"""
Multi-GPU (DistributedDataParallel) QLoRA fine-tuning of an EXL3 model, native
path (no HuggingFace Transformers).

Why DDP and not FSDP for QLoRA-on-EXL3
--------------------------------------
Only the LoRA ``a``/``b`` are trainable; the EXL3 base is frozen and tiny (that's
the whole point of low-bpw quant). So we don't want to *shard* the base -- we
replicate the (small, quantized) model on every GPU, shard the *batch*, and
all-reduce only the LoRA gradients (a few MB). That is exactly DDP. FSDP would
have to learn to all-gather the packed trellis buffers for no memory benefit.

We do NOT wrap the module in ``nn.parallel.DistributedDataParallel`` -- the model
is mostly frozen buffers plus a custom ``autograd.Function`` and gradient
checkpointing, which DDP's bucketing handles awkwardly. Instead we average the
LoRA gradients by hand after backward; with so few trainable params this is
trivial and fully equivalent.

Launch with torchrun (one process per GPU):

    torchrun --standalone --nproc_per_node=4 examples/qlora_train_native_ddp.py \
        --model /path/to/exl3_model \
        --out   out/exl3_qlora_adapter \
        --dataset TeeZee/dolly-15k-pirate-speech \
        --instruction-key instruction --context-key context --response-key response \
        --lora-r 64 --alpha 64 --batch 16 --steps 800
        # NB: use --lora-r, not --r -- torchrun intercepts the abbreviated "--r".

Effective batch = --batch * nproc_per_node * --grad-accum.

NOTE: this has been written to standard DDP idioms but NOT yet run on a real
multi-GPU box. Validate the first run carefully (see the checklist at the bottom).
"""

import argparse
import os
import random
import sys

import torch
import torch.distributed as dist

# Reuse the single-GPU example's data helpers (same dir on sys.path under torchrun).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import build_sft_examples, collate  # noqa: E402

from exllamav3 import Config, Model, Tokenizer  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402


def ddp_setup():
    """Read torchrun env, init the process group, pin this rank to its GPU."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    # NB: not "--r" -- torchrun greedily abbreviation-matches "--r" against its
    # own --rdzv-*/--role/--run-path options and errors out before our script
    # ever sees it. "--lora-r" sidesteps that; dest stays "r" so code is unchanged.
    ap.add_argument("--lora-r", dest="r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16, help="Per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--dataset", default="superdrew100/UwU_Alpaca_data")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="input")
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--no-clean-text", action="store_true")
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--uppercase-response", action="store_true",
                    help="Smoke test: train to RESPOND IN ALL CAPS (dense, "
                         "unambiguous proof the training path works).")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="Adapter dir to resume from (e.g. a single-GPU checkpoint). "
                         "Loaded on every rank before the broadcast. --r/--targets "
                         "must match the checkpoint; optimizer state is not restored.")
    args = ap.parse_args()

    rank, local_rank, world_size = ddp_setup()
    device = f"cuda:{local_rank}"
    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Every rank loads a full copy of the (small, quantized) model on its GPU.
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(device=device, progressbar=is_main(rank))
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 2. Build the differentiable QLoRA model (frozen base + trainable adapters).
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        compute_dtype=cdt, gradient_checkpointing=not args.no_grad_ckpt,
    )
    net.train()
    if is_main(rank):
        print(f" -- world_size {world_size}, trainable LoRA params: "
              f"{net.num_trainable():,} (r={args.r}, alpha={args.alpha})")

    # 3a. Optionally resume from a checkpoint (e.g. stop a single-GPU run, then
    #     continue on N GPUs). Every rank loads the same file; the broadcast below
    #     then guarantees bit-identical starting weights regardless.
    if args.resume:
        net.load_adapter(args.resume)

    # 3b. Sync initial adapter weights from rank 0 so every rank starts identical
    #     (lora_a is randomly initialized; lora_b is zero, but broadcast both to be
    #     safe). Without this, ranks would diverge from step 1.
    for p in net.lora_parameters():
        dist.broadcast(p.data, src=0)

    # 4. Data. Build the full set identically on every rank (same seed/order), then
    #    take a disjoint stride-shard so each GPU trains on different examples.
    examples = build_sft_examples(
        model, tokenizer, args.dataset, args.max_samples, args.seq_len,
        instruction_key=args.instruction_key, context_key=args.context_key,
        response_key=args.response_key, split=args.dataset_split,
        clean_text=not args.no_clean_text,
        min_response_words=args.min_response_words,
        uppercase_response=args.uppercase_response,
    )
    shard = examples[rank::world_size]
    assert shard, "no training examples on this rank"
    if is_main(rank):
        print(f" -- {len(examples)} examples total, ~{len(shard)} per rank")

    opt = torch.optim.AdamW(net.lora_parameters(), lr=args.lr)

    def batches():
        order = list(range(len(shard)))
        # Per-rank RNG so shards reshuffle independently each epoch.
        rng = random.Random(1234 + rank)
        while True:
            rng.shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [shard[j] for j in order[i:i + args.batch]]

    def allreduce_grads():
        # Average LoRA grads across ranks == DDP. Do it once per optimizer step,
        # after accumulation, before clipping.
        for p in net.lora_parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

    def save(tag):
        # Single writer; every rank holds identical adapters after the all-reduce.
        if is_main(rank):
            net.save_adapter(args.out, base_model_name_or_path=args.model)
            print(f"{tag} adapter -> {args.out}")
        dist.barrier()

    bgen = batches()
    opt.zero_grad(set_to_none=True)
    ema = None
    step = 0
    try:
        for step in range(1, args.steps + 1):
            accum_loss = 0.0
            for _ in range(args.grad_accum):
                batch = next(bgen)
                input_ids, labels, attn = collate(batch, pad_id)
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk)
                (loss / args.grad_accum).backward()
                accum_loss += loss.item() / args.grad_accum

            allreduce_grads()
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.lora_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            opt.zero_grad(set_to_none=True)

            # Average the loss across ranks just for a representative log line.
            lt = torch.tensor(accum_loss, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            global_loss = lt.item() / world_size
            ema = global_loss if ema is None else 0.9 * ema + 0.1 * global_loss
            if is_main(rank):
                print(f"  step {step:>5}/{args.steps} | loss {global_loss:6.4f} | "
                      f"ema {ema:6.4f} | grad {gnorm:7.4f}")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")
    except KeyboardInterrupt:
        if is_main(rank):
            print(f"\nInterrupted at step {step}; saving.")
        if step > 0:
            save("[interrupted]")

    save("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# First-run checklist on a real multi-GPU box (this script is untested on HW):
#   * `nvidia-smi` shows all N GPUs busy and at similar memory.
#   * Loss matches a single-GPU run with the same EFFECTIVE batch
#     (batch * nproc_per_node) -- if DDP loss is ~N x too small or large, the
#     grad averaging / loss reduction is off.
#   * Disjoint shards: rank logs should show ~total/N examples per rank.
#   * Adapter saved by rank 0 loads + steers in qlora_infer_native.py exactly
#     like the single-GPU adapter.
#   * If NCCL hangs at init, set NCCL_DEBUG=INFO and check the interface/port.
# ---------------------------------------------------------------------------
