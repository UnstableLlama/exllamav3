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
import time

import torch
import torch.distributed as dist

# Reuse the single-GPU example's data helpers (same dir on sys.path under torchrun).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import (  # noqa: E402
    build_sft_examples, build_lm_examples, collate, make_lr_scheduler,
    resolve_steps_and_warmup, ThroughputMeter,
)

from exllamav3 import Config, Model, Tokenizer  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402


def ddp_setup():
    """Read torchrun env, init the process group, pin this rank to its GPU."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # Pass device_id so NCCL collectives/barriers know this rank's device --
    # without it, barrier() warns and can HANG at teardown.
    dist.init_process_group(backend="nccl",
                            device_id=torch.device("cuda", local_rank))
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
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the LoRA params (default 0.01).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"],
                    default="none",
                    help="LR schedule after warmup: none/linear/cosine (to 0).")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Fraction of total steps to warm up the LR from 0 "
                         "(e.g. 0.05-0.1). Ignored if --warmup-steps>0.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when >0.")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, set --steps to cover this many passes over the "
                         "FULL training set (one step = batch*world*grad-accum "
                         "examples), so the schedule matches the epoch count.")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16, help="Per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--dataset", default="superdrew100/UwU_Alpaca_data")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="input")
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--messages-key", default=None,
                    help="Column holding OpenAI-style single-turn messages (e.g. "
                         "'messages' for UnstableLlama/semancy). When set, the user "
                         "turn is the prompt and the assistant turn the supervised "
                         "response; the flat instruction/response keys are ignored.")
    ap.add_argument("--prompt-format", choices=["auto", "mistral", "metharme"],
                    default="auto",
                    help="Chat format. auto: the model's native template "
                         "(Llama-3, Mistral [INST], mistral3 [SYSTEM_PROMPT]/[INST]). "
                         "mistral: explicit <s>[INST]{q}[/INST]{a}</s> (= auto for "
                         "the mistral3 arch, e.g. Mistral-Medium-3.5). metharme: "
                         "Pygmalion <|user|>{q}<|model|>{a}</s>.")
    ap.add_argument("--no-clean-text", action="store_true")
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--uppercase-response", action="store_true",
                    help="Smoke test: train to RESPOND IN ALL CAPS (dense, "
                         "unambiguous proof the training path works).")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the training rows once (deterministically, same "
                         "order on every rank) before the --val-frac carve and "
                         "sharding, so the held-out split is random and shards are "
                         "well mixed. Matched with the single-GPU/BNB arms by seed.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for --shuffle (also the random-subset seed when "
                         "--max-samples caps the rows). Default 0.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--train-embeddings", action="store_true",
                    help="Also FULLY train the input embeddings (modules_to_save). "
                         "Big (vocab x hidden): raises VRAM AND the per-step LoRA-"
                         "grad all-reduce by ~that size on every rank. On a tied "
                         "model this also trains the head.")
    ap.add_argument("--train-head", action="store_true",
                    help="Also FULLY train the LM head (modules_to_save); uses a "
                         "supervised-position CE so the head gets a gradient. Tied "
                         "model => same as --train-embeddings.")
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="Attention kernel: auto (FlashAttention-2 when available "
                         "on CUDA fp16/bf16, else eager), flash (require), or eager "
                         "(reference, O(t^2) memory). Flash is O(t) -- long context.")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-split", default=None,
                    help="Use this split of the dataset (e.g. 'test') as the "
                         "held-out eval set, instead of carving --val-frac off "
                         "train. Real held-out data; takes precedence over "
                         "--val-frac. Built identically on every rank.")
    ap.add_argument("--eval-dataset", default=None,
                    help="Dataset id/path for --eval-split (defaults to --dataset).")
    ap.add_argument("--eval2-dataset", default=None,
                    help="A SECOND held-out eval set, reported alongside the "
                         "primary one each --eval-every and at the end (e.g. your "
                         "test set vs wikitext). --save-best stays keyed on the "
                         "PRIMARY eval. Built identically on every rank.")
    ap.add_argument("--eval2-split", default="test",
                    help="Split for --eval2-dataset (default 'test').")
    ap.add_argument("--eval2-text-key", default=None,
                    help="If set, treat --eval2-dataset as PLAIN TEXT and compute "
                         "an LM loss over packed --seq-len blocks (e.g. 'text' for "
                         "wikitext). If unset, built as a second SFT eval.")
    ap.add_argument("--eval2-max-samples", type=int, default=0,
                    help="Cap source rows for --eval2-dataset (0 = all).")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="Hold out this fraction of train for held-out eval loss; "
                         "the SAME deterministic split as the single-GPU / BNB "
                         "arms. Held out before sharding so it never leaks into "
                         "training. Ignored if --eval-split is set.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Report held-out loss every N steps (needs --val-frac>0).")
    ap.add_argument("--save-best", action="store_true",
                    help="Save only when held-out loss improves (needs "
                         "--val-frac + --eval-every); keeps the best checkpoint "
                         "instead of an overfit endpoint.")
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
        train_embeddings=args.train_embeddings, train_head=args.train_head,
        attn_impl=args.attn_impl,
    )
    net.train()
    if is_main(rank):
        ms = net.modules_to_save_parameters()
        print(f" -- world_size {world_size}, trainable params: "
              f"{net.num_trainable():,} (r={args.r}, alpha={args.alpha}"
              f"{', +modules_to_save (' + str(sum(p.numel() for p in ms)) + ')' if ms else ''})")

    # 3a. Optionally resume from a checkpoint (e.g. stop a single-GPU run, then
    #     continue on N GPUs). Every rank loads the same file; the broadcast below
    #     then guarantees bit-identical starting weights regardless.
    if args.resume:
        net.load_adapter(args.resume)

    # 3b. Sync initial trainable weights from rank 0 so every rank starts
    #     identical (lora_a is random, lora_b zero; embed/head copies are
    #     identical already but broadcast for safety). Without this, ranks would
    #     diverge from step 1.
    for p in net.trainable_parameters():
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
        messages_key=args.messages_key,
        prompt_format=args.prompt_format,
        shuffle=args.shuffle, shuffle_seed=args.shuffle_seed,
    )
    # Held-out eval set, built identically on every rank. Prefer the dataset's own
    # eval split (real held-out data); otherwise carve the first val_frac off
    # train BEFORE sharding so those rows never leak into any training shard.
    val_examples = []
    if args.eval_split:
        val_examples = build_sft_examples(
            model, tokenizer, args.eval_dataset or args.dataset, 0, args.seq_len,
            instruction_key=args.instruction_key, context_key=args.context_key,
            response_key=args.response_key, split=args.eval_split,
            clean_text=not args.no_clean_text,
            min_response_words=args.min_response_words,
            uppercase_response=args.uppercase_response,
            messages_key=args.messages_key,
            prompt_format=args.prompt_format,
        )
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]

    # Optional SECOND held-out eval set (e.g. wikitext LM), built identically on
    # every rank so all ranks stay in lockstep when evaluating it.
    val2_examples = []
    eval2_label = ""
    if args.eval2_dataset:
        eval2_label = args.eval2_dataset.split("/")[-1]
        if args.eval2_text_key:
            val2_examples = build_lm_examples(
                tokenizer, args.eval2_dataset, args.eval2_split, args.seq_len,
                text_key=args.eval2_text_key, max_samples=args.eval2_max_samples)
        else:
            val2_examples = build_sft_examples(
                model, tokenizer, args.eval2_dataset, args.eval2_max_samples,
                args.seq_len, instruction_key=args.instruction_key,
                context_key=args.context_key, response_key=args.response_key,
                split=args.eval2_split, clean_text=not args.no_clean_text,
                min_response_words=args.min_response_words,
                uppercase_response=args.uppercase_response,
                messages_key=args.messages_key, prompt_format=args.prompt_format)
    shard = examples[rank::world_size]
    assert shard, "no training examples on this rank"

    # Finalize step count (from --epochs over the FULL train set) and warmup.
    eff_batch = args.batch * world_size * args.grad_accum
    args.steps, warmup_steps = resolve_steps_and_warmup(args, len(examples), eff_batch)
    if is_main(rank):
        print(f" -- {len(examples)} train examples total, ~{len(shard)} per rank; "
              f"{len(val_examples)} held out for eval "
              f"({'split ' + args.eval_split if args.eval_split else 'val_frac'})")
        print(f" -- {args.steps} steps, eff_batch {eff_batch}, "
              f"scheduler={args.scheduler}, warmup={warmup_steps}, "
              f"weight_decay={args.weight_decay}")

    opt = torch.optim.AdamW(net.param_groups(args.weight_decay), lr=args.lr)
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)

    def batches():
        order = list(range(len(shard)))
        # Per-rank RNG so shards reshuffle independently each epoch.
        rng = random.Random(1234 + rank)
        while True:
            rng.shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [shard[j] for j in order[i:i + args.batch]]

    def allreduce_grads():
        # Average trainable grads across ranks == DDP. Once per optimizer step,
        # after accumulation, before clipping. NB: with --train-embeddings/-head
        # the embed/head grads (vocab x hidden) are reduced here too -- much
        # larger than the usual few-MB LoRA grads.
        for p in net.trainable_parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

    def save(tag):
        # Single writer; every rank holds identical adapters after the all-reduce.
        if is_main(rank):
            net.save_adapter(args.out, base_model_name_or_path=args.model)
            print(f"{tag} adapter -> {args.out}")
        dist.barrier()

    def eval_loss(exs):
        # All ranks compute the same loss (replicated, synced adapters) so they
        # stay in lockstep; only rank 0 prints. Mean per-example loss, batch 1 --
        # identical to the single-GPU and BNB arms. Works for SFT and plain-LM
        # eval sets alike.
        if not exs:
            return None
        net.eval()
        total = 0.0
        with torch.no_grad():
            for ex in exs:
                ii, ll, aa = collate([ex], pad_id)
                total += net.compute_loss(ii, ll, attention_mask=aa,
                                          chunk=args.ce_chunk).item()
        net.train()
        return total / len(exs)

    def evaluate():
        return eval_loss(val_examples)

    bgen = batches()
    opt.zero_grad(set_to_none=True)
    ema = None
    step = 0
    best_val = float("inf")
    tok_seen, tot_seen, t0 = 0, 0, time.time()
    meter = ThroughputMeter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1, args.steps + 1):
            step_t0 = time.time()
            accum_loss = 0.0
            step_sup = step_tot = 0
            for _ in range(args.grad_accum):
                batch = next(bgen)
                input_ids, labels, attn = collate(batch, pad_id)
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk)
                (loss / args.grad_accum).backward()
                accum_loss += loss.item() / args.grad_accum
                step_sup += int((labels != -100).sum())
                step_tot += int(attn.sum())

            allreduce_grads()
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.trainable_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            # Average the loss across ranks just for a representative log line.
            lt = torch.tensor(accum_loss, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            global_loss = lt.item() / world_size
            # Live tok/s is this rank's only (no per-step collective); shards are
            # balanced so multiply by world_size for an aggregate estimate. The
            # final [PERF] line all-reduces the true total.
            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            _, tot_tps = meter.rates()
            ema = global_loss if ema is None else 0.9 * ema + 0.1 * global_loss
            if is_main(rank):
                print(f"  step {step:>5}/{args.steps} | loss {global_loss:6.4f} | "
                      f"ema {ema:6.4f} | grad {gnorm:7.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | "
                      f"~{tot_tps * world_size:,.0f} tok/s")

            if (args.eval_every and step % args.eval_every == 0
                    and (val_examples or val2_examples)):
                vl = evaluate()
                v2 = eval_loss(val2_examples)
                if is_main(rank):
                    parts = []
                    if vl is not None:
                        parts.append(f"held-out {vl:.4f}")
                    if v2 is not None:
                        parts.append(f"{eval2_label} {v2:.4f}")
                    print(f"    [eval] step {step}: " + " | ".join(parts))
                if args.save_best and vl is not None and vl < best_val:
                    best_val = vl
                    save(f"[best step {step}, val {vl:.4f}]")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")
    except KeyboardInterrupt:
        # Don't clobber the best-val checkpoint with the current (later, likely
        # worse) weights when --save-best is on; it's already saved.
        if args.save_best and val_examples:
            if is_main(rank):
                print(f"\nInterrupted at step {step}; keeping best-val adapter.")
        else:
            if is_main(rank):
                print(f"\nInterrupted at step {step}; saving.")
            if step > 0:
                save("[interrupted]")

    # With --save-best the best-val checkpoint is already saved; don't clobber it.
    if not (args.save_best and val_examples):
        save("Done.")

    # Held-out loss (all ranks compute identically; rank 0 reports) + global
    # throughput (sum supervised tokens across ranks) + this rank's peak VRAM.
    dt = time.time() - t0
    val_loss = evaluate()
    val2_loss = eval_loss(val2_examples)
    tok_t = torch.tensor([float(tok_seen), float(tot_seen)], device=device)
    dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
    if is_main(rank):
        if val_loss is not None:
            tag = f" (best kept: {best_val:.4f})" if args.save_best else ""
            print(f"\n[EVAL] held-out loss (EXL3 arm, DDP): {val_loss:.4f}{tag} "
                  f"over {len(val_examples)} examples")
        if val2_loss is not None:
            print(f"[EVAL] eval2 ({eval2_label}) loss: {val2_loss:.4f} "
                  f"over {len(val2_examples)} examples")
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[PERF] {tok_t[0].item() / dt if dt else 0:,.0f} sup tok/s, "
              f"{tok_t[1].item() / dt if dt else 0:,.0f} tot tok/s (all ranks) | "
              f"peak VRAM/GPU {peak_gb:.2f} GB | {dt:.0f}s, "
              f"{step} steps, world_size {world_size}")
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
