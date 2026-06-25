"""
BNB-NF4 QLoRA baseline arm — the comparison point for QLoRA-on-EXL3.

Trains the SAME model / data / LoRA config as examples/qlora_train_native.py and
qlora_train_native_ddp.py (the EXL3 arms); the ONLY difference is the frozen
base-weight format (bitsandbytes NF4 here vs the EXL3 trellis there). Everything
else is matched so the comparison isolates the quantization format:

  - same Llama-3 chat prompt + completion-only masking (prompt tokens = -100)
  - same `datasets` shuffle(seed=0)+select and the same deterministic val split
  - same LoRA (r/alpha/targets, dropout 0, bias none), fp32 adapters, bf16 compute
  - same optimizer (AdamW, lr, weight decay) + the same LR schedule helper
    (--scheduler/--warmup-*, identical to the EXL3 arm), grad-clip
  - same OpenAI `messages` loader (--messages-key) and real test-split eval
    (--eval-split), so a matched run needs identical flags on both arms
  - held-out loss computed identically (mean per-example completion loss, batch 1)

Runs in a SEPARATE venv (transformers + bitsandbytes + peft + accelerate +
datasets) so it cannot disturb the pinned torch/EXL3 extension in qlora-venv.
Point --model at the bf16/fp16 HF safetensors (bnb quantizes to NF4 on load).

Single GPU:
    ~/exl3/bnb-venv/bin/python examples/qlora_train_bnb.py \
        --model /path/to/Llama-3.2-3B-Instruct-bf16 \
        --out /mnt/two/adapters/yoda_bnb --dataset /mnt/two/data/yoda_refined.jsonl \
        --r 64 --alpha 64 --batch 16 --grad-accum 2 --steps 500 --val-frac 0.05 \
        --eval-every 25 --gen-out /mnt/two/data/yoda_bnb_samples.jsonl

Multi-GPU (DDP; --batch is PER-GPU, effective = batch * nproc * grad-accum):
    ~/exl3/bnb-venv/bin/torchrun --standalone --nproc_per_node=2 \
        examples/qlora_train_bnb.py --model ... --out ... --dataset ... \
        --r 64 --alpha 64 --batch 16 --steps 500 --val-frac 0.05 --eval-every 25
"""

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import time
from collections import deque

import torch
import torch.distributed as dist


class ThroughputMeter:  # identical to qlora_train_native.ThroughputMeter
    """Rolling tok/s over a sliding window of recent steps, for a live readout.
    Tracks supervised (labels != -100) and total (non-pad) tokens separately."""

    def __init__(self, window=20):
        self.buf = deque(maxlen=window)   # (dt, supervised_tokens, total_tokens)

    def update(self, dt, supervised, total):
        self.buf.append((float(dt), int(supervised), int(total)))

    def rates(self):
        tt = sum(b[0] for b in self.buf)
        if tt <= 0:
            return 0.0, 0.0
        return sum(b[1] for b in self.buf) / tt, sum(b[2] for b in self.buf) / tt


# Run-log schema + writer: identical to qlora_train_native (inlined -- the BNB arm
# runs in a separate venv and can't import the exllamav3 path), so both arms append
# to the same "mega CSV" with the same columns for a matched EXL3-vs-NF4 comparison.
RUN_LOG_FIELDS = [
    "timestamp", "arm", "status", "model", "arch", "out",
    "dataset", "eval_split", "eval_dataset", "eval2_dataset",
    "r", "alpha", "lr", "scheduler", "warmup_steps", "weight_decay",
    "batch", "grad_accum", "world_size", "eff_batch",
    "epochs", "steps_planned", "steps_done", "seq_len",
    "targets", "compute_dtype", "attn_impl", "parallel", "shuffle",
    "max_samples", "train_embeddings", "train_head", "prompt_format",
    "trainable_params", "n_train", "n_val", "n_eval2",
    "start_loss", "end_loss", "best_val", "best_val_step",
    "start_val", "start_eval2", "final_val", "final_eval2",
    "total_s", "s_per_step", "sup_tok_s", "tot_tok_s", "peak_vram_gb",
    "notes",
]


def append_run_log(path, record):
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header is not None and header != RUN_LOG_FIELDS:
            bak = path + ".bak"
            os.replace(path, bak)
            print(f"[run-log] schema changed; moved old log to {bak}")
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in RUN_LOG_FIELDS})
    print(f"[run-log] appended 1 row to {path}")


EOT = "<|eot_id|>"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Same fixed eval prompts as qlora_infer_native.py, so the generated-sample
# density score is comparable across arms.
EVAL_PROMPTS = [
    "Tell me about your day.",
    "Give me some advice about love.",
    "Explain how the water cycle works.",
    "What should I have for dinner tonight?",
]

_STAGE_DIR = re.compile(r"\[[^\]]*\]|\*[^*]*\*")
_WHITESPACE = re.compile(r"\s+")


def clean_style_text(s):  # identical to qlora_train_native.py
    s = _STAGE_DIR.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def llama3_prompt(user):  # identical to Llama.default_chat_prompt (no system)
    return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def extract_single_turn(messages):  # identical to qlora_train_native.py
    """(user_text, assistant_text) from an OpenAI-style single-turn messages list.
    Last user turn before the first assistant turn = prompt; that assistant turn =
    target. System turns ignored. ("", "") if either is missing -> row skipped."""
    user_text, asst_text = "", ""
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role == "user":
            user_text = content
        elif role == "assistant":
            asst_text = content
            break
    return user_text, asst_text


def make_lr_scheduler(optimizer, name, total_steps, warmup_steps):
    """Transformers-free LR scheduler (none/linear/cosine) with linear warmup,
    identical to qlora_train_native.py so the arms stay matched. Matches HF's
    get_{linear,cosine}_schedule_with_warmup."""
    name = (name or "none").lower()
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if name in ("none", "constant"):
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown scheduler '{name}' (expected none/linear/cosine)")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_steps_and_warmup(args, num_train_examples, effective_batch):
    """Finalize args.steps (from --epochs) and compute warmup steps, identical to
    qlora_train_native.py."""
    if getattr(args, "epochs", 0) and args.epochs > 0:
        eff = max(1, int(effective_batch))
        steps_per_epoch = max(1, math.ceil(num_train_examples / eff))
        args.steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    warmup = (args.warmup_steps if getattr(args, "warmup_steps", 0) and args.warmup_steps > 0
              else int(round(getattr(args, "warmup_ratio", 0.0) * args.steps)))
    return args.steps, max(0, warmup)


def build_sft_examples(tok, args, split=None, dataset=None, max_samples=None,
                       shuffle=False, config_name=None):
    """Mirror of qlora_train_native.build_sft_examples using the HF tokenizer.
    The underlying Llama-3 tokenizer is the same, so token IDs match the EXL3
    arm for identical text; add_special_tokens=False (the chat string already
    contains <|begin_of_text|>) reproduces add_bos=False + encode_special.

    split/dataset/max_samples override the args defaults (used to build the
    held-out eval set from a real split, e.g. 'test'). args.messages_key picks
    the OpenAI `messages` layout, matching the EXL3 arm."""
    from datasets import load_dataset
    name = dataset or args.dataset
    split = split or args.dataset_split
    max_samples = args.max_samples if max_samples is None else max_samples
    if os.path.exists(name):
        ext = os.path.splitext(name)[1].lower()
        builder = {".json": "json", ".jsonl": "json",
                   ".parquet": "parquet", ".csv": "csv"}.get(ext, "json")
        ds = load_dataset(builder, data_files=name, split=split)
    elif config_name:
        ds = load_dataset(name, config_name, split=split)
    else:
        ds = load_dataset(name, split=split)
    # Match qlora_train_native: shuffle the full set when asked (or, as before,
    # when capping rows). shuffle_seed defaults to 0, so the cap behavior and the
    # --val-frac split stay identical to the EXL3 arm for matched runs.
    seed = getattr(args, "shuffle_seed", 0)
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    messages_key = getattr(args, "messages_key", None)
    examples = []
    for ex in ds:
        if messages_key:
            instr, resp = extract_single_turn(ex.get(messages_key))
            ctx = ""
        else:
            instr = (ex.get(args.instruction_key) or "").strip()
            ctx = (ex.get(args.context_key) or "").strip()
            resp = (ex.get(args.response_key) or "").strip()
        if not args.no_clean_text:
            instr, ctx, resp = (clean_style_text(instr), clean_style_text(ctx),
                                clean_style_text(resp))
        if not resp or len(resp.split()) < args.min_response_words:
            continue
        if messages_key and not instr:
            continue  # malformed messages row: no user turn to prompt with
        if args.uppercase_response:
            resp = resp.upper()
        user = instr if not ctx else f"{instr}\n\n{ctx}"
        prompt_ids = tok(llama3_prompt(user), add_special_tokens=False)["input_ids"]
        resp_ids = tok(resp + EOT, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + resp_ids)[:args.seq_len]
        labels = ([-100] * len(prompt_ids) + list(resp_ids))[:args.seq_len]
        if all(l == -100 for l in labels):
            continue
        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


def build_lm_examples(tok, dataset_name, split, seq_len, text_key="text",
                      max_samples=0, config_name=None):
    """Plain-text LM eval set (e.g. wikitext), mirror of
    qlora_train_native.build_lm_examples using the HF tokenizer. Packs the text
    column into non-overlapping seq_len blocks, every token supervised. Same
    underlying tokenizer + packing as the EXL3 arm => identical blocks and a
    comparable nats/token loss."""
    from datasets import load_dataset
    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        builder = {".json": "json", ".jsonl": "json",
                   ".parquet": "parquet", ".csv": "csv"}.get(ext, "json")
        ds = load_dataset(builder, data_files=dataset_name, split=split)
    elif config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    bos = tok.bos_token_id
    buf, examples = [], []
    for row in ds:
        text = row.get(text_key) or ""
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        buf.extend(ids)
        while len(buf) >= seq_len:
            block = buf[:seq_len]
            buf = buf[seq_len:]
            examples.append({"input_ids": block, "labels": list(block)})
    return examples


def collate(batch, pad_id):  # identical padding to qlora_train_native.collate
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16/fp16 HF model dir")
    ap.add_argument("--out", required=True, help="Adapter output dir (PEFT)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="input")
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--messages-key", default=None,
                    help="Column holding OpenAI-style single-turn messages (e.g. "
                         "'messages' for UnstableLlama/semancy). When set, the user "
                         "turn is the prompt and the assistant turn the supervised "
                         "response; the flat instruction/response keys are ignored. "
                         "Matches the EXL3 arm's --messages-key.")
    ap.add_argument("--no-clean-text", action="store_true")
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--uppercase-response", action="store_true")
    # --lora-r (not --r): under torchrun, argparse abbrev-matches "--r" to
    # torchrun's own --rdzv-*/--role options. dest stays "r".
    ap.add_argument("--lora-r", dest="r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the LoRA params (default 0.01, "
                         "matching the EXL3 arm's torch-AdamW default).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"],
                    default="none",
                    help="LR schedule after warmup: none/linear/cosine (to 0). "
                         "Matches the EXL3 arm's --scheduler.")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Fraction of total steps to warm up the LR from 0 "
                         "(e.g. 0.05-0.1). Ignored if --warmup-steps>0.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when >0.")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, set --steps to cover this many passes over the "
                         "FULL training set (one step = batch*world*grad-accum "
                         "examples), so the schedule matches the epoch count.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8, help="Per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap source rows (0 = use all). Match this to the EXL3 "
                         "arm so both train/eval on the same split.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the training rows once (deterministically) "
                         "before the --val-frac carve and training. Matches the "
                         "EXL3 arm given the same --shuffle-seed.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for --shuffle (also the random-subset seed when "
                         "--max-samples caps the rows). Default 0.")
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
                         "PRIMARY eval. Matches the EXL3 arm's --eval2-*.")
    ap.add_argument("--eval2-split", default="test",
                    help="Split for --eval2-dataset (default 'test').")
    ap.add_argument("--eval2-config", default=None,
                    help="HF dataset config for --eval2-dataset (e.g. "
                         "'wikitext-2-raw-v1' for 'wikitext').")
    ap.add_argument("--eval2-text-key", default=None,
                    help="If set, treat --eval2-dataset as PLAIN TEXT and compute "
                         "an LM loss over packed --seq-len blocks (e.g. 'text' for "
                         "wikitext). If unset, built as a second SFT eval.")
    ap.add_argument("--eval2-max-samples", type=int, default=0,
                    help="Cap source rows for --eval2-dataset (0 = all).")
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Also report held-out loss every N steps (needs "
                         "--val-frac > 0 or --eval-split). 0 = only at the end.")
    ap.add_argument("--save-best", action="store_true",
                    help="Save the adapter only when held-out loss improves "
                         "(needs --val-frac + --eval-every). Avoids keeping an "
                         "overfit endpoint.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-out", default=None,
                    help="Write greedy generations on the eval prompts here "
                         "(jsonl, 'output' field) for score_style_density.py")
    ap.add_argument("--gen-max-new-tokens", type=int, default=120)
    ap.add_argument("--run-log", default="qlora_runs.csv",
                    help="Append one metadata row per run to this CSV (rank 0). "
                         "Same schema as the EXL3 arm, so both append to the same "
                         "mega-CSV for matched comparison. Empty string disables.")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # DDP: same manual pattern as qlora_train_native_ddp.py (replicate the small
    # NF4 model per GPU, shard the batch, all-reduce only the LoRA grads). No
    # torch-DDP wrapper, so bitsandbytes' 4-bit params don't trip its bucketing.
    ddp = "RANK" in os.environ
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        # device_id so NCCL barriers/collectives use this rank's GPU (without it
        # barrier() warns and can hang at teardown).
        dist.init_process_group(backend="nccl",
                                device_id=torch.device("cuda", local_rank))
    else:
        rank, local_rank, world_size = 0, 0, 1
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=not args.no_grad_ckpt)
    lcfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    if is_main:
        model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    # Start every rank from identical adapters.
    if ddp:
        for p in params:
            dist.broadcast(p.data, src=0)

    examples = build_sft_examples(tok, args, shuffle=args.shuffle)
    assert examples, "no usable training examples"
    # Held-out eval set. Prefer the dataset's own eval split (real held-out data);
    # otherwise carve the first val_frac off train BEFORE sharding so it never
    # leaks into any rank's training data and matches the other arms exactly.
    val_examples = []
    if args.eval_split:
        val_examples = build_sft_examples(
            tok, args, split=args.eval_split,
            dataset=args.eval_dataset or args.dataset, max_samples=0)
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]

    # Optional SECOND held-out eval set (e.g. wikitext LM), mirroring the EXL3 arm.
    val2_examples = []
    eval2_label = ""
    if args.eval2_dataset:
        eval2_label = args.eval2_dataset.split("/")[-1]
        if args.eval2_text_key:
            val2_examples = build_lm_examples(
                tok, args.eval2_dataset, args.eval2_split, args.seq_len,
                text_key=args.eval2_text_key, max_samples=args.eval2_max_samples,
                config_name=args.eval2_config)
        else:
            val2_examples = build_sft_examples(
                tok, args, split=args.eval2_split, dataset=args.eval2_dataset,
                max_samples=args.eval2_max_samples, config_name=args.eval2_config)
    shard = examples[rank::world_size] if ddp else examples

    # Finalize step count (from --epochs over the FULL train set) and warmup.
    eff_batch = args.batch * world_size * args.grad_accum
    args.steps, warmup_steps = resolve_steps_and_warmup(args, len(examples), eff_batch)
    if is_main:
        print(f" -- {len(examples)} train examples ({len(shard)}/rank), "
              f"{len(val_examples)} val "
              f"({'split ' + args.eval_split if args.eval_split else 'val_frac'})")
        print(f" -- {args.steps} steps, eff_batch {eff_batch}, "
              f"scheduler={args.scheduler}, warmup={warmup_steps}, "
              f"weight_decay={args.weight_decay}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)

    def batches():
        order = list(range(len(shard)))
        rng = random.Random(1234 + rank if ddp else 0)
        while True:
            rng.shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [shard[j] for j in order[i:i + args.batch]]

    def eval_loss(exs):
        # All ranks compute the same loss (replicated, synced adapters) so they
        # stay in lockstep; mean per-example loss, batch 1. Works for SFT and
        # plain-LM eval sets alike.
        if not exs:
            return None
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ex in exs:
                ii, ll, aa = collate([ex], pad_id)
                out = model(input_ids=ii.to(device), attention_mask=aa.to(device),
                            labels=ll.to(device))
                total += out.loss.item()
        model.train()
        return total / len(exs)

    def evaluate():
        return eval_loss(val_examples)

    def save(tag):
        if is_main:
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            print(f"{tag} adapter -> {args.out}")
        if ddp:
            dist.barrier()

    bgen = batches()
    model.train()
    opt.zero_grad(set_to_none=True)
    ema, tok_seen, tot_seen, t0, best_val = None, 0, 0, time.time(), float("inf")
    step = 0
    best_val_step = 0
    start_loss = end_loss = None
    start_val = start_eval2 = None
    last_eval_step, last_val, last_eval2 = -1, None, None
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    status = "completed"
    meter = ThroughputMeter()

    # Baseline eval at step 0 (no-op adapter = base NF4 model); rank 0 prints.
    if val_examples or val2_examples:
        start_val = evaluate()
        start_eval2 = eval_loss(val2_examples)
        if is_main:
            parts = []
            if start_val is not None:
                parts.append(f"held-out {start_val:.4f}")
            if start_eval2 is not None:
                parts.append(f"{eval2_label} {start_eval2:.4f}")
            print("    [eval] step 0 (baseline): " + " | ".join(parts))

    t0 = time.time()                       # training timer after the baseline eval
    torch.cuda.reset_peak_memory_stats(device)
    try:
      for step in range(1, args.steps + 1):
        step_t0 = time.time()
        accum = 0.0
        step_sup = step_tot = 0
        for _ in range(args.grad_accum):
            batch = next(bgen)
            ii, ll, aa = collate(batch, pad_id)
            out = model(input_ids=ii.to(device), attention_mask=aa.to(device),
                        labels=ll.to(device))
            (out.loss / args.grad_accum).backward()
            accum += out.loss.item() / args.grad_accum
            step_sup += int((ll != -100).sum())
            step_tot += int(aa.sum())
        if ddp:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm
                                               or float("inf")).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        # Live tok/s (this rank; ×world_size for an aggregate estimate -- the
        # final [PERF] line all-reduces the true total).
        tok_seen += step_sup
        tot_seen += step_tot
        meter.update(time.time() - step_t0, step_sup, step_tot)
        _, tot_tps = meter.rates()
        if start_loss is None:
            start_loss = accum
        end_loss = accum
        ema = accum if ema is None else 0.9 * ema + 0.1 * accum
        if is_main:
            print(f"  step {step:>5}/{args.steps} | loss {accum:6.4f} | "
                  f"ema {ema:6.4f} | grad {gnorm:7.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"~{tot_tps * world_size:,.0f} tok/s")
        if (args.eval_every and step % args.eval_every == 0
                and (val_examples or val2_examples)):
            vl = evaluate()
            v2 = eval_loss(val2_examples)
            last_eval_step, last_val, last_eval2 = step, vl, v2
            if is_main:
                parts = []
                if vl is not None:
                    parts.append(f"held-out {vl:.4f}")
                if v2 is not None:
                    parts.append(f"{eval2_label} {v2:.4f}")
                print(f"    [eval] step {step}: " + " | ".join(parts))
            if vl is not None and vl < best_val:
                best_val = vl
                best_val_step = step
                if args.save_best:
                    save(f"[best step {step}, val {vl:.4f}]")
    except KeyboardInterrupt:
        status = "interrupted"

    dt = time.time() - t0
    if not (args.save_best and val_examples):
        save("Done.")

    # Reuse the last in-loop eval if it landed on the final step, else compute once.
    if last_eval_step == step:
        val_loss, val2_loss = last_val, last_eval2
    else:
        if is_main and (val_examples or val2_examples):
            print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss = evaluate()
        val2_loss = eval_loss(val2_examples)
    tok_t = torch.tensor([float(tok_seen), float(tot_seen)], device=device)
    if ddp:
        dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
    if is_main:
        if val_loss is not None:
            tag = f" (best kept: {best_val:.4f})" if args.save_best else ""
            print(f"\n[EVAL] held-out loss (BNB-NF4 arm): {val_loss:.4f}{tag} "
                  f"over {len(val_examples)} examples")
        if val2_loss is not None:
            print(f"[EVAL] eval2 ({eval2_label}) loss: {val2_loss:.4f} "
                  f"over {len(val2_examples)} examples")
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[PERF] {tok_t[0].item() / dt if dt else 0:,.0f} sup tok/s, "
              f"{tok_t[1].item() / dt if dt else 0:,.0f} tot tok/s "
              f"({'all ranks' if ddp else '1 GPU'}) | peak VRAM/GPU {peak_gb:.2f} "
              f"GB | {dt:.0f}s, {args.steps} steps, world_size {world_size}")

    if is_main:
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        archs = getattr(model.config, "architectures", None) or [""]
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "bnb-nf4", "status": status,
            "model": args.model, "arch": archs[0], "out": args.out,
            "dataset": args.dataset, "eval_split": args.eval_split or "",
            "eval_dataset": args.eval_dataset or "", "eval2_dataset": args.eval2_dataset or "",
            "r": args.r, "alpha": args.alpha, "lr": args.lr,
            "scheduler": args.scheduler, "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay, "batch": args.batch,
            "grad_accum": args.grad_accum, "world_size": world_size,
            "eff_batch": eff_batch, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(TARGET_MODULES), "compute_dtype": "bfloat16",
            "attn_impl": "hf-sdpa", "parallel": "ddp" if ddp else "single",
            "shuffle": int(bool(args.shuffle)), "max_samples": args.max_samples,
            "train_embeddings": 0, "train_head": 0, "prompt_format": "llama3",
            "trainable_params": sum(p.numel() for p in params),
            "n_train": len(examples), "n_val": len(val_examples),
            "n_eval2": len(val2_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "start_val": rnd(start_val), "start_eval2": rnd(start_eval2),
            "final_val": rnd(val_loss), "final_eval2": rnd(val2_loss),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(tok_t[0].item() / dt) if dt else "",
            "tot_tok_s": round(tok_t[1].item() / dt) if dt else "",
            "peak_vram_gb": rnd(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            "notes": "",
        })

    if args.gen_out and is_main:
        model.eval()
        recs = []
        with torch.no_grad():
            for p in EVAL_PROMPTS:
                ids = tok(llama3_prompt(p), add_special_tokens=False,
                          return_tensors="pt")["input_ids"].to(device)
                out = model.generate(ids, max_new_tokens=args.gen_max_new_tokens,
                                     do_sample=False,
                                     eos_token_id=tok.convert_tokens_to_ids(EOT),
                                     pad_token_id=pad_id)
                text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                recs.append({"instruction": p, "input": "", "output": text})
                print(f"\n> {p}\n{text}")
        with open(args.gen_out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nGenerations written to {args.gen_out} "
              f"(score with examples/score_style_density.py)")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
