"""
BNB-NF4 QLoRA baseline arm — the comparison point for QLoRA-on-EXL3.

This trains the SAME model / data / LoRA config as examples/qlora_train_native.py
(the EXL3 arm), with the ONLY difference being the frozen base-weight format:
bitsandbytes NF4 here vs the EXL3 trellis there. Everything else is matched so
the comparison isolates the quantization format:

  - same Llama-3 chat prompt + completion-only masking (prompt tokens = -100)
  - same `datasets` shuffle(seed=0)+select and the same deterministic val split
  - same LoRA (r/alpha/targets, dropout 0, bias none), fp32 adapters, bf16 compute
  - same optimizer (AdamW, lr, no weight decay, constant LR), grad-clip
  - held-out loss computed identically (mean per-example completion loss, batch 1)

Runs in a SEPARATE venv (transformers + bitsandbytes + peft + accelerate +
datasets) so it cannot disturb the pinned torch/EXL3 extension in qlora-venv.
Point --model at the bf16/fp16 HF safetensors (bnb quantizes to NF4 on load).

    ~/exl3/bnb-venv/bin/python examples/qlora_train_bnb.py \
        --model /path/to/Llama-3.1-8B-Instruct-bf16 \
        --out   /mnt/two/adapters/yoda_bnb \
        --dataset /mnt/two/data/yoda_refined.jsonl \
        --r 64 --alpha 64 --batch 8 --steps 500 --val-frac 0.05 \
        --gen-out /mnt/two/data/yoda_bnb_samples.jsonl
"""

import argparse
import json
import os
import random
import re
import time

import torch

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


def build_sft_examples(tok, args):
    """Mirror of qlora_train_native.build_sft_examples using the HF tokenizer.
    The underlying Llama-3 tokenizer is the same, so token IDs match the EXL3
    arm for identical text; add_special_tokens=False (the chat string already
    contains <|begin_of_text|>) reproduces add_bos=False + encode_special."""
    from datasets import load_dataset
    name = args.dataset
    if os.path.exists(name):
        ext = os.path.splitext(name)[1].lower()
        builder = {".json": "json", ".jsonl": "json",
                   ".parquet": "parquet", ".csv": "csv"}.get(ext, "json")
        ds = load_dataset(builder, data_files=name, split=args.dataset_split)
    else:
        ds = load_dataset(name, split=args.dataset_split)
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.max_samples))

    examples = []
    for ex in ds:
        instr = (ex.get(args.instruction_key) or "").strip()
        ctx = (ex.get(args.context_key) or "").strip()
        resp = (ex.get(args.response_key) or "").strip()
        if not args.no_clean_text:
            instr, ctx, resp = (clean_style_text(instr), clean_style_text(ctx),
                                clean_style_text(resp))
        if not resp or len(resp.split()) < args.min_response_words:
            continue
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
    ap.add_argument("--no-clean-text", action="store_true")
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--uppercase-response", action="store_true")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-out", default=None,
                    help="Write greedy generations on the eval prompts here "
                         "(jsonl, 'output' field) for score_style_density.py")
    ap.add_argument("--gen-max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=not args.no_grad_ckpt)
    lcfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    examples = build_sft_examples(tok, args)
    assert examples, "no usable training examples"
    print(f" -- {len(examples)} SFT examples")
    val_examples = []
    if args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val; {len(examples)} train")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)  # no weight decay, constant LR

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(0).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    def evaluate():
        if not val_examples:
            return None
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ex in val_examples:
                ii, ll, aa = collate([ex], pad_id)
                out = model(input_ids=ii.cuda(), attention_mask=aa.cuda(),
                            labels=ll.cuda())
                total += out.loss.item()
        model.train()
        return total / len(val_examples)

    bgen = batches()
    model.train()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    ema, tok_seen, t0 = None, 0, time.time()
    for step in range(1, args.steps + 1):
        accum = 0.0
        for _ in range(args.grad_accum):
            batch = next(bgen)
            ii, ll, aa = collate(batch, pad_id)
            out = model(input_ids=ii.cuda(), attention_mask=aa.cuda(),
                        labels=ll.cuda())
            (out.loss / args.grad_accum).backward()
            accum += out.loss.item() / args.grad_accum
            tok_seen += int((ll != -100).sum())
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm
                                               or float("inf")).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema = accum if ema is None else 0.9 * ema + 0.1 * accum
        print(f"  step {step:>5}/{args.steps} | loss {accum:6.4f} | "
              f"ema {ema:6.4f} | grad {gnorm:7.4f}")

    dt = time.time() - t0
    tok_per_s = tok_seen / dt if dt > 0 else 0.0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"Adapter written to {args.out}")

    val_loss = evaluate()
    if val_loss is not None:
        print(f"\n[EVAL] held-out loss (BNB-NF4 arm): {val_loss:.4f} "
              f"over {len(val_examples)} examples")
    print(f"[PERF] {tok_per_s:,.0f} supervised tok/s | "
          f"peak VRAM {peak_gb:.2f} GB | {dt:.0f}s for {args.steps} steps")

    if args.gen_out:
        model.eval()
        recs = []
        with torch.no_grad():
            for p in EVAL_PROMPTS:
                ids = tok(llama3_prompt(p), add_special_tokens=False,
                          return_tensors="pt")["input_ids"].cuda()
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


if __name__ == "__main__":
    main()
