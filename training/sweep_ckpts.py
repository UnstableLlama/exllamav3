"""
Held-out loss sweep across QLoRA checkpoints.

Loads the EXL3 base + differentiable net ONCE, builds the held-out split via the
trainer's own build_sft_examples (completion-masked chat loss, so the number is
directly comparable to the trainer's "[eval] step 0 (baseline)" print), then
load_adapter() per checkpoint and reports loss + perplexity. Reuses exllamav3's
loader and the trainer module -- no reload per checkpoint.
"""
import argparse, glob, os, math, torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.training.native_llama import NativeLlamaQLoRA
from training.qlora_train_native import build_sft_examples, collate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True, help="dir holding checkpoint-* subdirs")
ap.add_argument("--dataset", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--messages-key", default="messages")
ap.add_argument("--prompt-format", default="auto")
ap.add_argument("--seq-len", type=int, default=1024)
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--alpha", type=float, default=64.0)
ap.add_argument("--expert-r", type=int, default=0)
ap.add_argument("--targets", nargs="+", required=True)
ap.add_argument("--use-per-device", type=int, nargs="+", default=None)
ap.add_argument("--ce-chunk", type=int, default=64)
ap.add_argument("--baseline", action="store_true", help="also eval the zero adapter (= base)")
args = ap.parse_args()

config = Config.from_directory(args.model)
model = Model.from_config(config)
load_kwargs = {}
if args.use_per_device is not None:
    load_kwargs["use_per_device"] = args.use_per_device
model.load(progressbar=True, **load_kwargs)
print(f" -- active devices {list(model.active_devices)}, output {model.output_device}")
tokenizer = Tokenizer.from_config(config)
pad_id = tokenizer.pad_token_id
if pad_id is None or pad_id < 0:
    pad_id = tokenizer.eos_token_id or 0

net = NativeLlamaQLoRA(
    model, r=args.r, alpha=args.alpha, target_modules=args.targets,
    compute_dtype=torch.bfloat16, gradient_checkpointing=False,
    attn_impl="auto", offload_activations=False, use_liger=False,
    expert_r=args.expert_r,
)
print(f" -- {net.describe_attn()}")

exs = build_sft_examples(
    model, tokenizer, args.dataset, 0, args.seq_len,
    split=args.split, messages_key=args.messages_key,
    prompt_format=args.prompt_format, shuffle=False,
)
print(f" -- {len(exs)} held-out examples from split '{args.split}'\n")

def eval_loss():
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ex in exs:
            input_ids, labels, attn, pos_ids, seg_ids = collate([ex], pad_id)
            l = net.compute_loss(input_ids, labels, attention_mask=attn,
                                 chunk=args.ce_chunk, position_ids=pos_ids, seg_ids=seg_ids)
            total += l.item(); n += 1
    return total / n

rows = []
if args.baseline:
    vl = eval_loss()
    rows.append(("baseline(0)", 0, vl))
    print(f"  baseline (no adapter): loss {vl:.4f}  ppl {math.exp(vl):.3f}")

ckpts = sorted(glob.glob(os.path.join(args.out, "checkpoint-*")),
               key=lambda p: int(p.rsplit("-", 1)[1]))
for c in ckpts:
    step = int(c.rsplit("-", 1)[1])
    net.load_adapter(c)
    vl = eval_loss()
    rows.append((os.path.basename(c), step, vl))
    print(f"  step {step:>4}: loss {vl:.4f}  ppl {math.exp(vl):.3f}")

print("\n=== summary (held-out loss) ===")
best = min(rows, key=lambda r: r[2])
for name, step, vl in rows:
    mark = "  <-- BEST" if (name, step, vl) == best else ""
    print(f"  {name:>14}  step {step:>4}  loss {vl:.4f}{mark}")
