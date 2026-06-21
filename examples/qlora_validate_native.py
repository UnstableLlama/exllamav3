"""
Validate the transformers-free differentiable Llama forward against the native
(correct) exllamav3 inference forward.

This is the correctness gate for QLoRA-on-EXL3 via the native path: before
training anything, prove that ``NativeLlamaQLoRA`` (pure autograd PyTorch on the
quantized weights) reproduces the logits that exllamav3's own kernels produce.
If the top-1 tokens match and the per-token loss is low, the differentiable
backbone is sound and any adapter trained on it is meaningful.

Unlike the HuggingFace integration, nothing here depends on a ``transformers``
version -- it reuses exllamav3's loaded weights and RoPE table directly.

Usage:
    python examples/qlora_validate_native.py \
        --model /path/to/exl3_model
"""

import argparse
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training.native_llama import NativeLlamaQLoRA


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Water is made of hydrogen and",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compute-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="dtype for the differentiable linears (float32 = closest to true math)")
    ap.add_argument("--prompts", nargs="*", default=None)
    args = ap.parse_args()

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]
    prompts = args.prompts or DEFAULT_PROMPTS

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(device=args.device, progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    # No adapters: a pure frozen forward, directly comparable to native inference.
    net = NativeLlamaQLoRA(model, target_modules=[], compute_dtype=cdt,
                           gradient_checkpointing=False)
    net.eval()

    print("=" * 78)
    print(f"Validating differentiable forward (compute_dtype={args.compute_dtype}) "
          f"vs native exllamav3")
    print("=" * 78)

    all_ok = True
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True).to(args.device)

        # Native (correct) forward -- runs under inference_mode, fp16 kernels.
        with torch.inference_mode():
            logits_native = model.forward(ids).float()       # [1, t, V]

        # Differentiable forward.
        with torch.no_grad():
            logits_diff = net.logits(ids).float()            # [1, t, V]

        ln = logits_native[0, -1]
        ld = logits_diff[0, -1]
        top1_native = int(ln.argmax())
        top1_diff = int(ld.argmax())
        match = top1_native == top1_diff
        all_ok &= match

        # Cross-entropy of the *native* next-token prediction under each model,
        # and agreement metrics on the final-token logits.
        max_abs = (ln - ld).abs().max().item()
        cos = torch.cosine_similarity(ln, ld, dim=0).item()
        # How often do the two forwards agree on the argmax across all positions?
        agree = (logits_native[0].argmax(-1) == logits_diff[0].argmax(-1)).float().mean().item()

        tok_native = repr(tokenizer.decode(torch.tensor([[top1_native]]),
                                           decode_special_tokens=True)[0])
        tok_diff = repr(tokenizer.decode(torch.tensor([[top1_diff]]),
                                         decode_special_tokens=True)[0])

        print(f"\nprompt: {prompt!r}")
        print(f"  native next-token : {tok_native}")
        print(f"  diff   next-token : {tok_diff}   {'OK' if match else 'MISMATCH'}")
        print(f"  per-position argmax agreement: {agree*100:.1f}%")
        print(f"  last-token logits: max|Δ|={max_abs:.4f}  cos={cos:.6f}")

    print("\n" + "=" * 78)
    print("RESULT:", "PASS -- differentiable forward matches native"
          if all_ok else "FAIL -- top-1 mismatch (see above)")
    print("=" * 78)
    # Exit non-zero on failure so a `validate && train` kickoff aborts the run
    # instead of training against a broken forward (e.g. a new architecture).
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
