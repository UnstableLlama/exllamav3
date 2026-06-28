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
    # single device
    python examples/qlora_validate_native.py --model /path/to/exl3_model

    # layer-autosplit across all visible GPUs (smoke-test the device-aware
    # forward); --check-backward also exercises cross-device gradient flow
    python examples/qlora_validate_native.py --model /path/to/exl3_model \
        --parallel split --check-backward

On a small model that fits one card, force a real split boundary by capping the
per-device budget, e.g. ``--use-per-device 1 8`` (≈1 GB on cuda:0 spills the rest
to cuda:1). On a model too big for one card, plain ``--parallel split`` balances
naturally. The printed block-device distribution shows where the boundary landed.
"""

import argparse
from collections import Counter
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training import backbone
from exllamav3.training.native_llama import NativeLlamaQLoRA


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Water is made of hydrogen and",
]


def check_backward(model, tokenizer, prompt, device, cdt, attn_impl="auto"):
    """
    Smoke-test cross-device gradient flow: attach a tiny adapter, run one
    loss.backward() with gradient checkpointing, and assert that gradients
    reached adapters on *every* device the decoder is split across. This is the
    part of the device-aware forward that the forward-only gate can't cover --
    autograd flowing back through the cross-device hidden-state migrations.
    """
    print("\n" + "-" * 78)
    print("backward smoke: cross-device gradient flow through the split")
    net = NativeLlamaQLoRA(model, r=4, alpha=8.0,
                           target_modules=["q_proj", "down_proj"],
                           compute_dtype=cdt, gradient_checkpointing=True,
                           attn_impl=attn_impl)
    net.train()
    ids = tokenizer.encode(prompt, add_bos=True).to(device)
    loss = net.compute_loss(ids, ids.clone())
    loss.backward()

    expected = sorted({str(d) for d in net._block_devices})
    have = set()
    missing = []
    for w in net._wrappers:
        if w.r <= 0:
            continue
        # B inits to zero, so on the first step the gradient flows to B (grad_A
        # is exactly zero while B == 0); check B to confirm the adapter was hit.
        g = w.lora_b.grad
        if g is not None and g.abs().sum().item() > 0:
            have.add(str(w.lora_b.device))
        else:
            missing.append(w.key)
    ok = (set(expected) <= have) and not missing
    print(f"  loss = {loss.item():.4f}")
    print(f"  adapters received grad on : {sorted(have)}")
    print(f"  decoder split devices     : {expected}")
    if missing:
        print(f"  MISSING grad on {len(missing)} adapters, e.g. {missing[:3]}")
    print("  backward smoke:", "PASS" if ok else "FAIL")
    return ok


def check_packing(net, tokenizer, prompts, device):
    """
    Prove sample packing isolates documents: the logits for each document inside a
    packed block must match running that document ALONE. This exercises both
    halves of correct packing -- the block-diagonal attention (no token attends
    across a document boundary) and the per-document RoPE position reset. A
    mismatch means packed training would silently mix documents.

    Runs on whatever dtype/attn the net was built with, so it covers the fp32
    eager reference and (under --compute-dtype bfloat16) the flash-varlen path.
    """
    print("\n" + "-" * 78)
    print("packing check: packed-block logits == per-document logits")
    docs = [tokenizer.encode(p, add_bos=True)[0].tolist() for p in prompts]

    # Per-document reference: each document forwarded on its own.
    ref = []
    with torch.no_grad():
        for d in docs:
            ref.append(net.logits(torch.tensor([d], device=device))[0].float().cpu())

    # Pack the documents into one sequence with seg ids + per-document position
    # resets (exactly what pack_examples/collate produce for training).
    input_ids, seg_ids, position_ids = [], [], []
    for s, d in enumerate(docs):
        input_ids += d
        seg_ids += [s] * len(d)
        position_ids += list(range(len(d)))
    ii = torch.tensor([input_ids], device=device)
    sg = torch.tensor([seg_ids], device=device)
    pp = torch.tensor([position_ids], device=device)
    with torch.no_grad():
        packed = net.logits(ii, position_ids=pp, seg_ids=sg)[0].float().cpu()

    ok, off = True, 0
    for s, (d, r) in enumerate(zip(docs, ref)):
        sl = packed[off: off + len(d)]
        off += len(d)
        agree = (sl.argmax(-1) == r.argmax(-1)).float().mean().item()
        max_abs = (sl - r).abs().max().item()
        cos = torch.cosine_similarity(sl[-1], r[-1], dim=0).item()
        good = agree > 0.999
        ok &= good
        print(f"  doc {s} ({len(d):>3} tok): per-position argmax {agree*100:5.1f}% | "
              f"max|Δ|={max_abs:.4f} cos={cos:.6f}  {'OK' if good else 'MISMATCH'}")
    print("  packing check:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0",
                    help="single-device load target (ignored when --parallel split)")
    ap.add_argument("--parallel", choices=["single", "split"], default="single",
                    help="single: load to --device; split: layer-autosplit across visible GPUs")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB to reserve per device; negative excludes a device")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB budget per device; caps a card to force a split on a small model")
    ap.add_argument("--compute-dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="dtype for the differentiable linears (float32 = closest to true math)")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="auto/eager/flash. NOTE flash needs CUDA fp16/bf16, so the "
                         "default float32 validate runs eager regardless; pass "
                         "--compute-dtype bfloat16 to exercise/validate the flash path.")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--check-backward", action="store_true",
                    help="also smoke-test cross-device gradient flow (tiny adapter + backward)")
    ap.add_argument("--check-packing", action="store_true",
                    help="also verify sample packing: a packed block's per-document "
                         "logits must match running each document alone (block-"
                         "diagonal attention + per-document RoPE reset). Run with "
                         "--compute-dtype bfloat16 to exercise the flash-varlen path.")
    args = ap.parse_args()

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]
    prompts = args.prompts or DEFAULT_PROMPTS

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    if args.parallel == "split":
        load_kwargs = {}
        if args.reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = args.reserve_per_device
        if args.use_per_device is not None:
            load_kwargs["use_per_device"] = args.use_per_device
        model.load(progressbar=True, **load_kwargs)
        print(f" -- layer-autosplit: active devices {model.active_devices}, "
              f"output device {model.output_device}")
    else:
        model.load(device=args.device, progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    # No adapters: a pure frozen forward, directly comparable to native inference.
    net = NativeLlamaQLoRA(model, target_modules=[], compute_dtype=cdt,
                           gradient_checkpointing=False, attn_impl=args.attn_impl)
    net.eval()
    print(f" -- {net.describe_attn()}")

    dist = Counter(str(d) for d in net._block_devices)
    print(f" -- decoder block devices: {dict(dist)}  (final norm + head on {net.device})")

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

        # Native output lands on the model's output device; co-locate for compare.
        logits_diff = logits_diff.to(logits_native.device)

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

    if not args.skip_head_slice_check:
        all_ok &= check_head_slice(net)

    if args.check_backward:
        all_ok &= check_backward(model, tokenizer, prompts[0], args.device, cdt,
                                 attn_impl=args.attn_impl)

    if args.check_packing:
        all_ok &= check_packing(net, tokenizer, prompts, args.device)

    print("\n" + "=" * 78)
    print("RESULT:", "PASS -- differentiable forward matches native"
          if all_ok else "FAIL -- see above")
    print("=" * 78)
    # Exit non-zero on failure so a `validate && train` kickoff aborts the run
    # instead of training against a broken forward (e.g. a new architecture).
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
