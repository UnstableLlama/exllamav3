# QLoRA on EXL3 — feasibility notes & proof of concept

> Status: **research / proof of concept.** Nothing here is wired into the
> inference path. This documents whether QLoRA-style fine-tuning *can* be
> built on top of EXL3-quantized weights, and provides a minimal,
> gradient-checked building block to prove the foundation is sound.

> **Update (2026-06):** the HF-Trainer integration this doc sketches
> (`hf_qlora.py` + `examples/qlora_train.py`) has been **removed**; it was
> superseded by a transformers-free native forward (`training/native_llama.py`,
> driven by `training/qlora_train_native.py`). The autograd building blocks
> below (`EXL3LoRAFunction`, fused cross-entropy) are unchanged and still
> underpin the native path. Mentions of `attach_qlora` / HF `Trainer` are
> historical.

## The question

Can we fine-tune an EXL3 (QTIP-style trellis) quantized model with QLoRA —
freeze the quantized base weights, train small low-rank adapters?

## The short answer

Yes, the foundation is sound. The commonly assumed blocker — "you can't
backpropagate through a 3-bit trellis code" — **does not apply to QLoRA**,
because QLoRA freezes the base weights and trains only the adapters. The
real work is breadth: ExLlamaV3 is an inference engine with no autograd, so
a training path has to be built alongside the existing forward.

## Why the trellis is not the problem

There are two different things people conflate:

| Task | Backprop *through* the quantizer? | Hard in EXL3? |
|------|-----------------------------------|---------------|
| **QLoRA** (train adapters, freeze base) | **No** — frozen weight is a constant | **No** |
| **QAT** (train the quantized weights)   | **Yes** — through the Viterbi argmax | **Yes** (needs e.g. BCJR-QAT) |

For QLoRA the frozen weight is a constant in the graph. The only gradients
needed are:

- `grad_x = grad_y @ W_effᵀ` — so loss reaches earlier layers, and
- `grad_A`, `grad_B` — to update the adapter,

all ordinary matmuls **once the effective FP16 weight `W_eff` has been
reconstructed from the trellis.** EXL3 already exposes that reconstruction:

- `LinearEXL3.get_weight_tensor()` returns the full effective weight
  `[in, out]` with sign flips (`suh`/`svh`) and Hadamard rotations folded
  in, and
- `LinearEXL3.reconstruct_hgemm()` / `ext.reconstruct(...)` already
  materialise the dequantized weight on the inference path for large
  batches.

This is exactly how bitsandbytes QLoRA works with NF4: dequantize on the
fly, matmul, never differentiate the quantizer. EXL3's advantage is that
its 3–4 bpw weights are far more accurate than NF4, so "QLoRA on EXL3"
means higher-fidelity adapters in the same VRAM.

## What this PoC contains

`exllamav3/training/qlora_linear.py`:

- `reference_forward(...)` — plain-torch ground truth; the dequantized
  weight is detached to a constant and autograd does the rest.
- `EXL3LoRAFunction` — a memory-efficient `autograd.Function` whose
  backward **re-reconstructs** `W_eff` from a `weight_fn` closure instead
  of stashing the full FP16 weight in the saved-activation set. This is the
  shape a real EXL3 training kernel wants: trade a little recompute for a
  lot of activation memory.
- `QLoRALinear` — an `nn.Module` wrapper. Base weight is reconstructed on
  the fly every forward (never copied/updated); only `lora_a`/`lora_b` are
  trainable. `from_exl3_linear()` builds one around a loaded ExLlamaV3
  `Linear` (works for EXL3 and fp16 inners). `B` is zero-initialised so a
  fresh adapter is an exact no-op.

`tests/test_qlora_grad.py` — three tiers:

1. **CPU / float64 `gradcheck`** of the hand-written backward (the real
   proof; needs only torch).
2. **Custom Function == autograd reference** (analytic vs analytic, exact).
3. **Real EXL3 layer on GPU** (opt-in via `--model`): forward parity of
   `x @ get_weight_tensor()` against the layer's own kernel forward, plus
   backward agreement with autograd.

Run:

```bash
python tests/test_qlora_grad.py                  # tiers 1–2, any machine w/ torch
python tests/test_qlora_grad.py --model /path/to/exl3_model   # + tier 3 on GPU
```

## Step 2 — trainable model via the HF integration

The Transformers integration (`exllamav3/integration/transformers.py`)
replaces **only the linear layers** with EXL3 (`Exl3HfLinear`); norms,
attention, RoPE and the LM-head loss stay as stock, autograd-friendly
PyTorch. So the *whole* model becomes trainable the moment the EXL3 linears
are differentiable — no need to reimplement any other backward.

`exllamav3/training/hf_qlora.py` builds on that:

- `Exl3LoRALinear` — wraps a frozen `Exl3HfLinear`, reconstructs the base
  weight on the fly (`get_weight_tensor`), and routes the forward through
  the gradchecked `EXL3LoRAFunction`. Base frozen, only `lora_a`/`lora_b`
  trainable, `B` zero-initialised (no-op at start).
- `attach_qlora(model, r, alpha, target_modules, ...)` — walks an HF model,
  swaps matching EXL3 linears for trainable wrappers, freezes everything
  else, returns the trainable parameter list.
- `save_lora_adapter(...)` — writes a PEFT-format adapter (correct A/B
  transpose + unscaled B) that the inference loader
  (`exllamav3/model/lora.py`) reproduces exactly.

`examples/qlora_train.py` ties it together with HF `Trainer`:
`patch_transformers()` → `from_pretrained` → `attach_qlora` → `Trainer.train`
→ `save_lora_adapter`. (Targets a real GPU + model; not run in the authoring
sandbox.)

`tests/test_qlora_train_loop.py` proves the mechanics end-to-end on CPU with
a mock EXL3 weight: loss decreases, the frozen base and all non-adapter
params are unchanged, only adapters move, and the saved PEFT orientation
reproduces the exact training-time delta. **Verified passing.**

## Step 3 — memory-readiness for 7-8B (checkpointing + fused CE)

Two additions in `exllamav3/training/` make the path usable on a single
consumer GPU at 7-8B scale:

- `prepare_model_for_qlora_training(model)` — enables gradient checkpointing
  (trade compute for activation memory), registers the input-embedding
  require-grad hook (without it, checkpointing silently drops gradients
  because the frozen embedding output doesn't require grad — the same fix
  PEFT's `prepare_model_for_kbit_training` applies), and disables the KV
  cache.
- `fused_ce.py` — `FusedLinearCrossEntropy`, a streaming linear
  cross-entropy that **never materialises the `[tokens, vocab]` logits or
  their gradient**, the single biggest memory spike at large vocab. It walks
  token chunks, computes loss and `grad_hidden`, and recomputes the
  (dequantized, frozen) head weight in the backward rather than storing it
  (a 128k-vocab head can be ~1 GB on its own). `qlora_causal_lm_loss(...)`
  wires it to any model with the standard HF
  `get_decoder()`/`get_output_embeddings()` interface, handling the causal
  shift and head orientation. Assumes a frozen head and no final-logit
  softcapping (Gemma2-style models should use the standard head).

`examples/qlora_train.py` now calls `prepare_model_for_qlora_training` and
uses a `QLoRATrainer` whose `compute_loss` routes through the fused head
(togglable with `--no-fused-ce` / `--no-grad-ckpt`).

`tests/test_fused_ce.py` proves it on CPU: loss and `grad_hidden` match
`F.cross_entropy` exactly (incl. `ignore_index`), results are invariant to
chunk size, the backward passes `gradcheck`, and `qlora_causal_lm_loss`
reproduces HF-style shifted CausalLM loss with gradients flowing into the
backbone. **Verified passing.**

## Demo dataset & before/after

`examples/qlora_train.py` defaults to
[`TeeZee/dolly-15k-pirate-speech`](https://hf.co/datasets/TeeZee/dolly-15k-pirate-speech)
(CC-BY-SA-3.0): the Dolly-15k instruction set with responses rewritten in
pirate speak. The style is deliberately obvious so the effect of fine-tuning
is unmistakable — a successful run makes the model answer with "Arrr", "matey",
"th'", "batten down the hatches", etc. Training uses completion-only masking
(prompt tokens set to `-100`), so the loss is computed only over the pirate
response. `examples/qlora_infer.py` generates from the base vs the adapted
model on the same prompts for a direct side-by-side.

## Status summary

Done and CPU-verified:

- ✅ Differentiable EXL3 linear (frozen base) with gradchecked backward.
- ✅ Trainable model via the HF integration (only linears are EXL3; the rest
  is stock differentiable PyTorch), adapter attach + PEFT-format save.
- ✅ Gradient checkpointing wiring (`prepare_model_for_qlora_training`).
- ✅ Fused linear cross-entropy head (no `[tokens × vocab]` logit spike).
- ✅ Optimizer state — trivial; only adapters train, handled by HF Trainer.

## What is NOT done yet (the rest of the roadmap)

1. **Run on real hardware.** Everything above the EXL3 kernel is exercised by
   CPU tests with mock weights; the GPU integration (`examples/qlora_train.py`,
   tier-3 of the grad test) has not been executed in the authoring sandbox.
   First real step: fine-tune a small EXL3 model end-to-end on a GPU and
   confirm loss + a learned adapter loaded back for inference.
2. **Native (non-HF) differentiable forward** — only needed if you want to
   train without Transformers. The native inference forward runs under
   `@torch.inference_mode` with fused, backward-less kernels; it would have to
   be routed through torch-native equivalents (note `RMSNorm.forward_torch`
   already exists) with flash-attn in training config. The HF path sidesteps
   this entirely.
3. **Model coverage caveats** for the fused head: final-logit softcapping
   (Gemma2) and non-standard head layouts need the standard CE path
   (`--no-fused-ce`).
4. **Multi-GPU**: single-GPU → layer-pipeline (matches the existing device
   split) → FSDP much later (EXL3's packed trellis tensors don't shard like
   bf16 params, so FSDP-QLoRA needs real design work).

## Bottom line

Feasible. The foundational pieces (trellis→FP16 reconstruction, LoRA hooks,
a torch-native RMSNorm) already exist, and the "can't differentiate the
trellis" objection is a non-issue for QLoRA. The remaining work is a whole
differentiable forward plus standard training infrastructure — breadth, not
a single impossible blocker. This PoC nails down the one load-bearing
assumption (correct gradients through a frozen, dequantized EXL3 weight)
with a finite-difference gradcheck.
