# QLoRA on EXL3 — feasibility notes & proof of concept

> Status: **research / proof of concept.** Nothing here is wired into the
> inference path. This documents whether QLoRA-style fine-tuning *can* be
> built on top of EXL3-quantized weights, and provides a minimal,
> gradient-checked building block to prove the foundation is sound.

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

## What is NOT done yet (the rest of the roadmap)

This proves one linear layer. A full training path additionally needs:

1. **A differentiable model forward.** The inference forward runs under
   `@torch.inference_mode` and uses fused, backward-less kernels
   (`ext.rms_norm`, fused MLP, `exl3_mgemm`, `had_r_128`, softcap, paged
   flash-attn). Training mode must disable `inference_mode` and route
   through torch-native / autograd-friendly equivalents (note `RMSNorm`
   already has a `forward_torch`), with flash-attn in its training (varlen,
   backward-enabled) configuration.
2. **Gradient checkpointing** — essentially mandatory given per-layer weight
   reconstruction.
3. **Fused / cut cross-entropy** at the LM head to avoid the
   `[tokens × vocab]` logit+grad memory spike.
4. **Optimizer state** — trivial in size (adapters only), but wire it up.
5. **Adapter save** in PEFT format (the inference `LoRA` loader in
   `exllamav3/model/lora.py` already handles load; mirror it for save).
6. **Multi-GPU**: start single-GPU → layer-pipeline (matches the existing
   device split) → FSDP much later (EXL3's packed trellis tensors don't
   shard like bf16 params, so FSDP-QLoRA needs real design work).

A pragmatic alternative to writing a trainer: expose EXL3 linears as
autograd-friendly `nn.Module`s (this PoC is step one) and reuse the
existing HF Transformers integration + PEFT + `Trainer`/axolotl for the
loop, owning only the EXL3 forward/backward.

## Bottom line

Feasible. The foundational pieces (trellis→FP16 reconstruction, LoRA hooks,
a torch-native RMSNorm) already exist, and the "can't differentiate the
trellis" objection is a non-issue for QLoRA. The remaining work is a whole
differentiable forward plus standard training infrastructure — breadth, not
a single impossible blocker. This PoC nails down the one load-bearing
assumption (correct gradients through a frozen, dequantized EXL3 weight)
with a finite-difference gradcheck.
