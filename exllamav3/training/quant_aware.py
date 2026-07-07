"""
Quantization-aware LoRA training modes for the native QLoRA trainer.

The problem (handoff backlog #1)
--------------------------------
The deploy path for a trained adapter is merge-and-requantize: fold
``ΔW = s·A@B`` into the bf16 weights and re-quantize the merged model to an
EXL3 trellis. Requantization replaces the base's quantization error ``ε``
with a fresh realization ``ε'`` of the same magnitude, so:

  * any part of the trained delta that (implicitly) compensated the specific
    ``ε`` the adapter was trained on is wasted at deploy, and
  * any delta component whose magnitude sits below the quantization-noise
    floor is destroyed outright (the Session-3 "attenuated LoRA" finding,
    which worsens at 2.5-3 bpw).

Ordinary QLoRA training is blind to both: it optimizes against one fixed,
exactly-known ``W_q``. The two modes here make the *training forward* see the
deploy-time uncertainty, so the optimizer is forced to put the adapter's
energy where it survives requantization by construction.

The two operators
-----------------
``noise`` -- pseudo-quantization-noise injection (the NIPQ / QuantNoise idea,
    arXiv:2206.00820): every optimizer micro-batch, the frozen weight served
    to the forward AND its backward recompute is ``W_q + δ`` with fresh
    ``δ ~ N(0, diag(σ²))`` scaled per output channel to the layer's measured
    (or estimated) quantization-error magnitude. Fully differentiable (the
    noise is an additive constant per step -- no STE, no gradient mismatch);
    the adapter can no longer fit the particular error realization of the
    trained-against quant and must learn signal that clears the noise floor
    in expectation. This is the closest differentiable proxy of "the merged
    model will be requantized with an error you cannot know yet".

``ste`` -- delta-quantization straight-through (QA-LoRA's intent transplanted
    to a trellis base): the forward sees ``W_q + Q(ΔW)`` where ``Q`` snaps the
    *effective adapter delta* to a per-output-channel uniform grid whose step
    matches the quantization-error magnitude (``q = √12·σ``, the uniform
    quantizer with error std σ); the backward treats ``Q`` as identity for
    A/B (straight-through) while grad_x flows through the snapped weight the
    forward actually used. Deltas below half a grid step contribute NOTHING
    to the forward -- exactly the deploy behavior -- so the loss can only be
    reduced by delta components big enough to survive. Deterministic (no
    seeding needed): ``Q(0) = 0``, so the mode is function-preserving at
    init for default/pissa/eva.

Why not QA-LoRA's own group-wise operator (arXiv:2309.14717): its exact
merge trick absorbs the adapter delta into the *zero points* of group-wise
affine quantization, requiring A to be constant within each input group.
The EXL3 trellis has no zero points or group scales to absorb anything --
the whole weight is re-fit through Hadamard rotations + Viterbi search --
so the exact-merge construction has no trellis analogue. What CAN be kept
is the objective (train so the quantized merge is what you optimized), which
is what the two operators above target.

σ: where the per-layer error scale comes from
---------------------------------------------
``ref`` (exact, preferred when the original weights are on disk): reconstruct
    ``W_q``, read the original ``W_ref`` (same lazy safetensors reader the
    qerr init uses) and measure ``σ_col = rms(W_ref - W_q)`` per output
    channel over the real (non-padded) region. Padded columns get σ = 0
    (never perturbed).

``heuristic`` (no reference needed): rate-distortion scaling. A near-optimal
    quantizer at K bits/weight on a Gaussian source leaves error variance
    ``≈ σ_w²·2^(-2K)`` (the EXL3/QTIP trellis operates close to this bound),
    so ``σ_col = std(W_q[:, col]) · 2^(-K)`` with K read from the trellis.
    Layers without a K (fp16 inners) are skipped -- no quantization error to
    model. ``--quant-aware-scale`` multiplies either source, so the heuristic
    can be calibrated against a ref-measured run once and reused.

Determinism contract (gradient correctness)
-------------------------------------------
The frozen-weight closure is re-invoked by the grad-checkpoint recompute and
by ``EXL3LoRAFunction.backward``, and the recomputed weight MUST equal the
forward's bit-for-bit (a drifting weight silently corrupts gradients). The
noise is therefore drawn from a generator seeded by (net-level tick, stable
per-layer id): the tick advances once per *grad-enabled* ``net.forward`` call
(``NativeLlamaQLoRA.forward``), so the up-to-3 reconstructions within one
micro-batch's forward+backward all see the same δ, while the next micro-batch
draws fresh noise. No-grad forwards (eval, the DPO/KTO reference and KL
passes) neither advance the tick nor see noise (eval runs in ``net.eval()``,
and the reference forward bypasses the adapter closure entirely).
Consequence: at most one grad-enabled forward may be in flight before its
backward runs -- true for the SFT trainer (fwd+bwd per micro-batch) and for
the preference trainer's single 2·batch-row policy forward.
"""

from __future__ import annotations
from typing import Optional
import torch

from . import backbone

# Distinct odd multipliers decorrelate the (layer, tick) seed stream; the
# exact values are arbitrary, they only need to be fixed (resume/replay give
# the same noise for the same tick) and odd (bijective mod 2^64).
_SEED_LAYER_MULT = 0x9E3779B97F4A7C15
_SEED_TICK_MULT = 0xBF58476D1CE4E5B9


def qa_seed(base_seed: int, layer_id: int, tick: int) -> int:
    """Deterministic 63-bit seed for (layer, micro-batch tick)."""
    s = (int(base_seed) + layer_id * _SEED_LAYER_MULT + tick * _SEED_TICK_MULT)
    return s & 0x7FFF_FFFF_FFFF_FFFF


@torch.no_grad()
def heuristic_sigma(wrapper) -> Optional[torch.Tensor]:
    """
    Estimated per-output-channel quantization-error std ``[out]`` (fp32, on
    the layer's device) from the rate-distortion heuristic, or ``None`` for a
    layer with no quantized storage (fp16 inner -- nothing to model).
    """
    k = backbone.linear_quant_bits(wrapper.linear)
    if k is None:
        return None
    base = backbone.frozen_weight_closure(wrapper.linear, torch.float32)()
    sigma = base.std(dim=0) * (2.0 ** -k)
    return sigma.contiguous()


@torch.no_grad()
def ref_sigma(wrapper, refs) -> torch.Tensor:
    """
    Measured per-output-channel quantization-error std ``[out]`` (fp32, on the
    layer's device): rms over the real input rows of ``W_ref - W_q``, exactly
    the error the merge-and-requantize step will re-roll. Padded output
    columns (outside the reference model) get σ = 0. ``refs`` is a
    ``lora_init.RefWeights`` over the ORIGINAL unquantized model dir.
    """
    name = f"{wrapper.key}.weight"
    ref = refs.get(name)
    if ref is None:
        raise SystemExit(
            f"--quant-aware ref model: tensor {name!r} not found -- is this "
            f"the original model for this quant?")
    base = backbone.frozen_weight_closure(wrapper.linear, torch.float32)()
    ref = ref.t().to(base.device, torch.float32)     # HF [out,in] -> [in,out]
    in_hf, out_hf = ref.shape
    in_pad, out_pad = base.shape
    if in_hf > in_pad or out_hf > out_pad:
        raise SystemExit(
            f"--quant-aware: reference weight {name} [{in_hf},{out_hf}] "
            f"larger than the quantized layer [{in_pad},{out_pad}] -- wrong "
            f"reference model?")
    err = ref - base[:in_hf, :out_hf]
    sigma = torch.zeros(out_pad, dtype=torch.float32, device=base.device)
    # rms (not centered std): the systematic component of the error is part
    # of what a requantize re-rolls, so it belongs in the perturbation scale.
    sigma[:out_hf] = err.pow(2).mean(dim=0).sqrt()
    return sigma.contiguous()


@torch.no_grad()
def configure_quant_aware(
    net,
    mode: str,
    scale: float = 1.0,
    ref_model_dir: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> None:
    """
    Enable a quantization-aware training mode (``"noise"`` or ``"ste"``) on
    every adapted (r > 0) per-linear wrapper of a ``NativeLlamaQLoRA``; the
    non-adapted frozen layers are left exact (their weights are untouched by
    the merge, and perturbing them adds variance the adapter cannot answer).
    ``mode in (None, "", "none")`` disables. σ comes from ``ref_model_dir``
    (measured, exact) when given, else the K-bits heuristic; ``scale``
    multiplies it. Call any time after construction (and again after a
    resume -- this is run configuration, not learned state; nothing here is
    persisted in checkpoints).
    """
    wrappers = [w for w in net._wrappers if w.r > 0]
    if mode in (None, "", "none"):
        for w in wrappers:
            w.qa_mode = ""
            w.qa_sigma = None
        net._qa_state = None
        return
    assert mode in ("noise", "ste"), f"unknown quant_aware mode {mode!r}"
    assert scale > 0.0, "--quant-aware-scale must be > 0"

    refs = None
    if ref_model_dir:
        from .lora_init import RefWeights
        refs = RefWeights(ref_model_dir)

    state = {"tick": 0, "seed": int(seed)}
    n_on, n_skip = 0, 0
    rel: list[float] = []
    for i, w in enumerate(wrappers):
        sigma = ref_sigma(w, refs) if refs is not None else heuristic_sigma(w)
        if sigma is None:
            n_skip += 1
            w.qa_mode = ""
            w.qa_sigma = None
            continue
        base_rms = float(backbone.frozen_weight_closure(
            w.linear, torch.float32)().pow(2).mean().sqrt())
        rel.append(float(sigma.mean()) * scale / max(base_rms, 1e-12))
        w.qa_sigma = (sigma * scale).contiguous()
        w.qa_mode = mode
        w.qa_layer_id = i
        w.qa_state = state
        n_on += 1

    net._qa_state = state if n_on else None
    if verbose and n_on:
        rel.sort()
        src = "measured vs ref model" if refs is not None else "K-bits heuristic"
        print(f" -- quant-aware {mode}: {n_on} adapted linears perturbed "
              f"({src}, scale {scale:g}); relative error scale "
              f"median {rel[len(rel) // 2] * 100:.2f}%, "
              f"min {rel[0] * 100:.2f}%, max {rel[-1] * 100:.2f}%"
              + (f"; {n_skip} unquantized layers skipped" if n_skip else ""))
    elif verbose:
        print(" -- quant-aware: no quantized adapted linears found; mode is a "
              "no-op (fp16 base?)")


def wrap_weight_closure(wrapper, base_fn):
    """
    Compose the quant-aware perturbation onto a wrapper's frozen-weight
    closure (called from ``DiffLinear._weight_closure``; ``base_fn`` already
    includes any pissa residual). Returns ``base_fn`` untouched when the mode
    is off or the wrapper is in eval mode -- eval / validation always see the
    exact weights.
    """
    mode = getattr(wrapper, "qa_mode", "")
    sigma = getattr(wrapper, "qa_sigma", None)
    if not mode or sigma is None or not wrapper.training:
        return base_fn

    if mode == "noise":
        # Capture the tick NOW (closure-creation time = this micro-batch's
        # forward): the checkpoint recompute re-runs DiffLinear.forward and
        # captures the then-current tick, which the determinism contract in
        # the module docstring guarantees is the same one.
        seed = qa_seed(wrapper.qa_state["seed"], wrapper.qa_layer_id,
                       wrapper.qa_state["tick"])

        def noisy_fn():
            w = base_fn()
            g = torch.Generator(device=w.device)
            g.manual_seed(seed)
            noise = torch.randn(w.shape, generator=g, device=w.device,
                                dtype=w.dtype)
            # w + noise * sigma_col, out of place (an fp16 inner's base_fn can
            # return the stored weight itself; never write into it).
            return torch.addcmul(w, noise, sigma.to(w.dtype))

        return noisy_fn

    # ste: forward weight = base + (Q(Δ) - Δ) with Δ = the effective adapter
    # delta; EXL3LoRAFunction then adds the un-snapped low-rank term back, so
    # the total forward is base + Q(Δ) while A/B gradients flow through the
    # identity (straight-through) and grad_x through the snapped weight.
    a, b, s = wrapper.lora_a, wrapper.lora_b, wrapper.scale
    a0, b0 = wrapper.init_a0, wrapper.init_b0

    def ste_fn():
        w = base_fn()
        with torch.no_grad():
            ad = a.detach().to(w.dtype)
            bd = b.detach().to(w.dtype)
            delta = (ad @ bd).mul_(s)
            if a0 is not None:
                # pissa: the delta that reaches a merge is s·(AB - A0B0).
                delta = torch.addmm(delta, a0.to(w.dtype), b0.to(w.dtype),
                                    alpha=-s)
            q = (sigma.to(w.dtype) * (12.0 ** 0.5))         # uniform step
            qs = torch.where(q > 0, q, torch.ones_like(q))  # guard σ=0 cols
            t = delta.div_(qs)                               # reuse the temp
            resid = (t.round() - t) * q                      # σ=0 -> exactly 0
        return w + resid

    return ste_fn
