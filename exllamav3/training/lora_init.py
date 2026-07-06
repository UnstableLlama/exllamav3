"""
LoRA initialization strategies for the native QLoRA trainer.

Why initialization matters here
-------------------------------
The trainer's runs are short (tiny SFT sets, a few epochs): with the PEFT
default (kaiming A, B = 0) the adapter spends a meaningful fraction of the
whole run just growing off the ground (|B| is typically still climbing at the
final step). Initialization methods attack exactly that. Two are implemented,
both SVD-based and both computed once at startup from weights the trainer
already reconstructs:

``pissa`` -- Principal Singular values and Singular vectors Adaptation
    (Meng et al. 2024, NeurIPS spotlight). Factor the top-r principal
    component out of the frozen base weight into the *trainable* adapter, so
    training updates the directions that matter most first. The reference
    formulation retrains against a residual base ``W_res = W - principal``;
    our base is a frozen EXL3 trellis that cannot be rewritten, so the
    residual is realized as a frozen rank-r offset instead:

        effective W  =  W_q  +  s·(A@B - A0@B0)
                     = (W_q - s·A0@B0) + s·A@B      # == W_res + s·A@B

    with A/B trainable and initialized to A0/B0 (so the model starts exactly
    at the base). The offset is folded into the frozen-weight closure (see
    ``DiffLinear``), costing one rank-r matmul per weight reconstruction --
    noise next to the trellis dequant it rides on.

``qerr`` -- quantization-error-correction init (the LoftQ idea, single-shot).
    Initialize the adapter to the top-r SVD of the *quantization error*
    ``E = W_ref - W_q`` (W_ref = the original unquantized weight), so at step
    0 the model is the closest rank-r approximation of the ORIGINAL bf16
    model and training proceeds from there. No offset, no bookkeeping: the
    nonzero start is the point. Needs ``--init-ref-model`` (the original HF
    model dir). The lower the bitrate, the bigger E -- this is the natural
    companion of low-bpw EXL3 training.

Both divide the factors by sqrt(scale) so the adapter's *scaled* contribution
``s·A@B`` equals the intended SVD component regardless of alpha/r/rslora.

SVD determinism note: ``svd_niter > 0`` uses randomized ``torch.svd_lowrank``
(seconds per model, the PiSSA "fast SVD" recipe); it is NOT deterministic
across runs, which is why pissa checkpoints persist A0/B0 in a sidecar file
(``pissa_init.safetensors``) instead of recomputing them on resume.
"""

from __future__ import annotations
from typing import Optional
import glob
import json
import os
import time
import torch

from . import backbone


@torch.no_grad()
def principal_factors(
    w: torch.Tensor,
    r: int,
    niter: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Rank-r principal factorization of ``w`` ([in, out], float32).

    Returns ``(a, b, captured, total)`` with ``a`` [in, r], ``b`` [r, out],
    ``a @ b`` = the top-r SVD approximation of ``w``, ``captured`` = the
    approximation's squared Frobenius mass and ``total`` = ``||w||_F^2``
    (their ratio is the explained-variance fraction). ``niter > 0`` selects
    randomized SVD (``torch.svd_lowrank`` with ``niter`` subspace iterations,
    slight oversampling); ``niter == 0`` computes the exact full SVD.
    """
    assert w.dim() == 2
    assert 0 < r <= min(w.shape), f"rank {r} > min dim {min(w.shape)}"
    w = w.float()
    if niter > 0:
        q = min(r + 8, min(w.shape))
        u, s, v = torch.svd_lowrank(w, q=q, niter=niter)
        u, s, v = u[:, :r], s[:r], v[:, :r]
    else:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u, s, v = u[:, :r], s[:r], vh[:r].t()
    sq = s.clamp_min(0).sqrt()
    a = (u * sq).contiguous()                 # [in, r]
    b = (v * sq).t().contiguous()             # [r, out]
    captured = float((s * s).sum())
    total = float((w * w).sum())
    return a, b, captured, total


class RefWeights:
    """
    Lazy reader for the ORIGINAL (unquantized) model's weights, by HF tensor
    name, from a local safetensors model dir (single file or sharded with
    ``model.safetensors.index.json``). Used by qerr init to form the
    quantization error; nothing is loaded until a tensor is requested.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self._name_to_file: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        index = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf8") as f:
                weight_map = json.load(f)["weight_map"]
            for name, fn in weight_map.items():
                self._name_to_file[name] = os.path.join(directory, fn)
        else:
            from safetensors import safe_open
            files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
            if not files:
                raise FileNotFoundError(
                    f"--init-ref-model: no *.safetensors under {directory}")
            for fn in files:
                with safe_open(fn, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        self._name_to_file[name] = fn

    def get(self, name: str) -> Optional[torch.Tensor]:
        from safetensors import safe_open
        fn = self._name_to_file.get(name)
        if fn is None:
            return None
        h = self._handles.get(fn)
        if h is None:
            h = safe_open(fn, framework="pt", device="cpu")
            self._handles[fn] = h
        return h.get_tensor(name)


@torch.no_grad()
def apply_init_lora(
    net,
    mode: str,
    ref_model_dir: Optional[str] = None,
    svd_niter: int = 16,
    verbose: bool = True,
) -> None:
    """
    Apply an SVD-based LoRA init to every per-linear target wrapper of a
    ``NativeLlamaQLoRA``. ``mode`` is ``"pissa"`` or ``"qerr"`` (``"default"``
    is a no-op -- the constructor's kaiming/zeros already happened).

    Embed/head LoRA (``lora_embed``/``lora_head``) keeps the default init:
    pissa on the embedding is ill-defined (token-indexed A) and the head is
    frozen fp16, not trellis-quantized, so qerr has no error to correct.
    """
    if mode in (None, "", "default"):
        return
    assert mode in ("pissa", "qerr"), f"unknown init_lora mode {mode!r}"

    refs = None
    if mode == "qerr":
        if not ref_model_dir:
            raise SystemExit(
                "--init-lora qerr needs --init-ref-model (the ORIGINAL "
                "unquantized HF model dir) to form the quantization error.")
        refs = RefWeights(ref_model_dir)

    t0 = time.time()
    n = 0
    var_fracs: list[float] = []
    for w in net._wrappers:
        if w.r <= 0:
            continue
        # Full-precision reconstruction of the frozen weight, on its device.
        base = backbone.frozen_weight_closure(w.linear, torch.float32)()

        if mode == "pissa":
            a0, b0, cap, tot = principal_factors(base, w.r, niter=svd_niter)
            root_s = w.scale ** 0.5
            a0 = a0 / root_s
            b0 = b0 / root_s
            w.lora_a.copy_(a0)
            w.lora_b.copy_(b0)
            w.set_init_offset(a0, b0)
        else:  # qerr
            name = f"{w.key}.weight"
            ref = refs.get(name)
            if ref is None:
                raise SystemExit(
                    f"--init-lora qerr: tensor {name!r} not found in "
                    f"{ref_model_dir} -- is this the original model for "
                    f"this quant?")
            ref = ref.t().to(base.device, torch.float32)   # HF [out,in] -> [in,out]
            in_hf, out_hf = ref.shape
            in_pad, out_pad = base.shape
            if in_hf > in_pad or out_hf > out_pad:
                raise SystemExit(
                    f"--init-lora qerr: reference weight {name} "
                    f"[{in_hf},{out_hf}] larger than the quantized layer "
                    f"[{in_pad},{out_pad}] -- wrong reference model?")
            err = -base
            err[:in_hf, :out_hf] += ref
            # Never "correct" the padding region: padded input rows/output
            # cols are outside the real model and must stay untouched.
            if in_hf < in_pad:
                err[in_hf:, :] = 0
            if out_hf < out_pad:
                err[:, out_hf:] = 0
            a, b, cap, tot = principal_factors(err, w.r, niter=svd_niter)
            root_s = w.scale ** 0.5
            w.lora_a.copy_(a / root_s)
            w.lora_b.copy_(b / root_s)
        del base
        var_fracs.append(cap / max(tot, 1e-30))
        n += 1

    net.init_lora = mode
    if verbose and n:
        vf = sorted(var_fracs)
        what = ("principal weight variance" if mode == "pissa"
                else "quantization-error variance")
        print(f" -- init-lora {mode}: {n} adapters initialized in "
              f"{time.time() - t0:.1f}s (rank-{net.r} captured {what}: "
              f"median {vf[len(vf) // 2] * 100:.1f}%, "
              f"min {vf[0] * 100:.1f}%, max {vf[-1] * 100:.1f}%)")
