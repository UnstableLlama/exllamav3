"""
CPU tests for the SVD-based LoRA inits (exllamav3/training/lora_init.py):

  * principal_factors recovers an exactly low-rank matrix (and its
    explained-variance accounting is coherent).
  * pissa: DiffLinear with the frozen offset is function-preserving at step 0
    (the effective delta s·(AB - A0B0) starts at exactly zero), and once B
    moves, the forward matches the residual-base math.
  * qerr: apply_init_lora against a real (temp-dir) reference safetensors
    reconstructs a synthetic low-rank quantization error exactly, respects
    the alpha/r scale folding, and never touches the padding region.
  * pissa save/load: adapter_model.safetensors carries the CONVERTED rank-2r
    standard-LoRA delta (correct for external consumers) while the sidecar
    round-trips the exact fp32 training state including the offsets.
  * pissa offsets are stored on-device in the compute dtype (VRAM) with exact
    fp32 masters kept for save/export.
  * eva: the streaming sketch recovers the activations' top right-singular
    subspace; the apply path shares one sketch across same-input sites
    (q/k/v), drops pad-token rows, keeps B at zero (bit-exact function
    preservation) and demands a data pre-pass.

All fp32 on CPU; no GPU, no real model.
"""

import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# test_native_llama sets up the ext-free "exl3train" package shim (the
# training modules import cleanly without the CUDA extension) and provides
# MockLinear; reuse both so DiffLinear is the same class object.
from test_native_llama import MockLinear, _load, _nl

_li = _load("lora_init")
principal_factors = _li.principal_factors
apply_init_lora = _li.apply_init_lora
EvaSketch = _li.EvaSketch

DiffLinear = _nl.DiffLinear


def _rank_k(in_f, out_f, k, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(in_f, k, generator=g)
    v = torch.randn(k, out_f, generator=g)
    return scale * (u @ v)


def test_principal_factors_exact():
    torch.manual_seed(0)
    w = _rank_k(48, 40, 5, seed=1)
    for niter in (0, 16):
        a, b, cap, tot = principal_factors(w, 5, niter=niter)
        assert a.shape == (48, 5) and b.shape == (5, 40)
        err = (a @ b - w).norm() / w.norm()
        assert err < 1e-4, f"rank-5 recovery failed (niter={niter}): rel {err:.2e}"
        assert abs(cap - tot) / tot < 1e-4, "explained variance should be ~100%"
    # Truncation below the true rank captures strictly less.
    a, b, cap, tot = principal_factors(w, 2, niter=0)
    assert cap < tot * 0.999
    assert (a @ b - w).norm() > 0
    print("[lora_init] principal_factors exact/truncated recovery PASSED")


def _pissa_wrapper(in_f=32, out_f=24, r=4, alpha=2.0, seed=3):
    """DiffLinear with a hand-applied pissa init (offset installed)."""
    lin = MockLinear(in_f, out_f, key="model.layers.0.self_attn.q_proj",
                     dtype=torch.float32)
    dl = DiffLinear(lin, r=r, alpha=alpha, compute_dtype=torch.float32)
    w = lin.inner.get_weight_tensor()
    a0, b0, _, _ = principal_factors(w, r, niter=0)
    root_s = dl.scale ** 0.5
    a0, b0 = a0 / root_s, b0 / root_s
    with torch.no_grad():
        dl.lora_a.copy_(a0)
        dl.lora_b.copy_(b0)
    dl.set_init_offset(a0, b0)
    return dl, lin


def test_pissa_function_preserving_and_residual_math():
    torch.manual_seed(0)
    dl, lin = _pissa_wrapper()
    w = lin.inner.get_weight_tensor()
    x = torch.randn(5, dl.in_features)

    # Step 0: adapter == offset, so the forward must equal the pure base.
    y = dl(x)
    ref = x @ w
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"pissa step-0 not function-preserving: rel {rel:.2e}"

    # After B moves, forward = x @ (W - s·A0B0) + s·x@A@B (the residual base).
    with torch.no_grad():
        dl.lora_b.add_(0.05 * torch.randn_like(dl.lora_b))
    y = dl(x)
    s = dl.scale
    ref = (x @ (w - s * (dl.init_a0 @ dl.init_b0))
           + s * (x @ dl.lora_a) @ dl.lora_b)
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"pissa residual-base forward mismatch: rel {rel:.2e}"
    print("[lora_init] pissa function-preservation + residual math PASSED")


def test_qerr_init_reconstructs_error():
    torch.manual_seed(0)
    in_pad, out_pad = 16, 12       # padded (quantized-layer) dims
    in_hf, out_hf = 14, 10         # true (reference-model) dims
    r, k_err = 8, 4                # adapter rank > true error rank -> exact
    key = "model.layers.0.self_attn.q_proj"

    lin = MockLinear(in_pad, out_pad, key=key, dtype=torch.float32)
    w_q = lin.inner.get_weight_tensor()
    err = torch.zeros(in_pad, out_pad)
    err[:in_hf, :out_hf] = _rank_k(in_hf, out_hf, k_err, seed=7, scale=0.1)
    # Reference weight = quantized + error, HF orientation [out, in], unpadded.
    ref_hf = (w_q + err)[:in_hf, :out_hf].t().contiguous()

    dl = DiffLinear(lin, r=r, alpha=2 * r, compute_dtype=torch.float32)  # s = 2
    net = types.SimpleNamespace(_wrappers=[dl], r=r, init_lora="default")

    with tempfile.TemporaryDirectory() as td:
        from safetensors.torch import save_file
        save_file({f"{key}.weight": ref_hf},
                  os.path.join(td, "model.safetensors"))
        apply_init_lora(net, "qerr", ref_model_dir=td, svd_niter=0,
                        verbose=False)

    assert net.init_lora == "qerr"
    assert dl.init_a0 is None, "qerr must not install a frozen offset"
    delta = dl.scale * (dl.lora_a @ dl.lora_b)
    rel = (delta - err).norm() / err.norm()
    assert rel < 1e-4, f"qerr delta != quantization error: rel {rel:.2e}"
    # Padding region untouched even though -W_q was nonzero there.
    assert delta[in_hf:, :].abs().max() < 1e-6
    assert delta[:, out_hf:].abs().max() < 1e-6
    # And the forward now reproduces the (rank-r-repaired) reference model.
    x = torch.randn(5, in_pad)
    y = dl(x)
    ref = x @ (w_q + err)
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"qerr step-0 forward != repaired model: rel {rel:.2e}"
    print("[lora_init] qerr error-reconstruction + padding guard PASSED")


def _headless_net(wrappers, r, alpha):
    net = _nl.NativeLlamaQLoRA.__new__(_nl.NativeLlamaQLoRA)
    nn.Module.__init__(net)
    net._wrappers = wrappers
    net.r, net.lora_alpha, net.use_rslora = r, float(alpha), False
    net.embed_weight = net.head_weight = None
    net.embed_lora_a = net.head_lora_a = None
    net.lora_embed = net.lora_head = False
    net.tie_word_embeddings = False
    return net


def test_pissa_save_convert_and_sidecar_roundtrip():
    torch.manual_seed(0)
    r, alpha = 4, 2.0
    dl1, lin1 = _pissa_wrapper(seed=3)
    # A second wrapper with a different key, sharing the init recipe.
    lin2 = MockLinear(32, 24, key="model.layers.0.self_attn.v_proj",
                      dtype=torch.float32)
    dl2 = DiffLinear(lin2, r=r, alpha=alpha, compute_dtype=torch.float32)
    a0, b0, _, _ = principal_factors(lin2.inner.get_weight_tensor(), r, niter=0)
    a0, b0 = a0 / dl2.scale ** 0.5, b0 / dl2.scale ** 0.5
    with torch.no_grad():
        dl2.lora_a.copy_(a0)
        dl2.lora_b.copy_(b0)
    dl2.set_init_offset(a0, b0)

    # "Train": move both Bs off the offset.
    with torch.no_grad():
        for dl in (dl1, dl2):
            dl.lora_b.add_(0.03 * torch.randn_like(dl.lora_b))

    net = _headless_net([dl1, dl2], r, alpha)
    net.init_lora = "pissa"

    with tempfile.TemporaryDirectory() as td:
        net.save_adapter(td)
        from safetensors.torch import load_file
        import json
        cfg = json.load(open(os.path.join(td, "adapter_config.json")))
        assert cfg["r"] == 2 * r and cfg["lora_alpha"] == 2 * r
        assert cfg["init_lora"] == "pissa" and cfg["init_lora_r"] == r

        # Converted export: a standard loader applying scale alpha/r = 1.0
        # must land on the true delta s·(AB - A0B0), to fp16 accuracy.
        state = load_file(os.path.join(td, "adapter_model.safetensors"))
        for dl in (dl1, dl2):
            key = f"base_model.model.{dl.key}"
            a = state[f"{key}.lora_A.weight"].t().float()   # [in, 2r]
            b = state[f"{key}.lora_B.weight"].t().float()   # [2r, out]
            assert a.shape == (dl.in_features, 2 * r)
            got = a @ b
            want = dl.scale * (dl.lora_a @ dl.lora_b
                               - dl.init_a0 @ dl.init_b0)
            denom = max(want.norm().item(), 1e-6)
            assert (got - want).norm().item() / denom < 5e-3, \
                f"converted export delta mismatch for {dl.key}"

        # Sidecar resume: a FRESH default-init net restores the exact fp32
        # state (adapters + offsets) and computes the identical forward.
        dl1b = DiffLinear(MockLinear(32, 24, key=dl1.key, dtype=torch.float32),
                          r=r, alpha=alpha, compute_dtype=torch.float32)
        dl2b = DiffLinear(MockLinear(32, 24, key=dl2.key, dtype=torch.float32),
                          r=r, alpha=alpha, compute_dtype=torch.float32)
        # Resume must not depend on rebuilding from the same base weights --
        # copy them so forwards are comparable.
        dl1b.linear.inner._w.copy_(lin1.inner.get_weight_tensor())
        dl2b.linear.inner._w.copy_(lin2.inner.get_weight_tensor())
        net2 = _headless_net([dl1b, dl2b], r, alpha)
        n = net2.load_adapter(td)
        assert n == 2 and net2.init_lora == "pissa"
        for orig, res in ((dl1, dl1b), (dl2, dl2b)):
            assert torch.equal(orig.lora_a, res.lora_a)
            assert torch.equal(orig.lora_b, res.lora_b)
            assert torch.equal(orig.init_a0, res.init_a0)
            assert torch.equal(orig.init_b0, res.init_b0)
            x = torch.randn(3, orig.in_features)
            assert torch.allclose(orig(x), res(x), atol=1e-6)
    print("[lora_init] pissa converted export + sidecar resume PASSED")


def _orthonormal(in_f, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(in_f, k, generator=g))[0]


def _subspace_gap(v_est, v_ref):
    """Frobenius distance between the column-span projectors."""
    return ((v_est @ v_est.t()) - (v_ref @ v_ref.t())).norm().item()


def test_eva_sketch_recovers_subspace():
    torch.manual_seed(0)
    in_f, r, n = 40, 5, 600
    g = torch.Generator().manual_seed(11)
    basis = _orthonormal(in_f, r, seed=12)                      # [in, r]
    z = torch.randn(n, r, generator=g) * torch.tensor([8., 6., 5., 4., 3.])
    x = z @ basis.t() + 0.01 * torch.randn(n, in_f, generator=g)

    _, s_ref, _ = torch.linalg.svd(x, full_matrices=False)
    cap_ref = float((s_ref[:r] ** 2).sum())
    for niter in (0, 8):
        sk = EvaSketch(in_f, k=r + 8, niter=niter)
        for chunk in x.split(64):                               # stream in folds
            sk.update(chunk)
        assert sk.rows == n
        v, cap, tot = sk.top(r)
        assert v.shape == (in_f, r)
        eye = v.t() @ v
        assert (eye - torch.eye(r)).abs().max() < 1e-4, "A not orthonormal"
        gap = _subspace_gap(v, basis)
        assert gap < 0.05, f"sketch missed the signal subspace (niter={niter}, gap {gap:.3f})"
        assert abs(cap - cap_ref) / cap_ref < 0.02, \
            f"captured variance off vs exact SVD (niter={niter})"
        assert abs(tot - float(x.pow(2).sum())) / tot < 1e-5

    # Too few rows for the requested rank must fail loudly, not silently.
    sk = EvaSketch(10, k=8)
    sk.update(torch.randn(2, 10))
    try:
        sk.top(4)
        assert False, "rank-starved sketch should raise"
    except SystemExit:
        pass
    print("[lora_init] eva sketch subspace recovery PASSED")


class _EvaNet(nn.Module):
    """Minimal stand-in exposing the surface _apply_eva touches: _wrappers,
    .r, .forward(**batch) that invokes the wrappers the way the real block
    forward does (q/k on the SAME tensor; o on its own)."""

    def __init__(self, wrappers, r, basis, basis_o, junk):
        super().__init__()
        self._wrappers = wrappers
        self.mods = nn.ModuleList(wrappers)
        self.r = r
        self.init_lora = "default"
        self.basis = basis        # [r_true, in] signal subspace for q/k
        self.basis_o = basis_o    # [r_true, in] different subspace for o
        self.junk = junk          # [in] direction ONLY pad rows excite

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                seg_ids=None):
        b, t = input_ids.shape
        g = torch.Generator().manual_seed(int(input_ids.sum().item()))
        x = torch.randn(b, t, self.basis.shape[0], generator=g) @ self.basis
        x2 = torch.randn(b, t, self.basis_o.shape[0], generator=g) @ self.basis_o
        if attention_mask is not None:
            pad = attention_mask == 0
            # Pad rows carry huge activations in a direction orthogonal to the
            # signal; if the pre-pass fails to drop them, they dominate the SVD.
            x[pad] = 50.0 * self.junk
            x2[pad] = 50.0 * self.junk
        q, k, o = self._wrappers
        return q(x) + k(x) + o(x2)


def test_eva_apply_shared_sites_masking_and_preservation():
    torch.manual_seed(0)
    in_f, out_f, r, r_true = 24, 16, 4, 6
    basis = _orthonormal(in_f, r_true, seed=5).t()              # [r_true, in]
    basis_o = _orthonormal(in_f, r_true, seed=6).t()
    junk = torch.randn(in_f, generator=torch.Generator().manual_seed(7))
    junk -= basis.t() @ (basis @ junk)                          # orthogonal to signal
    junk /= junk.norm()

    keys = [f"model.layers.0.self_attn.{l}"
            for l in ("q_proj", "k_proj", "o_proj")]
    wrappers = [DiffLinear(MockLinear(in_f, out_f, key=k2, dtype=torch.float32),
                           r=r, alpha=float(r), compute_dtype=torch.float32)
                for k2 in keys]
    net = _EvaNet(wrappers, r, basis, basis_o, junk)

    x_probe = torch.randn(3, in_f)
    y_before = [w(x_probe) for w in wrappers]

    mask = torch.ones(2, 8, dtype=torch.long)
    mask[:, 5:] = 0                                             # right padding
    batches = [dict(input_ids=torch.arange(2 * 8).reshape(2, 8) + 100 * i,
                    attention_mask=mask) for i in range(3)]
    apply_init_lora(net, "eva", svd_niter=8, verbose=False,
                    eva_batches=iter(batches))

    assert net.init_lora == "eva"
    q, k, o = wrappers
    # q/k consumed the same tensor -> one shared sketch -> identical A.
    assert torch.equal(q.lora_a, k.lora_a)
    assert not torch.equal(q.lora_a, o.lora_a)
    for w in wrappers:
        assert w.init_a0 is None, "eva must not install a frozen offset"
        assert w.lora_b.abs().max().item() == 0.0, "eva must keep B at zero"
        eye = w.lora_a.t() @ w.lora_a
        assert (eye - torch.eye(r)).abs().max() < 1e-4, "A not orthonormal"
    # Pad rows were dropped: their (huge) junk direction is absent from A.
    assert (q.lora_a.t() @ junk).abs().max() < 1e-3, \
        "pad-token activations leaked into the eva init"
    # B = 0: the forward is bit-identical to the pre-init model.
    for w, y0 in zip(wrappers, y_before):
        assert torch.equal(w(x_probe), y0), "eva step 0 not function-preserving"

    # And the pre-pass is mandatory.
    try:
        apply_init_lora(net, "eva", verbose=False)
        assert False, "eva without a data pre-pass should raise"
    except SystemExit:
        pass
    print("[lora_init] eva shared sites + pad masking + preservation PASSED")


def test_pissa_offset_compute_dtype_storage():
    # Under a half-precision compute dtype the on-device offsets are stored in
    # that dtype (they are cast to it on every reconstruction anyway, so this
    # only halves their VRAM), while the fp32 masters keep the exact factors
    # for the sidecar / converted export / apply_to_native.
    torch.manual_seed(0)
    lin = MockLinear(32, 24, key="model.layers.0.self_attn.q_proj",
                     dtype=torch.float32)
    dl = DiffLinear(lin, r=4, alpha=4.0, compute_dtype=torch.bfloat16)
    w = lin.inner.get_weight_tensor()
    a0, b0, _, _ = principal_factors(w, 4, niter=0)
    root_s = dl.scale ** 0.5
    a0, b0 = a0 / root_s, b0 / root_s
    with torch.no_grad():
        dl.lora_a.copy_(a0)
        dl.lora_b.copy_(b0)
    dl.set_init_offset(a0, b0)

    assert dl.init_a0.dtype == torch.bfloat16
    assert dl.init_b0.dtype == torch.bfloat16
    assert dl.init_a0_master.dtype == torch.float32
    assert torch.equal(dl.init_a0_master, a0)
    assert torch.equal(dl.init_b0_master, b0)

    # The served weight is the residual base at bf16 precision (fused addmm).
    got = dl._weight_closure()().float()
    ref = (w - dl.scale * (a0 @ b0)).float()
    tol = 0.05 * w.abs().max().item()          # bf16 rounding of the base
    assert (got - ref).abs().max().item() < tol, \
        "bf16 residual weight too far from the fp32 residual"
    print("[lora_init] pissa compute-dtype offset storage PASSED")


def test_rslora_scale():
    lin = MockLinear(16, 16, key="model.layers.0.self_attn.q_proj",
                     dtype=torch.float32)
    dl = DiffLinear(lin, r=16, alpha=32.0, use_rslora=True,
                    compute_dtype=torch.float32)
    assert abs(dl.scale - 32.0 / 4.0) < 1e-9   # alpha / sqrt(r)
    print("[lora_init] rslora scale PASSED")


def main():
    from util import run_timed
    run_timed([
        test_principal_factors_exact,
        test_pissa_function_preserving_and_residual_math,
        test_qerr_init_reconstructs_error,
        test_pissa_save_convert_and_sidecar_roundtrip,
        test_pissa_offset_compute_dtype_storage,
        test_eva_sketch_recovers_subspace,
        test_eva_apply_shared_sites_masking_and_preservation,
        test_rslora_scale,
    ], label="lora-init")
    print("\nAll LoRA-init checks passed.")


if __name__ == "__main__":
    main()
