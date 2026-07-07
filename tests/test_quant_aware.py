"""
CPU tests for the quantization-aware LoRA training modes
(exllamav3/training/quant_aware.py):

  * σ measurement: the K-bits heuristic (per-column std · 2^-K, skipping
    unquantized layers) and the exact ref-model measurement (per-column rms of
    W_ref - W_q, zero on padded columns).
  * noise mode: deterministic within a tick (the gradient-correctness
    contract: forward, checkpoint recompute and Function backward must see the
    same weight), fresh across ticks, empirically matched to σ·scale, exact in
    eval mode and when disabled, and composing with a pissa residual base.
  * noise-mode gradients: grad_x is consistent with the exact noisy weight the
    forward used (recreated from the same tick).
  * ste mode: bit-exact function preservation at Δ=0 (default AND pissa
    inits), sub-floor deltas contribute nothing, larger deltas snap to
    q·round(Δ/q), σ=0 columns are never touched, and A/B still receive
    straight-through gradients while grad_x flows through the snapped weight.
  * configure/enable/disable round-trip on a (headless) NativeLlamaQLoRA.

All fp32 on CPU; no GPU, no real model.
"""

import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_native_llama import MockLinear, _load, _nl
from util import run_timed

# native_llama's own `from . import quant_aware` already loaded the module
# under the shim package; reuse THAT instance so the closures DiffLinear calls
# and the functions tested here share one module (no double-load aliasing).
_qa = sys.modules.get("exl3train.quant_aware") or _load("quant_aware")
_li = sys.modules.get("exl3train.lora_init") or _load("lora_init")

DiffLinear = _nl.DiffLinear
configure_quant_aware = _qa.configure_quant_aware
heuristic_sigma = _qa.heuristic_sigma
ref_sigma = _qa.ref_sigma


def _wrapper(in_f=32, out_f=24, r=4, alpha=4.0, k_bits=4, seed=0,
             key="model.layers.0.self_attn.q_proj"):
    torch.manual_seed(seed)
    lin = MockLinear(in_f, out_f, key=key, dtype=torch.float32)
    if k_bits is not None:
        lin.inner.K = k_bits          # masquerade as a K-bit trellis layer
    dl = DiffLinear(lin, r=r, alpha=alpha, compute_dtype=torch.float32)
    return dl, lin


def _net(wrappers):
    net = types.SimpleNamespace(_wrappers=wrappers)
    return net


def test_heuristic_sigma():
    dl, lin = _wrapper(k_bits=4)
    w = lin.inner.get_weight_tensor()
    sig = heuristic_sigma(dl)
    ref = w.std(dim=0) * 2.0 ** -4
    assert torch.allclose(sig, ref), "heuristic sigma != per-col std * 2^-K"
    dl2, _ = _wrapper(k_bits=None, key="model.layers.0.self_attn.k_proj")
    assert heuristic_sigma(dl2) is None, "fp16 (no-K) layer must be skipped"
    print("[quant_aware] heuristic sigma PASSED")


def test_ref_sigma_measures_error_and_padding():
    in_pad, out_pad = 16, 12
    in_hf, out_hf = 14, 10
    key = "model.layers.0.self_attn.q_proj"
    dl, lin = _wrapper(in_pad, out_pad, key=key, seed=1)
    w_q = lin.inner.get_weight_tensor()
    err = 0.05 * torch.randn(in_hf, out_hf)
    ref_hf = (w_q[:in_hf, :out_hf] + err).t().contiguous()  # HF [out, in]

    with tempfile.TemporaryDirectory() as td:
        from safetensors.torch import save_file
        save_file({f"{key}.weight": ref_hf},
                  os.path.join(td, "model.safetensors"))
        refs = _li.RefWeights(td)
        sig = ref_sigma(dl, refs)

    assert sig.shape == (out_pad,)
    expect = err.pow(2).mean(dim=0).sqrt()
    assert torch.allclose(sig[:out_hf], expect, atol=1e-6), \
        "ref sigma != per-col rms of (W_ref - W_q)"
    assert (sig[out_hf:] == 0).all(), "padded columns must get sigma = 0"
    print("[quant_aware] ref sigma measurement + padding PASSED")


def test_noise_determinism_and_stats():
    dl, lin = _wrapper(in_f=64, out_f=48, seed=2)
    w_q = lin.inner.get_weight_tensor()
    net = _net([dl])
    scale = 0.7
    configure_quant_aware(net, "noise", scale=scale, verbose=False)
    assert dl.qa_mode == "noise" and dl.qa_sigma is not None

    # Same tick -> the closure must produce the SAME weight on every call
    # (forward + checkpoint recompute + Function backward see one draw).
    f1 = dl._weight_closure_qa()
    w1 = f1()
    w1b = f1()
    f2 = dl._weight_closure_qa()   # recompute path: a fresh closure, same tick
    assert torch.equal(w1, w1b) and torch.equal(w1, f2()), \
        "noise must be deterministic within one tick"
    assert not torch.equal(w1, w_q), "noise mode must actually perturb"

    # New tick -> fresh noise. (net._qa_state and dl.qa_state are the same
    # shared dict; the real net bumps it once per grad-enabled forward.)
    dl.qa_state["tick"] += 1
    w2 = dl._weight_closure_qa()()
    assert not torch.equal(w1, w2), "a new tick must draw fresh noise"

    # Empirical scale: pool the noise over many ticks; per-column std must
    # match sigma (already includes the scale knob).
    draws = []
    for t in range(200):
        dl.qa_state["tick"] = 100 + t
        draws.append(dl._weight_closure_qa()() - w_q)
    noise = torch.stack(draws)                       # [T, in, out]
    emp = noise.reshape(-1, noise.shape[-1]).std(dim=0)
    rel = ((emp - dl.qa_sigma).abs() / dl.qa_sigma).max()
    assert rel < 0.05, f"empirical noise std off by {rel:.1%}"
    assert torch.allclose(dl.qa_sigma,
                          heuristic_sigma(dl) * scale), "scale knob not folded"

    # Eval mode and disable -> exact weights.
    dl.eval()
    assert torch.equal(dl._weight_closure_qa()(), w_q), "eval must be exact"
    dl.train()
    configure_quant_aware(net, "none", verbose=False)
    assert torch.equal(dl._weight_closure_qa()(), w_q), "disable must be exact"
    print("[quant_aware] noise determinism / stats / gating PASSED")


def test_noise_grad_consistency():
    dl, lin = _wrapper(in_f=16, out_f=12, r=3, alpha=3.0, seed=3)
    net = _net([dl])
    configure_quant_aware(net, "noise", scale=1.0, verbose=False)
    with torch.no_grad():
        dl.lora_b.add_(0.1 * torch.randn_like(dl.lora_b))

    x = torch.randn(5, dl.in_features, requires_grad=True)
    y = dl(x)
    g = torch.randn_like(y)
    y.backward(g)

    w_noisy = dl._weight_closure_qa()()   # same tick -> the exact weight used
    s = dl.scale
    a, b = dl.lora_a.detach(), dl.lora_b.detach()
    exp_gx = g @ w_noisy.t() + s * (g @ b.t()) @ a.t()
    exp_gb = s * (x.detach() @ a).t() @ g
    assert torch.allclose(x.grad, exp_gx, atol=1e-5), \
        "grad_x inconsistent with the noisy forward weight"
    assert torch.allclose(dl.lora_b.grad, exp_gb, atol=1e-5), \
        "grad_b must be the plain LoRA gradient"
    print("[quant_aware] noise grad consistency PASSED")


def test_ste_function_preservation_and_snapping():
    dl, lin = _wrapper(in_f=20, out_f=16, r=4, alpha=4.0, seed=4)
    w_q = lin.inner.get_weight_tensor()
    net = _net([dl])
    configure_quant_aware(net, "ste", scale=1.0, verbose=False)
    q = dl.qa_sigma * 12.0 ** 0.5
    x = torch.randn(7, dl.in_features)

    # Delta = 0 (default init): bit-exact base, in the closure AND the forward.
    assert torch.equal(dl._weight_closure_qa()(), w_q), \
        "ste at zero delta must be bit-exact base"
    assert torch.allclose(dl(x), x @ w_q, atol=1e-6)

    # Sub-floor delta: |Δ| < q/2 everywhere -> the forward must IGNORE it.
    with torch.no_grad():
        dl.lora_b.normal_()
        delta = dl.scale * (dl.lora_a @ dl.lora_b)
        lim = (0.4 * q / delta.abs().max(dim=0).values.clamp_min(1e-12)
               ).clamp(max=1.0)
        dl.lora_b.mul_(lim.min())     # uniform shrink below every column floor
    delta = dl.scale * (dl.lora_a @ dl.lora_b)
    assert (delta.abs() < q * 0.5 - 1e-9).all()
    y = dl(x)
    assert torch.allclose(y, x @ w_q, atol=1e-5), \
        "sub-floor delta must contribute nothing to the ste forward"

    # Large delta: total forward == x @ (W_q + q*round(Δ/q)).
    with torch.no_grad():
        dl.lora_b.mul_(50.0)
    delta = dl.scale * (dl.lora_a @ dl.lora_b)
    snapped = (delta / q).round() * q
    y = dl(x)
    ref = x @ (w_q + snapped)
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"ste forward != snapped-delta forward: rel {rel:.2e}"

    # sigma = 0 columns are never touched.
    with torch.no_grad():
        dl.qa_sigma[3] = 0.0
    wf = dl._weight_closure_qa()()
    assert torch.equal(wf[:, 3], w_q[:, 3]), "sigma=0 column must stay exact"

    # Straight-through gradients: A/B get the PLAIN LoRA grads while grad_x
    # flows through the snapped weight the forward used.
    with torch.no_grad():
        dl.qa_sigma[3] = q[3] / 12.0 ** 0.5  # restore
    x = torch.randn(5, dl.in_features, requires_grad=True)
    y = dl(x)
    g = torch.randn_like(y)
    y.backward(g)
    s, a, b = dl.scale, dl.lora_a.detach(), dl.lora_b.detach()
    delta = s * (a @ b)
    w_used = w_q + (delta / q).round() * q
    # The closure serves W_q + Q(Δ) - Δ and the Function adds the live low-rank
    # term back, so the TOTAL forward (and its input grad) is W_q + Q(Δ):
    # exactly the deploy-time weight, no double-counted LoRA term.
    exp_gx = g @ w_used.t()
    exp_gb = s * (x.detach() @ a).t() @ g
    assert torch.allclose(dl.lora_b.grad, exp_gb, atol=1e-5), \
        "ste grad_b must be the straight-through (plain LoRA) gradient"
    assert torch.allclose(x.grad, exp_gx, atol=1e-4), \
        "ste grad_x must flow through the snapped forward weight"
    print("[quant_aware] ste preservation / snapping / STE grads PASSED")


def test_ste_composes_with_pissa_offset():
    from test_lora_init import _pissa_wrapper
    dl, lin = _pissa_wrapper()
    lin.inner.K = 4
    w_q = lin.inner.get_weight_tensor()
    net = _net([dl])
    configure_quant_aware(net, "ste", scale=1.0, verbose=False)

    # pissa init: effective delta s·(AB - A0B0) = 0 -> exact base model.
    x = torch.randn(6, dl.in_features)
    y = dl(x)
    rel = (y - x @ w_q).norm() / (x @ w_q).norm()
    assert rel < 1e-5, f"ste+pissa step 0 not function-preserving: rel {rel:.2e}"

    # After B moves: forward == x @ (W_q + q*round(Δ_eff/q)) with the
    # EFFECTIVE delta s·(AB - A0B0) (what a merge would carry).
    with torch.no_grad():
        dl.lora_b.add_(5.0 * torch.randn_like(dl.lora_b))
    q = dl.qa_sigma * 12.0 ** 0.5
    s = dl.scale
    d_eff = s * (dl.lora_a @ dl.lora_b - dl.init_a0 @ dl.init_b0)
    ref = x @ (w_q + (d_eff / q).round() * q)
    y = dl(x)
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"ste+pissa snapped forward mismatch: rel {rel:.2e}"
    print("[quant_aware] ste + pissa effective-delta composition PASSED")


def test_noise_composes_with_pissa_offset():
    from test_lora_init import _pissa_wrapper
    dl, lin = _pissa_wrapper()
    lin.inner.K = 4
    w_q = lin.inner.get_weight_tensor()
    net = _net([dl])
    configure_quant_aware(net, "noise", scale=1.0, verbose=False)

    # The noise rides ON TOP of the pissa residual base; the adapter (== the
    # offset at init) cancels it, so the forward is base + noise exactly.
    x = torch.eye(dl.in_features)
    w_eff = dl._weight_closure_qa()()             # residual + noise
    y = dl(x)                                     # + s·AB (cancels -s·A0B0)
    noise = w_eff - (w_q - dl.scale * (dl.init_a0 @ dl.init_b0))
    ref = w_q + noise
    rel = (y - ref).norm() / ref.norm()
    assert rel < 1e-5, f"noise+pissa forward mismatch: rel {rel:.2e}"
    assert noise.abs().max() > 0
    print("[quant_aware] noise + pissa composition PASSED")


def test_noise_under_gradient_checkpointing():
    # The crux risk: with use_reentrant=False checkpointing the block forward
    # is re-run during backward, creating a NEW weight closure. Same tick ->
    # same noise -> the checkpointed run must produce bit-identical gradients
    # to the uncheckpointed one (a mismatch here is the silent-corruption
    # failure mode the determinism contract exists to prevent).
    import torch.utils.checkpoint as ckpt

    def build(seed):
        torch.manual_seed(seed)
        dl1, _ = _wrapper(16, 16, r=2, alpha=2.0, seed=seed,
                          key="model.layers.0.self_attn.q_proj")
        dl2, _ = _wrapper(16, 12, r=2, alpha=2.0, seed=seed + 1,
                          key="model.layers.0.self_attn.o_proj")
        with torch.no_grad():
            for dl in (dl1, dl2):
                dl.lora_b.normal_(std=0.1)
        return dl1, dl2

    grads = {}
    for use_ckpt in (False, True):
        dl1, dl2 = build(11)
        net = _net([dl1, dl2])
        configure_quant_aware(net, "noise", scale=1.0, verbose=False)
        net._qa_state["tick"] = 7          # same tick either way
        torch.manual_seed(99)
        x = torch.randn(4, 16, requires_grad=True)

        def block(h):
            return dl2(torch.nn.functional.silu(dl1(h)))

        y = (ckpt.checkpoint(block, x, use_reentrant=False) if use_ckpt
             else block(x))
        y.pow(2).sum().backward()
        grads[use_ckpt] = (x.grad.clone(), dl1.lora_b.grad.clone(),
                           dl2.lora_b.grad.clone(), y.detach().clone())

    for a, b in zip(grads[False], grads[True]):
        assert torch.equal(a, b), \
            "checkpointed grads/outputs differ from the plain run -- the " \
            "noise recompute is not deterministic"
    print("[quant_aware] noise under grad-checkpointing (bit-exact) PASSED")


def test_configure_on_headless_net_and_skip_report():
    # One quantized + one fp16 (no-K) wrapper: configure must enable the
    # first, skip the second, and disable cleanly.
    dl1, _ = _wrapper(k_bits=3, key="model.layers.0.self_attn.q_proj", seed=5)
    dl2, _ = _wrapper(k_bits=None, key="model.layers.0.self_attn.k_proj", seed=6)
    frozen = DiffLinear(MockLinear(8, 8, key="model.layers.0.mlp.x",
                                   dtype=torch.float32), r=0,
                        compute_dtype=torch.float32)
    net = _nl.NativeLlamaQLoRA.__new__(_nl.NativeLlamaQLoRA)
    nn.Module.__init__(net)
    net._wrappers = [dl1, dl2, frozen]

    net.set_quant_aware("noise", scale=2.0, verbose=False)
    assert dl1.qa_mode == "noise" and dl1.qa_state is net._qa_state
    assert dl2.qa_mode == "" and dl2.qa_sigma is None
    assert net.quant_aware == "noise" and net.quant_aware_scale == 2.0
    assert net._qa_state == {"tick": 0, "seed": 0}

    net.set_quant_aware("none", verbose=False)
    assert net._qa_state is None and dl1.qa_mode == ""
    assert net.quant_aware == "none"
    print("[quant_aware] configure / skip / disable on net PASSED")


def test_seed_stream_layer_and_tick_decorrelated():
    s = {(l, t): _qa.qa_seed(0, l, t) for l in range(8) for t in range(8)}
    assert len(set(s.values())) == len(s), "seed collisions in (layer, tick)"
    assert all(0 <= v < 2 ** 63 for v in s.values())
    print("[quant_aware] seed stream PASSED")


if __name__ == "__main__":
    run_timed([
        test_heuristic_sigma,
        test_ref_sigma_measures_error_and_padding,
        test_noise_determinism_and_stats,
        test_noise_grad_consistency,
        test_ste_function_preservation_and_snapping,
        test_ste_composes_with_pissa_offset,
        test_noise_composes_with_pissa_offset,
        test_noise_under_gradient_checkpointing,
        test_configure_on_headless_net_and_skip_report,
        test_seed_stream_layer_and_tick_decorrelated,
    ], label="quant_aware")
    print("All quant_aware tests PASSED")
