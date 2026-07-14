"""
CPU tests for the differentiable BlockSparseMLP (mixture of experts) path in
``native_llama._moe_out`` (Qwen3-MoE / Qwen3.5-MoE with the std softmax top-k
router, optional shared expert + sigmoid shared gate).

No GPU, no compiled extension, no real model: the training modules are loaded
under a synthetic package (so their relative imports resolve without importing
the full exllamav3 package, which would build the CUDA ext), EXL3 linears are
mocked as frozen random weights, and the checks are:

  * ``_moe_out`` (top-k over logits, softmax over the selected k) matches an
    independent HF-style reference (softmax over ALL experts, top-k of the
    probabilities, renormalize -- the ``norm_topk_prob=True`` math), incl. the
    shared expert behind ``sigmoid(shared_gate(x))`` -- this equality IS the
    proof that the two routing formulations coincide;
  * a Qwen3-MoE-shaped block (no shared expert) with a post-softmax
    per-expert scale also matches the reference;
  * fp64 gradcheck through routing softmax + expert scatter/gather w.r.t. the
    block input;
  * backprop reaches expert + shared-expert LoRA adapters while the frozen
    base weights and the router (r=0, no params) stay untouched, and an
    expert that receives no tokens contributes no gradient;
  * ``_mlp_out`` dispatches on ``meta["mlp_kind"]`` (dense stays dense);
  * the Gemma4 MoE layout (``alt_residual_channel``): routing + routed
    experts read the RAW post-attention residual through their own pre-norms
    (router pre-norm carrying the ``hidden**-0.5`` constant scale) while the
    shared expert reads the block's normed input; routed/shared post-norms;
    GeGLU; per-expert scale -- against an independent reference, unit-level,
    fp64-gradchecked (w.r.t. BOTH input channels), and wired through a full
    ``_block_forward`` (sandwich post-norms + layer_scalar) so a
    wrong-channel regression cannot hide behind the unit test.

Run:  python tests/test_moe.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_qll = _load("qlora_linear")
_fce = _load("fused_ce")
_gdn = _load("gdn")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA


# ----------------------------------------------------------------------------
# Mock EXL3 linear (same shape as test_gdn's).
# ----------------------------------------------------------------------------
class _MockInner:
    def __init__(self, weight):
        self._w = weight                 # [in, out], frozen
        self.trellis = weight            # device inference
        self.bias = None

    def get_weight_tensor(self):
        return self._w

    def get_bias_tensor(self):
        return None


class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, key, scale=0.05, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.key = key
        self.device = torch.device("cpu")
        w = torch.randn(in_features, out_features, dtype=dtype) * scale
        self.register_buffer("frozen_weight", w)
        self.inner = _MockInner(self.frozen_weight)
        self.lora_a_tensors = {}
        self.lora_b_tensors = {}


def _headless_net():
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = torch.float32
    n.attn_impl = "eager"
    return n


# ----------------------------------------------------------------------------
# MoE block builder: per-expert gate/up/down + router (+ optional shared
# expert behind a sigmoid shared gate), mirroring what NativeLlamaQLoRA.__init__
# builds for a BlockSparseMLP block.
# ----------------------------------------------------------------------------
def _norm_w(dim, dtype):
    return nn.Parameter(1.0 + 0.05 * torch.randn(dim, dtype=dtype),
                        requires_grad=False)


def _spec(weight, eps=1e-5, bias=0.0, scale=1.0):
    # Mirror of backbone.norm_spec.
    return {"weight": weight, "eps": eps, "bias": bias, "scale": scale}


def _build_moe(d, inter, num_experts, top_k, shared_inter=0, shared_gate=True,
               per_expert_scale=None, r=0, expert_r=None, dtype=torch.float32,
               router_scale=0.5, activation="silu", gemma4=False,
               router_type="std", e_score_bias=None, routed_scaling=1.0):
    expert_r = r if expert_r is None else expert_r
    lins = {"router": MockLinear(d, num_experts, "blk.mlp.gate",
                                 scale=router_scale, dtype=dtype)}
    entry = types.SimpleNamespace()
    entry.is_moe = True
    entry.router = DiffLinear(lins["router"], r=0, compute_dtype=dtype)
    entry.expert_gates, entry.expert_ups, entry.expert_downs = [], [], []
    for e in range(num_experts):
        g = MockLinear(d, inter, f"blk.mlp.experts.{e}.gate_proj", dtype=dtype)
        u = MockLinear(d, inter, f"blk.mlp.experts.{e}.up_proj", dtype=dtype)
        dn = MockLinear(inter, d, f"blk.mlp.experts.{e}.down_proj", dtype=dtype)
        lins[f"g{e}"], lins[f"u{e}"], lins[f"d{e}"] = g, u, dn
        entry.expert_gates.append(DiffLinear(g, r=expert_r, compute_dtype=dtype))
        entry.expert_ups.append(DiffLinear(u, r=expert_r, compute_dtype=dtype))
        entry.expert_downs.append(DiffLinear(dn, r=expert_r, compute_dtype=dtype))
    if shared_inter:
        sg = MockLinear(d, shared_inter, "blk.mlp.shared_expert.gate_proj", dtype=dtype)
        su = MockLinear(d, shared_inter, "blk.mlp.shared_expert.up_proj", dtype=dtype)
        sd = MockLinear(shared_inter, d, "blk.mlp.shared_expert.down_proj", dtype=dtype)
        lins["sg"], lins["su"], lins["sd"] = sg, su, sd
        entry.gates = [DiffLinear(sg, r=r, compute_dtype=dtype)]
        entry.ups = [DiffLinear(su, r=r, compute_dtype=dtype)]
        entry.downs = [DiffLinear(sd, r=r, compute_dtype=dtype)]
        if shared_gate:
            shg = MockLinear(d, 1, "blk.mlp.shared_expert_gate", dtype=dtype)
            lins["shg"] = shg
            entry.shared_gate = DiffLinear(shg, r=0, compute_dtype=dtype)
        else:
            entry.shared_gate = None
    else:
        entry.gates, entry.ups, entry.downs = [], [], []
        entry.shared_gate = None

    if gemma4:
        # Gemma4 layout: routing + routed experts fed from the RAW residual
        # via their own pre-norms (the router pre-norm carries the
        # hidden**-0.5 constant scale of the checkpoint's `router.scale`
        # norm), routed/shared post-norms, no shared gate (callers pass
        # shared_gate=False).
        lins["n_rp"] = _norm_w(d, dtype)
        lins["n_ep"] = _norm_w(d, dtype)
        lins["n_po"] = _norm_w(d, dtype)
        entry.router_pre_spec = _spec(lins["n_rp"], scale=d ** -0.5)
        entry.routed_pre_spec = _spec(lins["n_ep"])
        entry.routed_post_spec = _spec(lins["n_po"])
        if shared_inter:
            lins["n_sp"] = _norm_w(d, dtype)
            entry.shared_post_spec = _spec(lins["n_sp"])

    meta = {
        "mlp_kind": "moe",
        "router_type": router_type,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "per_expert_scale": per_expert_scale,
        "routed_scaling_factor": routed_scaling,
        "e_score_bias": e_score_bias,
        "activation": activation,
        "shared_activation": activation if shared_inter else None,
        "alt_residual_channel": bool(gemma4),
    }
    return entry, meta, lins


def _ref_moe(meta, lins, x):
    """Independent HF-style reference (Qwen3MoeSparseMoeBlock /
    Qwen2MoeSparseMoeBlock math): router softmax over ALL experts, top-k of
    the probabilities, renormalize (norm_topk_prob=True), one-hot expert loop,
    plus the shared expert times sigmoid(shared_gate). Shares no code with
    _moe_out (which does top-k over LOGITS then softmax over the k -- the
    equality tested here is exactly the norm_topk_prob equivalence)."""
    b, t, d = x.shape
    y = x.reshape(-1, d)
    E, k = meta["num_experts"], meta["num_experts_per_tok"]

    logits = y @ lins["router"].frozen_weight
    probs = torch.softmax(logits.float(), dim=-1)
    topw, topi = torch.topk(probs, k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True)
    if meta.get("per_expert_scale") is not None:
        topw = topw * meta["per_expert_scale"].float()[topi]

    out = torch.zeros_like(y, dtype=torch.float32)
    expert_mask = F.one_hot(topi, num_classes=E)                 # [nt, k, E]
    for e in range(E):
        tok, kth = torch.where(expert_mask[..., e] > 0)
        if tok.numel() == 0:
            continue
        xe = y[tok]
        h = F.silu(xe @ lins[f"g{e}"].frozen_weight) * (xe @ lins[f"u{e}"].frozen_weight)
        de = (h @ lins[f"d{e}"].frozen_weight).float()
        out[tok] += de * topw[tok, kth].unsqueeze(-1)

    if "sg" in lins:
        sh = F.silu(y @ lins["sg"].frozen_weight) * (y @ lins["su"].frozen_weight)
        sh = (sh @ lins["sd"].frozen_weight).float()
        if "shg" in lins:
            sh = sh * torch.sigmoid((y @ lins["shg"].frozen_weight).float())
        out = out + sh
    return out.view(b, t, d).to(x.dtype)


def _ref_rmsnorm(x, w, eps=1e-5, scale=1.0):
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    xn = x.float() * torch.rsqrt(var) * scale
    return xn if w is None else xn * w.float()


def _ref_moe_gemma4(meta, lins, x_normed, residual):
    """Independent reference for the Gemma4 MoE layout, transcribed from the
    inference module (``BlockSparseMLP.forward`` with alt_residual_channel):
    routing input = rmsnorm(residual) * d**-0.5 (the `router.scale` norm);
    HF-style softmax-all + top-k + renormalize (+ per-expert scale); routed
    experts on rmsnorm(residual) (pre_feedforward_layernorm_2) with GeGLU;
    routed sum normed by post_feedforward_layernorm_2; shared expert on the
    NORMED block input, its output normed by post_feedforward_layernorm_1;
    sum. Shares no code with _moe_out."""
    b, t, d = x_normed.shape
    y = x_normed.reshape(-1, d)
    yr = residual.reshape(-1, d)
    E, k = meta["num_experts"], meta["num_experts_per_tok"]
    actfn = F.silu if meta["activation"] == "silu" \
        else (lambda z: F.gelu(z, approximate="tanh"))

    z = _ref_rmsnorm(yr, lins["n_rp"], scale=d ** -0.5)
    logits = z.to(yr.dtype) @ lins["router"].frozen_weight
    probs = torch.softmax(logits.float(), dim=-1)
    topw, topi = torch.topk(probs, k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True)
    if meta.get("per_expert_scale") is not None:
        topw = topw * meta["per_expert_scale"].float()[topi]

    ye = _ref_rmsnorm(yr, lins["n_ep"]).to(yr.dtype)
    out = torch.zeros(y.shape[0], d, dtype=torch.float32)
    expert_mask = F.one_hot(topi, num_classes=E)                 # [nt, k, E]
    for e in range(E):
        tok, kth = torch.where(expert_mask[..., e] > 0)
        if tok.numel() == 0:
            continue
        xe = ye[tok]
        h = actfn(xe @ lins[f"g{e}"].frozen_weight) * (xe @ lins[f"u{e}"].frozen_weight)
        de = (h @ lins[f"d{e}"].frozen_weight).float()
        out[tok] += de * topw[tok, kth].unsqueeze(-1)
    out = _ref_rmsnorm(out, lins["n_po"])

    if "sg" in lins:
        sh = actfn(y @ lins["sg"].frozen_weight) * (y @ lins["su"].frozen_weight)
        sh = (sh @ lins["sd"].frozen_weight).float()
        sh = _ref_rmsnorm(sh, lins["n_sp"])
        out = out + sh
    return out.view(b, t, d).to(x_normed.dtype)


# ----------------------------------------------------------------------------
# Forward matches the HF-style reference.
# ----------------------------------------------------------------------------
def test_moe_matches_hf_reference():
    torch.manual_seed(0)
    d, inter, E, k = 16, 8, 8, 3
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=12, shared_gate=True)
    net = _headless_net()

    x = torch.randn(2, 7, d)
    out = net._mlp_out(meta, entry, x)
    ref = _ref_moe(meta, lins, x)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"MoE forward mismatch vs HF-style reference: max|Δ|={err}"
    print(f"[moe] shared-expert MoE matches HF-style reference (max|Δ|={err:.2e}) PASSED")


def test_moe_qwen3_shape_with_scale():
    # Qwen3-MoE shape: no shared expert; also exercises per_expert_scale.
    torch.manual_seed(1)
    d, inter, E, k = 12, 6, 6, 2
    pes = (1.0 + 0.2 * torch.randn(E)).to(torch.bfloat16)
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=0,
                                   per_expert_scale=pes)
    net = _headless_net()

    x = torch.randn(3, 5, d)
    out = net._mlp_out(meta, entry, x)
    ref = _ref_moe(meta, lins, x)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"MoE (no shared, per-expert scale) mismatch: max|Δ|={err}"
    print(f"[moe] no-shared-expert MoE + per-expert scale matches reference "
          f"(max|Δ|={err:.2e}) PASSED")


def _ref_moe_dots(meta, lins, x):
    """Independent reference for the "dots" sigmoid router (AFMoE /
    dots.llm1), transcribed from the HF AfmoeMoE routing math (== the
    inference ``routing_dots`` / ``ext.routing_ds3_nogroup`` kernel):
    sigmoid scores; e_score_correction_bias added for the top-k SELECTION
    only; selected experts weighted by their UNBIASED scores normalized over
    the selected set (+1e-20) times routed_scaling_factor. Optional ungated
    shared expert added on top (AFMoE has no shared gate). Shares no code
    with _moe_out."""
    b, t, d = x.shape
    y = x.reshape(-1, d)
    E, k = meta["num_experts"], meta["num_experts_per_tok"]

    logits = (y @ lins["router"].frozen_weight).float()
    scores = torch.sigmoid(logits)
    sel = scores if meta.get("e_score_bias") is None \
        else scores + meta["e_score_bias"].float()
    _, topi = torch.topk(sel, k, dim=-1)
    topw = scores.gather(1, topi)
    topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
    topw = topw * meta["routed_scaling_factor"]

    out = torch.zeros_like(y, dtype=torch.float32)
    expert_mask = F.one_hot(topi, num_classes=E)                 # [nt, k, E]
    for e in range(E):
        tok, kth = torch.where(expert_mask[..., e] > 0)
        if tok.numel() == 0:
            continue
        xe = y[tok]
        h = F.silu(xe @ lins[f"g{e}"].frozen_weight) * (xe @ lins[f"u{e}"].frozen_weight)
        de = (h @ lins[f"d{e}"].frozen_weight).float()
        out[tok] += de * topw[tok, kth].unsqueeze(-1)

    if "sg" in lins:
        sh = F.silu(y @ lins["sg"].frozen_weight) * (y @ lins["su"].frozen_weight)
        sh = (sh @ lins["sd"].frozen_weight).float()
        if "shg" in lins:
            sh = sh * torch.sigmoid((y @ lins["shg"].frozen_weight).float())
        out = out + sh
    return out.view(b, t, d).to(x.dtype)


def test_moe_dots_matches_reference():
    # AFMoE shape: dots router with selection bias + routed scaling, ungated
    # shared expert.
    torch.manual_seed(5)
    d, inter, E, k = 16, 8, 8, 3
    e_bias = 0.5 * torch.randn(E)
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=12,
                                   shared_gate=False, router_type="dots",
                                   e_score_bias=e_bias, routed_scaling=2.826)
    net = _headless_net()

    x = torch.randn(2, 7, d)
    out = net._mlp_out(meta, entry, x)
    ref = _ref_moe_dots(meta, lins, x)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"dots-router MoE mismatch vs reference: max|Δ|={err}"
    print(f"[moe] dots-router (AFMoE) MoE matches reference (max|Δ|={err:.2e}) PASSED")


def test_moe_dots_bias_selects_but_never_weights():
    # A large positive bias on one expert must FORCE its selection for every
    # token, while its routing weight stays the plain (unbiased) sigmoid
    # score -- the defining split of the dots router.
    torch.manual_seed(6)
    d, inter, E, k = 12, 6, 6, 2
    e_bias = torch.zeros(E)
    e_bias[3] = 100.0                    # always selected, never up-weighted
    entry, meta, lins = _build_moe(d, inter, E, k, router_type="dots",
                                   e_score_bias=e_bias, routed_scaling=1.0)
    net = _headless_net()
    x = torch.randn(1, 9, d)

    out = net._mlp_out(meta, entry, x)
    ref = _ref_moe_dots(meta, lins, x)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"dots bias-selection mismatch: max|Δ|={err}"

    # Confirm the bias actually changed selection: expert 3 must appear in
    # every token's top-k of the BIASED scores, and (with these logits) not
    # in every token's top-k of the unbiased ones.
    logits = (x.reshape(-1, d) @ lins["router"].frozen_weight).float()
    scores = torch.sigmoid(logits)
    _, topi_b = torch.topk(scores + e_bias, k, dim=-1)
    assert bool((topi_b == 3).any(dim=-1).all()), "bias failed to force selection"
    _, topi_u = torch.topk(scores, k, dim=-1)
    assert not bool((topi_u == 3).any(dim=-1).all()), \
        "test vacuous: expert 3 was already always selected unbiased"
    print("[moe] dots e_score bias forces selection, never enters weights PASSED")


def test_moe_dots_gradcheck():
    torch.manual_seed(7)
    d, inter, E, k = 6, 4, 4, 2
    entry, meta, _ = _build_moe(d, inter, E, k, shared_inter=4,
                                shared_gate=False, dtype=torch.float64,
                                router_type="dots",
                                e_score_bias=0.3 * torch.randn(E, dtype=torch.float64),
                                routed_scaling=2.0)
    net = _headless_net()
    net.compute_dtype = torch.float64

    x = torch.randn(1, 5, d, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return net._mlp_out(meta, entry, inp)

    # topk is piecewise-constant in the input; random fp64 scores are nowhere
    # near a selection tie, so the local gradient is well-defined.
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)
    print("[moe] fp64 gradcheck through dots routing + experts PASSED")


# ----------------------------------------------------------------------------
# fp64 gradcheck through routing softmax + expert gather/scatter.
# ----------------------------------------------------------------------------
def test_moe_gradcheck():
    torch.manual_seed(2)
    d, inter, E, k = 6, 4, 4, 2
    entry, meta, _ = _build_moe(d, inter, E, k, shared_inter=4, shared_gate=True,
                                dtype=torch.float64)
    net = _headless_net()
    net.compute_dtype = torch.float64

    x = torch.randn(1, 5, d, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return net._mlp_out(meta, entry, inp)

    # topk is piecewise-constant in the input; random fp64 logits are nowhere
    # near a routing tie, so the local gradient is well-defined.
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)
    print("[moe] fp64 gradcheck through routing + experts + shared gate PASSED")


# ----------------------------------------------------------------------------
# Backward reaches the adapters; base + router stay frozen; a token-less
# expert contributes no gradient.
# ----------------------------------------------------------------------------
def test_moe_backward_reaches_adapters():
    torch.manual_seed(3)
    # k == E: every expert sees every token, so every expert adapter gets a
    # deterministic gradient.
    d, inter, E, k = 12, 6, 2, 2
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=8,
                                   shared_gate=True, r=4, expert_r=2)
    net = _headless_net()

    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    # B starts at zero (adapter no-op); nudge it so lora_a also sees gradient
    # through a nonzero product.
    adapted = ([entry.expert_gates[e] for e in range(E)]
               + [entry.expert_ups[e] for e in range(E)]
               + [entry.expert_downs[e] for e in range(E)]
               + [entry.gates[0], entry.ups[0], entry.downs[0]])
    for dl in adapted:
        with torch.no_grad():
            dl.lora_b.add_(torch.randn_like(dl.lora_b) * 0.02)

    x = torch.randn(2, 5, d, requires_grad=True)
    out = net._mlp_out(meta, entry, x)
    out.square().mean().backward()

    for dl in adapted:
        assert dl.lora_a.grad is not None and dl.lora_a.grad.abs().sum() > 0, \
            f"no gradient reached lora_a of {dl.key}"
        assert dl.lora_b.grad is not None and dl.lora_b.grad.abs().sum() > 0, \
            f"no gradient reached lora_b of {dl.key}"
    assert entry.expert_gates[0].r == 2 and entry.gates[0].r == 4, \
        "expert_r should differ from the shared-expert r in this test"
    assert entry.router.lora_a is None, "router must stay frozen (r=0)"
    assert entry.shared_gate.lora_a is None, "shared gate must stay frozen (r=0)"
    assert x.grad is not None and x.grad.abs().sum() > 0, \
        "no gradient reached the block input (routing weights must be differentiable)"
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_]), \
            f"frozen base weight of {k_} changed"
    print("[moe] backward reaches expert + shared adapters (mixed ranks); "
          "base/router/shared-gate frozen PASSED")


def test_moe_unrouted_expert_gets_no_grad():
    torch.manual_seed(4)
    d, inter, E, k = 10, 5, 4, 1
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=0, r=3)
    # Force every token onto expert 2: with W[:, 2] = +1 and all other columns
    # -1, any strictly positive input gives logit_2 = sum(x) > 0 > -sum(x) =
    # logit_other, so top-1 routing always picks expert 2.
    with torch.no_grad():
        lins["router"].frozen_weight[:, :] = -1.0
        lins["router"].frozen_weight[:, 2] = 1.0
    x = torch.randn(1, 6, d).abs() + 0.1          # positive inputs -> expert 2 wins
    x.requires_grad_(True)

    for e in range(E):
        with torch.no_grad():
            entry.expert_gates[e].lora_b.add_(0.02 * torch.randn_like(
                entry.expert_gates[e].lora_b))

    net = _headless_net()
    out = net._mlp_out(meta, entry, x)
    out.square().mean().backward()

    assert entry.expert_gates[2].lora_a.grad is not None \
        and entry.expert_gates[2].lora_a.grad.abs().sum() > 0, \
        "routed expert 2 should receive gradient"
    for e in (0, 1, 3):
        g = entry.expert_gates[e].lora_a.grad
        assert g is None or g.abs().sum() == 0, \
            f"expert {e} received tokens/grad but should be unrouted"
    print("[moe] token-less experts are skipped and receive no gradient PASSED")


# ----------------------------------------------------------------------------
# Gemma4 MoE layout: alt residual channel + extra norms + GeGLU.
# ----------------------------------------------------------------------------
def test_moe_gemma4_matches_reference():
    torch.manual_seed(6)
    d, inter, E, k = 16, 8, 6, 2
    pes = (1.0 + 0.2 * torch.randn(E)).to(torch.bfloat16)
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=12,
                                   shared_gate=False, per_expert_scale=pes,
                                   activation="gelu", gemma4=True)
    net = _headless_net()

    # DISTINCT tensors for the two input channels: the normed block input
    # (shared expert) vs the raw post-attention residual (routing + routed
    # experts). Feeding the same tensor would let a wrong-channel bug pass.
    x_normed = torch.randn(2, 7, d)
    residual = torch.randn(2, 7, d) * 3.0
    out = net._mlp_out(meta, entry, x_normed, residual)
    ref = _ref_moe_gemma4(meta, lins, x_normed, residual)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"Gemma4 MoE forward mismatch vs reference: max|Δ|={err}"

    # Channel sanity: routing/experts must actually depend on the residual --
    # a different residual with the same normed input must change the output.
    out2 = net._mlp_out(meta, entry, x_normed, torch.randn(2, 7, d) * 3.0)
    assert (out - out2).abs().max().item() > 1e-3, \
        "Gemma4 MoE output ignores the residual channel"
    print(f"[moe] Gemma4 layout (alt residual + 4 norms + GeGLU + scale) "
          f"matches reference (max|Δ|={err:.2e}) PASSED")


def test_moe_gemma4_gradcheck():
    torch.manual_seed(7)
    d, inter, E, k = 6, 4, 4, 2
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=4,
                                   shared_gate=False, activation="gelu",
                                   gemma4=True, dtype=torch.float64)
    net = _headless_net()
    net.compute_dtype = torch.float64

    xn = torch.randn(1, 5, d, dtype=torch.float64, requires_grad=True)
    rs = torch.randn(1, 5, d, dtype=torch.float64, requires_grad=True)

    def fn(a, b):
        return net._mlp_out(meta, entry, a, b)

    # Gradient w.r.t. BOTH channels: the shared expert path (normed input) and
    # the routing + routed-expert path (residual). topk is piecewise-constant;
    # random fp64 logits sit nowhere near a routing tie.
    assert torch.autograd.gradcheck(fn, (xn, rs), eps=1e-6, atol=1e-4)
    print("[moe] Gemma4 fp64 gradcheck through both input channels PASSED")


def test_moe_gemma4_backward_reaches_adapters():
    torch.manual_seed(8)
    d, inter, E, k = 12, 6, 2, 2                    # k == E: all experts hit
    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=8,
                                   shared_gate=False, activation="gelu",
                                   gemma4=True, r=4, expert_r=2)
    net = _headless_net()

    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()
                     if hasattr(l, "frozen_weight")}
    adapted = ([entry.expert_gates[e] for e in range(E)]
               + [entry.expert_ups[e] for e in range(E)]
               + [entry.expert_downs[e] for e in range(E)]
               + [entry.gates[0], entry.ups[0], entry.downs[0]])
    for dl in adapted:
        with torch.no_grad():
            dl.lora_b.add_(torch.randn_like(dl.lora_b) * 0.02)

    x_normed = torch.randn(2, 5, d, requires_grad=True)
    residual = torch.randn(2, 5, d, requires_grad=True)
    out = net._mlp_out(meta, entry, x_normed, residual)
    out.square().mean().backward()

    for dl in adapted:
        assert dl.lora_a.grad is not None and dl.lora_a.grad.abs().sum() > 0, \
            f"no gradient reached lora_a of {dl.key}"
    assert entry.router.lora_a is None, "router must stay frozen (r=0)"
    assert x_normed.grad is not None and x_normed.grad.abs().sum() > 0, \
        "no gradient into the normed input (shared-expert channel)"
    assert residual.grad is not None and residual.grad.abs().sum() > 0, \
        "no gradient into the residual (routing/routed-expert channel)"
    for k_, l in lins.items():
        if hasattr(l, "frozen_weight"):
            assert torch.equal(l.frozen_weight, frozen_before[k_]), \
                f"frozen base weight of {k_} changed"
    print("[moe] Gemma4 backward reaches expert + shared adapters through both "
          "channels; base/router frozen PASSED")


def test_moe_gemma4_block_wiring():
    # Full _block_forward: the MoE must receive the RAW post-attention
    # residual (what inference stashes as params["residual"]) -- not the
    # mlp-normed input -- composed with Gemma's sandwich post-norms and the
    # per-layer scalar. An independent full-block reference catches any
    # wrong-channel plumbing that the unit test (which passes residual
    # explicitly) cannot.
    torch.manual_seed(9)
    d, nq, nkv, hd = 16, 4, 2, 8
    inter, E, k = 8, 4, 2
    dtype = torch.float32

    entry, meta, lins = _build_moe(d, inter, E, k, shared_inter=12,
                                   shared_gate=False, activation="gelu",
                                   gemma4=True)
    # Attention half (plain GQA; the Gemma attention features have their own
    # tests in test_native_llama).
    lins["q"] = MockLinear(d, nq * hd, "blk.self_attn.q_proj", dtype=dtype)
    lins["k"] = MockLinear(d, nkv * hd, "blk.self_attn.k_proj", dtype=dtype)
    lins["v"] = MockLinear(d, nkv * hd, "blk.self_attn.v_proj", dtype=dtype)
    lins["o"] = MockLinear(nq * hd, d, "blk.self_attn.o_proj", dtype=dtype)
    entry.q_proj = DiffLinear(lins["q"], r=0, compute_dtype=dtype)
    entry.k_proj = DiffLinear(lins["k"], r=0, compute_dtype=dtype)
    entry.v_proj = DiffLinear(lins["v"], r=0, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=0, compute_dtype=dtype)
    entry.q_norm_spec = entry.k_norm_spec = entry.v_norm_spec = None
    lins["n_attn"] = _norm_w(d, dtype)
    lins["n_mlp"] = _norm_w(d, dtype)
    lins["n_attn_post"] = _norm_w(d, dtype)
    lins["n_mlp_post"] = _norm_w(d, dtype)
    entry.attn_norm_spec = _spec(lins["n_attn"])
    entry.mlp_norm_spec = _spec(lins["n_mlp"])
    entry.attn_post_spec = _spec(lins["n_attn_post"])
    entry.mlp_post_spec = _spec(lins["n_mlp_post"])

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    meta.update({
        "kind": "attn", "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": hd,
        "sm_scale": hd ** -0.5, "inv_freq": inv_freq, "attn_factor": 1.0,
        "sliding_window": -1, "softcap": 0.0, "use_k_as_v": False,
        "layer_scalar": 0.9,
    })

    net = _headless_net()
    b, t = 2, 6
    hidden = torch.randn(b, t, d, dtype=dtype)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    attn_bias = net._attn_bias(None, t, hidden.device, torch.float32)
    out = net._block_forward(meta, entry, hidden, positions, attn_bias)

    # Independent reference block.
    def rope(x):
        freqs = positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().unsqueeze(2), emb.sin().unsqueeze(2)
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rot * sin

    normed = _ref_rmsnorm(hidden, lins["n_attn"])
    q = rope((normed @ lins["q"].frozen_weight).view(b, t, nq, hd)).transpose(1, 2)
    kk = rope((normed @ lins["k"].frozen_weight).view(b, t, nkv, hd)).transpose(1, 2)
    vv = (normed @ lins["v"].frozen_weight).view(b, t, nkv, hd).transpose(1, 2)
    rep = nq // nkv
    kk = kk.repeat_interleave(rep, 1)
    vv = vv.repeat_interleave(rep, 1)
    scores = (q @ kk.transpose(-1, -2)) * meta["sm_scale"]
    mask = torch.triu(torch.full((t, t), float("-inf")), 1)
    ctx = torch.softmax(scores + mask, -1) @ vv
    attn_out = ctx.transpose(1, 2).reshape(b, t, nq * hd) @ lins["o"].frozen_weight
    resid = hidden + _ref_rmsnorm(attn_out, lins["n_attn_post"])

    x_normed = _ref_rmsnorm(resid, lins["n_mlp"]).to(dtype)
    moe_out = _ref_moe_gemma4(meta, lins, x_normed, resid.to(dtype))
    ref = resid + _ref_rmsnorm(moe_out, lins["n_mlp_post"])
    ref = ref * meta["layer_scalar"]

    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"Gemma4 MoE block wiring mismatch vs reference: max|Δ|={err}"
    print(f"[moe] Gemma4 MoE block: raw-residual routing channel + sandwich + "
          f"layer_scalar match full-block reference (max|Δ|={err:.2e}) PASSED")


# ----------------------------------------------------------------------------
# Dense dispatch unchanged.
# ----------------------------------------------------------------------------
def test_dense_dispatch_unchanged():
    torch.manual_seed(5)
    d, inter = 8, 16
    g = MockLinear(d, inter, "blk.mlp.gate_proj")
    u = MockLinear(d, inter, "blk.mlp.up_proj")
    dn = MockLinear(inter, d, "blk.mlp.down_proj")
    entry = types.SimpleNamespace(
        gates=[DiffLinear(g, r=0, compute_dtype=torch.float32)],
        ups=[DiffLinear(u, r=0, compute_dtype=torch.float32)],
        downs=[DiffLinear(dn, r=0, compute_dtype=torch.float32)],
    )
    meta = {"activation": "silu"}          # no mlp_kind key: legacy meta dicts
    net = _headless_net()
    x = torch.randn(2, 4, d)
    out = net._mlp_out(meta, entry, x)
    ref = F.silu(x @ g.frozen_weight) * (x @ u.frozen_weight) @ dn.frozen_weight
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"dense MLP dispatch changed: max|Δ|={err}"
    print(f"[moe] dense _mlp_out dispatch unchanged (max|Δ|={err:.2e}) PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util import run_timed
    run_timed([
        test_moe_matches_hf_reference,
        test_moe_qwen3_shape_with_scale,
        test_moe_dots_matches_reference,
        test_moe_dots_bias_selects_but_never_weights,
        test_moe_dots_gradcheck,
        test_moe_gradcheck,
        test_moe_backward_reaches_adapters,
        test_moe_unrouted_expert_gets_no_grad,
        test_moe_gemma4_matches_reference,
        test_moe_gemma4_gradcheck,
        test_moe_gemma4_backward_reaches_adapters,
        test_moe_gemma4_block_wiring,
        test_dense_dispatch_unchanged,
    ], label="moe")
