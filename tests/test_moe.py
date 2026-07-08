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
  * ``_mlp_out`` dispatches on ``meta["mlp_kind"]`` (dense stays dense).

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
def _build_moe(d, inter, num_experts, top_k, shared_inter=0, shared_gate=True,
               per_expert_scale=None, r=0, expert_r=None, dtype=torch.float32,
               router_scale=0.5):
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

    meta = {
        "mlp_kind": "moe",
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "per_expert_scale": per_expert_scale,
        "activation": "silu",
        "shared_activation": "silu" if shared_inter else None,
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
        test_moe_gradcheck,
        test_moe_backward_reaches_adapters,
        test_moe_unrouted_expert_gets_no_grad,
        test_dense_dispatch_unchanged,
    ], label="moe")
