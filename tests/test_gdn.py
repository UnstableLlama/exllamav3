"""
CPU tests for the differentiable GatedDeltaNet path (Qwen3.5/3.6 linear
attention) in ``exllamav3/training/gdn.py`` + ``native_llama._gdn_forward``,
and for the interleaved attention output gate (Qwen3.5 full-attn layers).

No GPU, no compiled extension, no real model: the training modules are loaded
under a synthetic package (so their relative imports resolve without importing
the full exllamav3 package, which would build the CUDA ext), EXL3 linears are
mocked as frozen random weights, and the checks are:

  * ``gdn_delta_rule_reference`` matches a verbatim transcription of the
    inference module's own validated reference
    (``torch_recurrent_gated_delta_rule``), including L2 norm and GQA
    grouping, and gradchecks in fp64;
  * ``gdn_causal_conv1d_silu`` matches the inference conv
    (``causal_conv1d_update_function_torch``) run from a zero state;
  * a full GDN block forward (``_gdn_forward``) matches an independent
    plain-torch composition (norm -> projections -> conv+silu -> beta/g ->
    delta rule -> gated RMSNorm -> o_proj + MLP);
  * backprop through a GDN block reaches the LoRA adapters on
    qkv/z/out while the frozen base and the untargeted b/a stay untouched;
  * an interleaved-gate attention block matches a reference that chunks
    q_proj into [q | gate] and multiplies the attention output by
    sigmoid(gate).

Run:  python tests/test_gdn.py
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
# Mock EXL3 linear (same shape as test_native_llama's).
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


def _spec(weight, eps=1e-5, bias=0.0, scale=1.0):
    return {"weight": weight, "eps": eps, "bias": bias, "scale": scale}


def _headless_net():
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = torch.float32
    n.attn_impl = "eager"
    return n


# ----------------------------------------------------------------------------
# Verbatim transcription of the inference module's reference implementation
# (modules/gated_delta_net_fn/gated_delta_rule.py:torch_recurrent_gated_delta_rule),
# the ground truth the CUDA kernel is validated against. Expects q/k/v with the
# SAME head count, so the GQA expansion happens in the caller.
# ----------------------------------------------------------------------------
def _inference_reference_delta_rule(query, key, value, g, beta):
    def l2norm(x, dim=-1, eps=1e-6):
        inv_norm = 1 / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)

    query, key, value = query.float(), key.float(), value.float()
    beta, g = beta.float(), g.float()

    for i in range(sequence_length):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = last_recurrent_state * k_t.unsqueeze(-1)
        kv_mem = kv_mem.sum(dim=-2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale

    return core_attn_out


def test_delta_rule_matches_inference_reference():
    torch.manual_seed(0)
    b, t, nk, dk, grp, dv = 2, 7, 2, 8, 2, 6
    nv = nk * grp
    q = torch.randn(b, t, nk, dk)
    k = torch.randn(b, t, nk, dk)
    v = torch.randn(b, t, nv, dv)
    g = -torch.rand(b, t, nv) * 0.5           # log decay <= 0
    beta = torch.sigmoid(torch.randn(b, t, nv))

    out = _gdn.gdn_delta_rule_reference(q, k, v, g, beta)

    # The inference reference wants uniform head counts: expand q/k the way the
    # split in_proj_qkv layout maps value heads to key heads (j -> j // grp).
    q_e = q.repeat_interleave(grp, dim=2)
    k_e = k.repeat_interleave(grp, dim=2)
    ref = _inference_reference_delta_rule(q_e, k_e, v, g, beta)

    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"delta rule mismatch vs inference reference: max|Δ|={err}"
    print(f"[gdn] delta rule matches inference reference (max|Δ|={err:.2e}) PASSED")


def test_delta_rule_gradcheck():
    torch.manual_seed(1)
    b, t, nk, dk, grp, dv = 1, 4, 1, 3, 2, 2
    nv = nk * grp
    q = torch.randn(b, t, nk, dk, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, t, nk, dk, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, t, nv, dv, dtype=torch.float64, requires_grad=True)
    g = (-torch.rand(b, t, nv, dtype=torch.float64) * 0.5).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, nv, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda *a: _gdn.gdn_delta_rule_reference(*a), (q, k, v, g, beta),
        eps=1e-6, atol=1e-6)
    print("[gdn] delta rule reference gradcheck PASSED")


# ----------------------------------------------------------------------------
# Conv: transcription of causal_conv1d_update_function_torch with zero state.
# ----------------------------------------------------------------------------
def _inference_reference_conv(x, weight, bias):
    bsz, dim, seq_len = x.shape
    conv_kernel_size = weight.shape[-1]
    conv_state = torch.zeros(bsz, dim, conv_kernel_size, dtype=x.dtype)
    y = torch.cat([conv_state[:, :, :conv_kernel_size], x], dim=-1).to(weight.dtype)
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding=0, groups=dim)
    y = F.silu(y[:, :, -seq_len:])
    return y.to(x.dtype)


def test_conv_matches_inference_reference():
    torch.manual_seed(2)
    b, dim, t, kernel = 2, 10, 9, 4
    x = torch.randn(b, dim, t)
    w = torch.randn(dim, kernel) * 0.5
    bias = torch.randn(dim) * 0.1
    out = _gdn.gdn_causal_conv1d_silu(x, w, bias)
    ref = _inference_reference_conv(x, w, bias)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"causal conv mismatch vs inference reference: max|Δ|={err}"
    # And without bias.
    err2 = (_gdn.gdn_causal_conv1d_silu(x, w, None)
            - _inference_reference_conv(x, w, None)).abs().max().item()
    assert err2 < 1e-6
    print(f"[gdn] causal conv+silu matches inference reference (max|Δ|={err:.2e}) PASSED")


# ----------------------------------------------------------------------------
# Full GDN block.
# ----------------------------------------------------------------------------
def _build_gdn_block(d, nk, dk, grp, dv, inter, r=0, dtype=torch.float32):
    nv = nk * grp
    k_dim, v_dim = nk * dk, nv * dv
    conv_dim = 2 * k_dim + v_dim
    kernel = 4

    lins = {
        "qkv": MockLinear(d, conv_dim, "blk.linear_attn.in_proj_qkv", dtype=dtype),
        "z": MockLinear(d, v_dim, "blk.linear_attn.in_proj_z", dtype=dtype),
        "b": MockLinear(d, nv, "blk.linear_attn.in_proj_b", dtype=dtype),
        "a": MockLinear(d, nv, "blk.linear_attn.in_proj_a", dtype=dtype),
        "o": MockLinear(v_dim, d, "blk.linear_attn.out_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }

    norm_a = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)
    norm_m = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)
    gdn_norm_w = nn.Parameter(1.0 + 0.02 * torch.randn(dv, dtype=dtype), requires_grad=False)

    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a)
    entry.mlp_norm_spec = _spec(norm_m)
    entry.attn_post_spec = None
    entry.mlp_post_spec = None
    entry.gdn_norm_spec = _spec(gdn_norm_w, eps=1e-6)
    entry.qkv_proj = DiffLinear(lins["qkv"], r=r, compute_dtype=dtype)
    entry.z_proj = DiffLinear(lins["z"], r=r, compute_dtype=dtype)
    entry.b_proj = DiffLinear(lins["b"], r=0, compute_dtype=dtype)
    entry.a_proj = DiffLinear(lins["a"], r=0, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]

    meta = {
        "kind": "gdn",
        "num_k_heads": nk, "num_v_heads": nv,
        "k_head_dim": dk, "v_head_dim": dv,
        "conv_kernel_size": kernel,
        "beta_scale": 1.0,
        "a_log": torch.randn(nv, dtype=dtype) * 0.3,
        "dt_bias": torch.randn(nv, dtype=dtype) * 0.3,
        "conv1d_weight": (torch.randn(conv_dim, kernel, dtype=dtype) * 0.4),
        "conv1d_bias": torch.randn(conv_dim, dtype=dtype) * 0.1,
        "activation": "silu",
        "layer_scalar": None,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    ref_w.update({"attn_norm": norm_a, "mlp_norm": norm_m, "gdn_norm": gdn_norm_w})
    return entry, meta, ref_w, lins


def _ref_rmsnorm(x, w, eps=1e-5):
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.rsqrt(var)) * w.float()


def _ref_gdn_block(meta, w, hidden):
    """Independent composition of the GDN block from the transcribed inference
    pieces (conv reference + delta-rule reference + GatedRMSNorm.forward_torch
    semantics), sharing no code with training.gdn."""
    b, t, d = hidden.shape
    nk, nv = meta["num_k_heads"], meta["num_v_heads"]
    dk, dv = meta["k_head_dim"], meta["v_head_dim"]
    grp = nv // nk
    k_dim, v_dim = nk * dk, nv * dv

    normed = _ref_rmsnorm(hidden, w["attn_norm"])
    qkv = normed @ w["qkv"]
    z = (normed @ w["z"]).view(b, t, nv, dv)
    b_raw = normed @ w["b"]
    a_raw = normed @ w["a"]
    beta = torch.sigmoid(b_raw.float()) * meta["beta_scale"]
    g = -meta["a_log"].float().exp() * F.softplus(a_raw.float() + meta["dt_bias"].float())

    x = _inference_reference_conv(qkv.transpose(1, 2), meta["conv1d_weight"],
                                  meta["conv1d_bias"]).transpose(1, 2)
    q, k, v = torch.split(x, [k_dim, k_dim, v_dim], dim=-1)
    q = q.view(b, t, nk, dk).repeat_interleave(grp, dim=2)
    k = k.view(b, t, nk, dk).repeat_interleave(grp, dim=2)
    v = v.view(b, t, nv, dv)
    core = _inference_reference_delta_rule(q, k, v, g, beta)   # [b,t,nv,dv]

    # GatedRMSNorm.forward_torch: rmsnorm * w, then * silu(gate) (fp32).
    var = core.float().pow(2).mean(-1, keepdim=True) + 1e-6
    core = core.float() * torch.rsqrt(var) * w["gdn_norm"].float()
    core = core * F.silu(z.float())
    hidden = hidden + core.reshape(b, t, v_dim) @ w["o"]

    normed2 = _ref_rmsnorm(hidden, w["mlp_norm"])
    a = F.silu(normed2 @ w["gate"]) * (normed2 @ w["up"])
    return hidden + a @ w["down"]


def test_gdn_block_matches_reference():
    torch.manual_seed(3)
    d, nk, dk, grp, dv, inter = 16, 2, 8, 2, 6, 32
    entry, meta, ref_w, _ = _build_gdn_block(d, nk, dk, grp, dv, inter, r=0)
    net = _headless_net()

    b, t = 2, 6
    hidden = torch.randn(b, t, d)
    out = net._gdn_forward(meta, entry, hidden)
    ref = _ref_gdn_block(meta, ref_w, hidden)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"GDN block forward mismatch vs reference: max|Δ|={err}"
    print(f"[gdn] block forward matches plain-torch reference (max|Δ|={err:.2e}) PASSED")


def test_gdn_block_backward_reaches_adapters():
    torch.manual_seed(4)
    d, nk, dk, grp, dv, inter = 12, 2, 4, 2, 4, 24
    entry, meta, _, lins = _build_gdn_block(d, nk, dk, grp, dv, inter, r=4)
    net = _headless_net()

    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    # B starts at zero (adapter no-op); nudge it so lora_a also sees gradient
    # through a nonzero product.
    for dl in (entry.qkv_proj, entry.z_proj, entry.o_proj):
        with torch.no_grad():
            dl.lora_b.add_(torch.randn_like(dl.lora_b) * 0.02)

    hidden = torch.randn(2, 5, d, requires_grad=True)
    out = net._gdn_forward(meta, entry, hidden)
    out.square().mean().backward()

    for name, dl in (("qkv", entry.qkv_proj), ("z", entry.z_proj), ("o", entry.o_proj)):
        assert dl.lora_a.grad is not None and dl.lora_a.grad.abs().sum() > 0, \
            f"no gradient reached lora_a of {name}"
        assert dl.lora_b.grad is not None and dl.lora_b.grad.abs().sum() > 0, \
            f"no gradient reached lora_b of {name}"
    for name, dl in (("b", entry.b_proj), ("a", entry.a_proj)):
        assert dl.lora_a is None, f"{name} should be frozen (r=0)"
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_]), \
            f"frozen base weight of {k_} changed"
    print("[gdn] backward reaches qkv/z/o adapters; base + b/a frozen PASSED")


# ----------------------------------------------------------------------------
# Interleaved attention output gate (Qwen3.5 full-attn layers).
# ----------------------------------------------------------------------------
def test_interleaved_gate_block_matches_reference():
    torch.manual_seed(5)
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    dtype = torch.float32
    norm_a = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)
    norm_m = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)
    lins = {
        "q": MockLinear(d, 2 * nq * hd, "blk.self_attn.q_proj", dtype=dtype),  # [q | gate]
        "k": MockLinear(d, nkv * hd, "blk.self_attn.k_proj", dtype=dtype),
        "v": MockLinear(d, nkv * hd, "blk.self_attn.v_proj", dtype=dtype),
        "o": MockLinear(nq * hd, d, "blk.self_attn.o_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a)
    entry.mlp_norm_spec = _spec(norm_m)
    entry.attn_post_spec = entry.mlp_post_spec = None
    entry.q_norm_spec = entry.k_norm_spec = entry.v_norm_spec = None
    entry.q_proj = DiffLinear(lins["q"], r=0, compute_dtype=dtype)
    entry.k_proj = DiffLinear(lins["k"], r=0, compute_dtype=dtype)
    entry.v_proj = DiffLinear(lins["v"], r=0, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=0, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    meta = {
        "kind": "attn",
        "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": hd,
        "sm_scale": hd ** -0.5, "inv_freq": inv_freq, "attn_factor": 1.0,
        "sliding_window": -1, "softcap": 0.0, "activation": "silu",
        "use_k_as_v": False, "interleaved_gate": True, "layer_scalar": None,
    }

    b, t = 2, 5
    hidden = torch.randn(b, t, d, dtype=dtype)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    net = _headless_net()
    net.use_liger = False
    attn_bias = net._attn_bias(None, t, hidden.device, torch.float32)
    out = net._block_forward(meta, entry, hidden, positions, attn_bias)

    # Reference: chunk q_proj output into [q | gate] per head, run plain GQA
    # attention with RoPE, multiply the flattened context by sigmoid(gate).
    def ref_rope(x, positions):
        freqs = positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().unsqueeze(2), emb.sin().unsqueeze(2)
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rot * sin

    normed = _ref_rmsnorm(hidden, norm_a)
    qg = (normed @ lins["q"].frozen_weight).view(b, t, nq, 2 * hd)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(b, t, nq * hd)
    k = (normed @ lins["k"].frozen_weight).view(b, t, nkv, hd)
    v = (normed @ lins["v"].frozen_weight).view(b, t, nkv, hd)
    q, k = ref_rope(q, positions), ref_rope(k, positions)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    rep = nq // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * meta["sm_scale"]
    mask = torch.triu(torch.full((t, t), float("-inf")), 1)
    ctx = torch.softmax(scores + mask, -1) @ v
    ctx = ctx.transpose(1, 2).reshape(b, t, nq * hd)
    ctx = ctx * torch.sigmoid(gate)
    hidden_ref = hidden + ctx @ lins["o"].frozen_weight
    normed2 = _ref_rmsnorm(hidden_ref, norm_m)
    a = F.silu(normed2 @ lins["gate"].frozen_weight) * (normed2 @ lins["up"].frozen_weight)
    ref = hidden_ref + a @ lins["down"].frozen_weight

    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"interleaved-gate block mismatch vs reference: max|Δ|={err}"
    print(f"[gdn] interleaved-gate attn block matches reference (max|Δ|={err:.2e}) PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util import run_timed
    run_timed([
        test_delta_rule_matches_inference_reference,
        test_delta_rule_gradcheck,
        test_conv_matches_inference_reference,
        test_gdn_block_matches_reference,
        test_gdn_block_backward_reaches_adapters,
        test_interleaved_gate_block_matches_reference,
    ], label="gdn")
    print("\nALL GDN TESTS PASSED")
