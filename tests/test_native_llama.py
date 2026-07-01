"""
CPU tests for the transformers-free differentiable Llama forward
(``exllamav3/training/native_llama.py``).

No GPU, no compiled extension, no transformers, no real model: this loads the
training modules under a synthetic package (so their relative imports resolve
without importing the full exllamav3 package, which would build the CUDA ext),
mocks the EXL3 linears as frozen random weights, and checks:

  * ``DiffLinear`` reproduces ``x @ W + scale·x@A@B`` and gradchecks, with the
    frozen base receiving no gradient;
  * a single decoder block's forward matches an independent plain-torch
    reference (RMSNorm + GQA/NeoX-RoPE attention + SwiGLU + residuals);
  * backprop through the block + a fused-CE head reaches the LoRA adapters in
    every projection while leaving the base weights untouched.

Run:  python tests/test_native_llama.py
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
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA
_rotate_half_neox = _nl._rotate_half_neox
fused_linear_cross_entropy = _fce.fused_linear_cross_entropy


# ----------------------------------------------------------------------------
# Mock EXL3 linear: a frozen random weight masquerading as a trellis layer.
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
    def __init__(self, in_features, out_features, key, scale=0.05, dtype=torch.float64):
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


def _ns_norm(dim, dtype=torch.float64):
    ns = types.SimpleNamespace()
    ns.weight = nn.Parameter(1.0 + 0.02 * torch.randn(dim, dtype=dtype), requires_grad=False)
    return ns


# ----------------------------------------------------------------------------
# Independent plain-torch reference for one Llama decoder block.
# ----------------------------------------------------------------------------
def _ref_rmsnorm(x, w, eps):
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.rsqrt(var)) * w.float()


def _ref_rope(x, inv_freq, positions):
    # x: [b,t,nh,hd]
    freqs = positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    emb = torch.cat((freqs, freqs), -1)
    cos = emb.cos().unsqueeze(2)
    sin = emb.sin().unsqueeze(2)
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def _ref_block(meta, weights, hidden, positions):
    eps_a, eps_m = meta["attn_eps"], meta["mlp_eps"]
    nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]
    sm = meta["sm_scale"]
    inv_freq = meta["inv_freq"]
    b, t, _ = hidden.shape

    normed = _ref_rmsnorm(hidden, weights["attn_norm"], eps_a)
    q = (normed @ weights["q"]).view(b, t, nq, hd)
    k = (normed @ weights["k"]).view(b, t, nkv, hd)
    v = (normed @ weights["v"]).view(b, t, nkv, hd)
    q = _ref_rope(q, inv_freq, positions)
    k = _ref_rope(k, inv_freq, positions)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    rep = nq // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * sm
    mask = torch.triu(torch.full((t, t), float("-inf"), dtype=scores.dtype), 1)
    scores = scores + mask
    ctx = torch.softmax(scores, -1) @ v
    ctx = ctx.transpose(1, 2).reshape(b, t, nq * hd)
    hidden = hidden + ctx @ weights["o"]

    normed2 = _ref_rmsnorm(hidden, weights["mlp_norm"], eps_m)
    a = F.silu(normed2 @ weights["gate"]) * (normed2 @ weights["up"])
    hidden = hidden + a @ weights["down"]
    return hidden


def _spec(weight, eps=1e-5, bias=0.0, scale=1.0):
    # Mirror of backbone.norm_spec (weight None = unweighted).
    return {"weight": weight, "eps": eps, "bias": bias, "scale": scale}


def _ref_rmsnorm_b(x, w, eps, bias):
    # RMSNorm with optional (weight + bias); w=None => unweighted (Gemma v-norm).
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    xn = x.float() * torch.rsqrt(var)
    if w is None:
        return xn
    return xn * (w.float() + bias) if bias != 0.0 else xn * w.float()


def _ref_block_gemma(meta, weights, hidden, positions, feats):
    """Independent reference for a block with the Gemma/Qwen3 features enabled:
    (1+w) norms, per-head q/k/v norm, sliding-window mask, attn softcap, GeGLU,
    and sandwich post-norms (x = x + post_norm(sublayer_out))."""
    eps_a, eps_m = meta["attn_eps"], meta["mlp_eps"]
    nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]
    sm, inv_freq = meta["sm_scale"], meta["inv_freq"]
    window, softcap = meta["sliding_window"], meta["softcap"]
    bias = weights["norm_bias"]
    b, t, _ = hidden.shape

    normed = _ref_rmsnorm_b(hidden, weights["attn_norm"], eps_a, bias)
    q = (normed @ weights["q"]).view(b, t, nq, hd)
    k = (normed @ weights["k"]).view(b, t, nkv, hd)
    v = (normed @ weights["v"]).view(b, t, nkv, hd)
    if feats.get("qk_norm"):
        q = _ref_rmsnorm_b(q, weights["q_norm"], eps_a, bias)
        k = _ref_rmsnorm_b(k, weights["k_norm"], eps_a, bias)
    if feats.get("v_norm"):
        v = _ref_rmsnorm_b(v, None, eps_a, 0.0)
    q = _ref_rope(q, inv_freq, positions)
    k = _ref_rope(k, inv_freq, positions)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    rep = nq // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * sm
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    mask = torch.zeros(t, t, dtype=scores.dtype)
    mask.masked_fill_(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    if window and window > 0:
        mask.masked_fill_(torch.tril(torch.ones(t, t, dtype=torch.bool), -window), float("-inf"))
    scores = scores + mask
    ctx = torch.softmax(scores, -1) @ v
    ctx = ctx.transpose(1, 2).reshape(b, t, nq * hd)
    attn_out = ctx @ weights["o"]
    if feats.get("post_norm"):
        hidden = hidden + _ref_rmsnorm_b(attn_out, weights["attn_post"], eps_a, bias)
    else:
        hidden = hidden + attn_out

    normed2 = _ref_rmsnorm_b(hidden, weights["mlp_norm"], eps_m, bias)
    actfn = F.silu if meta["activation"] == "silu" \
        else (lambda z: F.gelu(z, approximate="tanh"))
    a = actfn(normed2 @ weights["gate"]) * (normed2 @ weights["up"])
    mlp_out = a @ weights["down"]
    if feats.get("post_norm"):
        hidden = hidden + _ref_rmsnorm_b(mlp_out, weights["mlp_post"], eps_m, bias)
    else:
        hidden = hidden + mlp_out
    if meta.get("layer_scalar") is not None:
        hidden = hidden * meta["layer_scalar"]
    return hidden


def _build_block(d, nq, nkv, hd, inter, dtype=torch.float64, r=0, *,
                 qk_norm=False, v_norm=False, post_norm=False,
                 activation="silu", window=-1, softcap=0.0, norm_bias=0.0,
                 layer_scalar=None):
    """Build a synthetic block + matching reference weights. Flags toggle the
    Gemma/Qwen3 features so one builder covers the plain and extended paths."""
    norm_a = _ns_norm(d, dtype)
    norm_m = _ns_norm(d, dtype)
    lins = {
        "q": MockLinear(d, nq * hd, "blk.self_attn.q_proj", dtype=dtype),
        "k": MockLinear(d, nkv * hd, "blk.self_attn.k_proj", dtype=dtype),
        "v": MockLinear(d, nkv * hd, "blk.self_attn.v_proj", dtype=dtype),
        "o": MockLinear(nq * hd, d, "blk.self_attn.o_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a.weight, bias=norm_bias)
    entry.mlp_norm_spec = _spec(norm_m.weight, bias=norm_bias)
    # Optional per-head q/k/v norms (q/k weighted; v unweighted, as in Gemma).
    qn = nn.Parameter(1.0 + 0.02 * torch.randn(hd, dtype=dtype), requires_grad=False)
    kn = nn.Parameter(1.0 + 0.02 * torch.randn(hd, dtype=dtype), requires_grad=False)
    entry.q_norm_spec = _spec(qn, bias=norm_bias) if qk_norm else None
    entry.k_norm_spec = _spec(kn, bias=norm_bias) if qk_norm else None
    entry.v_norm_spec = _spec(None) if v_norm else None
    # Optional sandwich post-norms.
    pa = _ns_norm(d, dtype)
    pm = _ns_norm(d, dtype)
    entry.attn_post_spec = _spec(pa.weight, bias=norm_bias) if post_norm else None
    entry.mlp_post_spec = _spec(pm.weight, bias=norm_bias) if post_norm else None
    entry.q_proj = DiffLinear(lins["q"], r=r, compute_dtype=dtype)
    entry.k_proj = DiffLinear(lins["k"], r=r, compute_dtype=dtype)
    entry.v_proj = DiffLinear(lins["v"], r=r, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=r, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=r, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=r, compute_dtype=dtype)]

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    meta = {
        "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": hd,
        "sm_scale": hd ** -0.5, "attn_eps": 1e-5, "mlp_eps": 1e-5,
        "inv_freq": inv_freq, "attn_factor": 1.0,
        "sliding_window": window, "softcap": softcap, "activation": activation,
        "use_k_as_v": False, "layer_scalar": layer_scalar,
    }
    ref_weights = {
        "attn_norm": norm_a.weight, "mlp_norm": norm_m.weight,
        "q": lins["q"].frozen_weight, "k": lins["k"].frozen_weight,
        "v": lins["v"].frozen_weight, "o": lins["o"].frozen_weight,
        "gate": lins["gate"].frozen_weight, "up": lins["up"].frozen_weight,
        "down": lins["down"].frozen_weight,
        "q_norm": qn, "k_norm": kn, "attn_post": pa.weight, "mlp_post": pm.weight,
        "norm_bias": norm_bias,
    }
    return entry, meta, ref_weights, lins


def _headless_net():
    # We only need the bound helper methods (_rmsnorm/_apply_rope/_attn_bias/
    # _block_forward); none touch construction-time state.
    return NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)


def test_difflinear_matches_reference_and_gradchecks():
    torch.manual_seed(0)
    d_in, d_out, r = 6, 8, 3
    lin = MockLinear(d_in, d_out, "x", dtype=torch.float64)
    dl = DiffLinear(lin, r=r, alpha=2.0 * r, compute_dtype=torch.float64)
    with torch.no_grad():
        dl.lora_b.copy_(torch.randn(r, d_out, dtype=torch.float64) * 0.1)

    x = torch.randn(4, d_in, dtype=torch.float64)
    y = dl(x)
    # Adapters are fp32 master weights; EXL3LoRAFunction casts them to the input
    # dtype internally, so the fp64 reference casts them to double too.
    a64 = dl.lora_a.detach().double()
    b64 = dl.lora_b.detach().double()
    ref = x @ lin.frozen_weight + dl.scale * (x @ a64 @ b64)
    assert torch.allclose(y, ref, atol=1e-10), "DiffLinear forward mismatch"

    # gradcheck wrt input and adapters (all fp64 for numerical stability); the
    # base weight is a constant supplied through the weight_fn closure.
    def f(x_, a_, b_):
        return _qll.EXL3LoRAFunction.apply(
            x_, a_, b_, None, dl.scale, lambda: lin.frozen_weight
        )
    a = a64.clone().requires_grad_(True)
    b = b64.clone().requires_grad_(True)
    xin = x.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (xin, a, b), eps=1e-6, atol=1e-6)
    print("[difflinear] forward matches reference + gradcheck PASSED")


def test_block_matches_reference():
    torch.manual_seed(1)
    # The residual stream runs in fp32 (the block promotes via .float()), so the
    # whole block is an fp32 computation; compare against the fp32 reference.
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    entry, meta, refw, lins = _build_block(d, nq, nkv, hd, inter, dtype=torch.float32, r=0)
    net = _headless_net()

    b, t = 2, 5
    hidden = torch.randn(b, t, d, dtype=torch.float32)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    attn_bias = net._attn_bias(None, t, hidden.device, torch.float32)

    out = net._block_forward(meta, entry, hidden, positions, attn_bias)
    ref = _ref_block(meta, refw, hidden, positions)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"block forward mismatch vs reference: max|Δ|={err}"
    print(f"[block] forward matches plain-torch reference (max|Δ|={err:.2e}) PASSED")


def test_gemma_block_matches_reference():
    # The Gemma/Qwen3 feature path: (1+w) norms, per-head q/k/v norm, alternating
    # sliding window, attn softcap, GeGLU, and sandwich post-norms -- all at once.
    torch.manual_seed(3)
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    feats = {"qk_norm": True, "v_norm": True, "post_norm": True}
    entry, meta, refw, lins = _build_block(
        d, nq, nkv, hd, inter, dtype=torch.float32, r=0,
        qk_norm=True, v_norm=True, post_norm=True,
        activation="gelu", window=3, softcap=50.0, norm_bias=1.0,
        layer_scalar=0.7,
    )
    net = _headless_net()

    b, t = 2, 7
    hidden = torch.randn(b, t, d, dtype=torch.float32)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    attn_bias = net._attn_bias(None, t, hidden.device, torch.float32, window=3)

    out = net._block_forward(meta, entry, hidden, positions, attn_bias)
    ref = _ref_block_gemma(meta, refw, hidden, positions, feats)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"gemma block forward mismatch vs reference: max|Δ|={err}"
    print(f"[block-gemma] q/k/v-norm + sandwich + GeGLU + sliding + softcap "
          f"matches reference (max|Δ|={err:.2e}) PASSED")


def test_block_backward_reaches_adapters_only():
    torch.manual_seed(2)
    d, nq, nkv, hd, inter, r = 16, 4, 2, 8, 32, 4
    entry, meta, refw, lins = _build_block(d, nq, nkv, hd, inter, dtype=torch.float32, r=r)
    net = _headless_net()

    # B inits to zero (no-op adapter), which makes grad_A exactly zero on the
    # first step; give B a nonzero value so grad flow into A is exercised too.
    adapters_all = [entry.q_proj, entry.k_proj, entry.v_proj, entry.o_proj,
                    entry.gates[0], entry.ups[0], entry.downs[0]]
    with torch.no_grad():
        for adp in adapters_all:
            adp.lora_b.copy_(torch.randn_like(adp.lora_b) * 0.1)

    # Snapshot frozen base weights.
    base0 = {k: v.frozen_weight.clone() for k, v in lins.items()}

    b, t, V = 2, 6, 20
    hidden = torch.randn(b, t, d, dtype=torch.float32, requires_grad=True)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    attn_bias = net._attn_bias(None, t, hidden.device, torch.float32)

    head = torch.randn(d, V, dtype=torch.float32)        # frozen LM head
    labels = torch.randint(0, V, (b, t))

    out = net._block_forward(meta, entry, hidden, positions, attn_bias)
    loss = fused_linear_cross_entropy(out, lambda: head, labels, chunk=8, shift=True)
    loss.backward()

    # Every adapter must have received a gradient (signal flows through rope,
    # attention, GQA repeat, softmax, SwiGLU and both residual joins).
    adapters = [entry.q_proj, entry.k_proj, entry.v_proj, entry.o_proj,
                entry.gates[0], entry.ups[0], entry.downs[0]]
    for adp in adapters:
        assert adp.lora_a.grad is not None and adp.lora_a.grad.abs().sum() > 0, \
            f"no grad into lora_a of {adp.key}"
        assert adp.lora_b.grad is not None, f"no grad into lora_b of {adp.key}"
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0, "no grad into input"

    # Optimiser step must not touch the frozen base.
    opt = torch.optim.SGD([p for a in adapters for p in (a.lora_a, a.lora_b)], lr=0.1)
    opt.step()
    for k, v in lins.items():
        assert torch.equal(v.frozen_weight, base0[k]), f"frozen base {k} changed!"
    print("[block] backward reaches all adapters, base stays frozen PASSED")


def test_packing_block_isolation():
    # Sample packing: a packed block's per-document outputs must EQUAL running each
    # document alone -- the block-diagonal mask (seg_ids) plus per-document RoPE
    # position reset. This is the eager reference path (the flash-varlen path is
    # GPU-only). A leak here would mean packed training silently mixes documents.
    torch.manual_seed(5)
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    entry, meta, refw, lins = _build_block(d, nq, nkv, hd, inter, dtype=torch.float32, r=0)
    net = _headless_net()

    L0, L1 = 4, 3
    doc0 = torch.randn(1, L0, d, dtype=torch.float32)
    doc1 = torch.randn(1, L1, d, dtype=torch.float32)

    def run_alone(doc):
        t = doc.shape[1]
        pos = torch.arange(t).unsqueeze(0)
        bias = net._attn_bias(None, t, doc.device, torch.float32)
        return net._block_forward(meta, entry, doc, pos, bias)

    ref0, ref1 = run_alone(doc0), run_alone(doc1)

    # Pack the two documents into one sequence: seg ids mark membership, positions
    # reset per document, and the bias is built block-diagonal from seg_ids.
    packed = torch.cat([doc0, doc1], dim=1)                  # [1, L0+L1, d]
    seg = torch.tensor([[0] * L0 + [1] * L1])
    pos = torch.tensor([list(range(L0)) + list(range(L1))])
    bias = net._attn_bias(None, L0 + L1, packed.device, torch.float32, seg_ids=seg)
    out = net._block_forward(meta, entry, packed, pos, bias)

    e0 = (out[:, :L0] - ref0).abs().max().item()
    e1 = (out[:, L0:] - ref1).abs().max().item()
    assert e0 < 1e-5 and e1 < 1e-5, f"packing leak: doc0 max|Δ|={e0}, doc1 max|Δ|={e1}"
    print(f"[packing] packed block == per-document (max|Δ|={max(e0, e1):.2e}) PASSED")


def test_packing_pad_no_nan():
    # Trailing pad positions (attention_mask=0; seg id inherits the last document)
    # must not produce NaNs at real positions: a pad query attends back into its
    # document (never fully masked), and real queries never attend to pad keys.
    torch.manual_seed(6)
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    entry, meta, refw, lins = _build_block(d, nq, nkv, hd, inter, dtype=torch.float32, r=0)
    net = _headless_net()

    L0, L1, pad = 3, 2, 2
    t = L0 + L1 + pad
    hidden = torch.randn(1, t, d, dtype=torch.float32)
    seg = torch.tensor([[0] * L0 + [1] * L1 + [1] * pad])   # pad inherits last doc seg
    am = torch.tensor([[1] * (L0 + L1) + [0] * pad])
    pos = torch.tensor([list(range(L0)) + list(range(L1)) + [0] * pad])
    bias = net._attn_bias(am, t, hidden.device, torch.float32, seg_ids=seg)
    out = net._block_forward(meta, entry, hidden, pos, bias)
    assert torch.isfinite(out[:, :L0 + L1]).all(), "NaN/inf at real positions under packing"
    print("[packing] trailing pads produce no NaN at real positions PASSED")


def test_modules_to_save_param_groups():
    # Full-trained embed/head (modules_to_save) must be optimized but excluded
    # from weight decay (decaying a whole embedding table is harmful), while LoRA
    # params keep it. Build a headless net and set only the attrs the helpers read.
    net = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    torch.nn.Module.__init__(net)                       # set up _parameters/_buffers
    net._wrappers = []                                  # no LoRA wrappers here
    net.embed_weight = torch.nn.Parameter(torch.zeros(5, 3))
    net.head_weight = torch.nn.Parameter(torch.zeros(3, 5))

    assert net.lora_parameters() == []
    ms = net.modules_to_save_parameters()
    assert ms == [net.embed_weight, net.head_weight]
    assert net.trainable_parameters() == [net.embed_weight, net.head_weight]
    assert net.num_trainable() == 5 * 3 + 3 * 5

    groups = net.param_groups(weight_decay=0.01)
    # group 0 = LoRA (wd kept, here empty); group 1 = embed/head (wd forced to 0).
    assert groups[0]["weight_decay"] == 0.01 and groups[0]["params"] == []
    assert groups[1]["weight_decay"] == 0.0
    assert groups[1]["params"] == [net.embed_weight, net.head_weight]

    # With no modules_to_save, there is no second group at all.
    net.embed_weight = net.head_weight = None
    g = net.param_groups(weight_decay=0.01)
    assert len(g) == 1 and g[0]["weight_decay"] == 0.01
    print("[modules_to_save] param-group split (embed/head = no weight decay) PASSED")


def test_supervised_head_ce_chunking_matches_naive():
    # The train_head / lora_head / final-softcap loss path must not materialize
    # all supervised logits at once, but chunking must preserve the exact mean CE
    # semantics and gradients for every trainable surface.
    torch.manual_seed(7)
    n, d, v, r = 23, 9, 31, 4
    labels = torch.randint(0, v, (n,))
    labels[::4] = -100
    valid = labels != -100
    scale = 1.7
    softcap = 12.0

    dtype = torch.float32  # production head-LoRA params are fp32; logits are CE-upcast to fp32
    h_ref = torch.randn(n, d, dtype=dtype, requires_grad=True)
    w_ref = torch.randn(d, v, dtype=dtype, requires_grad=True)
    a_ref = torch.randn(d, r, dtype=dtype, requires_grad=True)
    b_ref = torch.randn(r, v, dtype=dtype, requires_grad=True)

    logits = h_ref[valid] @ w_ref + scale * ((h_ref[valid] @ a_ref) @ b_ref)
    logits = softcap * torch.tanh(logits / softcap)
    loss_ref = F.cross_entropy(logits, labels[valid])
    loss_ref.backward()

    h = h_ref.detach().clone().requires_grad_(True)
    w = w_ref.detach().clone().requires_grad_(True)
    a = a_ref.detach().clone().requires_grad_(True)
    b = b_ref.detach().clone().requires_grad_(True)

    net = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    net.lora_head = True
    net.head_lora_a = a
    net.head_lora_b = b
    net._module_lora_scale = scale
    net.final_softcap = softcap

    loss = net._supervised_head_cross_entropy(h, labels, valid, w, token_chunk=5)
    loss.backward()

    assert torch.allclose(loss, loss_ref, atol=1e-6), "chunked supervised CE loss mismatch"
    assert torch.allclose(h.grad, h_ref.grad, atol=1e-6), "hidden grad mismatch"
    assert torch.allclose(w.grad, w_ref.grad, atol=1e-6), "head weight grad mismatch"
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-6), "head LoRA A grad mismatch"
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-6), "head LoRA B grad mismatch"
    print("[head-ce] supervised chunked CE matches naive logits PASSED")


def test_supervised_head_ce_accepts_inference_weight():
    # Frozen EXL3 head reconstruction can return an inference tensor. The
    # supervised-head autograd path must not save/clone the whole head; it should
    # recompute through the frozen-head custom Function and still produce the same
    # hidden gradient as a normal cloned-weight reference.
    torch.manual_seed(8)
    n, d, v = 17, 7, 19
    with torch.inference_mode():
        frozen_weight = torch.randn(d, v, dtype=torch.float32)
    labels = torch.randint(0, v, (n,))
    labels[::5] = -100
    valid = labels != -100

    h_ref = torch.randn(n, d, dtype=torch.float32, requires_grad=True)
    logits = h_ref[valid] @ frozen_weight.clone()
    loss_ref = F.cross_entropy(logits, labels[valid])
    loss_ref.backward()

    h = h_ref.detach().clone().requires_grad_(True)
    net = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    net.lora_head = False
    net.final_softcap = 0.0
    loss = net._supervised_head_cross_entropy(
        h, labels, valid, None, token_chunk=4, weight_fn=lambda: frozen_weight)
    loss.backward()

    assert torch.allclose(loss, loss_ref, atol=1e-6), "inference-weight CE loss mismatch"
    assert torch.allclose(h.grad, h_ref.grad, atol=1e-6), "inference-weight hidden grad mismatch"
    print("[head-ce] frozen inference-tensor head path PASSED")


def main():
    test_difflinear_matches_reference_and_gradchecks()
    test_block_matches_reference()
    test_gemma_block_matches_reference()
    test_block_backward_reaches_adapters_only()
    test_packing_block_isolation()
    test_packing_pad_no_nan()
    test_modules_to_save_param_groups()
    test_supervised_head_ce_chunking_matches_naive()
    test_supervised_head_ce_accepts_inference_weight()
    print("\nAll native-Llama differentiable-forward checks passed.")


if __name__ == "__main__":
    main()
