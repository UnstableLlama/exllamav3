"""
The single seam between the native training forward and exllamav3 internals.

``exllamav3/training/native_llama.py`` reconstructs a differentiable
Llama/Mistral decoder on top of an already-loaded ``exllamav3.Model``. Doing so
requires reading exllamav3's *internal* module layout: the ``..modules`` types,
the loaded RoPE table, RMSNorm epsilons, and the trellis weight reconstruction.
Every such reach lives here and nowhere else, so the training code above depends
on a small, named surface instead of scattered attribute access.

Why isolate it: this is the natural API boundary for the work. If it is ever
promoted into exllamav3 as a supported training entry point, this is the file
that moves (or becomes a thin shim); ``native_llama.py`` would be unaffected.
And a standalone trainer that pins exllamav3 would depend on exactly this
surface and nothing deeper.

``..modules`` is imported lazily, inside the functions that need it, so that
importing this module never triggers the CUDA extension build.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch


# --- top-level decoder layout ----------------------------------------------

def split_decoder(model):
    """
    Return ``(embed, blocks, final_norm, lm_head)`` from a loaded exllamav3
    ``Model``, validating the overall module layout. ``blocks`` is the list of
    ``TransformerBlock`` modules, in order.
    """
    from ..modules import Embedding, TransformerBlock, RMSNorm, Linear
    mods = list(model.modules)
    assert isinstance(mods[0], Embedding), \
        f"expected Embedding as first module, got {type(mods[0]).__name__}"
    assert isinstance(mods[-2], RMSNorm), \
        f"expected final RMSNorm as penultimate module, got {type(mods[-2]).__name__}"
    assert isinstance(mods[-1], Linear), \
        f"expected Linear LM head as last module, got {type(mods[-1]).__name__}"
    blocks = [m for m in mods if isinstance(m, TransformerBlock)]
    assert blocks, "no TransformerBlock modules found; unsupported architecture"
    return mods[0], blocks, mods[-2], mods[-1]


# --- per-block structure ---------------------------------------------------

def assert_block_supported(block):
    """
    Reject architectures the native forward can't faithfully reproduce, loudly,
    so a mismatch is an explicit error rather than a silently wrong forward.

    The native block forward reproduces a pre-norm softmax-attention decoder and
    reads every norm / activation / scale from the loaded modules, so it covers
    Llama/Mistral/Qwen2 (plain), Qwen3 (q/k-norm), and Gemma3/4 (q/k/v-norm +
    sandwich post-norms + GeGLU + sliding/full window + per-layer head dims).
    What it still cannot do is rejected here: linear/recurrent attention
    (GatedDeltaNet -> Qwen3.5/3.6), MoE, attention output gating, mRoPE, partial
    rotary, and non-NeoX RoPE.
    """
    from ..modules import GatedMLP, Attention, SlidingAttention
    key = getattr(block, "key", "?")
    attn = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    assert attn is not None and mlp is not None, \
        f"{key}: block must have both attention and MLP (parallel/no-op blocks unsupported)"
    # Softmax attention only -- GatedDeltaNet (linear/recurrent attention, used by
    # Qwen3.5/3.6 for ~3/4 of layers) needs a differentiable recurrent forward we
    # do not have.
    assert isinstance(attn, (Attention, SlidingAttention)), \
        f"{key}: only softmax Attention/SlidingAttention is supported, got " \
        f"{type(attn).__name__} (linear/recurrent attention is unsupported)"
    assert isinstance(mlp, GatedMLP), \
        f"{key}: only GatedMLP is supported (no MoE), got {type(mlp).__name__}"
    assert mlp.activation_fn in ("silu", "gelu"), \
        f"{key}: only SiLU/GeLU gated MLP is supported, got activation {mlp.activation_fn!r}"
    assert getattr(mlp, "act_limit", 0.0) in (0.0, None), \
        f"{key}: gated-MLP act_limit is not supported"
    # q/k/v norms ARE supported now (read from the modules); only attention output
    # gating (interleaved/headwise gate, used by Qwen3.5 full-attn layers) is not.
    assert getattr(attn, "g_proj", None) is None and not getattr(attn, "interleaved_gate", False), \
        f"{key}: attention output gating is not supported"
    assert attn.rope is not None and attn.rope.inv_freq is not None, \
        f"{key}: model loaded without a RoPE table; cannot build positional encoding"
    assert attn.rope.mrope_section is None, f"{key}: mRoPE is not supported"
    assert attn.rope.rope_settings.rope_style.name == "NEOX", \
        f"{key}: only NeoX-style RoPE is supported, got {attn.rope.rope_settings.rope_style.name}"
    assert attn.rope.inv_freq.numel() * 2 == attn.head_dim, \
        f"{key}: partial-rotary RoPE is not supported (rotary_dim != head_dim)"


def block_metadata(block) -> dict:
    """
    Plain-data description of one decoder block's attention / RoPE / norm config,
    consumed by the differentiable block forward. Tensors are referenced, not
    copied (``inv_freq`` is the loaded RoPE table itself).
    """
    attn = block.attn
    sw = getattr(attn, "sliding_window", -1)
    return {
        "num_q_heads": attn.num_q_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "sm_scale": attn.sm_scale,
        # RoPE: the llama3-scaled inv_freq lives on the loaded RoPE object.
        "inv_freq": attn.rope.inv_freq,
        "attn_factor": attn.rope.attn_factor,
        # Per-layer attention window: >0 means sliding (band) attention, else full
        # causal (Gemma alternates local-sliding / global-full layers).
        "sliding_window": int(sw) if sw not in (None, 0) else -1,
        # tanh logit softcapping on the attention scores (Gemma2; 0 = none).
        "softcap": float(getattr(attn, "logit_softcapping", 0.0) or 0.0),
        # gated-MLP activation ("silu" or "gelu"/GeGLU for Gemma).
        "activation": block.mlp.activation_fn,
        # Some Gemma layers reuse the K projection as V (no separate v_proj).
        "use_k_as_v": bool(getattr(attn, "use_k_as_v", False)),
        # Gemma applies a learned per-layer scalar to the whole residual stream at
        # block end (TransformerBlock.forward: x *= layer_scalar_f). None elsewhere.
        "layer_scalar": getattr(block, "layer_scalar_f", None),
    }


def norm_spec(norm) -> Optional[dict]:
    """
    Plain-data description of an ``RMSNorm`` module for the native forward:
    ``{weight, eps, bias, scale}`` (``weight`` is the frozen tensor, or ``None``
    when unweighted). Reproduces ``RMSNorm.forward_torch`` exactly --
    ``y = (x / rms(x)) * scale * (weight + bias)`` -- so Gemma's ``(1 + weight)``
    convention and unweighted v-norm are handled by reading the module's own
    ``constant_bias`` / ``constant_scale`` / ``unweighted`` rather than hardcoding.
    Returns ``None`` for a missing norm.
    """
    if norm is None:
        return None
    from ..modules import RMSNorm
    assert isinstance(norm, RMSNorm), \
        f"native forward only supports RMSNorm, got {type(norm).__name__}"
    return {
        "weight": None if getattr(norm, "unweighted", False) else norm.weight,
        "eps": norm.rms_norm_eps,
        "bias": float(getattr(norm, "constant_bias", 0.0)),
        "scale": float(getattr(norm, "constant_scale", 1.0)),
    }


def block_norms(block):
    """Return the ``(attn_norm, mlp_norm)`` modules of one block."""
    return block.attn_norm, block.mlp_norm


def block_post_norms(block):
    """Return the optional ``(attn_post_norm, mlp_post_norm)`` modules of one
    block (Gemma sandwich norms). Either is ``None`` for a plain pre-norm block."""
    return getattr(block, "attn_post_norm", None), getattr(block, "mlp_post_norm", None)


def attn_qkv_norms(block):
    """Return the optional ``(q_norm, k_norm, v_norm)`` modules of one block's
    attention (Qwen3: q/k; Gemma: q/k/v; Llama/Mistral/Qwen2: all ``None``)."""
    a = block.attn
    return getattr(a, "q_norm", None), getattr(a, "k_norm", None), getattr(a, "v_norm", None)


def head_softcap(lm_head) -> float:
    """Final-logit tanh softcapping on the LM head (Gemma2; 0 = none)."""
    return float(getattr(lm_head, "softcap", 0.0) or 0.0)


def block_device(block):
    """
    The device a block's weights live on (set at load; differs per block under a
    layer-autosplit load, identical for a single-device load).
    """
    return block.device


def to_device(x: torch.Tensor, device) -> torch.Tensor:
    """
    Migrate ``x`` to ``device`` the way exllamav3's own layer-split forward does
    (``Module.prepare_for_device``, ``modules/module.py``): a direct copy, or a
    bounce through CPU when ``no_p2p_copy`` is set (env ``EXLLAMA_NO_P2P_COPY``),
    for rigs without GPU peer access. A no-op when already on ``device``.

    Unlike the native forward (``@torch.inference_mode``), this runs inside the
    training graph; ``.to`` / ``.cpu`` are autograd-friendly, so gradients flow
    back across the boundary.
    """
    if x.device == device:
        return x
    # Lazy import: only reached at runtime on a real multi-device model, never in
    # the single-device CPU tests (which return above).
    from ..modules import module as _module
    if _module.no_p2p_copy:
        return x.cpu().to(device)
    return x.to(device)


def attn_projections(block):
    """Return the ``(q_proj, k_proj, v_proj, o_proj)`` linears of one block."""
    a = block.attn
    return a.q_proj, a.k_proj, a.v_proj, a.o_proj


def mlp_projections(block):
    """
    Return ``(gates, ups, downs)`` linear lists of one block's gated MLP. Each is
    a list because a very wide MLP may be sliced across the intermediate dim.
    """
    m = block.mlp
    return m.gates, m.ups, m.downs


def rms_norm_eps(norm) -> float:
    """The epsilon of an exllamav3 ``RMSNorm`` module."""
    return norm.rms_norm_eps


# --- variable-length (packed) attention ------------------------------------

def attn_varlen(q, k, v, cu_seqlens, max_seqlen, sm_scale,
                window: int = -1, softcap: float = 0.0):
    """
    Variable-length (packed) attention for the training forward, via exllamav3's
    own autograd-capable flash wrapper -- the O(t) primitive that lets sample
    packing isolate documents without ever building a ``[t, t]`` mask.

    ``q`` / ``k`` / ``v`` are ``[total_tokens, num_heads, head_dim]``: every
    document of a packed batch concatenated into one token stream, with
    ``cu_seqlens`` (int32, shape ``[num_docs + 1]``, cumulative document lengths)
    marking the per-document boundaries so a document never attends across one.
    ``max_seqlen`` is the longest document length.

    Routed through ``attention_fn.attn_dispatch`` with no cache, so it skips every
    paged/cache backend and lands on ``fn_flash_attn_varlen_func`` -- the upstream
    ``flash_attn_varlen_func``, which (unlike exllamav3's inference kernels) is NOT
    wrapped in ``inference_mode`` and has a real backward. Requires
    ``head_dim <= 256`` (FA2 limit; the caller routes larger heads elsewhere).

    ``window > 0`` is a sliding window expressed as exllamav3's per-layer
    ``sliding_window`` (a token attends to itself + ``window - 1`` previous tokens
    = ``window`` total). We hand ``attn_dispatch`` ``window - 1`` because its
    ``get_window_size`` wraps the value as the FA2 left-window ``(w, 0)`` -- so the
    result matches the eager reference's ``-window`` diagonal exactly. ``softcap``
    applies tanh logit softcapping. Returns ``[total_tokens, num_heads, head_dim]``.
    """
    from ..modules.attention_fn.dispatch import attn_dispatch
    # attn_dispatch reads [bsz, q_len, num_heads, head_dim]; the varlen backend
    # asserts bsz == 1 and squeezes it, reading boundaries from cu_seqlens. Present
    # the flattened stream as a single batch row; the result is [total, nh, hd].
    window_size = (int(window) - 1) if (window and window > 0) else None
    return attn_dispatch(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        cache=None,
        causal=True,
        sm_scale=float(sm_scale),
        cu_seqlens=cu_seqlens,
        max_seqlen=int(max_seqlen),
        window_size=window_size,
        softcap=float(softcap or 0.0),
    )


# --- frozen quantized linears ----------------------------------------------

# Optional dequant profiling (``--profile-dequant``). When enabled, every
# frozen-weight reconstruction (block linears, LM head, head slices) times
# itself into this mutable dict -- answering "how much of a training step is
# trellis reconstruction", the load-bearing question for the dequant-count
# optimizations (see doc/qlora_optimization_audit.md A1). Costs a device sync
# either side of each reconstruction while enabled (a diagnostic mode; the
# measured share of wall time is still representative). Disabled = one global
# read per call, negligible next to the matmul it precedes.
_DEQUANT_PROFILE: Optional[dict] = None


def profile_dequant(state: Optional[dict]) -> None:
    """Enable (pass a dict with ``calls``/``s`` keys) or disable (pass None)
    reconstruction timing for all frozen-weight closures."""
    global _DEQUANT_PROFILE
    _DEQUANT_PROFILE = state


def _timed_reconstruct(fn):
    """Wrap a weight-producing closure so it accumulates into the profile dict
    when profiling is on. The check runs at call time, so enabling/disabling
    mid-run needs no closure rebuild."""
    def wrapped(*a):
        p = _DEQUANT_PROFILE
        if p is None:
            return fn(*a)
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = fn(*a)
        if w.is_cuda:
            torch.cuda.synchronize(w.device)
        p["calls"] += 1
        p["s"] += time.perf_counter() - t0
        return w
    return wrapped


def is_loaded(linear) -> bool:
    """True once a native ``Linear`` has its inner (trellis / fp16) weight."""
    return getattr(linear, "inner", None) is not None


def linear_device(linear):
    """Device a linear's weights live on (the trellis device when quantized)."""
    try:
        return linear.inner.trellis.device
    except Exception:
        return getattr(linear, "device", None)


def frozen_weight_closure(linear, dtype: torch.dtype) -> Callable[[], torch.Tensor]:
    """
    Closure that reconstructs the frozen effective weight (``[in, out]``) from the
    EXL3 trellis on every call, cast to ``dtype``. Recomputing rather than caching
    is what lets the backward pass avoid stashing the dense weight.
    """
    inner = linear.inner
    return _timed_reconstruct(lambda: inner.get_weight_tensor().to(dtype))


def frozen_bias(linear, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """The linear's frozen bias cast to ``dtype``, or ``None`` if it has none."""
    get_bias = getattr(linear.inner, "get_bias_tensor", None)
    if get_bias is None:
        return None
    b = get_bias()
    return b.to(dtype) if b is not None else None


def head_weight_closure(lm_head) -> Callable[[], torch.Tensor]:
    """
    Closure for the frozen LM-head weight in ``[hidden, vocab]`` orientation (no
    dtype cast; the fused-CE head promotes to >=fp32 internally).
    """
    inner = lm_head.inner
    return _timed_reconstruct(lambda: inner.get_weight_tensor())


def head_weight_slice_closure(lm_head):
    """
    For chunked-vocab head loss: return ``(slice_fn, out_features, granularity)``
    where ``slice_fn(n_start, n_features) -> [hidden, n_features]`` reconstructs only
    those output columns, or ``None`` if the head can't slice efficiently.

    An EXL3 head reconstructs just the requested columns (``get_weight_tensor_slice``)
    so the fused CE never materializes the full ``[hidden, vocab]`` weight + its fp32
    upcast -- the dominant memory spike on the output device for large vocabularies.
    Other head types (e.g. an unquantized ``[hidden, vocab]`` tensor) fall back to a
    plain column index, which still avoids the full-vocab fp32 logits/softmax.
    """
    inner = lm_head.inner
    sliced = getattr(inner, "get_weight_tensor_slice", None)
    if sliced is not None:
        gran = getattr(inner, "RECONSTRUCT_SLICE_GRANULARITY_N", None)
        if gran is None:
            # Module-level constant on the EXL3 linear's module.
            import exllamav3.modules.quant.exl3 as _exl3
            gran = _exl3.RECONSTRUCT_SLICE_GRANULARITY_N
        return _timed_reconstruct(lambda s, n: sliced(s, n)), inner.out_features, gran
    # Generic fallback: index the full (already-resident) weight. No reconstruction
    # spike to avoid, but the chunked CE still bounds the logits/softmax memory.
    get_full = getattr(inner, "get_weight_tensor", None)
    if get_full is None:
        return None
    out_features = getattr(inner, "out_features", None)
    if out_features is None:
        return None
    return _timed_reconstruct(lambda s, n: get_full()[:, s:s + n]), out_features, 1


# --- token embedding -------------------------------------------------------

def embed_weight(embed) -> torch.Tensor:
    """The input-embedding weight tensor (``[vocab, hidden]``) of an ``Embedding``
    module -- the tensor to clone when fully training the embeddings."""
    return embed.embedding.weight


def embed_apply(embed, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the ``Embedding`` module's optional multiplier / normalization to
    already-looked-up hidden states (shared by the frozen and trainable paths)."""
    if getattr(embed, "multiplier", 1.0) != 1.0:
        hidden = hidden * embed.multiplier
    if getattr(embed, "normalize", False):
        hidden = hidden * (hidden.shape[-1] ** 0.5)
    return hidden


def embed_tokens(embed, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Look up token embeddings via exllamav3's ``Embedding`` module, applying its
    optional multiplier / normalization. Returns hidden states on the embedding's
    own device (which may be CPU even when the decoder is on GPU).
    """
    table = embed.embedding
    hidden = table(input_ids.to(table.weight.device))
    return embed_apply(embed, hidden)


# --- runtime LoRA slots (so native generation reflects the adapter) --------

def set_runtime_lora(linear, owner, a: torch.Tensor, b: torch.Tensor) -> None:
    """
    Install adapter tensors into a native ``Linear``'s runtime LoRA slots, keyed
    by ``owner``, so ``model.forward`` / generation applies them. ``a`` / ``b``
    are moved to the linear's device.
    """
    linear.lora_a_tensors[owner] = a.to(linear.device)
    linear.lora_b_tensors[owner] = b.to(linear.device)


def clear_runtime_lora(linear, owner) -> None:
    """Remove ``owner``'s adapter tensors from a native ``Linear``'s LoRA slots."""
    linear.lora_a_tensors.pop(owner, None)
    linear.lora_b_tensors.pop(owner, None)
