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
    """
    from ..modules import GatedMLP
    key = getattr(block, "key", "?")
    attn = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    assert attn is not None and mlp is not None, \
        f"{key}: block must have both attention and MLP (parallel/no-op blocks unsupported)"
    assert isinstance(mlp, GatedMLP), \
        f"{key}: only GatedMLP (SiLU/GeLU gated) MLPs are supported, got {type(mlp).__name__}"
    assert mlp.activation_fn in ("silu",), \
        f"{key}: only SiLU gated MLP is supported, got activation {mlp.activation_fn!r}"
    assert getattr(mlp, "act_limit", 0.0) in (0.0, None), \
        f"{key}: gated-MLP act_limit is not supported"
    assert attn.q_norm is None and attn.k_norm is None, \
        f"{key}: attention q/k norms are not supported by the native QLoRA forward"
    assert getattr(attn, "v_norm", None) is None, f"{key}: attention v_norm unsupported"
    assert getattr(attn, "g_proj", None) is None and not getattr(attn, "interleaved_gate", False), \
        f"{key}: attention output gating is not supported"
    assert getattr(attn, "sliding_window", -1) in (-1, 0, None), \
        f"{key}: sliding-window attention is not supported"
    assert not getattr(attn, "logit_softcapping", 0.0), \
        f"{key}: attention logit softcapping is not supported"
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
    return {
        "num_q_heads": attn.num_q_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "sm_scale": attn.sm_scale,
        "attn_eps": rms_norm_eps(block.attn_norm),
        "mlp_eps": rms_norm_eps(block.mlp_norm),
        # RoPE: the llama3-scaled inv_freq lives on the loaded RoPE object.
        "inv_freq": attn.rope.inv_freq,
        "attn_factor": attn.rope.attn_factor,
    }


def block_norms(block):
    """Return the ``(attn_norm, mlp_norm)`` modules of one block."""
    return block.attn_norm, block.mlp_norm


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


# --- frozen quantized linears ----------------------------------------------

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
    return lambda: inner.get_weight_tensor().to(dtype)


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
    return lambda: inner.get_weight_tensor()


# --- token embedding -------------------------------------------------------

def embed_tokens(embed, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Look up token embeddings via exllamav3's ``Embedding`` module, applying its
    optional multiplier / normalization. Returns hidden states on the embedding's
    own device (which may be CPU even when the decoder is on GPU).
    """
    table = embed.embedding
    hidden = table(input_ids.to(table.weight.device))
    if getattr(embed, "multiplier", 1.0) != 1.0:
        hidden = hidden * embed.multiplier
    if getattr(embed, "normalize", False):
        hidden = hidden * (hidden.shape[-1] ** 0.5)
    return hidden


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
