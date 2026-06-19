"""
Fused linear cross-entropy for memory-efficient LM-head training.

The LM head is the single biggest activation-memory spike in LLM
fine-tuning: materialising logits of shape ``[tokens, vocab]`` (plus their
gradient) for a 128k+ vocabulary dwarfs everything else. This module computes
the cross-entropy loss and the gradient w.r.t. the hidden states **without
ever holding the full logit tensor**, by streaming over token chunks.

Assumptions matching QLoRA:
  * the LM-head weight is **frozen** (no LoRA on the head), so we only need
    the gradient w.r.t. the hidden states, not w.r.t. the weight. This lets
    us recompute the (dequantized) head weight in the backward instead of
    storing it -- a head weight can be ~1 GB on its own.
  * weight orientation is ``logits = hidden @ W`` with ``W`` of shape
    ``[hidden, vocab]`` -- exactly what ``LinearEXL3.get_weight_tensor()``
    returns.

The core ``FusedLinearCrossEntropy`` autograd Function is gradchecked in
``tests/test_fused_ce.py`` against ``torch.nn.functional.cross_entropy``.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn.functional as F


WeightFn = Callable[[], torch.Tensor]

DEFAULT_CHUNK = 1024
IGNORE_INDEX = -100


class FusedLinearCrossEntropy(torch.autograd.Function):
    """
    loss = cross_entropy(hidden @ weight, labels), computed chunked over the
    token dimension. Returns a scalar. Gradient is provided for ``hidden``
    only; the head weight is treated as a frozen constant.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,        # [N, d]  (already flattened + aligned)
        labels: torch.Tensor,        # [N]
        weight_fn: WeightFn,         # () -> [d, V]  (frozen)
        chunk: int,
        ignore_index: int,
    ) -> torch.Tensor:
        weight = weight_fn()
        n, d = hidden.shape
        # Upcast low-precision (half/bf16) to fp32 for a stable softmax, but
        # preserve fp64 so gradcheck stays exact.
        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)

        valid = labels != ignore_index
        m = int(valid.sum().item())
        denom = max(m, 1)

        loss_sum = hidden.new_zeros((), dtype=compute_dtype)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            h_c = hidden[start:end].to(compute_dtype)
            lbl_c = labels[start:end]
            valid_c = valid[start:end]
            if not bool(valid_c.any()):
                continue
            logits_c = h_c @ weight.to(compute_dtype)           # [c, V]
            logp_c = F.log_softmax(logits_c, dim=-1)
            safe_lbl = lbl_c.clamp_min(0)
            nll = -logp_c.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1)  # [c]
            nll = torch.where(valid_c, nll, torch.zeros_like(nll))
            loss_sum = loss_sum + nll.sum()

        loss = loss_sum / denom

        ctx.save_for_backward(hidden, labels)
        ctx.weight_fn = weight_fn
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        return loss.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, labels = ctx.saved_tensors
        weight = ctx.weight_fn()                 # recomputed, not stored
        chunk = ctx.chunk
        ignore_index = ctx.ignore_index
        denom = ctx.denom
        n, d = hidden.shape

        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)
        w = weight.to(compute_dtype)
        grad_hidden = torch.empty_like(hidden, dtype=compute_dtype)
        g_scale = grad_output.to(compute_dtype)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            h_c = hidden[start:end].to(compute_dtype)
            lbl_c = labels[start:end]
            valid_c = (lbl_c != ignore_index)

            logits_c = h_c @ w                                  # [c, V]
            probs_c = torch.softmax(logits_c, dim=-1)           # [c, V]
            # grad wrt logits = (softmax - onehot) / denom, zeroed on ignored.
            safe_lbl = lbl_c.clamp_min(0)
            probs_c.scatter_add_(
                -1, safe_lbl.unsqueeze(-1),
                torch.full_like(safe_lbl, -1.0, dtype=compute_dtype).unsqueeze(-1),
            )
            probs_c = torch.where(valid_c.unsqueeze(-1), probs_c,
                                  torch.zeros_like(probs_c))
            probs_c = probs_c / denom
            grad_hidden[start:end] = (probs_c @ w.t()) * g_scale

        return grad_hidden.to(hidden.dtype), None, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight_fn: WeightFn,
    labels: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
    ignore_index: int = IGNORE_INDEX,
    shift: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper that handles the causal-LM label shift and flattening.

    :param hidden:    ``[B, T, d]`` hidden states from the backbone.
    :param weight_fn: callable returning the frozen head weight ``[d, V]``
                      (e.g. ``lambda: lm_head.inner.get_weight_tensor()``).
    :param labels:    ``[B, T]`` token ids, ``ignore_index`` where masked.
    :param shift:     if True, predict token ``t+1`` from hidden at ``t``
                      (standard causal shift).
    """
    if shift:
        hidden = hidden[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    d = hidden.shape[-1]
    hidden_flat = hidden.reshape(-1, d)
    labels_flat = labels.reshape(-1)
    return FusedLinearCrossEntropy.apply(
        hidden_flat, labels_flat, weight_fn, chunk, ignore_index
    )
