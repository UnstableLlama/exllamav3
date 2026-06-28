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


WeightSliceFn = Callable[[int, int], torch.Tensor]   # (n_start, n_features) -> [d, n_features]

DEFAULT_VOCAB_CHUNK = 32768


class FusedLinearCrossEntropyVocabChunked(torch.autograd.Function):
    """
    Same loss/grad as :class:`FusedLinearCrossEntropy`, but the head weight is
    reconstructed and matmul'd in **vocab-column chunks** so the full
    ``[d, vocab]`` weight (and its fp32 upcast + the ``[tokens, vocab]`` logits)
    is never held at once -- only ``[d, vocab_chunk]`` and ``[token_chunk,
    vocab_chunk]`` tiles. This bounds the memory spike on the output device for
    big-vocabulary models (e.g. Gemma's 262k head), where the one-shot
    reconstruction is the dominant peak.

    The loop is **vocab-outer, token-inner**: each vocab chunk's weight is
    reconstructed exactly once per forward and once per backward (an online
    softmax carries per-token running max/sum across chunks), so the total
    reconstruction work equals the single-shot path -- no extra dequant cost.
    Head is frozen (gradient for ``hidden`` only). Gradchecked in
    ``tests/test_fused_ce.py`` against ``F.cross_entropy``.
    """

    @staticmethod
    def forward(ctx, hidden, labels, weight_slice_fn, vocab_size, vocab_chunk,
                token_chunk, ignore_index, granularity):
        n, d = hidden.shape
        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)
        # Vocab chunk size must be a multiple of the reconstruction granularity
        # (so an EXL3 slice stays had_n-aligned); round, keep >= granularity.
        vc = max(granularity, (vocab_chunk // granularity) * granularity)

        valid = labels != ignore_index
        denom = max(int(valid.sum().item()), 1)

        neg_inf = torch.finfo(compute_dtype).min
        run_max = hidden.new_full((n,), neg_inf, dtype=compute_dtype)   # running max [N]
        run_sum = hidden.new_zeros((n,), dtype=compute_dtype)           # running sum-exp [N]
        tgt_logit = hidden.new_zeros((n,), dtype=compute_dtype)         # target logit [N]
        h_all = hidden.to(compute_dtype)

        for v0 in range(0, vocab_size, vc):
            v1 = min(v0 + vc, vocab_size)
            w_v = weight_slice_fn(v0, v1 - v0).to(compute_dtype)        # [d, vw]
            for t0 in range(0, n, token_chunk):
                t1 = min(t0 + token_chunk, n)
                logits = h_all[t0:t1] @ w_v                            # [tc, vw]
                # Online-softmax running stats over the vocab dimension.
                chunk_max = logits.max(dim=-1).values                  # [tc]
                new_max = torch.maximum(run_max[t0:t1], chunk_max)
                run_sum[t0:t1] = (run_sum[t0:t1] * torch.exp(run_max[t0:t1] - new_max)
                                  + torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1))
                run_max[t0:t1] = new_max
                # Capture the target logit for tokens whose label is in this chunk.
                lbl = labels[t0:t1]
                in_chunk = (lbl >= v0) & (lbl < v1)
                local = (lbl - v0).clamp_(0, v1 - v0 - 1)
                got = logits.gather(-1, local.unsqueeze(-1)).squeeze(-1)
                tgt_logit[t0:t1] = torch.where(in_chunk, got, tgt_logit[t0:t1])

        lse = run_max + torch.log(run_sum)                              # [N]
        nll = torch.where(valid, lse - tgt_logit, torch.zeros_like(lse))
        loss = nll.sum() / denom

        ctx.save_for_backward(hidden, labels, lse)
        ctx.weight_slice_fn = weight_slice_fn
        ctx.vocab_size = vocab_size
        ctx.vc = vc
        ctx.token_chunk = token_chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        return loss.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, labels, lse = ctx.saved_tensors
        weight_slice_fn = ctx.weight_slice_fn
        vocab_size, vc = ctx.vocab_size, ctx.vc
        token_chunk = ctx.token_chunk
        ignore_index = ctx.ignore_index
        denom = ctx.denom
        n, d = hidden.shape

        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)
        h_all = hidden.to(compute_dtype)
        g_scale = grad_output.to(compute_dtype)
        grad_hidden = hidden.new_zeros((n, d), dtype=compute_dtype)
        valid = (labels != ignore_index)

        # Same vocab-outer order: reconstruct each weight chunk once, accumulate
        # grad_hidden += (softmax - onehot)/denom @ W_v^T over chunks. softmax for
        # this chunk = exp(logits - lse) (lse is the full-vocab normalizer saved
        # in forward), so no second normalization pass is needed.
        for v0 in range(0, vocab_size, vc):
            v1 = min(v0 + vc, vocab_size)
            w_v = weight_slice_fn(v0, v1 - v0).to(compute_dtype)        # [d, vw]
            for t0 in range(0, n, token_chunk):
                t1 = min(t0 + token_chunk, n)
                logits = h_all[t0:t1] @ w_v                            # [tc, vw]
                p = torch.exp(logits - lse[t0:t1].unsqueeze(-1))       # softmax slice
                lbl = labels[t0:t1]
                in_chunk = (lbl >= v0) & (lbl < v1)
                local = (lbl - v0).clamp_(0, v1 - v0 - 1)
                # Subtract the one-hot target inside this chunk.
                onehot = torch.zeros_like(p)
                onehot.scatter_(-1, local.unsqueeze(-1),
                                in_chunk.to(p.dtype).unsqueeze(-1))
                p = p - onehot
                p = torch.where(valid[t0:t1].unsqueeze(-1), p, torch.zeros_like(p))
                grad_hidden[t0:t1] += (p @ w_v.t()) / denom

        grad_hidden *= g_scale
        return grad_hidden.to(hidden.dtype), None, None, None, None, None, None, None


def fused_linear_cross_entropy_vocab_chunked(
    hidden: torch.Tensor,
    weight_slice_fn: WeightSliceFn,
    labels: torch.Tensor,
    vocab_size: int,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
    token_chunk: int = DEFAULT_CHUNK,
    ignore_index: int = IGNORE_INDEX,
    granularity: int = 1,
    shift: bool = True,
) -> torch.Tensor:
    """Vocab-chunked variant of :func:`fused_linear_cross_entropy`.

    :param weight_slice_fn: ``(n_start, n_features) -> [d, n_features]`` returning a
                            column slice of the frozen head weight.
    :param vocab_size:      total vocabulary (output columns).
    :param vocab_chunk:     output-column tile size (rounded down to ``granularity``).
    :param granularity:     required alignment of a slice (e.g. EXL3 ``had_n``=128).
    """
    if shift:
        hidden = hidden[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    d = hidden.shape[-1]
    hidden_flat = hidden.reshape(-1, d)
    labels_flat = labels.reshape(-1)
    return FusedLinearCrossEntropyVocabChunked.apply(
        hidden_flat, labels_flat, weight_slice_fn, vocab_size, vocab_chunk,
        token_chunk, ignore_index, granularity,
    )


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
