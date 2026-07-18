"""
Energy-Based Fine-Tuning (EBFT) reward math.

Feature-matching rewards for on-policy fine-tuning, after "Matching Features,
Not Tokens: Energy-Based Fine-Tuning of Language Models" (Jelassi et al.,
arXiv:2603.12248). Semantics follow the authors' reference implementation
(sjelassi/ebft_openrlhf: openrlhf/utils/embedding_utils.py and
ebft_experience_maker.py / compute_baseline) rather than the paper's Appendix
B.4 closed forms, in the one place they differ -- see whiten_features below.

Per (context, ground-truth window) group with n sampled rollouts:

  * features: concat of the residual stream at ~25/50/75% depth blocks, taken
    at the LAST token of the window, jointly L2-normalized (the reference's
    ``hidden_state_method=concat`` + ``embed_method=last_token``);
  * whitening: X in [n, D] row-whitened via W = U diag(1/S) U^T from the thin
    SVD of X (pseudo-inverse tolerance 1e-5 relative to S_max), applied to
    both the rollout features and the (replicated) ground-truth feature;
  * alignment_j  = 2 * cosine(x_w_j, y_w_j)                       (want high)
  * diversity_j  = 2 * mean_{j' != j} <x_w_j, x_w_j'>             (want low)
  * reward_j     = align_coef * alignment_j - div_coef * diversity_j
  * RLOO baseline (paper Appendix E eq. 94, reference compute_baseline):
      b_align_j = (sum_{j'} align_{j'} - align_j) / (n - 1)
      b_div_j   = (sum_{j'} div_{j'} - 2 div_j) / (n - 2)   [0 when n <= 2]
      b_j       = align_coef * b_align_j - div_coef * b_div_j

Under whitening the Gram matrix of distinct rollouts is ~identity, so the
diversity term fires (only) on duplicate/near-duplicate rollouts -- it is a
degeneracy penalty, not a spread bonus. The reference's default coefficients
are align_coef=1.0, div_coef=0.5 (the paper's alpha=0.5 alignment bias).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def whiten_features(gen: torch.Tensor, gt: torch.Tensor,
                    tol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-space whitening of rollout features, reference semantics.

    ``gen``: [B, n, D] rollout features per group; ``gt``: [B, D] ground-truth
    features. Returns ``(gen_w [B, n, D], gt_w [B, n, D])`` where
    ``gen_w = W @ gen`` with ``W = U diag(1/S) U^T`` from the thin SVD of each
    group's [n, D] matrix (singular values below ``tol * S_max`` are zeroed --
    rank-deficient groups become projections, which is what makes duplicate
    rollouts well-defined).

    NOTE the ground truth is whitened by the same [n, n] row operator applied
    to n replicated copies -- each output row is a SCALAR multiple of the raw
    ground-truth vector, NOT the paper's D-space ``(Sigma^+)^(1/2) y``. Under
    the cosine alignment this reduces to comparing whitened rollouts against
    the raw ground-truth direction (possible sign flips included). This
    matches the reference implementation exactly and is intentional."""
    B, n, D = gen.shape
    X = gen.float()
    Y = gt.float().unsqueeze(1).expand(B, n, D)
    try:
        U, S, _ = torch.linalg.svd(X, full_matrices=False)   # U [B,n,n], S [B,n]
    except torch.linalg.LinAlgError:
        # Reference fallback 1: tiny noise to break exact degeneracy.
        noise = 1e-6 * X.abs().mean()
        try:
            U, S, _ = torch.linalg.svd(X + noise * torch.randn_like(X),
                                       full_matrices=False)
        except torch.linalg.LinAlgError:
            # Reference fallback 2: give up on whitening for this batch.
            return X, Y
    smax = S.max(dim=-1, keepdim=True).values
    inv_s = torch.where(S > tol * smax, 1.0 / (S + 1e-12), torch.zeros_like(S))
    W = (U * inv_s.unsqueeze(-2)) @ U.transpose(-1, -2)      # [B, n, n]
    return W @ X, W @ Y


def ebft_rewards(gen_feats: torch.Tensor, gt_feats: torch.Tensor, *,
                 whiten: bool = True, align_coef: float = 1.0,
                 div_coef: float = 0.5, whiten_tol: float = 1e-5,
                 ) -> dict[str, torch.Tensor]:
    """Feature-matching rewards + RLOO baseline for one batch of groups.

    ``gen_feats``: [B, n, D] L2-normalized rollout features; ``gt_feats``:
    [B, D] L2-normalized ground-truth features. All math in fp32, no grad.

    Returns a dict of [B, n] tensors: ``reward``, ``baseline``, ``advantage``
    (= reward - baseline), ``alignment``, ``diversity``, plus scalar
    diagnostics ``cfm`` ([B], unwhitened conditional feature-matching loss
    ||mean_j phi_j - phi(y)||^2, the paper's calibration metric) and
    ``fm_whiten`` ([B], the eq. 53 proxy mean_j(align_j - div_j / 2), higher
    is better -- the paper's figures plot its negation)."""
    assert gen_feats.dim() == 3 and gt_feats.dim() == 2, \
        f"expected [B,n,D] and [B,D], got {tuple(gen_feats.shape)} / {tuple(gt_feats.shape)}"
    B, n, D = gen_feats.shape
    assert n >= 2, "EBFT needs n >= 2 samples per group (RLOO baseline)"
    with torch.no_grad():
        X = gen_feats.float()
        y = gt_feats.float()

        # Unwhitened CFM diagnostic on the raw (normalized) features.
        cfm = (X.mean(dim=1) - y).pow(2).sum(dim=-1)                  # [B]

        if whiten:
            Xw, Yw = whiten_features(X, y, tol=whiten_tol)
        else:
            Xw, Yw = X, y.unsqueeze(1).expand(B, n, D)

        # alignment_j = 2 cos(x_j, y_j); diversity_j = 2 mean_{j'!=j} <x_j, x_j'>
        alignment = 2.0 * F.cosine_similarity(Xw, Yw, dim=-1)          # [B, n]
        gram = Xw @ Xw.transpose(-1, -2)                               # [B, n, n]
        gram = gram - torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
        diversity = 2.0 * gram.sum(dim=-1) / (n - 1)                   # [B, n]

        reward = align_coef * alignment - div_coef * diversity

        # RLOO baseline; the diversity term depends on the other samples, so
        # its leave-one-out correction differs from the naive mean (eq. 94).
        b_align = (alignment.sum(dim=1, keepdim=True) - alignment) / (n - 1)
        if n > 2:
            b_div = (diversity.sum(dim=1, keepdim=True) - 2.0 * diversity) / (n - 2)
        else:
            b_div = torch.zeros_like(diversity)
        baseline = align_coef * b_align - div_coef * b_div

        fm_whiten = (alignment - diversity / 2.0).mean(dim=1)          # [B]

    return {
        "reward": reward, "baseline": baseline,
        "advantage": reward - baseline,
        "alignment": alignment, "diversity": diversity,
        "cfm": cfm, "fm_whiten": fm_whiten,
    }


@torch.no_grad()
def sample_rollouts(net, input_ids: torch.Tensor, row_lens: torch.Tensor,
                    gen_len: int, *, temperature: float = 0.6,
                    top_k: int = 0, top_p: float = 1.0,
                    generator: Optional[torch.Generator] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact on-policy sampler over the differentiable forward (no KV cache).

    ``input_ids``: [R, L] right-padded context rows; ``row_lens``: [R] context
    lengths. Appends ``gen_len`` sampled tokens per row with ``gen_len``
    full forwards -- wasteful vs a cached generator but EXACTLY the policy the
    REINFORCE gradient is computed under (the differentiable path), so v1
    training has zero sampling/scoring mismatch. Returns ``(rows [R, L'],
    lens [R])`` with the samples appended at each row's cursor.

    Logits are taken per-row at the cursor position from the pre-head hidden
    state (never materializing [R, T, vocab]); the LM head closure applies any
    final softcap, matching net.logits."""
    assert temperature > 0.0, "greedy rollouts collapse the RLOO baseline; use temperature > 0"
    R, L = input_ids.shape
    device = input_ids.device
    rows = torch.cat([input_ids,
                      input_ids.new_zeros(R, gen_len)], dim=1)
    lens = row_lens.clone()
    head_w = net.lm_head_weight_fn()
    was_training = net.training
    net.eval()
    try:
        for _ in range(gen_len):
            t = int(lens.max())
            attn = (torch.arange(t, device=device).unsqueeze(0)
                    < lens.unsqueeze(1)).long()
            hidden = net.forward(rows[:, :t], attention_mask=attn)      # [R, t, d]
            idx = (lens - 1).to(hidden.device)
            last = hidden[torch.arange(R, device=hidden.device), idx]   # [R, d]
            w = head_w()
            logits = last.to(w.device).to(w.dtype) @ w                  # [R, V]
            if net.final_softcap:
                logits = net.final_softcap * torch.tanh(logits / net.final_softcap)
            logits = logits.float() / temperature
            if top_k and top_k > 0:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sp = sorted_logits.softmax(dim=-1)
                # mask tokens whose EXCLUSIVE cumulative prob passes top_p
                # (the top token always survives)
                mask = sp.cumsum(dim=-1) - sp >= top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits)
            probs = logits.softmax(dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            rows[torch.arange(R, device=device), lens] = nxt.to(device)
            lens = lens + 1
    finally:
        if was_training:
            net.train()
    return rows[:, :int(lens.max())], lens
