"""
CPU tests for EBFT (energy-based fine-tuning) reward math
(exllamav3/training/ebft.py):

  * whitening: row-space W = U diag(1/S) U^T semantics, rank-deficiency
    tolerance, the duplicate-gating property (whitened Gram ~ I for distinct
    rollouts; duplicates get 1/n_k cross-sims -- reference-code scaling,
    which is 1/n of the paper's Appendix B.4 eq. 48 values);
  * alignment/diversity/reward composition (2x cosine alignment, 2x mean
    pairwise dot diversity, align_coef/div_coef mixing);
  * the corrected RLOO baseline (paper eq. 94 / reference compute_baseline):
    LOO mean for the alignment part, (sum - 2x)/(n-2) for the diversity part,
    zeroed at n <= 2;
  * degenerate inputs stay finite (all-identical rollouts, zero features);
  * the exact sampler's top-k / top-p filters.

No GPU / compiled extension / real model needed. Run:
    python tests/test_ebft.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


ebft = _load("ebft")


def test_whitening_orthonormalizes_distinct_rollouts():
    torch.manual_seed(0)
    B, n, D = 4, 4, 96
    X = F.normalize(torch.randn(B, n, D), dim=-1)
    y = F.normalize(torch.randn(B, D), dim=-1)
    Xw, Yw = ebft.whiten_features(X, y)
    gram = Xw @ Xw.transpose(-1, -2)
    off = gram - torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-4, "whitened distinct rollouts must be ~orthogonal"
    assert torch.allclose(gram.diagonal(dim1=-2, dim2=-1),
                          torch.ones(B, n), atol=1e-4)
    # gt rows are scalar multiples of raw y (reference semantics)
    cos = F.cosine_similarity(Yw, y.unsqueeze(1).expand_as(Yw), dim=-1).abs()
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)
    print("  whitening (distinct/orthonormal + gt scalar-multiple): OK")


def test_duplicate_gating():
    torch.manual_seed(1)
    B, n, D = 3, 4, 64
    X = F.normalize(torch.randn(B, n, D), dim=-1)
    y = F.normalize(torch.randn(B, D), dim=-1)
    Xd = X.clone()
    Xd[:, 1] = Xd[:, 0]
    rw = ebft.ebft_rewards(Xd, y)
    # duplicate pair (n_k = 2): cross-sim 1/2 => DT = 2 * (1/2) / (n-1) = 1/3
    assert torch.allclose(rw["diversity"][:, :2],
                          torch.full((B, 2), 1.0 / 3.0), atol=1e-4)
    assert rw["diversity"][:, 2:].abs().max() < 1e-4
    # all-identical group (n_k = 4): sims 1/4 => DT = 2 * 3 * (1/4) / 3 = 1/2
    Xs = X[:, :1].expand(B, n, D).contiguous()
    rws = ebft.ebft_rewards(Xs, y)
    assert torch.allclose(rws["diversity"],
                          torch.full((B, n), 0.5), atol=1e-4)
    assert torch.isfinite(rws["reward"]).all()
    assert torch.isfinite(rws["advantage"]).all()
    print("  duplicate gating (pair 1/3, identical 1/2, finite): OK")


def test_rloo_baseline():
    torch.manual_seed(2)
    B, n, D = 5, 4, 48
    X = F.normalize(torch.randn(B, n, D), dim=-1)
    y = F.normalize(torch.randn(B, D), dim=-1)
    ac, dc = 0.7, 0.4
    rw = ebft.ebft_rewards(X, y, align_coef=ac, div_coef=dc)
    align, div = rw["alignment"], rw["diversity"]
    exp_align = (align.sum(1, keepdim=True) - align) / (n - 1)
    exp_div = (div.sum(1, keepdim=True) - 2 * div) / (n - 2)
    exp_b = ac * exp_align - dc * exp_div
    assert torch.allclose(rw["baseline"], exp_b, atol=1e-5)
    assert torch.allclose(rw["advantage"], rw["reward"] - rw["baseline"], atol=1e-6)
    # n = 2: diversity correction must be zeroed, alignment part plain LOO
    rw2 = ebft.ebft_rewards(X[:, :2], y, align_coef=ac, div_coef=dc)
    a2 = rw2["alignment"]
    assert torch.allclose(rw2["baseline"], ac * (a2.sum(1, keepdim=True) - a2),
                          atol=1e-5)
    print("  RLOO baseline (coefs, n=2 gating): OK")


def test_reward_composition_and_no_whiten():
    torch.manual_seed(3)
    B, n, D = 2, 4, 32
    X = F.normalize(torch.randn(B, n, D), dim=-1)
    y = F.normalize(torch.randn(B, D), dim=-1)
    rw = ebft.ebft_rewards(X, y, whiten=False, align_coef=1.0, div_coef=0.5)
    exp_align = 2 * F.cosine_similarity(X, y.unsqueeze(1).expand_as(X), dim=-1)
    gram = X @ X.transpose(-1, -2)
    gram = gram - torch.diag_embed(gram.diagonal(dim1=-2, dim2=-1))
    exp_div = 2 * gram.sum(-1) / (n - 1)
    assert torch.allclose(rw["alignment"], exp_align, atol=1e-5)
    assert torch.allclose(rw["diversity"], exp_div, atol=1e-5)
    assert torch.allclose(rw["reward"], exp_align - 0.5 * exp_div, atol=1e-5)
    # unwhitened CFM diagnostic
    exp_cfm = (X.mean(1) - y).pow(2).sum(-1)
    assert torch.allclose(rw["cfm"], exp_cfm, atol=1e-5)
    print("  reward composition (no-whiten closed forms + cfm): OK")


def test_zero_features_finite():
    B, n, D = 2, 4, 16
    X = torch.zeros(B, n, D)
    y = torch.zeros(B, D)
    rw = ebft.ebft_rewards(X, y)
    for k in ("reward", "baseline", "advantage", "alignment", "diversity"):
        assert torch.isfinite(rw[k]).all(), f"{k} not finite on zero features"
    print("  zero features stay finite: OK")


def test_sampler_filters():
    """top-k / top-p masks in sample_rollouts, tested standalone on a logits
    row (the sampler body inlines these; keep the semantics pinned here)."""
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    # top-k = 2 keeps exactly the two largest
    kth = logits.topk(2, dim=-1).values[:, -1:]
    masked = logits.masked_fill(logits < kth, float("-inf"))
    assert (masked[0, :2] > float("-inf")).all() and (masked[0, 2:] == float("-inf")).all()
    # top-p: exclusive-cumsum mask always keeps the top token
    sp = logits.softmax(dim=-1)
    mask = sp.cumsum(dim=-1) - sp >= 0.1   # tiny p
    assert not mask[0, 0], "top token must always survive top-p"
    assert mask[0, -1]
    print("  sampler top-k/top-p semantics: OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_whitening_orthonormalizes_distinct_rollouts()
    test_duplicate_gating()
    test_rloo_baseline()
    test_reward_composition_and_no_whiten()
    test_zero_features_finite()
    test_sampler_filters()
    print("ALL EBFT TESTS PASSED")
