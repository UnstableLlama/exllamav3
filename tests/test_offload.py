"""
Correctness + bounded-pool tests for the async activation offloader
(``exllamav3/training/offload.py``, S36; pool cap added S47).

The offloader wraps a grad-checkpointed block loop, DtoH-copying every saved
activation to a pinned host buffer on a side stream and HtoD-copying it back in
backward. Two properties must hold:

  1. VALUE-EXACT. Copies don't round: a same-seed forward/backward run under the
     offloader must produce bit-identical gradients to the same run with no
     offloader. (The module's own verification gate.)

  2. BOUNDED POOL (S47). The free-buffer pool must NOT grow without bound when
     the per-step activation shape varies (EBFT random anchors / unpacked SFT
     variable seq length) — the leak that OOM'd a 1B EBFT run and a 128B SFT
     run. And it must NOT thrash (evict + re-alloc every step) on a large but
     shape-STABLE run — the working-set floor protects that case.

All tiers need CUDA (pinned buffers + side streams + events). SKIP without it.

Run directly::

    python tests/test_offload.py
"""

from __future__ import annotations
import os
import sys
import importlib.util
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Load offload.py directly (torch-only; avoids importing the exllamav3 package
# and its CUDA-extension build for a test that only needs torch + a GPU).
_spec = importlib.util.spec_from_file_location(
    "exllamav3_offload",
    os.path.join(_ROOT, "exllamav3", "training", "offload.py"),
)
offload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(offload)
AsyncActivationOffload = offload.AsyncActivationOffload

try:
    from util import run_timed
except ImportError:  # pytest collection from repo root
    from tests.util import run_timed

_HAS_CUDA = torch.cuda.is_available()


def _skip_if_no_cuda():
    if not _HAS_CUDA:
        print("  (SKIP: no CUDA — the offloader needs pinned buffers + streams)")
        return True
    return False


def _run_blocks(x, w, n_blocks, offloader=None):
    """A small checkpoint-free block loop whose intermediates are big enough to
    offload. matmul saves both operands; tanh saves its output — several
    >min_bytes tensors per pass. Returns (loss, x.grad)."""
    ctx = offloader if offloader is not None else _nullctx()
    with ctx:
        h = x
        for _ in range(n_blocks):
            h = torch.tanh(h @ w)
        loss = (h * h).sum()
    loss.backward()
    return float(loss.detach()), x.grad.detach().clone()


class _nullctx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_value_exact():
    """Gradients under the offloader are bit-identical to no offload."""
    if _skip_if_no_cuda():
        return
    dev = "cuda"
    shape = (2, 2048, 512)          # 2*2048*512*4 = 8 MB fp32 > 1 MB min_bytes
    torch.manual_seed(0)
    w = torch.randn(512, 512, device=dev, dtype=torch.float32) * 0.05

    def fresh_x():
        torch.manual_seed(1234)
        return torch.randn(*shape, device=dev, dtype=torch.float32,
                           requires_grad=True)

    x0 = fresh_x()
    loss0, g0 = _run_blocks(x0, w, n_blocks=6, offloader=None)

    off = AsyncActivationOffload()
    x1 = fresh_x()
    loss1, g1 = _run_blocks(x1, w, n_blocks=6, offloader=off)

    assert loss0 == loss1, f"loss differs: {loss0} vs {loss1}"
    assert torch.equal(g0, g1), \
        f"grad differs under offload: max|Δ|={(g0 - g1).abs().max().item()}"
    print("    value-exact: losses + grads bit-identical with/without offload")


def test_bounded_under_varying_shapes():
    """The leak fix: with a different activation shape every 'step', the pool
    must stay bounded (not accumulate one bucket per shape forever)."""
    if _skip_if_no_cuda():
        return
    dev = "cuda"
    off = AsyncActivationOffload(max_pool_bytes=1 << 20)  # tiny fixed floor; the
    #                                    working-set floor (1.5x a pass) governs.
    w = torch.randn(512, 512, device=dev, dtype=torch.float32) * 0.05

    peak_pool = 0
    for step in range(40):
        # Vary the sequence length every step, exactly like EBFT rollouts /
        # unpacked SFT: a brand-new (shape, dtype) pool key each time.
        seqlen = 1024 + (step * 137) % 2048
        torch.manual_seed(step)
        x = torch.randn(2, seqlen, 512, device=dev, dtype=torch.float32,
                        requires_grad=True)
        _run_blocks(x, w, n_blocks=6, offloader=off)
        torch.cuda.synchronize()
        peak_pool = max(peak_pool, off._pool_bytes)

    # A single pass offloads ~n_blocks activations. The bound is the effective
    # cap (max(floor, 1.5x working set)) plus at most one bucket in flight.
    cap = max(off.max_pool_bytes, int(off._working_set * 3 // 2))
    largest_bucket = off._working_set  # generous slack: one whole pass
    assert peak_pool <= cap + largest_bucket, (
        f"pool grew unbounded: peak {peak_pool/1e6:.1f} MB > "
        f"cap {cap/1e6:.1f} + slack {largest_bucket/1e6:.1f} MB "
        f"(working_set {off._working_set/1e6:.1f} MB)")
    # And the leak is genuinely evicted — the pool is nowhere near 40 passes.
    assert peak_pool < 40 * off._working_set, "eviction never fired"
    print(f"    varying shapes: pool bounded at {peak_pool/1e6:.1f} MB over 40 "
          f"steps (working set {off._working_set/1e6:.1f} MB) — leak evicted")


def test_no_thrash_when_shape_stable():
    """The no-regression guard: a repeated fixed shape reaches steady state at
    ~one pass worth and stays there — buffers are reused, none evicted."""
    if _skip_if_no_cuda():
        return
    dev = "cuda"
    off = AsyncActivationOffload(max_pool_bytes=1 << 20)  # small; floor protects
    w = torch.randn(512, 512, device=dev, dtype=torch.float32) * 0.05

    samples = []
    for step in range(10):
        torch.manual_seed(step)
        x = torch.randn(2, 1536, 512, device=dev, dtype=torch.float32,
                        requires_grad=True)
        _run_blocks(x, w, n_blocks=6, offloader=off)
        torch.cuda.synchronize()
        samples.append(off._pool_bytes)

    steady = samples[2:]  # allow warmup passes to populate the pool
    assert max(steady) == min(steady), \
        f"pool not steady on a fixed shape (thrash?): {samples}"
    # Steady pool == exactly one pass's worth of reused buffers.
    assert steady[0] == off._working_set, (
        f"steady pool {steady[0]} != one working set {off._working_set} — "
        f"buffers not being reused")
    print(f"    stable shape: pool steady at {steady[0]/1e6:.1f} MB == one pass "
          f"({off._working_set/1e6:.1f} MB), no thrash")


def main():
    run_timed(
        [test_value_exact,
         test_bounded_under_varying_shapes,
         test_no_thrash_when_shape_stable],
        label="test_offload",
    )
    print("ALL PASS")


if __name__ == "__main__":
    main()
