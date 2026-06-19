"""
Correctness tests for the fused linear cross-entropy head.

Validates that streaming the logits over token chunks gives exactly the same
loss and hidden-state gradient as the naive
``cross_entropy(hidden @ weight, labels)`` -- and that the chunk size does not
change the result. Runs on CPU; needs only torch.

Run:  python tests/test_fused_ce.py
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
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_fce = _load("fused_ce")
FusedLinearCrossEntropy = _fce.FusedLinearCrossEntropy
fused_linear_cross_entropy = _fce.fused_linear_cross_entropy
IGNORE_INDEX = _fce.IGNORE_INDEX


def _naive(hidden, weight, labels, ignore_index=IGNORE_INDEX):
    logits = hidden @ weight
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def test_loss_and_grad_parity():
    torch.manual_seed(0)
    n, d, v = 40, 16, 50
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight

    h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (n,))

    h2 = h1.detach().clone().requires_grad_(True)

    loss_ref = _naive(h1, weight, labels)
    loss_ref.backward()

    loss_fused = FusedLinearCrossEntropy.apply(h2, labels, wf, 7, IGNORE_INDEX)
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), \
        f"loss mismatch: {loss_ref.item()} vs {loss_fused.item()}"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-9), "grad_hidden mismatch"
    print(f"[fce] loss/grad parity PASSED  (loss={loss_fused.item():.6f})")


def test_ignore_index():
    torch.manual_seed(1)
    n, d, v = 30, 12, 40
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))
    labels[::3] = IGNORE_INDEX  # mask out a third of tokens

    h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)

    loss_ref = _naive(h1, weight, labels); loss_ref.backward()
    loss_fused = FusedLinearCrossEntropy.apply(h2, labels, wf, 5, IGNORE_INDEX)
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), "loss mismatch (ignore_index)"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-9), "grad mismatch (ignore_index)"
    # Ignored rows must get zero gradient.
    assert torch.allclose(h2.grad[::3], torch.zeros_like(h2.grad[::3]), atol=1e-12), \
        "ignored tokens received gradient"
    print("[fce] ignore_index PASSED")


def test_chunk_invariance():
    torch.manual_seed(2)
    n, d, v = 64, 16, 60
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))

    results = []
    for chunk in (1, 7, 64, 1000):
        h = torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(9),
                        requires_grad=True)
        loss = FusedLinearCrossEntropy.apply(h, labels, wf, chunk, IGNORE_INDEX)
        loss.backward()
        results.append((loss.detach().clone(), h.grad.clone()))

    l0, g0 = results[0]
    for li, gi in results[1:]:
        assert torch.allclose(l0, li, atol=1e-10), "loss varies with chunk size"
        assert torch.allclose(g0, gi, atol=1e-10), "grad varies with chunk size"
    print("[fce] chunk invariance PASSED")


def test_gradcheck():
    torch.manual_seed(3)
    n, d, v = 12, 8, 20
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))
    h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropy.apply(h_, labels, wf, 5, IGNORE_INDEX),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[fce] gradcheck PASSED")


def test_convenience_shift():
    """fused_linear_cross_entropy with shift must match a manual shifted CE."""
    torch.manual_seed(4)
    b, t, d, v = 2, 9, 16, 30
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    hidden = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (b, t))

    h2 = hidden.detach().clone().requires_grad_(True)
    # Reference: shift then flat CE.
    logits = (h2[:, :-1] @ weight).reshape(-1, v)
    tgt = labels[:, 1:].reshape(-1)
    loss_ref = F.cross_entropy(logits, tgt); loss_ref.backward()

    loss_fused = fused_linear_cross_entropy(hidden, wf, labels, chunk=4)
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), "shifted loss mismatch"
    assert torch.allclose(hidden.grad, h2.grad, atol=1e-9), "shifted grad mismatch"
    print("[fce] convenience shift PASSED")


def test_causal_lm_loss_wiring():
    """
    qlora_causal_lm_loss must reproduce HF-style shifted CausalLM loss, using
    the standard get_decoder()/get_output_embeddings() interface and a plain
    Linear head (orientation check).
    """
    import torch.nn as nn
    _hfq = _load("hf_qlora")

    b, t, d, v = 2, 10, 16, 30
    torch.manual_seed(5)

    class MockDecoderOut:
        def __init__(self, h): self.last_hidden_state = h

    class MockDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d, d)
        def forward(self, input_ids=None, attention_mask=None, **kw):
            # input_ids here are float embeddings for the mock
            return MockDecoderOut(self.proj(input_ids))

    class MockCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = MockDecoder()
            self.lm_head = nn.Linear(d, v, bias=False)
        def get_decoder(self): return self.decoder
        def get_output_embeddings(self): return self.lm_head

    model = MockCausalLM().double()
    emb = torch.randn(b, t, d, dtype=torch.float64)        # stand-in for token embeddings
    labels = torch.randint(0, v, (b, t))

    loss_fused = _hfq.qlora_causal_lm_loss(model, emb, labels, chunk=4)

    # Reference: full forward + HF-style shifted CE.
    hidden = model.decoder(input_ids=emb).last_hidden_state
    logits = model.lm_head(hidden)
    loss_ref = F.cross_entropy(
        logits[:, :-1].reshape(-1, v), labels[:, 1:].reshape(-1))

    assert torch.allclose(loss_ref, loss_fused, atol=1e-9), \
        f"causal LM loss mismatch: {loss_ref.item()} vs {loss_fused.item()}"
    # And gradient must flow back into the decoder params.
    loss_fused.backward()
    assert model.decoder.proj.weight.grad is not None, "no grad into decoder"
    print("[fce] causal LM loss wiring PASSED")


def main():
    test_loss_and_grad_parity()
    test_ignore_index()
    test_chunk_invariance()
    test_gradcheck()
    test_convenience_shift()
    test_causal_lm_loss_wiring()
    print("\nAll fused cross-entropy checks passed.")


if __name__ == "__main__":
    main()
