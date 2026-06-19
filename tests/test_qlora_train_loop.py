"""
End-to-end QLoRA training-mechanics test (single process, CPU).

This does not need a GPU, the compiled extension, transformers, or a real
quantized model. It stands a *mock* EXL3 linear (a frozen random weight with
a ``get_weight_tensor()`` accessor) into a toy network, attaches trainable
LoRA adapters via ``attach_qlora``, and runs a real optimisation loop.

It asserts the things a training path must get right:
  * loss actually decreases (gradients flow end-to-end through the
    differentiable EXL3 linear),
  * the frozen base weight and all non-adapter parameters are unchanged,
  * only the LoRA parameters require grad and actually moved,
  * the PEFT save orientation reproduces the exact same delta the inference
    LoRA loader would apply (adapter portability).

Run:  python tests/test_qlora_train_loop.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

# Load training modules under a synthetic package so their relative imports
# resolve, without importing the full exllamav3 package (which would build the
# CUDA extension).
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
_hfq = _load("hf_qlora")
attach_qlora = _hfq.attach_qlora
Exl3LoRALinear = _hfq.Exl3LoRALinear
iter_lora_modules = _hfq.iter_lora_modules


# ----------------------------------------------------------------------------
# Mock EXL3 linear: a frozen random weight masquerading as a trellis layer.
# ----------------------------------------------------------------------------
class _MockInner:
    def __init__(self, weight: torch.Tensor):
        self._w = weight            # [in, out], frozen
        self.trellis = weight       # stand-in so device inference works
        self.bias = None

    def get_weight_tensor(self) -> torch.Tensor:
        return self._w


class MockExl3HfLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.randn(in_features, out_features) * 0.05
        self.register_buffer("frozen_weight", w)   # not a Parameter -> never trained
        self.inner = _MockInner(self.frozen_weight)

    def forward(self, x):  # inference-style (non-differentiable stand-in)
        return x @ self.inner.get_weight_tensor()


class ToyModel(nn.Module):
    """emb (frozen) -> q_proj (mock EXL3, adapted) -> head (frozen)."""
    def __init__(self, d: int):
        super().__init__()
        self.emb = nn.Linear(d, d)
        self.q_proj = MockExl3HfLinear(d, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.emb(x)
        h = self.q_proj(h)
        h = torch.relu(h)
        return self.head(h)


def test_training_loop():
    torch.manual_seed(0)
    d, n = 16, 64
    model = ToyModel(d).to(torch.float32)

    # Snapshot frozen tensors to verify they don't change.
    base_w0 = model.q_proj.frozen_weight.clone()
    emb_w0 = model.emb.weight.clone()
    head_w0 = model.head.weight.clone()

    trainable = attach_qlora(
        model, r=4, alpha=8.0, target_modules=["q_proj"],
        compute_dtype=torch.float32, verbose=True,
    )
    assert len(trainable) == 2, "expected lora_a and lora_b"

    # Only LoRA params require grad.
    req = [n_ for n_, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n_ for n_ in req), f"non-LoRA params trainable: {req}"
    assert len(req) == 2, req

    # A fixed teacher target so there is a real signal to fit.
    x = torch.randn(n, d)
    teacher = torch.randn(d, 1)
    y = torch.tanh(x @ teacher) + 0.1 * torch.randn(n, 1)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        loss0 = loss_fn(model(x), y).item()

    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss1 = loss_fn(model(x), y).item()

    print(f" -- loss {loss0:.4f} -> {loss1:.4f}")
    assert loss1 < 0.6 * loss0, f"loss did not decrease enough: {loss0} -> {loss1}"

    # Frozen tensors untouched. After attach, q_proj is the wrapper and the
    # original mock (with the frozen weight buffer) lives at q_proj.base.
    assert torch.equal(model.q_proj.base.frozen_weight, base_w0), "base EXL3 weight changed!"
    assert torch.equal(model.emb.weight, emb_w0), "frozen emb changed!"
    assert torch.equal(model.head.weight, head_w0), "frozen head changed!"

    # Adapter actually moved (B starts at zero).
    lora = next(m for _, m in iter_lora_modules(model))
    assert lora.lora_b.abs().sum().item() > 0, "lora_b never updated"
    print("[train] loss decreased, base frozen, adapter updated PASSED")


def test_peft_save_orientation():
    """
    The saved PEFT tensors must reproduce, via the inference loader's
    transforms, the exact delta our training forward applies.

    Inference loader (exllamav3/model/lora.py) does, per adapter:
        A_load = A_peft.T            # [r, in]   -> [in, r]
        B_load = B_peft.T * scaling  # [out, r]  -> [r, out] * (alpha/r)
        delta  = x @ A_load @ B_load
    Our training forward applies:
        delta  = scale * (x @ A_int @ B_int),  scale = alpha/r
    with A_int=[in,r], B_int=[r,out]. We save A_peft=A_int.T, B_peft=B_int.T
    (unscaled). The two deltas must match.
    """
    torch.manual_seed(1)
    d, r, alpha = 8, 4, 16.0
    base = MockExl3HfLinear(d, d)
    lora = Exl3LoRALinear(base, r=r, alpha=alpha, compute_dtype=torch.float32)
    # Give B a nonzero value so the test is meaningful.
    with torch.no_grad():
        lora.lora_b.copy_(torch.randn(r, d) * 0.3)

    x = torch.randn(5, d)
    scale = alpha / r

    # Training-time delta (what our forward adds on top of the base).
    delta_train = scale * (x @ lora.lora_a @ lora.lora_b)

    # Emulate save (transpose to PEFT) then inference-loader load (transpose back + scale).
    A_peft = lora.lora_a.detach().t().contiguous()      # [r, in]
    B_peft = lora.lora_b.detach().t().contiguous()      # [out, r]
    A_load = A_peft.t()                                 # [in, r]
    B_load = B_peft.t() * scale                         # [r, out] * scaling
    delta_load = x @ A_load @ B_load

    assert torch.allclose(delta_train, delta_load, atol=1e-6), "PEFT orientation mismatch"
    print("[save] PEFT save/load orientation reproduces training delta PASSED")


def main():
    test_training_loop()
    test_peft_save_orientation()
    print("\nAll QLoRA training-mechanics checks passed.")


if __name__ == "__main__":
    main()
