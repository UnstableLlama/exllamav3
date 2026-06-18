"""
Differentiable LoRA-over-frozen-EXL3 linear (QLoRA proof of concept).

Background
----------
QLoRA freezes the (quantized) base weights and trains only the low-rank
adapter matrices A and B. Crucially, *no gradient ever needs to flow
through the quantizer*: the frozen weight behaves as a constant in the
computation graph, exactly like NF4 in bitsandbytes-based QLoRA. All we
need for a correct backward pass is:

  - ``grad_x``  -- to propagate loss to earlier layers, and
  - ``grad_A`` / ``grad_B`` -- to update the adapter.

both of which are ordinary matmuls once the effective FP16 weight has been
reconstructed from the EXL3 trellis. EXL3 already exposes that
reconstruction via ``LinearEXL3.get_weight_tensor()`` (the full effective
weight, with the sign flips ``suh``/``svh`` and Hadamard rotations folded
in), so the "scary" part -- differentiating a 3-bit trellis code -- never
enters the picture for QLoRA.

This module provides two things:

  1. ``reference_forward`` -- the dead-simple version that just runs plain
     torch ops with the dequantized weight detached to a constant, and
     lets autograd do everything. This is the ground truth.

  2. ``EXL3LoRAFunction`` -- a memory-efficient ``autograd.Function`` that
     does *not* stash the dequantized weight for the backward pass but
     instead re-derives it from the (cheap, already-on-device) trellis via
     a ``weight_fn`` closure. This is the shape of the kernel a real
     training path would use: it keeps the big FP16 weight out of the saved
     activation set. Its hand-written backward is validated against (1) and
     against ``torch.autograd.gradcheck`` in ``tests/test_qlora_grad.py``.

Nothing here touches the inference path. It is deliberately standalone so
the "is this even real?" question can be answered by a single gradcheck.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn as nn


WeightFn = Callable[[], torch.Tensor]


def reference_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    scale: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Ground-truth forward using only plain torch ops.

    ``weight`` is the dequantized effective weight, shape ``[in, out]``,
    such that the base projection is ``y = x @ weight``. It is treated as a
    constant (its ``requires_grad`` is irrelevant -- we never want a grad
    for the frozen base). ``lora_a`` is ``[in, r]`` and ``lora_b`` is
    ``[r, out]``, matching the orientation ExLlamaV3's inference LoRA loader
    already uses (``x @ A @ B``).

    Autograd handles the entire backward for free; this exists so the
    custom Function below can be checked against it.
    """
    w_const = weight.detach()
    y = x @ w_const
    if lora_a is not None and lora_b is not None:
        y = y + scale * ((x @ lora_a) @ lora_b)
    if bias is not None:
        y = y + bias
    return y


class EXL3LoRAFunction(torch.autograd.Function):
    """
    Memory-efficient differentiable LoRA-over-frozen-weight linear.

    The forward reconstructs the dequantized weight via ``weight_fn`` (e.g.
    ``lambda: linear.inner.get_weight_tensor()``), uses it, and then throws
    it away. The backward re-reconstructs it rather than carrying the full
    FP16 weight through the saved-tensors set, mirroring how a real EXL3
    training kernel would trade a little recompute for a lot of activation
    memory.

    Tensor shapes
    -------------
      x      : ``[*, in]``
      weight : ``[in, out]``     (frozen, returned by ``weight_fn``)
      A      : ``[in, r]``       (trainable)
      B      : ``[r, out]``      (trainable)
      bias   : ``[out]``         (optional, trainable)
      y      : ``[*, out]``
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lora_a: Optional[torch.Tensor],
        lora_b: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale: float,
        weight_fn: WeightFn,
    ) -> torch.Tensor:
        weight = weight_fn()  # [in, out], constant w.r.t. autograd
        assert weight.dim() == 2, "weight_fn must return a 2D [in, out] tensor"

        y = x @ weight
        if lora_a is not None and lora_b is not None:
            y = y + scale * ((x @ lora_a) @ lora_b)
        if bias is not None:
            y = y + bias

        # Deliberately do NOT save `weight`; recompute it in backward.
        ctx.save_for_backward(x, lora_a, lora_b)
        ctx.scale = scale
        ctx.weight_fn = weight_fn
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, lora_a, lora_b = ctx.saved_tensors
        scale = ctx.scale
        weight = ctx.weight_fn()  # recomputed, not stored

        in_features = x.shape[-1]
        out_features = grad_y.shape[-1]
        xf = x.reshape(-1, in_features)
        gf = grad_y.reshape(-1, out_features)

        # grad wrt input: base path always present.
        grad_x = grad_y @ weight.transpose(-1, -2)

        grad_a = grad_b = grad_bias = None

        if lora_a is not None and lora_b is not None:
            # y_lora = scale * (x @ A) @ B
            # let P = x @ A  -> [N, r]
            #     g_B = grad_y @ B^T -> [N, r]
            g_through_b = gf @ lora_b.transpose(-1, -2)          # [N, r]
            # contribution of LoRA branch to grad_x
            grad_x = grad_x + scale * (g_through_b @ lora_a.transpose(-1, -2)).view_as(grad_x)
            if ctx.needs_input_grad[1]:
                grad_a = scale * (xf.transpose(-1, -2) @ g_through_b)   # [in, r]
            if ctx.needs_input_grad[2]:
                p = xf @ lora_a                                          # [N, r]
                grad_b = scale * (p.transpose(-1, -2) @ gf)             # [r, out]

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = gf.sum(dim=0)

        # Frozen weight (None), scale (None), weight_fn (None) get no grad.
        if not ctx.needs_input_grad[0]:
            grad_x = None

        return grad_x, grad_a, grad_b, grad_bias, None, None


def qlora_linear_forward(
    x: torch.Tensor,
    weight_fn: WeightFn,
    lora_a: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional entry point using the memory-efficient Function."""
    return EXL3LoRAFunction.apply(x, lora_a, lora_b, bias, scale, weight_fn)


class QLoRALinear(nn.Module):
    """
    nn.Module wrapper turning a frozen ExLlamaV3 ``Linear`` into a trainable
    QLoRA layer.

    The base weight is never copied or updated; on every forward it is
    reconstructed on the fly from the EXL3 trellis (or read directly, for an
    fp16 base) through ``weight_fn``. Only ``lora_a`` / ``lora_b`` are
    registered parameters and receive gradients.

    Parameters
    ----------
    weight_fn:
        Callable returning the frozen ``[in, out]`` effective weight. For a
        loaded EXL3 layer this is typically
        ``lambda: linear.inner.get_weight_tensor()``. The returned tensor's
        dtype/device define the compute dtype/device of the base projection.
    in_features, out_features:
        Padded feature dims of the wrapped layer.
    r, alpha:
        LoRA rank and scaling (``scale = alpha / r``, optionally rslora).
    """

    def __init__(
        self,
        weight_fn: WeightFn,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.weight_fn = weight_fn
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        denom = (r ** 0.5) if use_rslora else r
        self.scale = float(alpha) / float(denom)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # PEFT-style init: A ~ kaiming/normal, B = 0 so the adapter starts
        # as a no-op and training begins from the exact base model.
        self.lora_a = nn.Parameter(torch.empty(in_features, r, dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros(r, out_features, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

        # Bias from the base layer is frozen (not trained here).
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_in = self.dropout(x) if self.dropout is not None else x
        # Note: dropout differs between the two LoRA branches only if applied
        # before the base; here the base path uses the raw x and the adapter
        # uses dropped x. To keep the PoC's gradcheck exact we route both
        # through the same Function, so apply dropout to the adapter inputs
        # by folding it into A is not possible -- keep dropout off for grad
        # checking (default dropout=0.0).
        return EXL3LoRAFunction.apply(
            lora_in, self.lora_a, self.lora_b, self.bias, self.scale, self.weight_fn
        )

    @classmethod
    def from_exl3_linear(
        cls,
        linear,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> "QLoRALinear":
        """
        Build a trainable wrapper around a loaded ExLlamaV3 ``Linear``.

        Works for both EXL3 and fp16 inner layers, since both expose
        ``get_weight_tensor()`` returning the ``[in, out]`` effective weight.
        """
        inner = linear.inner
        device = linear.device

        def weight_fn() -> torch.Tensor:
            w = inner.get_weight_tensor()      # [in, out], half
            return w.to(dtype)

        bias = None
        get_bias = getattr(inner, "get_bias_tensor", None)
        if get_bias is not None:
            b = get_bias()
            if b is not None:
                bias = b.to(dtype)

        return cls(
            weight_fn=weight_fn,
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            use_rslora=use_rslora,
            bias=bias,
            dtype=dtype,
            device=device,
        )
