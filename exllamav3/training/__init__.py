"""
Experimental training support for ExLlamaV3.

This subpackage is a *proof of concept* exploring whether QLoRA-style
fine-tuning is feasible on top of EXL3-quantized weights. It currently
contains a single, self-contained building block: a differentiable
LoRA-over-frozen-EXL3 linear layer.

Nothing here is wired into the main (inference) forward path, which runs
under ``torch.inference_mode`` and cannot participate in autograd. See
``doc/qlora_feasibility.md`` for the design rationale and roadmap.
"""

from .qlora_linear import (
    EXL3LoRAFunction,
    qlora_linear_forward,
    QLoRALinear,
    reference_forward,
)
from .hf_qlora import (
    Exl3LoRALinear,
    attach_qlora,
    iter_lora_modules,
    save_lora_adapter,
    load_lora_adapter,
    prepare_model_for_qlora_training,
    qlora_causal_lm_loss,
    DEFAULT_TARGET_MODULES,
)
from .fused_ce import (
    FusedLinearCrossEntropy,
    fused_linear_cross_entropy,
    DEFAULT_CHUNK,
)
# native_llama imports exllamav3.modules lazily (inside __init__), so importing
# the symbols here does not pull in the CUDA extension at package import time.
from .native_llama import (
    NativeLlamaQLoRA,
    DiffLinear,
)

__all__ = [
    "EXL3LoRAFunction",
    "qlora_linear_forward",
    "QLoRALinear",
    "reference_forward",
    "Exl3LoRALinear",
    "attach_qlora",
    "iter_lora_modules",
    "save_lora_adapter",
    "load_lora_adapter",
    "prepare_model_for_qlora_training",
    "qlora_causal_lm_loss",
    "DEFAULT_TARGET_MODULES",
    "FusedLinearCrossEntropy",
    "fused_linear_cross_entropy",
    "DEFAULT_CHUNK",
    "NativeLlamaQLoRA",
    "DiffLinear",
]
