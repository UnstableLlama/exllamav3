"""
QLoRA training on top of the ExLlamaV3 HuggingFace Transformers integration.

The Transformers integration (``exllamav3/integration/transformers.py``)
replaces only the *linear* layers of a model with EXL3 layers
(``Exl3HfLinear``); every other component -- norms, attention, RoPE, the LM
head's cross-entropy -- remains stock, autograd-friendly PyTorch. That means
the *entire* model becomes trainable the moment the EXL3 linears expose a
differentiable forward. This module provides exactly that, plus the plumbing
to attach trainable LoRA adapters and save them in PEFT format.

Design
------
- The quantized base weight is **frozen**. On every forward it is
  reconstructed on the fly from the trellis (``LinearEXL3.get_weight_tensor``)
  and used as a constant -- identical in spirit to bitsandbytes QLoRA
  dequantizing NF4. No gradient ever flows through the quantizer.
- Only the low-rank ``lora_a`` / ``lora_b`` parameters are trainable.
- The differentiable forward/backward is ``EXL3LoRAFunction`` from
  ``qlora_linear.py``, whose hand-written backward is gradchecked.

Typical use::

    from exllamav3.integration.transformers import patch_transformers
    from exllamav3.training.hf_qlora import attach_qlora, save_lora_adapter

    patch_transformers()
    model = AutoModelForCausalLM.from_pretrained(exl3_dir, device_map="cuda")
    attach_qlora(model, r=16, alpha=32, target_modules=["q_proj", "v_proj"])
    # ... train with HF Trainer or a plain loop ...
    save_lora_adapter(model, "out/adapter")   # PEFT format, loadable for inference
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional
import json
import os
import torch
import torch.nn as nn

from .qlora_linear import EXL3LoRAFunction
from .fused_ce import fused_linear_cross_entropy, DEFAULT_CHUNK, IGNORE_INDEX


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


class Exl3LoRALinear(nn.Module):
    """
    Trainable LoRA wrapper around a frozen EXL3 linear.

    Wraps (does not copy) an existing ``Exl3HfLinear`` so its trellis buffers
    stay put and are never updated. Adds ``lora_a``/``lora_b`` parameters and
    routes the forward through ``EXL3LoRAFunction`` so gradients flow to the
    adapter and to the input, but never into the quantized weight.
    """

    def __init__(
        self,
        base: nn.Module,
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert getattr(base, "inner", None) is not None, \
            "base Exl3HfLinear must be finalized (have .inner) before wrapping"
        self.base = base                       # frozen; holds trellis + .inner
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        denom = (r ** 0.5) if use_rslora else r
        self.scale = float(alpha) / float(denom)
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora
        self.compute_dtype = compute_dtype
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Adapters are fp32 master weights (compute happens in compute_dtype via
        # casting inside EXL3LoRAFunction). fp32 params keep the optimizer stable
        # and satisfy the fp16 GradScaler, which refuses to unscale fp16 grads.
        dev = self._infer_device()
        self.lora_a = nn.Parameter(torch.empty(self.in_features, r, dtype=torch.float32, device=dev))
        self.lora_b = nn.Parameter(torch.zeros(r, self.out_features, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

        # Freeze everything in the base.
        for p in self.base.parameters(recurse=True):
            p.requires_grad_(False)

    def _infer_device(self):
        try:
            return self.base.inner.trellis.device
        except Exception:
            return None

    def _weight_fn(self) -> Callable[[], torch.Tensor]:
        inner = self.base.inner
        cdt = self.compute_dtype
        # get_weight_tensor() returns the full effective [in, out] weight (half).
        return lambda: inner.get_weight_tensor().to(cdt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xc = x.to(self.compute_dtype)
        lora_in = self.dropout(xc) if self.dropout is not None else xc
        bias = getattr(self.base.inner, "bias", None)
        if bias is not None:
            bias = bias.to(self.compute_dtype)
        # NOTE: dropout (if any) is applied to both branches via lora_in; with
        # the default dropout=0.0 this is exact. For nonzero dropout the base
        # path also sees dropped inputs, matching PEFT's pre-projection dropout.
        y = EXL3LoRAFunction.apply(
            lora_in, self.lora_a, self.lora_b, bias, self.scale, self._weight_fn()
        )
        return y.to(in_dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"scale={self.scale:.4g}, compute_dtype={self.compute_dtype}")


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _name_matches(name: str, targets: Iterable[str]) -> bool:
    leaf = name.split(".")[-1]
    return any(leaf == t or name.endswith("." + t) for t in targets)


def attach_qlora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    target_modules: Optional[Iterable[str]] = None,
    dropout: float = 0.0,
    use_rslora: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
    freeze_base: bool = True,
    verbose: bool = True,
) -> list[nn.Parameter]:
    """
    Replace matching EXL3 linears in ``model`` with trainable LoRA wrappers.

    Returns the list of trainable LoRA parameters. After this call, only those
    parameters require grad; everything else (quantized weights, norms,
    embeddings) is frozen.

    ``target_modules`` matches by leaf name (e.g. ``"q_proj"``), PEFT-style.
    Defaults to the standard attention + MLP projections.
    """
    # Import here so this module is importable without transformers installed
    # (e.g. for the CPU mechanics test, which uses its own mock layers).
    try:
        from exllamav3.integration.transformers import Exl3HfLinear
        exl3_types: tuple = (Exl3HfLinear,)
    except Exception:
        exl3_types = tuple()

    targets = list(target_modules) if target_modules is not None else DEFAULT_TARGET_MODULES

    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    to_replace = []
    for name, module in model.named_modules():
        is_exl3 = exl3_types and isinstance(module, exl3_types)
        # Also allow duck-typed modules exposing .inner.get_weight_tensor (test mocks).
        is_ducktyped = (not is_exl3
                        and hasattr(module, "inner")
                        and hasattr(getattr(module, "inner"), "get_weight_tensor")
                        and hasattr(module, "in_features"))
        if (is_exl3 or is_ducktyped) and _name_matches(name, targets):
            to_replace.append((name, module))

    trainable: list[nn.Parameter] = []
    for name, module in to_replace:
        wrapper = Exl3LoRALinear(
            module, r=r, alpha=alpha, dropout=dropout,
            use_rslora=use_rslora, compute_dtype=compute_dtype,
        )
        _set_submodule(model, name, wrapper)
        trainable.extend([wrapper.lora_a, wrapper.lora_b])

    if verbose:
        n_params = sum(p.numel() for p in trainable)
        print(f" -- attach_qlora: wrapped {len(to_replace)} layers, "
              f"{n_params:,} trainable params (r={r}, alpha={alpha})")
        if not to_replace:
            print(" -- attach_qlora: WARNING no matching EXL3 layers found "
                  f"(target_modules={targets})")

    return trainable


def prepare_model_for_qlora_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
) -> nn.Module:
    """
    Make a (mostly frozen, EXL3) HF model ready for adapter training.

    Mirrors what PEFT's ``prepare_model_for_kbit_training`` does for the bits
    that matter here:

      * enable gradient checkpointing (trade compute for activation memory --
        essential at 7-8B), and
      * register the input-embedding hook so activations require grad. Without
        this, checkpointing silently drops gradients because the frozen
        embedding output has ``requires_grad=False`` and nothing in the
        checkpointed region is a leaf that needs grad.
      * disable the KV cache (incompatible with training / checkpointing).
    """
    if hasattr(model, "config"):
        model.config.use_cache = False

    if use_gradient_checkpointing:
        # Non-reentrant checkpointing plays well with frozen params + custom
        # autograd Functions and doesn't need the input-grad hack, but we set
        # the hook anyway for the reentrant fallback and for safety.
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    return model


def _head_weight_fn(head: nn.Module) -> Callable[[], torch.Tensor]:
    """
    Return a closure giving the head weight in ``[hidden, vocab]`` orientation
    (so ``logits = hidden @ W``), for either an EXL3 head or a plain Linear.
    """
    inner = getattr(head, "inner", None)
    if inner is not None and hasattr(inner, "get_weight_tensor"):
        return lambda: inner.get_weight_tensor()
    if hasattr(head, "get_weight_tensor"):
        return lambda: head.get_weight_tensor()
    if isinstance(head, nn.Linear):
        # nn.Linear weight is [vocab, hidden]; transpose to [hidden, vocab].
        return lambda: head.weight.t()
    raise TypeError(f"Unsupported LM head type for fused CE: {type(head)}")


def qlora_causal_lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    chunk: int = DEFAULT_CHUNK,
    ignore_index: int = IGNORE_INDEX,
    **decoder_kwargs,
) -> torch.Tensor:
    """
    Causal-LM loss via the fused linear cross-entropy head.

    Runs the decoder backbone to hidden states, then computes the shifted
    cross-entropy without ever materialising the full ``[tokens, vocab]``
    logits. Assumes the LM head is frozen (standard QLoRA) and that the model
    does not apply final-logit softcapping (e.g. Gemma2 needs the standard
    head instead).

    Works with any model exposing the standard HF ``get_decoder()`` /
    ``get_output_embeddings()`` interface. Unwraps DataParallel/DDP.
    """
    core = model.module if hasattr(model, "module") else model
    decoder = core.get_decoder()
    out = decoder(input_ids=input_ids, attention_mask=attention_mask, **decoder_kwargs)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    head = core.get_output_embeddings()
    weight_fn = _head_weight_fn(head)
    return fused_linear_cross_entropy(
        hidden, weight_fn, labels, chunk=chunk, ignore_index=ignore_index, shift=True
    )


def iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, Exl3LoRALinear):
            yield name, module


def save_lora_adapter(
    model: nn.Module,
    directory: str,
    base_model_name_or_path: Optional[str] = None,
) -> None:
    """
    Save attached adapters in PEFT format, loadable by both PEFT and the
    inference-side ``exllamav3.model.lora.LoRA`` loader.

    PEFT convention: ``lora_A`` is ``[r, in]`` and ``lora_B`` is ``[out, r]``,
    keyed ``base_model.model.<path>.lora_A.weight`` etc. Our internal tensors
    are ``A=[in, r]`` / ``B=[r, out]``, so we transpose on save. The
    ``alpha``/``r`` scaling is reproduced by the loader from the config, so we
    save the *unscaled* B.
    """
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)

    state: dict[str, torch.Tensor] = {}
    r = alpha = None
    use_rslora = False
    target_leaves: set[str] = set()

    for name, module in iter_lora_modules(model):
        r = module.r
        alpha = module.lora_alpha
        use_rslora = module.use_rslora
        target_leaves.add(name.split(".")[-1])
        key = f"base_model.model.{name}"
        state[f"{key}.lora_A.weight"] = module.lora_a.detach().t().contiguous().to(torch.float16).cpu()
        state[f"{key}.lora_B.weight"] = module.lora_b.detach().t().contiguous().to(torch.float16).cpu()

    if r is None:
        raise ValueError("No Exl3LoRALinear modules found in model; nothing to save.")

    save_file(state, os.path.join(directory, "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": use_rslora,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": sorted(target_leaves),
        "base_model_name_or_path": base_model_name_or_path,
    }
    with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf8") as f:
        json.dump(config, f, indent=2)

    print(f" -- saved LoRA adapter ({len(target_leaves)} target types) to {directory}")


def load_lora_adapter(
    model: nn.Module,
    directory: str,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Attach an adapter saved by :func:`save_lora_adapter` (PEFT format) onto an
    HF model for inference. Reads the config for r/alpha/targets, wraps the
    matching EXL3 linears, and loads the A/B weights (transposing back from the
    PEFT ``[r, in]`` / ``[out, r]`` orientation to our internal layout).
    """
    from safetensors.torch import load_file

    with open(os.path.join(directory, "adapter_config.json"), encoding="utf8") as f:
        cfg = json.load(f)

    attach_qlora(
        model,
        r=cfg["r"],
        alpha=cfg["lora_alpha"],
        target_modules=cfg.get("target_modules"),
        use_rslora=cfg.get("use_rslora", False),
        compute_dtype=compute_dtype,
        verbose=False,
    )

    wrappers = {name: mod for name, mod in iter_lora_modules(model)}
    state = load_file(os.path.join(directory, "adapter_model.safetensors"))

    loaded = 0
    for key, tensor in state.items():
        # base_model.model.<name>.lora_A.weight  /  .lora_B.weight
        if not key.startswith("base_model.model."):
            continue
        body = key[len("base_model.model."):]
        if body.endswith(".lora_A.weight"):
            name, half = body[:-len(".lora_A.weight")], "A"
        elif body.endswith(".lora_B.weight"):
            name, half = body[:-len(".lora_B.weight")], "B"
        else:
            continue
        mod = wrappers.get(name)
        if mod is None:
            continue
        with torch.no_grad():
            if half == "A":  # [r, in] -> [in, r]
                mod.lora_a.copy_(tensor.t().to(mod.lora_a.dtype).to(mod.lora_a.device))
            else:            # [out, r] -> [r, out]
                mod.lora_b.copy_(tensor.t().to(mod.lora_b.dtype).to(mod.lora_b.device))
        loaded += 1

    print(f" -- loaded LoRA adapter from {directory} ({loaded} tensors into "
          f"{len(wrappers)} modules)")
    return model
