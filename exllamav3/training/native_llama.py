"""
Transformers-free differentiable Llama forward over native EXL3 weights.

Why this exists
---------------
The HF Transformers integration (``exllamav3/integration/transformers.py``)
turns an EXL3 model into a trainable graph by replacing only the linear
layers and letting stock Transformers supply norms/attention/RoPE. That works
in principle but couples training to a single Transformers version: the EXL3
Llama-3.2 weights were calibrated against transformers 4.45, and 5.x changed
``llama3`` RoPE handling, producing a correct-per-layer but garbage-overall
forward (see ``doc/qlora_handoff.md``).

This module sidesteps that entirely. It reconstructs the Llama decoder forward
in plain, autograd-friendly PyTorch directly on top of exllamav3's *own* loaded
modules -- the same weights, RoPE settings (``RoPE.inv_freq``), norms and
attention scale that the native (correct) inference forward uses. There is no
``transformers`` import anywhere in the path, so it cannot be broken by an
upstream version bump.

The quantized base weights stay frozen and are reconstructed on the fly from
the trellis via ``LinearEXL3.get_weight_tensor()`` (orientation ``[in, out]``,
``y = x @ W``). Only the low-rank ``lora_a`` / ``lora_b`` adapters train, routed
through the gradchecked :class:`EXL3LoRAFunction`. The LM head is handled by the
streaming :func:`fused_linear_cross_entropy`, so the ``[tokens, vocab]`` logit
tensor is never materialized during training.

Scope: Llama-family decoders (``LlamaModel`` and architectures that reuse the
same ``TransformerBlock`` = pre-norm attention (GQA + NeoX RoPE) + pre-norm
gated-SiLU MLP, e.g. Mistral/Qwen2). Models with q/k norms, sliding windows,
logit softcapping, MoE or interleaved/headwise gates are rejected explicitly so
failures are loud rather than silently wrong.
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qlora_linear import EXL3LoRAFunction
from .fused_ce import fused_linear_cross_entropy, DEFAULT_CHUNK, IGNORE_INDEX


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

# Leaf name -> the role we expect it to play in the differentiable forward.
_ROLE_BY_LEAF = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
}


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    # Matches exllamav3.util.rope._rotate_half_neox: split the head dim in half.
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class DiffLinear(nn.Module):
    """
    Differentiable linear over a frozen native ``Linear`` module.

    The frozen effective weight is reconstructed on every call from the EXL3
    trellis (or read directly for an fp16 inner layer) via ``get_weight_tensor``
    and treated as a constant -- no gradient ever flows into the quantized base.
    Optionally carries trainable LoRA ``a``/``b`` (fp32 master weights); when
    absent the layer is a pure frozen projection. Either way the forward/backward
    runs through the gradchecked :class:`EXL3LoRAFunction`, which recomputes the
    weight in the backward pass instead of stashing it (activation-memory win).

    Shapes: ``a`` is ``[in, r]``, ``b`` is ``[r, out]`` (``y = x @ W + s·x@a@b``).
    """

    def __init__(
        self,
        linear: nn.Module,
        r: int = 0,
        alpha: float = 16.0,
        use_rslora: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert getattr(linear, "inner", None) is not None, \
            "native Linear must be loaded (have .inner) before wrapping"
        self.linear = linear                 # frozen; holds trellis / fp16 weight
        self.key = linear.key
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype
        self.r = r
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora

        if r > 0:
            denom = (r ** 0.5) if use_rslora else r
            self.scale = float(alpha) / float(denom)
            dev = self._infer_device()
            # B starts at zero so the adapter is a no-op at init (training begins
            # from the exact base model); A uses the PEFT kaiming init.
            self.lora_a = nn.Parameter(torch.empty(self.in_features, r, dtype=torch.float32, device=dev))
            self.lora_b = nn.Parameter(torch.zeros(r, self.out_features, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        else:
            self.scale = 1.0
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        # The wrapped native Linear (an exllamav3 ABC Module, not an nn.Module)
        # holds its weights as plain tensors / buffers, never nn.Parameters, so
        # there is nothing to freeze: no gradient can ever reach the base.

    def _infer_device(self):
        try:
            return self.linear.inner.trellis.device
        except Exception:
            return getattr(self.linear, "device", None)

    def _weight_fn(self) -> Callable[[], torch.Tensor]:
        inner = self.linear.inner
        cdt = self.compute_dtype
        return lambda: inner.get_weight_tensor().to(cdt)

    def _bias(self) -> Optional[torch.Tensor]:
        get_bias = getattr(self.linear.inner, "get_bias_tensor", None)
        if get_bias is None:
            return None
        b = get_bias()
        return b.to(self.compute_dtype) if b is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.to(self.compute_dtype)
        return EXL3LoRAFunction.apply(
            xc, self.lora_a, self.lora_b, self._bias(), self.scale, self._weight_fn()
        )

    def extra_repr(self) -> str:
        return (f"key={self.key}, in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, compute_dtype={self.compute_dtype}")


class NativeLlamaQLoRA(nn.Module):
    """
    Differentiable Llama decoder built on a loaded exllamav3 ``Model``.

    Construct from a model already loaded with native exllamav3 (the path that
    forwards correctly on the quantized weights), then train LoRA adapters with
    a plain PyTorch loop -- no HuggingFace Transformers anywhere.

    Example::

        from exllamav3 import Config, Model
        from exllamav3.training.native_llama import NativeLlamaQLoRA

        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        model.load(device="cuda:0")
        net = NativeLlamaQLoRA(model, r=16, alpha=32,
                               target_modules=["q_proj", "v_proj"])
        loss = net.compute_loss(input_ids, labels)   # fused-CE, frozen head
        loss.backward()
        ...
        net.save_adapter("out/pirate")               # PEFT format
    """

    def __init__(
        self,
        model: nn.Module,
        r: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[Iterable[str]] = None,
        use_rslora: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        # Import here so the module is importable without a CUDA build present.
        from ..modules import (
            Embedding, TransformerBlock, RMSNorm, Attention, GatedMLP, Linear,
        )

        self.model = model
        self.compute_dtype = compute_dtype
        self.gradient_checkpointing = gradient_checkpointing
        targets = set(target_modules) if target_modules is not None else set(DEFAULT_TARGET_MODULES)
        self.target_modules = sorted(targets)
        self.r = r
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora

        mods = list(model.modules)

        # Embedding is the first module; final RMSNorm + LM head are the last two.
        assert isinstance(mods[0], Embedding), \
            f"expected Embedding as first module, got {type(mods[0]).__name__}"
        self.embed = mods[0]
        assert isinstance(mods[-2], RMSNorm), \
            f"expected final RMSNorm as penultimate module, got {type(mods[-2]).__name__}"
        self.final_norm = mods[-2]
        assert isinstance(mods[-1], Linear), \
            f"expected Linear LM head as last module, got {type(mods[-1]).__name__}"
        self.lm_head = mods[-1]

        blocks = [m for m in mods if isinstance(m, TransformerBlock)]
        assert blocks, "no TransformerBlock modules found; unsupported architecture"

        self.blocks = nn.ModuleList()
        self._block_meta = []
        wrappers: list[DiffLinear] = []

        def wrap(linear, leaf):
            is_target = leaf in targets
            w = DiffLinear(
                linear,
                r=r if is_target else 0,
                alpha=alpha,
                use_rslora=use_rslora,
                compute_dtype=compute_dtype,
            )
            wrappers.append(w)
            return w

        for blk in blocks:
            attn: Attention = blk.attn
            mlp: GatedMLP = blk.mlp
            self._assert_supported(blk, attn, mlp)

            # MLP may be sliced across intermediate dim for very wide models;
            # wrap every slice and sum the down-projections (mirrors GatedMLP).
            gates = [wrap(g, "gate_proj") for g in mlp.gates]
            ups = [wrap(u, "up_proj") for u in mlp.ups]
            downs = [wrap(d, "down_proj") for d in mlp.downs]

            entry = nn.Module()
            entry.attn_norm = blk.attn_norm
            entry.mlp_norm = blk.mlp_norm
            entry.q_proj = wrap(attn.q_proj, "q_proj")
            entry.k_proj = wrap(attn.k_proj, "k_proj")
            entry.v_proj = wrap(attn.v_proj, "v_proj")
            entry.o_proj = wrap(attn.o_proj, "o_proj")
            entry.gates = nn.ModuleList(gates)
            entry.ups = nn.ModuleList(ups)
            entry.downs = nn.ModuleList(downs)
            self.blocks.append(entry)

            self._block_meta.append({
                "num_q_heads": attn.num_q_heads,
                "num_kv_heads": attn.num_kv_heads,
                "head_dim": attn.head_dim,
                "sm_scale": attn.sm_scale,
                "attn_eps": blk.attn_norm.rms_norm_eps,
                "mlp_eps": blk.mlp_norm.rms_norm_eps,
                # RoPE (llama3-scaled inv_freq lives on the loaded RoPE object).
                "inv_freq": attn.rope.inv_freq,
                "attn_factor": attn.rope.attn_factor,
            })

        self._wrappers = wrappers
        self.final_eps = self.final_norm.rms_norm_eps
        # The decoder device (norms / linears / RoPE). May differ from the
        # embedding's device, which is loaded on CPU (prefer_cpu). Assumes the
        # whole decoder sits on one device (true for single-GPU loads).
        self.device = self.final_norm.weight.device

        # Sanity: every requested target name actually matched something.
        matched = {w.key.split(".")[-1] for w in wrappers if w.r > 0}
        missing = targets - matched
        if missing:
            raise ValueError(
                f"target_modules {sorted(missing)} matched no linear in the model "
                f"(available leaves: {sorted({w.key.split('.')[-1] for w in wrappers})})"
            )

    # --- architecture guard ------------------------------------------------

    @staticmethod
    def _assert_supported(blk, attn, mlp):
        from ..modules import GatedMLP
        key = getattr(blk, "key", "?")
        assert attn is not None and mlp is not None, \
            f"{key}: block must have both attention and MLP (parallel/no-op blocks unsupported)"
        assert isinstance(mlp, GatedMLP), \
            f"{key}: only GatedMLP (SiLU/GeLU gated) MLPs are supported, got {type(mlp).__name__}"
        assert mlp.activation_fn in ("silu",), \
            f"{key}: only SiLU gated MLP is supported, got activation {mlp.activation_fn!r}"
        assert getattr(mlp, "act_limit", 0.0) in (0.0, None), \
            f"{key}: gated-MLP act_limit is not supported"
        assert attn.q_norm is None and attn.k_norm is None, \
            f"{key}: attention q/k norms are not supported by the native QLoRA forward"
        assert getattr(attn, "v_norm", None) is None, f"{key}: attention v_norm unsupported"
        assert getattr(attn, "g_proj", None) is None and not getattr(attn, "interleaved_gate", False), \
            f"{key}: attention output gating is not supported"
        assert getattr(attn, "sliding_window", -1) in (-1, 0, None), \
            f"{key}: sliding-window attention is not supported"
        assert not getattr(attn, "logit_softcapping", 0.0), \
            f"{key}: attention logit softcapping is not supported"
        assert attn.rope is not None and attn.rope.inv_freq is not None, \
            f"{key}: model loaded without a RoPE table; cannot build positional encoding"
        assert attn.rope.mrope_section is None, f"{key}: mRoPE is not supported"
        assert attn.rope.rope_settings.rope_style.name == "NEOX", \
            f"{key}: only NeoX-style RoPE is supported, got {attn.rope.rope_settings.rope_style.name}"
        assert attn.rope.inv_freq.numel() * 2 == attn.head_dim, \
            f"{key}: partial-rotary RoPE is not supported (rotary_dim != head_dim)"

    # --- forward -----------------------------------------------------------

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # Matches RMSNorm.forward_torch: normalize in fp32, then apply weight.
        var = x.float().pow(2).mean(dim=-1, keepdim=True) + eps
        xn = x.float() * torch.rsqrt(var)
        return xn * weight.float()

    def _apply_rope(self, x: torch.Tensor, inv_freq: torch.Tensor, attn_factor: float,
                    position_ids: torch.Tensor) -> torch.Tensor:
        # x: [b, t, n_heads, head_dim] (fp32). position_ids: [b, t].
        freqs = position_ids.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0).unsqueeze(0)  # [b,t,hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)                  # [b,t,hd]  (NeoX layout)
        cos = (emb.cos() * attn_factor).unsqueeze(2)             # [b,t,1,hd]
        sin = (emb.sin() * attn_factor).unsqueeze(2)
        return x * cos + _rotate_half_neox(x) * sin

    def _attn_bias(self, attention_mask: Optional[torch.Tensor], t: int,
                   device, dtype) -> torch.Tensor:
        # Additive attention bias: causal (upper triangle masked) AND, if given, a
        # [b, t] key-padding mask. Assumes right-padding (pads at the end), so no
        # real-token query row is ever fully masked -> softmax can't produce NaN.
        neg = float("-inf")
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
        bias = torch.zeros(1, 1, t, t, dtype=dtype, device=device)
        bias = bias.masked_fill(causal[None, None], neg)
        if attention_mask is not None:
            key_pad = (attention_mask == 0)[:, None, None, :]    # [b,1,1,t]
            bias = bias.masked_fill(key_pad, neg)
        return bias

    def _block_forward(self, meta, entry, hidden, position_ids, attn_bias):
        bsz, t, _ = hidden.shape
        nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]

        # --- attention ---
        normed = self._rmsnorm(hidden, entry.attn_norm.weight, meta["attn_eps"])
        q = entry.q_proj(normed).view(bsz, t, nq, hd).float()
        k = entry.k_proj(normed).view(bsz, t, nkv, hd).float()
        v = entry.v_proj(normed).view(bsz, t, nkv, hd).float()

        q = self._apply_rope(q, meta["inv_freq"], meta["attn_factor"], position_ids)
        k = self._apply_rope(k, meta["inv_freq"], meta["attn_factor"], position_ids)

        # [b, n_heads, t, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if nq != nkv:                                            # GQA: expand KV groups
            rep = nq // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * meta["sm_scale"]  # [b,nq,t,t]
        scores = scores + attn_bias.to(scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)                            # [b,nq,t,hd]
        ctx = ctx.transpose(1, 2).reshape(bsz, t, nq * hd)
        attn_out = entry.o_proj(ctx).float()
        hidden = hidden + attn_out

        # --- gated MLP ---
        normed2 = self._rmsnorm(hidden, entry.mlp_norm.weight, meta["mlp_eps"])
        mlp_out = None
        for gate, up, down in zip(entry.gates, entry.ups, entry.downs):
            a = F.silu(gate(normed2)) * up(normed2)
            d = down(a).float()
            mlp_out = d if mlp_out is None else mlp_out + d
        hidden = hidden + mlp_out
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return final-norm hidden states ``[b, t, d]`` in fp32."""
        bsz, t = input_ids.shape
        # The embedding may live on CPU (it has prefer_cpu=True and is loaded on
        # CPU even for a single-device model), while the decoder lives on the GPU.
        # Run the lookup on the embedding's device, then move to the decoder.
        dec_device = self.device
        emb_device = self.embed.embedding.weight.device

        hidden = self.embed.embedding(input_ids.to(emb_device))
        if getattr(self.embed, "multiplier", 1.0) != 1.0:
            hidden = hidden * self.embed.multiplier
        if getattr(self.embed, "normalize", False):
            hidden = hidden * (hidden.shape[-1] ** 0.5)
        hidden = hidden.to(dec_device).float()

        if attention_mask is not None:
            attention_mask = attention_mask.to(dec_device)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.clamp_min(0)
            else:
                position_ids = torch.arange(t, device=dec_device).unsqueeze(0).expand(bsz, t)
        position_ids = position_ids.to(dec_device)

        # Gradient checkpointing needs at least one input that requires grad, but
        # the base embedding is frozen. Detach to a leaf and flag it so the
        # checkpointed blocks (whose only trainable params are the LoRA adapters)
        # build a backward graph. No gradient is lost: the embedding is frozen.
        ckpt = self.gradient_checkpointing and self.training
        if ckpt:
            hidden = hidden.detach().requires_grad_(True)

        attn_bias = self._attn_bias(attention_mask, t, dec_device, torch.float32)

        for meta, entry in zip(self._block_meta, self.blocks):
            if ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._block_forward, meta, entry, hidden, position_ids, attn_bias,
                    use_reentrant=False,
                )
            else:
                hidden = self._block_forward(meta, entry, hidden, position_ids, attn_bias)

        hidden = self._rmsnorm(hidden, self.final_norm.weight, self.final_eps)
        return hidden

    # --- heads -------------------------------------------------------------

    def lm_head_weight_fn(self) -> Callable[[], torch.Tensor]:
        """Frozen LM-head weight closure in ``[hidden, vocab]`` orientation."""
        inner = self.lm_head.inner
        return lambda: inner.get_weight_tensor()

    def logits(self, input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Materialize full logits ``[b, t, vocab]`` (validation / small batches)."""
        hidden = self.forward(input_ids, attention_mask)
        w = self.lm_head_weight_fn()()
        return hidden.to(w.dtype) @ w

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk: int = DEFAULT_CHUNK,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Shifted causal-LM cross-entropy via the streaming fused head."""
        hidden = self.forward(input_ids, attention_mask)
        labels = labels.to(hidden.device)
        return fused_linear_cross_entropy(
            hidden, self.lm_head_weight_fn(), labels,
            chunk=chunk, ignore_index=ignore_index, shift=True,
        )

    # --- adapter parameters / IO ------------------------------------------

    def lora_parameters(self) -> list[nn.Parameter]:
        ps: list[nn.Parameter] = []
        for w in self._wrappers:
            if w.r > 0:
                ps += [w.lora_a, w.lora_b]
        return ps

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.lora_parameters())

    @torch.no_grad()
    def apply_to_native(self, scaling: float = 1.0) -> None:
        """
        Push the *current* adapter weights into the underlying native ``Linear``
        modules' runtime LoRA slots, so that ``model.forward`` / generation
        reflects the adapter. Independent of the training forward (which reads
        only the frozen base weight), so this is safe to call mid-training to
        sample progress. Call :meth:`remove_from_native` to revert to base.

        The native ``Linear.apply_lora`` computes ``x @ A @ B`` with no extra
        scale, so the LoRA scale is folded into B here.
        """
        for w in self._wrappers:
            if w.r <= 0:
                continue
            lin = w.linear
            a = w.lora_a.detach().to(torch.float16)
            b = (w.lora_b.detach() * (w.scale * scaling)).to(torch.float16)
            lin.lora_a_tensors[self] = a.to(lin.device)
            lin.lora_b_tensors[self] = b.to(lin.device)

    @torch.no_grad()
    def remove_from_native(self) -> None:
        """Remove this adapter from the native modules' runtime LoRA slots."""
        for w in self._wrappers:
            if w.r <= 0:
                continue
            w.linear.lora_a_tensors.pop(self, None)
            w.linear.lora_b_tensors.pop(self, None)

    def save_adapter(self, directory: str,
                     base_model_name_or_path: Optional[str] = None) -> None:
        """
        Write the trained adapters in PEFT format, keyed by the native module
        path so they load with both PEFT and exllamav3's ``LoRA.from_directory``
        (and hence ``examples/qlora_infer_native.py``).

        Internal tensors are ``a=[in, r]`` / ``b=[r, out]``; PEFT stores
        ``lora_A=[r, in]`` / ``lora_B=[out, r]``, so we transpose on save and
        emit the *unscaled* B (the loader reapplies alpha/r).
        """
        from safetensors.torch import save_file
        os.makedirs(directory, exist_ok=True)

        state: dict[str, torch.Tensor] = {}
        target_leaves: set[str] = set()
        r = alpha = None
        use_rslora = self.use_rslora
        for w in self._wrappers:
            if w.r <= 0:
                continue
            r, alpha = w.r, w.lora_alpha
            target_leaves.add(w.key.split(".")[-1])
            key = f"base_model.model.{w.key}"
            state[f"{key}.lora_A.weight"] = w.lora_a.detach().t().contiguous().to(torch.float16).cpu()
            state[f"{key}.lora_B.weight"] = w.lora_b.detach().t().contiguous().to(torch.float16).cpu()

        if r is None:
            raise ValueError("No trainable LoRA adapters to save.")

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
        print(f" -- saved native QLoRA adapter ({len(target_leaves)} target types) to {directory}")

    def load_adapter(self, directory: str) -> int:
        """
        Load adapter weights previously written by :meth:`save_adapter` back into
        the trainable wrappers, to *continue* training from a checkpoint. Inverts
        the save transpose (PEFT ``lora_A=[r, in]`` / ``lora_B=[out, r]`` ->
        internal ``a=[in, r]`` / ``b=[r, out]``).

        Only the adapter weights are restored, NOT optimizer state -- AdamW
        resumes cold (a brief, harmless re-warmup for LoRA). The target modules /
        rank must match the current model (a shape mismatch raises). Returns the
        number of wrappers loaded.
        """
        from safetensors.torch import load_file
        path = os.path.join(directory, "adapter_model.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No adapter_model.safetensors in {directory}")
        state = load_file(path)

        loaded = 0
        for w in self._wrappers:
            if w.r <= 0:
                continue
            key = f"base_model.model.{w.key}"
            ak, bk = f"{key}.lora_A.weight", f"{key}.lora_B.weight"
            if ak not in state or bk not in state:
                raise KeyError(f"checkpoint missing tensors for {w.key} ({ak})")
            a = state[ak].t()  # [r, in] -> [in, r]
            b = state[bk].t()  # [out, r] -> [r, out]
            if a.shape != w.lora_a.shape or b.shape != w.lora_b.shape:
                raise ValueError(
                    f"adapter shape mismatch for {w.key}: checkpoint "
                    f"a{tuple(a.shape)}/b{tuple(b.shape)} vs model "
                    f"a{tuple(w.lora_a.shape)}/b{tuple(w.lora_b.shape)} "
                    f"-- do --r/--targets match the checkpoint?"
                )
            with torch.no_grad():
                w.lora_a.copy_(a.to(w.lora_a.dtype).to(w.lora_a.device))
                w.lora_b.copy_(b.to(w.lora_b.dtype).to(w.lora_b.device))
            loaded += 1

        if loaded == 0:
            raise ValueError("No trainable LoRA adapters matched the checkpoint.")
        print(f" -- resumed {loaded} adapters from {directory} (optimizer state not restored)")
        return loaded
