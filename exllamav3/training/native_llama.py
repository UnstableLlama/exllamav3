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

from . import backbone
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
        assert backbone.is_loaded(linear), \
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
            dev = backbone.linear_device(self.linear)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.to(self.compute_dtype)
        return EXL3LoRAFunction.apply(
            xc, self.lora_a, self.lora_b,
            backbone.frozen_bias(self.linear, self.compute_dtype),
            self.scale,
            backbone.frozen_weight_closure(self.linear, self.compute_dtype),
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
        train_embeddings: bool = False,
        train_head: bool = False,
    ):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self.gradient_checkpointing = gradient_checkpointing
        targets = set(target_modules) if target_modules is not None else set(DEFAULT_TARGET_MODULES)
        self.target_modules = sorted(targets)
        self.r = r
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora

        # All reach into exllamav3's internal module layout goes through backbone:
        # embedding first, final RMSNorm + LM head last, transformer blocks between.
        self.embed, blocks, self.final_norm, self.lm_head = backbone.split_decoder(model)

        self.blocks = nn.ModuleList()
        self._block_meta = []
        self._block_devices = []   # per-block device (differs under layer-autosplit)
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
            backbone.assert_block_supported(blk)
            q_proj, k_proj, v_proj, o_proj = backbone.attn_projections(blk)
            gate_lins, up_lins, down_lins = backbone.mlp_projections(blk)
            attn_norm, mlp_norm = backbone.block_norms(blk)

            # MLP may be sliced across intermediate dim for very wide models;
            # wrap every slice and sum the down-projections (mirrors GatedMLP).
            gates = [wrap(g, "gate_proj") for g in gate_lins]
            ups = [wrap(u, "up_proj") for u in up_lins]
            downs = [wrap(d, "down_proj") for d in down_lins]

            entry = nn.Module()
            entry.attn_norm = attn_norm
            entry.mlp_norm = mlp_norm
            entry.q_proj = wrap(q_proj, "q_proj")
            entry.k_proj = wrap(k_proj, "k_proj")
            entry.v_proj = wrap(v_proj, "v_proj")
            entry.o_proj = wrap(o_proj, "o_proj")
            entry.gates = nn.ModuleList(gates)
            entry.ups = nn.ModuleList(ups)
            entry.downs = nn.ModuleList(downs)
            self.blocks.append(entry)

            self._block_meta.append(backbone.block_metadata(blk))
            self._block_devices.append(backbone.block_device(blk))

        self._wrappers = wrappers
        self.final_eps = backbone.rms_norm_eps(self.final_norm)
        # Output device = where the final norm + LM head live. Under a single
        # load this is the one decoder device; under layer-autosplit it is the
        # last device in the split (modules[-1].device). The embedding is loaded
        # on CPU (prefer_cpu) regardless.
        self.device = self.final_norm.weight.device
        self._head_device = backbone.linear_device(self.lm_head)

        # Sanity: every requested target name actually matched something.
        matched = {w.key.split(".")[-1] for w in wrappers if w.r > 0}
        missing = targets - matched
        if missing:
            raise ValueError(
                f"target_modules {sorted(missing)} matched no linear in the model "
                f"(available leaves: {sorted({w.key.split('.')[-1] for w in wrappers})})"
            )

        # --- optional full-trained input/output embeddings (modules_to_save) ---
        # LoRA normally freezes embed_tokens / lm_head; these flags fully train
        # them (a trainable fp32 copy reconstructed once from the frozen base),
        # PEFT-`modules_to_save` style, and save them alongside the adapter.
        #
        # exllamav3 always loads a *separate* embedding and lm_head, even for a
        # tied model (the tied weight is materialized into both), so we train
        # whichever is requested independently -- no shared-parameter special
        # case. (tie_word_embeddings is recorded in the saved config only as a
        # hint for a downstream merge/re-quantize step.)
        self.train_embeddings = bool(train_embeddings)
        self.train_head = bool(train_head)
        self.tie_word_embeddings = bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        self.embed_weight = None    # [vocab, hidden], trainable, or None
        self.head_weight = None     # [hidden, vocab], trainable, or None

        # The base embedding is often loaded on CPU (prefer_cpu); put the trainable
        # copies on the GPU compute devices so training isn't bottlenecked on a CPU
        # optimizer / matmul. First-block device for the input embedding, head
        # device for the output projection (identical under single-device / DDP).
        if self.train_embeddings:
            w = backbone.embed_weight(self.embed).detach().to(torch.float32)   # [V, d]
            self.embed_weight = nn.Parameter(w.clone().to(self._block_devices[0]))
        if self.train_head:
            hw = backbone.head_weight_closure(self.lm_head)().detach().to(torch.float32)  # [d, V]
            self.head_weight = nn.Parameter(hw.clone().to(self._head_device))

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
        # Under a layer-autosplit load each block lives on its own device; the
        # hidden state is migrated across boundaries below (mirroring exllamav3's
        # forward_ls). Start on the first block's device. The embedding is loaded
        # on CPU (prefer_cpu); backbone.embed_tokens runs the lookup there.
        first_device = self._block_devices[0]
        if self.embed_weight is not None:
            # Trainable input embedding: lookup against the fp32 master weight,
            # then the same multiplier/normalize the frozen path applies.
            ew = self.embed_weight
            looked_up = F.embedding(input_ids.to(ew.device), ew)
            hidden = backbone.embed_apply(self.embed, looked_up).to(first_device).float()
        else:
            hidden = backbone.embed_tokens(self.embed, input_ids).to(first_device).float()

        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.clamp_min(0)
            else:
                position_ids = torch.arange(t, device=first_device).unsqueeze(0).expand(bsz, t)
        position_ids = position_ids.to(first_device)

        # Gradient checkpointing needs at least one input that requires grad. With
        # a frozen embedding the hidden state doesn't, so detach to a leaf and flag
        # it (no gradient is lost: the embedding is frozen). But when the embedding
        # is TRAINABLE, hidden already requires grad and carries the path back to
        # embed_weight -- detaching would sever it, so only detach when needed.
        ckpt = self.gradient_checkpointing and self.training
        if ckpt and not hidden.requires_grad:
            hidden = hidden.detach().requires_grad_(True)

        attn_bias = self._attn_bias(attention_mask, t, first_device, torch.float32)

        cur_device = first_device
        for meta, entry, dev in zip(self._block_meta, self.blocks, self._block_devices):
            # Cross the device boundary if this block sits on another card. All
            # no-ops under a single-device load (dev == cur_device throughout).
            if dev != cur_device:
                hidden = backbone.to_device(hidden, dev)
                position_ids = backbone.to_device(position_ids, dev)
                attn_bias = backbone.to_device(attn_bias, dev)
                cur_device = dev
            if ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._block_forward, meta, entry, hidden, position_ids, attn_bias,
                    use_reentrant=False,
                )
            else:
                hidden = self._block_forward(meta, entry, hidden, position_ids, attn_bias)

        # Final norm + head live on the output device (the last split device).
        hidden = backbone.to_device(hidden, self.device)
        hidden = self._rmsnorm(hidden, self.final_norm.weight, self.final_eps)
        return hidden

    # --- heads -------------------------------------------------------------

    def lm_head_weight_fn(self) -> Callable[[], torch.Tensor]:
        """Frozen LM-head weight closure in ``[hidden, vocab]`` orientation."""
        return backbone.head_weight_closure(self.lm_head)

    def logits(self, input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Materialize full logits ``[b, t, vocab]`` (validation / small batches)."""
        hidden = self.forward(input_ids, attention_mask)
        w = self.lm_head_weight_fn()()
        return backbone.to_device(hidden, w.device).to(w.dtype) @ w

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk: int = DEFAULT_CHUNK,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Shifted causal-LM cross-entropy.

        Frozen head: the streaming fused head (never materializes ``[tokens,
        vocab]``). Trainable head (``train_head``): standard autograd so the head
        weight gets a gradient -- but logits are computed only at the *supervised*
        positions (labels != ignore_index), so memory scales with the number of
        supervised tokens, not the full sequence."""
        hidden = self.forward(input_ids, attention_mask)
        if self.train_head:
            d = hidden.shape[-1]
            hs = hidden[:, :-1, :].reshape(-1, d)                 # shift
            lbl = labels[:, 1:].reshape(-1).to(hs.device)
            valid = lbl != ignore_index
            w = self.head_weight                                  # [d, V]
            if not bool(valid.any()):
                # No supervised tokens: keep the head in the graph with a 0 grad.
                return (hs.sum() * 0.0) + (w.sum() * 0.0)
            hs = backbone.to_device(hs[valid], w.device).to(w.dtype)
            logits = hs @ w
            return F.cross_entropy(logits, lbl[valid].to(w.device))
        # The fused head matmuls hidden against the frozen head weight, which
        # lives on the head's device; co-locate them (no-op single-device).
        hidden = backbone.to_device(hidden, self._head_device)
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

    def modules_to_save_parameters(self) -> list[nn.Parameter]:
        """Trainable full embed/head params (PEFT ``modules_to_save``), if any.
        For a tied model this is the single shared embedding weight."""
        ps: list[nn.Parameter] = []
        if self.embed_weight is not None:
            ps.append(self.embed_weight)
        if self.head_weight is not None:
            ps.append(self.head_weight)
        return ps

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All trainable params: LoRA adapters + any full-trained embed/head."""
        return self.lora_parameters() + self.modules_to_save_parameters()

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Optimizer param groups: weight decay on the LoRA params, but NONE on
        the full embed/head (weight-decaying a whole embedding table is harmful)."""
        groups = [{"params": self.lora_parameters(), "weight_decay": weight_decay}]
        ms = self.modules_to_save_parameters()
        if ms:
            groups.append({"params": ms, "weight_decay": 0.0})
        return groups

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

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
            a = w.lora_a.detach().to(torch.float16)
            b = (w.lora_b.detach() * (w.scale * scaling)).to(torch.float16)
            backbone.set_runtime_lora(w.linear, self, a, b)

    @torch.no_grad()
    def remove_from_native(self) -> None:
        """Remove this adapter from the native modules' runtime LoRA slots."""
        for w in self._wrappers:
            if w.r <= 0:
                continue
            backbone.clear_runtime_lora(w.linear, self)

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

        # Fully-trained embed/head (PEFT modules_to_save). Kept in a SEPARATE file
        # so the LoRA loader (which expects only lora_A/lora_B) is undisturbed;
        # they are large full matrices, applied by merging into the base, not via
        # the runtime LoRA slots. Saved in HF orientation ([vocab, hidden]) under
        # the standard module names.
        modules_to_save: list[str] = []
        ms_state: dict[str, torch.Tensor] = {}
        if self.embed_weight is not None:
            ms_state["model.embed_tokens.weight"] = \
                self.embed_weight.detach().to(torch.float16).cpu()           # [V, d]
            modules_to_save.append("embed_tokens")
        if self.head_weight is not None:
            ms_state["lm_head.weight"] = \
                self.head_weight.detach().t().contiguous().to(torch.float16).cpu()  # [d,V]->[V,d]
            modules_to_save.append("lm_head")
        if ms_state:
            save_file(ms_state, os.path.join(directory, "modules_to_save.safetensors"))

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
            "modules_to_save": modules_to_save,
            "tie_word_embeddings": self.tie_word_embeddings,
            "base_model_name_or_path": base_model_name_or_path,
        }
        with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf8") as f:
            json.dump(config, f, indent=2)
        extra = f" + modules_to_save {modules_to_save}" if modules_to_save else ""
        print(f" -- saved native QLoRA adapter ({len(target_leaves)} target types"
              f"{extra}) to {directory}")

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

        # Restore fully-trained embed/head if present and currently enabled.
        ms_path = os.path.join(directory, "modules_to_save.safetensors")
        if os.path.exists(ms_path) and (self.embed_weight is not None
                                        or self.head_weight is not None):
            ms = load_file(ms_path)
            with torch.no_grad():
                if self.embed_weight is not None and "model.embed_tokens.weight" in ms:
                    e = ms["model.embed_tokens.weight"]            # [V, d]
                    self.embed_weight.copy_(e.to(self.embed_weight.dtype).to(self.embed_weight.device))
                if self.head_weight is not None and "lm_head.weight" in ms:
                    h = ms["lm_head.weight"].t()                   # [V,d] -> [d,V]
                    self.head_weight.copy_(h.to(self.head_weight.dtype).to(self.head_weight.device))
            print(f" -- resumed modules_to_save from {directory}")

        print(f" -- resumed {loaded} adapters from {directory} (optimizer state not restored)")
        return loaded
