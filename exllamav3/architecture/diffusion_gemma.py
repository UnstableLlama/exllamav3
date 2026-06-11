from __future__ import annotations
import torch
from typing_extensions import override
from .gemma4 import Gemma4Config, Gemma4TextModel, Gemma4VisionModel, _prepare_noncausal_mm_spans
from ..modules.arch_specific.diffusion_gemma import DiffusionGemmaSelfConditioning
from ..modules.attn import prepare_for_attn

# DiffusionGemma is a Gemma4 derivative that generates text by block diffusion: blocks ("canvases") of
# canvas_length tokens are iteratively denoised by the decoder, which attends bidirectionally within the
# canvas and reads (but never updates) the KV cache built by the causal encoder. Encoder and decoder share
# all transformer weights; the decoder additionally has a small self-conditioning block. See
# exllamav3/generator/block_diffusion.py for the sampling loop.
#
# Reference: https://huggingface.co/docs/transformers/model_doc/diffusion_gemma

class DiffusionGemmaConfig(Gemma4Config):
    arch_string = "DiffusionGemmaForBlockDiffusion"

    @override
    def get_model_classes(self):
        return {"text": DiffusionGemmaModel, "vision": DiffusionGemmaVisionModel}

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        # The checkpoint stores one copy of the weights tied between the HF encoder and decoder submodels.
        # Probe for the stored prefix before the parent constructor needs it.
        super().__init__(directory, **kwargs)

        # DiffusionGemma has no enable_moe_block key; every layer has the parallel dense MLP + routed
        # experts block
        self.enable_moe_block = True
        assert self.num_experts > 0, "DiffusionGemma requires text_config->num_experts"
        assert self.num_experts_per_tok > 0, "DiffusionGemma requires text_config->top_k_experts"
        assert self.moe_intermediate_size > 0, "DiffusionGemma requires text_config->moe_intermediate_size"

        # Full-attention layers have no v_proj; values are the K projection (with V-norm applied). This is
        # unconditional in DiffusionGemma, with no attention_k_eq_v key
        self.attention_k_eq_v = True

        # Block diffusion parameters
        self.canvas_length = self.read_cfg(int, "canvas_length", 256)

        # Locate the text stack in the checkpoint (tied weights are stored once)
        self.text_key_prefix = None
        for prefix in ("model.decoder", "model.encoder.language_model", "model.language_model"):
            if self.stc.has_tensor(f"{prefix}.embed_tokens.weight"):
                self.text_key_prefix = prefix
                break
        assert self.text_key_prefix is not None, \
            "Cannot locate text model tensors (tried model.decoder, model.encoder.language_model)"

        self.sc_key_prefix = None
        for prefix in (f"{self.text_key_prefix}.self_conditioning", "model.decoder.self_conditioning"):
            if self.stc.has_tensor(f"{prefix}.gate_proj.weight"):
                self.sc_key_prefix = prefix
                break
        assert self.sc_key_prefix is not None, \
            "Cannot locate self_conditioning tensors in checkpoint"

        for prefix in ("model.encoder", "model"):
            if len(self.stc.list_tensors(prefix = f"{prefix}.vision_tower")):
                self.vision_key_prefix = prefix
                break
        else:
            self.vision_key_prefix = "model.encoder"


class DiffusionGemmaModel(Gemma4TextModel):
    config_class = DiffusionGemmaConfig

    def __init__(
        self,
        config: DiffusionGemmaConfig,
        swa_full: bool = True,
        **kwargs
    ):
        # swa_full is accepted (model_init passes it for all architectures) but always forced on:
        # sliding-window layers must use the regular paged cache rather than SWA recurrent states, so
        # that repeated denoising passes over uncommitted canvas positions never mutate rolling window
        # state
        super().__init__(
            config,
            key_prefix = config.text_key_prefix,
            swa_full = True,
            **kwargs
        )

        # The soft-embedding projection in the self-conditioning block needs the embedding table on-device
        embedding = self.modules[0]
        embedding.caps["prefer_cpu"] = False

        self_conditioning = DiffusionGemmaSelfConditioning(
            config = config,
            key = config.sc_key_prefix,
            hidden_size = config.hidden_size,
            intermediate_size = config.intermediate_size,
            rms_norm_eps = config.rms_norm_eps,
            embedding = embedding,
        )
        self.modules.insert(1, self_conditioning)
        self.first_block_idx += 1
        self.last_kv_module_idx += 1
        self.logit_layer_idx += 1

        self.caps.update({
            "block_diffusion": True,
            "canvas_length": config.canvas_length,
        })

    @override
    def load_gen(self, *args, **kwargs):
        # Denoising passes produce logits for the whole canvas, so make sure the loader reserves output
        # buffers for at least canvas_length positions regardless of how the model is loaded
        kwargs["max_output_size"] = max(kwargs.get("max_output_size") or 0, self.config.canvas_length)
        kwargs["max_output_factor"] = max(kwargs.get("max_output_factor") or 1, 2)
        yield from super().load_gen(*args, **kwargs)

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if params.get("diffusion_decode"):
            # Denoising pass: the canvas attends bidirectionally to itself and to the entire cache (full
            # attention layers) or the trailing window of the cache (sliding layers)
            seqlen = input_ids.shape[-1]
            params["non_causal_spans"] = [(0, seqlen, True)]
        else:
            # Encoder passes (prefill and canvas commit) are causal, except for image spans
            _prepare_noncausal_mm_spans(input_ids, params)
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<bos>"
        if system_prompt:
            p += f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n"
        else:
            p += f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
        p += "<start_of_turn>model\n"
        return p


class DiffusionGemmaVisionModel(Gemma4VisionModel):
    pass
