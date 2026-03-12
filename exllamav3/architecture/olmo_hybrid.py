from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    Linear,
    GatedMLP,
)
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence
from ..modules.olmo_gated_delta_net import OlmoGatedDeltaNet


def read_olmo_hybrid_layer_types(
    config: Config,
    num_layers: int,
    full_attention_interval: int,
) -> list[str]:
    layer_types = config.read_cfg(list, "layer_types", None)
    if layer_types is not None:
        assert len(layer_types) == num_layers, \
            "Length of layer_types key doesn't match number of hidden layers"
        for t in layer_types:
            if t not in ["linear_attention", "full_attention"]:
                raise ValueError(f"Unknown layer type in layer_types: {t}")
        return layer_types

    return [
        "full_attention" if (idx + 1) % full_attention_interval == 0 else "linear_attention"
        for idx in range(num_layers)
    ]


class OlmoHybridConfig(Config):
    arch_string = "OlmoHybridForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": OlmoHybridModel},
            **kwargs
        )

        # Attention params
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.full_attention_interval = self.read_cfg(int, "full_attention_interval", 4)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params (gated delta net)
        self.linear_conv_kernel_dim = self.read_cfg(int, "linear_conv_kernel_dim", 4)
        self.linear_num_key_heads = self.read_cfg(int, "linear_num_key_heads", no_default)
        self.linear_num_value_heads = self.read_cfg(int, "linear_num_value_heads", no_default)
        self.linear_key_head_dim = self.read_cfg(int, "linear_key_head_dim", no_default)
        self.linear_value_head_dim = self.read_cfg(int, "linear_value_head_dim", no_default)
        self.linear_allow_neg_eigval = self.read_cfg(bool, "linear_allow_neg_eigval", False)

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_olmo_hybrid_layer_types(
            self,
            self.num_hidden_layers,
            self.full_attention_interval,
        )

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000.0,
        )


class OlmoHybridModel(Model):
    config_class = OlmoHybridConfig

    def __init__(
        self,
        config: OlmoHybridConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        kp = "model"

        self.modules += [
            Embedding(
                config = config,
                key = f"{kp}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_linear = config.layer_types[idx] == "linear_attention"

            if is_linear:
                # Linear attention layers (GatedDeltaNet):
                #   input_layernorm (pre-attn norm)
                #   linear_attn.*
                #   post_attention_layernorm (pre-mlp norm)
                #   mlp.*
                self.modules += [
                    TransformerBlock(
                        config = config,
                        key = f"{kp}.layers.{idx}",
                        attn_norm = RMSNorm(
                            config = config,
                            key = f"{kp}.layers.{idx}.input_layernorm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        attn = OlmoGatedDeltaNet(
                            config = config,
                            key = f"{kp}.layers.{idx}.linear_attn",
                            layer_idx = idx,
                            hidden_size = config.hidden_size,
                            k_head_dim = config.linear_key_head_dim,
                            v_head_dim = config.linear_value_head_dim,
                            num_k_heads = config.linear_num_key_heads,
                            num_v_heads = config.linear_num_value_heads,
                            rms_norm_eps = config.rms_norm_eps,
                            conv_kernel_size = config.linear_conv_kernel_dim,
                            allow_neg_eigval = config.linear_allow_neg_eigval,
                            qmap = "block.attn",
                            out_dtype = torch.float,
                        ),
                        mlp_norm = RMSNorm(
                            config = config,
                            key = f"{kp}.layers.{idx}.post_attention_layernorm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        mlp = GatedMLP(
                            config = config,
                            key = f"{kp}.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                        ),
                    )
                ]
            else:
                # Full attention layers (post-norm):
                #   self_attn.* (with q_norm, k_norm)
                #   post_attention_layernorm (post-attn norm)
                #   mlp.*
                #   post_feedforward_layernorm (post-mlp norm)
                self.modules += [
                    TransformerBlock(
                        config = config,
                        key = f"{kp}.layers.{idx}",
                        attn = Attention(
                            config = config,
                            key = f"{kp}.layers.{idx}.self_attn",
                            layer_idx = idx,
                            hidden_size = config.hidden_size,
                            head_dim = config.head_dim,
                            num_q_heads = config.num_q_heads,
                            num_kv_heads = config.num_kv_heads,
                            rope_settings = config.rope_settings,
                            sm_scale = None,
                            key_q = "q_proj",
                            key_k = "k_proj",
                            key_v = "v_proj",
                            key_o = "o_proj",
                            qmap = "block.attn",
                            out_dtype = torch.float,
                            q_norm = RMSNorm(
                                config = config,
                                key = f"{kp}.layers.{idx}.self_attn.q_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                span_heads = True,
                            ),
                            k_norm = RMSNorm(
                                config = config,
                                key = f"{kp}.layers.{idx}.self_attn.k_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                span_heads = True,
                            ),
                        ),
                        attn_post_norm = RMSNorm(
                            config = config,
                            key = f"{kp}.layers.{idx}.post_attention_layernorm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        mlp = GatedMLP(
                            config = config,
                            key = f"{kp}.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                        ),
                        mlp_post_norm = RMSNorm(
                            config = config,
                            key = f"{kp}.layers.{idx}.post_feedforward_layernorm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                    )
                ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{kp}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{kp}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({"recurrent_states": True})
        self.caps.update({"supports_tp": False})

        self.g_rope = RoPE("cpu", config.rope_settings)

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p
