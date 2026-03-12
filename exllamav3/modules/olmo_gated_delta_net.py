from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm
from .gated_delta_net import (
    GDN_RecurrentState,
    prepare_for_recurrence,
    causal_conv1d_update_function,
    causal_conv1d_fwd_function,
    fused_recurrent_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ModuleNotFoundError:
    chunk_gated_delta_rule = None

try:
    from ..ext import exllamav3_ext as ext
except ImportError:
    ext = None


class OlmoGatedDeltaNet(Module):
    """
    Gated Delta Net module for OLMo Hybrid.

    OLMo uses separate projections and convolutions:
        q_proj, k_proj, v_proj     (separate linear projections)
        g_proj                     (gate, equivalent to Qwen's in_proj_z)
        b_proj, a_proj             (beta/alpha)
        q_conv1d, k_conv1d, v_conv1d  (separate depthwise convolutions)
        o_norm                     (output norm)
        o_proj                     (output projection)
        A_log, dt_bias             (recurrence parameters)

    The forward pass math is identical to the standard GatedDeltaNet —
    only the weight layout and projection steps differ.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        allow_neg_eigval: bool = False,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "OlmoGatedDeltaNet"

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.allow_neg_eigval = allow_neg_eigval
        self.out_dtype = out_dtype

        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads

        # Separate q/k/v projections
        self.q_proj = Linear(
            config, f"{key}.q_proj", hidden_size, self.k_dim,
            qmap = qmap + ".input", out_dtype = torch.float
        )
        self.k_proj = Linear(
            config, f"{key}.k_proj", hidden_size, self.k_dim,
            qmap = qmap + ".input", out_dtype = torch.float
        )
        self.v_proj = Linear(
            config, f"{key}.v_proj", hidden_size, self.v_dim,
            qmap = qmap + ".input", out_dtype = torch.float
        )
        self.register_submodule(self.q_proj)
        self.register_submodule(self.k_proj)
        self.register_submodule(self.v_proj)

        # Gate projection (g_proj in OLMo, equivalent to in_proj_z in Qwen)
        self.g_proj = Linear(
            config, f"{key}.g_proj", hidden_size, self.v_dim,
            qmap = qmap + ".input", out_dtype = torch.float
        )
        self.register_submodule(self.g_proj)

        # Beta and alpha projections
        self.b_proj = Linear(
            config, f"{key}.b_proj", hidden_size, self.num_v_heads,
            qmap = None, out_dtype = torch.float, pad_to = 1
        )
        self.a_proj = Linear(
            config, f"{key}.a_proj", hidden_size, self.num_v_heads,
            qmap = None, out_dtype = torch.float, pad_to = 1
        )
        self.register_submodule(self.b_proj)
        self.register_submodule(self.a_proj)

        # Output projection
        self.o_proj = Linear(
            config, f"{key}.o_proj",
            self.v_head_dim * self.num_v_heads, hidden_size,
            qmap = qmap + ".output", out_dtype = self.out_dtype
        )
        self.register_submodule(self.o_proj)

        # Output norm (o_norm in OLMo, equivalent to norm in Qwen)
        self.norm = GatedRMSNorm(
            config, f"{key}.o_norm", self.rms_norm_eps, out_dtype = torch.half
        )
        self.register_submodule(self.norm)

        # Non-Linear parameters loaded directly from safetensors
        self.a_log = None
        self.dt_bias = None
        self.q_conv1d_weight = None
        self.q_conv1d_bias = None
        self.k_conv1d_weight = None
        self.k_conv1d_bias = None
        self.v_conv1d_weight = None
        self.v_conv1d_bias = None

        self.key_a_log = f"{key}.A_log"
        self.key_dt_bias = f"{key}.dt_bias"
        self.key_q_conv1d_weight = f"{key}.q_conv1d.weight"
        self.key_q_conv1d_bias = f"{key}.q_conv1d.bias"
        self.key_k_conv1d_weight = f"{key}.k_conv1d.weight"
        self.key_k_conv1d_bias = f"{key}.k_conv1d.bias"
        self.key_v_conv1d_weight = f"{key}.v_conv1d.weight"
        self.key_v_conv1d_bias = f"{key}.v_conv1d.bias"

        # Total dim for the fused q+k+v used by conv and recurrence
        self.conv_dim = self.k_dim + self.k_dim + self.v_dim

        self.caps.update({
            "recurrent_cache": True
        })

    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)
        self.a_log = self.config.stc.get_tensor(
            self.key_a_log, self.device, optional = False, allow_bf16 = True
        )
        self.dt_bias = self.config.stc.get_tensor(
            self.key_dt_bias, self.device, optional = False, allow_bf16 = True
        )
        self.q_conv1d_weight = self.config.stc.get_tensor(
            self.key_q_conv1d_weight, self.device, optional = False, allow_bf16 = True
        )
        self.q_conv1d_bias = self.config.stc.get_tensor(
            self.key_q_conv1d_bias, self.device, optional = True, allow_bf16 = True
        )
        self.k_conv1d_weight = self.config.stc.get_tensor(
            self.key_k_conv1d_weight, self.device, optional = False, allow_bf16 = True
        )
        self.k_conv1d_bias = self.config.stc.get_tensor(
            self.key_k_conv1d_bias, self.device, optional = True, allow_bf16 = True
        )
        self.v_conv1d_weight = self.config.stc.get_tensor(
            self.key_v_conv1d_weight, self.device, optional = False, allow_bf16 = True
        )
        self.v_conv1d_bias = self.config.stc.get_tensor(
            self.key_v_conv1d_bias, self.device, optional = True, allow_bf16 = True
        )
        self.norm.load(device, **kwargs)

    @override
    def unload(self):
        self.a_log = None
        self.dt_bias = None
        self.q_conv1d_weight = None
        self.q_conv1d_bias = None
        self.k_conv1d_weight = None
        self.k_conv1d_bias = None
        self.v_conv1d_weight = None
        self.v_conv1d_bias = None
        self.norm.unload()
        super().unload()

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        bsz, seqlen, _ = x.shape

        # Previous state
        rs = params.get("recurrent_states")
        if rs is not None:
            rs = rs[self.layer_idx]
            conv_state = rs.last_conv_state if rs.last_conv_state is not None else \
                torch.zeros(
                    (bsz, self.conv_dim, self.conv_kernel_size),
                    dtype = torch.bfloat16, device = x.device
                )
            recurrent_state = rs.last_recurrent_state if rs.last_recurrent_state is not None else \
                torch.zeros(
                    (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float, device = self.device
                )
            save_state = True
        else:
            conv_state = None
            recurrent_state = None
            save_state = False

        # --- Projections (separate, OLMo style) ---
        q = self.q_proj.forward(x, params)[..., :self.k_dim]                           # [bsz, seq, k_dim]
        k = self.k_proj.forward(x, params)[..., :self.k_dim]                           # [bsz, seq, k_dim]
        v = self.v_proj.forward(x, params)[..., :self.v_dim]                           # [bsz, seq, v_dim]
        z = self.g_proj.forward(x, params)[..., :self.v_dim]                           # [bsz, seq, v_dim]
        z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)                     # gate
        b = self.b_proj.forward(x, params)                                              # [bsz, seq, num_v_heads]
        a = self.a_proj.forward(x, params)                                              # [bsz, seq, num_v_heads]

        # Compute beta and g from b, a, dt_bias, A_log
        # (same math as gated_delta_net_fused_op_2)
        beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
        g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

        if ext is not None:
            ext.gated_delta_net_fused_op_2(
                b, a,
                self.dt_bias,
                self.a_log,
                beta, g
            )
        else:
            # Fallback: manual computation
            beta.copy_(torch.sigmoid(b).to(torch.bfloat16))
            dt = F.softplus(a + self.dt_bias.unsqueeze(0).unsqueeze(0))
            g.copy_(-dt * self.a_log.float().exp().unsqueeze(0).unsqueeze(0))

        # Scale beta for negative eigenvalue mode (allows range [0, 2] instead of [0, 1])
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # --- Convolutions (separate per q/k/v, OLMo style) ---
        # Concatenate for conv state tracking, then split back
        # Shape: [bsz, q_dim+k_dim+v_dim, seq]
        mixed_qkv = torch.cat([q, k, v], dim = -1).transpose(1, 2).to(torch.bfloat16).contiguous()

        # Fuse the three conv1d weights into one for the depthwise conv
        # This is mathematically equivalent to three separate convolutions
        # because it's depthwise (groups=channels)
        fused_conv_weight = torch.cat(
            [self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight],
            dim = 0
        ).squeeze(1)

        fused_conv_bias = None
        if self.q_conv1d_bias is not None and self.k_conv1d_bias is not None and self.v_conv1d_bias is not None:
            fused_conv_bias = torch.cat([self.q_conv1d_bias, self.k_conv1d_bias, self.v_conv1d_bias], dim = 0)

        if conv_state is None:
            if save_state:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                rs.last_conv_state = conv_state
            mixed_qkv = causal_conv1d_fwd_function(
                mixed_qkv,
                fused_conv_weight,
                fused_conv_bias,
            )
        else:
            mixed_qkv = causal_conv1d_update_function(
                mixed_qkv,
                conv_state,  # updated inplace
                fused_conv_weight,
                fused_conv_bias,
            )

        # --- Recurrence (identical to standard GatedDeltaNet) ---

        # Use chunked rule when advantageous and available
        if seqlen >= self.num_v_heads and chunk_gated_delta_rule is not None:
            mixed_qkv = mixed_qkv.transpose(1, 2)

            q_out, k_out, v_out = torch.split(
                mixed_qkv, [self.k_dim, self.k_dim, self.v_dim], dim = -1
            )
            q_out = q_out.view(bsz, seqlen, -1, self.k_head_dim)
            k_out = k_out.view(bsz, seqlen, -1, self.k_head_dim)
            v_out = v_out.view(bsz, seqlen, -1, self.v_head_dim)

            # Grouped attn
            if self.num_v_groups > 1:
                q_out = q_out.repeat_interleave(self.num_v_groups, dim = 2)
                k_out = k_out.repeat_interleave(self.num_v_groups, dim = 2)

            core_attn_out, recurrent_state = chunk_gated_delta_rule(
                q_out, k_out, v_out,
                g = g,
                beta = beta,
                initial_state = recurrent_state,
                output_final_state = save_state,
                use_qk_l2norm_in_kernel = True,
            )

        else:
            mixed_qkv = mixed_qkv.transpose(1, 2)
            q_out, k_out, v_out = torch.split(
                mixed_qkv, [self.k_dim, self.k_dim, self.v_dim], dim = -1
            )
            q_out = q_out.view(bsz, seqlen, -1, self.k_head_dim)
            k_out = k_out.view(bsz, seqlen, -1, self.k_head_dim)
            v_out = v_out.view(bsz, seqlen, -1, self.v_head_dim)

            if self.num_v_groups > 1:
                q_out = q_out.repeat_interleave(self.num_v_groups, dim = 2)
                k_out = k_out.repeat_interleave(self.num_v_groups, dim = 2)

            if recurrent_state is None:
                recurrent_state = torch.zeros(
                    (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float,
                    device = self.device
                )

            # Use FLA fused recurrent kernel instead of CUDA kernel, which has
            # alignment requirements (k_head_dim must be divisible by 64) that
            # OLMo's k_head_dim=96 does not satisfy
            core_attn_out, recurrent_state = fused_recurrent_gated_delta_rule(
                q_out, k_out, v_out, g, beta,
                recurrent_state, save_state,
                use_qk_l2norm_in_kernel = True,
            )

        # --- Norm + output projection ---
        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)
        x = self.o_proj.forward(core_attn_out, params)

        # Update cache
        if save_state:
            rs.last_recurrent_state = recurrent_state
            rs.last_conv_state = conv_state
            if not rs.batched:
                rs.position += seqlen
            else:
                rs.positions = [r + seqlen for r in rs.positions]

        return to2(x, out_dtype, self.out_dtype)

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.q_conv1d_weight, self.key_q_conv1d_weight),
            (self.q_conv1d_bias, self.key_q_conv1d_bias),
            (self.k_conv1d_weight, self.key_k_conv1d_weight),
            (self.k_conv1d_bias, self.key_k_conv1d_bias),
            (self.v_conv1d_weight, self.key_v_conv1d_weight),
            (self.v_conv1d_bias, self.key_v_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t

    def new_recurrent_state(self):
        return GDN_RecurrentState()

    @override
    def optimizer_targets(self):
        return [[
            self.q_proj.optimizer_targets(),
            self.k_proj.optimizer_targets(),
            self.v_proj.optimizer_targets(),
        ]]

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()

    def tp_export(self, plan, producer):
        raise NotImplementedError()

    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError()
