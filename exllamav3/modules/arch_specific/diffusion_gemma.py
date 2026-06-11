from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ...model.config import Config
from .. import Module, Linear, RMSNorm, Embedding


class DiffusionGemmaSelfConditioning(Module):
    """
    Self-conditioning block for the DiffusionGemma diffusion decoder. Converts the logits from the previous
    denoising step into soft embeddings (softmax over the vocabulary, projected through the tied embedding
    table), passes them through a gated MLP and adds the result to the canvas input embeddings.

    Only active when params["diffusion_decode"] is set. In encoder mode (prefill/commit passes, calibration)
    the module is an identity function, matching the HF reference in which the encoder has no
    self-conditioning path.

    Note that with no self-conditioning signal (the first denoising step), the input embeddings still pass
    through the final unweighted RMSNorm.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        embedding: Embedding,
        out_dtype: torch.dtype | None = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "SelfConditioning"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_dtype = out_dtype

        # Tied embedding table used to compute soft embeddings. Owned by the model, not registered here
        self.embedding = embedding

        self.pre_norm = RMSNorm(
            config = config,
            key = f"{key}.pre_norm",
            rms_norm_eps = rms_norm_eps,
        )
        self.post_norm = RMSNorm(
            config = config,
            key = f"{key}.post_norm",
            rms_norm_eps = rms_norm_eps,
            unweighted = True,
        )
        self.gate_proj = Linear(
            config = config,
            key = f"{key}.gate_proj",
            in_features = hidden_size,
            out_features = intermediate_size,
        )
        self.up_proj = Linear(
            config = config,
            key = f"{key}.up_proj",
            in_features = hidden_size,
            out_features = intermediate_size,
        )
        self.down_proj = Linear(
            config = config,
            key = f"{key}.down_proj",
            in_features = intermediate_size,
            out_features = hidden_size,
            allow_input_padding = True,
        )

        self.register_submodule(self.pre_norm)
        self.register_submodule(self.post_norm)
        self.register_submodule(self.gate_proj)
        self.register_submodule(self.up_proj)
        self.register_submodule(self.down_proj)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if not params.get("diffusion_decode"):
            return x

        sc_logits = params.get("self_conditioning_logits")
        if sc_logits is not None:
            bsz, seqlen, _ = x.shape
            weight = self.embedding.embedding.weight
            probs = torch.softmax(sc_logits.float(), dim = -1)
            probs = probs.to(device = weight.device, dtype = weight.dtype)
            soft_emb = probs.view(-1, probs.shape[-1]) @ weight
            soft_emb = soft_emb.view(bsz, seqlen, self.hidden_size).to(x.device)
            soft_emb *= self.embedding.multiplier

            normed = self.pre_norm.forward_torch(soft_emb, params, out_dtype = torch.half)
            g = self.gate_proj.forward(normed, params)
            u = self.up_proj.forward(normed, params)
            sc = self.down_proj.forward(F.gelu(g, approximate = "tanh") * u, params)
            y = x + sc.to(x.dtype)
        else:
            # Zero signal: the gated MLP contributes nothing, but the post-norm still applies
            y = x

        return self.post_norm.forward_torch(y, params, out_dtype = out_dtype or self.out_dtype)
