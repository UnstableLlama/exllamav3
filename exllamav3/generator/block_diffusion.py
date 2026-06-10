from __future__ import annotations
import json
import os
from dataclasses import dataclass, replace
from typing import Callable

import torch

from ..model.model import Model
from ..cache.cache import Cache

"""
Standalone block-diffusion generation loop for diffusion language models (DiffusionGemma).

Generation alternates between two uses of the same weight-tied transformer stack:

  1. Encoder passes: ordinary causal prefill of new tokens (the prompt, then each finished canvas),
     extending the KV cache.
  2. Denoising passes: a canvas of canvas_length tokens, initialized with uniform-random token ids, is
     repeatedly forwarded in decoder mode (bidirectional within the canvas, read-only attention to the
     cache). After each pass, low-entropy tokens from the model's proposal are accepted and the rest are
     re-noised, until the canvas is stable and confident or max_denoising_steps is reached.

The denoising forward writes the canvas K/V into the not-yet-committed page(s) past the cache's committed
length. Each denoising step simply overwrites this scratch region, and the encoder pass that commits the
finished canvas overwrites it with the final values, so the committed cache contents always correspond to
encoder-mode (causal) passes only.

Faithful port of the reference loop in transformers' generation_diffusion_gemma.py (EntropyBoundSampler,
LinearTemperatureScheduleLogitsProcessor, StableAndConfidentStoppingCriteria).
"""


@dataclass
class BlockDiffusionSettings:
    # Maximum denoising steps per canvas
    max_denoising_steps: int = 48
    # Entropy bound for the EB sampler; higher accepts more tokens per step (arXiv:2505.24857)
    entropy_bound: float = 0.1
    # Linear temperature schedule, from t_max (first step) to t_min (last step)
    t_min: float = 0.4
    t_max: float = 0.8
    # Early stopping: argmax canvas unchanged for this many steps...
    stability_threshold: int = 1
    # ...and mean token entropy of the temperature-scaled logits below this value
    confidence_threshold: float = 0.005

    @staticmethod
    def from_directory(directory: str) -> "BlockDiffusionSettings":
        """
        Read defaults from a model directory's generation_config.json, where present, falling back to the
        reference defaults for any unset key.
        """
        settings = BlockDiffusionSettings()
        path = os.path.join(directory, "generation_config.json")
        if not os.path.exists(path):
            return settings
        with open(path, encoding = "utf8") as f:
            gc = json.load(f)
        overrides = {}
        for key in ("max_denoising_steps", "t_min", "t_max", "stability_threshold", "confidence_threshold"):
            if gc.get(key) is not None:
                overrides[key] = gc[key]
        sampler_config = gc.get("sampler_config")
        if isinstance(sampler_config, dict) and sampler_config.get("entropy_bound") is not None:
            overrides["entropy_bound"] = sampler_config["entropy_bound"]
        return replace(settings, **overrides)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Entropy (nats) of the categorical distribution over the last dim, computed stably in fp32. Same
    formulation as torch.distributions.Categorical.entropy().
    """
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim = -1)
    log_probs = torch.clamp(log_probs, min = torch.finfo(log_probs.dtype).min)
    return -(log_probs.exp() * log_probs).sum(dim = -1)


def eb_accept_mask(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    """
    Entropy-bound acceptance: accept the k lowest-entropy positions such that

        sum_i^k entropy_i - max(entropy_1, ..., entropy_k) <= entropy_bound

    which bounds the joint mutual information between the accepted tokens, so they can be sampled
    independently in one step.

    :param entropy:
        Per-position entropy, shape (bsz, seq_len)
    :return:
        Boolean acceptance mask, shape (bsz, seq_len). At least one position (the lowest-entropy one) is
        always accepted.
    """
    sorted_entropy, sorted_indices = torch.sort(entropy, dim = -1, descending = False)
    cumulative = torch.cumsum(sorted_entropy, dim = -1)
    # sorted_entropy is also the running maximum, since the sort is ascending
    selected_sorted = (cumulative - sorted_entropy) <= entropy_bound
    mask = torch.zeros_like(selected_sorted)
    mask.scatter_(dim = -1, index = sorted_indices, src = selected_sorted)
    return mask


class BlockDiffusionGenerator:
    """
    Minimal single-sequence generator for block-diffusion models. Operates directly on a Model and Cache,
    outside the batching Generator/Job framework.
    """

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer = None,
        settings: BlockDiffusionSettings | None = None,
    ):
        """
        :param model:
            Loaded model instance with the "block_diffusion" capability (e.g. DiffusionGemmaModel). Load
            with max_output_size >= the model's canvas length so the autosplit loader reserves space for
            full-canvas logits.

        :param cache:
            Cache with max_num_tokens >= prompt length + max_new_tokens, rounded up to a multiple of the
            page size, plus one extra page of scratch space for the canvas being denoised.

        :param tokenizer:
            Optional tokenizer. When provided, generate() also returns decoded text.

        :param settings:
            Sampling settings. If None, defaults are read from the model directory's generation_config.json.
        """
        assert model.caps.get("block_diffusion"), \
            f"{model.config.architecture} is not a block diffusion model"
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.config = model.config
        self.canvas_length = model.caps.get("canvas_length")
        self.settings = settings or BlockDiffusionSettings.from_directory(model.config.directory)
        self.vocab_size = model.config.vocab_size

    def _base_params(self) -> dict:
        return {
            "attn_mode": "flash_attn",
            "cache": self.cache,
            "batch_shape": (1, self.cache.max_num_tokens),
        }

    def _prefill(self, input_ids: torch.Tensor, past_len: int, max_chunk_size: int = 2048):
        seq_len = input_ids.shape[-1]
        for a in range(0, seq_len, max_chunk_size):
            b = min(a + max_chunk_size, seq_len)
            params = self._base_params()
            params["past_len"] = past_len + a
            self.model.prefill(input_ids[:, a:b], params)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        prompt: str | None = None,
        max_new_tokens: int = 256,
        settings: BlockDiffusionSettings | None = None,
        eos_token_ids: list[int] | None = None,
        seed: int | None = None,
        on_canvas: Callable[[torch.Tensor], None] | None = None,
        on_draft: Callable[[int, torch.Tensor], None] | None = None,
    ) -> dict:
        """
        Generate a completion.

        :param input_ids:
            Prompt token ids, shape (1, seq_len). Mutually exclusive with prompt.

        :param prompt:
            Raw prompt string, tokenized with the generator's tokenizer (special tokens enabled).

        :param max_new_tokens:
            Generation stops after the canvas containing this many new tokens; the returned sequence is
            trimmed to at most this many new tokens.

        :param settings:
            Optional per-call settings override.

        :param eos_token_ids:
            Token ids ending the sequence. Defaults to the model config's eos_token_id list. Tokens in the
            final canvas after the first EOS are discarded.

        :param seed:
            Optional RNG seed for canvas initialization, re-noising and sampling.

        :param on_canvas:
            Called with the (trimmed) new token ids, shape (1, n), after each canvas is finalized.

        :param on_draft:
            Called after every denoising step with (step_index, argmax canvas ids). The argmax canvas is the
            current best estimate of the block, as streamed by HF's TextDiffusionStreamer drafts.

        :return:
            dict with:
                "sequence_ids": full sequence including prompt, shape (1, n)
                "new_ids": generated tokens only, shape (1, n)
                "text": decoded completion (when a tokenizer is available)
                "eos": True if generation ended with an EOS token
                "num_canvases": number of canvases generated
                "denoising_steps": total denoising (decoder forward) passes
                "tokens_per_forward": generated tokens / denoising steps
        """
        settings = settings or self.settings
        assert (input_ids is None) != (prompt is None), "Specify either input_ids or prompt"
        if prompt is not None:
            assert self.tokenizer is not None, "Prompt string given but generator has no tokenizer"
            input_ids = self.tokenizer.encode(prompt, encode_special_tokens = True)
        assert input_ids.shape[0] == 1, "BlockDiffusionGenerator requires batch size 1"

        if eos_token_ids is None:
            eos_token_ids = self.config.eos_token_id_list
        eos_token_ids = [e for e in (eos_token_ids or []) if e is not None]

        canvas_length = self.canvas_length
        device = self.model.modules[-1].device
        rng = torch.Generator(device = device)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()
        eos_tensor = torch.tensor(eos_token_ids, dtype = torch.long, device = device)

        # Prefill prompt (encoder mode, causal)
        prompt_len = input_ids.shape[-1]
        assert prompt_len + canvas_length <= self.cache.max_num_tokens, \
            f"Cache too small for prompt of {prompt_len} tokens plus one canvas"
        self._prefill(input_ids, past_len = 0)
        committed_len = prompt_len

        sequence_ids = input_ids.to(device)
        new_ids = torch.empty((1, 0), dtype = torch.long, device = device)
        uncommitted_canvas = None
        finished = False
        found_eos = False
        num_canvases = 0
        total_steps = 0

        while not finished:
            # Commit the previous canvas to the cache (encoder mode, causal)
            if uncommitted_canvas is not None:
                if committed_len + 2 * canvas_length > self.cache.max_num_tokens:
                    break
                self._prefill(uncommitted_canvas, past_len = committed_len)
                committed_len += canvas_length
                uncommitted_canvas = None

            # Denoising loop for the next canvas
            canvas = torch.randint(
                low = 0, high = self.vocab_size, size = (1, canvas_length),
                generator = rng, device = device,
            )
            argmax_canvas = canvas
            sc_logits = None
            argmax_history = torch.full(
                (max(settings.stability_threshold, 1), 1, canvas_length), -1,
                dtype = torch.long, device = device,
            )

            for cur_step in reversed(range(1, settings.max_denoising_steps + 1)):
                total_steps += 1

                params = self._base_params()
                params.update({
                    "past_len": committed_len,
                    "diffusion_decode": True,
                    "self_conditioning_logits": sc_logits,
                })
                logits = self.model.forward(canvas, params).float()

                # Linear temperature schedule; cur_step counts down, so temperature anneals t_max -> t_min
                temperature = settings.t_min + (
                    (settings.t_max - settings.t_min) * (cur_step / settings.max_denoising_steps)
                )
                logits = logits / temperature
                logits = logits.to(device)

                entropy = token_entropy(logits)
                probs = torch.softmax(logits, dim = -1)
                proposal = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples = 1, generator = rng
                ).view(1, canvas_length)
                argmax_canvas = logits.argmax(dim = -1)

                # Accept approximately-independent low-entropy tokens, re-noise the rest
                accept = eb_accept_mask(entropy, settings.entropy_bound)
                renoise = torch.randint(
                    low = 0, high = self.vocab_size, size = (1, canvas_length),
                    generator = rng, device = device,
                )
                canvas = torch.where(accept, proposal, renoise)

                # Self-conditioning input for the next step is this step's temperature-scaled logits
                sc_logits = logits.half()

                if on_draft is not None:
                    on_draft(settings.max_denoising_steps - cur_step, argmax_canvas)

                # Adaptive stopping: stable argmax canvas and confident (low mean entropy)
                if settings.stability_threshold == 0:
                    stable = True
                else:
                    stable = bool((argmax_history == argmax_canvas.unsqueeze(0)).all())
                    argmax_history = torch.roll(argmax_history, shifts = -1, dims = 0)
                    argmax_history[-1] = argmax_canvas
                confident = entropy.mean().item() < settings.confidence_threshold
                if stable and confident:
                    break

            num_canvases += 1
            canvas_ids = argmax_canvas

            # Finalize: cut at the first EOS token, enforce max_new_tokens
            is_eos = torch.isin(canvas_ids, eos_tensor)
            keep = canvas_length
            if is_eos.any():
                keep = int(is_eos[0].nonzero()[0].item()) + 1
                finished = True
                found_eos = True
            if new_ids.shape[-1] + keep >= max_new_tokens:
                keep = min(keep, max_new_tokens - new_ids.shape[-1])
                finished = True

            kept_ids = canvas_ids[:, :keep]
            new_ids = torch.cat((new_ids, kept_ids), dim = -1)
            sequence_ids = torch.cat((sequence_ids, kept_ids), dim = -1)
            uncommitted_canvas = canvas_ids

            if on_canvas is not None:
                on_canvas(kept_ids)

        result = {
            "sequence_ids": sequence_ids.cpu(),
            "new_ids": new_ids.cpu(),
            "eos": found_eos,
            "num_canvases": num_canvases,
            "denoising_steps": total_steps,
            "tokens_per_forward": new_ids.shape[-1] / max(total_steps, 1),
        }
        if self.tokenizer is not None:
            text = self.tokenizer.decode(new_ids.cpu(), decode_special_tokens = False)
            result["text"] = text[0] if isinstance(text, list) else text
        return result
