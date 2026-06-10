# Plan: DiffusionGemma (block text diffusion) support in exllamav3

*Research notes and implementation plan, 2026-06-10.*

DiffusionGemma (`google/diffusiongemma-26B-A4B-it`, Apache 2.0, released 2026-06-10) is Google's
open-weights block text diffusion model: a 26B-total / ~4B-active MoE that generates 256-token
blocks ("canvases") in parallel by iterative denoising, claiming up to ~4x faster generation than
comparable AR models (1000+ t/s on H100, 700+ t/s on RTX 5090, ~18 GB VRAM quantized).

The headline finding of this research: **DiffusionGemma is a Gemma 4 derivative**, and exllamav3
already implements nearly the entire weight-bearing architecture in
`exllamav3/architecture/gemma4.py`. The new work is concentrated in (1) a thin architecture
subclass, (2) one small new module (self-conditioning), and (3) a block-diffusion sampling loop —
for which exllamav3's attention stack already exposes every primitive needed (`causal=False`,
`non_causal_spans`, position overrides, paged-cache scratch writes).

References (HF `transformers` v5.11, `src/transformers/models/diffusion_gemma/`):
- `modular_diffusion_gemma.py` — model definition (inherits Gemma4 components)
- `generation_diffusion_gemma.py` — block-diffusion generation loop, EB sampler
- `configuration_diffusion_gemma.py` — config schema
- Docs: https://huggingface.co/docs/transformers/model_doc/diffusion_gemma
- EB sampler paper: https://arxiv.org/pdf/2505.24857

---

## 1. How DiffusionGemma works

### 1.1 Weights

`DiffusionGemmaForBlockDiffusion` = one Gemma4-style transformer stack used in two *modes*, plus
two small extras:

- **Shared stack** (encoder and decoder are weight-tied via `_tied_weights_keys`; the checkpoint
  stores one copy, under the decoder prefix — verify exact key names against the safetensors
  index at implementation time):
  - Scaled embeddings (`embed_tokens` × √hidden_size, bf16-rounded like Gemma4).
  - Per layer: `input_layernorm` → attention → `post_attention_layernorm` (+residual), then the
    Gemma4 "MoE block": dense `mlp` (via `pre_feedforward_layernorm`, out through
    `post_feedforward_layernorm_1`) **in parallel with** routed experts (router on the *pre-norm
    residual*, experts via `pre/post_feedforward_layernorm_2`), summed, through
    `post_feedforward_layernorm`, +residual, × `layer_scalar`.
  - Attention: mixed `sliding_attention` / `full_attention` layer types; Q/K/V RMSNorms (V-norm
    unweighted); `sm_scale = 1.0`; separate RoPE settings per layer type; on full-attention
    layers `global_head_dim` (512), `num_global_key_value_heads`, and **K==V** (no `v_proj`;
    values are the K projection with the V-norm applied). Sliding layers have a normal `v_proj`
    and `head_dim` (256), window `sliding_window` (512 default).
  - Final `norm`, `lm_head` (tied to embeddings) with `final_logit_softcapping = 30.0`.
  - Gemma4 features *removed* in DiffusionGemma: per-layer embeddings (PLE), KV-shared layers,
    double-wide MLP, audio/video. MoE is always on.
- **`DiffusionGemmaSelfConditioning`** (decoder-only, untied): gated MLP (gate/up/down) with a
  weighted pre-RMSNorm and *unweighted* post-RMSNorm. Input: soft embeddings computed from the
  previous denoising step's logits: `softmax(logits) @ embed_tokens.weight × embed_scale`.
  Output: `post_norm(inputs_embeds + mlp(pre_norm(soft_emb)))`. On the first denoising step the
  signal is zeros — note the result is **not** identity (the unweighted post-norm still runs), so
  the module must execute in decoder mode even without self-conditioning logits, and must *not*
  execute in encoder mode.
- **Vision tower**: Gemma4's vision encoder + `embed_vision` projector (image-text-to-text),
  under the encoder prefix.

Config schema (`model_type: diffusion_gemma`, `architectures: ["DiffusionGemmaForBlockDiffusion"]`):
`text_config` uses the *same key names Gemma4Config already reads* (`head_dim`,
`global_head_dim`, `num_global_key_value_heads`, `layer_types`, `sliding_window`, `num_experts`,
`top_k_experts`, `moe_intermediate_size`, `rope_parameters->{sliding,full}_attention`,
`final_logit_softcapping`, `use_bidirectional_attention`, …) plus top-level `canvas_length`
(256), `boi/eoi/image_token_id`. Doc defaults: vocab 262144, sliding_window 512, 5:1
sliding:full layer ratio, max_position_embeddings 131072. Exact 26B values (layer count, hidden
size, expert count) must be read from the released `config.json` when implementing.

### 1.2 Generation algorithm (block diffusion)

Outer loop = autoregression over 256-token canvases; inner loop = denoising of one canvas.

1. **Encode (prefill)**: run uncached tokens through the stack in *encoder mode* — ordinary
   causal attention (`use_bidirectional_attention != "all"`) with the usual causal+sliding
   masks — appending K/V to the cache. This is exactly an exllamav3 prefill.
2. **Denoise** the next canvas (up to `max_denoising_steps`, default 48):
   - Canvas starts as **uniform-random token ids** (not mask tokens).
   - *Decoder mode* forward over the 256 canvas tokens: embeddings + self-conditioning;
     **bidirectional self-attention within the canvas, read-only attention to the encoder KV
     cache** (decoder never commits K/V). Positions continue after the cache
     (`cache_len … cache_len+255`). Full-attention layers see the whole cache + canvas;
     sliding layers see the last `sliding_window` cached tokens + the whole canvas.
   - `lm_head` + softcap → logits (canvas_length × vocab).
   - **Linear temperature schedule**: `t = t_min + (t_max−t_min)·(step_remaining/max_steps)`,
     defaults t_max 0.8 → t_min 0.4.
   - Sample a proposal canvas (`multinomial`), keep `argmax` canvas separately.
   - **Entropy-bound (EB) acceptance**: sort positions by token entropy ascending; accept the
     largest k with `cumsum(entropy) − running_max ≤ entropy_bound` (default 0.1); rejected
     positions are **re-noised with fresh uniform-random tokens**.
   - **Adaptive stop** when the argmax canvas is unchanged for `stability_threshold` (1) steps
     *and* mean token entropy < `confidence_threshold` (0.005).
   - This step's processed logits become the next step's self-conditioning signal.
3. **Commit**: append the final `argmax` canvas to the sequence and run it through the
   *encoder* (causal) to extend the KV cache. Scan the canvas for EOS (pad after the first EOS,
   stop if all sequences finished); otherwise go to 2 for the next canvas.

Generation defaults (from `DiffusionGemmaGenerationConfig`): `max_new_tokens 256`,
`max_denoising_steps 48`, `entropy_bound 0.1`, `t_min 0.4`, `t_max 0.8`,
`stability_threshold 1`, `confidence_threshold 0.005`. The released `generation_config.json`
may override these — read it at load time.

Note for `use_bidirectional_attention == "all"` checkpoints: the encoder itself goes
bidirectional *within each encode call* (block-causal overall). exllamav3 expresses this with
`non_causal_spans` + an atomic-prefill cap, both of which already exist for Gemma4 vision.
Whether the released checkpoint sets this must be checked in its `config.json`.

---

## 2. Gap analysis: exllamav3 vs requirements

| Requirement | Status in exllamav3 |
|---|---|
| Gemma4 layer stack (Q/K/V norms, K==V global layers, head_dim 512, sm_scale 1.0, softcaps, layer_scalar, per-layer-type RoPE) | ✅ `architecture/gemma4.py` (`Gemma4TextModel`) |
| Parallel dense-MLP + routed-experts block, router on pre-norm residual, fused `experts.gate_up_proj`/`down_proj` checkpoints, per-expert scale | ✅ `modules/block_sparse_mlp.py` via gemma4's `BlockSparseMLP(..., shared_experts=…, alt_residual_channel=True)` |
| head_dim 512 attention kernels | ✅ bighead scalar / triton paged backends (flash-attn rejects dim > 256) — `modules/attention_fn/` |
| Non-causal attention with paged KV cache | ✅ `params["causal"]=False` and `non_causal_spans` flow through `attn.py:683-754` → `attention_fn/dispatch.py`; span machinery (`common.py:get_non_causal_span_arglist`) computes `causal=False` + window `(max(W,l), l−1)` per span — built for Gemma4 bidirectional vision, directly reusable for the canvas |
| Read-only attention over cache + uncommitted block | ✅ by construction: backends write new K/V at `cache_seqlens` into allocated pages, but committed length (`pagetable` `kv_position`) only advances when the caller says so — repeated denoise passes just overwrite the same scratch region. `PAGE_SIZE` (256) == `canvas_length` (256): the scratch is exactly one page |
| Position overrides for canvas (`cache_len…cache_len+255`) | ✅ `positions`/`position_ids` params (`attn.py:26-40`); default `positions = cache_seqlens` is already correct for the decoder pass |
| Vision tower (Gemma4) | ✅ `Gemma4VisionModel` (needs key-prefix parameterization, see 3.1) |
| Gemma tokenizer (262k vocab) | ✅ |
| Self-conditioning module | ❌ new (small gated MLP + 2 norms) |
| Block-diffusion sampling loop (temperature schedule, EB sampler, renoise, adaptive stop, EOS-in-canvas) | ❌ new (pure PyTorch, ~200 lines) |
| Generator/Job support for multi-token non-AR decode steps | ❌ new (phase 2) |
| Speculative decoding / draft models | n/a — must be disabled for this arch |

The conversion/quantization pipeline needs **no structural changes**: encoder and decoder share
weights, so the stack is quantized once and used in both modes; the conversion calibration
forward (causal) is exactly the encoder mode.

---

## 3. Implementation plan

### Phase 1 — model + standalone block-diffusion generation (the core deliverable)

#### 3.1 Architecture definition — `exllamav3/architecture/diffusion_gemma.py`

- `DiffusionGemmaConfig(Gemma4Config)`, `arch_string = "DiffusionGemmaForBlockDiffusion"`.
  - Reuse the Gemma4 `text_config` parsing wholesale (same keys). Force the MoE path on
    (DiffusionGemma has no `enable_moe_block` flag; experts are always present).
  - Read `canvas_length` (top level, default 256) and `use_bidirectional_attention`.
  - Skip/conditionalize Gemma4-only keys that DiffusionGemma deletes (PLE, KV-sharing).
  - `tie_word_embeddings` default **True**.
- `DiffusionGemmaModel(Gemma4TextModel)` built with:
  - `key_prefix` pointing at the stored text-stack prefix (expected `model.decoder`; confirm
    against the safetensors index — `Gemma4TextModel.__init__` already takes `key_prefix`).
  - `swa_full = True` for v1, so sliding layers live in the ordinary paged cache (window applied
    at attention time) instead of `SlidingAttention`/`SWAState` rolling buffers. This sidesteps
    state mutation during the denoising loop entirely; an SWAState-aware optimization can come
    later (its `stash`/`rewind` API would support it).
  - A new `SelfConditioning` module inserted immediately after `Embedding` (see 3.2).
  - `lm_head` with `alt_key` = embeddings (tied), `softcap = final_logit_softcapping`.
  - `caps`: `{"block_diffusion": True, "canvas_length": N, "supports_tp": False}` initially; no
    draft-model/spec-dec support.
  - `prepare_inputs` override: in decoder mode (a `params["diffusion_decode"] = True` flag) set
    `params["causal"] = False` for full bidirectionality over cache+canvas on full-attention
    layers — or equivalently `params["non_causal_spans"] = [(0, canvas, True)]`, which also
    handles the sliding-window case via the existing span arglist. In encoder mode, behave
    exactly like Gemma4 (and honor `use_bidirectional_attention == "all"` with spans + atomic
    prefill if the released config sets it).
- `DiffusionGemmaVisionModel(Gemma4VisionModel)` with the `model.vision_tower`/`model.embed_vision`
  key prefixes parameterized to the encoder-side names in this checkpoint. (Small refactor of
  `gemma4.py` to take a vision key prefix; default unchanged.)
- Register in `exllamav3/architecture/architectures.py`.

#### 3.2 New module — `exllamav3/modules/arch_specific/diffusion_gemma.py`

`SelfConditioning(Module)` — composed of existing `RMSNorm` (weighted pre, unweighted post) and
three `Linear`s, with the gated-MLP forward. Behavior:

- Encoder mode (`diffusion_decode` not set): pass-through.
- Decoder mode: read `params["self_conditioning_logits"]` (may be `None` → zeros signal);
  compute soft embeddings `softmax(logits, fp32) @ E × embed_scale`; output
  `post_norm(x + mlp(pre_norm(soft)))`. The soft-embedding matmul needs access to the embedding
  table — keep a reference to the `Embedding` module (CPU/meta-safe: resolve at load time).

Keep this module **unquantized (fp16)** in v1 by registering its tensors via
`get_additional_compiled_tensors` (the same mechanism that keeps the Gemma4 vision tower
unquantized). It is three `hidden×intermediate` matrices — negligible next to 26B — and it never
sees calibration activations in the causal calibration pass, so quantizing it blind would be the
only risky part of conversion. This makes conversion work with **zero pipeline changes**.

#### 3.3 Block-diffusion sampling loop — `exllamav3/generator/block_diffusion.py`

A standalone `BlockDiffusionSampler`/`generate_block_diffusion()` working directly against
`Model` + `Cache` (mirroring HF's loop; batch size 1 in v1):

```
prefill(prompt, params={attn_mode: "flash_attn", cache, past_len: 0})        # encoder, causal
L = len(prompt)
for each canvas:
    canvas = randint(vocab, 256); sc_logits = None
    for step in reversed(1..max_steps):
        logits = model.forward(canvas, params={
            attn_mode: "flash_attn", cache, past_len: L,        # cache NOT advanced
            diffusion_decode: True, self_conditioning_logits: sc_logits,
        })                                                       # positions default to L..L+255
        logits /= t_min + (t_max−t_min)·(step/max_steps)
        proposal = multinomial(softmax(logits)); argmax_c = argmax(logits)
        canvas   = EB_accept_and_renoise(canvas, proposal, logits, entropy_bound)
        if stable(argmax_c, stability_threshold) and mean_entropy < confidence_threshold: break
        sc_logits = logits
    prefill(argmax_c, params={attn_mode: "flash_attn", cache, past_len: L})  # encoder commit
    L += 256; emit tokens; EOS scan → pad/stop
```

Components (each a faithful port of the HF reference, unit-testable against it):
`LinearTemperatureSchedule`, `EntropyBoundSampler` (accept + renoise), 
`StableAndConfidentStop`, EOS-in-canvas finalization. Stream per-canvas, with an optional
per-step "draft" callback like HF's `TextDiffusionStreamer.put_draft`.

Plus `examples/diffusion_gemma.py` demonstrating chat generation with draft visualization.

**Decoder-pass KV mechanics (v1):** rely on the existing dispatch path — canvas K/V are written
into the (already allocated, not-yet-committed) page(s) at `cache_seqlens = L` each step and
simply overwritten by the next step; the final encoder commit overwrites them with the real
values and advances `past_len`. No new kernels, works on every backend (flash-attn ≤256-dim,
bighead/triton for 512-dim global layers, SDPA fallback).

**Known approximation to resolve during implementation:** for *sliding* layers the span
machinery yields a per-token relative window `(max(W,256), 255)`, while HF's materialized mask
gives every canvas token the same `[L−W+1, L]` cache slice + full canvas. (HF's own
un-padded/un-compiled fast path also delegates to backend-relative windows, so both behaviors
exist in the reference.) Add a parity test vs HF; if the difference matters, implement the exact
variant: gather the last `min(W, L)` cached tokens once per canvas (`CacheLayer.get_kv`), and run
`flash_attn_func(q, [K_slice;K_canvas], …, causal=False)` per step. The same explicit-concat
variant also avoids re-quantization noise when a quantized KV cache is used (scratch writes
go through the quantized layer), so it's worth having as a switchable path regardless.

#### 3.4 Conversion/quantization

- Standard `convert_model.py` flow works once the architecture is registered: causal
  calibration == encoder mode; MoE experts calibrate with `calibration_all_experts` (as gemma4);
  `lm_head` calibrates on encoder-mode hiddens (same weights/distribution family as decoder
  outputs — acceptable; revisit if quality suffers).
- Self-conditioning stays fp16 (3.2). Optional later refinement: calibrate it with synthetic
  decoder-mode activations (zeros + soft embeddings from a few denoise steps on calibration
  prompts) and quantize.
- Verify tied-weight handling: with `tie_word_embeddings=True`, `lm_head` loads from
  `…embed_tokens` via `alt_key` (existing mechanism).

#### 3.5 Tests / validation

1. Unit tests: EB sampler, temperature schedule, stability/confidence stop vs the HF reference
   implementations on synthetic logits (no model needed).
2. Mask-parity test: decoder-pass attention vs HF `create_diffusion_decoder_attention_mask`
   semantics on toy shapes, per layer type, incl. the sliding-window question above.
3. End-to-end logit parity: unquantized EXL3 load vs HF `forward()` for (a) encoder prefill,
   (b) one decoder denoise step with and without self-conditioning. Tolerance ≈ existing
   parity tests for gemma4.
4. Quantized smoke test + eval: quantize at ~4 bpw, run the block-diffusion loop on a handful of
   prompts; track tokens/forward (HF's own throughput metric) and output quality. AR perplexity
   does not apply to this model (lm_head only ever sees decoder-mode states); use generative
   evals (e.g. existing HumanEval harness) for quant-quality comparisons.

### Phase 2 — Generator/Job integration (paged, multi-user)

- `Generator` detects `model.caps["block_diffusion"]` and routes such jobs through a canvas
  decode step instead of token-by-token `iterate_gen`: reserve pages through the existing
  allocator so the scratch page is always present (canvas == one page), run the denoise loop,
  then commit via the normal prefill path — **page hashing/dedup/reuse keep working unchanged**
  because committed pages only ever contain encoder-committed tokens.
- Jobs emit one canvas (EOS-trimmed) per outer iteration; surface intermediate drafts in the
  job results dict for streaming UIs.
- Batching: start with one diffusion job's denoise loop per `iterate()` round (interleaved with
  other jobs' prefill/decode chunks, which the scheduler already supports). True batched
  denoising across jobs needs varlen non-causal attention with per-job KV lengths — doable via
  `flash_attn_varlen` later; explicitly out of scope for v1.
- Disable/guard: draft models, n-gram drafting, token healing, logit-level samplers designed for
  AR (the diffusion sampler owns the canvas), `stop_on_loop`-style features that assume 1
  token/iteration. Stop-strings can be applied at canvas-finalization time.
- Multi-turn reuse: `past_key_values`-style continuation falls out naturally (the cache is a
  normal exllamav3 cache; prompts of turn N+1 prefill on top).

### Phase 3 — performance polish (optional, after correctness)

- Fuse the denoise-step post-processing (temperature, softmax, entropy, sort/cumsum accept,
  renoise) into one or two kernels or CUDA graphs; HF leans on `torch.compile` for the same
  reason. 262k-vocab softmax+entropy at 256 positions dominates the non-GEMM time.
- Self-conditioning soft-embedding matmul is `(256×262k)@(262k×h)` per step — consider top-k
  truncation of the softmax (quality-test first) or fp16 GEMM via the embedding table directly.
- Cache the unchanged context K/V dequantized across the ≤48 steps of one canvas
  (explicit-concat path) when using quantized KV cache.
- Revisit `swa_full=False` with SWAState stash/rewind for long-context memory savings.
- Tensor-parallel support (gemma4 itself is `supports_tp: False` today, so nothing regresses).

---

## 4. Risks / open items (to verify when the checkpoint is in hand)

1. **Exact `config.json` values** for the 26B release (layer count/hidden/experts/`layer_types`,
   `use_bidirectional_attention`, rope params) — schema is known, values pending; HF MCP access
   to the repo was flaky during research.
2. **Safetensors key prefixes** for the tied stack (`model.decoder.*` expected) and the vision
   tower / `embed_vision` / `self_conditioning` keys.
3. **Sliding-layer decoder mask exactness** (3.3) — needs the parity test to choose
   span-window vs explicit-concat.
4. **`generation_config.json`** released defaults vs the in-code defaults listed above.
5. **Quantization sensitivity of diffusion decoding**: denoising may be more sensitive to weight
   error than AR decoding (errors compound across steps *within* a canvas but also self-correct
   via re-noising). Needs empirical eval; the 26B-A4B MoE at 3–4 bpw targeting ~18 GB consumer
   GPUs is exactly exllamav3's sweet spot, and "fits in 18 GB quantized" is Google's own pitch —
   strong motivation for this feature.
6. Chat template / special tokens for the `-it` checkpoint (expected Gemma-style `<start_of_turn>`
   family; wire into `default_chat_prompt`).
