# DiffusionGemma in exllamav3 — handoff notes

*Session handoff, 2026-06-11. Companion to `doc/diffusion_gemma_plan.md` (research notes and original
implementation plan).*

## What this is

Full support for `google/diffusiongemma-26B-A4B-it` (released 2026-06-10) in exllamav3: architecture,
quantization-compatible model definition, the block-diffusion sampling algorithm, integration with the
standard `Generator`/`Job` API so existing front-ends work unchanged, and a first round of performance
work. Implemented, unit-tested, and validated end-to-end by the project owner through an external chat
front-end (ezexl3) on the real checkpoint.

## Where the work lives

Branch `claude/youthful-sagan-w98xvv`, tip `96e0de7` (verified present on GitHub). The owner's `up`
branch carries the first two commits as PRs #24–#26 plus a manual fix (see "incident log" below);
`96e0de7` cherry-picks cleanly on top.

Commits, in order:

| Commit | Content |
|---|---|
| `fae9d1f` | Research notes / implementation plan (`doc/diffusion_gemma_plan.md`) |
| `0b8cc1e` | Phase 1: architecture, self-conditioning module, standalone block-diffusion generator, tests |
| `48eda9f` | Fix: accept (and override) the `swa_full` kwarg that `model_init` passes to every architecture |
| `9e7ab02` | Phase 2: block diffusion through the standard `Generator`/`Job`/`iterate()` API |
| `96e0de7` | Performance: optimized denoising step, exact sliding-window decoder attention, instrumentation |

Files:

- `exllamav3/architecture/diffusion_gemma.py` — `DiffusionGemmaConfig` / `DiffusionGemmaModel` /
  `DiffusionGemmaVisionModel`
- `exllamav3/modules/arch_specific/diffusion_gemma.py` — `DiffusionGemmaSelfConditioning`
- `exllamav3/generator/block_diffusion.py` — `BlockDiffusionSettings`, `CanvasDenoiser`,
  `BlockDiffusionGenerator`, sampler math (`token_entropy`, `eb_accept_mask`, `gumbel_sample`)
- `exllamav3/generator/generator.py` — `bd_mode` decode path (`iterate_block_diffusion` and `bd_*`
  helpers)
- `exllamav3/modules/attn.py` — `diffusion_decode` sliding-window branch in `decode_flash_attn`
- `exllamav3/architecture/gemma4.py` — vision key prefix via `config.vision_key_prefix` (default
  unchanged), vision rope_theta also read from `rope_parameters`
- `examples/diffusion_gemma.py`, `tests/test_block_diffusion.py` (37 CPU tests),
  `tests/smoke_diffusion_gemma_arch.py` (checkpoint + GPU, incl. a Generator/Job round trip)

## How the model works (one paragraph)

One Gemma4-style MoE transformer stack (30 layers, 5:1 sliding:full, hidden 2816, 128 experts top-8,
K==V on 512-dim global heads) used in two modes with fully tied weights. *Encoder mode* is an ordinary
causal forward that extends the KV cache (prompt prefill and committing each finished block). *Decoder
mode* denoises a 256-token "canvas": uniform-random token ids plus a self-conditioning signal (previous
step's probabilities through the tied embedding table and a small gated MLP) are forwarded with
bidirectional attention over [context window + canvas], then the entropy-bound sampler accepts the
approximately-independent low-entropy tokens and re-noises the rest, until the canvas is stable and
confident (or 48 steps). The finished block is appended autoregressively and the loop repeats.

## Design decisions worth knowing

1. **The whole thing rides on existing exllamav3 primitives.** Canvas K/V land in the not-yet-committed
   page past the cache's committed length (`PAGE_SIZE` == `canvas_length` == 256, so the scratch is
   exactly one page); each denoising step overwrites it and the commit prefill replaces it with real
   values. No new kernels; the dispatch's `causal=False` / `non_causal_spans` machinery (built for
   Gemma4 vision) provides bidirectionality, and the 512-dim global layers ride the existing
   triton-paged backend (flash-attn rejects dim > 256).
2. **Conversion/quantization needed zero pipeline changes, by construction.** Encoder mode == causal
   calibration. The self-conditioning module is a pass-through outside decoder mode and its Linears
   carry no qmap, so they are stored fp16 automatically. Quantize with `convert_model.py` as usual.
3. **`swa_full=True` always.** Sliding layers live in the regular paged cache instead of
   `SlidingAttention`/`SWAState` rolling buffers, so denoising passes can never corrupt recurrent
   state. Costs cache memory at long context; an SWAState-aware mode (stash/rewind) is possible later.
4. **Generator integration is a parallel decode path, not a rewrite.** `Generator` detects
   `model.caps["block_diffusion"]` and routes `enqueue`/`iterate` through a serial canvas loop emitting
   the standard `started`/`prefill`/`streaming` result dicts. Jobs queue normally but execute one at a
   time. The cache is treated as one flat sequence with longest-common-token-prefix reuse across jobs
   (multi-turn chat does not re-prefill history). Structural features raise at enqueue (CFG, filters,
   token healing, multimodal embeddings); soft ones warn once and are ignored (banned strings, loop
   detection, `min_new_tokens`, AR samplers, logits/probs returns).
5. **The model enforces its own load requirements.** `DiffusionGemmaModel.load_gen` forces
   `max_output_size >= canvas_length` (and `max_output_factor >= 2`) so unmodified loaders reserve VRAM
   for full-canvas logits.
6. **Checkpoint key prefixes are probed, not assumed.** The tied stack is stored once (found under
   `model.decoder`); the vision tower under `model.encoder`. `DiffusionGemmaConfig` probes candidates
   against the safetensors collection at load time.
7. **The embedding table stays on GPU** (`prefer_cpu` overridden, ~1.5 GB at fp16) because
   self-conditioning needs a (256 × 262144) @ (262144 × 2816) matmul per denoising step.

## Performance state and the remaining gap

Symptoms reported on first real run: ~100 t/s (expected 300–500) and degradation over long
generations. Diagnosis and fixes in `96e0de7`:

- **Sampling overhead**: the step ran several independent full-vocabulary normalizations plus
  `torch.multinomial` (very slow at 262k vocab) plus multiple device syncs. Now: one `log_softmax`
  serves entropy, both samples and self-conditioning with buffer reuse; Gumbel-max sampling replaces
  multinomial (identical distribution); one sync per step on the combined stopping flag.
- **Degradation**: the original sliding-layer decoder attention used flash-attn's *relative* window —
  per-token shifted, diverging from the reference *uniform* window once context exceeded
  `sliding_window` (1024). Worse convergence → more steps → slower and lower quality, onset correlating
  with generation length. The denoise pass now gathers the trailing window slice and attends
  bidirectionally over [slice + canvas]: exact per the HF mask, constant-cost in context length, and no
  canvas K/V writes (so no per-step requantization churn on quantized caches) for those layers.
- **Instrumentation**: streaming results report `canvas_steps`; the final result adds
  `denoising_steps` and `tokens_per_forward`. Throughput ≈ tokens_per_forward × forward rate; if
  canvases consistently burn all 48 steps, adaptive stopping isn't firing and settings need tuning.

Tuning without code changes: `generation_config.json` in the model directory is read at generator
construction — `sampler_config.entropy_bound` (0.1 default; 0.2–0.4 accepts more tokens per step),
`max_denoising_steps`, `confidence_threshold`, `t_min`/`t_max`, `stability_threshold`. Programmatic:
`Generator(..., block_diffusion_settings=BlockDiffusionSettings(...))`.

Known remaining gaps to the headline numbers (Google: 700+ t/s on RTX 5090): the HF reference gets
there with `torch.compile(mode="reduce-overhead")` + CUDA graphs over the encoder, decoder, sampler and
stopping criteria. Per-step kernel-launch overhead and the eager sampler are the next targets (below).
Data still wanted from real runs: `tokens_per_forward` early vs late in long generations, whether
quantized KV cache (`-cq`) is used, and whether flash-attn is installed (vs triton/SDPA fallbacks).

## Forward: suggested next steps, in rough priority order

1. **Throughput follow-up with the new instrumentation.** If steps/canvas ≈ 48 always, investigate
   convergence (see parity item below) before micro-optimizing.
2. **CUDA-graph / torch.compile the denoising step.** The forward is shape-stable (1×256) per canvas;
   the sampler is a fixed tensor program. Graph-capturing forward+sampler and syncing only on the
   stopping flag every step (or every k steps) is the single biggest remaining win.
3. **HF logit-parity harness.** Compare encoder prefill and one decoder step (with and without
   self-conditioning) against transformers on the bf16 checkpoint. The riskiest numerics are the
   self-conditioning block (embed-scale rounding, unweighted post-norm) and the decoder mask edges
   (window slice is `[L-W+1, L)` matching HF's non-compiled path; HF's compiled path sees one extra
   token). Any convergence problem will show up here first.
4. **Quantized-cache global layers**: per-canvas memoization of the dequantized context K/V for the 5
   full-attention layers (context only changes between canvases; currently dequantized every step,
   growing with context).
5. **Quantization sensitivity eval**: quantize at 3/4/5 bpw and track `tokens_per_forward` (a built-in
   quality proxy: a worse model is less confident, accepts fewer tokens per step and converges slower)
   plus a generative eval. Calibration currently runs encoder-mode only; if decoder-mode quality lags,
   consider mixed calibration with random-canvas suffixes.
6. **VRAM: soft-embedding matmul.** Options: top-k-truncated probabilities (quality-test first), an
   fp8 embedding copy, or chunked matmul. Would free ~1.5 GB for small-VRAM setups.
7. **Multimodal prompts** through both generation paths (the vision tower and Gemma4 span machinery are
   already wired; `enqueue` currently rejects embeddings for diffusion jobs). Validate
   `use_bidirectional_attention == "vision"` image spans interacting with canvas commits, and the
   `processor_config.json` assumptions in `Gemma4Config`.
8. **Batched denoising across jobs** (varlen non-causal attention with per-job context lengths) if
   multi-user serving matters; the serial path is correct but doesn't scale concurrency.
9. **Upstreaming.** The `gemma4.py` changes are backwards-compatible and everything else is additive.
   When proposing to turboderp/exllamav3, the deliberately-contained integration points (one gated
   branch in `attn.py`, one `bd_mode` path in `generator.py`) should make review tractable.
10. Small polish: optional draft streaming through Generator results; `min_new_tokens` semantics
    (suppress EOS in early canvases) if anyone needs it; stop-string `token_ids` alignment is
    approximate (text is exact).

## Backward: incident log and lessons

- **`swa_full` kwarg collision** (`48eda9f`): `model_init.py` passes `swa_full=` to
  `Model.from_config` for *every* architecture. New architecture classes that hardcode parent kwargs
  must accept-and-override, not just set them.
- **Squash-merge kept dead code**: the owner merged the branch onto `up` as PRs #24–#26; the conflict
  resolution kept both the old "not supported" assert *and* its replacement, so the assert kept firing
  even though the integration was present. Diffing `up` against the source branch found the 4 stray
  lines immediately. After a merge with conflicts, diff against the source branch, not just the PR.
- **A push silently failed to reach GitHub**: the session's git proxy reported success
  ("[new branch]") for a push that never landed (and the branch had also been deleted after the PR
  merges). Verified and re-pushed via the GitHub API. Lesson: after pushing from this environment,
  confirm the remote tip independently when it matters.
- **`torch.multinomial` at 262k vocab is a trap**; Gumbel-max is a drop-in equivalent.
- **Relative vs uniform attention windows matter for diffusion convergence** in a way they don't for
  AR decoding — the approximation was invisible until context exceeded the window, then degraded
  quality and speed together.
- Numerical detail: entropy must be computed as `-(p·log p)` from `log_softmax` (matching
  `Categorical.entropy()`); the `lse − Σp·logits` form loses precision exactly in the low-entropy
  regime the confidence threshold lives in (caught by a unit test at tolerance).

## Quick reference

```bash
# Standalone generation with live denoising visualization
python examples/diffusion_gemma.py -m <model_dir> -p "Why is the sky blue?" --show_drafts

# Smoke checks against a real checkpoint (graph, K==V, self-conditioning, both generation APIs)
python tests/smoke_diffusion_gemma_arch.py --model_dir <model_dir> --generate

# CPU unit tests (sampler math vs references, Generator decode-path logic)
pytest tests/test_block_diffusion.py

# Quantize (no special flags needed)
python -m exllamav3.conversion.convert_model -i <hf_dir> -o <exl3_dir> -b 4
```
