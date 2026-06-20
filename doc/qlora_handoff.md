# QLoRA-on-EXL3 — Handoff & Next-Step Plan

> Working session handoff. Branch: **`claude/magical-mayer-fq6z4i`**.
> Goal of the original task: prove whether **QLoRA fine-tuning on EXL3-quantized
> weights** is possible, and get a visible end-to-end demo (fine-tune a small
> model to talk like a pirate).

---

## 0. RESOLVED — QLoRA-on-EXL3 works end-to-end (transformers-free)

> Completed on branch `claude/determined-gauss-suq9gx`. The original question
> ("is QLoRA fine-tuning on EXL3-quantized weights possible?") is **answered:
> yes**, demonstrated end-to-end on the GPU box with the real model, with **no
> `transformers` dependency in the path at all**.

**What was run and confirmed (Llama-3.2-1B-Instruct, EXL3 4bpw, single GPU):**

1. **Forward validated against native.** `examples/qlora_validate_native.py`
   PASSED: the differentiable forward's logits match exllamav3's own (correct)
   inference forward — top-1 next-token identical on every prompt, **100%
   per-position argmax agreement**, last-token logits `cos ≈ 0.999999`,
   `max|Δ| ≈ 0.02–0.03` (just fp32-vs-native-fp16 rounding). e.g.
   "The capital of France is" → ` Paris`. This was the whole ballgame: the
   backbone that produced garbage under transformers 5.x is correct here on the
   same quantized weights.

2. **Training works.** `examples/qlora_train_native.py` (plain PyTorch loop, only
   `pip install datasets`) trained adapters on `TeeZee/dolly-15k-pirate-speech`.
   Healthy diagnostics throughout: first loss ~2–3 (NOT ~11 random), grad norm
   20–50 (gradients reaching adapters), `|B|` climbing monotonically 0→13, EMA
   loss falling 2.78→~2.35 then plateauing at the data's irreducible-loss floor.

3. **Adapter saves + reloads natively + steers generation.**
   `examples/qlora_infer_native.py` loads the PEFT adapter via the native
   `LoRA.from_directory` loader (224 tensors = 32 layers × 7 targets) and the
   output measurably changes vs base. Cranking `--lora-scaling` proved the
   learned direction is exactly the dataset's pirate transform: at ~5× effective
   the generation collapses into `"be be be …"` — the arrr library's dominant
   `is/are/am → be` substitution, over-amplified. Coherent-but-clearly-pirate
   sweet spot is `--lora-scaling ~1.4` (effective ~2.8×).

**Caveat on the *visible* demo (not a code issue):** the chosen dataset is a
**light, inconsistent** pirate conversion (the `arrr` library: `the→th'`,
`is→be`, `you→ye`, `my→me`, occasional canned phrases; responses also lowercased
+ terse; many rows show no pirate markers at all — verified by previewing the
rows). So at the trained scale (`--lora-scaling 1.0`) the effect is subtle, and
the most consistently learnable signal is the lowercasing/terseness, not the
sparse substitutions. For a *naturally* dramatic pirate at scale 1.0, swap in a
heavier-pirate dataset (the loader only needs instruction/response-style fields);
nothing about the training path needs to change.

**Recommended workflow now: §0 (next section). The transformers-5.x
investigation in §4–5 is fully superseded and only of historical interest.**

---

## 0b. Transformers-free native path — implementation details (option 2)

> Added on branch `claude/determined-gauss-suq9gx`.

Rather than keep fighting the transformers-5.x RoPE bug (§4–5 below), the
**fallback option 2 (§6) is now built**: a self-contained, autograd-friendly
Llama forward on exllamav3's *own* loaded weights — **no `transformers` import in
the training path at all**, so it cannot be broken by an upstream version bump.
This is now the recommended path.

**New code (all CUDA-free to import; pure torch):**
- `exllamav3/training/native_llama.py`
  - `NativeLlamaQLoRA(model, r, alpha, target_modules, compute_dtype, …)` —
    builds a differentiable Llama decoder (RMSNorm + GQA/NeoX-RoPE attention +
    SwiGLU + residuals) directly over a **loaded native `exllamav3.Model`**. It
    reuses the exact RoPE table (`attn.rope.inv_freq`), norms, and `sm_scale`
    the correct native inference forward uses. Frozen base weights are
    reconstructed on the fly via `get_weight_tensor()`; only LoRA `a`/`b` (fp32)
    train, routed through the gradchecked `EXL3LoRAFunction`. Head loss via the
    streaming `fused_linear_cross_entropy` (no `[tokens, vocab]` logits).
    `.compute_loss()`, `.logits()`, `.save_adapter()` (PEFT format),
    `.apply_to_native()/.remove_from_native()` (so generation reflects the
    adapter mid-train). Unsupported features (q/k-norm, sliding window,
    softcapping, MoE, gating, partial/mRoPE, non-NeoX) are rejected loudly.
  - `DiffLinear` — differentiable frozen-base + optional-LoRA linear.
- `examples/qlora_validate_native.py` — **the correctness gate.** Compares the
  differentiable forward's logits against the native (correct) forward,
  per-prompt: top-1 token agreement, per-position argmax agreement, last-token
  `max|Δ|` / cosine. Run this FIRST.
- `examples/qlora_train_native.py` — plain PyTorch training loop (no HF Trainer /
  transformers / accelerate; just `pip install datasets`). Pirate SFT,
  completion-only masking via the native Llama-3 chat template, fused-CE,
  gradient checkpointing, live native samples.
- `tests/test_native_llama.py` — CPU tests (torch only): `DiffLinear` matches
  reference + gradcheck; one decoder block matches an independent plain-torch
  reference to <1e-4; backward reaches every adapter while the base stays
  frozen. (Other CPU tests in §3 still apply.)

**Workflow (on the GPU box, any transformers version or none):**
```
# 1. PROVE the forward is correct (no training):
python examples/qlora_validate_native.py --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/
#    Expect: PASS, "The capital of France is" -> ' Paris', high argmax agreement.

# 2. TRAIN (transformers-free):
python examples/qlora_train_native.py \
    --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/ \
    --out   /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/pirate2
#    Expect: first loss ~2-4 and dropping; live samples turn piratey.

# 3. VERIFY on the native inference path (already worked before):
python examples/qlora_infer_native.py \
    --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/ \
    --adapter /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/pirate2
```

**Status: RUN AND CONFIRMED on the GPU box** — see §0 for the full results
(validate PASSED with 100% argmax agreement; training healthy; adapter
saves/reloads natively and steers generation). Issues found and fixed during the
real run, all on this branch: CPU-embedding vs GPU-decoder device split; KV cache
must be allocated before `model.load()`; exact prompt/response label masking. The
§4–5 transformers-5.x investigation below is fully superseded (historical only).

---

## 0c. Session 2 — PROVEN end-to-end on a visible demo; dataset density is the key variable

> Branch `claude/zen-franklin-g5hedw` (merged to master). Many runs on
> Llama-3.2-1B and -3B (4bpw), single-GPU and 2× RTX 3090 (DDP).

### Headline result

The EXL3-QLoRA path is **proven end-to-end with an unambiguous visible demo.** An
**ALL-CAPS smoke test** (`--uppercase-response`: train the model to RESPOND IN
CAPS) gave a clean, controllable before/after on Llama-3.2-1B 4bpw:
- BASE → normal mixed case; ADAPTED → SHOUTS IN CAPS, strength scaling with
  `--lora-scaling` (subtle at 1.0 after a short run, consistent across all prompts
  at 2–3). No ambiguity: differentiable forward over the trellis-quantized base,
  frozen base + trained LoRA, save/reload/steer — all working on quantized weights.

### THE key lesson: signal *density*, not rank/steps/model size

Every earlier style demo was muddy for ONE reason — sparse/inconsistent training
signal. The uppercase test isolated the variable:
- **Uppercase** — *every token of every row* changes → learns cleanly, shows
  controllably (can't reach low loss without it, can't hide it at decode). ✅
- **Pirate** (`TeeZee/dolly-15k-pirate-speech`) — `arrr`-library swaps
  (the→th', is→be, you→ye) are *sparse* and many rows barely pirate. Trained hard
  (r64, ~2 epochs, 1B & 3B) it DOES show th'/be/ye when cranked (effective ~2.5–3
  on 1B; coherent further up on 3B) but collapses into "be be be" past that. Real
  but light.
- **UwU** (`superdrew100/UwU_Alpaca_data`) — style in *rare* tokens
  (emoji/caps/OwO). Loss fell to ~0.7 (English backbone fit) but markers stayed
  low-probability and never surfaced at decode, even at scale 2 + temp; only soft
  persona traces ("shy being") leaked. Sparse-marker styles don't transfer to a
  small model's greedy/low-temp generation.
- **Shakespeare** (`Roudranil/shakespearean-...`) — dense register BUT a *play
  script*: tangential monologues + stage directions. At strength it degenerated
  (parroted `*stage directions*`, repetition); tamed it went bland. Data-structure
  ceiling.

**Recipe for a good visible demo: a DENSE, CONSISTENT style on CLEAN
instruction→answer Q&A.** Uppercase is the trivial (no-LLM) instance of exactly
that. A heavily-styled *generated* dataset (every row strongly transformed) would
behave like the caps demo — clean and controllable. Light off-the-shelf style
sets (pirate) or sparse-marker ones (UwU) won't give a clean scale-1.0 demo on a
small model.

### Tooling added this session (on the branch / merged to master)

`examples/qlora_train_native.py` (+ DDP variant):
- **Dataset-agnostic loader** — `--instruction-key/--context-key/--response-key`,
  `--dataset-split`. Defaults are Alpaca (`instruction`/`input`/`output`, dataset
  `superdrew100/UwU_Alpaca_data`). Pirate (Dolly schema):
  `--dataset TeeZee/dolly-15k-pirate-speech --instruction-key instruction
  --context-key context --response-key response`.
- **`--uppercase-response`** — the dense smoke test (uppercases only the response).
- **`clean_style_text`** (default on; `--no-clean-text`) — strips
  `[stage directions]`/`*actions*` + normalizes whitespace; `--min-response-words`
  drops junk-short rows. Use `--no-clean-text` for UwU (keeps `*action*` flavor).
- **Checkpointing** — `--save-every N` + save-on-Ctrl-C (previously saved only at
  the end, so early-stopping discarded everything).
- **Resume** — `--resume <adapter_dir>` + `NativeLlamaQLoRA.load_adapter()`
  (inverse of `save_adapter`). NOTE optimizer state is NOT restored (cold AdamW
  re-warmup; harmless for LoRA); `--r`/`--targets` must match the checkpoint.

`examples/qlora_infer_native.py`:
- **Sampling controls** — `--temperature/--min-p/--top-p/--top-k/--seed`. Library
  default is temp 0.8 + min_p 0.08, which truncates the low-prob tail and hides
  sparse-marker styles; `--temperature 0` = greedy. `--lora-scaling` unchanged.

`examples/qlora_train_native_ddp.py` (NEW): multi-GPU via DDP (see §0d).

### Multi-GPU (DDP) — confirmed working on hardware (2× RTX 3090)

Both GPUs 100% util; disjoint data shards (~total/N per rank); loss tracks a
single-GPU run at the same *effective* batch. **GPU1 was on PCIe ×4 and it didn't
matter** — only the tiny LoRA grads are all-reduced, so the slow lane isn't a
bottleneck. That's exactly why DDP (not FSDP) fits QLoRA-on-EXL3.

```
torchrun --standalone --nproc_per_node=2 examples/qlora_train_native_ddp.py \
    --model /mnt/two/Weights/<model>/4/ --out /mnt/two/Weights/<model>/4/run \
    --dataset ... --lora-r 128 --alpha 128 --batch 16 --steps 600 --save-every 100
```
- Run once in one terminal — `torchrun` spawns one process per GPU. Only rank 0
  prints/saves (so the log looks single — confirm both GPUs with `nvidia-smi`).
- Resume a single-GPU checkpoint on N GPUs: add `--resume <dir>` (loaded on every
  rank before the broadcast). Effective batch = `--batch × nproc × --grad-accum`.
- DDP script has NO live `🎭` samples / `|B|` column — confirm via the infer sweep.

### Tuning lessons

- **Effective strength = `(alpha/r) × --lora-scaling`.** Single-GPU default is
  r=32/alpha=64 (ratio 2.0); DDP default r=64/alpha=64 (ratio 1.0). Use ratio 1.0
  (`--alpha == --r`) for an intuitive knob.
- **Loss plateau ≠ done.** EMA flattens fast; style keeps firming past it.
  Pirate-hard *broke through* a second time (~2.0→~1.4 around 1 epoch) learning the
  deeper swaps. EMA is a local logging var — it "resets" on resume (meaningless);
  watch raw loss.
- **Harder training → lower inference scale.** A harder-trained adapter is stronger
  per unit scale, so the coherent sweet spot moves DOWN; sweep low first.
- **Bigger base holds coherence under amplification** (3B coherent at higher scale
  than 1B before "be be be" collapse).
- **Live samples run at effective `alpha/r`** — light/sparse styles show nothing
  there even when learned; judge by the inference scale sweep. (Dense styles like
  uppercase DO show live.)

### Gotchas hit (and fixed)

- **torchrun eats `--r`** (abbrev-matches `--rdzv-*`/`--role`). DDP script uses
  `--lora-r` (dest `r`); single-GPU `--r` is fine.
- **OOM from `--batch 48 --no-grad-ckpt`** on wordy data — full attention
  activations × layers exceed 24GB. Fix: drop `--no-grad-ckpt` (checkpointing on)
  and/or lower `--batch`. Only use `--no-grad-ckpt` with VRAM to spare.
- **`<|eot_id|>` spam** in some generations — infer script sets no EOT stop
  condition, so it runs into new assistant turns. Cosmetic; adapter unaffected.
- **`.../4/` in chat commands is a placeholder** for the full model path.

### Run status
- **Uppercase smoke test (1B):** ✅ PROVEN — clean CAPS before/after, scalable.
- **Pirate-hard (1B r64, resumed single→2-GPU):** light but real pirate at
  effective ~2.5–3; collapses past that.
- **Pirate-hard (3B r128, 2 epochs, DDP):** th'/ye at scale ~3, coherent (no hard
  collapse) but still light — dataset ceiling.
- **UwU (1B/3B):** soft persona only; sparse markers don't surface. Not recommended.
- **Shakespeare:** rejected (play-script structure).

### Recommended next step
A **dense funny style on clean Q&A** — *generate* it (take Alpaca/Dolly prompts,
rewrite every answer in a strong style with a local model) so every row is heavily
styled; it'll then behave like the caps demo. OR move to the flagship: low-bpw
(2.5–3) bigger-model fine-tune on a real task with a metric, benchmarked vs what
BNB NF4 can fit (the actually-valuable result — see §0d / implications).

---

## 0d. Multi-GPU strategy (rationale)

"Multi-GPU" splits by *goal*, and QLoRA changes which tool fits, because only the
tiny LoRA params train and the frozen quantized base is small:

- **DDP (data parallel) — easy, the right default for throughput.** Replicate the
  small quantized model per GPU, shard the batch, all-reduce only the LoRA grads
  (a few MB). Built (`qlora_train_native_ddp.py`), confirmed on 2× 3090. We
  hand-average the LoRA grads rather than wrapping in
  `nn.parallel.DistributedDataParallel`, because the module is mostly frozen
  buffers + a custom `autograd.Function` + grad checkpointing, which DDP's
  bucketing handles awkwardly.
- **Pipeline / layer-split — moderate, for models too big for one GPU.** exllamav3
  already splits layers across GPUs for *inference*; the native training forward
  would need to be made device-aware (move hidden states across block boundaries;
  autograd handles cross-device grads). Not built.
- **FSDP — hard, and usually the WRONG tool here.** Its value is sharding huge
  *trainable* params + optimizer; here the trainable surface is tiny and the frozen
  base is a packed trellis that doesn't shard like bf16. You'd gain ~nothing and do
  real engineering to make the packed format FSDP-compatible (cf. Answer.AI's
  FSDP-QLoRA, a genuine research project for exactly this). EXL3's compression also
  partly dissolves FSDP's main use case: a 70B at 2.5bpw is ~22GB → fits one 24GB
  card, so you may never need to shard the model — DDP for throughput +
  pipeline-split for long context is enough.

**Implications (the real prize):** EXL3 makes a bitrate regime *trainable* that
BNB NF4 can't reach (NF4 is unusable ≤3bpw; EXL3's trellis stays coherent at
2.5–3bpw). So QLoRA on a 2.5bpw 70B fits a single 24GB card, and you train the
adapter against the *exact* weights you deploy (no train/serve quant mismatch).
Expected outcome: rough parity with BNB at 4-bit, clear EXL3 win in the low-bitrate
regime. The flagship experiment to substantiate it: same model fine-tuned BNB-NF4
vs EXL3-4bpw vs EXL3-~2.5bpw at matched VRAM, compared on a real downstream metric
+ tokens/sec.

---

---

## 1. TL;DR status (historical — see §0 for the resolved status)

> This section describes the state *before* the transformers-free native path was
> built and run. The "Blocker" below was resolved by §0/§0b, not by fixing the
> transformers-5.x forward. Kept for context.

- **The QLoRA-on-EXL3 mechanism is built and verified.** Differentiable EXL3
  linear, fused cross-entropy head, adapter attach/save/load — all gradcheck-
  verified on CPU, and the per-layer forward matches the EXL3 kernel to 0.07%.
- **Native exllamav3 inference + native LoRA loading both work** (coherent base
  generation, adapter applies).
- **Blocker:** the only *differentiable* forward we have for training is the HF
  Transformers integration, and it is **broken on every transformers version
  available on this machine**:
  - transformers **5.x**: EXL3 quantizer engages, per-layer weights correct,
    but the assembled forward produces garbage (RoPE/attention mismatch).
  - transformers **4.56 / 4.57**: quantizer does **not** engage at all — model
    loads with random weights.
- **Plan (option 1, chosen):** go back to transformers **5.x** (the only version
  where the quantizer engages), diagnose the localized forward bug (almost
  certainly RoPE), patch it, then train for real.

The previously-trained adapter at `.../4/pirate` is **garbage** (trained against
the broken 5.x forward, final loss ~10.37 ≈ random). Discard it.

---

## 2. Key paths & environment

**Model (EXL3, 4bpw):** `/mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/`
- Llama-3.2-1B-Instruct, **tied embeddings** (`tie_word_embeddings: true`).
- `config.json` is complete and correct (`max_position_embeddings: 131072`,
  `rope_scaling: {rope_type: llama3, factor: 32, ...}`, `quantization_config:
  {quant_method: exl3, version 0.0.21, bits 4, head_bits 6}`).
- **`config.json` says `transformers_version: 4.45.0.dev0`** — the model (and its
  EXL3 calibration) was produced against transformers 4.45. This is the leading
  suspect for the 5.x forward mismatch.

**Bad adapter (discard):** `/mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/pirate`

**Two venvs:**
- **`~/exl3/tabbyAPI/venv`** ("tabby") — the user's main venv. Uses exllamav3
  **natively** (transformers-independent). Do **NOT** further mutate it. (It was
  temporarily changed during this session; ideally restore it with
  `pip install "transformers==5.10.2" kernels`.)
- **`~/exl3/qlora-venv`** — the isolated venv we built for this work. Current
  consistent state:
  - torch **2.8.0+cu128**, triton 3.4.0
  - flash-attn **2.8.3.post1** (prebuilt wheel, cu12torch2.8 cxx11abiTRUE cp312)
  - transformers **4.56.1**, tokenizers 0.22, datasets, accelerate
  - pydantic **2.10.6** (pinned; 2.13 breaks formatron 0.5.0)
  - **xformers uninstalled** (optional fallback backend; was version-conflicting)
  - **no `kernels`** (conflicts with huggingface_hub strict dataclasses)
  - exllamav3 installed **editable** (`-e`) pointing at `~/exl3/private/exllamav3`;
    CUDA extension JIT-built and cached in `~/.cache/torch_extensions`.

**Env pitfalls (learned the hard way):**
- Do **not** re-run `pip install -e exllamav3` — its deps (xformers/
  flash-linear-attention) drag torch up to 2.12 and break the prebuilt EXL3 `.so`
  (ABI mismatch: `undefined symbol: ...c10_cuda_check_implementation`). If torch
  moves, pin it back: `pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128`.
- xformers must match torch exactly; simplest is to leave it uninstalled (the
  package's existing `except ModuleNotFoundError` guard handles absence).
- Keep pydantic `<2.11`.

---

## 3. What was built (all committed on the branch)

### Training library — `exllamav3/training/`
- **`qlora_linear.py`**
  - `EXL3LoRAFunction` — memory-efficient `autograd.Function`. Forward
    `y = x @ W_eff + scale·(x@A@B) + bias`; backward recomputes `W_eff` from a
    `weight_fn` closure instead of storing it. Adapters can be fp32 master
    weights while compute is bf16/fp16 (cast inside fwd/bwd; no-op for the
    float64 gradcheck). **gradcheck-verified.**
  - `reference_forward` (plain-autograd ground truth), `qlora_linear_forward`,
    `QLoRALinear` (standalone nn.Module).
  - Key fact: `W_eff = LinearEXL3.get_weight_tensor()`, shape `[in, out]`,
    so `y = x @ W_eff`. **Verified equal to the EXL3 kernel forward to
    rel_err 0.00067** (and `W.t()` gives 1.41 — orientation confirmed).
- **`fused_ce.py`** — `FusedLinearCrossEntropy`: streaming linear cross-entropy
  over token chunks; never materializes `[tokens, vocab]` logits; recomputes the
  frozen head weight in backward. `qlora_causal_lm_loss(model, ...)` wires it via
  `get_decoder()` / `get_output_embeddings()` (unwraps DataParallel). Promotes to
  ≥fp32 internally. **All correctness tests pass** (matches `F.cross_entropy`,
  ignore_index, chunk-invariant, gradcheck, shifted-CausalLM wiring).
- **`hf_qlora.py`**
  - `Exl3LoRALinear` — trainable wrapper over a frozen `Exl3HfLinear`; base
    weight reconstructed on the fly; only `lora_a`/`lora_b` (fp32) train; B=0 init.
  - `attach_qlora(model, r, alpha, target_modules, ...)` — swaps matching EXL3
    linears for trainable wrappers, freezes everything else.
  - `prepare_model_for_qlora_training(model)` — gradient checkpointing +
    `enable_input_require_grads()` + `use_cache=False`.
  - `save_lora_adapter` / `load_lora_adapter` — PEFT format, compatible with both
    PEFT and the native `exllamav3.model.lora.LoRA` loader (verified orientation).

### Examples — `examples/`
- **`qlora_train.py`** — HF Trainer QLoRA. Defaults: dataset
  `TeeZee/dolly-15k-pirate-speech`, completion-only label masking, bf16 compute,
  fp32 adapters, fused-CE `compute_loss`, gradient checkpointing, live pirate
  sampling every N steps (`--sample-every 0` to disable). Monkeypatches
  `transformers.trainer.validate_quantization_for_training` to bypass the
  "purely quantized" guard (works on 5.x; see §5 note for 4.56).
- **`qlora_infer.py`** — HF before/after (depends on a working HF forward; broken
  until the forward bug is fixed).
- **`qlora_infer_native.py`** — **WORKS.** Native exllamav3 forward + native
  `LoRA.from_directory`. Use this to validate any adapter regardless of the HF
  mess.

### Tests — `tests/` (all pass on CPU, torch only)
- `test_qlora_grad.py` (tiers 1–2 always; tier 3 GPU/model opt-in),
  `test_qlora_train_loop.py`, `test_fused_ce.py`.

### Library fix kept (legit, not a workaround)
- `exllamav3/integration/transformers.py`: `Exl3HfLinear.weight` is now a frozen
  `nn.Parameter` (was a bare tensor) — fixes a crash in modern transformers'
  tied-weight finalizer (`get_parameter('...weight')` → "is not an nn.Parameter").

### Docs
- `doc/qlora_feasibility.md` — the design rationale / roadmap.
- `doc/qlora_handoff.md` — this file.

---

## 4. The bug to fix (the whole ballgame)

On **transformers 5.10.2** (the only version where the quantizer engages):
- `AutoModelForCausalLM.from_pretrained` engages `Exl3HfQuantizer`; 113
  `Exl3HfLinear` modules present; one probed layer matches the kernel to 0.07%.
- `embed_tokens`: healthy (`mean_abs 0.016`, fp16, cuda, no NaN).
- final-norm weight `mean_abs 2.35` (plausible for Llama-3.2; unverified).
- **But the full forward is garbage:** `"The capital of France is"` →
  `loss 15.7` (random ≈ `ln(128256)=11.76`), top-5 next-token = junk
  (`ĠComfort`, `Ġtrack`, …). Generation is word-salad. Training `train_loss ≈ 10.37`.

**Localization already established:** `qlora_causal_lm_loss` builds logits from
the **decoder's hidden states** × the **verified-correct** `lm_head` weight
(`get_weight_tensor`). It still gives ~10.37 → **the backbone (decoder) produces
bad hidden states**, not just the head. Backbone linears + embeddings are correct
→ the break is in the stock-transformers assembly: **RoPE / attention / norm**.

**Leading hypothesis: RoPE.** The model + EXL3 calibration are from transformers
4.45 (`config.json`), and 5.x changed `llama3` rope handling
(`modeling_rope_utils.standardize_rope_params`, etc.). Wrong positional encoding
→ wrong attention → garbage hidden states. (A later 5.x reinstall even *crashed*
in `standardize_rope_params` accessing `max_position_embeddings` — an extra clue
that rope handling is the fragile area, though that particular crash was env
churn.)

**Quantizer-engagement matrix (important):**
| transformers | quantizer engages? | forward correct? |
|---|---|---|
| 5.10.2 | **yes** (113 layers) | no (rope-garbage) |
| 4.57.x | no (random weights) | n/a |
| 4.56.1 | no (random weights) | n/a |

So 5.x is the only viable base; the task is to fix its forward.

---

## 5. Plan for option 1 (fix the 5.x forward)

**Step 0 — get a clean transformers 5.x env where the quantizer engages.**
Either switch `~/exl3/qlora-venv` to 5.x or make a parallel one. In qlora-venv:
```
pip install "transformers==5.10.2"
```
Watch for dep churn (it may pull newer tokenizers/hub; keep `pydantic<2.11`,
keep xformers uninstalled, keep `kernels` uninstalled). Re-confirm import:
```
python -c "import exllamav3, transformers; print(transformers.__version__)"
```
Sanity that the quantizer engages (want a count in the hundreds, not 0):
```
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM
from exllamav3.integration.transformers import patch_transformers, Exl3HfLinear
patch_transformers()
m=AutoModelForCausalLM.from_pretrained("/mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/", device_map="cuda", dtype=torch.float16)
print("EXL3 linears:", sum(isinstance(x,Exl3HfLinear) for x in m.modules()))
PY
```

**Step 1 — localize where hidden states go bad** (the probe that never ran):
```
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from exllamav3.integration.transformers import patch_transformers
patch_transformers()
d="/mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/"
tok=AutoTokenizer.from_pretrained(d)
m=AutoModelForCausalLM.from_pretrained(d, device_map="cuda", dtype=torch.float16)
ids=tok("The capital of France is", return_tensors="pt").input_ids.to(m.device)
out=m(input_ids=ids, output_hidden_states=True)
for i,h in enumerate(out.hidden_states):
    v=h[0,-1].float()
    print(f"h{i:2d} norm {v.norm():8.2f} absmax {v.abs().max():8.2f} nan {bool(torch.isnan(v).any())}")
print("top5:", tok.convert_ids_to_tokens(out.logits[0,-1].topk(5).indices.tolist()))
PY
```
Interpretation:
- norms explode / NaN at layer N → that block's attention/MLP (rope!).
- norms sane but top-5 junk → final norm or lm_head.
- already bad at h0/h1 → embedding / first block.

**Step 2 — confirm RoPE by differencing against the native (correct) forward.**
The native exllamav3 model loads and forwards correctly; use it as the oracle.
Compare HF vs native hidden states / attention for the same `input_ids` at layer 0
(q/k after rope). exllamav3's own rope implementation
(`exllamav3/util/rope.py`, `exllamav3_ext` rope, `RopeSettings/RopeStyle`) is the
reference for what the weights expect (llama3 scaling: factor 32, low 1, high 4,
orig_max 8192, theta 5e5).

**Step 3 — fix.** Most likely one of:
- Force transformers to compute the llama3 rope the 4.45-compatible way (override
  `config.rope_scaling`, or set the rotary implementation explicitly), or
- Patch the integration to inject a correct rotary embedding for these models, or
- If it turns out to be attention (e.g. an `attn_implementation` default change in
  5.x), set `attn_implementation="eager"`/`"sdpa"` explicitly at load.

Iterate against Step 1's probe until `loss` on "The capital of France is" is low
(~2–4) and top-5 is `[' Paris', ...]`.

**Step 4 — train for real & verify.**
```
CUDA_VISIBLE_DEVICES=0 python examples/qlora_train.py \
  --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/ \
  --out   /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/pirate2 \
  --sample-every 0
```
Expect first loss ~2–4 and dropping. Then verify the adapter on the **native**
path (transformers-independent, always works):
```
CUDA_VISIBLE_DEVICES=0 python examples/qlora_infer_native.py \
  --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/ \
  --adapter /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/pirate2
```
Success = BASE coherent English, ADAPTED coherent pirate ("Arrr", "matey", "th'").

**Trainer guard note:** on 5.x the "purely quantized" check is
`validate_quantization_for_training` (already monkeypatched in
`qlora_train.py`). If on some version it instead raises from `Trainer.__init__`
directly (as seen on 4.56, `trainer.py:566`), patch that path too / subclass to
skip. Not expected on 5.10.2.

---

## 6. Fallback (option 2, if 5.x forward proves unfixable)

Write a small **transformers-free differentiable Llama forward** (RMSNorm, GQA +
llama3 RoPE, SwiGLU) on exllamav3's native weights + `EXL3LoRAFunction` +
`FusedLinearCrossEntropy`. Validate logits against the native forward for one
input. ~200 lines, immune to transformers version drift. More work, but it can't
be broken upstream.

---

## 7. Quick reference — what's proven vs assumed

Proven (don't re-verify):
- `x @ get_weight_tensor()` == EXL3 kernel forward (rel_err 6.7e-4), orientation `[in,out]`.
- `EXL3LoRAFunction` and `FusedLinearCrossEntropy` backprops correct (gradcheck).
- Native inference + native `LoRA.from_directory` of our PEFT adapter work.
- CPU training-loop mechanics (mock EXL3 weight) reduce loss, freeze base, move adapters.

Assumed / unverified:
- That RoPE is the specific 5.x forward bug (strong hypothesis, not yet pinned).
- final-norm correctness on 5.x.
