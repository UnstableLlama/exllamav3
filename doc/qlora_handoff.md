# QLoRA-on-EXL3 — Handoff & Next-Step Plan

> Working session handoff. Branch: **`claude/magical-mayer-fq6z4i`**.
> Goal of the original task: prove whether **QLoRA fine-tuning on EXL3-quantized
> weights** is possible, and get a visible end-to-end demo (fine-tune a small
> model to talk like a pirate).

> **Maintenance note (2026-06):** the superseded HF-Trainer training path
> (`exllamav3/training/hf_qlora.py`, `examples/qlora_train.py`,
> `examples/qlora_infer.py`, and `tests/test_qlora_train_loop.py`) has been
> **removed**. The transformers-free native path (§0/§0b) is the only supported
> route. Sections §3–§6 below describe the removed code and the abandoned
> transformers-5.x investigation; they are kept for historical context only.
> All exllamav3-internal reach-through used by the native forward now lives in
> the single seam `exllamav3/training/backbone.py`.

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

### Session 3 — Yoda generated demo + EXL3-vs-BNB parity + the LoRA-on-quant finding

Long session. Three arcs: (A) build a dense Yoda dataset and prove the visible
demo; (B) stand up a controlled EXL3-vs-bitsandbytes QLoRA comparison; (C) a real
finding about why LoRAs look weak on EXL3-quantized bases at inference.

#### A. Yoda dataset + the density nuance

Picked **Yoda-speak** as the dense style (syntactic clause inversion over COMMON
tokens — unlike pirate/UwU). Off-the-shelf search was dry (only `dvgodoy/
yoda_sentences`, 720 translation pairs), so we **generate**.

- `examples/make_style_dataset.py` — rewrites ONLY the response of a normal
  instruction set (default `yahma/alpaca-cleaned`) into a target style via a LOCAL
  exllamav3 model; Alpaca-schema JSONL. Styles: `yoda`/`archaic`/`pirate`/
  `corporate`; `--refine-from` does a stricter second pass over an existing set.
  Use a strong instruct model as `--gen-model` (we used `TheDrummer_Rocinante-XL-
  16B-v1` 4bpw). Gotchas fixed live: create out-dir up front; stop at EOT (RP
  finetunes role-play a whole convo past the answer); drop `min_new_tokens` (it
  triggered a sampler cuda/cpu mask bug); reject prompt-echo/refusal rows.
- `examples/score_style_density.py` — Yoda-ness metric = clause-final inversion
  rate (sentences ending in aux/pronoun/contraction, plus a front-loaded
  "displaced subject" signal). **Known blind spot: noun-subject fronting ("Ended
  the war did") and main-verb endings — needs POS to catch — so the score is a
  conservative LOWER BOUND. Do NOT hard-filter on it** (it can't tell good
  noun-subject Yoda from junk; both score ~0).
- `qlora_train_native.py` loader now also reads a **local file path** for
  `--dataset` (json/jsonl/parquet/csv); DDP inherits it.

**THE density nuance (correction to S0c):** Yoda is dense at the SENTENCE level
but **sparse at the TOKEN level** — most words within an inverted sentence are
still normal order, so teacher-forced loss got low (~0.4) without the model
committing to inversions at greedy decode. So on the quantized base it needed
`--lora-scaling ~3` to surface (1B collapsed there; 3B held coherence and gave
clean Yoda). A density audit confirmed bimodality (~27% of rows barely inverted);
a Rocinante **refine pass** improved quality but the metric undercounted the gain
(it produces noun-subject Yoda the metric can't see). Dataset density was NOT the
bottleneck — see arc C for the real reason the 4bpw demo looked weak.

#### B. EXL3-vs-BNB QLoRA comparison harness (the flagship, parity check)

Goal: same model / data / LoRA / optimizer, only the frozen-weight format differs
(EXL3-4bpw vs bitsandbytes NF4), on Llama-3.2-3B, Yoda data. **Chose NOT to use
Axolotl** (it can't train EXL3 at all; mixing frameworks confounds; its dep tree
threatens the pinned torch/EXL3 `.so`). Instead a minimal matched loop:

- `examples/qlora_train_bnb.py` (NEW) — transformers 4-bit NF4 + PEFT LoRA in a
  hand loop mirroring `qlora_train_native.py` byte-for-byte (same Llama-3 chat
  prompt, completion-only masking, `datasets` shuffle(seed=0)+select, val split,
  LoRA targets/r/alpha, AdamW/lr/clip, bf16). Optional DDP (manual LoRA-grad
  all-reduce, same as the EXL3 DDP arm). Runs in the same venv (just
  `pip install bitsandbytes peft accelerate`) or an isolated one.
- Added to all three trainers: `--val-frac` + **identical held-out eval loss**
  (mean per-example, batch 1), `--eval-every`, `--save-best` (keep the best-val
  checkpoint; Ctrl-C won't clobber it), `[PERF]` tok/s + peak VRAM. `--gen-out`
  on the BNB trainer and `qlora_infer_native.py` dump samples for the scorer.
- Fixes found running it: NCCL teardown hang → pass `device_id` to
  `init_process_group`; `--r` eaten by torchrun → use `--lora-r`; **default
  `--max-samples` mismatch (bnb 4000 vs ddp 0) silently trained the arms on
  different data** → aligned to 0 (use all); match EFFECTIVE batch across arms
  (`--batch` is per-GPU under DDP).

**Overfitting caught by the eval curve:** at `lr 2e-4`, r=64, ~4 epochs, train
loss hit 0.09 while **held-out loss rose to 3.11** — the endpoint adapter was
memorized garbage (this, not the dataset, is why scale-3 gens degenerated). Fix:
`--lr 1e-4` + all data + `--save-best` → clean minimum ~2.0.

**Results (matched, 5329 train / 280 val, lr 1e-4, r64/a64, eff-batch 32):**
- **Held-out loss: near-identical (~2.0 both arms) — 4-bit PARITY confirmed.**
  EXL3 was a hair lower at matched steps.
- **EXL3 is more memory-efficient:** it ships fused-CE (never materializes the
  `[tokens×128k]` logits), so it fit `--batch 16`; the stock BNB/PEFT loss OOM'd
  at 16 on a 24GB card and needed `--batch 8 --grad-accum 2`.
- **EXL3 converged in ~⅓ the steps** — but that's an effective-LR/init difference
  between our `EXL3LoRAFunction` and PEFT (grad norms ~2.5 vs ~0.6 at the same
  nominal lr), NOT a quant property. Compare loss FLOORS, not steps.
- **The visible demo WORKS:** QLoRA-on-EXL3 Yoda, applied to bf16, gives clean
  coherent inversion ("Dinner for tonight, what should you? Many options, there
  are."). Both arms produce strong Yoda → quality parity too.

#### C. FINDING — LoRAs are attenuated on EXL3-quantized bases at inference

Controlled test (user's `ezexl3` frontend, correct Llama-3 templates, 0%/100%
scaling, BOTH the EXL3- and BNB-trained adapters, applied to BOTH the bf16 and the
4bpw base): **only the bf16 base produced strong steering; the 4bpw base
attenuated every adapter.** Correlated with LOW rank/alpha.

`Linear.apply_lora` is **base-agnostic** (`delta = x@a@b` added identically for
EXL3 and bf16; the `alpha/r` scale is folded into `B` at load) — so this is NOT an
application bug. Mechanism: the EXL3 trellis adds a per-output quantization
perturbation `x·ε`; a low-rank/low-alpha LoRA's delta is small and gets **buried
in `ε` on the quantized base**, while on bf16 (no `ε`) the same delta dominates and
the style shows. Higher rank/alpha grows the delta past the quant floor — exactly
the observed rank/alpha correlation. This is why our earlier "weak Yoda on 4bpw"
looked like a dataset/training problem when it was really a **signal-to-quant-noise
ratio at inference**; the adapter was good all along (loss + bf16 gens prove it).

**Implications:**
- Evaluate adapters on bf16 for a fair comparison (sidesteps a confound that hits
  both arms) — done; parity holds.
- Deploying a swappable low-rank adapter ON the EXL3 base is attenuated. Options:
  **(a) higher rank/alpha** so the delta clears the quant floor, or **(b) merge the
  adapter into bf16 and re-quantize** (delta becomes part of `W`; no small signal
  to bury — the clean deploy path, loses hot-swap).
- **This gets WORSE at 2.5–3bpw (bigger `ε`)** — so for the low-bitrate prize,
  merge-and-requantize is likely the right deployment story. Flagged before that
  experiment.

#### Recommended next steps
1. **Merge-and-requantize test** (chosen, arc C option b): merge the EXL3 Yoda
   adapter into bf16, re-quantize to 4bpw, confirm the style survives on the
   quantized base. Validates the real deployment path.
2. Optional: rank/alpha sweep on the 4bpw base to map the quant-floor threshold
   (turns the qualitative finding into a curve — useful to the exllamav3 community).
3. Then the headline low-bitrate run: EXL3-2.5/3bpw (where NF4 can't follow),
   deployed via merge-requantize, on a real metric.

### Session 4 — real-task training: schedulers, `messages`/test-split eval, prompt formats, full embed/head, BOS fix

> Branch `claude/epic-planck-n892gw`. Turned the proven QLoRA-on-EXL3 path into a
> real-task trainer, driven by a live run on **UnstableLlama/semancy** (a
> philosophy fine-tune: 436 train / 116 test, OpenAI-style single-turn `messages`,
> no system prompts). All code changes; the runs happen on the GPU box (2×3090).
> Everything below is **run-confirmed** unless noted.

**At a glance — what's new this session (all in `examples/qlora_train_native.py` +
the DDP variant; the BNB arm mirrors everything except `--prompt-format`,
`--inspect`, and `--train-embeddings/--train-head`):**

| Flag / change | What it does |
|---|---|
| `--messages-key` | Load OpenAI `messages` rows (user→prompt, assistant→supervised response). |
| `--scheduler {none,linear,cosine}` + `--warmup-ratio`/`--warmup-steps` | Transformers-free LR schedule with warmup (HF-equivalent `LambdaLR`). Default `none` = old constant LR. |
| `--weight-decay` (def 0.01) | Explicit AdamW weight decay (was torch's implicit 0.01). |
| `--epochs` | Derive `--steps` from train size × effective batch (folds in `world_size` under DDP). |
| `--eval-split` / `--eval-dataset` | Held-out eval loss on a **real** split (e.g. `test`) instead of carving `--val-frac` off train. |
| `--inspect N` | Decode the first N built examples (prompt vs supervised span, turn-end check) and exit. Tokenization gate; also a native-forward feasibility check. |
| `--prompt-format {auto,mistral,metharme}` | Chat format: model-native, explicit Mistral `[INST]`, or Pygmalion `<|user|>/<|model|>`. |
| `--train-embeddings` / `--train-head` | Fully train embed_tokens / lm_head (PEFT `modules_to_save`) and save them, not just LoRA. |
| **BOS fix** | `build_sft_examples` was emitting a **doubled `<|begin_of_text|>`** + a stray BOS on the response; now normalized to exactly one. (Found by `--inspect`.) |

Shared helpers (`build_sft_examples`, `collate`, `make_lr_scheduler`,
`resolve_steps_and_warmup`, `format_prompt_and_eot`, `extract_single_turn`) live
in `qlora_train_native.py`; the DDP script imports them. The BNB arm
(`qlora_train_bnb.py`) inlines byte-identical copies (separate venv, can't import
the exllamav3 path) for `--messages-key`, scheduler/warmup/`--weight-decay`,
`--epochs`, and `--eval-split` — so a matched EXL3-vs-NF4 run needs only the same
flags on both. (`--prompt-format` and `--train-embeddings/--train-head` are
native-arm only for now.)

**New trainer features (both `qlora_train_native.py` and the DDP variant; shared
helpers live in `qlora_train_native.py` and are imported by the DDP script):**
- **`--messages-key`** — OpenAI `messages` loader. For single-turn rows it takes
  the user turn as the prompt and the assistant turn as the supervised response
  (system turns ignored; the dataset has none). `extract_single_turn()` does the
  pull; the rest of the completion-only masking path is unchanged, so the answer
  (plus `<|eot_id|>`) is supervised and the prompt is `-100`. Takes precedence
  over the flat `--instruction/context/response-key`.
- **LR schedulers** — `--scheduler {none,linear,cosine}` + `--warmup-ratio`
  (fraction of steps) or `--warmup-steps` (absolute). `make_lr_scheduler()` is a
  transformers-free `LambdaLR` matching HF's `get_{linear,cosine}_schedule_with_warmup`
  exactly: LR ramps 0→base over warmup, then linear-to-0 or half-cosine-to-0.
  Default `none` keeps the old constant-LR behavior (matched-arm runs unaffected).
  Per-step `sched.step()`; current LR is logged each step.
- **`--weight-decay`** (default 0.01) — now explicit on the AdamW over the LoRA
  params (torch's AdamW default was already 0.01; this just makes it a knob).
- **`--epochs`** — if >0, computes `--steps` from the train-set size and the
  *effective* batch (`batch*grad_accum`, ×`world_size` under DDP), so the schedule
  length matches the requested passes. e.g. 436 rows, eff-batch 16, 2 epochs → 56
  steps (warmup 6 at ratio 0.1).
- **`--eval-split` / `--eval-dataset`** — held-out eval loss on a *real* split
  (e.g. semancy's 116-row `test`) instead of carving `--val-frac` off train.
  Built identically on every rank under DDP; works with `--eval-every`/`--save-best`.
- **`--inspect N`** (single-GPU script) — decodes the first N built examples,
  showing the masked prompt span vs the supervised response span and **warning if
  the response was truncated by `--seq-len`** (so the model would never see the
  turn-end token). Run once to verify tokenization on any new dataset/schema
  before a long run.

**Tokenization** for the `messages` path: `default_chat_prompt()` (Llama) injects
**no** system prompt when none is passed, emits the standard Llama-3 template
ending at the assistant header; the response is tokenized separately and
terminated with the architecture-correct turn-end token, prompt masked `-100`.

**Doubled-BOS bug found by `--inspect` (fixed).** Running `--inspect 3` on the GPU
box revealed every prompt began with `<|begin_of_text|><|begin_of_text|>` and the
**response span began with a stray `<|begin_of_text|>`**. Cause: the special-token
encode path (`tokenizer.py:243/254`) calls the underlying HF tokenizer with
`add_special_tokens=True`, and the Llama-3 tokenizer has `add_bos_token=true`, so
it auto-prepends BOS on *every* `encode()` — on top of the literal BOS in the chat
template, and again on the separately-encoded response. exllamav3's own `add_bos`
flag is independent of this. This is non-standard (real Llama-3 = exactly one BOS,
none mid-sequence) and meant the EXL3 arm wasn't even BOS-matched to the BNB arm
(which uses `add_special_tokens=False` and so gets a single BOS). **Fix:**
`build_sft_examples` now normalizes to one leading BOS and strips the response's
spurious one (no-op for tokenizers that don't auto-prepend, so it's general). The
fix lives in the trainer, NOT the tokenizer (the inference path relies on the
auto-BOS). Prior pirate/Yoda runs carried this double BOS and still trained/
validated (the forward math was unaffected), but new runs should use the fix.
Pure-logic checks (scheduler shape, epoch→step math, messages extraction, BOS
normalization) pass; the rest needs the GPU box (no torch/CUDA in the container).

**Recommended semancy run (per the research plan: r=16/α=32, lr 1e-4, cosine,
~10% warmup, wd 0.01, eff-batch 16, completions-only, all linear targets, 2 epochs):**
```
# 0. Verify tokenization first (single-GPU, exits after printing):
python examples/qlora_train_native.py --model /path/to/exl3_model \
    --dataset UnstableLlama/semancy --messages-key messages \
    --no-clean-text --seq-len 2048 --inspect 3
#    Confirm: PROMPT is the user turn, RESPONSE is the assistant turn, and
#    "ends with turn-end token? True" (else raise --seq-len).

# 1. Train (2× GPU DDP; --batch 8 × 2 GPUs = eff-batch 16):
torchrun --standalone --nproc_per_node=2 examples/qlora_train_native_ddp.py \
    --model /path/to/exl3_model --out /path/to/out/semancy \
    --dataset UnstableLlama/semancy --messages-key messages --no-clean-text \
    --eval-split test --eval-every 10 --save-best \
    --lora-r 16 --alpha 32 --lr 1e-4 --scheduler cosine --warmup-ratio 0.1 \
    --weight-decay 0.01 --batch 8 --grad-accum 1 --epochs 2 --seq-len 2048
```
- **`--no-clean-text` is important here**: the default cleaner strips `[...]`/`*...*`
  and collapses newlines — fine for play-script style sets, **wrong for a
  reasoning dataset** (would delete bracketed content and paragraph structure).
- **`--seq-len`**: philosophy/reasoning answers can be long; 512 (the default)
  likely truncates them. Use `--inspect` to check, then size `--seq-len` so most
  responses end in the turn-end token. Bigger seq-len costs VRAM (drop `--batch`
  or keep grad-checkpointing on if OOM).
- Recall **finding C**: a low-rank adapter (r=16) is *attenuated* on the EXL3
  base at inference. Evaluate on bf16 (or merge-and-requantize) for a fair read;
  the held-out `test` loss is the format-independent signal.

**BNB-NF4 arm matched too.** `examples/qlora_train_bnb.py` got the same flags
(`--messages-key`, `--scheduler`/`--warmup-ratio`/`--warmup-steps`,
`--weight-decay`, `--epochs`, `--eval-split`/`--eval-dataset`) with the helpers
*inlined* (it runs in the separate transformers+bitsandbytes+peft venv and can't
import the exllamav3 path) but byte-identical to the EXL3 arm, so a matched
EXL3-vs-NF4 comparison on semancy just needs the same flags on both:
```
# EXL3-4bpw arm: the DDP command above.
# BNB-NF4 arm (point --model at the bf16 HF weights; same eff-batch via --batch):
~/exl3/bnb-venv/bin/torchrun --standalone --nproc_per_node=2 \
    examples/qlora_train_bnb.py --model /path/to/Llama-bf16 --out /path/to/out/semancy_bnb \
    --dataset UnstableLlama/semancy --messages-key messages --no-clean-text \
    --eval-split test --eval-every 10 --save-best \
    --lora-r 16 --alpha 32 --lr 1e-4 --scheduler cosine --warmup-ratio 0.1 \
    --weight-decay 0.01 --batch 8 --grad-accum 1 --epochs 2 --seq-len 2048
```
Compare the held-out `test` loss floors (per S3-B: compare floors, not steps —
the arms differ in effective LR/init). BNB may need a smaller `--batch` +
`--grad-accum` to hit eff-batch 16 within 24GB (it lacks the EXL3 arm's fused-CE).

**semancy run result (Llama-3.2-1B 4bpw, 2×3090 DDP, r16/α32, lr1e-4 cosine,
warmup6, 56 steps):** clean — both GPUs, cosine schedule exactly as designed
(LR→1e-4 by step6, →0 at step56). Train loss 3.53→2.14; held-out `test` loss
3.30→**3.09** (`--save-best`); peak VRAM **5.26 GB/GPU** (huge headroom); 179s.
Eval flattened ~3.09 after step30 with a widening train/eval gap → near plateau;
3.09 is high vs the style runs (~2.0) but expected (dense reasoning + a
deliberately-hard generalization test split, not memorization). Likely
capacity-bound on 1B — a 3B/8B base is the bigger lever than more 1B epochs.

### Prompt formats (`--prompt-format`) + Mistral

`build_sft_examples` got a `--prompt-format {auto,mistral,metharme}` knob (single +
DDP; `format_prompt_and_eot()` is the seam):
- **auto** (default, unchanged): the model's own template via
  `default_chat_prompt` — Llama-3 headers, Mistral `<s>[INST] … [/INST]`, etc.,
  with the architecture-correct turn-end token.
- **mistral**: explicit `<s>[INST]{q}[/INST]{a}</s>` (no spaces; `[INST]`/`[/INST]`
  are control tokens). Identical to **auto** for the `mistral3` arch — which is
  what **Mistral-Medium-3.5-128B** (and Small/Medium 3.x) loads as: its
  `mistral3.py:243` `default_chat_prompt` already emits
  `<s>[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT][INST]{q}[/INST]` (the current V13
  control-token format), so **Medium 3.5 needs no new format — auto already
  covers it**; `mistral` just makes it explicit / arch-independent. (Caveats for
  actually training it: it's a 128B — needs a multi-GPU EXL3 quant; and `--inspect`
  doubles as the native-forward feasibility check since it builds the net and
  `assert_block_supported` would reject an unsupported block.)
- **metharme**: Pygmalion format `<s><|user|>{q}<|model|>{a}</s>`. On a base
  model the `<|user|>`/`<|model|>` markers are **plain text** (not registered
  special tokens) — the model learns them as a literal pattern, the standard way
  these tunes train; EOS ends the turn. The literal `<s>` + the existing
  BOS-normalization → exactly one leading BOS, none mid-sequence (verified).
  Mistral did **not** do this before (its `default_chat_prompt` is `[INST]`).

### Training the embeddings + LM head (`--train-embeddings` / `--train-head`)

LoRA freezes `embed_tokens`/`lm_head`; these flags fully train them
(PEFT `modules_to_save` semantics) and save them. Single + DDP; in
`NativeLlamaQLoRA`:
- A trainable **fp32 copy** of the embedding (`[vocab,hidden]`) and/or head
  (`[hidden,vocab]`) is reconstructed once from the frozen base and placed on the
  GPU compute device (the base embedding is on CPU under `prefer_cpu`; a CPU
  optimizer would crawl). exllamav3 loads a *separate* embed and head even for a
  tied model, so they're trained **independently** — no shared-param special case
  (the saved config records `tie_word_embeddings` only as a merge-time hint).
- **Head loss path:** the fused-CE head is frozen-only (returns no weight grad),
  so `--train-head` switches `compute_loss` to a standard-autograd cross-entropy
  computed **only at the supervised positions** (labels≠ignore) — the head gets a
  gradient and memory scales with supervised tokens, not the full `[tokens,vocab]`.
- **Grad-checkpointing fix:** `forward` used to `detach()` the embedding (frozen
  assumption); it now only detaches when `hidden` doesn't already require grad, so
  a trainable embedding's gradient isn't severed.
- Optimizer: embed/head go in a **0-weight-decay** group (`param_groups()`);
  decaying a whole embedding table is harmful. LoRA keeps its weight decay.
- Saved to `modules_to_save.safetensors` (HF orientation `[vocab,hidden]`) beside
  the adapter, kept OUT of `adapter_model.safetensors` so the LoRA loader is
  undisturbed; `adapter_config.json` lists `modules_to_save`. `load_adapter`
  restores them on resume. CPU test: `test_modules_to_save_param_groups`.

**Cost / caveats:** these are big matrices. On the tied 1B (vocab 128k×2048 ≈
262M params) that's ~4 GB (fp32 master+grad+Adam m,v) — fine. On a 16B (vocab
131k×~6144 ≈ 805M *each*, untied) it's ~13 GB *per* matrix — under **DDP it's
replicated per card AND the grad is all-reduced every step** (the embed/head grad,
not just the few-MB LoRA grad), so `--train-embeddings`/`--train-head` on a 16B
will OOM 24 GB and is slow over PCIe. Realistic on small models, or large models
only with `--parallel split` / more VRAM. Live samples + the native infer path do
NOT yet reflect a trained embed/head (only the runtime LoRA slots are wired) — the
trained matrices are for the merge/re-quantize deploy path.

**Mistral caveat:** the native forward (`backbone.assert_block_supported`,
backbone.py:71) **rejects sliding-window attention**, so Mistral-7B-v0.1
(sliding_window=4096) won't load; v0.2/v0.3/Nemo/Small (sliding_window=null) are
fine. Verify a new Mistral base with `--inspect` first (it now reports the
metharme spans + the EOS turn-end check).

Run a Mistral metharme SFT by adding `--prompt-format metharme` to the usual
command (point `--model` at a no-sliding-window Mistral EXL3). Inference must use
the same metharme template for the adapter to fire (the single-GPU live `sample`
already does; set it in your own frontend for real inference). The BNB arm still
uses `[INST]`/native only — mirror `--prompt-format` into `qlora_train_bnb.py` if
a metharme EXL3-vs-NF4 comparison is wanted.

### Session 4 — status, run-confirmed, and open items

**Run-confirmed on the box (2×3090):**
- semancy DDP run end-to-end on Llama-3.2-1B 4bpw: cosine schedule, `--epochs`,
  `--eval-split test`, `--save-best` all working (result above; eval `test` loss
  ~3.09, 5.26 GB/GPU, 179 s).
- `--inspect` on the 1B (Llama) and on **Rocinante-XL-16B** (mistral, metharme):
  both show one BOS / correct spans / `ends with turn-end token? True`. The 16B
  loaded fine through the native forward (no sliding-window rejection), so
  mistral-family + metharme works on a real 16B.
- `--prompt-format metharme` on the 16B produced exactly
  `<s><|user|>{q}<|model|>{a}</s>`.
- BOS fix verified (the doubled-BOS dump → single BOS after the fix).

**Not yet run (write-confirmed only — verify before relying on):**
- `--train-embeddings` / `--train-head` on real hardware (logic + a CPU test pass;
  the head-grad path and the save/load round-trip haven't run on GPU yet). Smoke
  it on the 1B first.
- The 16B **metharme training** run itself (only `--inspect` was run; use DDP
  `--batch 2 --grad-accum 4` for eff-batch 16, or `--batch 1 --grad-accum 8` if it
  OOMs on load).
- Mistral-Medium-3.5: it's the `mistral3` arch and `--prompt-format auto`/`mistral`
  already emit its V13 `[SYSTEM_PROMPT]`/`[INST]` format, but no EXL3 quant of a
  128B was trained (needs multi-GPU `--parallel split`).

**Open items / would-be-nice next:**
- Mirror `--prompt-format` and `--train-embeddings/--train-head` into the BNB arm
  (PEFT supports `modules_to_save` natively) for matched comparisons.
- Make live samples / the native infer path reflect a trained embed/head (today
  only the runtime LoRA slots are wired; trained matrices are for merge/requantize).
- Tests for the new paths need a GPU/model (CPU tests can't build the full net):
  only `test_modules_to_save_param_groups` is pure-CPU. Run `tests/test_native_llama.py`
  + `tests/test_fused_ce.py` + `tests/test_qlora_grad.py` after pulling.
- For semancy specifically: a bigger base (3B/8B) is the likely win over more 1B
  epochs (the ~3.09 floor looks capacity-bound); judge style on bf16 / merge per
  finding C, not on the attenuated 4bpw base.

### Session 5 — broader architectures: Gemma3/4 + Qwen3 dense in the native forward

> Goal: train current dense models (Gemma4, Qwen3.x) on the EXL3 native path.
> The native forward was scoped to one block shape (pre-norm GQA + NeoX-RoPE
> softmax attention + pre-norm SiLU GatedMLP) and `assert_block_supported`
> rejected everything else. This session **generalizes** that forward instead of
> forking a per-arch one.

**The triage (read the arch files, not memory — exllamav3 already has inference
support for all of these):**
- **Qwen3.5 / 3.6 are hybrid linear-attention** (`architecture/qwen3_5.py`): even
  the non-MoE `Qwen3_5ForCausalLM` builds ~3/4 of layers as **`GatedDeltaNet`**
  (delta rule + exp gating + causal Conv1D + L2 q/k-norm) and the rest as gated
  full-attention. Training them needs a **differentiable Gated DeltaNet** (its own
  recurrent forward+backward) — a separate research-grade project, **not done**.
  This is the hard blocker; "dense" ≠ standard transformer for Qwen3.5/3.6.
- **Gemma4 dense text** (`architecture/gemma4.py`) is **softmax** attention — no
  linear-attn blocker — but adds: q/k/v-norm, **sandwich post-norms**, **GeGLU**,
  alternating **sliding/full** layers with **per-layer head dims**, Gemma `(1+w)`
  RMSNorm, embedding scaling, optional logit softcapping. All differentiable.
- **Qwen3 (3.0) dense** = standard transformer + **q/k-norm** only (falls out for
  free from the generalization).

**RUN-CONFIRMED on the box (gemma-4-12B-it 4bpw, 1×GPU, fp32 eager):**
`qlora_validate_native.py` **PASSED** — 100% per-position argmax agreement vs
exllamav3's own forward on every prompt, last-token `cos ≈ 0.999998`,
`max|Δ| ≈ 0.08` (just fp32-vs-native-fp16 rounding), `--check-backward` PASS.
The decisive bug was the missing **per-layer `layer_scalar`** (below): without it
the forward was per-block-correct but compounded to garbage over 48 layers (0%
argmax, cos~0). The `sm_scale=1.0` question is settled — correct as-is (Gemma's
query scaling is folded into the weights; the near-perfect cosine confirms it).
Still to run: the bf16 flash/SDPA path (`--compute-dtype bfloat16`) and a real
training run.

**What was changed (compiles + CPU logic checks pass; forward now GPU-validated on
gemma-4-12B as above):**
- `training/backbone.py`: `assert_block_supported` relaxed to allow q/k/v-norm,
  GeGLU, sliding window, sandwich post-norms and softcap, while still rejecting
  GatedDeltaNet (linear attn), MoE, attention output gating, mRoPE, partial
  rotary and non-NeoX RoPE. New `norm_spec()` (reproduces `RMSNorm.forward_torch`:
  `y = x/rms(x) * constant_scale * (weight + constant_bias)`, so Gemma's `(1+w)`
  and unweighted v-norm are read from the module, not hardcoded), `block_post_norms()`,
  `attn_qkv_norms()`, `head_softcap()`; `block_metadata` gains `sliding_window`,
  `softcap`, `activation`, `use_k_as_v`.
- `training/native_llama.py`: `_rmsnorm` → spec-driven `_norm`; `_block_forward`
  now does per-head q/k/v-norm (pre-RoPE), attn softcap, sandwich post-norms
  (`x = x + post_norm(sublayer_out)` — confirmed against the fused `rms_norm` CUDA
  kernel, `norm.cu:205`), GeGLU, and `use_k_as_v` (V = raw K projection); `_attn_bias`
  gains a sliding-window band; `forward` builds an attention bias per
  (window, device); final norm uses `_norm`; final-logit softcapping handled in
  `logits()`/`compute_loss()` (the fused-CE path can't softcap, so it falls back
  to a supervised-position CE when a final cap is present). **Reduces
  bit-identically to the old Llama/Mistral/Qwen2 path when all features are off.**
- `tests/test_native_llama.py`: updated to the new entry/meta interface and added
  `test_gemma_block_matches_reference` (q/k/v-norm + sandwich + GeGLU + sliding +
  softcap + `(1+w)`) vs an independent reference.

**Loading:** `Model.from_config` defaults to `component="text"`, so a multimodal
`Gemma4ForConditionalGeneration` checkpoint loads its **text decoder** only; the
`split_decoder` embedding-first/head-last layout holds. Use `--sample-every 0` for
a first Gemma4 run (live generation would exercise the SWA/recurrent cache path,
irrelevant to training).

**MUST DO before trusting a Gemma4 run (the correctness gate):**
```
# CPU first (torch only): the differentiable block vs an independent reference.
python tests/test_native_llama.py
# Then on the box: logits parity vs exllamav3's own forward on a real Gemma4 dense:
python examples/qlora_validate_native.py --model /path/to/Gemma4-dense-exl3
#   Expect high per-position argmax agreement + tiny last-token max|Δ|.
# Only then: python examples/qlora_train_native.py --model ... --sample-every 0 ...
```
Open: q/k-norm is folded into exllamav3's RoPE kernel in the native path (we apply
norm-then-RoPE, mathematically equal) — the validate gate is what confirms it on
real weights. Gemma4's `sm_scale=1.0` (vs Gemma2/3 `query_pre_attn_scalar**-0.5`)
is read from the module, so no special-casing — but eyeball it in validation.

#### FlashAttention-2 fast path (long-context training)

The native block forward was **eager** — it materialized the `[b, n_q, t, t]`
score matrix in fp32 — which is the O(t²) memory wall at long context.
exllamav3's own FA2 is inference-only (`@torch.inference_mode` kernels), so the
training path now uses the upstream **`flash_attn` package's autograd-capable
`flash_attn_func`** instead (already installed in the qlora-venv, 2.8.3).

- `native_llama._block_forward` gained a `use_flash` branch: flash takes
  `[b, t, nh, hd]` directly (no transpose / no GQA expand), and the causal mask,
  Gemma sliding window (`window_size=(W-1, 0)`) and softcap go through flags — so
  the eager `[t, t]` bias is **not built** when flash is active (that omission is
  what realizes the saving). Eager stays as the reference / CPU / fp32 / gradcheck
  path.
- `--attn-impl {auto,eager,flash}` on both trainers + the validate script
  (`NativeLlamaQLoRA(attn_impl=...)`). **auto** = flash when the package imports
  AND the run is CUDA fp16/bf16, decided per-forward; fp32 / CPU always run eager.
- **VRAM saved ≈ ~2 × b · n_q · t² · 4 bytes** (the eliminated score matrix +
  softmax). With grad-checkpointing on (default) that is the **per-block** peak —
  independent of model depth / hidden size, only `b · n_q · t²`. Examples (b=1):
  n_q=32 → ~4 GB at t=4k, ~16 GB at t=8k; n_q=16 (Gemma) → half. Net effect:
  roughly 4–8× more context (or batch) on the same card in the t=4k–16k range.
  Flash does NOT touch the residual-stream checkpoints (`~layers · b · t · d · 4`),
  which become the next term at extreme context.
- **Per-block mode (Gemma mixes head sizes).** FA2 only supports `head_dim <= 256`,
  but Gemma4's *global* layers exceed that. So the mode is chosen per block:
  **flash** (head_dim ≤ 256, % 8 — the sliding/local layers), **SDPA**
  (`F.scaled_dot_product_attention`, head_dim > 256 full-causal no-softcap — keeps
  the big-head global layers O(t) too, so they don't re-impose the t² peak), else
  **eager**. Only eager blocks build the `[t, t]` bias.
- **Validation:** the default fp32 `qlora_validate_native.py` exercises *eager*
  (mem-efficient kernels need fp16/bf16). To validate flash+SDPA, run with
  `--compute-dtype bfloat16`; expect a slightly looser match than fp32 eager, still
  high argmax agreement. Relies on right-padding (which `collate` guarantees).

**Two upstream Gemma4 native-forward fixes (needed for the validate oracle / live
sampling on GPU, hit while running the gate on gemma-4-12B):**
- `architecture/gemma4.py` `_prepare_noncausal_mm_spans`: built boundary tensors
  with CPU literals and cat'd them with a CUDA tensor → device-mismatch crash on
  any GPU text forward. Now built on `ids.device`.
- `modules/attn.py`: the no-cache SDPA fallback (taken for `head_dim > 256` since
  the `bighead` kernel is paged/cache-only) returns a transposed, non-contiguous
  `[b, t, nh, hd]`; `o.view(...)` → `o.reshape(...)` in both `decode_flash_attn`
  paths. (xformers is NOT required — it's optional and was ABI-mismatched in the
  venv; the SDPA fallback is correct, just slower.)

Status: flash/SDPA path is GPU-only (CPU suite covers the eager reference). The
bf16 flash/SDPA path itself is **run-confirmed indirectly** — the Gemma4-12B
training run below descends cleanly in bf16 (which engages flash+SDPA) — but the
dedicated bf16 `qlora_validate_native.py --compute-dtype bfloat16` parity pass is
still worth running once (see Session 6 open items).

---

### Session 6 — Gemma4 run-confirmed end-to-end; tooling (logging, dual-eval, UX)

> Branch `claude/confident-goodall-9bnuwp`. Long session. Turned the Gemma4
> support into a confirmed working trainer on real hardware (gemma-4-12B-it 4bpw)
> and built out the run-quality tooling. Everything below is **run-confirmed on
> the box (1×GPU and 2×GPU DDP)** unless marked otherwise.

**Headline: QLoRA-on-EXL3 works on Gemma4-12B.**
- Forward parity gate PASSED (fp32): `qlora_validate_native.py` on gemma-4-12B-it
  4bpw → **100% per-position argmax agreement** vs exllamav3's own forward,
  last-token `cos ≈ 0.999998`, `max|Δ| ≈ 0.08`.
- Training run confirmed (semancy, 1×GPU r16/α32 lr1e-4 cosine, and 2×GPU DDP
  lr5e-5 eff-batch16): clean descent (train 5.1→2.4), held-out `test` **2.51**
  (vs the 1B's ~3.09 floor — bigger base is the lever, as predicted). The
  wikitext dual-eval *also* dropped during the philosophy SFT = **positive
  transfer, not forgetting** (watch for the eventual crossover under harder/longer
  training — that's the forgetting signal the dual-eval exists to catch).

**The decisive Gemma4 bug (and two upstream fixes):**
- **`layer_scalar`** — Gemma4's `TransformerBlock` carries
  `key_layer_scalar="layer_scalar"` and multiplies the whole residual stream by a
  learned per-layer scalar at block end (`transformer.py:222`). Omitting it made
  the forward per-block-correct but **garbage over 48 layers** (0% argmax,
  cos~0). Now read via `backbone.block_metadata["layer_scalar"]` and applied in
  `_block_forward`. This was *the* fix that took the validate from FAIL → PASS.
- **`architecture/gemma4.py`** — `_prepare_noncausal_mm_spans` built boundary
  tensors with CPU literals and `cat`'d them with a CUDA tensor → device-mismatch
  crash on any GPU text forward. Build on `ids.device`.
- **`modules/attn.py`** — the no-cache SDPA fallback (reached for `head_dim>256`;
  Gemma4 global layers, since the `bighead` kernel is paged/cache-only) returns a
  transposed non-contiguous tensor; `o.view(...)` → `o.reshape(...)` in both
  `decode_flash_attn` paths. (xformers is **not** required — it was ABI-mismatched
  in the venv; the SDPA fallback is correct, just slower.)
- `sm_scale=1.0` for Gemma4 (vs Gemma2/3 `query_pre_attn_scalar**-0.5`) is
  confirmed correct as-is (read from the module; the ~perfect cosine proves the
  query scaling is folded into the weights).

**FlashAttention-2 fast path is now per-block + visible:**
- `_block_forward` picks **flash** (head_dim≤256), **SDPA** (head_dim>256
  full-causal no-softcap — keeps Gemma's big-head global layers O(t)), or **eager**
  per block; `describe_attn()` prints the plan at startup (e.g.
  `attn: 40×flash, 8×sdpa [impl=auto, flash_attn available]`). `--attn-impl
  {auto,eager,flash}` on the trainers + validate. (Fixed a cosmetic bug where the
  summary mislabeled everything `eager` because it tested `block.device.type`
  on a device *string*; the forward itself keys off the live `hidden.is_cuda` and
  was always correct.)

**Run-quality tooling added this session (all three arms — native single, DDP,
BNB — unless noted):**
- **`--run-log` CSV (default `qlora_runs.csv`)** — one metadata row per run,
  written on normal finish AND Ctrl-C (`status=completed|interrupted`): model,
  arch, all hyperparameters, eff-batch, train/val/eval2 sizes, start/end train
  loss, **start_val/start_eval2 (baseline)**, best_val + best_val_step,
  final_val/final_eval2, total_s, s_per_step, sup/tot tok/s, peak VRAM. Shared
  `append_run_log`+`RUN_LOG_FIELDS` in `qlora_train_native.py` (DDP imports; BNB
  inlines an identical copy). **Self-heals** on schema change: if an existing
  file's header differs it moves the old file to `<path>.bak` and starts fresh.
- **Live rolling tok/s** on the per-step line (`ThroughputMeter`, train-step
  compute only); `[PERF]` now reports both supervised and total tok/s.
- **`--shuffle` / `--shuffle-seed`** — deterministically shuffle rows before the
  `--val-frac` carve. With `--eval-split` (a real test split) train is shuffled
  and the split is preserved (they're separate `load_dataset` calls).
- **Second eval set** — `--eval2-dataset` (+ `--eval2-split` / `--eval2-config` /
  `--eval2-text-key` / `--eval2-max-samples`). With `--eval2-text-key` it's a
  plain-text LM loss over packed `seq_len` blocks (e.g. wikitext); `--eval2-config`
  supplies the HF config (`wikitext-2-raw-v1`). Each block now starts with **one
  BOS** when the model uses one (gated on `bos_token_id`; Qwen → none), so the
  absolute LM number is meaningful, not just the trend. `build_lm_examples` shared
  (native) / inlined (BNB) with matched tokenization, so EXL3-vs-NF4 is comparable.
- **Baseline (step-0) eval** — when an eval set is selected, the held-out (and
  eval2) loss is computed once before step 1 (no-op adapter = base model) and
  printed `[eval] step 0 (baseline): ...`; logged as `start_val`/`start_eval2`.
  Training timer + VRAM peak (re)start after it.
- **No more "looks hung after Done."** — the post-loop final eval is reused from
  the last in-loop eval when it landed on the final step (else computed once with
  a `-- computing final held-out eval (GPU busy, not hung) ...` notice). Was a
  duplicate full pass on a big model.
- **torchrun redirect** — `qlora_train_native.py` launched under torchrun
  (RANK/WORLD_SIZE in env) exits with a one-line pointer to the DDP script instead
  of an argparse wall (the intuitive `--parallel ddp` footgun).

**Throughput note:** native EXL3 training is ~136 tok/s (1×) / ~215 tok/s (2×GPU
DDP, all-ranks est) on the 12B — correctness-first, not speed-tuned (trellis
reconstruction ×3 under checkpointing + SDPA on the big-head global layers). The
`[PERF]`/`s_per_step` CSV columns are there to track this across configs.

**Open items / next day:**
1. **Run the bf16 parity pass**: `qlora_validate_native.py --compute-dtype
   bfloat16` on gemma-4-12B to formally confirm the flash/SDPA logits vs native
   (expect looser than fp32 — ~0.99x cos — but high argmax). Confirms the
   `attn:` line shows flash actually engaging.
2. **Matched EXL3-vs-NF4 Gemma4 run** (the flagship): same flags on
   `qlora_train_bnb.py` (point `--model` at bf16 HF Gemma4); compare held-out
   floors + the run-log rows. Note BNB lacks `--prompt-format`/Gemma specifics —
   verify the chat formatting matches.
3. **Qwen3.5 / 3.6** remain unsupported — they need a differentiable **Gated
   DeltaNet** (linear/recurrent attention; ~3/4 of layers). This is the open
   research frontier; start with a single-layer forward-parity check against
   exllamav3's `GatedDeltaNet` before committing to the backward.
4. Speed: if wall-clock matters for long runs, profile trellis-reconstruction vs
   SDPA-big-head cost; a fused/ cached reconstruction or training-time flash for
   head_dim>256 (when a kernel supports it) are the levers.

### Session 7 — retained checkpoint history (`--checkpoint-every`) + the three save modes clarified

> Goal: keep a rollback-able history of the LoRA across a run, not just one
> latest/best/endpoint copy. Also pinned down the "is it still saving the lowest
> held-out val?" question.

**The three save modes are now distinct (single-GPU + DDP):**
| Flag | Where it writes | Behavior |
|---|---|---|
| `--save-every N` | `--out` (overwritten) | One **latest** copy, refreshed every N steps. |
| `--save-best` | `--out` (overwritten) | One **best-val** copy; only rewritten when the **PRIMARY** held-out eval improves (`val_examples` from `--eval-split`/`--val-frac`; eval2/wikitext does NOT drive it). |
| **`--checkpoint-every N`** (NEW) | `--out/checkpoint-<step>` (retained) | A **history** — each is kept, never overwritten. Roll back / pick from any. |

- **`--checkpoint-every N`** saves the adapter to `--out/checkpoint-<step>`
  (zero-padded 8-wide so `ls` sorts numerically) every N steps, independent of the
  other two modes. **`--keep-checkpoints K`** caps retention (delete oldest;
  `0` = keep all) — matters under `--train-embeddings/--train-head` where each
  checkpoint carries the big embed/head matrices, not just the few-MB LoRA.
- Shared helpers `checkpoint_dir()`/`list_checkpoints()`/`prune_checkpoints()` live
  in `qlora_train_native.py`; the DDP script imports them and writes from rank 0
  only with a `dist.barrier()` (same single-writer pattern as `save()`). Not yet
  mirrored into the BNB arm (separate venv; add if a matched run needs it).

**Full resume (`--resume` now restores optimizer + LR schedule + step):** every
save target (the best at `--out`, a `--save-every` copy, and each
`--checkpoint-every` dir) now also writes a small **`trainer_state.pt`** beside
the adapter (AdamW state_dict + `LambdaLR` state + `step` + `best_val`/`ema`). On
`--resume <dir>` the trainer loads it (when present) and **continues**: the
optimizer moments, the warmup/cosine position, and the step counter all pick up
where they left off, instead of the old behavior of cold-restarting the schedule
from step 0 (which would replay warmup mid-run — wrong for the short 56-step
semancy cosine). `--reset-optimizer` keeps the old weights-only behavior (use when
changing LR/schedule or resuming across a different GPU count / device topology).
A dir with only adapter weights (a pre-Session-7 checkpoint or a foreign PEFT
adapter) falls back to weights-only automatically. Helpers
`save_trainer_state()`/`load_trainer_state()`/`restore_optimizer_state()` are
shared (DDP imports them); under `--parallel split` `restore_optimizer_state`
moves each AdamW tensor onto its param's device (params span GPUs); under DDP
every rank loads the same rank-0 state (ranks are identical after each
all-reduce). `torch.load(weights_only=False)` since it's our own trusted file.
**Write-confirmed only** (no torch in the container) — smoke a stop/resume on the
box: run a few steps with `--checkpoint-every`, Ctrl-C, then `--resume
<out>` and confirm the printed `continuing at step N` + LR match where it stopped.

**Two ergonomics fixes (during the live gemma-4-31B run):**
- **`--eval2-max-blocks N`** — caps the number of packed eval2 LM blocks directly
  (wikitext packed into 285 blocks at seq-len 1024, swamping the 116-example
  primary `test` set). `--eval2-max-samples` only caps *source rows*, which is
  unpredictable after packing; `--eval2-max-blocks 116` sizes eval2 to ~match the
  primary set. Added to `build_lm_examples` (shared, so DDP gets it via import) +
  both trainers' CLI. (BNB arm inlines `build_lm_examples` — mirror there if a
  matched run needs it.)
- **Text cleaning is now opt-in (`--clean-text`), was opt-out (`--no-clean-text`).**
  The default is now **no cleaning** — right for the reasoning/code/markdown data
  these runs mostly use (brackets and paragraph structure are content). Pass
  `--clean-text` to strip `[stage directions]`/`*actions*` + normalize whitespace
  for play-script style sets. `--no-clean-text` is kept as a **deprecated no-op**
  (warns once) so existing commands don't break — drop it from new commands. Both
  trainers; the BNB arm still has the old `--no-clean-text`-on-by-default (flip it
  there too for a matched run).

**Chunked-vocab LM head (`--head-vocab-chunk N`) — relieves the output-card OOM
on big-vocab models.** The gemma-4-31B split run OOM'd on cuda:1 at the **first
training step**, inside the head-weight reconstruction (`get_weight_tensor` →
`preapply_had_r`, a **5.25 GiB** fp32 spike for the 262k-vocab head): the fused CE
reconstructed the whole `[hidden, vocab]` head at once on the output device, on
top of the grad graph. (Eval survived because it ran under `no_grad`.)
- **Fix:** reconstruct + matmul the head in **vocab-column tiles**. The EXL3 kernel
  already slices (`ext.reconstruct_slice`, 128-aligned); the three head transforms
  are all slice-safe at that granularity (`preapply_had_l`/`su` mix only the input
  dim; `preapply_had_r` is **block-diagonal over the output dim at `had_n`=128**;
  `sv` is per-column) — so a 128-aligned column slice reconstructs **bit-identically**
  to the matching slice of the full weight. New `LinearEXL3.get_weight_tensor_slice`,
  exposed via `backbone.head_weight_slice_closure` (falls back to plain column
  indexing for a dense head).
- **`FusedLinearCrossEntropyVocabChunked`** (fused_ce.py) computes the **same loss
  and grad** via an **online softmax** (per-token running max/sum across vocab
  chunks) + a two-pass backward that reuses the saved per-token LSE. Loop is
  **vocab-outer, token-inner**, so each weight chunk is reconstructed exactly once
  per forward and once per backward — **total dequant work = the single-shot path,
  no extra cost**. Peak head memory drops from `~[hidden, vocab]` to
  `~[hidden, chunk]` (+ a `[token_chunk, chunk]` logits tile): at `--head-vocab-chunk
  32768` on the 31B that's the 5.25 GiB spike → ~0.7 GiB. Wired into `compute_loss`
  (frozen-head path only; `train_head`/final-softcap still take the materialized
  path) behind `--head-vocab-chunk` (single + DDP; `NativeLlamaQLoRA(head_vocab_chunk=)`).
  **Default 0 (off)** — opt in after validating.
- **Validation:** the loss + analytic grad were checked against naive CE
  (finite-difference, all vocab/token-chunk combos, ignore-index) — **algorithm
  confirmed**. The torch autograd Function is gradchecked in `tests/test_fused_ce.py`
  (`test_vocab_chunked_*`, incl. *bit-identical to the single-shot fused head*) —
  **run on the box** (no torch in the container). The EXL3 slice reconstruction is
  GPU-only and is now gated automatically: **`qlora_validate_native.py` runs a
  head-slice check** (`get_weight_tensor()[:, a:b]` vs `get_weight_tensor_slice(a, b)`
  at the first/middle/last aligned chunk; expects bit-identical, folds into the
  PASS/FAIL and the non-zero exit, SKIPs for an unsliceable head, `--skip-head-slice-check`
  to opt out). So the standard pre-run gate already covers it; then run
  `python tests/test_fused_ce.py` for the autograd gradcheck and a 1-step train
  smoke with `--head-vocab-chunk 32768` (watch cuda:1 peak drop and the loss match
  an off run). Once trusted it should let you **drop the `--use-per-device` juggling
  and raise `--batch`/`--seq-len`**.

**Re "was it still only saving the lowest held-out val?" — diagnosis (no
regression):** saving the lowest val is **opt-in via `--save-best`**, not the
default; without it a run saves the **endpoint** ("Done."). With `--save-best`
the logic is intact (`qlora_train_native_ddp.py` best-tracking branch), but the
`[best step N, val …]` line only prints when the val *improves*, so once the
held-out curve plateaus/rises (the Gemma4 run flattened ~step 30) it stops
printing those lines even though the earlier best checkpoint is correctly kept.
So a DDP run "not indicating it anymore" = either `--save-best` wasn't passed, or
it was and the eval had already plateaued. `--checkpoint-every` sidesteps the
reliance on a single best/endpoint adapter.

**Status: write-confirmed** (helpers unit-tested for ordering + prune; both
scripts parse). Not yet exercised on the GPU box — smoke it with a small
`--checkpoint-every` + `--keep-checkpoints` on the next run.

**Big-run plan: gemma-4-31B-it on semancy, layer-split across 2×24GB**
(`qlora_train_native.py --parallel split`, single process — NOT ddp, which would
replicate the ~17 GB base per card). The recommended invocation reuses the
established r16/α32 lr1e-4 cosine profile + the Gemma4 specifics
(`--sample-every 0` to avoid the SWA/recurrent cache path, `--no-clean-text`,
`--messages-key messages`, `--eval-split test`, auto prompt/attn) + the wikitext
dual-eval + `--checkpoint-every`. Run the forward-correctness gate **under the
same split** first (`qlora_validate_native.py --parallel split --use-per-device
8 24 --check-backward`) — first 31B Gemma4 on the device-aware split forward, so
prove parity before the run. **Greedy-autosplit footgun:** the 17 GB base fits
one card, so without a per-device cap the whole model lands on cuda:0 and cuda:1
idles — `--use-per-device 8 24` caps cuda:0 near half the base to force the split;
watch the per-card VRAM line and tune (the 262k-vocab head is end-heavy on
cuda:1). **Not yet run** — the gate result + the held-out `test` floor (vs 1B
~3.09 / 12B 2.51) are the first things to record next session.

**MTP note:** gemma-4-31B's checkpoint may carry MTP tensors, but exllamav3's
`gemma4.py` registers only `{"text", "vision"}` components (no `"mtp"`), so MTP is
**not loaded** — our training path (`component="text"`) never touches it, and it's
not wired for inference on this arch either. No special handling needed; nothing
breaks. The real consideration is downstream: fine-tuning the trunk shifts its
hidden states, so a *frozen* MTP draft head's speculative-acceptance rate would
drift — retraining MTP alongside would need (a) a Gemma4 `"mtp"` component in
exllamav3 and (b) a differentiable MTP forward + multi-token loss in our path
(neither exists). MTP speeds up *inference* (self-speculative decoding), not
training.

---

### Session 8 — 8k context on 2×3090: long-context OOM, run-confirmed but BARELY (more to do)

> Goal: train gemma-4-12B-it 3bpw QLoRA at **`--seq-len 8192 --pack`**, r64/α64,
> `--parallel split` across 2×24GB (RTX 3090), batch 1, 262M trainable params.
> End state: **running, but on the ragged edge** of cuda:1 — see open items. Three
> structural fixes landed this session; one more (query-tiled big-head attention)
> is the next lever. This note supersedes the earlier incremental framing in the
> commit history (notably the "ride the mem-efficient backend" idea, which was
> wrong — see below).

**The arch (confirmed from `config.json`).** `head_dim: 256`,
**`global_head_dim: 512`**, `num_attention_heads: 16`, `num_key_value_heads: 8`,
`hidden_size: 3840`, `vocab_size: 262144`. So 40 sliding layers are head_dim 256
(→ `flash`) and **8 global (full-attention) layers are head_dim 512** (→ `sdpa`).
`describe_attn()` → `40×flash, 8×sdpa`.

**The real wall: head_dim 512 has NO O(t) attention kernel on Ampere.** FA2 caps at
256 (`flash_attn_2.py`: `dim > 256 → None`); torch's mem-efficient SDPA backend
**also caps at 256**; exllamav3's `bighead_scalar` is inference-only (needs a KV
cache + `q_len < 8`). So the 8 global layers **always** run the SDPA **math**
backend, which materializes the `[nq, L, L]` score matrix **in fp32** (the math
backend upcasts regardless of input dtype). At L≈8k that is ~4 GB *per global
layer*. Grad-checkpointing keeps one layer live, so the peak is one such matrix —
fine **if the card has room**, fatal when it doesn't. **This is exactly how
HF/Axolotl run gemma-4 too** (they eat the same O(t²) math on the global layers);
they fit a 31B at 8k only because everything *else* is lean. Our OOM was never the
4 GB itself — it was that cuda:1 was already maxed.

**Confirmed culprit allocation:** the OOM was `Tried to allocate 3.97 GiB` in the
backward = `16 × 8161² × 4 bytes` exactly — a near-full **single-document** packed
block (one ~8.2k-token doc) hitting a head_dim-512 global layer. The score matrix
scales with the **longest document in a block**, not the block size.

**Correction to the earlier framing (important for next time):** the
`sdpa`-branch changes that "keep the mem-efficient backend eligible" (per-document
`is_causal`, no mask, **hand-expanded GQA**) were built on a false premise —
**the mem-efficient backend never engages at head_dim 512**, so those layers are
always math. What actually helped: (a) the **per-document split** still matters
(it bounds the math score matrix to the longest *document* instead of the full
8192 block); but (b) the **GQA `repeat_interleave` expansion is now pointless
overhead** for head_dim 512 (math handles GQA via `enable_gqa`) — see open items.

**Three fixes that got it running (in order of impact):**
1. **bf16 activations (the big one).** The matmuls already ran in `compute_dtype`
   (the QLoRA linear casts input + reconstructs the frozen weight in bf16), but
   `_block_forward` then **upcast every activation back to fp32** via `.float()`
   (q/k/v, ctx, attn_out, mlp_out) and kept the **whole residual stream in fp32** —
   ~2× the memory of an HF/Axolotl bf16 forward. *That* was the structural reason a
   12B needed two cards where Axolotl fits a 31B. Fix: drop the `.float()` upcasts
   so activations follow `compute_dtype`; keep fp32 only where it matters — `_norm`
   and `_apply_rope` compute internals in fp32 but **return in the input dtype** (HF
   RMSNorm convention); the eager *reference* path still does scores/softmax in fp32;
   the **final** norm returns fp32 so the head/CE dtype contract is unchanged;
   `layer_scalar` is cast back to the residual dtype (an fp32 scalar would re-promote
   the whole stream). ~Halves activation + grad-checkpoint + attention-backward
   memory. **Bit-identical in fp32** (every new dtype op is a no-op when
   `compute_dtype` is fp32), so the fp32 validate gate and the fp32 CPU block tests
   are unaffected. NOTE: bf16 does **not** shrink the math score matrix (it stays
   fp32) — it frees everything *else* so that spike has room.
2. **`--optim {adamw,adamw8bit,paged_adamw8bit}`** (`build_optimizer`). `torch.AdamW`
   keeps `m`/`v` in fp32 = 8 bytes/param = ~2.1 GB for the 262M-param r=64 adapter,
   allocated lazily on the **first `optimizer.step()`** — which is why an early run
   passed step-0 eval, trained a few steps, then OOM'd at the first *in-training*
   eval (the moments had materialized). bitsandbytes 8-bit moments → ~2 bytes/param
   (~4× less, ~1.6 GB freed at r=64; `paged_` also offloads to host on a spike).
   Negligible quality cost (QLoRA paper uses paged 8-bit Adam). Default stays
   `adamw`. Needs `bitsandbytes` in the venv. **Native trainer only so far.**
3. **Per-document SDPA under packing** (`sdpa` branch): gather non-pad tokens, loop
   `is_causal` SDPA per document span (`pack["cu_seqlens"]`), scatter back — bounds
   the math score matrix to the longest document. `pack_ctx` feeds both `flash` and
   `sdpa`; the old explicit-`[t,t]`-mask path is gone. FlexAttention was tried first
   and does NOT fit a 24GB consumer SM at head_dim 512 (`No valid triton configs …
   Required: 200704, limit: 101376`) — removed.

Also fixed this session: a **corrupted #100/#101 squash-merge** had left
`native_llama.py` with dangling `_flex_*` references (NameError at construction);
master was repaired (`3f28d48`).

**Current run config (run-confirmed, barely):** `--parallel split --use-per-device
9 24 --seq-len 8192 --pack --r 64 --alpha 64 --head-vocab-chunk 32768 --optim
paged_adamw8bit --scheduler cosine`. Split = `{cuda:0: 36 blocks, cuda:1: 12
blocks + final norm + head}`; **cuda:1 is the constraint** (it carries the 262k
head *and* 2 of the 8 global layers, so a long-doc block spikes ~4 GB right where
free memory is tightest). Loss descends cleanly (4.1 → 2.4 by step 5).

**Open items / next session (in priority order):**
1. **Query-tiled big-head attention — THE next lever.** Bound the global-layer
   score matrix to `[16, q_tile, L]` (e.g. q_tile=2048 → ~1 GB vs ~4 GB) with a
   flash-style online-softmax accumulation so **both forward and backward** stay
   bounded (a naive query loop re-bloats the backward unless it's a custom
   `autograd.Function` or nested-checkpointed per tile). This makes head_dim 512 @
   8k fit with real margin regardless of document length, and is what unlocks
   higher rank (128, like the confirmed 2×3090 runs) / longer context.
2. **Drop the pointless GQA expansion** in the `sdpa` branch. Since head_dim 512 is
   always the math backend, `repeat_interleave(KV → nq)` just wastes memory — revert
   to `enable_gqa=True` (math handles GQA, keeps K/V at `nkv` heads). Small win, but
   free, and fold it into the query-tiling rewrite.
3. **Rebalance the split off cuda:1.** It holds the head + 2 global layers. Either
   force the global layers onto cuda:0, or lower `--head-vocab-chunk` to 8192
   (~375 MiB more margin), or shift blocks via `--use-per-device`.
4. **Mirror `--optim` into the DDP + BNB arms** for matched EXL3-vs-NF4 runs.
5. **bf16 parity gate** still worth running once: `qlora_validate_native.py
   --compute-dtype bfloat16` (the `flash`/`sdpa` paths run only in fp16/bf16).
6. **Throughput**: layer-split is serial (cuda:0 100% / cuda:1 0% see-saw); true
   overlap needs pipeline-parallel micro-batching, not built. ~420 tok/s is near the
   floor for this split. Not memory-related, but the next efficiency frontier after
   query-tiling.

---

### Session 9 — bf16 flash/packing parity gate CLOSED; review pass + doc/comment fixes

> Branch `claude/exllama-qlora-review-xginxq`. A review of the whole QLoRA-on-EXL3
> body of work, plus the bf16 packing verification that had been the standing
> open item since Session 6, plus two box-free code/doc accuracy fixes.

**bf16 forward parity is now confirmed for the flash + packing path (was the #1
open verification gap).** Run-confirmed on the box:
- **`qlora_validate_native.py --check-packing --compute-dtype bfloat16`**:
  `attn: 16×flash` with packing **PASS** — 100% per-position argmax agreement,
  last-token `cos ≈ 0.99996`. This is the first time the bf16 *flash* path (not
  just fp32 eager) is differenced against exllamav3's own forward, and the first
  with sample packing engaged.
- **`tests/test_native_llama.py`**: packing **document-isolation** + **pad-NaN**
  checks PASS.
- **bf16 `--pack` training run** (4000 docs → 1413 packed blocks, **82.5% filled**)
  descends cleanly on `16×flash`, grad norms 7–30, `|B|` climbing — confirming the
  **flash-varlen backward** and the **`o[keep] = of` scatter under
  grad-checkpointing** (a real risk area: a custom scatter inside a checkpointed
  block, re-run on recompute).

**Remaining parity sub-gate (precise scope):** the verified run is `16×flash`, i.e.
**all head_dim ≤ 256 — no `sdpa` blocks exercised**. The **big-head `sdpa` path in
bf16** (Gemma4's 8 global layers, head_dim 512, the per-document SDPA loop) is still
only validated *indirectly* via the 12B/8k training descent. To close it the same
way, run `qlora_validate_native.py --compute-dtype bfloat16` on a **Gemma4** base so
`describe_attn` reports `…×sdpa` and the gate covers those blocks. Lower risk than
flash was (plain `is_causal` SDPA, no custom scatter) — but it is the last
uncovered attention branch, and it would **also** serve as the verification for the
pending GQA-expansion removal (next item).

**Code/doc accuracy fixes this session (box-free, no numeric change):**
- **`native_llama._block_forward` (`sdpa` branch) — comment corrected.** The old
  comment still claimed the per-document SDPA "keeps the mem-efficient backend
  eligible" via hand-expanded GQA + no mask. Session 8 established that head_dim 512
  *always* hits the SDPA **math** backend (mem-efficient/flash cap at 256), so that
  rationale was wrong. Comment now states the reality; the `repeat_interleave` GQA
  expansion is annotated as **pure overhead pending removal** (the math backend
  reads `enable_gqa` directly). **The expansion itself was intentionally left in
  place** — removing it changes the attention path and there were no test runs to
  confirm it; do it in the query-tiled big-head rewrite (Session 8 open #1/#2) and
  cover it with the bf16 Gemma4 `sdpa` gate above.
- **Gemma2 `--head-vocab-chunk` no-op warning.** `compute_loss` routes any model
  with final-logit softcapping (Gemma2) to the materialized supervised-position head
  path *before* the vocab-chunk branch, because the chunked-vocab CE can't apply the
  tanh cap — so `--head-vocab-chunk` is silently a no-op there. That is exactly the
  case you'd want it (Gemma2 = 256k vocab + softcap). `NativeLlamaQLoRA.__init__` now
  prints a one-line heads-up so the bypass isn't silent; **no loss-path change**.
  (A real fix would be a softcap-aware chunked CE — not built; flag if Gemma2 with
  long supervised spans becomes a target.)

**Review findings still open (carried forward, unchanged):**
- **bf16 `sdpa` big-head parity gate** on Gemma4 (above) — the one real remaining
  correctness check.
- **Drop the GQA `repeat_interleave`** in the big-head `sdpa` path (free margin on
  the constrained card; do it with the query-tiling rewrite, verify via the gate).
- **Mirror `--optim {adamw,adamw8bit,paged_adamw8bit}`** into the DDP + BNB arms
  (currently native-single-GPU only) so a matched EXL3-vs-NF4 flagship run has equal
  optimizer-state memory. (Session 8 open #4.)
- **Micro-nit:** `EXL3LoRAFunction.backward` reconstructs the frozen weight and
  computes `grad_x` *before* checking `needs_input_grad[0]`; the reconstruction is
  wasted when `grad_x` isn't needed. In practice the first wrapped linear's input
  always requires grad, so cost ≈ 0 today — noted only for completeness.

**Next (box runs, when you're running again):**
1. The bf16 Gemma4 `sdpa` parity gate (closes the last correctness hole).
2. A real packed run with a **held-out eval** to quantify the tok/s win from `--pack`
   and confirm the loss floor is unchanged vs unpacked.
3. The low-bitrate flagship (EXL3 2.5/3bpw where NF4 can't follow, on a real
   metric) — still the highest-value *unrealized* result (§0d).

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
