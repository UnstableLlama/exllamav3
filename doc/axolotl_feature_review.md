# Axolotl feature review — what they've shipped that we haven't (2026-07-20)

> Companion to `doc/qlora_optimization_audit.md` (Session 11, 2026-07-01). That
> audit was a *performance/VRAM* pass — packing fill, CCE, LoRA+, selective
> checkpointing, fused kernels, sequence parallelism — and most of its shortlist
> is now built (BFD packing, fused-CE dtype fix, GA normalization, fast dequant,
> batched eval). This doc is the other axis: the **training methods and
> objectives** axolotl has added since, mapped against what we already have, and
> ranked by value-per-effort **for this pipeline specifically** (quantized
> immutable base, merge-and-requantize deploy, RP-model releases, 1–2×3090).
>
> The ranking reweights a generic "axolotl has X, we don't" list for two things
> a feature-parity checklist can't see: (a) which gaps we can build cheaply
> because we already have the plumbing (EBFT rollouts, the DPO/KTO reference
> trick, original-model loading for qerr), and (b) which axolotl features are
> actively the *wrong* tool for an EXL3 base and should stay unbuilt.

## What axolotl has now (recent releases, categorized)

Pulled from the axolotl README + docs, 2026-07. Grouped; **bold = we don't have
it**, plain = we already have an equivalent.

- **Fine-tuning:** full FT, LoRA, QLoRA, ~~ReLoRA~~, ~~DoRA~~. We have LoRA/QLoRA
  on the trellis, embed/head full + low-rank, SVD inits (pissa/eva/qerr) that
  axolotl mostly *doesn't* have. DoRA we evaluated and deferred (Session 13
  decision record: changes every step, dequant-bound backward, edge is at low
  rank we don't run, win is void when shipping merged).
- **Preference / RL:** DPO, IPO, KTO (we have these), **ORPO**, **GDPO**,
  **GRPO + Async GRPO**, **Reward Modelling (RM)**, **Process Reward Modelling
  (PRM)**. Reference-free preference (ORPO / SimPO-class) and verifiable-reward
  RL (GRPO) are our two real methods gaps. We have EBFT (novel; not in axolotl)
  as our on-policy objective, but **no scalar-reward RL**.
- **Distillation:** **KD via teacher top-k logprobs** (offline, precomputed with
  vLLM; KL loss; composes with QLoRA). We have nothing here — and it's more
  interesting for us than for axolotl (below).
- **Optimizers:** **Muon** (distributed, FSDP2), **Dion**, **Schedule-Free
  AdamW**, **CAME**, **ADOPT**, Optimi/Flash AdamW. We have adamw / adamw8bit /
  paged_adamw8bit only.
- **Regularizers:** **NEFTune** (noisy embedding FT). We don't have it.
- **Quantization:** QAT for int8/int4/**FP8/NVFP4/MXFP4**, **BitNet 1.58**. We
  have our own `--quant-aware {noise,ste}` (built S17, box-shelved S25 —
  early-stopping beat it) tuned to the *trellis* deploy path; axolotl's
  format-specific QAT doesn't map to EXL3.
- **Parallelism:** FSDP1/2, DeepSpeed, DDP (we have DDP), **Sequence /
  Context / Tensor / Expert Parallelism**, Ray multi-node, TiledMLP. Sequence
  parallelism is already logged in the optimization audit as big-engineering /
  FA2-head_dim-limited.
- **Kernels/attention:** Liger (we wire RMSNorm+SwiGLU), CCE (audit: can't read
  a trellis; dense-head only), Flash 2/3/4, **FlexAttention**, **SageAttention**,
  **ScatterMoE**. We have our own fast trellis dequant path + MoE forward.
- **Newer/niche:** text-diffusion training, EAFT (entropy-aware focal training),
  Scalable Softmax, FP8 mixed-precision. Low relevance here.

## Tier 1 — worth building, strong fit, reuses infrastructure we already have

### 1. GRPO / RLVR (verifiable-reward RL) — the biggest methods gap, and we're ~90% plumbed for it

GRPO (group-relative policy optimization, the reasoning-RL workhorse of 2025–26)
is the single most significant method we lack. The reason it's Tier 1 rather
than a big project: **the EBFT build already wrote every hard part of it.**

`exllamav3/training/ebft.py` + `qlora_train_ebft.py` already give us:
- on-policy rollouts on our own model — both the *exact* differentiable sampler
  and the *native* KV-cached sampler via `apply_to_native` (S41, 2.06× step);
- the **RLOO / leave-one-out baseline** (closed-form, CPU-tested);
- REINFORCE with `compute_logps` over the differentiable path;
- the **frozen reference = adapter-disabled base** trick (shared with DPO/KTO),
  i.e. a KL-to-reference term costs zero extra VRAM.

GRPO is then a *reward swap*: replace EBFT's dense whitened feature-matching
reward with a **scalar verifier reward** (exact-match / format / unit-test /
regex, or a reward model), normalize advantages **within the group** (GRPO's
group-mean/std instead of RLOO's leave-one-out — we already compute per-group
stats), and add the standard **KL-to-reference penalty**. The rollout window
becomes a full completion rather than a G=8 span, but that's a config change to
machinery that already samples completions. `doc/ebft.md` itself frames EBFT as
"matches RLVR with no verifier" — so RLVR/GRPO is the acknowledged comparison
point we never built.

- **Why it fits us:** verifiable RL on a quantized base with merge-and-requant
  deploy is a clean story; it's the method people now expect a finetuning stack
  to have; and it slots into the same YAML launcher / run-report path as EBFT.
- **Effort:** medium. New reward-fn interface + group-normalized advantage +
  KL term + a GRPO trainer that is mostly `qlora_train_ebft.py` with the reward
  and advantage stages replaced. The MoE caveat is inherited verbatim (native
  rollouts refused for routed-expert LoRA; exact sampler only).
- **Watch-outs:** REINFORCE LR sensitivity is real here too (see the EBFT LR
  findings — RL wanted 10–20× lower LR than SFT); reward hacking; verifier
  design is the actual work once the trainer exists.

### 2. Knowledge distillation from the bf16 teacher into the quantized student — more valuable for *us* than for axolotl

Axolotl's KD trains a student on a teacher's **top-k logprobs** (precomputed
offline, KL loss, composes with QLoRA). For axolotl the teacher and student are
usually different-size *bf16* models. **Our setting makes this unusually
pointed:** the natural teacher is the *original bf16 model* and the student is
*that same model quantized to EXL3 + a LoRA*. Distilling the teacher's output
distribution into the quantized student is a principled, direct attack on the
recurring theme of this whole repo — "LoRAs come out attenuated on quantized
bases" (S3), qerr's rank-r quant-error repair (S13), the quant-aware experiments
(S17/S25). It's the recover-the-quantization-loss idea, done at the output
distribution instead of the weights.

- **We already load the original model** for `init_lora: qerr` and
  `quant_aware_ref_model` (error measurement), so the teacher plumbing exists.
- **Two build shapes.** (a) *Offline*, axolotl-style: one pass caches the bf16
  teacher's top-k logprobs per token to disk, then train the quantized student
  to KL-match them — cheap per step, teacher runs once, but needs storage and a
  fixed dataset. (b) *Online self-distillation*: teacher = adapter-disabled base
  in **full bf16-reference** mode within the same process — but our base *is* the
  trellis, so the honest teacher is the separate bf16 model held resident, which
  is memory we usually spend on the quant instead. Offline is the better first
  cut for us.
- **Effort:** medium. The KL loss wants to live in the same chunked/fused
  frozen-head CE path we already stream (`fused_ce.py`, `ce_chunk`,
  `head_vocab_chunk`) so the `[tokens, vocab]` tensor never materializes on
  262k-vocab heads; top-k logprobs keep that bounded. A validate-gate for the
  KL head (like every other init/loss gate) before a long run.
- **Why it's interesting now:** it's the axolotl feature whose payoff is
  *larger* in our niche than in theirs, and it reuses loader + fused-head code
  we already own.

### 3. Reference-free preference: ORPO / SimPO — a small delta on `qlora_train_pref.py`

We have DPO/KTO (`qlora_train_pref.py`, both box-smoked S29/S30). Axolotl also
has **ORPO** (odds-ratio preference, monolithic — a penalty term *added to the
SFT loss*, **no reference model at all**) and the SimPO/CPO family
(reference-free, length-normalized reward). These matter for us because:

- **ORPO is cheaper than DPO here**, not more expensive: no reference forward at
  all (not even the adapter-disable pass), so it's the lightest preference
  objective we could offer. It's an added term on the CE we already compute.
- Reference-free preference is popular exactly for the **RP/style tunes** this
  repo targets, where a single-stage "SFT + preference nudge" is ergonomic.
- **Effort:** small. Same data schema and trainer as DPO/KTO; it's another
  `--method`/loss-variant on `qlora_train_pref.py` (which the backlog already
  wants wired into the YAML launcher, item 8). ORPO first (no reference forward),
  SimPO/CPO as loss variants after.

## Tier 2 — cheap wins, low risk

### 4. NEFTune (noisy embedding fine-tuning) — ~10 lines, well-cited SFT boost

Add scaled uniform noise to the embedding output during training only
(`x += U(-1,1) · α/√(L·d)`), off at eval/inference. Consistently helps
instruction-tuning quality; near-zero cost; nothing in our path fights it. Fits
cleanly next to the existing embedding-adapter code (`lora_embed`) and the
train-only masking we already do (dropout, quant-aware are train-only too). One
config key `neftune_alpha`, one gate line in the validate script (noise off →
step-0 exact). This is the highest quality-per-line item in the whole review.

### 5. New optimizers: Schedule-Free AdamW and Muon — drop-in `--optim` values

Our `--optim` is adamw / adamw8bit / paged_adamw8bit. Axolotl added a whole
suite; two are worth exposing:

- **Schedule-Free AdamW** removes the LR-schedule search entirely (no warmup/
  cosine to tune) — directly useful given how much of our A/B effort is LR
  sensitivity (EBFT especially). Pure optimizer swap.
- **Muon** (orthogonalized-momentum, higher sample efficiency) applies to 2D
  matrices — our LoRA A/B factors qualify. It's a *small experiment* rather than
  a sure win (Muon's gains are strongest on large full-FT weight matrices, not
  tiny LoRA factors), but it's a drop-in optimizer class, adjudicated on the run
  log like LoRA+.
- **Effort:** small each. Add the optimizer construction + a `--optim` value +
  a run-log field. No new math in our forward/backward. Gate: none needed beyond
  "does it train stably" — these don't change the graph.

## Tier 3 — noted, but lower priority or a poor fit for an EXL3 base

- **LoRA+ (differential A/B LR)** — already in the optimization audit (A10) as a
  cheap two-line change; adjudicate on the run log. Belongs on the list; just not
  new information from axolotl.
- **Sequence / Context parallelism** — already audited: FA2-based → head_dim ≤
  256, big engineering, our long-context multi-GPU is serial layer-split. Defer.
- **FSDP2 / DeepSpeed / Tensor / Expert parallelism** — the handoff argues this
  is the *wrong tool* for us (S-notes §FSDP): EXL3 compression dissolves FSDP's
  main use case (a 70B @2.5bpw ≈ 22 GB fits one 24 GB card), and the packed
  trellis tensors don't shard like bf16 params. DDP remains the right axis.
- **Format-specific QAT (FP8 / NVFP4 / MXFP4 / BitNet)** — we already have a
  trellis-specific quant-aware LoRA, and it was **box-shelved** (S25:
  early-stopping beat it). axolotl's format QAT doesn't map onto the EXL3
  deploy path. Skip unless the 2.5bpw + `ste` lane is ever revisited.
- **CCE / FlexAttention / SageAttention / ScatterMoE** — audit already covers
  CCE (can't read a trellis; dense-head only). The rest are backend swaps around
  attention/MoE where we run our own fast trellis path; low marginal value.
- **Reward / Process Reward Modelling** — natural *follow-on* to GRPO (item 1),
  not a standalone build. Sequence after the RL trainer exists.
- **torch.compile, text-diffusion, EAFT, Scalable Softmax** — niche or blocked
  by our custom autograd Functions (audit note). Not now.

## Bottom line — recommended order

1. **GRPO / RLVR** — biggest methods gap, and EBFT already built the rollout /
   RLOO / reference machinery, so it's a reward-and-advantage swap, not a green
   field. *(medium)*
2. **KD from the bf16 teacher into the quantized student** — the axolotl feature
   whose payoff is bigger in our niche than in theirs; reuses the original-model
   loader and the fused frozen-head CE we already own. *(medium)*
3. **ORPO / SimPO** — reference-free preference; a small loss-variant delta on
   the existing DPO/KTO trainer (ORPO needs *no* reference forward). *(small)*
4. **NEFTune** — ~10 lines, well-cited, near-zero risk. *(tiny)*
5. **Schedule-Free / Muon optimizers** — drop-in `--optim` values; adjudicate on
   the run log. *(small)*

Items 3–5 are box-free-verifiable deltas (CPU gate + step-0 checks, like every
other init/loss we've added). Items 1–2 need the same treatment as EBFT: a
correctness gate, then a matched A/B against an SFT/DPO baseline on the same
data before believing any result — RL and KD are both easy to fool yourself
with.

## Sources

- Axolotl README / docs (features, RL, optimizers): https://github.com/axolotl-ai-cloud/axolotl , https://docs.axolotl.ai/
- Axolotl KD trainer (top-k logprobs): https://github.com/axolotl-ai-cloud/axolotl/pull/2202
- Axolotl v0.7.0 (GRPO): https://axolotlai.substack.com/p/axolotl-update-v070-with-grpo
- Axolotl optimizers (Muon / Dion / Schedule-Free / CAME): https://docs.axolotl.ai/docs/optimizers.html
- GRPO: DeepSeekMath, https://arxiv.org/abs/2402.03300
- ORPO: https://arxiv.org/abs/2403.07691 · SimPO: https://arxiv.org/abs/2405.14734
- NEFTune: https://arxiv.org/abs/2310.05914
- Muon: https://kellerjordan.github.io/posts/muon/
- Prior internal context: `doc/qlora_optimization_audit.md`, `doc/ebft.md`,
  `doc/qlora_handoff.md` (Session 13 PEFT decision record; FSDP notes).
