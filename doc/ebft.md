# EBFT on EXL3 — design, status, and run guide

Energy-Based Fine-Tuning (EBFT) for the exl3-qlora native path. First known
implementation of EBFT on **LoRA / quantized** weights (the paper and its
reference code are full-fine-tuning only).

- **Paper:** "Matching Features, Not Tokens: Energy-Based Fine-Tuning of
  Language Models" (Jelassi et al., arXiv:2603.12248).
- **Reference code:** `sjelassi/ebft_openrlhf` (OpenRLHF fork, full-FT, Ray).
  Cloned and read to pin exact semantics; our reward math is faithful to the
  **code**, not the appendix, where they differ (see "Two deliberate
  deviations" below).
- **Built:** Session 39 (2026-07-17). Status: implementation complete +
  correctness-gated + one 30-step smoke run. **Not yet answered:** does
  EBFT-LoRA beat SFT-LoRA (needs the A/B — see "Open work").

---

## What EBFT is, in one paragraph

Instead of next-token CE (SFT) or a scalar verifier reward (RLVR), train the
policy so its on-policy rollouts match the *feature statistics* of ground-truth
completions. A frozen feature network φ (a copy of the model at init) embeds
both sampled and ground-truth completions; each rollout gets a dense reward =
whitened alignment-with-ground-truth minus leave-one-out similarity-to-siblings;
the policy is updated by REINFORCE with an RLOO baseline, plus a small CE term.
Paper results: beats SFT on downstream accuracy, matches RLVR with no verifier,
and *lowers* validation CE more than SFT does.

---

## Files (all uncommitted as of Session 39)

| File | Status | What |
|---|---|---|
| `exllamav3/training/ebft.py` | **new** (194 L) | reward math (`ebft_rewards`), whitening (`whiten_features`), exact sampler (`sample_rollouts`) |
| `training/qlora_train_ebft.py` | **new** (983 L) | the trainer (anchors-as-rows, self-test, eval, run-log) |
| `tests/test_ebft.py` | **new** (163 L) | CPU test suite, no GPU needed |
| `exllamav3/training/native_llama.py` | **modified** | `collect_hidden()` + `feature_block_indices()` feature taps in the forward |
| `training/README.md` | **modified** | file listing entry |
| `doc/qlora_handoff.md` | **modified** | Session 39 narrative note |

---

## Architecture / key decisions

**Frozen feature network φ = the adapter-disabled base.** EBFT's φ is a copy of
the model at init. With LoRA the policy at init *is* the frozen base, so
`net.adapters_disabled()` (the existing DPO/KTO reference trick) gives φ exactly
— zero extra VRAM, no second model. Holds for `--init-lora default/pissa/eva`;
**not** for `qerr` (its step-0 policy ≠ raw base), which the trainer warns about.

**Feature taps.** `collect_hidden(block_indices, positions)` context manager on
`NativeLlamaQLoRA` stashes the residual stream after selected blocks, optionally
gathered at given token positions (always pass positions in training — the full
fp32 stream is hundreds of MB/tap). `feature_block_indices()` picks ~25/50/75%
depth via the HF `hidden_states` convention the reference critic uses. Features
= concat across the 3 taps, last window token, jointly L2-normalized
(reference's `hidden_state_method=concat` + `embed_method=last_token`).

**Exact on-policy sampler (v1 choice).** `sample_rollouts` samples G tokens by
running the differentiable forward G times (no KV cache). Wasteful, but the
rollout is sampled under *exactly* the policy the REINFORCE gradient is later
computed against (`compute_logps` over the same path) — **zero sampling/scoring
mismatch**, unlike a vLLM-vs-trainer RLHF stack. Faster alternatives are the
first throughput lever (see Open work).

**Anchors-as-rows (v1 choice).** The reference amortizes rollouts across a whole
sequence with a Quiet-STaR custom attention mask (Appendix F). We instead sample
`--anchors` positions per sequence and materialize each `(context, window)` as
its own batch row. Same reward math per group, less prefix sharing. The strided
mask is the planned v2 throughput upgrade, not an algorithmic requirement.

**Loss.** `loss = rl_coef · mean_rows(−advantage · logp_mean) + ce_coef · CE`,
where `logp_mean` is the per-token-mean completion logprob (reference
`EBFTPolicyLoss` semantics; the PPO ratio is 1 for a single-step surrogate), the
advantage is the detached `reward − RLOO_baseline`, and CE is standard next-token
CE on the ground-truth sequences (the paper's γ term).

### Two deliberate deviations from the paper's appendix (both match the reference code, verified numerically)

1. **Ground-truth whitening.** The code whitens the gt feature by applying the
   `[n,n]` row operator to n replicated copies, so each output row is a *scalar
   multiple* of the raw gt vector — **not** the paper's D-space `(Σ⁺)^(1/2) y`.
   Under cosine alignment this reduces to comparing whitened rollouts against the
   raw gt direction. This is what the reference does; we match it.
2. **Diversity scale.** The code's whitened diversity is `1/n` of the appendix
   eq. 48 values (a duplicate pair at n=4 → **1/3**, not 4/3; all-identical n=4
   → 1/2). `tests/test_ebft.py` pins these exact numbers.

**Structural insight worth remembering:** after whitening, the Gram matrix of
*distinct* rollouts is ≈ identity, so the diversity term is essentially a
**duplicate penalty** — it only fires on repeated/near-identical rollouts (n_k>1
groups). That's why temp-0.6/G=8 duplicate rollouts (common at short horizons)
are handled by design rather than needing a special case.

---

## Verified (Session 39)

- **CPU suite** (`python tests/test_ebft.py`) — whitening orthonormalization, gt
  scalar-multiple property, duplicate-gating exact values, RLOO closed forms,
  no-whiten composition, zero-feature finiteness, sampler top-k/top-p. All pass.
- **Model self-test** (`--self-test` on Llama-3.2-1B-Instruct EXL3 @4bpw):
  - tap position-gather == full-stream slice;
  - `compute_logps` == gather(log_softmax(`net.logits`));
  - one full EBFT step is finite with grads on **all 112 adapter tensors and
    nothing else** (φ / base isolation confirmed).
- **30-step smoke train** (Llama-3.2-1B, semancy, r32, lr 1e-5, G8/n4/anchors4,
  temp 0.6, γ 0.03). Held-out eval moved the right way:

  | step | CE ↓ | CFM ↓ | reward ↑ |
  |---|---|---|---|
  | 0 | 3.4294 | 0.8623 | +0.4537 |
  | 10 | 3.3838 | 0.8134 | +0.4980 |
  | 20 | 3.3424 | 0.8165 | +0.4954 |
  | 30 | 3.2965 | 0.8145 | +0.4965 |

  Stable throughout: grad norm 0.9–1.9 (no spikes), |dB| 0→0.22 smooth, RL term
  oscillating ~0 (correct for a zero-mean RLOO advantage), dup 0–9%/step with
  whitened diversity penalizing exactly those steps. **~32 s/step**, 5.08 GB peak,
  76% of step time in forward (the sampler's G re-forwards, as predicted).

  ⚠️ This proves *correct and stable*, not *works*. 30 steps / 231 examples /
  one seed / γ=0.03 means CE could be moving on the CE term alone. Attribution
  needs the A/B.

---

## How to run

```bash
# 1. correctness gate on your model (fast, ~2 min on a 1B)
python training/qlora_train_ebft.py \
    --model /path/to/exl3_model --dataset <id-or-path> \
    --messages-key messages --self-test

# 2. a real run (validation-scale, Qwen2.5-1.5B / Llama-3.2-1B class)
python training/qlora_train_ebft.py \
    --model /path/to/exl3_model --out out/exl3_ebft \
    --dataset <id-or-path> --mode qa \
    --gen-len 8 --n-samples 4 --anchors 4 --temperature 0.6 \
    --lr 1e-5 --epochs 1 --eval-every 50 --val-frac 0.02

# 3. verify the adapter on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3_model --adapter out/exl3_ebft
```

Reference-code defaults are baked in: G=8, n=4, temp=0.6, align=1.0, div=0.5,
ce(γ)=0.03, betas=(0.9,0.95). `--mode pretrain` trains on raw packed text
(non-verifiable setting). Frozen LM head only; single-process (`--parallel
single|split`); no packing / `--train-head` / `--train-embeddings`.

**Key knobs:** `--lr` (paper full-FT is 1e-6; LoRA range 1e-5..1e-4 is
**unvalidated**), `--anchors`/`--n-samples`/`--batch` (rollout rows =
batch·anchors·n_samples, the real memory/compute knob), `--no-whiten` (ablation
— paper shows clear degradation without it), `--div-coef` (1.0 = pure
proper-scoring-rule; 0.5 = the paper's α=0.5 alignment bias).

---

## Open work (all needs user-approved GPU time — see standing orders in handoff §Session 38)

1. **The real question: EBFT-LoRA vs SFT-LoRA A/B** on the same data
   (Qwen2.5-1.5B or Llama-1B, OpenCodeInstruct-class). Watch **CFM ↓ AND CE ↓
   together** — the paper's signature. Consider a γ=0 arm to isolate pure
   feature-matching (if CE still drops there, cleanest possible confirmation).
2. **LoRA LR sweep** 1e-5..1e-4 (the one genuinely unknown hyperparameter).
3. **Throughput:** native-generator sampling via `apply_to_native` (already
   built, needs wiring) — **caveat:** routed-expert LoRA goes off-policy there
   (fused MoE kernels bypass runtime LoRA); the exact sampler has no such
   caveat, so MoE must use the exact sampler. Then Appendix-F strided-mask
   rollouts.
4. **Scale up** only if the A/B validates.

Cost note: A/B over ~5–10k examples is overnight-class at 1–1.5B on one 3090
(~32 s/step, 8 seq/step, sampler-forward-bound).

---

## Provenance / gotchas for a future session

- Reference repo lives in this session's scratchpad only (not in-repo); re-clone
  `sjelassi/ebft_openrlhf` if you need to re-check semantics. Key files read:
  `openrlhf/utils/embedding_utils.py` (whitening, alignment/diversity),
  `.../ebft_experience_maker.py` (`compute_baseline` = RLOO), `models/loss.py`
  (`EBFTPolicyLoss`), `models/critic.py` (feature extraction),
  `scripts/run_ebft_example.sh` (default hyperparameters).
- Naming collision: arXiv:2402.12419 is an *unrelated* 2024 "EBFT" (sparse-LLM
  block fine-tuning) that *does* mention LoRA — not prior art for this.
- Live-sample logging (`--sample-every` in the SFT trainer) is **not** wired
  into the EBFT trainer; add it via `apply_to_native` if wanted.
