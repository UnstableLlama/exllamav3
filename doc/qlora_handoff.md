# QLoRA-on-EXL3 — Handoff & Next-Step Plan

> Working session handoff. Branch: **`claude/magical-mayer-fq6z4i`**.
> Goal of the original task: prove whether **QLoRA fine-tuning on EXL3-quantized
> weights** is possible, and get a visible end-to-end demo (fine-tune a small
> model to talk like a pirate).

---

## 1. TL;DR status

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
