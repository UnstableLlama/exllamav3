# EXL3 QLoRA training scripts

User-facing entry points for the QLoRA-on-EXL3 training path. The importable
library code behind them lives in the `exllamav3.training` package
(`exllamav3/training/`); this directory holds the runnable scripts, in the same
way `examples/` holds upstream's inference examples. `examples/` itself is kept
byte-identical to upstream — everything fork-specific lives here.

Quick start (full version in the repo [README](../README.md)):

```bash
# 1. prove the differentiable forward is correct for YOUR model (run this first)
python training/qlora_validate_native.py --model /path/to/exl3-model --compute-dtype bfloat16

# 2. edit the config, then train
python training/qlora_train.py --config training/qlora_train_config.yaml

# 3. before/after comparison on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3-model --adapter out/my_adapter
```

## Files

- `qlora_train.py` + `qlora_train_config.yaml` — the YAML launcher (single
  command entry point; picks the single-GPU, layer-split, or DDP backend) and
  its fully-commented reference config.
- `qlora_train_native.py` — the single-GPU / layer-split SFT trainer (plain
  PyTorch, no transformers). Also home to the shared data/tokenization helpers
  the other trainers import.
- `qlora_train_native_ddp.py` — the multi-GPU DDP variant (run under
  `torchrun`).
- `qlora_train_pref.py` — DPO / KTO preference training on the native path.
- `qlora_validate_native.py` — the correctness gates: compares the
  differentiable training forward against exllamav3's own inference forward.
  Run this FIRST on any new model/architecture.
- `qlora_infer_native.py` — before/after generation with an adapter on the
  native inference path.
- `qlora_train_bnb.py` — the bitsandbytes-NF4 comparison arm (matched
  benchmark harness; needs its own transformers+peft+bitsandbytes venv).
- `experiments/` — one-off, experiment-specific tooling (dataset generation,
  style metrics, run scripts); kept for reproducibility, not part of the
  reusable path. See its README.

## MoE models (Qwen3-MoE, Qwen3.5-MoE, Gemma4 MoE)

Supported with the std softmax top-k router (incl. Qwen3.5-MoE's shared
expert + sigmoid shared gate, and the Gemma4 MoE layout: routing + routed
experts fed from the raw post-attention residual through their own pre-norms,
routed/shared post-norms, per-expert scale). Plain
`gate_proj`/`up_proj`/`down_proj` targets adapt the dense / shared-expert
paths only; opt in to the routed
experts with `--targets ... expert_gate_proj expert_up_proj expert_down_proj`
(consider a small `--expert-r` — it's one adapter pair per expert per layer).
The router is always frozen and no aux load-balancing loss is added. Caveat:
routed-expert adapters do not show up in native generation (the fused MoE
inference kernels bypass the runtime LoRA slots) — deploy them by
merge-and-requantize. See the Session 20/21 notes in `doc/qlora_handoff.md`.

## Docs

- `doc/qlora_handoff.md` — the full engineering log (per-session results,
  decision records, backlog).
- `doc/qlora_feasibility.md`, `doc/qlora_multigpu_plan.md`,
  `doc/qlora_optimization_audit.md` — design rationale and plans.
