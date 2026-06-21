#!/usr/bin/env bash
#
# Overnight 2-GPU QLoRA fine-tune of TheDrummer/Rocinante-XL-16B-v1 (a 54-layer
# Mistral-Nemo-12B depth merge) on a generated Yoda style set, via torchrun DDP.
#
# Usage:
#   bash examples/train_rocinante_yoda.sh MODEL_DIR DATA_JSONL [OUT_DIR]
#
#   MODEL_DIR   path to the local EXL3 (quantized) Rocinante directory
#   DATA_JSONL  path to the Yoda .jsonl (Alpaca schema: instruction/input/output)
#   OUT_DIR     adapter output dir (default: MODEL_DIR/yoda)
#
# Override the run shape with env vars, e.g. `BATCH=2 ACCUM=8 NGPU=2 bash ...`:
#   NGPU   GPUs / torchrun processes        (default 2)
#   BATCH  per-GPU micro-batch              (default 4; drop to 2 if you OOM)
#   ACCUM  grad-accum steps                 (default 4)
#   STEPS  training steps                   (default 2000)
#   SKIP_VALIDATE=1  skip the forward-correctness gate
#
# Effective batch = BATCH * NGPU * ACCUM (default 4*2*4 = 32).
#
# Why these defaults (from doc/qlora_handoff.md, Session 3):
#   * lr 1e-4 + --save-best: lr 2e-4 overfit (train 0.09 / held-out 3.11); 1e-4
#     and keeping the best-val checkpoint gave the clean ~2.0 minimum. --save-best
#     also means an overfit endpoint never clobbers the good adapter.
#   * r64/alpha64 (ratio 1.0): intuitive strength knob.
#   * a validate gate first: this is the first time the transformers-free forward
#     runs on a Mistral model, so prove it matches native before training all night.

set -euo pipefail

MODEL="${1:?usage: train_rocinante_yoda.sh MODEL_DIR DATA_JSONL [OUT_DIR]}"
DATA="${2:?need the Yoda .jsonl dataset path (Alpaca schema)}"
OUT="${3:-${MODEL%/}/yoda}"

NGPU="${NGPU:-2}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
STEPS="${STEPS:-2000}"

# Run from the repo root so the example paths below resolve.
cd "$(dirname "$0")/.."

mkdir -p "$OUT"
LOG="${OUT%/}.log"

echo "=============================================================================="
echo " Rocinante-XL-16B QLoRA (Yoda)  |  $NGPU GPU(s)  |  eff-batch $((BATCH*NGPU*ACCUM))"
echo " model : $MODEL"
echo " data  : $DATA"
echo " out   : $OUT"
echo " log   : $LOG"
echo "=============================================================================="

# 1. Correctness gate: differentiable forward must match native exllamav3 on this
#    (new, Mistral) architecture. set -e aborts the whole run if it exits non-zero.
if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    echo "== [1/2] validating differentiable forward on Rocinante (gates the run) =="
    python examples/qlora_validate_native.py --model "$MODEL"
else
    echo "== [1/2] SKIP_VALIDATE=1 -- skipping the forward-correctness gate =="
fi

# 2. The overnight DDP run. torchrun spawns one process per GPU; only rank 0
#    prints/saves. --save-best keeps the best held-out checkpoint at $OUT.
echo "== [2/2] launching $NGPU-GPU DDP QLoRA run =="
torchrun --standalone --nproc_per_node="$NGPU" examples/qlora_train_native_ddp.py \
    --model "$MODEL" --out "$OUT" \
    --dataset "$DATA" \
    --lora-r 64 --alpha 64 --lr 1e-4 \
    --batch "$BATCH" --grad-accum "$ACCUM" --seq-len 512 --steps "$STEPS" \
    --val-frac 0.02 --eval-every 100 --save-best \
    2>&1 | tee "$LOG"

echo "== done. adapter -> $OUT ; full log -> $LOG =="
echo "   verify (on bf16 base for a fair read; the 4bpw base attenuates LoRAs):"
echo "   python examples/qlora_infer_native.py --model $MODEL --adapter $OUT"
