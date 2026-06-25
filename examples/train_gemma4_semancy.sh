#!/usr/bin/env bash
#
# Big-model QLoRA fine-tune of gemma-4-31B-it (EXL3) on UnstableLlama/semancy,
# layer-split across 2x 24GB cards in ONE python process (--parallel split).
#
# Why split (not ddp): a 31B at 4bpw is ~17 GB of base weights -- it FITS one
# 24GB card, but then the optimizer/activations have nowhere to go and you hit
# the memory wall. ddp would replicate all 17 GB per card and make it worse.
# Layer-split puts ~half the base on each card AND splits the training state, so
# the run actually fits with headroom. (See doc/qlora_multigpu_plan.md.)
#
# GREEDY-AUTOSPLIT FOOTGUN (the important bit): exllamav3's autosplit fills
# cuda:0 to its budget, THEN spills to cuda:1 (model_ls.py). Because the 17 GB
# base fits cuda:0, WITHOUT a cap the whole model lands there and cuda:1 sits
# idle. USE_PER_DEVICE caps each card's LOAD so the layers actually split. The
# cap sizes only the load (base + a reference forward); training overhead
# (AdamW 2x moments + grads + activations) is allocated at RUNTIME in the
# leftover headroom -- so cap cuda:0 near HALF THE BASE (~8 GB), NOT half the
# card. The head (262k vocab) is heavy on the LAST device, so cuda:1 runs a bit
# fuller; if it OOMs at runtime, LOWER cuda:0's cap to push more layers off it.
# Watch the printed "decoder block devices:" / per-card VRAM line and tune.
#
# GEMMA4 specifics baked in:
#   --sample-every 0   live generation exercises Gemma's sliding-window /
#                      recurrent cache path (irrelevant to training, and a
#                      needless cache alloc under split) -- off.
#   --no-clean-text    semancy is reasoning data (bracketed content, paragraph
#                      structure); the default cleaner would mangle it.
#   --messages-key     semancy is OpenAI-style single-turn `messages`.
#   --eval-split test  real 116-row held-out split (not a --val-frac carve).
#   prompt-format auto Gemma's own chat template (default).
#   attn-impl auto     per-block flash (head<=256) / SDPA (big-head globals).
#
# MUST pass the forward-correctness gate first: this is the first 31B Gemma4 on
# the device-aware SPLIT forward -- prove its logits match native exllamav3
# before committing to the run. set -e aborts if the gate fails.
#
# Usage:
#   bash examples/train_gemma4_semancy.sh MODEL_DIR [OUT_DIR]
#
# Override with env vars:
#   USE_PER_DEVICE   per-card GB load budget list   (default "8 24"; TUNE!)
#   BATCH            micro-batch                     (default 2)
#   ACCUM            grad-accum steps                (default 8)   eff-batch=16
#   EPOCHS           passes over train               (default 2)
#   SEQLEN           max sequence length             (default 2048)
#   LORA_R / ALPHA   adapter rank / alpha            (default 16 / 32)
#   LR               learning rate                   (default 1e-4)
#   CKPT_EVERY       retained checkpoint interval    (default 10; 0 disables)
#   KEEP_CKPTS       cap retained checkpoints        (default 0 = keep all)
#   DATASET          override the dataset id/path     (default semancy)
#   SKIP_VALIDATE=1  skip the forward-correctness gate (NOT recommended here)
#
# Effective batch (split, single process): BATCH * ACCUM.

set -euo pipefail

MODEL="${1:?usage: train_gemma4_semancy.sh MODEL_DIR [OUT_DIR]}"
OUT="${2:-${MODEL%/}/semancy}"

USE_PER_DEVICE="${USE_PER_DEVICE:-8 24}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-8}"
EPOCHS="${EPOCHS:-2}"
SEQLEN="${SEQLEN:-2048}"
LORA_R="${LORA_R:-16}"
ALPHA="${ALPHA:-32}"
LR="${LR:-1e-4}"
CKPT_EVERY="${CKPT_EVERY:-10}"
KEEP_CKPTS="${KEEP_CKPTS:-0}"
DATASET="${DATASET:-UnstableLlama/semancy}"

# Run from repo root so example paths resolve.
cd "$(dirname "$0")/.."

mkdir -p "$OUT"
LOG="${OUT%/}.log"
EFF=$((BATCH * ACCUM))

echo "=============================================================================="
echo " gemma-4-31B-it QLoRA (semancy)  |  PARALLEL=split  |  eff-batch $EFF"
echo " model : $MODEL"
echo " data  : $DATASET   (eval-split: test)"
echo " out   : $OUT"
echo " split : USE_PER_DEVICE='$USE_PER_DEVICE'  (cap cuda:0 ~half the base)"
echo " lora  : r=$LORA_R alpha=$ALPHA lr=$LR cosine wd0.01  epochs=$EPOCHS seq=$SEQLEN"
echo " ckpt  : every $CKPT_EVERY steps -> $OUT/checkpoint-<step> (keep=$KEEP_CKPTS)"
echo " log   : $LOG"
echo "=============================================================================="

SPLIT_ARGS=(--parallel split)
[[ -n "$USE_PER_DEVICE" ]] && SPLIT_ARGS+=(--use-per-device $USE_PER_DEVICE)

# 1. Correctness gate (under the SAME split): the differentiable cross-device
#    forward must match native exllamav3 on this 31B Gemma4 before we train.
if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    echo "== [1/2] validating differentiable forward under split (gates the run) =="
    python examples/qlora_validate_native.py --model "$MODEL" \
        --check-backward "${SPLIT_ARGS[@]}"
else
    echo "== [1/2] SKIP_VALIDATE=1 -- skipping the forward-correctness gate =="
fi

# 2. The run. --save-best keeps the best held-out (test) checkpoint at $OUT;
#    --checkpoint-every keeps a retained history at $OUT/checkpoint-<step> so a
#    long split run is never reliant on a single best/endpoint adapter.
#    NOTE: single-process script uses --r (not --lora-r; the torchrun abbrev
#    collision only bites the ddp launcher).
echo "== [2/2] launching split QLoRA run =="
CKPT_ARGS=()
[[ "$CKPT_EVERY" != "0" ]] && CKPT_ARGS=(--checkpoint-every "$CKPT_EVERY" --keep-checkpoints "$KEEP_CKPTS")

python examples/qlora_train_native.py \
    --model "$MODEL" --out "$OUT" "${SPLIT_ARGS[@]}" \
    --dataset "$DATASET" --messages-key messages --no-clean-text \
    --eval-split test --eval-every 10 --save-best \
    --eval2-dataset wikitext --eval2-config wikitext-2-raw-v1 --eval2-text-key text \
    --r "$LORA_R" --alpha "$ALPHA" --lr "$LR" \
    --scheduler cosine --warmup-ratio 0.1 --weight-decay 0.01 \
    --batch "$BATCH" --grad-accum "$ACCUM" --epochs "$EPOCHS" --seq-len "$SEQLEN" \
    --sample-every 0 "${CKPT_ARGS[@]}" \
    2>&1 | tee "$LOG"

echo "== done. best adapter -> $OUT ; checkpoints -> $OUT/checkpoint-* ; log -> $LOG =="
echo "   compare the held-out 'test' floor vs prior runs (1B ~3.09, 12B 2.51)."
echo "   verify on the BF16 base for a fair read (the 4bpw base attenuates LoRAs):"
echo "   python examples/qlora_infer_native.py --model <gemma4-bf16> --adapter $OUT"
