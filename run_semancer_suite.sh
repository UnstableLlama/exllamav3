#!/usr/bin/env bash
# Sequential semancer suite: 4 models on UnstableLlama/semancy, smallest first.
# Per model: train (rsLoRA+PiSSA, 5e-5, 2ep, batch 4, save_best; gemmas packed
# bfd@2048, Qwens unpacked — GDN forbids packing) -> prune checkpoint history to
# top-4 held-out + final -> before/after decode (base vs best adapter, greedy,
# fixed seed). Everything recorded under out/semancer_suite/.
# On CUDA OOM the model retries with batch halved / grad_accum doubled (eff-batch
# preserved), twice at most; the first OOM line is logged so a lopsided GPU split
# is distinguishable from a genuine capacity limit. A failed model doesn't stop
# the suite.
set -u
cd /home/unstable/exl3/private/exllamav3

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS=out/semancer_suite
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.log"

note() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

# Fans up for the duration of the suite
nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=66" -a "[fan:1]/GPUTargetFanSpeed=66" \
                -a "[fan:2]/GPUTargetFanSpeed=66" -a "[fan:3]/GPUTargetFanSpeed=66" \
                >/dev/null 2>&1 && note "fans set to 66%"

run_one() {
    local name=$1 config=$2 model=$3 fmt=$4; shift 4
    local split_args=("$@")   # optional: --use-per-device 18 23

    # Train, with up to two OOM fallbacks (batch/2, grad_accum*2 each time).
    local log="" ok=0
    local fallbacks=("" "2 2" "1 4")
    for i in 0 1 2; do
        local cfg=$config
        log="$RESULTS/${name}_train.log"
        if [ "$i" -gt 0 ]; then
            read -r b ga <<< "${fallbacks[$i]}"
            cfg="$RESULTS/${name}_oomfb${i}.yaml"
            sed -E "s/^batch: .*/batch: $b/; s/^grad_accum: .*/grad_accum: $ga/" \
                "$config" > "$cfg"
            log="$RESULTS/${name}_train_oomfb${i}.log"
            note "$name: OOM fallback $i -> batch $b, grad_accum $ga"
        fi
        note "=== $name: TRAIN start (config $cfg) ==="
        if python training/qlora_train.py --config "$cfg" > "$log" 2>&1; then
            ok=1; break
        fi
        if grep -qi "out of memory" "$log"; then
            note "$name: CUDA OOM -- $(grep -im1 'out of memory' "$log" | cut -c1-160)"
        else
            note "=== $name: TRAIN FAILED, non-OOM (see $log) ==="
            return 1
        fi
    done
    if [ "$ok" -ne 1 ]; then
        note "=== $name: TRAIN FAILED after OOM fallbacks ==="
        return 1
    fi
    note "$name: train done -- $(grep -m1 '\[EVAL\] held-out' "$log" | cut -c1-160)"

    # Prune checkpoint history: top-4 by held-out eval + final (best-at-root kept).
    if python training/prune_ckpts.py --out "out/$name" --train-log "$log" --top 4 \
            > "$RESULTS/${name}_prune.log" 2>&1; then
        note "$name: checkpoints pruned to top-4 + final ($RESULTS/${name}_prune.log)"
    else
        note "$name: PRUNE FAILED, checkpoints left untouched ($RESULTS/${name}_prune.log)"
    fi

    # Before/after decode: base model vs best adapter, greedy, fixed seed.
    if ! python training/qlora_infer_native.py \
            --model "$model" --adapter "out/$name" \
            --prompt-format "$fmt" "${split_args[@]}" \
            --prompts "What is truth?" "What is love?" "Are you conscious?" \
            --temperature 0 --seed 42 --max-new-tokens 400 \
            --gen-out "$RESULTS/${name}_gen.jsonl" \
            > "$RESULTS/${name}_before_after.log" 2>&1; then
        note "=== $name: INFERENCE FAILED (see $RESULTS/${name}_before_after.log) ==="
        return 1
    fi
    note "=== $name: COMPLETE (before/after in $RESULTS/${name}_before_after.log) ==="
}

run_one semancer_qwen35_4b  semancer_qwen35_4b.yaml  /mnt/two/weights/Qwen_Qwen3.5-4B/8                   qwen3.5-nothink
run_one semancer_gemma12b   semancer_gemma12b.yaml   /home/unstable/weights/google_gemma-4-12B-it/6.00bpw gemma4-nothink  --use-per-device 8 23
run_one semancer_qwen36_27b semancer_qwen36_27b.yaml /mnt/two/weights/Qwen_Qwen3.6-27B/6                  qwen3.5-nothink --use-per-device 18 23
run_one semancer_gemma31b   semancer_gemma31b.yaml   /home/unstable/weights/googlegemma-4-31B-it/4        gemma4-nothink  --use-per-device 18 23

# Fans back down now that the GPUs are released
nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=30" -a "[fan:1]/GPUTargetFanSpeed=30" \
                -a "[fan:2]/GPUTargetFanSpeed=30" -a "[fan:3]/GPUTargetFanSpeed=30" \
                >/dev/null 2>&1 && note "fans back to 30%"

note "SUITE FINISHED"
