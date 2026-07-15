"""
Session 35 — attribute GPU time in a --torch-profile chrome trace to the
Session-32 forward-gap components.

Reads a trace.json(.gz) exported by --torch-profile, correlates every CUDA
kernel / memcpy back to its launching runtime call, sweeps the python_function
timeline to find the enclosing python stack at launch time, and buckets the
kernel's GPU time into: trellis reconstruct, Hadamard/sign activation
transforms, base GEMM, LoRA GEMMs, activation-offload copies, dtype casts,
attention, cross-entropy, optimizer, eager glue, other.

Usage: python training/experiments/profile_fwdgap_analyze.py \
           out/profile_fwdgap_llama3b/torch_profile/trace.json.gz [steps]
"""

import gzip
import json
import sys
from collections import Counter, defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def categorize(kname, stack, dims):
    if "reconstruct_kernel" in kname:
        return "reconstruct"
    joined = "\x00".join(stack)
    in_bm = "_base_matmul" in joined
    in_fn = "qlora_linear.py" in joined
    offload = ("save_on_cpu" in joined or "pack_hook" in joined
               or "unpack_hook" in joined)
    if "Memcpy" in kname or "memcpy" in kname.lower():
        if offload:
            return "offload_copy"
        if in_fn:
            return "cast_copy"
        return "other_copy"
    if "flash_attn" in joined or "flash_attention" in joined:
        return "attention"
    if "fused_ce" in joined:
        return "cross_entropy"
    if "optimizer" in joined.lower() or "adamw" in joined.lower():
        return "optimizer"
    gemm = ("gemm" in kname.lower() or "cutlass" in kname.lower()
            or "splitKreduce" in kname)
    if in_bm:
        if gemm:
            # 128-block Hadamard matmuls vs the big base GEMM
            if dims and any(d == [128, 128] for d in dims):
                return "had_transform"
            return "base_gemm"
        return "had_transform"                # suh/svh scaling, views
    if in_fn:
        if gemm:
            return "lora"                     # only adapter mms live here
        return "cast_copy"                    # .to() casts and adds
    if offload:
        return "offload_copy"
    if "checkpoint" in joined:
        return "ckpt_misc"
    if "native_llama.py" in joined:
        return "eager_glue"
    return "other"


def main(path, steps=10):
    ev = load(path)

    frames = defaultdict(list)          # tid -> [(ts, end, name)]
    for e in ev:
        if e.get("cat") == "python_function" and "dur" in e:
            frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for tid in frames:
        frames[tid].sort()

    mm_ops = defaultdict(list)          # tid -> [(ts, end, dims)]
    for e in ev:
        if e.get("cat") == "cpu_op" and e.get("name") == "aten::mm":
            d = e.get("args", {}).get("Input Dims")
            mm_ops[e["tid"]].append((e["ts"], e["ts"] + e["dur"], d))
    for tid in mm_ops:
        mm_ops[tid].sort()

    # queries: one per kernel/memcpy, at its launching runtime call's ts
    launch = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                launch[c] = (e["tid"], e["ts"])

    queries = defaultdict(list)         # tid -> [(ts, kname, dur)]
    uncorrelated = 0.0
    tot = 0.0
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur", 0.0)
        tot += dur
        c = e.get("args", {}).get("correlation")
        tidts = launch.get(c)
        if tidts is None:
            uncorrelated += dur
            continue
        queries[tidts[0]].append((tidts[1], e.get("name", ""), dur))

    cat_time = Counter()
    cat_calls = Counter()
    examples = defaultdict(Counter)
    cat_time["uncorrelated"] = uncorrelated

    for tid, qs in queries.items():
        qs.sort()
        fl = frames.get(tid, [])
        ol = mm_ops.get(tid, [])
        fi = oi = 0
        stack = []                       # active frames, outer -> inner
        cur_mm = None
        for ts, kname, dur in qs:
            while fi < len(fl) and fl[fi][0] <= ts:
                s, e2, name = fl[fi]
                fi += 1
                if e2 <= ts:
                    continue             # already finished before the query
                while stack and stack[-1][0] <= s:
                    stack.pop()          # pop frames that ended before start
                stack.append((e2, name))
            while stack and stack[-1][0] <= ts:
                stack.pop()
            while oi < len(ol) and ol[oi][0] <= ts:
                cur_mm = ol[oi]
                oi += 1
            dims = cur_mm[2] if cur_mm and cur_mm[1] > ts else None
            names = [n for _, n in stack]
            cat = categorize(kname, names, dims)
            cat_time[cat] += dur
            cat_calls[cat] += 1
            examples[cat][kname[:80]] += 1

    print(f"total GPU time {tot/1e6:.3f}s over {steps} steps "
          f"= {tot/1e3/steps:.1f}ms/step\n")
    print(f"{'category':<18}{'ms/step':>10}{'% GPU':>8}{'calls/step':>12}")
    for cat, t in cat_time.most_common():
        print(f"{cat:<18}{t/1e3/steps:>10.1f}{100*t/tot:>7.1f}%"
              f"{cat_calls[cat]/steps:>12.0f}")
    print("\ntop kernels per category:")
    for cat, t in cat_time.most_common():
        print(f"-- {cat}")
        for k, n in examples[cat].most_common(3):
            print(f"     {n:>6}x  {k}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
