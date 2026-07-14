"""
Prune a run's checkpoint-* history to the top-N by held-out eval loss plus the
final (highest-step) checkpoint. The best-checkpoint adapter at the run root
(--save-best) is never touched.

Eval losses are read from the training log ("[eval] step N: held-out X.XXXX"
lines) rather than re-evaluated on GPU — run with checkpoint_every ==
eval_every so every retained checkpoint has a matching eval.

Usage:
    python training/prune_ckpts.py --out out/run_dir --train-log run.log [--top 4]
    (add --dry-run to print the plan without deleting)
"""

import argparse
import os
import re
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir holding checkpoint-* subdirs")
    ap.add_argument("--train-log", required=True,
                    help="training stdout log with the [eval] lines")
    ap.add_argument("--top", type=int, default=4,
                    help="keep this many lowest-held-out-loss checkpoints")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.train_log, errors="replace") as f:
        evals = {int(m.group(1)): float(m.group(2))
                 for m in re.finditer(r"\[eval\] step (\d+): held-out ([0-9.]+)",
                                      f.read())}

    ckpts = {}
    for name in os.listdir(args.out):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and os.path.isdir(os.path.join(args.out, name)):
            ckpts[int(m.group(1))] = name
    if not ckpts:
        print(f"no checkpoint-* dirs in {args.out}; nothing to prune")
        return

    ranked = sorted((s for s in ckpts if s in evals), key=lambda s: evals[s])
    keep = set(ranked[:args.top])
    keep.add(max(ckpts))  # the final checkpoint, regardless of its loss

    print(f"{args.out}: {len(ckpts)} checkpoints, keeping {len(keep)}")
    for step in sorted(ckpts):
        loss = f"{evals[step]:.4f}" if step in evals else "no eval"
        tag = "KEEP" if step in keep else "drop"
        final = "  <- final" if step == max(ckpts) else ""
        print(f"  {tag}  checkpoint-{step:<6} held-out {loss}{final}")
        if step not in keep and not args.dry_run:
            shutil.rmtree(os.path.join(args.out, ckpts[step]))


if __name__ == "__main__":
    main()
