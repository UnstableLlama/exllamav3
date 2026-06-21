"""
Score and filter a styled dataset by STYLE DENSITY, with metrics.

Motivation
----------
A style LoRA only surfaces at lora_scaling 1.0 (the stable operating point --
scaling >1 degrades the base model) if the per-token style signal in the
training data is dense. Yoda is a syntactic transform that is dense at the
SENTENCE level but sparse at the TOKEN level, and the first generation pass
leaves many sentences in normal order, diluting the signal. This tool measures
how Yoda-like each row actually is, shows the distribution so you choose a
threshold from evidence (not a blind percentile), and writes the rows that pass.

Metric (style=yoda): clause-final inversion rate
-------------------------------------------------
Yoda's hallmark is putting the verb/auxiliary last: "...you have", "...it is",
"...learn you will", "...help you I can". Normal English almost never ends a
sentence on an auxiliary/modal or a bare subject pronoun. So we score a row as

    yoda_score = (# sentences ending in a Yoda marker) / (# sentences)

where a "Yoda marker" is the final word being an auxiliary/copula/modal, a
subject/object pronoun, or "not". Plain English text scores ~0.0-0.1 on this;
densely-inverted Yoda scores ~0.5+. It's a proxy, not a parser, but it's
monotonic with the thing we care about and trivial to audit.

Usage
-----
    # just report the distribution:
    python examples/score_style_density.py --in /mnt/two/data/yoda_alpaca.jsonl

    # write the rows at/above a chosen threshold:
    python examples/score_style_density.py --in /mnt/two/data/yoda_alpaca.jsonl \
        --out /mnt/two/data/yoda_dense.jsonl --min-score 0.5
"""

import argparse
import json
import re
import statistics

# Words whose presence as the LAST token of a sentence signals Yoda inversion.
AUX = {
    "is", "are", "am", "was", "were", "be", "been", "being",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "do", "does", "did", "have", "has", "had",
}
PRON = {"i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them"}
SUBJ_PRON = {"i", "you", "he", "she", "it", "we", "they"}
YODA_FINAL = AUX | PRON | {"not"}

# Split on sentence punctuation, but NOT a period between digits (so "1.7" and
# "U.S" style decimals don't create spurious sentence breaks).
_SENT_SPLIT = re.compile(r"(?<!\d)[.!?]+(?!\d)")
_WORD = re.compile(r"[A-Za-z']+")


def _is_inverted(words):
    """A sentence counts as Yoda-inverted if EITHER:
      (a) it ends on a verb/auxiliary/pronoun  ("...it is", "...you have"), or
      (b) a subject pronoun + auxiliary cluster appears NON-initially -- a
          displaced subject, the tell of front-loaded inversion like
          "Feel I do, ..." or "..., they do".  Requiring the cluster to be
          non-initial avoids matching normal SVO openers ("It is...", "I have...").
    """
    if words[-1] in YODA_FINAL:
        return True
    # Front-loaded inversion: a subject pronoun + auxiliary with NO subject
    # pronoun before it (the subject is displaced rightward, after a fronted
    # predicate). The "no subject before" guard rejects normal subordinate
    # clauses like "I think it is good" / "we know that they are here".
    seen_subj = False
    for i in range(len(words) - 1):
        if i >= 1 and not seen_subj and words[i] in SUBJ_PRON and words[i + 1] in AUX:
            return True
        if words[i] in SUBJ_PRON:
            seen_subj = True
    return False


def yoda_score(text, min_sentence_words=3):
    sents = []
    for chunk in _SENT_SPLIT.split(text):
        words = _WORD.findall(chunk.lower())
        if len(words) >= min_sentence_words:
            sents.append(words)
    if not sents:
        return 0.0, 0
    hits = sum(1 for w in sents if _is_inverted(w))
    return hits / len(sents), len(sents)


SCORERS = {"yoda": yoda_score}


def histogram(scores, bins=10, width=40):
    lines = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        n = sum(1 for s in scores if (lo <= s < hi) or (b == bins - 1 and s == 1.0))
        bar = "#" * round(width * n / max(1, len(scores)))
        lines.append(f"  [{lo:.1f},{hi:.1f}) {n:6d} {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Styled .jsonl")
    ap.add_argument("--out", default=None,
                    help="If set, write rows with score >= --min-score here")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--style", default="yoda", choices=list(SCORERS.keys()))
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--show", type=int, default=4,
                    help="Print this many example rows per score band")
    args = ap.parse_args()

    scorer = SCORERS[args.style]
    rows, scores = [], []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sc, nsent = scorer(rec.get(args.response_key) or "")
            rec["_score"] = sc
            rows.append(rec)
            scores.append(sc)

    if not rows:
        raise SystemExit("No rows found.")

    scores_sorted = sorted(scores)
    n = len(scores)

    def pct(p):
        return scores_sorted[min(n - 1, int(p * n))]

    print(f"\nScored {n} rows ({args.style} clause-final inversion rate)\n")
    print(f"  mean   {statistics.mean(scores):.3f}")
    print(f"  median {statistics.median(scores):.3f}")
    print(f"  p10/p25/p75/p90  "
          f"{pct(0.10):.3f} / {pct(0.25):.3f} / {pct(0.75):.3f} / {pct(0.90):.3f}")
    print(f"\nDistribution:\n{histogram(scores)}")

    print("\nRows passing each threshold:")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        k = sum(1 for s in scores if s >= t)
        print(f"  >= {t:.1f}: {k:6d}  ({100 * k / n:5.1f}%)")

    # Example rows near low / mid / high score, to calibrate the threshold.
    if args.show:
        for label, lo, hi in (("LOW (~0.0-0.2)", 0.0, 0.2),
                               ("MID (~0.4-0.6)", 0.4, 0.6),
                               ("HIGH (>=0.7)", 0.7, 1.01)):
            band = [r for r in rows if lo <= r["_score"] < hi][:args.show]
            if band:
                print(f"\n--- {label} ---")
                for r in band:
                    txt = (r.get(args.response_key) or "").replace("\n", " ")
                    print(f"  [{r['_score']:.2f}] {txt[:160]}")

    if args.out:
        kept = [r for r in rows if r["_score"] >= args.min_score]
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept:
                r.pop("_score", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(kept)} rows (score >= {args.min_score}) to {args.out}")
    else:
        print("\n(no --out given; reporting only. Re-run with --out + --min-score "
              "to write the filtered set.)")


if __name__ == "__main__":
    main()
