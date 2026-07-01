"""
Normalize "smart"/typographic quotes and apostrophes to plain ASCII across a
JSONL dataset.

Mixed quote styles between rows (some straight, some curly) teach the model
an inconsistent convention and can bump per-token loss on the rarer
curly-quote tokens. This walks every string value in every JSON object --
message content, metadata, everything -- not just a "content" field, so nothing
curly slips through.

Usage:
    python examples/fix_dataset_quotes.py data.jsonl -o data.clean.jsonl
    python examples/fix_dataset_quotes.py data.jsonl --in-place   # backs up to data.jsonl.bak first
    python examples/fix_dataset_quotes.py data.jsonl --dry-run    # report only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Curly/typographic quote & apostrophe variants -> plain ASCII equivalents.
QUOTE_MAP = {
    "‘": "'",  # ' LEFT SINGLE QUOTATION MARK
    "’": "'",  # ' RIGHT SINGLE QUOTATION MARK (the common "smart apostrophe")
    "‚": "'",  # , SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "′": "'",  # PRIME
    "ʼ": "'",  # MODIFIER LETTER APOSTROPHE
    "“": '"',  # " LEFT DOUBLE QUOTATION MARK
    "”": '"',  # " RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "‟": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "″": '"',  # DOUBLE PRIME
}
_TABLE = str.maketrans(QUOTE_MAP)


def normalize(s: str) -> str:
    return s.translate(_TABLE)


def walk(obj):
    """Recursively normalize every string in a JSON-compatible structure."""
    if isinstance(obj, str):
        return normalize(obj)
    if isinstance(obj, list):
        return [walk(v) for v in obj]
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Path to the .jsonl file to clean")
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("-o", "--output", help="Write cleaned JSONL here (input left untouched)")
    out.add_argument("--in-place", action="store_true",
                      help="Overwrite the input file after backing it up to <input>.bak")
    out.add_argument("--dry-run", action="store_true",
                      help="Report what would change; write nothing")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_lines = []
    total_lines = 0
    changed_lines = 0
    total_chars = 0

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                out_lines.append(line)
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{in_path}:{lineno}: invalid JSON ({e}); aborting, no output written")
            # Compare against the canonical re-serialization of the *original*
            # object (not the raw line) so this counts correctly whether the
            # source file used raw UTF-8 quote chars or \uXXXX escapes.
            orig_canonical = json.dumps(obj, ensure_ascii=False)
            cleaned = walk(obj)
            new_line = json.dumps(cleaned, ensure_ascii=False)
            if new_line != orig_canonical:
                changed_lines += 1
                total_chars += sum(orig_canonical.count(ch) for ch in QUOTE_MAP)
            out_lines.append(new_line)

    print(f"{total_lines} rows scanned, {changed_lines} rows changed, "
          f"{total_chars} quote/apostrophe characters normalized.", file=sys.stderr)

    if args.dry_run:
        print("--dry-run: nothing written.", file=sys.stderr)
        return

    if args.in_place:
        out_path = in_path
        backup_path = in_path.with_suffix(in_path.suffix + ".bak")
        shutil.copy2(in_path, backup_path)
        print(f"Backed up {in_path} -> {backup_path}", file=sys.stderr)
    else:
        out_path = Path(args.output)

    with out_path.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
