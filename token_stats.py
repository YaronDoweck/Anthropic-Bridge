#!/usr/bin/env python3
"""
Read OUTPUT_FILE (JSONL) and print input/output token totals per model.

Usage:
    python token_stats.py <output_file>
    OUTPUT_FILE=path/to/file python token_stats.py
"""

import json
import os
import sys
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OUTPUT_FILE")
    if not path:
        print("Usage: python token_stats.py <output_file>  (or set OUTPUT_FILE env var)")
        sys.exit(1)

    stats = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0})

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [line {lineno}] skipped (invalid JSON): {exc}", file=sys.stderr)
                continue

            model = entry.get("model", "<unknown>")
            s = stats[model]
            s["requests"] += 1
            s["input_tokens"] += entry.get("input_tokens") or 0
            s["output_tokens"] += entry.get("output_tokens") or 0

    if not stats:
        print("No entries found.")
        return

    col_w = max(len(m) for m in stats) + 2
    header = f"{'Model':<{col_w}}  {'Requests':>10}  {'Input tokens':>14}  {'Output tokens':>14}"
    print(header)
    print("-" * len(header))

    total_req = total_in = total_out = 0
    for model, s in sorted(stats.items()):
        req, inp, out = s["requests"], s["input_tokens"], s["output_tokens"]
        total_req += req
        total_in += inp
        total_out += out
        print(f"{model:<{col_w}}  {req:>10,}  {inp:>14,}  {out:>14,}")

    print("-" * len(header))
    print(f"{'TOTAL':<{col_w}}  {total_req:>10,}  {total_in:>14,}  {total_out:>14,}")


if __name__ == "__main__":
    main()
