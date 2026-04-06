#!/usr/bin/env python3
"""Extract the first word from each field in a JSONL file."""

import json
import sys


def trim(value):
    if isinstance(value, str):
        parts = value.split()
        return parts[0] if parts else ""
    if isinstance(value, dict):
        return {k: trim(v) for k, v in value.items()}
    if isinstance(value, list):
        return [trim(item) for item in value]
    return value


def process_jsonl(input_path, output_path):
    with open(input_path) as infile:
        lines = infile.readlines()

    with open(output_path, "w") as outfile:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result = {k: trim(v) for k, v in record.items()}
            outfile.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
        sys.exit(1)
    process_jsonl(sys.argv[1], sys.argv[2])
    print("Done.")
