#!/usr/bin/env python3
"""
summarize_logs.py -- Summarize LLM proxy JSONL log entries using Ollama.

Usage:
    python summarize_logs.py [--config PATH] [--model NAME] [--ollama-url URL]
                             [--input-file PATH] [--output-file PATH]

Reads the JSONL output file from the LLM proxy, sends each logged conversation
to a local Ollama model for summarization, and writes all summaries to a
Markdown file.
"""

import os
import sys
import json
import argparse
import tomllib
import datetime
import httpx


PROMPT_TEMPLATE = """\
Below is a logged LLM API exchange. Summarize in 2-3 sentences:
(1) what the user was asking or trying to accomplish, and
(2) what the assistant's response covered.

Metadata:
- Timestamp: {timestamp}
- Model: {model}
- Endpoint: {endpoint}
- Input tokens: {input_tokens}
- Output tokens: {output_tokens}
- Stop reason: {stop_reason}

Conversation:
{conversation}

Assistant response:
{response_text}

Provide a concise summary (2-3 sentences only, no bullet points).\
"""


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        print(f"Error: invalid TOML in {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LLM proxy JSONL log entries using Ollama."
    )
    parser.add_argument(
        "--config",
        default="summarizer.toml",
        help="Path to TOML config file (default: summarizer.toml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (overrides config)",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        default=None,
        help="Ollama base URL (overrides config)",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        default=None,
        help="Path to JSONL log file (overrides config)",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        default=None,
        help="Path for Markdown summary output (overrides config)",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"Warning: skipping line {lineno} in {path} -- JSON decode error: {exc}",
                        file=sys.stderr,
                    )
    except FileNotFoundError:
        print(f"Error: input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return entries


def _extract_text_from_content(content) -> str:
    """Extract plain text from a message content field.

    Content may be a plain string or a list of content blocks (dicts with
    a 'type' and 'text' key, as used by the Anthropic API).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def build_prompt(entry: dict) -> str:
    request_body = entry.get("request_body", {})
    parts = []

    # Optional system prompt
    system = request_body.get("system")
    if system is not None:
        system_text = _extract_text_from_content(system)
        system_text = system_text[:2000] + "... [truncated]" if len(system_text) > 2000 else system_text
        parts.append(f"[System]: {system_text}")

    # Messages
    for message in request_body.get("messages", []):
        role = message.get("role", "unknown").capitalize()
        content = _extract_text_from_content(message.get("content", ""))
        content = content[:2000] + "... [truncated]" if len(content) > 2000 else content
        parts.append(f"[{role}]: {content}")

    conversation = "\n\n".join(parts)
    # Cap total conversation to avoid exceeding the model's context window
    if len(conversation) > 6000:
        conversation = conversation[:6000] + "\n\n... [conversation truncated]"

    response_text = entry.get("response_text", "")
    if len(response_text) > 3000:
        response_text = response_text[:3000] + "... [truncated]"

    return PROMPT_TEMPLATE.format(
        timestamp=entry.get("timestamp", "N/A"),
        model=entry.get("model", "N/A"),
        endpoint=entry.get("endpoint", "N/A"),
        input_tokens=entry.get("input_tokens", "N/A"),
        output_tokens=entry.get("output_tokens", "N/A"),
        stop_reason=entry.get("stop_reason", "N/A"),
        conversation=conversation,
        response_text=response_text,
    )


def call_ollama(ollama_url: str, model: str, prompt: str) -> str:
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = httpx.post(url, json=payload, timeout=httpx.Timeout(120.0))
    resp.raise_for_status()
    result = resp.json().get("response", "").strip()
    if not result:
        raise ValueError(f"Ollama returned empty response. Raw body: {resp.text[:200]}")
    return result


def format_summary_entry(index: int, entry: dict, summary: str) -> str:
    return (
        f"## Entry {index}\n"
        f"\n"
        f"- **Timestamp:** {entry.get('timestamp', 'N/A')}\n"
        f"- **Model:** {entry.get('model', 'N/A')}\n"
        f"- **Endpoint:** {entry.get('endpoint', 'N/A')}\n"
        f"- **Tokens:** {entry.get('input_tokens', 'N/A')} in / {entry.get('output_tokens', 'N/A')} out\n"
        f"- **Stop reason:** {entry.get('stop_reason', 'N/A')}\n"
        f"\n"
        f"**Summary:** {summary}\n"
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI args override config values; config file is optional if all are provided via CLI
    model = args.model if args.model is not None else config.get("model")
    ollama_url = args.ollama_url if args.ollama_url is not None else config.get("ollama_url")
    input_file = args.input_file if args.input_file is not None else config.get("input_file")
    output_file = args.output_file if args.output_file is not None else config.get("output_file")

    missing = [k for k, v in [("model", model), ("ollama_url", ollama_url), ("input_file", input_file), ("output_file", output_file)] if v is None]
    if missing:
        print(f"Error: missing required value(s): {', '.join(missing)}. Provide via CLI or summarizer.toml.", file=sys.stderr)
        sys.exit(1)

    entries = read_jsonl(input_file)
    if not entries:
        print(f"No log entries found in {input_file}.")
        sys.exit(0)

    # Validate Ollama is reachable and model is pulled
    try:
        tags_resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        tags_resp.raise_for_status()
        pulled_models = {m["name"] for m in tags_resp.json().get("models", [])}
        # Normalize: treat "name" and "name:latest" as equivalent
        def _normalize(name: str) -> str:
            return name if ":" in name else f"{name}:latest"
        if _normalize(model) not in {_normalize(m) for m in pulled_models}:
            print(
                f"Error: model '{model}' is not pulled in Ollama. "
                f"Run: ollama pull {model}",
                file=sys.stderr,
            )
            sys.exit(1)
    except httpx.RequestError as exc:
        print(f"Error: cannot reach Ollama at {ollama_url} -- {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(entries)
    print(f"Processing {total} log entries with model '{model}'...\n")

    tmp_output = output_file + ".tmp"
    try:
        with open(tmp_output, "w", encoding="utf-8") as fh:
            # Write Markdown header
            fh.write("# LLM Proxy Log Summaries\n")
            fh.write("\n")
            fh.write(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"Source: {input_file}\n")
            fh.write(f"Total entries: {total}\n")
            fh.write(f"Summarizer model: {model}\n")
            fh.write("\n")
            fh.write("---\n")
            fh.write("\n")

            for i, entry in enumerate(entries, 1):
                m = entry.get("model", "N/A")
                ts = entry.get("timestamp", "N/A")
                print(f"[{i}/{total}] Summarizing entry {i} (model={m}, {ts})... ", end="", flush=True)

                try:
                    prompt = build_prompt(entry)
                    summary = call_ollama(ollama_url, model, prompt)
                    print("done.")
                except Exception as exc:
                    summary = f"[ERROR: Could not generate summary -- {exc}]"
                    print(f"FAILED: {exc}")

                fh.write(format_summary_entry(i, entry, summary))
                fh.write("\n---\n\n")
                fh.flush()

        os.replace(tmp_output, output_file)
    except Exception:
        # Clean up temp file on unexpected failure
        if os.path.exists(tmp_output):
            os.unlink(tmp_output)
        raise

    print(f"\nDone. Summaries written to: {output_file}")


if __name__ == "__main__":
    main()
