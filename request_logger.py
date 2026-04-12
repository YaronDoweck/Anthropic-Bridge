"""
RequestLogger: appends JSONL entries for every proxied request/response.
RawLogger: appends raw incoming request bodies to a single JSONL file.

Activated when the OUTPUT_DIR / RAW_LOG_DIR environment variables are set.
Each session writes to its own <session_id>.jsonl file inside the output directory.
All I/O is wrapped in try/except so a logging failure never crashes the proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


def _extract_session_id(request_body: dict) -> str:
    """Extract session_id from request_body.metadata.user_id JSON string."""
    try:
        user_id_str = request_body.get("metadata", {}).get("user_id", "")
        user_id = json.loads(user_id_str)
        session_id = user_id.get("session_id", "")
        if not session_id or not isinstance(session_id, str):
            return "unknown"
        # Sanitize: allow only alphanumeric, hyphens, underscores (covers UUIDs)
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        if not sanitized or not sanitized.strip('_'):
            return "unknown"
        return sanitized[:200]
    except Exception:
        return "unknown"


class RequestLogger:
    def __init__(self, output_dir: str) -> None:
        self._dir: Optional[str] = output_dir
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise OSError(f"Directory is not writable: {output_dir}")
        except OSError as exc:
            logger.warning(
                "RequestLogger: cannot use output directory %r: %s — logging disabled",
                output_dir,
                exc,
            )
            self._dir = None

    def log_entry(self, entry: dict, session_id: str = "unknown") -> None:
        """Append one JSON line to the session's output file. Never raises."""
        if self._dir is None:
            return
        try:
            path = os.path.join(self._dir, f"{session_id}.jsonl")
            line = json.dumps(entry, ensure_ascii=False)
            with self._get_session_lock(session_id):
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("RequestLogger: failed to write log entry: %s", exc)

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def log_buffered_response(
        self,
        request_body: dict,
        model: str,
        endpoint: str,
        response_body: bytes,
        timestamp: str,
        status_code: int = 200,
        session_id: Optional[str] = None,
    ) -> None:
        """Parse a complete (buffered) Anthropic-format response and log it."""
        if session_id is None:
            session_id = _extract_session_id(request_body)
        try:
            parsed = json.loads(response_body)
            content_blocks = parsed.get("content", [])
            text_parts = []
            thinking_parts = []
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
                elif block.get("type") == "tool_use":
                    name = block.get("name", "")
                    input_data = json.dumps(block.get("input", {}), ensure_ascii=False)
                    text_parts.append(f"[tool_use: {name}] {input_data}".strip())
            response_text = "\n".join(text_parts)
            thinking_text = "\n".join(thinking_parts)
            usage = parsed.get("usage", {})
            _raw_input = usage.get("input_tokens")
            input_tokens: Optional[int] = (
                _raw_input
                + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
            ) if _raw_input is not None else None
            output_tokens: Optional[int] = usage.get("output_tokens")
            stop_reason: Optional[str] = parsed.get("stop_reason")
        except Exception as exc:
            logger.warning("RequestLogger: failed to parse buffered response: %s", exc)
            response_text = ""
            thinking_text = ""
            input_tokens = None
            output_tokens = None
            stop_reason = None

        self.log_entry(
            {
                "timestamp": timestamp,
                "model": model,
                "endpoint": endpoint,
                "stream": False,
                "status_code": status_code,
                "request_body": request_body,
                "thinking_text": thinking_text,
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
            },
            session_id=session_id,
        )

    async def wrap_streaming_generator(
        self,
        request_body: dict,
        model: str,
        endpoint: str,
        original_generator: AsyncGenerator[bytes, None],
        timestamp: str,
        session_id: str = "unknown",
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator that yields every chunk from *original_generator* unchanged,
        while accumulating text, token counts, and stop_reason from SSE events.
        Writes a JSONL entry after the stream is exhausted (or on disconnect).
        """
        stream_start = time.monotonic()
        accumulated_text: list[str] = []
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        stop_reason: Optional[str] = None
        leftover = ""  # buffer for partial SSE lines split across chunks
        # Tool use tracking: index -> {name, partial_json_parts}
        tool_blocks: dict[int, dict] = {}
        # Thinking tracking: index -> [parts]
        thinking_blocks: dict[int, list] = {}
        current_block_index: Optional[int] = None
        first_chunk_time: Optional[float] = None
        estimated_tokens: int = 0
        last_rate_log_time: float = 0.0
        last_rate_log_tokens: int = 0

        try:
            async for chunk in original_generator:
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                    last_rate_log_time = first_chunk_time
                yield chunk
                # Parse SSE lines, handling chunk-boundary splits
                try:
                    decoded = leftover + chunk.decode("utf-8", errors="replace")
                    lines = decoded.split("\n")
                    # Last element may be incomplete — keep it as leftover
                    leftover = lines[-1]
                    for line in lines[:-1]:
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str in ("", "[DONE]"):
                            continue
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event_data.get("type")

                        if event_type == "message_start":
                            msg = event_data.get("message", {})
                            usage = msg.get("usage", {})
                            val = usage.get("input_tokens")
                            if val is not None:
                                input_tokens = (
                                    val
                                    + (usage.get("cache_read_input_tokens") or 0)
                                    + (usage.get("cache_creation_input_tokens") or 0)
                                )

                        elif event_type == "content_block_start":
                            block = event_data.get("content_block", {})
                            idx = event_data.get("index", 0)
                            current_block_index = idx
                            if block.get("type") == "tool_use":
                                tool_blocks[idx] = {
                                    "name": block.get("name", ""),
                                    "parts": [],
                                }
                            elif block.get("type") == "thinking":
                                thinking_blocks[idx] = []

                        elif event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    accumulated_text.append(text)
                                    estimated_tokens += max(1, len(text) // 4)
                                    now = time.monotonic()
                                    if first_chunk_time is not None and (now - last_rate_log_time >= 5.0 or estimated_tokens - last_rate_log_tokens >= 50):
                                        elapsed = now - first_chunk_time
                                        if elapsed > 0:
                                            logger.info("STREAM %s: ~%d estimated tokens in %.1fs = %.1f tok/s (in progress)", model, estimated_tokens, elapsed, estimated_tokens / elapsed)
                                            last_rate_log_time = now
                                            last_rate_log_tokens = estimated_tokens
                            elif delta.get("type") == "input_json_delta":
                                idx = event_data.get("index", current_block_index)
                                if idx in tool_blocks:
                                    partial = delta.get("partial_json", "")
                                    if partial:
                                        tool_blocks[idx]["parts"].append(partial)
                            elif delta.get("type") == "thinking_delta":
                                idx = event_data.get("index", current_block_index)
                                if idx in thinking_blocks:
                                    thinking = delta.get("thinking", "")
                                    if thinking:
                                        thinking_blocks[idx].append(thinking)
                                        estimated_tokens += max(1, len(thinking) // 4)
                                        now = time.monotonic()
                                        if first_chunk_time is not None and (now - last_rate_log_time >= 5.0 or estimated_tokens - last_rate_log_tokens >= 50):
                                            elapsed = now - first_chunk_time
                                            if elapsed > 0:
                                                logger.info("STREAM %s: ~%d estimated tokens in %.1fs = %.1f tok/s (in progress, thinking)", model, estimated_tokens, elapsed, estimated_tokens / elapsed)
                                                last_rate_log_time = now
                                                last_rate_log_tokens = estimated_tokens

                        elif event_type == "message_delta":
                            delta = event_data.get("delta", {})
                            sr = delta.get("stop_reason")
                            if sr is not None:
                                stop_reason = sr
                            usage = event_data.get("usage", {})
                            val = usage.get("output_tokens")
                            if val is not None:
                                output_tokens = val
                            val = usage.get("input_tokens")
                            if val is not None:
                                input_tokens = (
                                    val
                                    + (usage.get("cache_read_input_tokens") or 0)
                                    + (usage.get("cache_creation_input_tokens") or 0)
                                )

                except Exception as exc:
                    logger.warning("RequestLogger: error parsing SSE chunk: %s", exc)
        finally:
            thinking_text = ""
            if thinking_blocks:
                t_parts = []
                for thinking_parts_list in thinking_blocks.values():
                    t_parts.append("".join(thinking_parts_list))
                thinking_text = "\n".join(t_parts)

            response_parts = []
            if accumulated_text:
                response_parts.append("".join(accumulated_text))
            for block in tool_blocks.values():
                name = block["name"]
                partial_json = "".join(block["parts"])
                response_parts.append(f"[tool_use: {name}] {partial_json}".strip())
            response_text = "\n".join(response_parts)
            if first_chunk_time is not None:
                ttft = first_chunk_time - stream_start
                elapsed = time.monotonic() - first_chunk_time
                if elapsed > 0:
                    if output_tokens is not None:
                        logger.info("STREAM %s: TTFT=%.2fs | %d tokens in %.1fs = %.1f tok/s", model, ttft, output_tokens, elapsed, output_tokens / elapsed)
                    else:
                        logger.info("STREAM %s: TTFT=%.2fs | ~%d estimated tokens in %.1fs = ~%.1f tok/s (estimated)", model, ttft, estimated_tokens, elapsed, estimated_tokens / elapsed)
            else:
                ttft = time.monotonic() - stream_start
                logger.info("STREAM %s: no tokens received (TTFT>%.2fs)", model, ttft)
            entry = {
                "timestamp": timestamp,
                "model": model,
                "endpoint": endpoint,
                "stream": True,
                "status_code": 200,
                "request_body": request_body,
                "thinking_text": thinking_text,
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
            }
            # Run blocking file I/O in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda e=entry: self.log_entry(e, session_id=session_id))


class RawLogger:
    """Appends raw incoming request bodies to a single raw_requests.jsonl file."""

    def __init__(self, output_dir: str) -> None:
        self._path: Optional[str] = None
        self._lock = threading.Lock()
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise OSError(f"Directory is not writable: {output_dir}")
            self._path = os.path.join(output_dir, "raw_requests.jsonl")
        except OSError as exc:
            logger.warning(
                "RawLogger: cannot use output directory %r: %s — raw logging disabled",
                output_dir,
                exc,
            )

    def log(self, timestamp: str, method: str, path: str, body: str) -> None:
        """Append one raw request entry. Never raises."""
        if self._path is None:
            return
        try:
            entry = {"timestamp": timestamp, "direction": "request", "method": method, "path": path, "body": body}
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:
            logger.warning("RawLogger: failed to write entry: %s", exc)

    def log_response(self, timestamp: str, method: str, path: str, status_code: int, body: str) -> None:
        """Append one raw response entry. Never raises."""
        if self._path is None:
            return
        try:
            entry = {"timestamp": timestamp, "direction": "response", "method": method, "path": path, "status_code": status_code, "body": body}
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:
            logger.warning("RawLogger: failed to write response entry: %s", exc)
