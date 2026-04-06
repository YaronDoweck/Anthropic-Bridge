"""
RequestLogger: appends JSONL entries for every proxied request/response.

Activated when the OUTPUT_FILE environment variable is set.
All I/O is wrapped in try/except so a logging failure never crashes the proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class RequestLogger:
    def __init__(self, output_path: str) -> None:
        self._path: Optional[str] = output_path
        self._lock = threading.Lock()
        # Validate writability at startup
        try:
            with open(output_path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            logger.warning(
                "RequestLogger: cannot write to output file %r: %s — logging disabled",
                output_path,
                exc,
            )
            self._path = None

    def log_entry(self, entry: dict) -> None:
        """Append one JSON line to the output file. Never raises."""
        if self._path is None:
            return
        try:
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            logger.warning("RequestLogger: failed to write log entry: %s", exc)

    def log_buffered_response(
        self,
        request_body: dict,
        model: str,
        endpoint: str,
        response_body: bytes,
        timestamp: str,
        status_code: int = 200,
    ) -> None:
        """Parse a complete (buffered) Anthropic-format response and log it."""
        try:
            parsed = json.loads(response_body)
            content_blocks = parsed.get("content", [])
            response_text = "".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            usage = parsed.get("usage", {})
            input_tokens: Optional[int] = usage.get("input_tokens")
            output_tokens: Optional[int] = usage.get("output_tokens")
            stop_reason: Optional[str] = parsed.get("stop_reason")
        except Exception as exc:
            logger.warning("RequestLogger: failed to parse buffered response: %s", exc)
            response_text = ""
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
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
            }
        )

    async def wrap_streaming_generator(
        self,
        request_body: dict,
        model: str,
        endpoint: str,
        original_generator: AsyncGenerator[bytes, None],
        timestamp: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator that yields every chunk from *original_generator* unchanged,
        while accumulating text, token counts, and stop_reason from SSE events.
        Writes a JSONL entry after the stream is exhausted (or on disconnect).
        """
        accumulated_text: list[str] = []
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        stop_reason: Optional[str] = None
        leftover = ""  # buffer for partial SSE lines split across chunks

        try:
            async for chunk in original_generator:
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
                                input_tokens = val

                        elif event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    accumulated_text.append(text)

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
                                input_tokens = val

                except Exception as exc:
                    logger.warning("RequestLogger: error parsing SSE chunk: %s", exc)
        finally:
            entry = {
                "timestamp": timestamp,
                "model": model,
                "endpoint": endpoint,
                "stream": True,
                "request_body": request_body,
                "response_text": "".join(accumulated_text),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "stop_reason": stop_reason,
            }
            # Run blocking file I/O in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.log_entry, entry)
