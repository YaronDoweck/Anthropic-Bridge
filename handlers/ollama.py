"""
Ollama handler.

Forwards incoming Anthropic-format requests directly to Ollama's
Anthropic-compatible endpoint, with optional system-reminder extraction.
"""

from __future__ import annotations

import copy
import json
import re
import logging
from typing import Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import ModelConfig
from handlers.base import BaseHandler
from logging_config import format_body_short, get_debug_level

logger = logging.getLogger(__name__)

# Only safe, generic headers are forwarded to Ollama
_ALLOW_HEADERS = frozenset({"content-type", "accept", "user-agent", "accept-encoding"})

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>\s*(.*?)\s*</system-reminder>", re.DOTALL)


class OllamaHandler(BaseHandler):
    async def handle(
        self,
        request: Request,
        path: str,
        body: dict,
        model_config: ModelConfig,
    ) -> tuple[Union[Response, StreamingResponse], Optional[dict]]:
        url = f"{self.base_url}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"

        # Forward only safe headers
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() in _ALLOW_HEADERS
        }
        headers["content-type"] = "application/json"

        debug_level = get_debug_level()

        # Work on a deep copy to avoid mutating the caller's dict
        send_body = copy.deepcopy(body)

        if model_config.omit_claude_main_description and model_config.claude_system_instructions != "passthrough":
            send_body.pop("system", None)

        if model_config.claude_system_instructions == "strip":
            system = send_body.get("system", "")
            if isinstance(system, str):
                search_str = "</system-reminder>\n\n"
                if search_str in system:
                    send_body["system"] = system.split(search_str)[1]
        elif model_config.claude_system_instructions == "split":
            messages = send_body.get("messages", [])
            extracted: list[str] = []
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                blocks = _SYSTEM_REMINDER_RE.findall(content)
                if blocks:
                    extracted.extend(b.strip() for b in blocks)
                    msg["content"] = _SYSTEM_REMINDER_RE.sub("", content).strip()
            if extracted:
                existing = send_body.get("system", "")
                extra = "\n\n".join(extracted)
                if isinstance(existing, str) and existing:
                    send_body["system"] = existing + "\n\n" + extra
                elif isinstance(existing, list):
                    existing.append({"type": "text", "text": extra})
                else:
                    send_body["system"] = extra

        if debug_level == 1:
            logger.debug("OLLAMA REQUEST: %s", format_body_short(send_body))
        elif debug_level >= 2:
            logger.debug(
                "OLLAMA REQUEST:\n%s",
                json.dumps(send_body, indent=2, ensure_ascii=False),
            )

        is_streaming = body.get("stream", False)

        if is_streaming:
            response = await self._handle_streaming(url, headers, send_body, debug_level)
        else:
            response = await self._handle_buffered(url, headers, send_body, debug_level)
        return (response, send_body)

    async def _handle_buffered(
        self,
        url: str,
        headers: dict,
        body: dict,
        debug_level: int,
    ) -> Response:
        resp = await self.client.request(
            method="POST",
            url=url,
            headers=headers,
            content=json.dumps(body).encode(),
        )

        if resp.status_code != 200:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={"content-type": "application/json"},
            )

        if debug_level == 1:
            logger.debug("OLLAMA RESPONSE: status=%d", resp.status_code)
        elif debug_level >= 2:
            try:
                logger.debug(
                    "OLLAMA RESPONSE:\n%s",
                    json.dumps(resp.json(), indent=2, ensure_ascii=False),
                )
            except Exception:
                logger.debug("OLLAMA RESPONSE (raw): %s", resp.text[:500])

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={"content-type": resp.headers.get("content-type", "application/json")},
        )

    async def _handle_streaming(
        self,
        url: str,
        headers: dict,
        body: dict,
        debug_level: int,
    ) -> StreamingResponse:
        async def _stream_generator():
            accumulated_text = []
            async with self.client.stream(
                method="POST",
                url=url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    if debug_level >= 1:
                        try:
                            for line in chunk.decode().splitlines():
                                if line.startswith("data:"):
                                    data = json.loads(line[5:])
                                    if data.get("type") == "content_block_delta":
                                        delta_data = data.get("delta", {})
                                        accumulated_text.append(delta_data.get("text", "") or delta_data.get("thinking", ""))
                        except Exception:
                            pass
                    yield chunk
            if debug_level >= 1 and accumulated_text:
                text = "".join(accumulated_text)
                if debug_level == 1:
                    preview = text[:80].replace("\n", " ")
                    logger.debug("OLLAMA RESPONSE [stream]: %r", preview)
                else:
                    logger.debug("OLLAMA RESPONSE [stream]:\n%s", text)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
        )
