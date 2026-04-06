"""
Ollama handler.

Converts incoming Anthropic-format requests to OpenAI format (which Ollama's
/v1/chat/completions endpoint speaks), forwards them to Ollama, then converts
the response back to Anthropic format before returning it to the caller.
"""

from __future__ import annotations

import json
import re
import logging
import uuid
from typing import Union

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import ModelConfig
from converters.anthropic_openai import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic_response,
    convert_openai_stream_to_anthropic,
)
from handlers.base import BaseHandler
from logging_config import format_body_short, get_debug_level

logger = logging.getLogger(__name__)

# Anthropic-specific headers that Ollama does not understand
_DROP_HEADERS = frozenset({
    "host",
    "content-length",
    "transfer-encoding",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
})

# Only safe, generic headers are forwarded to Ollama
_ALLOW_HEADERS = frozenset({"content-type", "accept", "user-agent", "accept-encoding"})

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>\s*(.*?)\s*</system-reminder>", re.DOTALL)


def _map_path(path: str) -> str:
    """Map Anthropic paths to their Ollama equivalents."""
    if path == "v1/messages":
        return "v1/chat/completions"
    return path


class OllamaHandler(BaseHandler):
    async def handle(
        self,
        request: Request,
        path: str,
        body: dict,
        model_config: ModelConfig,
    ) -> Union[Response, StreamingResponse]:
        ollama_path = _map_path(path)
        url = f"{self.base_url}/{ollama_path}"
        if request.url.query:
            url += f"?{request.url.query}"

        # Forward only safe headers; drop Anthropic-specific ones
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() in _ALLOW_HEADERS
        }
        headers["content-type"] = "application/json"

        debug_level = get_debug_level()
        model_name = body.get("model", "")
        openai_body = convert_anthropic_to_openai_request(body)
        if model_config.omit_claude_main_description and model_config.claude_system_instructions != "passthrough":
            if len(openai_body["messages"]) >= 2:
                del openai_body["messages"][0]
            else:
                logger.warning(
                    "omit_claude_main_description=true but messages has fewer than 2 elements; skipping"
                )

        if model_config.claude_system_instructions == "strip":
            messages = openai_body.get("messages", [])
            if messages and isinstance(messages[0].get("content"), str):
                search_str = "</system-reminder>\n\n"
                if search_str in messages[0]["content"]:
                    messages[0]["content"] = messages[0]["content"].split(search_str)[1]
        elif model_config.claude_system_instructions == "split":
            messages = openai_body.get("messages", [])
            system_insert_idx = 0
            for msg in messages:
                if msg.get("role") == "system":
                    system_insert_idx += 1
                else:
                    break
            extracted: list[dict] = []
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                blocks = _SYSTEM_REMINDER_RE.findall(content)
                if blocks:
                    for block_text in blocks:
                        extracted.append({"role": "system", "content": block_text.strip()})
                    msg["content"] = _SYSTEM_REMINDER_RE.sub("", content).strip()
            for i, sys_msg in enumerate(extracted):
                messages.insert(system_insert_idx + i, sys_msg)
            openai_body["messages"] = messages

        if debug_level == 1:
            logger.debug("OLLAMA REQUEST: %s", format_body_short(openai_body))
        elif debug_level >= 2:
            logger.debug(
                "OLLAMA REQUEST:\n%s",
                json.dumps(openai_body, indent=2, ensure_ascii=False),
            )
        is_streaming = body.get("stream", False)

        if is_streaming:
            return await self._handle_streaming(url, headers, openai_body, model_name, debug_level)
        else:
            return await self._handle_buffered(url, headers, openai_body, model_name, debug_level)

    async def _handle_buffered(
        self,
        url: str,
        headers: dict,
        openai_body: dict,
        model_name: str,
        debug_level: int,
    ) -> Response:
        resp = await self.client.request(
            method="POST",
            url=url,
            headers=headers,
            content=json.dumps(openai_body).encode(),
        )

        if resp.status_code != 200:
            # Return Ollama's error as-is (already JSON, close enough)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={"content-type": "application/json"},
            )

        try:
            openai_resp = resp.json()
        except Exception as exc:
            logger.error("Failed to parse Ollama response as JSON: %s", exc)
            return Response(
                content=json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": "Upstream returned non-JSON response"},
                }).encode(),
                status_code=502,
                headers={"content-type": "application/json"},
            )

        try:
            anthropic_resp = convert_openai_to_anthropic_response(openai_resp, model_name)
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Failed to convert Ollama response: %s. Response was: %s", e, openai_resp)
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "api_error", "message": f"Invalid response from Ollama backend: {e}"}},
            )

        if debug_level == 1:
            logger.debug("OLLAMA RESPONSE: %s", format_body_short(anthropic_resp))
        elif debug_level >= 2:
            logger.debug(
                "OLLAMA RESPONSE:\n%s",
                json.dumps(anthropic_resp, indent=2, ensure_ascii=False),
            )

        return Response(
            content=json.dumps(anthropic_resp).encode(),
            status_code=200,
            headers={"content-type": "application/json"},
        )

    async def _handle_streaming(
        self,
        url: str,
        headers: dict,
        openai_body: dict,
        model_name: str,
        debug_level: int,
    ) -> StreamingResponse:
        message_id = f"msg_{uuid.uuid4().hex}"

        async def _stream_generator():
            accumulated_text = []
            async with self.client.stream(
                method="POST",
                url=url,
                headers=headers,
                content=json.dumps(openai_body).encode(),
            ) as resp:
                async for chunk in convert_openai_stream_to_anthropic(resp, model_name, message_id):
                    if debug_level >= 1:
                        try:
                            for line in chunk.decode().splitlines():
                                if line.startswith("data:"):
                                    data = json.loads(line[5:])
                                    if data.get("type") == "content_block_delta":
                                        accumulated_text.append(data.get("delta", {}).get("text", ""))
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
