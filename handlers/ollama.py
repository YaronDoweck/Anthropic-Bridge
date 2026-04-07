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
import uuid
from typing import Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import ModelConfig
from handlers.base import BaseHandler
from logging_config import format_body_short, get_debug_level
from converters.anthropic_openai import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic_response,
    convert_openai_stream_to_anthropic,
)

logger = logging.getLogger(__name__)

# Only safe, generic headers are forwarded to Ollama
_ALLOW_HEADERS = frozenset({"content-type", "accept", "user-agent", "accept-encoding"})

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>\s*(.*?)\s*</system-reminder>", re.DOTALL)

_STATUS_TO_ANTHROPIC_ERROR = {
    401: "authentication_error",
    429: "rate_limit_error",
    400: "invalid_request_error",
}


def _anthropic_error_response(status_code: int, message: str) -> JSONResponse:
    error_type = _STATUS_TO_ANTHROPIC_ERROR.get(
        status_code, "api_error" if status_code >= 500 else "invalid_request_error"
    )
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


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

        if model_config.ollama_format != "openai":
            if debug_level == 1:
                logger.debug("OLLAMA REQUEST: %s", format_body_short(send_body))
            elif debug_level >= 2:
                logger.debug(
                    "OLLAMA REQUEST:\n%s",
                    json.dumps(send_body, indent=2, ensure_ascii=False),
                )

        is_streaming = body.get("stream", False)

        sent_body = send_body
        if model_config.ollama_format == "openai":
            openai_url = f"{self.base_url}/v1/chat/completions"
            original_model = body.get("model", "")
            openai_body = convert_anthropic_to_openai_request(send_body)
            sent_body = openai_body
            if debug_level == 1:
                logger.debug("OLLAMA OPENAI REQUEST: %s", format_body_short(openai_body))
            elif debug_level >= 2:
                logger.debug(
                    "OLLAMA OPENAI REQUEST:\n%s",
                    json.dumps(openai_body, indent=2, ensure_ascii=False),
                )
            if is_streaming:
                response = await self._handle_streaming_openai(openai_url, headers, openai_body, original_model, debug_level)
            else:
                response = await self._handle_buffered_openai(openai_url, headers, openai_body, original_model, debug_level)
        else:
            if is_streaming:
                response = await self._handle_streaming(url, headers, send_body, debug_level)
            else:
                response = await self._handle_buffered(url, headers, send_body, debug_level)
        return (response, sent_body)

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

    async def _handle_buffered_openai(
        self,
        url: str,
        headers: dict,
        openai_body: dict,
        original_model: str,
        debug_level: int,
    ) -> Response:
        resp = await self.client.request(
            method="POST",
            url=url,
            headers=headers,
            content=json.dumps(openai_body).encode(),
        )

        if resp.status_code != 200:
            try:
                err = resp.json()
                message = err.get("error", {}).get("message", resp.text)
            except Exception:
                message = resp.text
            return _anthropic_error_response(resp.status_code, message)

        try:
            openai_resp = resp.json()
        except Exception:
            return _anthropic_error_response(500, "Invalid JSON response from Ollama")

        anthropic_resp = convert_openai_to_anthropic_response(openai_resp, original_model)

        if debug_level == 1:
            logger.debug("OLLAMA OPENAI RESPONSE: status=%d", resp.status_code)
        elif debug_level >= 2:
            logger.debug(
                "OLLAMA OPENAI RESPONSE:\n%s",
                json.dumps(anthropic_resp, indent=2, ensure_ascii=False),
            )

        return Response(
            content=json.dumps(anthropic_resp).encode(),
            status_code=200,
            headers={"content-type": "application/json"},
        )

    async def _handle_streaming_openai(
        self,
        url: str,
        headers: dict,
        openai_body: dict,
        original_model: str,
        debug_level: int,
    ) -> StreamingResponse:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def _stream_generator():
            accumulated_text = []
            async with self.client.stream(
                method="POST",
                url=url,
                headers=headers,
                content=json.dumps(openai_body).encode(),
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    try:
                        err = json.loads(error_body)
                        message = err.get("error", {}).get("message", error_body.decode())
                    except Exception:
                        message = error_body.decode()
                    error_type = _STATUS_TO_ANTHROPIC_ERROR.get(
                        resp.status_code, "api_error" if resp.status_code >= 500 else "invalid_request_error"
                    )
                    error_dict = {"type": "error", "error": {"type": error_type, "message": message}}
                    yield f"event: error\ndata: {json.dumps(error_dict)}\n\n".encode()
                    return

                async for chunk in convert_openai_stream_to_anthropic(resp, original_model, message_id):
                    if debug_level >= 1:
                        try:
                            line = chunk.decode()
                            if "text_delta" in line and "content_block_delta" in line:
                                data = json.loads(line.split("data: ", 1)[1])
                                accumulated_text.append(data.get("delta", {}).get("text", ""))
                        except Exception:
                            pass
                    yield chunk

            if debug_level >= 1 and accumulated_text:
                text = "".join(accumulated_text)
                if debug_level == 1:
                    preview = text[:80].replace("\n", " ")
                    logger.debug("OLLAMA OPENAI RESPONSE [stream]: %r", preview)
                else:
                    logger.debug("OLLAMA OPENAI RESPONSE [stream]:\n%s", text)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
        )
