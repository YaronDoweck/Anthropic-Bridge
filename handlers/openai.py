"""
OpenAI handler.

Accepts incoming Anthropic-format requests, converts them to OpenAI
/v1/chat/completions format, forwards them to an OpenAI-compatible API
endpoint, and converts the response back to Anthropic format.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional, Union

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

# Only safe, generic headers are forwarded to the upstream
_ALLOW_HEADERS = frozenset({"content-type", "accept", "user-agent", "accept-encoding"})

# Map from OpenAI HTTP status codes to Anthropic error types
_STATUS_TO_ERROR_TYPE: dict[int, str] = {
    401: "authentication_error",
    429: "rate_limit_error",
    400: "invalid_request_error",
}


def _openai_error_type(status_code: int) -> str:
    if status_code in _STATUS_TO_ERROR_TYPE:
        return _STATUS_TO_ERROR_TYPE[status_code]
    if 500 <= status_code < 600:
        return "api_error"
    return "api_error"


def _error_response(status_code: int, message: str) -> JSONResponse:
    error_type = _openai_error_type(status_code)
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


class OpenAIHandler(BaseHandler):
    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, http_client)
        self.api_key = api_key

    def _build_headers(self, request: Request) -> dict[str, str]:
        """Build forwarding headers: allowlist only, force content-type, inject Bearer if set."""
        headers: dict[str, str] = {
            k: v
            for k, v in request.headers.items()
            if k.lower() in _ALLOW_HEADERS
        }
        # Always override content-type to ensure JSON
        headers["content-type"] = "application/json"
        # Inject API key, overriding any client-supplied Authorization header
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def handle(
        self,
        request: Request,
        path: str,
        body: dict,
        model_config: ModelConfig,
    ) -> tuple[Union[Response, StreamingResponse], Optional[dict]]:
        # Always POST to /v1/chat/completions regardless of incoming path
        url = f"{self.base_url}/v1/chat/completions"

        headers = self._build_headers(request)

        # Remember the original model name for response conversion
        original_model = body.get("model", "")

        # Convert Anthropic request body → OpenAI format
        openai_body = convert_anthropic_to_openai_request(body)

        # OpenAI's own API requires max_completion_tokens; other compatible backends use max_tokens
        if "openai.com" in self.base_url and "max_tokens" in openai_body:
            openai_body["max_completion_tokens"] = openai_body.pop("max_tokens")

        debug_level = get_debug_level()
        if debug_level == 1:
            logger.debug("OPENAI REQUEST: %s", format_body_short(openai_body))
        elif debug_level >= 2:
            logger.debug(
                "OPENAI REQUEST:\n%s",
                json.dumps(openai_body, indent=2, ensure_ascii=False),
            )

        is_streaming = openai_body.get("stream", False)

        if is_streaming:
            response = await self._handle_streaming(url, headers, openai_body, original_model, debug_level)
        else:
            response = await self._handle_buffered(url, headers, openai_body, original_model, debug_level)

        return (response, None)

    async def _handle_buffered(
        self,
        url: str,
        headers: dict,
        openai_body: dict,
        original_model: str,
        debug_level: int,
    ) -> Union[Response, JSONResponse]:
        resp = await self.client.request(
            method="POST",
            url=url,
            headers=headers,
            content=json.dumps(openai_body).encode(),
        )

        if resp.status_code != 200:
            # Try to extract a meaningful message from the OpenAI error response
            message = f"OpenAI API error (status {resp.status_code})"
            try:
                err_body = resp.json()
                if isinstance(err_body, dict):
                    err_obj = err_body.get("error", {})
                    if isinstance(err_obj, dict) and err_obj.get("message"):
                        message = err_obj["message"]
            except Exception:
                pass
            logger.warning("OPENAI RESPONSE error: status=%d message=%r", resp.status_code, message)
            return _error_response(resp.status_code, message)

        try:
            openai_resp = resp.json()
        except Exception as exc:
            logger.error("OPENAI RESPONSE: failed to parse JSON: %s", exc)
            return _error_response(500, "Invalid JSON response from OpenAI API")

        if debug_level == 1:
            logger.debug("OPENAI RESPONSE: %s", format_body_short(openai_resp))
        elif debug_level >= 2:
            logger.debug(
                "OPENAI RESPONSE:\n%s",
                json.dumps(openai_resp, indent=2, ensure_ascii=False),
            )

        anthropic_resp = convert_openai_to_anthropic_response(openai_resp, original_model)

        if debug_level >= 2:
            logger.debug(
                "OPENAI→ANTHROPIC RESPONSE:\n%s",
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
        original_model: str,
        debug_level: int,
    ) -> Union[StreamingResponse, JSONResponse]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Open the stream now so we can inspect the status code before returning.
        # The context manager is transferred into the generator below.
        stream_ctx = self.client.stream(
            method="POST",
            url=url,
            headers=headers,
            content=json.dumps(openai_body).encode(),
        )
        resp = await stream_ctx.__aenter__()

        if resp.status_code != 200:
            try:
                raw = await resp.aread()
                err_body = json.loads(raw)
                err_obj = err_body.get("error", {})
                message = err_obj.get("message", f"OpenAI API error (status {resp.status_code})")
            except Exception:
                message = f"OpenAI API error (status {resp.status_code})"
            finally:
                await stream_ctx.__aexit__(None, None, None)
            logger.warning("OPENAI STREAM error: status=%d message=%r", resp.status_code, message)
            return _error_response(resp.status_code, message)

        async def _stream_generator():
            accumulated_text: list[str] = []
            try:
                async for chunk in convert_openai_stream_to_anthropic(resp, original_model, msg_id):
                    if debug_level >= 2:
                        try:
                            for line in chunk.decode().splitlines():
                                if line.startswith("data:"):
                                    data = json.loads(line[5:])
                                    if data.get("type") == "content_block_delta":
                                        delta = data.get("delta", {})
                                        accumulated_text.append(
                                            delta.get("text", "") or delta.get("thinking", "")
                                        )
                        except Exception:
                            pass
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)

            if debug_level >= 1 and accumulated_text:
                text = "".join(accumulated_text)
                if debug_level == 1:
                    preview = text[:80].replace("\n", " ")
                    logger.debug("OPENAI RESPONSE [stream]: %r", preview)
                else:
                    logger.debug("OPENAI RESPONSE [stream]:\n%s", text)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
        )
