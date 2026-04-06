"""
Anthropic pass-through handler.

Forwards all requests as-is to the Anthropic API. No format conversion is
performed. If the incoming request carries no auth headers, the handler
injects fallback credentials from the proxy's own environment.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from config import ModelConfig
from handlers.base import BaseHandler
from logging_config import format_body_short, get_debug_level

logger = logging.getLogger(__name__)

# Headers that must not be forwarded to the upstream
_EXCLUDED_REQUEST_HEADERS = frozenset({"host", "content-length", "transfer-encoding"})
# Headers that must not be copied from the upstream response
_EXCLUDED_RESPONSE_HEADERS = frozenset({
    "transfer-encoding",
    "content-encoding",
    "content-length",
    "set-cookie",
    "set-cookie2",
    "strict-transport-security",
    "alt-svc",
})


class AnthropicHandler(BaseHandler):
    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, http_client)
        self.api_key = api_key
        self.auth_token = auth_token

    async def handle(
        self,
        request: Request,
        path: str,
        body: dict,
        model_config: ModelConfig,
    ) -> Union[Response, StreamingResponse]:
        url = f"{self.base_url}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _EXCLUDED_REQUEST_HEADERS
        }

        # Inject fallback auth only if the incoming request has no auth headers
        has_auth = "x-api-key" in headers or "authorization" in headers
        if not has_auth:
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.api_key:
                headers["x-api-key"] = self.api_key

        debug_level = get_debug_level()
        if debug_level == 2:
            logger.debug("ANTHROPIC REQUEST: %s", format_body_short(body))
        elif debug_level >= 3:
            logger.debug(
                "ANTHROPIC REQUEST:\n%s",
                json.dumps(body, indent=2, ensure_ascii=False),
            )

        is_streaming = body.get("stream", False)

        if is_streaming:
            return await self._handle_streaming(request.method, url, headers, body, debug_level)
        else:
            return await self._handle_buffered(request.method, url, headers, body, debug_level)

    async def _handle_buffered(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        debug_level: int,
    ) -> Response:
        resp = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            content=json.dumps(body).encode(),
        )
        response_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _EXCLUDED_RESPONSE_HEADERS
        }

        if debug_level >= 2:
            try:
                parsed = json.loads(resp.content)
                if debug_level == 2:
                    logger.debug("ANTHROPIC RESPONSE: %s", format_body_short(parsed))
                else:
                    logger.debug(
                        "ANTHROPIC RESPONSE:\n%s",
                        json.dumps(parsed, indent=2, ensure_ascii=False),
                    )
            except Exception:
                logger.debug("ANTHROPIC RESPONSE [non-JSON body]")

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers,
        )

    async def _handle_streaming(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        debug_level: int,
    ) -> StreamingResponse:
        async def _stream_generator():
            accumulated_text = []
            async with self.client.stream(
                method=method,
                url=url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    if debug_level >= 2:
                        try:
                            for line in chunk.decode().splitlines():
                                if line.startswith("data:"):
                                    data = json.loads(line[5:])
                                    if data.get("type") == "content_block_delta":
                                        accumulated_text.append(data.get("delta", {}).get("text", ""))
                        except Exception:
                            pass
                    yield chunk
            if debug_level >= 2 and accumulated_text:
                text = "".join(accumulated_text)
                if debug_level == 2:
                    preview = text[:80].replace("\n", " ")
                    logger.debug("ANTHROPIC RESPONSE [stream]: %r", preview)
                else:
                    logger.debug("ANTHROPIC RESPONSE [stream]:\n%s", text)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
        )
