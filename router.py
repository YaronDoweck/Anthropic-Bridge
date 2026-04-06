"""
ProxyRouter: parses the incoming request body, looks up the model in config,
and dispatches to the appropriate handler.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Optional, Union

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import ProxyConfig
from handlers.anthropic import AnthropicHandler
from handlers.ollama import OllamaHandler
from request_logger import RequestLogger, RawLogger, _extract_session_id

logger = logging.getLogger(__name__)


def _error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _strip_date_postfix(model: str) -> str | None:
    """Return model name without trailing -YYYYMMDD postfix, or None if not present."""
    m = re.match(r'^(.+)-\d{8}$', model)
    return m.group(1) if m else None


class ProxyRouter:
    def __init__(
        self,
        config: ProxyConfig,
        http_client: httpx.AsyncClient,
        request_logger: Optional[RequestLogger] = None,
        raw_logger: Optional[RawLogger] = None,
    ) -> None:
        self.config = config
        self.request_logger = request_logger
        self.raw_logger = raw_logger
        self.handlers = {
            "anthropic": AnthropicHandler(
                base_url=config.anthropic_url,
                http_client=http_client,
                auth_token=os.getenv("ANTHROPIC_AUTH_TOKEN"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            ),
            "ollama": OllamaHandler(config.ollama_url, http_client),
        }

    async def route(
        self, request: Request, path: str
    ) -> Union[Response, StreamingResponse, JSONResponse]:
        # 1. Parse body JSON
        raw_body = await request.body()
        if not raw_body:
            return _error_response(400, "invalid_request_error", "Request body is empty")

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON body: %s", exc)
            return _error_response(400, "invalid_request_error", f"Invalid JSON body: {exc}")

        # 2. Extract model field
        model = body.get("model")
        if not model:
            return _error_response(
                400, "invalid_request_error", "Missing required field: 'model'"
            )

        # 2b. Apply model override if configured
        override = self.config.server.override_model
        if override:
            original_model = model
            model = override
            body["model"] = override
            logger.debug("OVERRIDE: '%s' -> '%s'", original_model, override)

        # 3. Look up ModelConfig (with date-postfix fallback)
        model_config = self.config.models.get(model)
        if model_config is None:
            stripped = _strip_date_postfix(model)
            if stripped is not None:
                model_config = self.config.models.get(stripped)
                if model_config is not None:
                    logger.info("Model '%s' matched config key '%s' (date-postfix stripped)", model, stripped)
        if model_config is None:
            return _error_response(
                400,
                "invalid_request_error",
                f"Model '{model}' is not configured. "
                f"Available models: {', '.join(sorted(self.config.models))}",
            )

        # 4. Dispatch to handler
        endpoint_type = model_config.endpoint
        handler = self.handlers.get(endpoint_type)
        if handler is None:
            logger.error("No handler registered for endpoint type '%s'", endpoint_type)
            return _error_response(
                500, "api_error", f"No handler registered for endpoint '{endpoint_type}'"
            )

        logger.info("REQUEST → %s [%s]", model, endpoint_type)
        timestamp = datetime.datetime.now().isoformat()

        try:
            response, sent_body = await handler.handle(request, path, body, model_config)
            log_body = sent_body if sent_body is not None else body
        except httpx.TimeoutException as exc:
            logger.error("Upstream timeout for model '%s': %s", model, exc)
            return _error_response(
                504, "api_error", "Upstream request timed out"
            )
        except httpx.ConnectError as exc:
            logger.error("Upstream connection error for model '%s': %s", model, exc)
            return _error_response(
                502, "api_error", "Could not connect to upstream endpoint"
            )
        except httpx.RequestError as exc:
            logger.error("Upstream request error for model '%s': %s", model, exc)
            return _error_response(
                502, "api_error", f"Upstream request failed: {type(exc).__name__}"
            )

        if self.request_logger is not None:
            session_id = _extract_session_id(body)
            if isinstance(response, StreamingResponse):
                orig_headers = dict(response.headers)
                response = StreamingResponse(
                    self.request_logger.wrap_streaming_generator(
                        log_body, model, endpoint_type, response.body_iterator, timestamp,
                        session_id=session_id
                    ),
                    status_code=response.status_code,
                    media_type=response.media_type,
                    headers=orig_headers,
                )
            else:
                self.request_logger.log_buffered_response(
                    log_body, model, endpoint_type,
                    getattr(response, "body", b""),
                    timestamp,
                    status_code=response.status_code,
                    session_id=session_id,
                )

        if isinstance(response, StreamingResponse):
            logger.info("RESPONSE ← %s [%s] streaming", model, endpoint_type)
        else:
            logger.info("RESPONSE ← %s [%s] status=%d", model, endpoint_type, response.status_code)

        return response
