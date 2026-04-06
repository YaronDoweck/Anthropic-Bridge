from __future__ import annotations

import os
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import logging

from config import ProxyConfig, load_config
from logging_config import setup_logging
from request_logger import RequestLogger, RawLogger
from router import ProxyRouter
from startup_checks import run_startup_checks


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _apply_env_overrides(config: ProxyConfig) -> None:
    env_host = os.getenv("HOST")
    if env_host is not None:
        config.server.host = env_host
    env_port = os.getenv("PORT")
    if env_port is not None:
        try:
            config.server.port = int(env_port)
        except ValueError:
            raise RuntimeError(f"Invalid PORT env var: {env_port!r} is not an integer")
    env_log_level = os.getenv("LOG_LEVEL")
    if env_log_level is not None:
        upper = env_log_level.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise RuntimeError(f"Invalid LOG_LEVEL env var: {env_log_level!r}. Must be one of {sorted(_VALID_LOG_LEVELS)}")
        config.server.log_level = upper
    env_debug_level = os.getenv("DEBUG_LEVEL")
    if env_debug_level is not None:
        try:
            config.server.debug_level = max(0, min(3, int(env_debug_level)))
        except ValueError:
            raise RuntimeError(f"Invalid DEBUG_LEVEL env var: {env_debug_level!r} is not an integer")
    env_output_dir = os.getenv("OUTPUT_DIR")
    if env_output_dir is not None:
        config.server.output_dir = env_output_dir
    env_raw_log_dir = os.getenv("RAW_LOG_DIR")
    if env_raw_log_dir is not None:
        config.server.raw_log_dir = env_raw_log_dir
    env_override_model = os.getenv("OVERRIDE_MODEL")
    if env_override_model is not None:
        config.server.override_model = env_override_model.strip()


load_dotenv()
config = load_config()
_apply_env_overrides(config)
if config.server.override_model and config.server.override_model not in config.models:
    available = ", ".join(sorted(config.models.keys()))
    raise RuntimeError(
        f"override_model '{config.server.override_model}' is not defined in [models]. "
        f"Available models: {available}"
    )
setup_logging(log_level=config.server.log_level, debug_level=config.server.debug_level)
_log = logging.getLogger(__name__)
if config.server.override_model:
    _log.warning("OVERRIDE MODE ACTIVE: all requests will use model '%s'", config.server.override_model)

router_instance: Optional[ProxyRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance
    ollama_proc = run_startup_checks(config)
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            output_dir = config.server.output_dir or None
            req_logger = RequestLogger(output_dir) if output_dir else None
            raw_log_dir = config.server.raw_log_dir or config.server.output_dir or None
            raw_logger = RawLogger(raw_log_dir) if raw_log_dir else None
            app.state.raw_logger = raw_logger
            router_instance = ProxyRouter(config, client, request_logger=req_logger, raw_logger=raw_logger)
            yield
    finally:
        router_instance = None
        app.state.raw_logger = None
        if ollama_proc is not None:
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()


app = FastAPI(lifespan=lifespan)

import datetime as _dt

@app.middleware("http")
async def _raw_logging_middleware(request: Request, call_next):
    raw_logger = getattr(app.state, "raw_logger", None)
    if raw_logger is None:
        return await call_next(request)

    raw_body = await request.body()
    timestamp = _dt.datetime.now().isoformat()
    body_str = raw_body.decode("utf-8", errors="replace")
    raw_logger.log(timestamp, request.method, request.url.path, body_str)

    response = await call_next(request)

    # Wrap the body iterator to capture chunks while still streaming to the client
    original_iterator = response.body_iterator
    method = request.method
    path = str(request.url.path)
    status_code = response.status_code

    async def capturing_iterator():
        chunks = []
        try:
            async for chunk in original_iterator:
                chunks.append(chunk)
                yield chunk
        finally:
            full_body = b"".join(chunks)
            raw_logger.log_response(timestamp, method, path, status_code, full_body.decode("utf-8", errors="replace"))

    response.body_iterator = capturing_iterator()
    return response


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str) -> Response:
    if router_instance is None:
        return JSONResponse(
            status_code=503,
            content={"type": "error", "error": {"type": "api_error", "message": "Proxy is starting up, please retry"}},
        )
    return await router_instance.route(request, path)


def main():
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
