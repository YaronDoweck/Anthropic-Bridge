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

from config import ProxyConfig, load_config
from logging_config import setup_logging
from request_logger import RequestLogger
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
    env_output_file = os.getenv("OUTPUT_FILE")
    if env_output_file is not None:
        config.server.output_file = env_output_file


load_dotenv()
config = load_config()
_apply_env_overrides(config)
setup_logging(log_level=config.server.log_level, debug_level=config.server.debug_level)

router_instance: Optional[ProxyRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_instance
    ollama_proc = run_startup_checks(config)
    try:
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            output_file = config.server.output_file or None
            req_logger = RequestLogger(output_file) if output_file else None
            router_instance = ProxyRouter(config, client, request_logger=req_logger)
            yield
    finally:
        router_instance = None
        if ollama_proc is not None:
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()


app = FastAPI(lifespan=lifespan)


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
