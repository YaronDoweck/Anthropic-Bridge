"""
Unit tests for OllamaHandler failover logic.

Tests cover:
  a. No failover configured — ConnectError propagates
  b. Failover on ConnectError (buffered) — retries on fallback, _using_fallback=True
  c. Failover on ConnectTimeout (streaming) — retries on fallback, _using_fallback=True
  d. Mid-stream ReadError does NOT trigger fallback
  e. Sticky: after failover, next request goes directly to fallback
  f. Health checker: marks primary recovered when probe succeeds
  g. Fallback also down — exception propagates
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handlers.ollama import OllamaHandler
from config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRIMARY_URL = "http://primary:11434"
FALLBACK_URL = "http://fallback:11434"


def _make_model_config(**kwargs) -> ModelConfig:
    defaults = dict(
        endpoint="ollama",
        claude_system_instructions="passthrough",
        omit_claude_main_description=False,
        ollama_format="anthropic",
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _make_request(stream: bool = False) -> MagicMock:
    """Minimal fake FastAPI Request."""
    req = MagicMock()
    req.headers = {}
    req.url = MagicMock()
    req.url.query = ""
    return req


def _buffered_body(stream: bool = False) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "max_tokens": 10,
    }


def _make_httpx_response(status: int = 200, content: bytes = b'{"type":"message"}') -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.headers = {"content-type": "application/json"}

    async def _aread():
        return content

    async def _aclose():
        pass

    async def _aiter_bytes():
        yield content

    resp.aread = _aread
    resp.aclose = _aclose
    resp.aiter_bytes = _aiter_bytes
    return resp


def _make_stream_response_with_read_error() -> MagicMock:
    """Returns a response that yields one chunk then raises ReadError."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}

    async def _aread():
        return b""

    async def _aclose():
        pass

    async def _aiter_bytes():
        yield b"data: partial\n\n"
        raise httpx.ReadError("mid-stream disconnection")

    resp.aread = _aread
    resp.aclose = _aclose
    resp.aiter_bytes = _aiter_bytes
    return resp


# ---------------------------------------------------------------------------
# a. No failover configured — ConnectError propagates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_failover_connect_error_propagates():
    """Without a fallback URL, ConnectError must propagate to the caller."""
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=None)
    model_cfg = _make_model_config()

    with pytest.raises(httpx.ConnectError):
        await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)

    assert handler._using_fallback is False


# ---------------------------------------------------------------------------
# b. Failover on ConnectError (buffered) — primary raises, handler retries on fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failover_on_connect_error_buffered():
    """
    When the primary URL raises ConnectError, the handler retries the request
    on the fallback URL and returns the successful response.
    _using_fallback must be True after.
    """
    mock_client = MagicMock()
    success_resp = _make_httpx_response()

    call_count = 0

    async def _request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        url = kwargs.get("url", args[1] if len(args) > 1 else "")
        if PRIMARY_URL in str(url):
            raise httpx.ConnectError("primary refused")
        return success_resp

    mock_client.request = _request

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()

    response, _ = await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)

    assert handler._using_fallback is True
    assert response.status_code == 200
    assert call_count == 2  # once on primary (fail), once on fallback (success)


# ---------------------------------------------------------------------------
# c. Failover on ConnectTimeout (streaming) — primary raises, handler retries on fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failover_on_connect_timeout_streaming():
    """
    During a streaming request, ConnectTimeout on the primary causes the handler
    to retry on the fallback. _using_fallback must be True after.
    """
    mock_client = MagicMock()
    success_stream = _make_httpx_response()

    # build_request must return a real-ish request object
    def _build_request(*args, **kwargs):
        req = MagicMock()
        req.url = MagicMock()
        req.url.__str__ = lambda s: kwargs.get("url", args[1] if len(args) > 1 else "")
        return req

    call_count = 0

    async def _send(req, **kwargs):
        nonlocal call_count
        call_count += 1
        url_str = str(req.url)
        if PRIMARY_URL in url_str:
            raise httpx.ConnectTimeout("primary timed out")
        return success_stream

    mock_client.build_request = _build_request
    mock_client.send = _send

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()
    body = _buffered_body(stream=True)

    response, _ = await handler.handle(_make_request(), "v1/messages", body, model_cfg)

    assert handler._using_fallback is True
    assert call_count == 2


# ---------------------------------------------------------------------------
# d. Mid-stream ReadError does NOT trigger fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_stream_read_error_no_fallback():
    """
    A ReadError raised inside aiter_bytes (after the connection opened successfully)
    must NOT trigger the fallback. _using_fallback stays False and an SSE error
    event is yielded.
    """
    mock_client = MagicMock()
    stream_resp = _make_stream_response_with_read_error()

    def _build_request(*args, **kwargs):
        req = MagicMock()
        req.url = MagicMock()
        url_val = kwargs.get("url", PRIMARY_URL)
        req.url.__str__ = lambda s: str(url_val)
        return req

    async def _send(req, **kwargs):
        return stream_resp

    mock_client.build_request = _build_request
    mock_client.send = _send

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()
    body = _buffered_body(stream=True)

    response, _ = await handler.handle(_make_request(), "v1/messages", body, model_cfg)

    # Consume the streaming response
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    combined = b"".join(chunks)

    assert handler._using_fallback is False, "_using_fallback must remain False for mid-stream errors"
    assert b"error" in combined.lower(), "Expected an SSE error event in the stream"


# ---------------------------------------------------------------------------
# e. Sticky: after failover, subsequent requests go directly to fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sticky_after_failover():
    """
    After a failover, subsequent requests must go directly to the fallback URL
    without attempting the primary first.
    """
    mock_client = MagicMock()
    success_resp = _make_httpx_response()
    urls_called: list[str] = []

    async def _request(*args, **kwargs):
        url = str(kwargs.get("url", ""))
        urls_called.append(url)
        if PRIMARY_URL in url and not handler._using_fallback:
            raise httpx.ConnectError("primary refused")
        return success_resp

    mock_client.request = _request

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()

    # First request: triggers failover
    urls_called.clear()
    await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)
    assert handler._using_fallback is True

    # Second request: should go directly to fallback (only one URL attempted)
    urls_called.clear()
    await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)

    assert len(urls_called) == 1, f"Expected only 1 request after failover, got {len(urls_called)}: {urls_called}"
    assert FALLBACK_URL in urls_called[0], f"Expected fallback URL, got {urls_called[0]!r}"


# ---------------------------------------------------------------------------
# f. Health checker: marks primary recovered
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_checker_marks_primary_recovered():
    """
    When _using_fallback is True and _probe_primary returns True (200 response),
    one health-check iteration must set _using_fallback back to False.
    """
    mock_client = MagicMock()
    probe_resp = MagicMock()
    probe_resp.status_code = 200
    mock_client.get = AsyncMock(return_value=probe_resp)

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    handler._using_fallback = True

    # Directly call _probe_primary and then _mark_primary to simulate one loop iteration
    recovered = await handler._probe_primary()
    assert recovered is True

    if recovered:
        handler._mark_primary()

    assert handler._using_fallback is False


@pytest.mark.asyncio
async def test_health_checker_does_not_recover_when_primary_down():
    """
    When _probe_primary returns False (connection error), _using_fallback stays True.
    """
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("still down"))

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    handler._using_fallback = True

    recovered = await handler._probe_primary()
    assert recovered is False
    assert handler._using_fallback is True


# ---------------------------------------------------------------------------
# g. Fallback also down — exception propagates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_also_down_propagates():
    """
    When both primary and fallback raise ConnectError, the exception must
    propagate and not be swallowed.
    """
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=httpx.ConnectError("everything is down"))

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()

    with pytest.raises(httpx.ConnectError):
        await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)


# ---------------------------------------------------------------------------
# Additional: _using_fallback stays False when no failover occurs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_failover_when_primary_healthy():
    """When the primary succeeds, _using_fallback must remain False."""
    mock_client = MagicMock()
    success_resp = _make_httpx_response()
    mock_client.request = AsyncMock(return_value=success_resp)

    handler = OllamaHandler(PRIMARY_URL, mock_client, fallback_url=FALLBACK_URL)
    model_cfg = _make_model_config()

    await handler.handle(_make_request(), "v1/messages", _buffered_body(), model_cfg)

    assert handler._using_fallback is False
