"""
Unit tests for router.py using FastAPI's TestClient with mocked HTTP upstreams.
These tests require no API keys and no running services.
"""

from __future__ import annotations

import json
import sys
import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "proxy.config"))


# ---------------------------------------------------------------------------
# Helper: build a TestClient whose lifespan skips real startup checks and
# injects a mocked httpx.AsyncClient that never actually dials any backend.
# ---------------------------------------------------------------------------

@contextmanager
def _make_client(mock_http_response=None):
    """
    Context manager that yields a configured TestClient for the proxy app.

    - run_startup_checks is patched to return None (no Ollama needed).
    - server.httpx.AsyncClient is patched so the lifespan injects a mock
      HTTP client that never makes real network calls.
    - config is loaded from the real proxy.config (using its absolute path).
    """
    from config import load_config
    real_config = load_config(CONFIG_PATH)

    if mock_http_response is None:
        mock_http_response = _openai_ok_response()

    mock_http_client = _build_mock_http_client(mock_http_response)

    import server as server_module

    with (
        patch.object(server_module, "config", real_config),
        patch("server.run_startup_checks", return_value=None),
        patch("server.httpx.AsyncClient", return_value=mock_http_client),
    ):
        with TestClient(server_module.app, raise_server_exceptions=False) as client:
            yield client


def _openai_ok_response(content: str = "Hello!") -> MagicMock:
    """Build a fake httpx.Response that looks like a valid OpenAI chat completion."""
    resp = MagicMock()
    resp.status_code = 200
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp.content = json.dumps(payload).encode()
    resp.json.return_value = payload
    resp.headers = {"content-type": "application/json"}
    return resp


def _build_mock_http_client(response: MagicMock) -> MagicMock:
    """
    Build a mock httpx.AsyncClient whose .request() coroutine returns *response*
    and whose async context manager (__aenter__/__aexit__) returns itself.
    """
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=response)

    # Support `async with client.stream(...)` used by streaming handlers
    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=response)
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=stream_ctx)

    # AsyncClient itself is used as an async context manager in the lifespan
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMissingModelField:
    def test_missing_model_field_returns_400(self):
        """A request body without a 'model' key must yield 400 invalid_request_error."""
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "model" in body["error"]["message"].lower()


class TestUnknownModel:
    def test_unknown_model_returns_400(self):
        """A model name not present in config must yield 400 invalid_request_error."""
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "this-model-does-not-exist",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "this-model-does-not-exist" in body["error"]["message"]


class TestInvalidJson:
    def test_invalid_json_returns_400(self):
        """A non-JSON body must yield 400 invalid_request_error."""
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                content=b"this is not json at all!!!",
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"

    def test_empty_body_returns_400(self):
        """An empty request body must yield 400."""
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                content=b"",
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"

    def test_partial_json_returns_400(self):
        """Truncated JSON must yield 400."""
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                content=b'{"model": "llama3", "messages": [',
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"


class TestValidOllamaRequest:
    def test_known_ollama_model_routed_successfully(self):
        """
        A valid request for a known Ollama model should be forwarded and the
        response converted back to Anthropic format (type='message').
        """
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "qwen3.5:9b",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 100,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert isinstance(body["content"], list)
        assert body["content"][0]["type"] == "text"

    def test_known_ollama_model_stop_sequences_forwarded(self):
        """
        stop_sequences in the Anthropic request should be converted to 'stop'
        before being sent to Ollama (we verify the router doesn't reject the field).
        """
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "qwen3.5:9b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stop_sequences": ["END"],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200


class TestErrorResponseShape:
    def test_error_body_schema(self):
        """
        Every error response must follow the Anthropic error schema:
        { "type": "error", "error": { "type": str, "message": str } }
        """
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={"model": "nonexistent-model", "messages": []},
            )
        body = resp.json()
        assert "type" in body
        assert body["type"] == "error"
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert isinstance(body["error"]["message"], str)

    def test_error_lists_available_models(self):
        """
        When an unknown model is requested the error message should hint at
        the configured models.
        """
        from config import load_config
        real_config = load_config(CONFIG_PATH)
        available = list(real_config.models.keys())

        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={"model": "no-such-model", "messages": []},
            )
        body = resp.json()
        message = body["error"]["message"]
        assert any(m in message for m in available), (
            f"Expected one of {available} in error message, got: {message!r}"
        )


# ---------------------------------------------------------------------------
# Helpers for override-model tests
# ---------------------------------------------------------------------------

import copy


@contextmanager
def _make_client_with_override(override_model: str, captured_requests: list | None = None):
    """
    Like _make_client but also sets config.server.override_model to the given
    value and optionally collects the kwargs passed to mock_client.request so
    tests can inspect what body was forwarded upstream.
    """
    from config import load_config
    real_config = load_config(CONFIG_PATH)

    # Deep-copy so we don't mutate the shared config object
    patched_config = copy.deepcopy(real_config)
    patched_config.server.override_model = override_model

    mock_http_response = _openai_ok_response()
    mock_http_client = _build_mock_http_client(mock_http_response)

    # Wrap request so callers can inspect what was forwarded
    if captured_requests is not None:
        original_request = mock_http_client.request

        async def _capturing_request(*args, **kwargs):
            captured_requests.append(kwargs)
            return await original_request(*args, **kwargs)

        mock_http_client.request = _capturing_request

    import server as server_module

    with (
        patch.object(server_module, "config", patched_config),
        patch("server.run_startup_checks", return_value=None),
        patch("server.httpx.AsyncClient", return_value=mock_http_client),
    ):
        with TestClient(server_module.app, raise_server_exceptions=False) as client:
            yield client


# ---------------------------------------------------------------------------
# Override-model tests
# ---------------------------------------------------------------------------

class TestOverrideModel:
    def test_override_replaces_unknown_model(self):
        """
        When override_model is set, a request using an unknown model name must
        succeed (200) because the router replaces it with the override before
        looking up the config.
        """
        with _make_client_with_override("claude-sonnet-4-6") as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "completely-unknown-model-xyz",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200

    def test_override_replaces_model_in_body(self):
        """
        When override_model is set, the body forwarded upstream must contain
        the override model name, not the original model sent by the client.
        """
        captured: list = []
        with _make_client_with_override("claude-sonnet-4-6", captured_requests=captured) as client:
            client.post(
                "/v1/messages",
                json={
                    "model": "some-unknown-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 50,
                },
            )

        assert len(captured) == 1, "Expected exactly one upstream request"
        forwarded_kwargs = captured[0]
        # The body is passed as the 'content' keyword arg (bytes) in httpx
        forwarded_content = forwarded_kwargs.get("content", b"")
        if isinstance(forwarded_content, bytes):
            forwarded_body = json.loads(forwarded_content)
        else:
            forwarded_body = forwarded_content
        assert forwarded_body["model"] == "claude-sonnet-4-6", (
            f"Expected upstream body model='claude-sonnet-4-6', got {forwarded_body.get('model')!r}"
        )

    def test_override_disabled_when_empty(self):
        """
        When override_model is '' (empty), requests with a known model name
        should route normally and return 200.
        """
        # _make_client uses the real config which has override_model="" by default
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 50,
                },
            )
        assert resp.status_code == 200

    def test_no_override_unknown_model_returns_400(self):
        """
        With no override configured, an unknown model name must yield 400.
        """
        with _make_client() as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "definitely-not-a-real-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
