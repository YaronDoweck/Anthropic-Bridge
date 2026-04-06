"""
Live integration tests against the Anthropic API.

These tests only run when ANTHROPIC_API_KEY is set in the environment.
They start a real uvicorn server on a random port via the `running_server`
session-scoped fixture defined in conftest.py.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN")

requires_api_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY and not ANTHROPIC_AUTH_TOKEN,
    reason="Neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set",
)

# Use the cheapest Haiku model for live tests
_CHEAP_MODEL = "claude-haiku-4-5"

# A model name that is definitely not in proxy.config
_INVALID_MODEL = "claude-does-not-exist-99999"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if ANTHROPIC_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {ANTHROPIC_AUTH_TOKEN}"
    elif ANTHROPIC_API_KEY:
        headers["x-api-key"] = ANTHROPIC_API_KEY
    return headers


def _parse_sse_events(raw: bytes) -> list[dict]:
    """
    Parse a raw SSE byte stream into a list of dicts with keys 'event' and 'data'.
    """
    events = []
    current_event: dict[str, str] = {}
    for raw_line in raw.decode().splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = line[len("data:"):].strip()
        elif line == "" and current_event:
            events.append(current_event)
            current_event = {}
    if current_event:
        events.append(current_event)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_api_key
def test_non_streaming_request(running_server):
    """
    POST /v1/messages with a real Anthropic model (non-streaming) must return
    a valid Anthropic message response.
    """
    url = f"{running_server}/v1/messages"
    payload = {
        "model": _CHEAP_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say the word 'hello' and nothing else."}],
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=_auth_headers())

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()

    # Anthropic message schema checks
    assert body.get("type") == "message", f"Expected type='message', got: {body.get('type')!r}"
    assert body.get("role") == "assistant", f"Expected role='assistant', got: {body.get('role')!r}"

    content = body.get("content")
    assert isinstance(content, list), f"Expected content to be a list, got: {type(content)}"
    assert len(content) >= 1, "Expected at least one content block"
    assert content[0].get("type") == "text", f"Expected first block type='text', got: {content[0].get('type')!r}"
    assert isinstance(content[0].get("text"), str), "Expected text to be a string"

    assert "usage" in body, "Expected 'usage' field in response"
    assert "input_tokens" in body["usage"]
    assert "output_tokens" in body["usage"]


@requires_api_key
def test_streaming_request(running_server):
    """
    POST /v1/messages with stream=True must produce SSE events in the correct
    Anthropic order: message_start, content_block_start, >= 1 content_block_delta,
    content_block_stop, message_delta, message_stop.
    """
    url = f"{running_server}/v1/messages"
    payload = {
        "model": _CHEAP_MODEL,
        "max_tokens": 32,
        "stream": True,
        "messages": [{"role": "user", "content": "Say the word 'hi'."}],
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=_auth_headers())

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    events = _parse_sse_events(resp.content)
    event_types = [e.get("event") for e in events if "event" in e]

    # Required event types that must appear in order
    required_in_order = [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    # Check all required events are present
    for et in required_in_order:
        assert et in event_types, f"Missing expected SSE event: {et!r}. Got: {event_types}"

    # Check ordering: each required event appears after the previous
    indices = {et: event_types.index(et) for et in required_in_order}
    for i in range(len(required_in_order) - 1):
        a = required_in_order[i]
        b = required_in_order[i + 1]
        assert indices[a] < indices[b], (
            f"Expected {a!r} before {b!r}, but {a!r} is at index {indices[a]} "
            f"and {b!r} is at index {indices[b]}"
        )

    # At least one content_block_delta must exist
    assert "content_block_delta" in event_types, (
        "Expected at least one content_block_delta event"
    )

    # content_block_delta must appear between content_block_start and content_block_stop
    first_delta_idx = event_types.index("content_block_delta")
    assert indices["content_block_start"] < first_delta_idx < indices["content_block_stop"], (
        "content_block_delta must appear between content_block_start and content_block_stop"
    )

    # Validate message_start data payload
    message_start_event = next(e for e in events if e.get("event") == "message_start")
    ms_data = json.loads(message_start_event["data"])
    assert ms_data.get("type") == "message_start"
    assert "message" in ms_data
    assert ms_data["message"].get("role") == "assistant"

    # Validate message_stop data payload
    message_stop_event = next(e for e in events if e.get("event") == "message_stop")
    ms_stop_data = json.loads(message_stop_event["data"])
    assert ms_stop_data.get("type") == "message_stop"


@requires_api_key
def test_error_response_format(running_server):
    """
    Sending a request with an invalid/unconfigured model name must return an
    error response that conforms to the Anthropic error schema.
    """
    url = f"{running_server}/v1/messages"
    payload = {
        "model": _INVALID_MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=_auth_headers())

    # Router should return 400 for unknown model
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

    body = resp.json()

    # Anthropic error schema
    assert body.get("type") == "error", f"Expected type='error', got: {body.get('type')!r}"
    assert "error" in body, "Expected 'error' key in response body"
    assert "type" in body["error"], "Expected 'type' key in error object"
    assert "message" in body["error"], "Expected 'message' key in error object"
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["type"] == "invalid_request_error"
