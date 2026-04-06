"""
Unit tests for converters/anthropic_openai.py and the normalize model name helper
in startup_checks.py.  These tests require no API keys and no running services.
"""

from __future__ import annotations

import sys
import os

# Ensure the proxy source root is importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from converters.anthropic_openai import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic_response,
)
from startup_checks import _normalize_model_name


# ---------------------------------------------------------------------------
# Request conversion tests
# ---------------------------------------------------------------------------

class TestConvertAnthropicToOpenaiRequest:
    def test_request_system_field_becomes_first_message(self):
        """The top-level 'system' field must be prepended as a system message."""
        body = {
            "model": "llama3",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = convert_anthropic_to_openai_request(body)
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_request_no_system_field_no_prepend(self):
        """Without a 'system' key no system message should be injected."""
        body = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = convert_anthropic_to_openai_request(body)
        assert result["messages"][0]["role"] == "user"
        assert len(result["messages"]) == 1

    def test_request_content_list_joined(self):
        """Content given as a list of text blocks must be joined into a single string."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " world"},
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        assert result["messages"][0]["content"] == "Hello world"

    def test_request_content_list_skips_non_text_blocks(self):
        """Non-text blocks (e.g. image) should be silently skipped, text blocks kept."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        assert result["messages"][0]["content"] == "describe this"

    def test_request_stop_sequences_mapped(self):
        """'stop_sequences' in Anthropic format must become 'stop' in OpenAI format."""
        body = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END", "STOP"],
        }
        result = convert_anthropic_to_openai_request(body)
        assert "stop" in result
        assert result["stop"] == ["END", "STOP"]
        assert "stop_sequences" not in result

    def test_request_optional_fields_passed_through(self):
        """temperature, top_p, max_tokens and stream must be copied verbatim."""
        body = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "stream": True,
        }
        result = convert_anthropic_to_openai_request(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 512
        assert result["stream"] is True

    def test_request_model_passed_through(self):
        """The 'model' field must be present in the converted output."""
        body = {
            "model": "llama3:8b",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = convert_anthropic_to_openai_request(body)
        assert result["model"] == "llama3:8b"

    def test_request_empty_messages(self):
        """An empty messages list is legal and should produce an empty messages list."""
        body = {"model": "llama3", "messages": []}
        result = convert_anthropic_to_openai_request(body)
        assert result["messages"] == []


# ---------------------------------------------------------------------------
# Response conversion tests
# ---------------------------------------------------------------------------

class TestConvertOpenaiToAnthropicResponse:
    def _make_openai_resp(
        self,
        content: str = "Hello!",
        finish_reason: str | None = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        resp_id: str = "chatcmpl-abc123",
    ) -> dict:
        return {
            "id": resp_id,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def test_response_basic_conversion(self):
        """choices[0].message.content must appear as a text content block."""
        openai_resp = self._make_openai_resp(content="Hi there!")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "Hi there!"

    def test_response_finish_reason_stop_maps_to_end_turn(self):
        """finish_reason='stop' must map to stop_reason='end_turn'."""
        openai_resp = self._make_openai_resp(finish_reason="stop")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["stop_reason"] == "end_turn"

    def test_response_finish_reason_length_maps_to_max_tokens(self):
        """finish_reason='length' must map to stop_reason='max_tokens'."""
        openai_resp = self._make_openai_resp(finish_reason="length")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["stop_reason"] == "max_tokens"

    def test_response_finish_reason_none_maps_to_end_turn(self):
        """finish_reason=None must map to stop_reason='end_turn'."""
        openai_resp = self._make_openai_resp(finish_reason=None)
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["stop_reason"] == "end_turn"

    def test_response_finish_reason_mapping(self):
        """Exhaustive check: stop→end_turn, length→max_tokens, None→end_turn."""
        cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            (None, "end_turn"),
        ]
        for finish_reason, expected_stop_reason in cases:
            openai_resp = self._make_openai_resp(finish_reason=finish_reason)
            result = convert_openai_to_anthropic_response(openai_resp, "llama3")
            assert result["stop_reason"] == expected_stop_reason, (
                f"finish_reason={finish_reason!r} expected stop_reason={expected_stop_reason!r}, "
                f"got {result['stop_reason']!r}"
            )

    def test_response_usage_mapping(self):
        """prompt_tokens must become input_tokens and completion_tokens → output_tokens."""
        openai_resp = self._make_openai_resp(prompt_tokens=42, completion_tokens=17)
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["usage"]["input_tokens"] == 42
        assert result["usage"]["output_tokens"] == 17

    def test_response_model_field_preserved(self):
        """The model name passed in must appear in the response."""
        openai_resp = self._make_openai_resp()
        result = convert_openai_to_anthropic_response(openai_resp, "llama3:8b")
        assert result["model"] == "llama3:8b"

    def test_response_id_preserved(self):
        """The OpenAI response id must be carried over to the Anthropic response."""
        openai_resp = self._make_openai_resp(resp_id="chatcmpl-XYZ")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["id"] == "chatcmpl-XYZ"

    def test_response_missing_usage_defaults_to_zero(self):
        """If the OpenAI response has no 'usage' key, tokens should default to 0."""
        openai_resp = self._make_openai_resp()
        del openai_resp["usage"]
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0


# ---------------------------------------------------------------------------
# Normalize model name tests
# ---------------------------------------------------------------------------

class TestNormalizeModelName:
    def test_bare_name_gets_latest_tag(self):
        """A name without ':' should have ':latest' appended."""
        assert _normalize_model_name("llama3") == "llama3:latest"

    def test_tagged_name_unchanged(self):
        """A name that already contains ':' must be returned unchanged."""
        assert _normalize_model_name("llama3:8b") == "llama3:8b"

    def test_latest_tag_unchanged(self):
        """'llama3:latest' should remain 'llama3:latest'."""
        assert _normalize_model_name("llama3:latest") == "llama3:latest"

    def test_other_tag_unchanged(self):
        """Any explicit tag other than latest must be preserved as-is."""
        assert _normalize_model_name("mistral:7b-instruct") == "mistral:7b-instruct"

    def test_empty_string(self):
        """An empty string contains no ':' so ':latest' should be appended."""
        assert _normalize_model_name("") == ":latest"
