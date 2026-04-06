"""
Unit tests for converters/anthropic_openai.py and the normalize model name helper
in startup_checks.py.  These tests require no API keys and no running services.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Ensure the proxy source root is importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from converters.anthropic_openai import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic_response,
    convert_openai_stream_to_anthropic,
    _extract_think_tags,
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


# ---------------------------------------------------------------------------
# Request conversion — tool_use / tool_result tests
# ---------------------------------------------------------------------------

class TestConvertAnthropicToOpenaiRequestToolUse:
    def test_request_tool_use_blocks_converted(self):
        """assistant message with tool_use blocks must produce tool_calls on the OpenAI message."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"] == "toolu_01"
        assert tc["function"]["name"] == "get_weather"

    def test_request_tool_use_with_text(self):
        """assistant message with both text and tool_use must produce content and tool_calls."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_02",
                            "name": "search",
                            "input": {"query": "weather"},
                        },
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_request_tool_result_becomes_tool_message(self):
        """user message with a tool_result block must produce a role:tool message."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "Sunny, 22°C",
                        }
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        # Expect exactly one tool message; the user text is empty so no user msg
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "toolu_01"
        assert msg["content"] == "Sunny, 22°C"

    def test_request_multiple_tool_results(self):
        """user message with multiple tool_result blocks must produce multiple tool messages."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "Result A",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_02",
                            "content": "Result B",
                        },
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"toolu_01", "toolu_02"}

    def test_request_tool_result_with_list_content(self):
        """tool_result whose content is a list of text blocks must be joined."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_03",
                            "content": [
                                {"type": "text", "text": "Hello "},
                                {"type": "text", "text": "world"},
                            ],
                        }
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "Hello world"

    def test_request_tool_result_with_string_content(self):
        """tool_result whose content is a plain string must be passed through unchanged."""
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_04",
                            "content": "plain string result",
                        }
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        msgs = result["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "plain string result"

    def test_request_arguments_serialized_as_string(self):
        """tool_use input dict must be serialized to a JSON string in the arguments field."""
        input_dict = {"location": "London", "unit": "celsius"}
        body = {
            "model": "llama3",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_05",
                            "name": "get_weather",
                            "input": input_dict,
                        }
                    ],
                }
            ],
        }
        result = convert_anthropic_to_openai_request(body)
        tc = result["messages"][0]["tool_calls"][0]
        args = tc["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == input_dict


# ---------------------------------------------------------------------------
# Non-streaming response conversion — tool_calls tests
# ---------------------------------------------------------------------------

class TestConvertOpenaiToAnthropicResponseToolCalls:
    def _make_openai_resp_with_tool_calls(
        self,
        tool_calls: list,
        content: str | None = None,
        finish_reason: str = "tool_calls",
    ) -> dict:
        return {
            "id": "chatcmpl-tool1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }

    def _make_tool_call(self, tc_id: str, name: str, arguments: dict) -> dict:
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    def test_response_tool_calls_converted(self):
        """OpenAI response with tool_calls must produce tool_use content blocks."""
        tc = self._make_tool_call("toolu_01", "get_weather", {"location": "Paris"})
        openai_resp = self._make_openai_resp_with_tool_calls([tc])
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        tool_use_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        block = tool_use_blocks[0]
        assert block["id"] == "toolu_01"
        assert block["name"] == "get_weather"

    def test_response_tool_calls_finish_reason(self):
        """finish_reason 'tool_calls' must map to stop_reason 'tool_use'."""
        tc = self._make_tool_call("toolu_01", "get_weather", {})
        openai_resp = self._make_openai_resp_with_tool_calls([tc], finish_reason="tool_calls")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        assert result["stop_reason"] == "tool_use"

    def test_response_tool_calls_with_text(self):
        """response with both content text and tool_calls must produce text and tool_use blocks."""
        tc = self._make_tool_call("toolu_02", "search", {"query": "AI"})
        openai_resp = self._make_openai_resp_with_tool_calls(
            [tc], content="I will search for that."
        )
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "I will search for that."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "search"

    def test_response_tool_calls_arguments_parsed(self):
        """arguments JSON string must be parsed to a dict in the input field."""
        args = {"city": "Tokyo", "days": 3}
        tc = self._make_tool_call("toolu_03", "forecast", args)
        openai_resp = self._make_openai_resp_with_tool_calls([tc])
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_block["input"] == args


# ---------------------------------------------------------------------------
# Streaming conversion — tool_calls tests
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run a coroutine synchronously for testing purposes."""
    return asyncio.get_event_loop().run_until_complete(coro)


class _MockStream:
    """
    Minimal mock of an httpx streaming response.
    Yields pre-formed SSE lines from a list of OpenAI stream chunk dicts.
    """

    def __init__(self, chunks: list[dict]):
        self._lines = []
        for chunk in chunks:
            self._lines.append(f"data: {json.dumps(chunk)}")
        self._lines.append("data: [DONE]")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _collect_stream(chunks: list[dict], model: str = "llama3") -> list[tuple[str, dict]]:
    """
    Run convert_openai_stream_to_anthropic against a mock stream and return
    a list of (event_type, data) tuples parsed from the yielded SSE bytes.
    """
    stream = _MockStream(chunks)

    async def _collect():
        events = []
        async for raw in convert_openai_stream_to_anthropic(stream, model, "msg_test"):
            text = raw.decode()
            # Each SSE block: "event: <type>\ndata: <json>\n\n"
            lines = [l for l in text.strip().splitlines() if l]
            event_type = lines[0].replace("event: ", "")
            data = json.loads(lines[1].replace("data: ", ""))
            events.append((event_type, data))
        return events

    return _run_async(_collect())


def _make_tool_call_chunk(
    index: int,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str = "",
    finish_reason: str | None = None,
) -> dict:
    """Build an OpenAI streaming chunk with a tool_calls delta."""
    tc_delta: dict = {"index": index}
    if tc_id is not None:
        tc_delta["id"] = tc_id
    if name is not None:
        tc_delta["function"] = {"name": name, "arguments": arguments}
    else:
        tc_delta["function"] = {"arguments": arguments}
    choice: dict = {"index": 0, "delta": {"tool_calls": [tc_delta]}}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return {"id": "chatcmpl-s1", "object": "chat.completion.chunk", "choices": [choice]}


class TestConvertOpenaiStreamToAnthropicToolCalls:
    def test_stream_tool_calls_emits_tool_use_blocks(self):
        """Mock stream with tool_calls deltas must produce correct Anthropic SSE events."""
        chunks = [
            _make_tool_call_chunk(0, tc_id="toolu_01", name="get_weather", arguments='{"loc'),
            _make_tool_call_chunk(0, arguments='ation": "Paris"}'),
            _make_tool_call_chunk(0, arguments="", finish_reason="tool_calls"),
        ]
        events = _collect_stream(chunks)

        event_types = [e[0] for e in events]
        # Must have message_start, ping, content_block_start (tool_use), at least one
        # content_block_delta, content_block_stop, message_delta, message_stop
        assert "message_start" in event_types
        assert "ping" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        # Verify the content_block_start is a tool_use block
        block_starts = [d for et, d in events if et == "content_block_start"]
        tool_use_starts = [b for b in block_starts if b["content_block"]["type"] == "tool_use"]
        assert len(tool_use_starts) == 1
        assert tool_use_starts[0]["content_block"]["name"] == "get_weather"
        assert tool_use_starts[0]["content_block"]["id"] == "toolu_01"

    def test_stream_tool_calls_no_text_block(self):
        """When response has only tool calls, no text content_block must be emitted and
        the first block index must be 0."""
        chunks = [
            _make_tool_call_chunk(0, tc_id="toolu_01", name="search", arguments='{"q": "x"}'),
            _make_tool_call_chunk(0, arguments="", finish_reason="tool_calls"),
        ]
        events = _collect_stream(chunks)

        block_starts = [d for et, d in events if et == "content_block_start"]
        # No text content_block should be present
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]
        assert len(text_starts) == 0, "No text content_block should be emitted for tool-only responses"

        # The tool_use block must have index 0 (first and only block)
        tool_starts = [b for b in block_starts if b["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 1
        assert tool_starts[0]["index"] == 0

    def test_stream_tool_calls_finish_reason(self):
        """finish_reason 'tool_calls' in a stream chunk must map to stop_reason 'tool_use'."""
        chunks = [
            _make_tool_call_chunk(0, tc_id="toolu_01", name="calc", arguments='{"x": 1}'),
            _make_tool_call_chunk(0, arguments="", finish_reason="tool_calls"),
        ]
        events = _collect_stream(chunks)

        message_delta_events = [d for et, d in events if et == "message_delta"]
        assert len(message_delta_events) == 1
        assert message_delta_events[0]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Non-streaming response — thinking/reasoning content tests
# ---------------------------------------------------------------------------

class TestConvertOpenaiToAnthropicResponseThinking:
    """Tests for thinking/reasoning block handling in non-streaming responses."""

    def _make_openai_resp(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        finish_reason: str = "stop",
    ) -> dict:
        message: dict = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        return {
            "id": "chatcmpl-think1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    def test_reasoning_content_produces_thinking_block_before_text(self):
        """reasoning_content field must appear as a thinking block before the text block."""
        openai_resp = self._make_openai_resp(
            content="Here is the answer.",
            reasoning_content="I thought about this carefully.",
        )
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "I thought about this carefully."
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "Here is the answer."

    def test_inline_think_tags_extracted_as_thinking_block(self):
        """Inline <think>...</think> in content must produce a thinking block and
        the remaining text must be in the text block."""
        openai_resp = self._make_openai_resp(
            content="<think>step by step reasoning</think>Final answer here."
        )
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        assert len(blocks) == 2
        thinking_blocks = [b for b in blocks if b["type"] == "thinking"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "step by step reasoning"
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Final answer here."

    def test_empty_think_tags_produce_no_thinking_block(self):
        """Empty <think></think> must not produce a thinking block."""
        openai_resp = self._make_openai_resp(content="<think></think>The answer is 42.")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        thinking_blocks = [b for b in blocks if b["type"] == "thinking"]
        assert len(thinking_blocks) == 0
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "The answer is 42."

    def test_no_thinking_content_unchanged(self):
        """Plain text response without reasoning must produce a single text block (regression)."""
        openai_resp = self._make_openai_resp(content="Just a normal reply.")
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Just a normal reply."

    def test_reasoning_content_takes_precedence_over_inline_think(self):
        """When both reasoning_content and inline <think> are present,
        only one thinking block must be emitted using reasoning_content."""
        openai_resp = self._make_openai_resp(
            content="<think>inline thinking</think>Some text.",
            reasoning_content="dedicated reasoning field",
        )
        result = convert_openai_to_anthropic_response(openai_resp, "llama3")
        blocks = result["content"]
        thinking_blocks = [b for b in blocks if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1, (
            f"Expected exactly one thinking block, got {len(thinking_blocks)}"
        )
        # Must use the dedicated reasoning_content, not the inline tag
        assert thinking_blocks[0]["thinking"] == "dedicated reasoning field"


# ---------------------------------------------------------------------------
# _extract_think_tags unit tests
# ---------------------------------------------------------------------------

class TestExtractThinkTags:
    """Unit tests for the _extract_think_tags helper function."""

    def test_single_block(self):
        """A single <think>...</think> block must be extracted."""
        thinking, remaining = _extract_think_tags("<think>my thoughts</think>answer")
        assert thinking == "my thoughts"
        assert remaining == "answer"

    def test_multiple_blocks_concatenated_with_newline(self):
        """Multiple <think> blocks must be concatenated with a newline."""
        text = "<think>block one</think>middle<think>block two</think>end"
        thinking, remaining = _extract_think_tags(text)
        assert thinking == "block one\nblock two"
        assert "block one" in thinking
        assert "block two" in thinking
        assert remaining == "middleend"

    def test_empty_block_returns_no_thinking(self):
        """<think></think> must produce empty thinking string and original remaining text."""
        thinking, remaining = _extract_think_tags("<think></think>some text")
        assert thinking == ""
        assert remaining == "some text"

    def test_unclosed_tag_returns_content_as_thinking(self):
        """An unclosed <think> tag must capture all text after the tag as thinking."""
        thinking, remaining = _extract_think_tags("prefix<think>open ended content")
        assert thinking == "open ended content"
        assert remaining == "prefix"

    def test_no_tags_returns_original_text(self):
        """Text with no <think> tags must be returned unchanged as remaining."""
        thinking, remaining = _extract_think_tags("just plain text no tags")
        assert thinking == ""
        assert remaining == "just plain text no tags"


# ---------------------------------------------------------------------------
# Streaming conversion — thinking/reasoning content tests
# ---------------------------------------------------------------------------

def _make_text_chunk(
    content: str,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> dict:
    """Build an OpenAI streaming chunk with a text delta (and optional reasoning_content)."""
    delta: dict = {"content": content}
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return {"id": "chatcmpl-s2", "object": "chat.completion.chunk", "choices": [choice]}


def _make_reasoning_chunk(reasoning: str) -> dict:
    """Build an OpenAI streaming chunk with only a reasoning_content delta."""
    choice: dict = {"index": 0, "delta": {"reasoning_content": reasoning}}
    return {"id": "chatcmpl-s2", "object": "chat.completion.chunk", "choices": [choice]}


class TestConvertOpenaiStreamThinking:
    """Tests for thinking/reasoning block handling in streaming responses."""

    def test_reasoning_content_deltas_produce_thinking_at_index_0_text_at_index_1(self):
        """reasoning_content deltas must produce a thinking block at index 0 and
        text content at index 1."""
        chunks = [
            _make_reasoning_chunk("step 1"),
            _make_reasoning_chunk("step 2"),
            _make_text_chunk("Final answer.", finish_reason="stop"),
        ]
        events = _collect_stream(chunks)

        block_starts = [d for et, d in events if et == "content_block_start"]
        thinking_starts = [b for b in block_starts if b["content_block"]["type"] == "thinking"]
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]

        assert len(thinking_starts) == 1, "Expected exactly one thinking block_start"
        assert thinking_starts[0]["index"] == 0, "Thinking block must be at index 0"

        assert len(text_starts) == 1, "Expected exactly one text block_start"
        assert text_starts[0]["index"] == 1, "Text block must be at index 1"

        # Check that thinking deltas contain our content
        thinking_deltas = [
            d["delta"]["thinking"]
            for et, d in events
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "thinking_delta"
        ]
        assert "step 1" in thinking_deltas
        assert "step 2" in thinking_deltas

    def test_inline_think_open_then_close_then_text_correct_block_sequence(self):
        """Inline <think> in first chunk then </think> then text in later chunks must
        produce correct thinking block at 0 then text block at 1."""
        chunks = [
            _make_text_chunk("<think>"),
            _make_text_chunk("inner reasoning"),
            _make_text_chunk("</think>actual text", finish_reason="stop"),
        ]
        events = _collect_stream(chunks)

        block_starts = [d for et, d in events if et == "content_block_start"]
        thinking_starts = [b for b in block_starts if b["content_block"]["type"] == "thinking"]
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]

        assert len(thinking_starts) == 1, "Expected exactly one thinking block_start"
        assert thinking_starts[0]["index"] == 0

        assert len(text_starts) == 1, "Expected exactly one text block_start"
        assert text_starts[0]["index"] == 1

    def test_closing_tag_split_across_chunks_not_emitted_as_thinking_content(self):
        """When </think> is split across two SSE chunks the partial closing tag
        must not be emitted as thinking content (H2 fix verification)."""
        # Chunk boundary splits "</think>" as "</thi" + "nk>"
        chunks = [
            _make_text_chunk("<think>some reasoning</thi"),
            _make_text_chunk("nk>text after"),
            _make_text_chunk("", finish_reason="stop"),
        ]
        events = _collect_stream(chunks)

        # Collect all thinking delta text
        thinking_delta_texts = [
            d["delta"]["thinking"]
            for et, d in events
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "thinking_delta"
        ]
        combined_thinking = "".join(thinking_delta_texts)

        # The closing tag fragment "</thi" or "nk>" must not appear in thinking content
        assert "</thi" not in combined_thinking, (
            f"Partial closing tag '</thi' must not appear in thinking content. Got: {combined_thinking!r}"
        )
        assert "nk>" not in combined_thinking, (
            f"Partial closing tag 'nk>' must not appear in thinking content. Got: {combined_thinking!r}"
        )

        # Verify text block was produced after the thinking block
        block_starts = [d for et, d in events if et == "content_block_start"]
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]
        assert len(text_starts) == 1, "Expected a text block after the thinking block"
        assert text_starts[0]["index"] == 1

    def test_thinking_plus_tool_calls_thinking_at_0_tool_at_2(self):
        """Stream with thinking + tool_calls and no text must produce thinking=0, tool=2
        (text block at index 1 is reserved but not started, H1 fix verification)."""
        chunks = [
            _make_reasoning_chunk("I will use a tool"),
            # tool_call delta — no text content
            _make_tool_call_chunk(0, tc_id="toolu_01", name="get_weather", arguments='{"city": "Rome"}'),
            _make_tool_call_chunk(0, arguments="", finish_reason="tool_calls"),
        ]
        events = _collect_stream(chunks)

        block_starts = [d for et, d in events if et == "content_block_start"]
        thinking_starts = [b for b in block_starts if b["content_block"]["type"] == "thinking"]
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]
        tool_starts = [b for b in block_starts if b["content_block"]["type"] == "tool_use"]

        assert len(thinking_starts) == 1, "Expected exactly one thinking block"
        assert thinking_starts[0]["index"] == 0, "Thinking must be at index 0"

        assert len(text_starts) == 0, "No text block should be emitted when there is no text"

        assert len(tool_starts) == 1, "Expected exactly one tool_use block"
        assert tool_starts[0]["index"] == 2, (
            f"Tool block must be at index 2 (0=thinking, 1=reserved text), "
            f"got index {tool_starts[0]['index']}"
        )

    def test_no_thinking_content_text_at_index_0(self):
        """Stream with only text content (no thinking) must produce text block at index 0
        (regression test)."""
        chunks = [
            _make_text_chunk("Hello "),
            _make_text_chunk("world.", finish_reason="stop"),
        ]
        events = _collect_stream(chunks)

        block_starts = [d for et, d in events if et == "content_block_start"]
        thinking_starts = [b for b in block_starts if b["content_block"]["type"] == "thinking"]
        text_starts = [b for b in block_starts if b["content_block"]["type"] == "text"]

        assert len(thinking_starts) == 0, "No thinking block expected for plain text response"
        assert len(text_starts) == 1, "Expected exactly one text block"
        assert text_starts[0]["index"] == 0, "Text block must be at index 0 when no thinking"
