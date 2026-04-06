"""
Bidirectional conversion between Anthropic and OpenAI message formats.
All functions are pure (no I/O) except the async streaming converter.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions for tool_use / tool_calls conversion
# ---------------------------------------------------------------------------

def _convert_tool_use_to_openai(block: dict) -> dict:
    """Convert an Anthropic tool_use content block to an OpenAI tool_calls entry."""
    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input", {})),
        },
    }


def _convert_tool_result_to_openai(block: dict) -> dict:
    """Convert an Anthropic tool_result content block to an OpenAI role:tool message."""
    content = block.get("content", "")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    elif not isinstance(content, str):
        content = str(content)
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": content,
    }


def _convert_openai_tool_call_to_anthropic(tc: dict) -> dict:
    """Convert an OpenAI tool_calls entry to an Anthropic tool_use content block."""
    arguments = tc.get("function", {}).get("arguments", "{}")
    try:
        input_data = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse tool call arguments as JSON: %r", arguments)
        input_data = {}
    return {
        "type": "tool_use",
        "id": tc.get("id", ""),
        "name": tc.get("function", {}).get("name", ""),
        "input": input_data,
    }


# ---------------------------------------------------------------------------
# Request conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _extract_system_reminders(text: str) -> tuple[list[str], str]:
    """Extract <system-reminder> tags from text, returning (tags, cleaned_text)."""
    pattern = r'<system-reminder>.*?</system-reminder>'
    found = re.findall(pattern, text, re.DOTALL)
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return found, cleaned


def _extract_think_tags(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from text. Returns (thinking_text, remaining_text).
    Multiple blocks are concatenated. Empty thinking returns ("", original_text)."""
    pattern = r'<think>(.*?)</think>'
    blocks = re.findall(pattern, text, re.DOTALL)
    if blocks:
        # Closed tag(s) found — use them even if all were empty
        thinking = "\n".join(b for b in blocks if b.strip())
        remaining = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking, remaining
    # No closed tags found — try unclosed tag
    unclosed = re.search(r'<think>(.*)', text, re.DOTALL)
    if unclosed:
        thinking = unclosed.group(1)
        remaining = text[:unclosed.start()].strip()
        return thinking, remaining
    return "", text


def convert_anthropic_to_openai_request(body: dict) -> dict:
    """Convert an Anthropic /v1/messages request body to OpenAI /v1/chat/completions format."""
    messages: list[dict] = []

    # Anthropic "system" field becomes a system message prepended to the list.
    # The value may be a plain string or a list of Anthropic content blocks.
    if "system" in body:
        system = body["system"]
        if isinstance(system, list):
            system = "".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            content_str = content
            tool_use_blocks: list[dict] = []
            tool_result_messages: list[dict] = []
        elif isinstance(content, list):
            text_parts: list[str] = []
            tool_use_blocks = []
            tool_result_messages = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_use_blocks.append(block)
                    elif btype == "tool_result":
                        tool_result_messages.append(_convert_tool_result_to_openai(block))
                    else:
                        logger.warning(
                            "Skipping unsupported content block type '%s' in conversion",
                            btype,
                        )
                else:
                    logger.warning("Skipping unexpected content block (not a dict): %r", block)
            content_str = "".join(text_parts)
        else:
            content_str = str(content)
            tool_use_blocks = []
            tool_result_messages = []

        if role == "user":
            reminders, content_str = _extract_system_reminders(content_str)
            content_str = content_str.strip()
            if reminders:
                messages.append({"role": "system", "content": "\n".join(reminders)})
            for tool_result_msg in tool_result_messages:
                messages.append(tool_result_msg)
            if content_str:
                messages.append({"role": "user", "content": content_str})
        elif role == "assistant":
            if tool_use_blocks:
                content_val = content_str if content_str else None
                messages.append({
                    "role": "assistant",
                    "content": content_val,
                    "tool_calls": [_convert_tool_use_to_openai(b) for b in tool_use_blocks],
                })
            else:
                messages.append({"role": "assistant", "content": content_str})
        else:
            messages.append({"role": role, "content": content_str})

    openai_body: dict = {"messages": messages}

    # Direct pass-throughs
    for key in ("model", "temperature", "top_p", "max_tokens", "stream"):
        if key in body:
            openai_body[key] = body[key]

    # Convert Anthropic tools → OpenAI tools
    if body.get("tools"):
        openai_tools = []
        for tool in body["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        openai_body["tools"] = openai_tools

    # Ask Ollama (and other OpenAI-compatible backends) to include token usage
    # in the final streaming chunk so we can log it accurately.
    if body.get("stream"):
        openai_body["stream_options"] = {"include_usage": True}

    # Anthropic → OpenAI field renames
    if "stop_sequences" in body:
        openai_body["stop"] = body["stop_sequences"]

    return openai_body


# ---------------------------------------------------------------------------
# Response conversion: OpenAI → Anthropic (non-streaming)
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[Optional[str], str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    None: "end_turn",
}


def convert_openai_to_anthropic_response(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI chat completion response to Anthropic message format."""
    choice = openai_resp["choices"][0]
    finish_reason = choice.get("finish_reason")
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage = openai_resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    message = choice["message"]
    content_blocks = []

    # Thinking / reasoning content (comes before text)
    reasoning_content = message.get("reasoning_content")

    # Text content
    text_content = message.get("content")

    # Extract inline <think> tags if no dedicated reasoning_content field
    if not reasoning_content and text_content and "<think>" in text_content:
        extracted, text_content = _extract_think_tags(text_content)
        if extracted:
            reasoning_content = extracted

    if reasoning_content:
        content_blocks.append({"type": "thinking", "thinking": reasoning_content, "signature": ""})

    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    # Tool calls
    for tc in message.get("tool_calls") or []:
        content_blocks.append(_convert_openai_tool_call_to_anthropic(tc))

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    return {
        "id": openai_resp.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming conversion: OpenAI SSE → Anthropic SSE
# ---------------------------------------------------------------------------

def _sse(event_type: str, data: dict) -> bytes:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def convert_openai_stream_to_anthropic(
    openai_stream,
    model: str,
    message_id: str,
) -> AsyncIterator[bytes]:
    """
    Consume an OpenAI SSE stream and yield Anthropic SSE events.

    Expected usage:
        async with client.stream(...) as resp:
            async for chunk in convert_openai_stream_to_anthropic(resp, model, msg_id):
                yield chunk
    """
    # 1. message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # 2. ping
    yield _sse("ping", {"type": "ping"})

    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    # Block index tracking:
    # Without thinking: text=0, tool_use=1+
    # With thinking:    thinking=0, text=1, tool_use=2+
    has_thinking = False
    thinking_block_started = False
    thinking_block_closed = False

    text_block_index = 0
    text_block_started = False
    text_block_closed = False
    next_block_index = 0

    # tool_calls_accum: openai_tc_index -> {"id", "name", "block_index"}
    tool_calls_accum: dict[int, dict] = {}

    # Inline <think> tag detection for models that embed reasoning in content
    inline_think_detected: Optional[bool] = None  # None=unknown, True=yes, False=no
    inline_think_complete = False
    inline_think_buffer = ""
    think_tail_buffer = ""

    async for line in openai_stream.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Could not parse OpenAI stream chunk: %r", payload)
            continue

        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

        # --- Reasoning/thinking content (dedicated field) ---
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            if not thinking_block_started:
                has_thinking = True
                text_block_index = 1
                next_block_index = 2
                thinking_block_started = True
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": reasoning_content},
            })

        # --- Text content (may contain inline <think> tags) ---
        text_content = delta.get("content")
        if text_content:
            # Only run inline detection if no dedicated reasoning_content field seen
            if not has_thinking and inline_think_detected is None:
                inline_think_buffer += text_content
                stripped = inline_think_buffer.lstrip()
                if stripped.startswith("<think>"):
                    # Inline think mode confirmed
                    inline_think_detected = True
                    has_thinking = True
                    text_block_index = 1
                    next_block_index = 2
                    thinking_block_started = True
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    })
                    after_tag = stripped[len("<think>"):]
                    if "</think>" in after_tag:
                        parts = after_tag.split("</think>", 1)
                        if parts[0]:
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "thinking_delta", "thinking": parts[0]},
                            })
                        # Close thinking block
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "signature_delta", "signature": ""},
                        })
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": 0,
                        })
                        thinking_block_closed = True
                        inline_think_complete = True
                        text_content = parts[1].strip() if parts[1].strip() else None
                        inline_think_buffer = ""
                    else:
                        # Buffer the last 8 chars of after_tag in case </think> is
                        # split across the SSE chunk boundary (same logic as continuation)
                        if after_tag:
                            if len(after_tag) > 8:
                                emit_part = after_tag[:-8]
                                think_tail_buffer = after_tag[-8:]
                                yield _sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "thinking_delta", "thinking": emit_part},
                                })
                            else:
                                think_tail_buffer = after_tag
                        inline_think_buffer = ""
                        text_content = None
                elif len(stripped) > 0 and not "<think>".startswith(stripped[:6]):
                    # Definitely not a think tag
                    inline_think_detected = False
                    text_content = inline_think_buffer
                    inline_think_buffer = ""
                elif len(inline_think_buffer) > 32:
                    # Buffer limit — no think tag coming
                    inline_think_detected = False
                    text_content = inline_think_buffer
                    inline_think_buffer = ""
                else:
                    text_content = None  # still buffering

            elif inline_think_detected is True and not inline_think_complete:
                # Inside inline think block; prepend any buffered tail from previous chunk
                text_content = think_tail_buffer + text_content
                think_tail_buffer = ""
                if "</think>" in text_content:
                    parts = text_content.split("</think>", 1)
                    if parts[0]:
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "thinking_delta", "thinking": parts[0]},
                        })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": ""},
                    })
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": 0,
                    })
                    thinking_block_closed = True
                    inline_think_complete = True
                    text_content = parts[1].strip() if parts[1].strip() else None
                else:
                    # </think> not yet seen; keep last 8 chars buffered in case the
                    # closing tag is split across the SSE chunk boundary.
                    if len(text_content) > 8:
                        emit_part = text_content[:-8]
                        think_tail_buffer = text_content[-8:]
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "thinking_delta", "thinking": emit_part},
                        })
                    else:
                        think_tail_buffer = text_content
                    text_content = None

            # Emit text content (original behaviour, now index-aware)
            if text_content:
                # Re-open text block at a new index if the previous one was closed by a tool call
                if text_block_closed:
                    text_block_index = next_block_index
                    next_block_index += 1
                    text_block_started = False
                    text_block_closed = False
                if not text_block_started:
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_block_started = True
                    next_block_index = text_block_index + 1
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": text_content},
                })

        # Handle tool_calls
        for tc_delta in delta.get("tool_calls") or []:
            tc_idx = tc_delta.get("index", 0)

            if tc_idx not in tool_calls_accum:
                # Close text block if it was open
                if text_block_started and not text_block_closed:
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": text_block_index,
                    })
                    text_block_closed = True

                block_index = next_block_index
                next_block_index += 1
                tool_calls_accum[tc_idx] = {
                    "id": tc_delta.get("id", ""),
                    "name": tc_delta.get("function", {}).get("name", ""),
                    "block_index": block_index,
                }
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_delta.get("id", ""),
                        "name": tc_delta.get("function", {}).get("name", ""),
                        "input": {},
                    },
                })

            tc_state = tool_calls_accum[tc_idx]
            args_chunk = tc_delta.get("function", {}).get("arguments", "")
            if args_chunk:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": tc_state["block_index"],
                    "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                })

    # Flush any unresolved inline_think_buffer as text
    if inline_think_buffer and inline_think_detected is None:
        text_content = inline_think_buffer
        inline_think_buffer = ""
        if text_content:
            if not text_block_started:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                text_block_started = True
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {"type": "text_delta", "text": text_content},
            })

    # Flush any tail buffer that was held back waiting for </think>
    if think_tail_buffer and thinking_block_started and not thinking_block_closed:
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": think_tail_buffer},
        })
        think_tail_buffer = ""

    # Close thinking block if still open (e.g. unclosed inline <think> tag)
    if thinking_block_started and not thinking_block_closed:
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": ""},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })
        thinking_block_closed = True

    # Close text block if open
    if text_block_started and not text_block_closed:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": text_block_index,
        })

    for tc_idx in sorted(tool_calls_accum.keys()):
        block_index = tool_calls_accum[tc_idx]["block_index"]
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })

    # message_delta
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })

    # message_stop
    yield _sse("message_stop", {"type": "message_stop"})
