"""
Bidirectional conversion between Anthropic and OpenAI message formats.
All functions are pure (no I/O) except the async streaming converter.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request conversion: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def convert_anthropic_to_openai_request(body: dict) -> dict:
    """Convert an Anthropic /v1/messages request body to OpenAI /v1/chat/completions format."""
    messages: list[dict] = []

    # Anthropic "system" field becomes a system message prepended to the list
    if "system" in body:
        messages.append({"role": "system", "content": body["system"]})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Content is a list of blocks — join text blocks, warn and skip others
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        logger.warning(
                            "Skipping unsupported content block type '%s' in conversion",
                            block.get("type"),
                        )
                else:
                    logger.warning("Skipping unexpected content block (not a dict): %r", block)
            messages.append({"role": role, "content": "".join(text_parts)})
        else:
            # Fallback: stringify whatever we got
            messages.append({"role": role, "content": str(content)})

    openai_body: dict = {"messages": messages}

    # Direct pass-throughs
    for key in ("model", "temperature", "top_p", "max_tokens", "stream"):
        if key in body:
            openai_body[key] = body[key]

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

    return {
        "id": openai_resp.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "text", "text": choice["message"].get("content", "")}
        ],
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
    openai_stream,  # httpx.Response used as async context manager
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

    # 2. content_block_start
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    # 3. ping (Anthropic clients expect this early)
    yield _sse("ping", {"type": "ping"})

    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    # 4. Stream content deltas
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

        # Accumulate usage if present (some providers include it in the last chunk)
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        # Track finish reason from the chunk that closes the stream
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

        content = delta.get("content")
        if content is not None:
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            })

    # 5. content_block_stop
    yield _sse("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    # 6. message_delta
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })

    # 7. message_stop
    yield _sse("message_stop", {"type": "message_stop"})
