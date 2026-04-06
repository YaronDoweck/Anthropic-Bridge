import json
import logging
import os


def get_debug_level() -> int:
    """Read DEBUG_LEVEL env var and return an int clamped to 0-3."""
    raw = os.getenv("DEBUG_LEVEL", "0")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return max(0, min(3, value))


def format_body_short(body: dict) -> str:
    """Return a single-line summary of a request or response body.

    Request shape: has a ``messages`` key.
    Response shape: has both ``content`` and ``stop_reason`` keys.
    Falls back to a compact JSON dump for anything else.
    """
    if not isinstance(body, dict):
        return str(body)[:120]
    if "messages" in body:
        # Request shape
        messages = body["messages"]
        model = body.get("model", "?")
        last_role = messages[-1].get("role", "?") if messages else "?"
        last_content = messages[-1].get("content", "") if messages else ""
        if isinstance(last_content, list):
            # Flatten content blocks to text
            last_content = " ".join(
                block.get("text", "") for block in last_content if isinstance(block, dict)
            )
        preview = str(last_content)[:80].replace("\n", " ")
        return f"model={model} msgs={len(messages)} last={last_role!r}: {preview!r}"
    elif "content" in body and "stop_reason" in body:
        # Response shape
        content = body.get("content", [])
        stop_reason = body.get("stop_reason", "?")
        usage = body.get("usage", {})
        text = ""
        if content and isinstance(content, list):
            text = content[0].get("text", "") if isinstance(content[0], dict) else ""
        preview = str(text)[:80].replace("\n", " ")
        return (
            f"stop_reason={stop_reason} "
            f"in={usage.get('input_tokens', '?')} "
            f"out={usage.get('output_tokens', '?')} "
            f"text={preview!r}"
        )
    else:
        s = json.dumps(body, ensure_ascii=False)
        return s[:120] + ("..." if len(s) > 120 else "")


def setup_logging(log_level: str | None = None, debug_level: int | None = None) -> None:
    if log_level is not None:
        log_level_name = log_level.upper()
    else:
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved_log_level = getattr(logging, log_level_name, logging.INFO)

    if debug_level is not None:
        resolved_debug_level = debug_level
    else:
        resolved_debug_level = get_debug_level()

    # DEBUG_LEVEL >= 1 forces the root logger to DEBUG regardless of LOG_LEVEL
    if resolved_debug_level >= 1:
        resolved_log_level = logging.DEBUG

    logging.basicConfig(
        level=resolved_log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_request(
    method: str,
    path: str,
    model: str,
    endpoint_type: str,
    status_code: int,
    latency_ms: float,
) -> None:
    logger = logging.getLogger("proxy.request")
    logger.info(
        "method=%s path=%s model=%s endpoint=%s status=%d latency_ms=%.1f",
        method,
        path,
        model,
        endpoint_type,
        status_code,
        latency_ms,
    )
