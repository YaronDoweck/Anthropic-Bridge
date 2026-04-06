# Project Overview: LLM Proxy

A FastAPI proxy that accepts **Anthropic-format API requests** and routes them to the **Anthropic API** (pass-through), a local **Ollama instance** (native Anthropic format), or an **OpenAI-compatible API** (with format conversion), based on the model name in the request.

---

## Architecture

```
Client (Anthropic format)
        │
        ▼
   server.py          FastAPI entrypoint, lifespan management
        │
        ▼
   router.py          Parses body → extracts model → looks up config → dispatches
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
handlers/ handlers/  handlers/
anthropic ollama     openai
.py       .py        .py
   │       │            │
   ▼       ▼            ▼
Anthropic Ollama      OpenAI-compatible
  API    (local,       API (converts
        Anthropic-     Anthropic↔OpenAI
        compatible)    format)
```

---

## File Map

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app, lifespan, catch-all route |
| `router.py` | Parse body, look up model, dispatch to handler, DEBUG logging |
| `config.py` | Load `proxy.config` TOML into `ProxyConfig` / `ModelConfig` / `ServerConfig` dataclasses |
| `logging_config.py` | Structured logging setup, `log_request()`, `get_debug_level()`, `format_body_short()` helpers |
| `startup_checks.py` | Validate Ollama is installed/running/models pulled at startup |
| `proxy.config` | TOML: server settings, endpoint URLs, and model→endpoint mapping |
| `.env` | Private credentials only: ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY |
| `handlers/base.py` | Abstract `BaseHandler` interface |
| `handlers/anthropic.py` | Pass-through to Anthropic API, injects fallback auth |
| `handlers/ollama.py` | Forwards to Ollama Anthropic-compatible endpoint; handles system-reminder extraction |
| `handlers/openai.py` | Converts Anthropic requests to OpenAI format, forwards to OpenAI-compatible API, converts response back |
| `converters/anthropic_openai.py` | Pure conversion functions: Anthropic↔OpenAI request/response/streaming |
| `tests/conftest.py` | Pytest fixtures: `proxy_config`, `test_port`, `running_server` |
| `tests/test_converters.py` | Unit tests for converters (no network) |
| `tests/test_router.py` | Router tests via TestClient with mocked httpx |
| `tests/test_anthropic_live.py` | Live integration tests (requires API key/token) |
| `scripts/` | start/stop service scripts for macOS (launchd), Linux (systemd), Windows (PS) |

---

## Configuration

### `proxy.config` (TOML)

```toml
[server]
host = "127.0.0.1"
port = 8000
log_level = "INFO"
debug_level = 0
# output_dir = "/path/to/logs"

[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://127.0.0.1:11434"
openai_url = "https://api.openai.com"

[models."claude-3-haiku-20240307"]
endpoint = "anthropic"

[models."llama3.2"]
endpoint = "ollama"

[models."gpt-4o"]
endpoint = "openai"
```

The `[server]` section is fully optional — all fields have defaults. Each model is a **TOML table** (not a simple string) so future fields can be added without breaking existing entries. Model names are **exact case-sensitive** matches against the `model` field in incoming requests.

#### `[server]` fields

| Field | Default | Description |
|-------|---------|-------------|
| `host` | `127.0.0.1` | Bind address |
| `port` | `8000` | Listen port |
| `log_level` | `INFO` | Python log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `debug_level` | `0` | Verbosity of request/response body logging: 0=off, 1=one-line summary, 2=full pretty-printed JSON, 3=reserved for future use. Setting ≥1 forces root logger to DEBUG. |
| `output_dir` | `""` | Path to directory for per-session JSONL log files. Each session writes to `<session_id>.jsonl` in this directory. Empty string means disabled. |
| `override_model` | `""` | When set, all requests are routed to this model regardless of the incoming `model` field. Must be defined in `[models]`. Empty = disabled. |

Environment variables override `proxy.config` values when set (e.g. `LOG_LEVEL=DEBUG uv run llm-proxy` still works). The env var names mirror the field names: `HOST`, `PORT`, `LOG_LEVEL`, `DEBUG_LEVEL`, `OUTPUT_DIR`, `OVERRIDE_MODEL`.

### `.env`

Keep this file for private credentials only — do not commit it.

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_AUTH_TOKEN` | Fallback bearer token (injected if client sends no auth) |
| `ANTHROPIC_API_KEY` | Fallback API key (injected if client sends no auth) |
| `OPENAI_API_KEY` | API key injected as `Authorization: Bearer` for OpenAI handler |

**Auth priority** (Anthropic handler): incoming `x-api-key` > incoming `Authorization` > `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY`

**Auth** (OpenAI handler): `OPENAI_API_KEY` from `.env` is always injected as `Authorization: Bearer <key>`, overriding any client-supplied header.

---

## Request Flow

1. Client POSTs Anthropic-format request to `/{path}` (e.g. `/v1/messages`)
2. `router.py` parses body, extracts `model`
3. Looks up `ModelConfig` in `proxy.config` — returns 400 if not found
4. Dispatches to handler based on `model_config.endpoint`
5. **Anthropic handler**: forwards all headers as-is (except hop-by-hop), injects fallback auth if needed, streams or buffers Anthropic SSE/JSON unchanged
6. **Ollama handler**: strips non-essential headers, applies system-reminder handling if configured, forwards request unchanged to Ollama's Anthropic-compatible endpoint, streams or buffers response unchanged
7. **OpenAI handler**: allowlists headers (content-type, accept, user-agent, accept-encoding only), injects `Authorization: Bearer` from `OPENAI_API_KEY`, converts request body from Anthropic to OpenAI format (`convert_anthropic_to_openai_request`), always POSTs to `{openai_url}/v1/chat/completions`. Non-streaming: converts OpenAI JSON response back to Anthropic format (`convert_openai_to_anthropic_response`). Streaming: pipes OpenAI SSE through `convert_openai_stream_to_anthropic` and yields Anthropic SSE events. Error responses (non-200) are mapped to Anthropic error schema: 401→authentication_error, 429→rate_limit_error, 400→invalid_request_error, 5xx→api_error.
8. All errors returned in Anthropic error schema: `{"type": "error", "error": {"type": "...", "message": "..."}}`

---

## Ollama System-Reminder Handling

Controlled per-model via `claude_system_instructions` in `proxy.config`:

- `passthrough` (default): body forwarded unchanged
- `strip`: strips everything up to and including `</system-reminder>\n\n` from the `system` field
- `split`: extracts `<system-reminder>` blocks from user messages and appends their content to the `system` field

`omit_claude_main_description = true` (default): removes the `system` field entirely before forwarding (has no effect when `claude_system_instructions = "passthrough"`).

---

## Startup Checks (Ollama)

If any model uses `endpoint = "ollama"`:
1. Check `ollama` binary exists — interactive: offer to install; non-interactive: hard fail
2. Check Ollama server is responding — interactive: auto-start; non-interactive: hard fail
3. Check all configured ollama models are pulled — interactive: offer to pull; non-interactive: hard fail
4. Model name comparison normalizes `:latest` (e.g. `llama3` matches `llama3:latest`)
5. If proxy started Ollama, it terminates that process on shutdown

---

## Running

```bash
uv sync                  # install dependencies
uv run llm-proxy         # start server
LOG_LEVEL=DEBUG uv run llm-proxy  # with full request/response logging

uv run pytest            # unit + router tests (no credentials needed)
ANTHROPIC_API_KEY=sk-... uv run pytest  # include live tests
```

---

## Adding a New Model

Edit `proxy.config` only — no code changes needed:
```toml
[models."new-model-name"]
endpoint = "ollama"   # or "anthropic"
```

## Adding a New Backend

1. Create `handlers/<name>.py` extending `BaseHandler`
2. Add endpoint URL to `[endpoints]` in `proxy.config`
3. Register handler in `ProxyRouter.__init__()` in `router.py`
4. Optionally add `converters/anthropic_<name>.py` for format conversion

---

## Known Limitations / Deferred Items

- Streaming errors from Ollama arrive as HTTP 200 with a partial event stream
- No request body size limit
- `log_request()` in `logging_config.py` is defined but not yet called from handlers
- Blocking subprocess calls in `startup_checks.py` run on the asyncio event loop thread (acceptable for startup-only use)
