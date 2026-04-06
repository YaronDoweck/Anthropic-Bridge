# LLM Proxy

An LLM routing proxy that accepts Anthropic-format requests and routes them to Anthropic or Ollama based on model name.

## How It Works

1. An incoming request arrives with a JSON body containing a `model` field.
2. The proxy looks up the model name in `proxy.config`.
3. Depending on the configured endpoint:
   - **anthropic** — the request is forwarded as-is (pass-through) to the Anthropic API.
   - **ollama** — the request body is converted from Anthropic format to Ollama's `/api/chat` format, sent to the local Ollama server, and the response is converted back to Anthropic format before being returned to the caller.

Streaming is supported for both endpoint types.

## Requirements

- Python 3.9+
- [UV](https://github.com/astral-sh/uv) package manager
- Ollama (optional — only required if you configure any `ollama` endpoint models)

## Installation

```bash
git clone <repository-url>
cd proxy
uv sync
cp .env.example .env   # edit with your settings
```

After cloning, make the service scripts executable:

```bash
chmod +x scripts/*.sh
```

## Configuration

Model routing is controlled by `proxy.config`, a TOML file with two sections.

### `[endpoints]`

Defines the base URLs for each upstream service. These are the defaults:

```toml
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url    = "http://127.0.0.1:11434"
```

### `[models.<name>]`

Each table entry maps a model name (exactly as it will appear in API requests) to an endpoint key (`"anthropic"` or `"ollama"`).

```toml
[models."claude-sonnet-4-6"]
endpoint = "anthropic"

[models."claude-opus-4-6"]
endpoint = "anthropic"

[models."claude-haiku-4-5"]
endpoint = "anthropic"

[models."Qwen3.5:9b"]
endpoint = "ollama"
```

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable               | Default       | Description                                                                                 |
|------------------------|---------------|---------------------------------------------------------------------------------------------|
| `HOST`                 | `127.0.0.1`   | Address the proxy server binds to                                                           |
| `PORT`                 | `8000`        | Port the proxy server listens on                                                            |
| `LOG_LEVEL`            | `INFO`        | Python logging level (`DEBUG`, `INFO`, `WARNING`, …)                                        |
| `ANTHROPIC_AUTH_TOKEN` | _(unset)_     | Optional fallback Anthropic bearer token. Injected as `Authorization: Bearer <token>` when the client sends no auth headers. |
| `ANTHROPIC_API_KEY`    | _(unset)_     | Optional fallback Anthropic API key. Injected as `x-api-key: <key>` when the client sends no auth headers and `ANTHROPIC_AUTH_TOKEN` is also unset. |

Auth header priority for Anthropic requests: incoming `x-api-key` > incoming `Authorization` > `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_API_KEY`.

The `.env` file is loaded automatically at startup via `python-dotenv`.

## Running

### Development

```bash
# Using the installed entry point
uv run llm-proxy

# Or directly
uv run python server.py
```

### As a Background Service

```bash
# macOS (launchd) or Linux (systemd user service)
./scripts/start-service.sh

# Stop the service
./scripts/stop-service.sh

# Windows (PowerShell)
.\scripts\start-service.ps1

# Stop on Windows
.\scripts\stop-service.ps1
```

## Usage Examples

Replace `12345` with the `PORT` value from your `.env`.

### Anthropic model — non-streaming

```bash
curl http://127.0.0.1:12345/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Anthropic model — streaming

```bash
curl http://127.0.0.1:12345/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  --no-buffer \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a joke."}]
  }'
```

### Ollama model

```bash
curl http://127.0.0.1:12345/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5:9b",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is 2 + 2?"}]
  }'
```

## Log Summarization

The proxy can log every request/response to a JSONL file by setting the `OUTPUT_FILE` environment variable in `.env`:

```
OUTPUT_FILE=/path/to/proxy_output.jsonl
```

To summarize the logged conversations using a local Ollama model:

1. Edit `summarizer.toml` to set the model name, Ollama URL, and file paths:

```toml
model      = "llama3.2"
ollama_url = "http://127.0.0.1:11434"
input_file = "proxy_output.jsonl"
output_file = "summaries.md"
```

2. Run the summarizer:

```bash
uv run python summarize_logs.py
```

All CLI options override the config file:

```bash
uv run python summarize_logs.py --model qwen3.5:9b --input-file /path/to/proxy_output.jsonl --output-file /path/to/summaries.md
```

The output is a Markdown file with one entry per logged exchange — metadata (timestamp, model, token counts) plus a 2–3 sentence summary of the conversation.

## Token Usage Stats

To print input/output token totals per model from the JSONL log:

```bash
python token_stats.py /path/to/proxy_output.jsonl
# or
OUTPUT_FILE=/path/to/proxy_output.jsonl python token_stats.py
```

Example output:

```
Model                    Requests    Input tokens   Output tokens
--------------------------------------------------------------------
claude-sonnet-4-6              12          45,231           8,102
Qwen3.5:9b                      5           9,800           2,300
--------------------------------------------------------------------
TOTAL                          17          55,031          10,402
```

## Adding a New Model

Open `proxy.config` and add a new `[models."<model-name>"]` table with an `endpoint` key set to either `"anthropic"` or `"ollama"`. The model name must match exactly what you will pass in the `"model"` field of your API requests. For an Ollama model, make sure the model has been pulled (`ollama pull <model-name>`) before starting the proxy — the startup checks will warn you if it has not been. No code changes are needed; the proxy reads the config file on every startup.

## Adding a New Endpoint Type

Supporting a backend other than Anthropic or Ollama requires implementing a new handler class (analogous to `handlers/anthropic.py` and `handlers/ollama.py`) that translates incoming Anthropic-format request bodies to the target API's format and converts responses back. Register the new handler in `ProxyRouter.__init__` inside `router.py`, then reference it by name in `proxy.config` entries.
