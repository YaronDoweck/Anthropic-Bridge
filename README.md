# anthropic-bridge

Claude Code is powerful on its own, but it gets dramatically more capable — and affordable — when you run multiple specialized subagents in parallel. The catch: Claude Code only supports a single model backend per session. You can point it at the Anthropic API and use Claude models, or you can configure a custom endpoint and use local models — but not both at once. Every subagent must use the same backend.

This project removes that limitation. `anthropic-bridge` is a lightweight local proxy that accepts Anthropic-format API requests and routes each one to the right backend based on the model name. Point Claude Code at the proxy instead of directly at Anthropic, then configure your subagents as you normally would — some using Claude Sonnet, others using a local Qwen or Llama model running in Ollama. The proxy handles the routing and format conversion transparently.

The practical payoff: you can assign expensive Anthropic models only to the subagents that genuinely need them — the orchestrator, the code reviewer, the final summarizer — while cheaper or fully free local models handle high-volume or lower-stakes work like log parsing, boilerplate generation, or internal tool calls. If you run a lot of agentic workloads, this can cut your API spend substantially without sacrificing quality where it matters.

It also gives you a clean place to add cross-cutting concerns. The proxy logs every request and response to JSONL, so you can audit exactly what your subagents are doing. There is a built-in summarizer that reads those logs and produces a readable summary using a local model. Token usage stats are tracked per model, so you can see at a glance where costs are accumulating.

## How It Works

```
Claude Code (or any Anthropic-compatible client)
        │
        ▼
  anthropic-bridge           ← you run this locally
        │
   reads model name
        │
   ┌────┴────┐
   ▼         ▼
Anthropic   Ollama
  API       (local)
```

1. Your client sends an Anthropic-format request with a `model` field.
2. The proxy looks up the model name in `proxy.config`.
3. If the model is mapped to `anthropic`, the request is forwarded unchanged to the Anthropic API.
4. If mapped to `ollama`, the request is converted from Anthropic format to OpenAI format, sent to the local Ollama server, and the response is converted back before being returned to the client.

Streaming is fully supported for both backends.

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- Ollama (optional — only needed if you configure any `ollama` endpoint models)

## Installation

```bash
git clone https://github.com/your-username/anthropic-bridge
cd anthropic-bridge
uv sync
cp .env.example .env   # add your Anthropic API key here
chmod +x scripts/*.sh
```

## Configuration

Server settings can also be configured in `proxy.config` under `[server]`:

| Field           | Default       | Description |
|-----------------|---------------|-------------|
| `host`          | `127.0.0.1`   | Bind address |
| `port`          | `8000`        | Listen port |
| `log_level`     | `INFO`        | Log verbosity |
| `debug_level`   | `0`           | Request/response body logging: `0`=off, `1`=summary, `2`=full JSON |
| `output_dir`    | _(disabled)_  | Directory for per-session JSONL log files |
| `override_model`| _(disabled)_  | Force all requests to a specific model (useful for testing) |

All routing is controlled by `proxy.config`. Map each model name you plan to use to either `"anthropic"` or `"ollama"`:

```toml
[server]
port = 8000

[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url    = "http://127.0.0.1:11434"

[models."claude-sonnet-4-6"]
endpoint = "anthropic"

[models."claude-haiku-4-5"]
endpoint = "anthropic"

[models."qwen3:8b"]
endpoint = "ollama"

[models."llama3.2"]
endpoint = "ollama"
```

Model names are case-sensitive and must match exactly what your client sends in the `model` field. For Claude Code subagents, set the model name in each subagent's configuration to whatever you define here.

### Using with Claude Code

Point Claude Code at the proxy instead of the Anthropic API directly. In your Claude Code settings or environment:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000
```

Now each subagent can be configured with any model name defined in `proxy.config`, and the proxy will route it to the right backend automatically.

## (Optional) Fallback API keys - Environment Variables

Copy `.env.example` to `.env` and set your credentials:

| Variable               | Default     | Description |
|------------------------|-------------|-------------|
| `ANTHROPIC_AUTH_TOKEN` | _(unset)_   | Fallback bearer token for Anthropic requests when the client sends no auth headers |
| `ANTHROPIC_API_KEY`    | _(unset)_   | Fallback API key (lower priority than `ANTHROPIC_AUTH_TOKEN`) |


## Running

### Development

```bash
uv run anthropic-bridge
```

With full request/response logging:

```bash
LOG_LEVEL=DEBUG uv run anthropic-bridge
```

### As a Background Service

```bash
# macOS (launchd) or Linux (systemd)
./scripts/start-service.sh
./scripts/stop-service.sh

# Windows (PowerShell)
.\scripts\start-service.ps1
.\scripts\stop-service.ps1
```

## Log Summarization

Enable JSONL logging by setting `output_dir` in `proxy.config`:

```toml
[server]
output_dir = "/path/to/logs"
```

Each session writes to a separate file. To summarize logged conversations using a local Ollama model, configure `summarizer.toml` and run:

```bash
uv run python summarize_logs.py
```

All options can be passed on the command line:

```bash
uv run python summarize_logs.py --model qwen3:8b --input-file session.jsonl --output-file summary.md
```

## Token Usage Stats

```bash
python token_stats.py /path/to/logs/session.jsonl
```

```
Model                    Requests    Input tokens   Output tokens
--------------------------------------------------------------------
claude-sonnet-4-6              12          45,231           8,102
qwen3:8b                        5           9,800           2,300
--------------------------------------------------------------------
TOTAL                          17          55,031          10,402
```

## Adding a New Model

Edit `proxy.config` only — no code changes needed:

```toml
[models."new-model-name"]
endpoint = "ollama"   # or "anthropic"
```

For Ollama models, pull the model first: `ollama pull <model-name>`. The proxy checks that all configured Ollama models are available at startup and prompts you to pull any that are missing.

## Adding a New Backend

1. Create `handlers/<name>.py` extending `BaseHandler`
2. Add an endpoint URL to `[endpoints]` in `proxy.config`
3. Register the handler in `ProxyRouter.__init__()` in `router.py`

## Testing

```bash
uv run pytest                          # unit + integration tests (no credentials needed)
ANTHROPIC_API_KEY=sk-... uv run pytest # include live Anthropic API tests
```
