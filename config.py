from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    debug_level: int = 0
    output_dir: str = ""
    raw_log_dir: str = ""
    override_model: str = ""


@dataclass
class ModelConfig:
    endpoint: str  # "anthropic" | "ollama" | "openai"
    claude_system_instructions: str = "passthrough"
    omit_claude_main_description: bool = True


@dataclass
class ProxyConfig:
    anthropic_url: str
    ollama_url: str = ""
    openai_url: str = ""
    models: dict[str, ModelConfig] = field(default_factory=dict)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(path: str = "proxy.config") -> ProxyConfig:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            raise RuntimeError(
                "tomli is required on Python < 3.11. Install it with: pip install tomli"
            )

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config file '{path}': {exc}") from exc

    endpoints = raw.get("endpoints", {})
    anthropic_url = endpoints.get("anthropic_url", "")
    ollama_url = endpoints.get("ollama_url", "")
    openai_url = endpoints.get("openai_url", "")

    models_raw = raw.get("models", {})
    models: dict[str, ModelConfig] = {}
    for model_name, model_data in models_raw.items():
        if "endpoint" not in model_data:
            raise RuntimeError(
                f"Model '{model_name}' in config is missing required 'endpoint' field"
            )
        claude_system_instructions = model_data.get("claude_system_instructions", "passthrough")
        _VALID_INSTRUCTION_MODES = ("passthrough", "strip", "split")
        if claude_system_instructions not in _VALID_INSTRUCTION_MODES:
            raise RuntimeError(
                f"Model '{model_name}': claude_system_instructions must be one of "
                f"{_VALID_INSTRUCTION_MODES}, got '{claude_system_instructions}'"
            )
        if "omit_claude_main_description" in model_data and claude_system_instructions == "passthrough":
            logger.warning(
                "Model '%s': omit_claude_main_description is set but has no effect "
                "when claude_system_instructions='passthrough'",
                model_name,
            )
        models[model_name] = ModelConfig(
            endpoint=model_data["endpoint"],
            claude_system_instructions=claude_system_instructions,
            omit_claude_main_description=model_data.get("omit_claude_main_description", True),
        )

    server = ServerConfig()
    server_raw = raw.get("server", {})
    if server_raw:
        if "host" in server_raw:
            server.host = str(server_raw["host"])
        if "port" in server_raw:
            server.port = int(server_raw["port"])
        if "log_level" in server_raw:
            log_level = str(server_raw["log_level"]).upper()
            if log_level not in _VALID_LOG_LEVELS:
                raise RuntimeError(
                    f"[server] log_level must be one of {_VALID_LOG_LEVELS}, got '{log_level}'"
                )
            server.log_level = log_level
        if "debug_level" in server_raw:
            debug_level = int(server_raw["debug_level"])
            if not (0 <= debug_level <= 3):
                raise RuntimeError(
                    f"[server] debug_level must be between 0 and 3, got {debug_level}"
                )
            server.debug_level = debug_level
        if "output_dir" in server_raw:
            server.output_dir = str(server_raw["output_dir"])
        elif "output_file" in server_raw:
            server.output_dir = str(server_raw["output_file"])
        if "raw_log_dir" in server_raw:
            server.raw_log_dir = str(server_raw["raw_log_dir"])
        if "override_model" in server_raw:
            server.override_model = str(server_raw["override_model"]).strip()

    # Validate that each model's endpoint has a corresponding non-empty URL
    _endpoint_urls = {
        "anthropic": anthropic_url,
        "ollama": ollama_url,
        "openai": openai_url,
    }
    for model_name, model_cfg in models.items():
        ep = model_cfg.endpoint
        url = _endpoint_urls.get(ep, "")
        if not url:
            raise RuntimeError(
                f"Model '{model_name}' uses endpoint '{ep}' but no URL is configured "
                f"for that endpoint. Add '{ep}_url' under [endpoints] in proxy.config."
            )

    return ProxyConfig(
        anthropic_url=anthropic_url.rstrip("/"),
        ollama_url=ollama_url.rstrip("/"),
        openai_url=openai_url.rstrip("/"),
        models=models,
        server=server,
    )
