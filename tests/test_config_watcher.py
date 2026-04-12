"""
Unit tests for ConfigWatcher (hot-swap) and config parsing (fallback URL).

Sections:
  1. ConfigWatcher tests
     a. Detects file change and calls apply_config with new config
     b. Does NOT reload on unchanged file
     c. Keeps old config on parse error (invalid TOML)
     d. Keeps old config when override_model is invalid

  2. Config parsing tests (ollama_fallback_url)
     a. ollama_fallback_url is parsed from config
     b. Validation: fallback URL without primary URL raises RuntimeError
     c. Trailing slash is stripped from ollama_fallback_url
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_watcher import ConfigWatcher
from config import ProxyConfig, ServerConfig, ModelConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TOML = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://127.0.0.1:11434"

[models."mymodel"]
endpoint = "ollama"
"""

_BASE_TOML_CHANGED = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://127.0.0.1:11434"

[models."mymodel"]
endpoint = "ollama"

[models."anothermodel"]
endpoint = "ollama"
"""

_INVALID_TOML = "this is not [valid toml ]["

_TOML_WITH_INVALID_OVERRIDE = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://127.0.0.1:11434"

[models."mymodel"]
endpoint = "ollama"

[server]
override_model = "nonexistent"
"""


def _write_temp_config(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return f.name


def _make_mock_router():
    """Return a mock router with an async apply_config."""
    router = MagicMock()
    router.apply_config = AsyncMock(return_value=None)
    return router


def _make_proxy_config() -> ProxyConfig:
    return ProxyConfig(
        anthropic_url="https://api.anthropic.com",
        ollama_url="http://127.0.0.1:11434",
        models={"mymodel": ModelConfig(endpoint="ollama")},
    )


# ---------------------------------------------------------------------------
# 1a. ConfigWatcher detects file change and calls apply_config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watcher_detects_change_and_calls_apply_config():
    """
    Write a config file, create a ConfigWatcher, modify the file, run one poll
    iteration, and verify router.apply_config was called with the new config.
    """
    path = _write_temp_config(_BASE_TOML)
    try:
        router = _make_mock_router()
        http_client = MagicMock()
        apply_env = MagicMock()  # no-op env override

        watcher = ConfigWatcher(
            path=path,
            router=router,
            http_client=http_client,
            apply_env_overrides=apply_env,
            interval_s=0.05,
        )

        # Seed the initial stat
        initial_stat = os.stat(path)
        watcher._last_stat = initial_stat

        # Modify the file to trigger change detection
        time.sleep(0.01)  # ensure mtime differs
        with open(path, "w", encoding="utf-8") as f:
            f.write(_BASE_TOML_CHANGED)
        # Touch again to ensure mtime_ns differs on fast filesystems
        os.utime(path, None)

        new_stat = os.stat(path)

        # Directly call _reload instead of running the full polling loop
        await watcher._reload(new_stat)

        router.apply_config.assert_called_once()
        new_cfg: ProxyConfig = router.apply_config.call_args[0][0]
        assert "anothermodel" in new_cfg.models, (
            f"Expected 'anothermodel' in new config models, got: {list(new_cfg.models.keys())}"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1b. ConfigWatcher does NOT reload on unchanged file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watcher_no_reload_on_unchanged_file():
    """
    When the file has not changed between two ticks, apply_config must not be called.
    """
    path = _write_temp_config(_BASE_TOML)
    try:
        router = _make_mock_router()
        http_client = MagicMock()
        apply_env = MagicMock()

        watcher = ConfigWatcher(
            path=path,
            router=router,
            http_client=http_client,
            apply_env_overrides=apply_env,
            interval_s=0.05,
        )

        stat = os.stat(path)
        watcher._last_stat = stat  # seed with current stat

        # Run poll loop for a brief time: no file changes expected
        async def _run_briefly():
            task = asyncio.create_task(watcher.run())
            await asyncio.sleep(0.12)  # 2+ intervals
            watcher.stop()
            await task

        await _run_briefly()

        router.apply_config.assert_not_called()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1c. ConfigWatcher keeps old config on parse error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watcher_keeps_old_config_on_parse_error():
    """
    Writing invalid TOML to the file must NOT cause apply_config to be called.
    The watcher logs an error and keeps the previous config.
    """
    path = _write_temp_config(_INVALID_TOML)
    try:
        router = _make_mock_router()
        http_client = MagicMock()
        apply_env = MagicMock()

        watcher = ConfigWatcher(
            path=path,
            router=router,
            http_client=http_client,
            apply_env_overrides=apply_env,
            interval_s=0.05,
        )
        # _last_stat=None so watcher sees it as "changed" on first poll
        watcher._last_stat = None

        new_stat = os.stat(path)
        await watcher._reload(new_stat)

        router.apply_config.assert_not_called()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1d. ConfigWatcher keeps old config when override_model is invalid
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watcher_keeps_old_config_when_override_model_invalid():
    """
    A config with override_model referencing a non-existent model must NOT
    cause apply_config to be called.
    """
    path = _write_temp_config(_TOML_WITH_INVALID_OVERRIDE)
    try:
        router = _make_mock_router()
        http_client = MagicMock()
        apply_env = MagicMock()

        watcher = ConfigWatcher(
            path=path,
            router=router,
            http_client=http_client,
            apply_env_overrides=apply_env,
            interval_s=0.05,
        )
        watcher._last_stat = None

        new_stat = os.stat(path)
        await watcher._reload(new_stat)

        router.apply_config.assert_not_called()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2a. ollama_fallback_url is parsed from config
# ---------------------------------------------------------------------------

def test_ollama_fallback_url_parsed():
    """
    A TOML with ollama_fallback_url set must result in
    ProxyConfig.ollama_fallback_url == that URL.
    """
    toml = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://primary:11434"
ollama_fallback_url = "http://fallback:11434"

[models."m"]
endpoint = "ollama"
"""
    path = _write_temp_config(toml)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    assert cfg.ollama_fallback_url == "http://fallback:11434"


# ---------------------------------------------------------------------------
# 2b. Validation: fallback URL without primary URL raises
# ---------------------------------------------------------------------------

def test_fallback_without_primary_raises():
    """
    Setting ollama_fallback_url without ollama_url must raise RuntimeError.
    """
    toml = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_fallback_url = "http://fallback:11434"
"""
    path = _write_temp_config(toml)
    try:
        with pytest.raises(RuntimeError, match="ollama_fallback_url"):
            load_config(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2c. Trailing slash is stripped from ollama_fallback_url
# ---------------------------------------------------------------------------

def test_fallback_url_trailing_slash_stripped():
    """
    A trailing slash in ollama_fallback_url must be stripped from
    ProxyConfig.ollama_fallback_url.
    """
    toml = """\
[endpoints]
anthropic_url = "https://api.anthropic.com"
ollama_url = "http://primary:11434"
ollama_fallback_url = "http://fallback:11434/"

[models."m"]
endpoint = "ollama"
"""
    path = _write_temp_config(toml)
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    assert cfg.ollama_fallback_url == "http://fallback:11434", (
        f"Expected trailing slash stripped, got: {cfg.ollama_fallback_url!r}"
    )
