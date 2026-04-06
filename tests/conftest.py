"""
Shared pytest fixtures for the LLM proxy test suite.
"""

from __future__ import annotations

import socket
import sys
import os
import threading
import time

import pytest

# Ensure the proxy source root is on sys.path for all test modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "proxy.config")


@pytest.fixture(scope="session")
def proxy_config():
    """Load the real proxy.config and return a ProxyConfig object."""
    from config import load_config
    return load_config(os.path.abspath(CONFIG_PATH))


@pytest.fixture
def test_port():
    """Find and return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def running_server():
    """
    Start the proxy app with uvicorn in a background thread.
    Yields the base URL (e.g. 'http://127.0.0.1:PORT').
    Stops the server when the session ends.

    Used only by live integration tests that need a real HTTP server.
    The startup_checks are mocked out so no Ollama binary is required.
    """
    import uvicorn
    from unittest.mock import patch

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # We must import app after patching startup_checks to avoid Ollama checks
    with patch("startup_checks.run_startup_checks", return_value=None):
        # Re-import server to pick up the patched startup_checks
        import importlib
        import server as server_module
        importlib.reload(server_module)
        app = server_module.app

    uv_config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(uv_config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to be ready
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            import httpx
            with httpx.Client() as client:
                client.get(base_url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
