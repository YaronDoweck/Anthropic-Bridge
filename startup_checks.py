from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time

import httpx

from config import ProxyConfig

logger = logging.getLogger("proxy.startup")


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _normalize_model_name(name: str) -> str:
    """Append ':latest' if the name contains no ':' separator."""
    return name if ":" in name else f"{name}:latest"


def _check_ollama_server(ollama_url: str, timeout: float = 3.0) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception as exc:
        logger.debug("Ollama server not reachable at %s: %s", ollama_url, exc)
        return False


def _get_available_models(ollama_url: str) -> list[str]:
    """Return list of model names available in the running Ollama server."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.warning("Failed to fetch available Ollama models: %s", exc)
        return []


def _install_ollama_interactive() -> None:
    """Attempt to install Ollama, interactively guiding the user."""
    print("[startup] Attempting to install Ollama...")

    if sys.platform == "darwin":
        if shutil.which("brew") is not None:
            print("[startup] Running: brew install ollama")
            try:
                result = subprocess.run(["brew", "install", "ollama"], timeout=300)
            except subprocess.TimeoutExpired:
                raise RuntimeError("brew install ollama timed out after 300 seconds.")
            if result.returncode != 0:
                raise RuntimeError("brew install ollama failed. Install Ollama manually from https://ollama.com")
        else:
            print("[startup] Homebrew not found. Install Ollama manually by running:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("  or visit https://ollama.com for other install options.")
            raise RuntimeError("Ollama not installed and Homebrew is unavailable. Install manually from https://ollama.com")

    elif sys.platform.startswith("linux"):
        print("[startup] Running: curl -fsSL https://ollama.com/install.sh | sh")
        try:
            result = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama install script timed out after 300 seconds.")
        if result.returncode != 0:
            raise RuntimeError("Ollama install script failed. Install manually from https://ollama.com")

    else:
        print("[startup] Unsupported OS for automatic install. Install Ollama manually from https://ollama.com")
        raise RuntimeError("Automatic Ollama install is not supported on this platform. Install from https://ollama.com")


def run_startup_checks(config: ProxyConfig) -> subprocess.Popen | None:
    """Run startup validation checks for Ollama if any models use the ollama endpoint.

    Returns the Popen object for 'ollama serve' if this function started it,
    or None if Ollama was already running (or not needed).
    """

    # Step 1: collect ollama models
    ollama_models = [
        name for name, model_cfg in config.models.items()
        if model_cfg.endpoint == "ollama"
    ]

    # Step 2: no ollama models configured — nothing to check
    if not ollama_models:
        return None

    print("[startup] Checking Ollama...")
    interactive = _is_interactive()

    # Step 3 & 4: check if ollama binary is installed
    if shutil.which("ollama") is None:
        if not interactive:
            logger.error(
                "Ollama is not installed but is required by config. "
                "Install it from https://ollama.com"
            )
            raise RuntimeError(
                "Ollama is not installed but required by config. "
                "Install it from https://ollama.com"
            )

        # Interactive: ask user
        answer = input(
            "[startup] Ollama is not installed but is required by config. "
            "Install it now? [y/N]: "
        ).strip().lower()

        if answer not in ("y", "yes"):
            raise RuntimeError(
                "Ollama not installed. Remove ollama models from proxy.config or install Ollama."
            )

        _install_ollama_interactive()

        # Verify install succeeded
        if shutil.which("ollama") is None:
            raise RuntimeError(
                "Ollama was not found after install attempt. "
                "Install it manually from https://ollama.com"
            )

    print("[startup] Ollama is installed.")

    # Step 5: check if Ollama server is running
    ollama_proc: subprocess.Popen | None = None
    local_ollama_url = "http://127.0.0.1:11434"
    active_ollama_url = config.ollama_url  # tracks which URL is actually reachable

    if not _check_ollama_server(config.ollama_url):
        # Primary URL unreachable — check if a local Ollama is already running
        local_already_running = _check_ollama_server(local_ollama_url)

        if local_already_running and config.ollama_url != local_ollama_url:
            print(
                f"[startup] Primary Ollama URL ({config.ollama_url}) is unreachable, "
                "but a local Ollama instance is already running. Continuing with local instance."
            )
            active_ollama_url = local_ollama_url
        elif local_already_running:
            print("[startup] A local Ollama instance is already running on 127.0.0.1:11434.")
        elif not interactive:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            )
        else:
            print("[startup] Ollama server is not running. Starting it...")

            # Start ollama serve in background
            ollama_proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=None,  # inherit terminal output
                stderr=None,
            )

            # Wait up to 5 seconds for the server to come up
            started = False
            for _ in range(10):
                time.sleep(0.5)
                if _check_ollama_server(local_ollama_url):
                    started = True
                    break

            if not started:
                ollama_proc.terminate()
                try:
                    ollama_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ollama_proc.kill()
                raise RuntimeError(
                    "Ollama server did not start within 5 seconds. "
                    "Run 'ollama serve' manually and retry."
                )

            active_ollama_url = local_ollama_url
            print("[startup] Ollama server started.")
    else:
        print("[startup] Ollama server is running.")

    # Step 6: check configured models are available
    print("[startup] Checking configured models...")
    available_raw = _get_available_models(active_ollama_url)
    available_normalized = {_normalize_model_name(m) for m in available_raw}
    missing = [
        name for name in ollama_models
        if _normalize_model_name(name) not in available_normalized
    ]

    if not missing:
        print("[startup] All models ready.")
        return ollama_proc

    logger.warning("The following Ollama models are not pulled: %s", ", ".join(missing))

    if not interactive:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"The following Ollama models are missing: {missing_list}. "
            "Pull them manually with: " + " && ".join(f"ollama pull {m}" for m in missing)
        )

    # Interactive: ask user to pull
    missing_list = ", ".join(missing)
    print(f"[startup] WARNING: The following model(s) are not pulled: {missing_list}")
    answer = input(f"Pull missing models now? [Y/n]: ").strip().lower()

    if answer in ("n", "no"):
        logger.warning(
            "Skipping pull for missing models: %s. "
            "Requests to these models will fail at runtime.",
            missing_list,
        )
        return ollama_proc

    all_pulled = True
    for model_name in missing:
        print(f"[startup] Pulling {model_name}...")
        try:
            result = subprocess.run(["ollama", "pull", model_name], timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("Timed out pulling model '%s' after 300 seconds.", model_name)
            all_pulled = False
            continue
        if result.returncode != 0:
            logger.warning("Failed to pull model '%s'. It will fail at runtime.", model_name)
            all_pulled = False

    if all_pulled:
        print("[startup] All models ready.")
    else:
        logger.warning(
            "Some models could not be pulled. Requests to those models will fail at runtime."
        )

    return ollama_proc
