#!/usr/bin/env bash
# stop-service.sh — stop and unload the LLM Proxy background service.
# Works on macOS (launchd) and Linux (systemd user service).
# Run: chmod +x scripts/stop-service.sh  (once, after cloning)

set -euo pipefail

OS="$(uname -s)"

# ---------------------------------------------------------------------------
# macOS — launchd
# ---------------------------------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    PLIST_FILE="$HOME/Library/LaunchAgents/com.llm-proxy.plist"

    if [ ! -f "$PLIST_FILE" ]; then
        echo "No plist found at $PLIST_FILE — service may not be installed." >&2
        exit 1
    fi

    launchctl unload -w "$PLIST_FILE"
    echo "LLM Proxy service stopped and unloaded."

# ---------------------------------------------------------------------------
# Linux — systemd user service
# ---------------------------------------------------------------------------
elif [[ "$OS" == Linux* ]]; then
    if ! systemctl --user is-active --quiet llm-proxy 2>/dev/null; then
        echo "llm-proxy service is not running." >&2
    fi

    systemctl --user disable --now llm-proxy
    echo "LLM Proxy service stopped and disabled."

else
    echo "Unsupported OS: $OS" >&2
    echo "Use scripts/stop-service.ps1 on Windows." >&2
    exit 1
fi
