#!/usr/bin/env bash
# start-service.sh — install and start the LLM Proxy as a background service.
# Works on macOS (launchd) and Linux (systemd user service).
# Run: chmod +x scripts/start-service.sh  (once, after cloning)

set -euo pipefail

# Resolve the project root regardless of where this script is called from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OS="$(uname -s)"

# ---------------------------------------------------------------------------
# macOS — launchd
# ---------------------------------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$PLIST_DIR/com.llm-proxy.plist"
    LOG_DIR="$HOME/.llm-proxy/logs"
    ENV_FILE="$PROJECT_ROOT/.env"

    mkdir -p "$PLIST_DIR"
    mkdir -p "$LOG_DIR"

    # Build EnvironmentVariables block from .env (skip comments and blank lines)
    ENV_BLOCK=""
    if [ -f "$ENV_FILE" ]; then
        while IFS='=' read -r key value; do
            # Skip comments and blank lines
            [[ "$key" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${key// }" ]]           && continue
            # Strip inline comments from value
            value="${value%%#*}"
            # Strip surrounding whitespace
            key="${key// /}"
            value="${value#"${value%%[! ]*}"}"
            value="${value%"${value##*[! ]}"}"
            ENV_BLOCK="${ENV_BLOCK}        <key>${key}</key>
        <string>${value}</string>
"
        done < "$ENV_FILE"
    fi

    cat > "$PLIST_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.llm-proxy</string>

    <key>ProgramArguments</key>
    <array>
        <string>uv</string>
        <string>run</string>
        <string>llm-proxy</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
${ENV_BLOCK}    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>

    <key>KeepAlive</key>
    <true/>

    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
PLIST

    # Unload first in case a stale entry exists (ignore errors)
    launchctl unload -w "$PLIST_FILE" 2>/dev/null || true

    launchctl load -w "$PLIST_FILE"

    echo "LLM Proxy service started."
    echo "Logs: $LOG_DIR/"
    echo "To stop: ./scripts/stop-service.sh"

# ---------------------------------------------------------------------------
# Linux — systemd user service
# ---------------------------------------------------------------------------
elif [[ "$OS" == Linux* ]]; then
    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE_FILE="$SERVICE_DIR/llm-proxy.service"
    ENV_FILE="$PROJECT_ROOT/.env"

    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE_FILE" <<SERVICE
[Unit]
Description=LLM Proxy

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
ExecStart=uv run llm-proxy
Restart=on-failure
EnvironmentFile=${ENV_FILE}

[Install]
WantedBy=default.target
SERVICE

    systemctl --user daemon-reload
    systemctl --user enable --now llm-proxy

    echo "LLM Proxy service started."
    echo "Run: journalctl --user -u llm-proxy -f"
    echo "To stop: ./scripts/stop-service.sh"

else
    echo "Unsupported OS: $OS" >&2
    echo "Use scripts/start-service.ps1 on Windows." >&2
    exit 1
fi
