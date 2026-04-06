# stop-service.ps1 — stop the background LLM Proxy process on Windows.
# Reads the PID written by start-service.ps1 and terminates the process.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PidFile = Join-Path $env:APPDATA "llm-proxy\llm-proxy.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "PID file not found at $PidFile — is the service running?" -ForegroundColor Yellow
    exit 1
}

$savedPid = Get-Content $PidFile -ErrorAction SilentlyContinue
if (-not $savedPid) {
    Write-Host "PID file is empty. Removing stale file." -ForegroundColor Yellow
    Remove-Item $PidFile -Force
    exit 1
}

$targetPid = [int]$savedPid
$proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "No process found with PID $targetPid. It may have already stopped." -ForegroundColor Yellow
    Remove-Item $PidFile -Force
    exit 0
}

Write-Host "Stopping LLM Proxy (PID $targetPid)..."
Stop-Process -Id $targetPid -Force

# Wait briefly and confirm.
Start-Sleep -Milliseconds 500
$check = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if ($check) {
    Write-Host "Process $targetPid did not exit cleanly. Try stopping it manually." -ForegroundColor Red
    exit 1
}

Remove-Item $PidFile -Force
Write-Host "LLM Proxy stopped."
