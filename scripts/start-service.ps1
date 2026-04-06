# start-service.ps1 — start the LLM Proxy as a background process on Windows.
# The process is detached from the current terminal; stdout/stderr are written
# to log files under $env:APPDATA\llm-proxy\logs\.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve project root (one level up from the scripts\ directory).
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

# Directories for logs and the PID file.
$AppDataDir = Join-Path $env:APPDATA "llm-proxy"
$LogDir     = Join-Path $AppDataDir "logs"
$PidFile    = Join-Path $AppDataDir "llm-proxy.pid"

# Create directories if they don't exist.
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Check whether a process from a previous run is still alive.
if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($existingPid) {
        $existingProcess = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($existingProcess) {
            Write-Host "LLM Proxy is already running (PID $existingPid)."
            Write-Host "Run .\scripts\stop-service.ps1 to stop it first."
            exit 1
        }
    }
    Remove-Item $PidFile -Force
}

$StdoutLog = Join-Path $LogDir "stdout.log"
$StderrLog = Join-Path $LogDir "stderr.log"

Write-Host "Starting LLM Proxy..."
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Logs         : $LogDir"

$proc = Start-Process `
    -FilePath "uv" `
    -ArgumentList "run", "llm-proxy" `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError  $StderrLog `
    -PassThru

# Save PID so stop-service.ps1 can find the process.
$proc.Id | Set-Content $PidFile

Write-Host ""
Write-Host "LLM Proxy started (PID $($proc.Id))."
Write-Host "Stdout log : $StdoutLog"
Write-Host "Stderr log : $StderrLog"
Write-Host "PID file   : $PidFile"
Write-Host ""
Write-Host "To stop the service run: .\scripts\stop-service.ps1"
