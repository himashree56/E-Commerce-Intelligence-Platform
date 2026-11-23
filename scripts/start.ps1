# ==============================================================================
# E-Commerce Intelligence Platform — Start Services Script (PowerShell)
# ==============================================================================
param (
    [switch]$WithDashboard,
    [int]$TimeoutSeconds = 60,
    [int]$DashboardPort = 8501
)
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "E-COMMERCE INTELLIGENCE PLATFORM — SERVICE STARTUP (POWERSHELL)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

# 1. Validate Docker Compose config
Write-Host "`n[1/3] Validating Docker Compose configuration..." -ForegroundColor Yellow
& docker compose config > $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker Compose configuration validation failed."
    exit 1
}
Write-Host "  Docker Compose configuration validated." -ForegroundColor Green

# 2. Start PostgreSQL container
Write-Host "`n[2/3] Starting PostgreSQL 16 container..." -ForegroundColor Yellow
& docker compose up -d postgres
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start postgres container."
    exit 1
}

# 3. Wait for PostgreSQL readiness
Write-Host "  Waiting for database health check (timeout: ${TimeoutSeconds}s)..." -ForegroundColor Yellow
$elapsed = 0
$healthy = $false

while ($elapsed -lt $TimeoutSeconds) {
    try {
        & $VenvPython "scripts/check_database.py" > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            $healthy = $true
            break
        }
    } catch {
        # continue polling
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $healthy) {
    Write-Error "PostgreSQL failed to become healthy within ${TimeoutSeconds} seconds."
    exit 1
}
Write-Host "  PostgreSQL 16 is ready and accepting connections." -ForegroundColor Green

# 4. Optional Dashboard Startup
if ($WithDashboard) {
    Write-Host "`n[3/3] Launching Streamlit analytics dashboard on port ${DashboardPort}..." -ForegroundColor Yellow
    & $VenvPython "scripts/run_dashboard.py" --port $DashboardPort
} else {
    Write-Host "`n[3/3] Platform services started successfully." -ForegroundColor Green
    Write-Host "  To launch dashboard, run: python scripts/run_dashboard.py --port ${DashboardPort}" -ForegroundColor Cyan
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "SERVICES OPERATIONAL [READY]" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
exit 0
