# ==============================================================================
# E-Commerce Intelligence Platform — Windows PowerShell Setup Script
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "E-COMMERCE INTELLIGENCE PLATFORM — SYSTEM SETUP (POWERSHELL)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# 1. Check Python
Write-Host "`n[1/5] Checking Python runtime..." -ForegroundColor Yellow
try {
    $pythonVer = & python --version 2>&1
    Write-Host "  Found: $pythonVer" -ForegroundColor Green
} catch {
    Write-Error "Python executable not found in PATH. Please install Python 3.10+ (recommended 3.12)."
    exit 1
}

# 2. Check or create virtual environment
Write-Host "`n[2/5] Checking virtual environment (.venv)..." -ForegroundColor Yellow
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creating virtual environment at .venv..." -ForegroundColor Cyan
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
}
Write-Host "  Virtual environment ready: $VenvPython" -ForegroundColor Green

# 3. Install requirements
Write-Host "`n[3/5] Installing/verifying project dependencies..." -ForegroundColor Yellow
$ReqFile = Join-Path $ProjectRoot "requirements.txt"
if (Test-Path $ReqFile) {
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r $ReqFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency installation encountered an error."
        exit 1
    }
    Write-Host "  Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Warning "requirements.txt not found."
}

# 4. Initialize .env configuration
Write-Host "`n[4/5] Checking environment configuration (.env)..." -ForegroundColor Yellow
$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"
if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "  Created .env from .env.example with local defaults." -ForegroundColor Green
    } else {
        Write-Warning "Neither .env nor .env.example found."
    }
} else {
    Write-Host "  .env configuration exists." -ForegroundColor Green
}

# 5. Execute environment validator
Write-Host "`n[5/5] Running environment validator..." -ForegroundColor Yellow
& $VenvPython "src/utils/env_validator.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment validation failed."
    exit 1
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETED SUCCESSFULLY [PASS]" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
exit 0
