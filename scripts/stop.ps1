# ==============================================================================
# E-Commerce Intelligence Platform — Stop Services Script (PowerShell)
# ==============================================================================
param (
    [switch]$Down
)
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "E-COMMERCE INTELLIGENCE PLATFORM — SERVICE SHUTDOWN (POWERSHELL)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

if ($Down) {
    Write-Host "`nRemoving containers and network (preserving persistent database volumes)..." -ForegroundColor Yellow
    & docker compose down
} else {
    Write-Host "`nStopping PostgreSQL container gracefully..." -ForegroundColor Yellow
    & docker compose stop postgres
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "SERVICES STOPPED [CLEAN SHUTDOWN]" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
exit 0
