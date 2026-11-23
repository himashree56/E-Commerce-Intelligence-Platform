# ==============================================================================
# E-Commerce Intelligence Platform — Run Tests Script (PowerShell)
# ==============================================================================
param (
    [switch]$VerboseOutput,
    [switch]$Coverage,
    [string]$Filter = "",
    [string]$Target = "tests/"
)
$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "E-COMMERCE INTELLIGENCE PLATFORM — TEST SUITE EXECUTION (POWERSHELL)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

$PytestArgs = @("-m", "pytest", $Target)
if ($VerboseOutput) {
    $PytestArgs += "-v"
} else {
    $PytestArgs += "-q"
}

if ($Filter -ne "") {
    $PytestArgs += @("-k", $Filter)
}

if ($Coverage) {
    $PytestArgs += @("--cov=src", "--cov-report=term-missing")
}

& $VenvPython $PytestArgs
$exitCode = $LASTEXITCODE

Write-Host "`n================================================================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "ALL TESTS PASSED SUCCESSFULLY [100% PASS RATE]" -ForegroundColor Green
} else {
    Write-Host "TEST SUITE COMPLETED WITH FAILURES (Exit Code: $exitCode)" -ForegroundColor Red
}
Write-Host "================================================================================" -ForegroundColor Cyan

exit $exitCode
