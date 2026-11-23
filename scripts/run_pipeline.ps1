# ==============================================================================
# E-Commerce Intelligence Platform — Run Pipeline Script (PowerShell)
# ==============================================================================
param (
    [ValidateSet("full", "inference", "validate")]
    [string]$Mode = "inference",
    [switch]$SkipDb,
    [string]$Stage = "",
    [int]$TimeoutSeconds = 600
)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

$ArgsList = @("scripts/run_pipeline.py", "--mode", $Mode, "--timeout", $TimeoutSeconds)
if ($SkipDb) {
    $ArgsList += "--skip-db"
}
if ($Stage -ne "") {
    $ArgsList += @("--stage", $Stage)
}

& $VenvPython $ArgsList
exit $LASTEXITCODE
