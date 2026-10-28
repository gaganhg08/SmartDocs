<#
.SYNOPSIS
    SmartDocs Application Launcher for Windows
.DESCRIPTION
    Activates the virtual environment and launches the Streamlit web interface.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_app.ps1
#>

[CmdletBinding()]
param()

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"

if (Test-Path $VenvStreamlit) {
    Write-Host "Launching SmartDocs via virtual environment..." -ForegroundColor Cyan
    & $VenvStreamlit run (Join-Path $ProjectRoot "app.py")
} elseif (Get-Command "streamlit" -ErrorAction SilentlyContinue) {
    Write-Host "Launching SmartDocs via system Streamlit..." -ForegroundColor Cyan
    streamlit run (Join-Path $ProjectRoot "app.py")
} else {
    Write-Host "[ERROR] Streamlit is not installed. Please run setup_windows.ps1 first:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1" -ForegroundColor Yellow
}
