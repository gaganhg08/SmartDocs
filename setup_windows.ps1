<#
.SYNOPSIS
    SmartDocs Automated Windows Setup Script
.DESCRIPTION
    Automates environment verification, virtual environment creation,
    dependency installation, environment file initialization, Ollama verification,
    and self-test execution for Windows users.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

function Write-Step {
    param([string]$Message)
    Write-Host "`n====================================================================" -ForegroundColor Cyan
    Write-Host ">>> $Message" -ForegroundColor Cyan
    Write-Host "====================================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-WarningMsg {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host @"
====================================================================
           SmartDocs - Automated Windows Environment Setup
====================================================================
"@ -ForegroundColor Magenta

# ------------------------------------------------------------------
# Step 1: Verify Python Installation and Version
# ------------------------------------------------------------------
Write-Step "Step 1: Checking Python Installation"

$PythonCmd = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PythonCmd = "py"
}

if (-not $PythonCmd) {
    Write-ErrorMsg "Python was not found in your system PATH."
    Write-Host "Please download and install Python 3.11 or 3.12 (64-bit) from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/windows/" -ForegroundColor White
    Write-Host "IMPORTANT: During installation, check the box 'Add python.exe to PATH'." -ForegroundColor Yellow
    exit 1
}

$PyVersionRaw = & $PythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$PyMajor = & $PythonCmd -c "import sys; print(sys.version_info.major)"
$PyMinor = & $PythonCmd -c "import sys; print(sys.version_info.minor)"
$Is64Bit = & $PythonCmd -c "import sys; print(sys.maxsize > 2**32)"

Write-Host "Detected Python command: $PythonCmd"
Write-Host "Detected Python version: $PyVersionRaw ($('64-bit' if $Is64Bit -eq 'True' else '32-bit'))"

if ($Is64Bit -ne "True") {
    Write-ErrorMsg "32-bit Python detected. SmartDocs requires 64-bit Python for FAISS and PyTorch."
    Write-Host "Please install 64-bit Python from https://www.python.org/downloads/windows/" -ForegroundColor Yellow
    exit 1
}

if ([int]$PyMajor -ne 3 -or [int]$PyMinor -lt 10) {
    Write-ErrorMsg "Unsupported Python version: $PyVersionRaw. SmartDocs requires Python 3.10 to 3.13."
    Write-Host "We recommend Python 3.11 or 3.12 for maximum compatibility." -ForegroundColor Yellow
    exit 1
}

Write-Success "Python $PyVersionRaw (64-bit) is supported."

# ------------------------------------------------------------------
# Step 2: Create Python Virtual Environment (.venv)
# ------------------------------------------------------------------
Write-Step "Step 2: Configuring Python Virtual Environment (.venv)"

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvPath..."
    & $PythonCmd -m venv $VenvPath
    if (-not (Test-Path $VenvPython)) {
        Write-ErrorMsg "Failed to create virtual environment."
        exit 1
    }
    Write-Success "Virtual environment created successfully."
} else {
    Write-Host "Virtual environment already exists at $VenvPath."
}

# ------------------------------------------------------------------
# Step 3: Upgrade pip and Install Dependencies
# ------------------------------------------------------------------
Write-Step "Step 3: Installing Python Dependencies"

Write-Host "Upgrading pip inside virtual environment..."
& $VenvPython -m pip install --upgrade pip --quiet

Write-Host "Installing packages from requirements.txt (this may take a couple minutes on first run)..."
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

if ($LASTEXITCODE -ne 0) {
    Write-ErrorMsg "Package installation encountered an error."
    Write-Host "Common troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Ensure Microsoft Visual C++ 2015-2022 Redistributable (x64) is installed:" -ForegroundColor White
    Write-Host "   https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor White
    Write-Host "2. Try running: .\.venv\Scripts\python.exe -m pip install -r requirements.txt --verbose" -ForegroundColor White
    exit 1
}
Write-Success "All dependencies installed successfully."

# ------------------------------------------------------------------
# Step 4: Configure Environment Variables (.env)
# ------------------------------------------------------------------
Write-Step "Step 4: Checking Environment Configuration"

$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
        Write-Success "Created .env configuration file from .env.example."
    } else {
        Write-WarningMsg ".env.example not found; skipping .env creation."
    }
} else {
    Write-Host ".env configuration file already exists."
}

# ------------------------------------------------------------------
# Step 5: Check Ollama Local LLM Server
# ------------------------------------------------------------------
Write-Step "Step 5: Checking Ollama Service"

$OllamaRunning = $false
try {
    $Response = Invoke-RestMethod -Uri "http://localhost:11434" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $OllamaRunning = $true
} catch {
    $OllamaRunning = $false
}

if ($OllamaRunning) {
    Write-Success "Ollama server is active and reachable at http://localhost:11434."
    
    # Check if llama3.2 is downloaded
    try {
        $Tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
        $Models = $Tags.models | ForEach-Object { $_.name }
        $HasModel = $Models | Where-Object { $_ -like "llama3.2*" }
        if ($HasModel) {
            Write-Success "Recommended model (llama3.2) is already installed in Ollama."
        } else {
            Write-WarningMsg "Model 'llama3.2' not found in Ollama."
            Write-Host "To download the recommended model, open a terminal and run:" -ForegroundColor Yellow
            Write-Host "   ollama pull llama3.2" -ForegroundColor White
        }
    } catch {
        Write-WarningMsg "Could not query Ollama models list."
    }
} else {
    Write-WarningMsg "Ollama is not currently running at http://localhost:11434."
    Write-Host "How to install and start Ollama:" -ForegroundColor Yellow
    Write-Host "  1. Download the Windows installer from: https://ollama.com/download/windows" -ForegroundColor White
    Write-Host "  2. Install and launch Ollama." -ForegroundColor White
    Write-Host "  3. Open PowerShell and run: ollama pull llama3.2" -ForegroundColor White
}

# ------------------------------------------------------------------
# Step 6: Run Self-Tests
# ------------------------------------------------------------------
Write-Step "Step 6: Running Automated Verification Tests"

Write-Host "Executing pytest suite..."
& $VenvPython -m pytest tests -q

if ($LASTEXITCODE -eq 0) {
    Write-Success "All automated tests passed successfully!"
} else {
    Write-WarningMsg "Some tests did not pass. Inspect the test output above."
}

# ------------------------------------------------------------------
# Summary and Next Steps
# ------------------------------------------------------------------
Write-Host @"

====================================================================
🎉 SmartDocs Setup Complete!
====================================================================

To launch SmartDocs:

  Option A (Quick Script):
    powershell -ExecutionPolicy Bypass -File .\run_app.ps1

  Option B (Manual Activation):
    .\.venv\Scripts\Activate.ps1
    streamlit run app.py

Then open your browser at: http://localhost:8501

To stop SmartDocs at any time, press Ctrl + C in the PowerShell window.
====================================================================
"@ -ForegroundColor Green
