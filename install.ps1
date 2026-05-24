<#
.SYNOPSIS
    Install Nine Men's Morris (NMM) on Windows.
.DESCRIPTION
    Creates a Python virtual environment, installs dependencies, and
    optionally installs Ollama for LLM commentary features.
.PARAMETER NoOllama
    Skip Ollama installation and model download entirely.
.PARAMETER Model
    Override the Ollama model to pull (default: read from data\settings.json,
    fallback to llama3.1:8b).
.EXAMPLE
    .\install.ps1
    .\install.ps1 -NoOllama
    .\install.ps1 -Model "mistral:7b"
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$NoOllama,
    [string]$Model = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$NMM_DIR  = $PSScriptRoot
$VENV_DIR = Join-Path $NMM_DIR ".venv"
$VENV_PY  = Join-Path $VENV_DIR "Scripts\python.exe"
$VENV_PIP = Join-Path $VENV_DIR "Scripts\pip.exe"

function Write-Info { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[NMM] ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host "  |   Nine Men's Morris -- Installer     |" -ForegroundColor Cyan
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host ""

# ── 1. Read model from settings.json ─────────────────────────────────────────
$SETTINGS = Join-Path $NMM_DIR "data\settings.json"
if ($Model -eq "" -and (Test-Path $SETTINGS)) {
    try {
        $s = Get-Content $SETTINGS -Raw | ConvertFrom-Json
        if ($s.ollama_model) { $Model = $s.ollama_model }
    } catch {}
}
if ($Model -eq "") { $Model = "llama3.1:8b" }

# ── 2. Python 3.10+ ───────────────────────────────────────────────────────────
Write-Info "Checking Python..."
$pythonCmd = $null
foreach ($cmd in @("python", "py", "python3")) {
    try {
        $ver = & $cmd --version 2>&1
        if ("$ver" -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $pythonCmd = $cmd
                Write-Info "Python $major.$minor found via '$cmd' -- OK"
                break
            } else {
                Write-Warn "'$cmd' is Python $major.$minor (3.10+ required), trying next..."
            }
        }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Fail "Python 3.10+ not found. Download from https://python.org and add it to PATH."
}

# ── 3. Virtual environment ────────────────────────────────────────────────────
if (Test-Path $VENV_PY) {
    Write-Info "Existing .venv found -- skipping creation."
} else {
    Write-Info "Creating virtual environment in .venv ..."
    & $pythonCmd -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment." }
}

Write-Info "Upgrading pip..."
& $VENV_PIP install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Fail "pip upgrade failed." }

# ── 4. Python dependencies ────────────────────────────────────────────────────
Write-Info "Installing Python requirements..."
& $VENV_PIP install --quiet -r (Join-Path $NMM_DIR "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install Python requirements." }
Write-Info "Python packages installed."

# ── 5. Ollama (optional) ──────────────────────────────────────────────────────
$installOllama = $false

if ($NoOllama) {
    Write-Info "Skipping Ollama (-NoOllama flag set). LLM features will be disabled."
} else {
    Write-Host ""
    Write-Host "  Ollama enables AI commentary and LLM-powered move analysis." -ForegroundColor White
    Write-Host "  The default model ($Model) is ~5 GB and requires a download." -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "[NMM] Install Ollama for LLM features? [Y/n]"
    if ($choice -eq "" -or $choice -match "^[Yy]") {
        $installOllama = $true
    } else {
        Write-Info "Skipping Ollama. Run install.ps1 again (without -NoOllama) to add it later."
    }
}

if ($installOllama) {
    # Check if already installed
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaExe) {
        $ollamaVer = (& ollama --version 2>&1 | Select-Object -First 1)
        Write-Info "Ollama already installed ($ollamaVer)."
    } else {
        Write-Info "Downloading Ollama installer..."
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        try {
            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
                -OutFile $installer -UseBasicParsing
        } catch {
            Write-Fail "Could not download Ollama installer. Download manually from https://ollama.com and re-run."
        }

        Write-Info "Running Ollama installer (follow the prompts)..."
        Start-Process -FilePath $installer -Wait
        Remove-Item $installer -Force -ErrorAction SilentlyContinue

        # Refresh PATH so ollama is visible in this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            Write-Warn "Ollama not found in PATH after install."
            Write-Warn "Restart your terminal, then run:  ollama pull $Model"
            $installOllama = $false
        } else {
            Write-Info "Ollama installed."
        }
    }
}

if ($installOllama) {
    # ── 6. Start Ollama service if not running ────────────────────────────────
    $ollamaReady = $false
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
            -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
        $ollamaReady = $true
        Write-Info "Ollama service already running."
    } catch {}

    if (-not $ollamaReady) {
        Write-Info "Starting Ollama service..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        $waited = 0
        while (-not $ollamaReady -and $waited -lt 15) {
            Start-Sleep -Seconds 1
            $waited++
            try {
                Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
                    -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null
                $ollamaReady = $true
            } catch {}
        }
        if ($ollamaReady) {
            Write-Info "Ollama service started."
        } else {
            Write-Warn "Ollama service did not respond in time. Start it manually: ollama serve"
        }
    }

    # ── 7. Pull LLM model ─────────────────────────────────────────────────────
    if ($ollamaReady) {
        Write-Info "Checking for model '$Model'..."
        & ollama show $Model 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Model '$Model' already present."
        } else {
            Write-Info "Pulling '$Model' -- this may take several minutes..."
            & ollama pull $Model
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Model pull failed. Run manually later: ollama pull $Model"
            } else {
                Write-Info "Model '$Model' ready."
            }
        }
    }
}

# ── 8. Create data directories ────────────────────────────────────────────────
foreach ($d in @("data\games", "data\session_memory", "data\chroma")) {
    $path = Join-Path $NMM_DIR $d
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}
Write-Info "Data directories ready."

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the game:" -ForegroundColor White
Write-Host "    .\run_nmm.bat" -ForegroundColor Cyan
Write-Host ""
