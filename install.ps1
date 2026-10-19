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
.PARAMETER Yes
    Non-interactive: assume "Yes" to all prompts (installs Ollama by default).
.EXAMPLE
    .\install.ps1
    .\install.ps1 -NoOllama
    .\install.ps1 -Model "mistral:7b"
    .\install.ps1 -Yes
.NOTES
    If PowerShell refuses to run this script with "running scripts is disabled",
    either double-click install.bat (recommended) or run from PowerShell:
        powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$NoOllama,
    [string]$Model = "",
    [switch]$Yes
)

# NOTE: Set-StrictMode is deliberately NOT enabled here.
# Strict mode interacts badly with $LASTEXITCODE inspection after native
# commands and with optional JSON properties, producing confusing failures.
$ErrorActionPreference = "Continue"

$NMM_DIR  = $PSScriptRoot
if (-not $NMM_DIR) { $NMM_DIR = (Get-Location).Path }

$VENV_DIR = Join-Path $NMM_DIR ".venv"
$VENV_PY  = Join-Path $VENV_DIR "Scripts\python.exe"

function Write-Info { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[NMM] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[NMM] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ── Locale detection + message file ──────────────────────────────────────────
$_ui = [System.Globalization.CultureInfo]::CurrentUICulture.TwoLetterISOLanguageName
$_supported = @("zh","de","hu","fa","ar","ru","ro","it","fr","pl","cs","es","da","nb","sv","nl","pt")
$_lang = if ($_supported -contains $_ui) { $_ui } else { "en" }
$_i18n = Join-Path $NMM_DIR "install_i18n\$_lang.ps1"
if (-not (Test-Path -LiteralPath $_i18n)) { $_i18n = Join-Path $NMM_DIR "install_i18n\en.ps1" }
. $_i18n

Write-Host ""
Write-Host $P_TITLE_1 -ForegroundColor Cyan
Write-Host $P_TITLE_2 -ForegroundColor Cyan
Write-Host $P_TITLE_3 -ForegroundColor Cyan
Write-Host ""
Write-Info ($P_INSTALL_DIR -f $NMM_DIR)

# Refuse to run inside paths PowerShell can't handle reliably.
if ($NMM_DIR -match '[\[\]]') {
    Write-Fail $P_PATH_ERROR
}

# === 1. Read model from settings.json =======================================
$SETTINGS = Join-Path $NMM_DIR "data\settings.json"
if ($Model -eq "" -and (Test-Path -LiteralPath $SETTINGS)) {
    try {
        $raw = Get-Content -LiteralPath $SETTINGS -Raw -ErrorAction Stop
        $s = $raw | ConvertFrom-Json -ErrorAction Stop
        if ($s -and ($s.PSObject.Properties.Name -contains 'ollama_model')) {
            if ($s.ollama_model) { $Model = [string]$s.ollama_model }
        }
    } catch {
        Write-Warn "Could not parse data\settings.json -- using default model."
    }
}
if ($Model -eq "") { $Model = "llama3.1:8b" }

# === 2. Find a usable Python 3.10+ ==========================================
Write-Info $P_PYTHON_SEARCH

function Get-PythonVersion {
    param([string]$Exe, [string[]]$PreArgs = @())
    try {
        $argList = @()
        $argList += $PreArgs
        $argList += "--version"
        # Capture both streams; some shims write to stderr.
        $output = & $Exe @argList 2>&1
        if ($null -eq $output) { return $null }
        $text = ($output | Out-String)
        if ($text -match "Python\s+(\d+)\.(\d+)") {
            return [pscustomobject]@{
                Major = [int]$Matches[1]
                Minor = [int]$Matches[2]
                Raw   = $text.Trim()
            }
        }
    } catch {
        return $null
    }
    return $null
}

# Build candidate list: each entry is @{ Exe = ...; Args = @(...) }
# 'py -3' is preferred on Windows because it skips the Microsoft Store stub.
$candidates = @(
    @{ Exe = "py";      Args = @("-3") },
    @{ Exe = "py";      Args = @() },
    @{ Exe = "python";  Args = @() },
    @{ Exe = "python3"; Args = @() }
)

$pythonExe  = $null
$pythonArgs = @()

foreach ($c in $candidates) {
    $cmd = Get-Command $c.Exe -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }

    # Skip the Windows App Execution Alias stub for python.exe (it lives under
    # WindowsApps and opens the Microsoft Store instead of running Python).
    if ($c.Exe -eq "python" -and $cmd.Source -match "WindowsApps") {
        Write-Warn "Skipping Microsoft Store 'python' stub at $($cmd.Source)."
        continue
    }

    $v = Get-PythonVersion -Exe $c.Exe -PreArgs $c.Args
    if ($null -eq $v) { continue }

    if ($v.Major -gt 3 -or ($v.Major -eq 3 -and $v.Minor -ge 10)) {
        $pythonExe  = $c.Exe
        $pythonArgs = $c.Args
        $displayCmd = if ($c.Args.Count -gt 0) { "$($c.Exe) $($c.Args -join ' ')" } else { $c.Exe }
        Write-Info ($P_PYTHON_FOUND -f $v.Major, $v.Minor, $displayCmd)
        break
    } else {
        Write-Warn ($P_PYTHON_OLD -f $c.Exe, $v.Major, $v.Minor)
    }
}

if (-not $pythonExe) {
    Write-Host ""
    Write-Host "  $P_PYTHON_MISSING" -ForegroundColor Red
    Write-Host "  $P_PYTHON_URL"     -ForegroundColor White
    Write-Host "  $P_PYTHON_PATH"    -ForegroundColor White
    Write-Host ""
    Write-Fail $P_PYTHON_FAIL
}

# === 3. Create the virtual environment ======================================
if (Test-Path -LiteralPath $VENV_PY) {
    Write-Info $P_VENV_EXISTS
} else {
    Write-Info $P_VENV_CREATE
    & $pythonExe @pythonArgs -m venv "$VENV_DIR"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VENV_PY)) {
        Write-Fail $P_VENV_FAIL
    }
}

# === 4. Upgrade pip via 'python -m pip' (NEVER call pip.exe directly on Windows) ===
# Calling pip.exe to upgrade itself fails on Windows because the .exe is held
# open while it tries to overwrite itself. 'python -m pip' avoids this.
Write-Info $P_PIP_UPGRADE
& $VENV_PY -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Warn $P_PIP_WARN
}

# === 5. Install Python requirements =========================================
$REQS = Join-Path $NMM_DIR "requirements.txt"
if (-not (Test-Path -LiteralPath $REQS)) {
    Write-Fail "requirements.txt not found at $REQS"
}

Write-Info $P_REQS_INSTALL
& $VENV_PY -m pip install --disable-pip-version-check -r "$REQS"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Warn "Some packages failed to install."
    Write-Warn $P_REQS_CPP
    Write-Warn $P_REQS_CPP_URL
    Write-Warn $P_REQS_CPP_RERUN
    Write-Fail $P_REQS_FAIL
}
Write-Info $P_REQS_DONE

# === 6. Ollama (optional) ===================================================
$installOllama = $false

if ($NoOllama) {
    Write-Info $P_OLLAMA_SKIP
} elseif ($Yes) {
    $installOllama = $true
    Write-Info $P_OLLAMA_AUTO
} else {
    Write-Host ""
    Write-Host "  $P_OLLAMA_PROMPT1" -ForegroundColor White
    Write-Host "  $($P_OLLAMA_PROMPT2 -f $Model)" -ForegroundColor White
    Write-Host ""
    $choice = Read-Host $P_OLLAMA_YN
    if ($choice -eq "" -or $choice -match "^[Yy]") {
        $installOllama = $true
    } else {
        Write-Info $P_OLLAMA_SKIP2
    }
}

if ($installOllama) {
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaExe) {
        $ollamaVer = (& ollama --version 2>&1 | Select-Object -First 1)
        Write-Info ($P_OLLAMA_EXISTS -f $ollamaVer)
    } else {
        Write-Info $P_OLLAMA_DOWNLOAD
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        try {
            # Use TLS 1.2+; older Windows defaults can refuse the GitHub redirect.
            try {
                [System.Net.ServicePointManager]::SecurityProtocol = `
                    [System.Net.ServicePointManager]::SecurityProtocol -bor `
                    [System.Net.SecurityProtocolType]::Tls12
            } catch {}

            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
                -OutFile $installer -UseBasicParsing -ErrorAction Stop
        } catch {
            Write-Warn ($P_OLLAMA_DL_FAIL -f $_.Exception.Message)
            Write-Warn $P_OLLAMA_DL_MANUAL
            $installOllama = $false
        }

        if ($installOllama) {
            Write-Info $P_OLLAMA_RUN
            try {
                Start-Process -FilePath $installer -Wait -ErrorAction Stop
            } catch {
                Write-Warn ($P_OLLAMA_RUN_FAIL -f $_.Exception.Message)
                $installOllama = $false
            }
            Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue

            # Refresh PATH so 'ollama' is visible in this session.
            $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $userPath    = [System.Environment]::GetEnvironmentVariable("Path", "User")
            $env:Path = "$machinePath;$userPath"

            if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
                Write-Warn $P_OLLAMA_PATH
                Write-Warn ($P_OLLAMA_PATH_HINT -f $Model)
                $installOllama = $false
            } else {
                Write-Info $P_OLLAMA_DONE
            }
        }
    }
}

if ($installOllama) {
    # === 7. Start Ollama service if not running =============================
    $ollamaReady = $false
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" `
            -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop | Out-Null
        $ollamaReady = $true
        Write-Info $P_OLLAMA_RUNNING
    } catch {}

    if (-not $ollamaReady) {
        Write-Info $P_OLLAMA_START
        try {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
        } catch {
            Write-Warn "Could not start 'ollama serve': $($_.Exception.Message)"
        }
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
            Write-Info $P_OLLAMA_DONE
        } else {
            Write-Warn $P_OLLAMA_TIMEOUT
        }
    }

    # === 8. Pull LLM model ===================================================
    if ($ollamaReady) {
        Write-Info ($P_MODEL_CHECK -f $Model)
        & ollama show $Model 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Info ($P_MODEL_EXISTS -f $Model)
        } else {
            Write-Info ($P_MODEL_PULL -f $Model)
            & ollama pull $Model
            if ($LASTEXITCODE -ne 0) {
                Write-Warn ($P_MODEL_PULL_FAIL -f $Model)
            } else {
                Write-Info ($P_MODEL_READY -f $Model)
            }
        }
    }
}

# === 9. Create data directories =============================================
foreach ($d in @(
    "data\games", "data\ai_games", "data\human_games", "data\session_memory",
    "data\chroma", "data\players", "data\weights", "data\endgame", "data\backups",
    "data\logs", "learned_ai\sentinel\checkpoints"
)) {
    $path = Join-Path $NMM_DIR $d
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}
Write-Info $P_DIRS_READY

# === 10. Default settings ====================================================
$TEMPLATE = Join-Path $NMM_DIR "data\settings.template.json"
if (-not (Test-Path -LiteralPath $SETTINGS) -and (Test-Path -LiteralPath $TEMPLATE)) {
    Copy-Item -LiteralPath $TEMPLATE -Destination $SETTINGS
    Write-Info $P_SETTINGS_CREATED
}

# === 11. Download databases ==================================================
$installDbs = Join-Path $NMM_DIR "tools\install_dbs.py"
if (Test-Path -LiteralPath $installDbs) {
    Write-Host ""
    & $VENV_PY $installDbs $NMM_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Warn $P_DB_ERROR
    }
} else {
    Write-Warn $P_DB_MISSING
}

# === Done ===================================================================
Write-Host ""
Write-Host "  $P_DONE" -ForegroundColor Green
Write-Host ""
Write-Host "  $P_START_HINT" -ForegroundColor White
Write-Host $P_START_CMD -ForegroundColor Cyan
Write-Host ""
