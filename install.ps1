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
    fallback to qwen3:4b-instruct).
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
    [switch]$Yes,
    [switch]$Repair
)

# NOTE: Set-StrictMode is deliberately NOT enabled here.
# Strict mode interacts badly with $LASTEXITCODE inspection after native
# commands and with optional JSON properties, producing confusing failures.
$ErrorActionPreference = "Continue"

$NMM_DIR  = $PSScriptRoot
if (-not $NMM_DIR) { $NMM_DIR = (Get-Location).Path }

$VENV_DIR  = Join-Path $NMM_DIR ".venv"
$VENV_PY   = Join-Path $VENV_DIR "Scripts\python.exe"

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
if ($Model -eq "") { $Model = "qwen3:4b-instruct" }

# === 2. Repair mode: delete .venv and reinstall ==============================
if ($Repair) {
    Write-Info "Repair mode: removing .venv (data\ and runtime\ are preserved)."
    if (Test-Path -LiteralPath $VENV_DIR) {
        Remove-Item -Recurse -Force -LiteralPath $VENV_DIR
    }
}

# === 3. Bootstrap private Python 3.12 via uv =================================
Write-Info "Ensuring private Python 3.12 ..."
$bootstrapScript = Join-Path $NMM_DIR "packaging\bootstrap_python.ps1"
if (-not (Test-Path -LiteralPath $bootstrapScript)) {
    Write-Fail "Cannot find packaging\bootstrap_python.ps1"
}
# Dot-source so $UV_CMD, $UV_PYTHON_INSTALL_DIR, $NMM_PYTHON are set here.
. $bootstrapScript
if (-not $UV_CMD -or -not $NMM_PYTHON) {
    Write-Fail "bootstrap_python.ps1 did not set required variables."
}

# === 4. Create virtual environment ===========================================
if (Test-Path -LiteralPath $VENV_PY) {
    $venvVer = & $VENV_PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    if ($venvVer -like "3.12*") {
        Write-Info "Virtual environment already exists (Python $venvVer)."
    } else {
        Write-Info "Virtual environment is Python $venvVer; recreating with 3.12."
        Remove-Item -Recurse -Force -LiteralPath $VENV_DIR
        & $UV_CMD venv --python $NMM_PYTHON "$VENV_DIR"
        if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment." }
    }
} else {
    Write-Info "Creating virtual environment ..."
    & $UV_CMD venv --python $NMM_PYTHON "$VENV_DIR"
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment." }
}

# === 5. Install Python requirements from locked files ========================
$PLAYER_LOCK = Join-Path $NMM_DIR "packaging\requirements-player.lock"
$TORCH_LOCK  = Join-Path $NMM_DIR "packaging\requirements-torch-cpu.lock"
if (-not (Test-Path -LiteralPath $PLAYER_LOCK)) {
    Write-Fail "requirements-player.lock not found at $PLAYER_LOCK"
}
if (-not (Test-Path -LiteralPath $TORCH_LOCK)) {
    Write-Fail "requirements-torch-cpu.lock not found at $TORCH_LOCK"
}

Write-Info "Installing Python requirements (may take a few minutes on first run) ..."
& $UV_CMD pip install --python $VENV_PY --require-hashes -r "$PLAYER_LOCK"
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install Python requirements." }

Write-Info "Installing CPU torch ..."
& $UV_CMD pip install --python $VENV_PY `
    --index-url https://download.pytorch.org/whl/cpu `
    --extra-index-url https://pypi.org/simple `
    --index-strategy unsafe-best-match `
    --require-hashes `
    -r "$TORCH_LOCK"
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install torch." }

Write-Info "Requirements installed."

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

# === Desktop shortcut ========================================================
$createShortcut = $false
$shortcutScript = Join-Path $NMM_DIR "tools\create_shortcut.ps1"

if ($Yes) {
    $createShortcut = $true
} elseif (Test-Path -LiteralPath $shortcutScript) {
    Write-Host ""
    $scChoice = Read-Host "  Create a Desktop shortcut? [Y/n]"
    if ($scChoice -eq "" -or $scChoice -match "^[Yy]") { $createShortcut = $true }
}

if ($createShortcut -and (Test-Path -LiteralPath $shortcutScript)) {
    try {
        & $shortcutScript
    } catch {
        Write-Warn "Shortcut creation failed: $($_.Exception.Message). Re-run: .\tools\create_shortcut.ps1"
    }
}

# === Done ===================================================================
Write-Host ""
Write-Host "  $P_DONE" -ForegroundColor Green
Write-Host ""
Write-Host "  $P_START_HINT" -ForegroundColor White
Write-Host $P_START_CMD -ForegroundColor Cyan
Write-Host ""
