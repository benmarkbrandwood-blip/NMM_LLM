# scripts/build_rust.ps1 — Optional: build the Rust acceleration core (nmm_core).
#
# NOT required to run the game. The Python engine works standalone and falls
# back transparently when nmm_core is absent (see ai/native_core.py). Building
# this extension accelerates the hot-path engine primitives + search.
#
# Usage (from project root):
#   ./scripts/build_rust.ps1
#
# Non-fatal: if the Rust toolchain is missing, prints instructions and exits 0
# so installers can call it without aborting setup.

$ErrorActionPreference = "Continue"

$NmmDir   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CrateDir = Join-Path $NmmDir "native\nmm_core"
$VenvDir  = Join-Path $NmmDir ".venv"

function Info($m) { Write-Host "[rust] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[rust] $m" -ForegroundColor Yellow }

# Pick interpreter / pip (prefer project venv)
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvPy) {
    $Py = $VenvPy
    Info "Using project venv: $VenvDir"
} else {
    $Py = "python"
    Warn "No .venv found - using system interpreter (python)."
}

# Rust toolchain check (non-fatal)
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) {
    Warn "Rust toolchain not found. The game runs fine WITHOUT it (Python fallback)."
    Warn "To enable acceleration, install Rust from https://rustup.rs and re-run this script."
    exit 0
}
Info "cargo found: $(cargo --version)"

# maturin
& $Py -c "import maturin" 2>$null
if ($LASTEXITCODE -ne 0) {
    Info "Installing maturin build backend..."
    & $Py -m pip install "maturin>=1.4,<2.0"
    if ($LASTEXITCODE -ne 0) { Warn "maturin install failed; skipping Rust build."; exit 0 }
}

# Build + install
Push-Location $CrateDir
try {
    if (Test-Path $VenvPy) {
        $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
        if (Test-Path $activate) { . $activate }
        Info "Building + installing nmm_core (maturin develop --release)..."
        maturin develop --release
        if ($LASTEXITCODE -ne 0) {
            Warn "maturin develop failed; trying wheel build + pip install."
            maturin build --release
            & $Py -m pip install --force-reinstall (Get-ChildItem target\wheels\nmm_core-*.whl | Select-Object -First 1).FullName
        }
    } else {
        Info "Building wheel (no venv) and pip-installing..."
        maturin build --release
        if ($LASTEXITCODE -eq 0) {
            & $Py -m pip install --force-reinstall (Get-ChildItem target\wheels\nmm_core-*.whl | Select-Object -First 1).FullName
        } else {
            Warn "Rust build failed; Python fallback remains active."; exit 0
        }
    }
} finally {
    Pop-Location
}

# Verify
& $Py -c "import nmm_core" 2>$null
if ($LASTEXITCODE -eq 0) {
    Info "Verified: 'import nmm_core' works. Acceleration enabled."
} else {
    Warn "nmm_core not importable after build; the game will use the Python fallback."
}
