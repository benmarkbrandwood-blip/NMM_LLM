# install_i18n/en.ps1 — English messages for install.ps1
# Format: plain strings; use -f operator for {0} substitution.
# UTF-8 BOM-less; loaded with `. "$NMM_DIR\install_i18n\$lang.ps1"`
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installer     |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Install directory: {0}"
$P_PATH_ERROR       = "Install path contains '[' or ']' which PowerShell handles poorly. Move the folder to a simpler path (e.g. C:\NMM) and re-run."

$P_PYTHON_SEARCH    = "Looking for Python 3.10 or newer..."
$P_PYTHON_FOUND     = "Using Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' is Python {1}.{2} -- need 3.10+; trying next."
$P_PYTHON_MISSING   = "Python 3.10 or newer was not found."
$P_PYTHON_URL       = "Install it from https://www.python.org/downloads/"
$P_PYTHON_PATH      = "IMPORTANT: tick 'Add python.exe to PATH' during install."
$P_PYTHON_FAIL      = "Python 3.10+ is required."

$P_VENV_EXISTS      = "Existing .venv found -- reusing it."
$P_VENV_CREATE      = "Creating virtual environment in .venv ..."
$P_VENV_FAIL        = "Failed to create virtual environment. Make sure the Python 'venv' module is available."

$P_PIP_UPGRADE      = "Upgrading pip / setuptools / wheel..."
$P_PIP_WARN         = "pip upgrade returned a non-zero exit code; continuing anyway."

$P_REQS_INSTALL     = "Installing Python requirements (this can take a few minutes)..."
$P_REQS_CPP         = "On Windows, chromadb sometimes needs the 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = "Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/"
$P_REQS_CPP_RERUN   = "then re-run install.bat."
$P_REQS_FAIL        = "Python dependency install failed."
$P_REQS_DONE        = "Python packages installed."

$P_OLLAMA_SKIP      = "Skipping Ollama (-NoOllama flag set). LLM features will be disabled."
$P_OLLAMA_AUTO      = "Auto-installing Ollama (-Yes flag set)."
$P_OLLAMA_PROMPT1   = "Ollama enables AI commentary and LLM-powered move analysis."
$P_OLLAMA_PROMPT2   = "The default model ({0}) is ~5 GB and requires a download."
$P_OLLAMA_YN        = "[NMM] Install Ollama for LLM features? [Y/n]"
$P_OLLAMA_SKIP2     = "Skipping Ollama. Run install.ps1 again (without -NoOllama) to add it later."
$P_OLLAMA_EXISTS    = "Ollama already installed ({0})."
$P_OLLAMA_DOWNLOAD  = "Downloading Ollama installer..."
$P_OLLAMA_DL_FAIL   = "Could not download Ollama installer automatically: {0}"
$P_OLLAMA_DL_MANUAL = "Install it manually from https://ollama.com then re-run install.bat."
$P_OLLAMA_RUN       = "Running Ollama installer (follow the prompts)..."
$P_OLLAMA_RUN_FAIL  = "Ollama installer did not complete: {0}"
$P_OLLAMA_PATH      = "Ollama not found in PATH after install."
$P_OLLAMA_PATH_HINT = "Restart your terminal, then run:  ollama pull {0}"
$P_OLLAMA_DONE      = "Ollama installed."
$P_OLLAMA_START     = "Starting Ollama service..."
$P_OLLAMA_RUNNING   = "Ollama service already running."
$P_OLLAMA_TIMEOUT   = "Ollama service did not respond in time. Start it manually: ollama serve"

$P_MODEL_CHECK      = "Checking for model '{0}'..."
$P_MODEL_EXISTS     = "Model '{0}' already present."
$P_MODEL_PULL       = "Pulling '{0}' -- this may take several minutes..."
$P_MODEL_PULL_FAIL  = "Model pull failed. Run manually later: ollama pull {0}"
$P_MODEL_READY      = "Model '{0}' ready."

$P_DIRS_READY       = "Data directories ready."
$P_SETTINGS_CREATED = "Default settings created at data\settings.json."
$P_DB_ERROR         = "Database setup encountered an error -- continue at /databases in the game."
$P_DB_MISSING       = "tools\install_dbs.py not found -- skipping database download."

$P_DONE             = "Installation complete!"
$P_START_HINT       = "To start the game:"
$P_START_CMD        = "    .\run_nmm.bat"
