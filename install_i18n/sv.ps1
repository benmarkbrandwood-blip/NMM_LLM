# install_i18n/sv.ps1 — Swedish messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installera    |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Installationskatalog: {0}"
$P_PATH_ERROR       = "Installationssökvägen innehåller '[' eller ']' som PowerShell hanterar dåligt. Flytta mappen till en enklare sökväg (t.ex. C:\NMM) och kör om."

$P_PYTHON_SEARCH    = 'Söker efter Python 3.10 eller nyare...'
$P_PYTHON_FOUND     = "Använder Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' är Python {1}.{2} -- behöver 3.10+; försöker nästa."
$P_PYTHON_MISSING   = 'Python 3.10 eller nyare hittades inte.'
$P_PYTHON_URL       = 'Installera det från https://www.python.org/downloads/'
$P_PYTHON_PATH      = "VIKTIGT: kryssa i 'Add python.exe to PATH' under installationen."
$P_PYTHON_FAIL      = 'Python 3.10+ krävs.'

$P_VENV_EXISTS      = 'Befintlig .venv hittad -- återanvänds.'
$P_VENV_CREATE      = 'Skapar virtuell miljö i .venv...'
$P_VENV_FAIL        = "Det gick inte att skapa den virtuella miljön. Kontrollera att Python-modulen 'venv' är tillgänglig."

$P_PIP_UPGRADE      = 'Uppgraderar pip / setuptools / wheel...'
$P_PIP_WARN         = 'pip-uppgradering returnerade en felkod; fortsätter ändå.'

$P_REQS_INSTALL     = 'Installerar Python-krav (detta kan ta några minuter)...'
$P_REQS_CPP         = "På Windows behöver chromadb ibland 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installera från https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'kör sedan install.bat igen.'
$P_REQS_FAIL        = 'Installation av Python-beroenden misslyckades.'
$P_REQS_DONE        = 'Python-paket installerade.'

$P_OLLAMA_SKIP      = 'Hoppar över Ollama (-NoOllama-flagga satt). LLM-funktioner inaktiveras.'
$P_OLLAMA_AUTO      = 'Installerar Ollama automatiskt (-Yes-flagga satt).'
$P_OLLAMA_PROMPT1   = 'Ollama möjliggör AI-kommentarer och LLM-driven draganalys.'
$P_OLLAMA_PROMPT2   = "Standardmodellen ({0}) är ~5 GB och kräver nedladdning."
$P_OLLAMA_YN        = '[NMM] Installera Ollama för LLM-funktioner? [Y/n]'
$P_OLLAMA_SKIP2     = 'Hoppar över Ollama. Kör install.ps1 igen (utan -NoOllama) för att lägga till det senare.'
$P_OLLAMA_EXISTS    = "Ollama redan installerat ({0})."
$P_OLLAMA_DOWNLOAD  = 'Laddar ner Ollama-installationsprogram...'
$P_OLLAMA_DL_FAIL   = "Det gick inte att ladda ner Ollama-installationsprogrammet automatiskt: {0}"
$P_OLLAMA_DL_MANUAL = 'Installera det manuellt från https://ollama.com och kör sedan install.bat igen.'
$P_OLLAMA_RUN       = 'Kör Ollama-installationsprogram (följ anvisningarna)...'
$P_OLLAMA_RUN_FAIL  = "Ollama-installationsprogrammet slutfördes inte: {0}"
$P_OLLAMA_PATH      = 'Ollama hittades inte i PATH efter installationen.'
$P_OLLAMA_PATH_HINT = "Starta om terminalen och kör:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installerat.'
$P_OLLAMA_START     = 'Startar Ollama-tjänsten...'
$P_OLLAMA_RUNNING   = 'Ollama-tjänsten körs redan.'
$P_OLLAMA_TIMEOUT   = 'Ollama-tjänsten svarade inte i tid. Starta den manuellt: ollama serve'

$P_MODEL_CHECK      = "Kontrollerar modell '{0}'..."
$P_MODEL_EXISTS     = "Modell '{0}' finns redan."
$P_MODEL_PULL       = "Laddar ner '{0}' -- detta kan ta flera minuter..."
$P_MODEL_PULL_FAIL  = "Modellnedladdning misslyckades. Kör manuellt senare: ollama pull {0}"
$P_MODEL_READY      = "Modell '{0}' redo."

$P_DIRS_READY       = 'Datakataloger redo.'
$P_SETTINGS_CREATED = 'Standardinställningar skapade i data\settings.json.'
$P_DB_ERROR         = 'Databasinställningen stötte på ett fel -- fortsätt via /databases i spelet.'
$P_DB_MISSING       = 'tools\install_dbs.py hittades inte -- databasnedladdning hoppas över.'

$P_DONE             = 'Installationen klar!'
$P_START_HINT       = 'Starta spelet:'
$P_START_CMD        = '    .\run_nmm.bat'
