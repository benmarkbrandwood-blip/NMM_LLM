# install_i18n/nb.ps1 — Norwegian Bokmål messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installering  |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Installasjonsmappe: {0}"
$P_PATH_ERROR       = "Installasjonsstien inneholder '[' eller ']' som PowerShell håndterer dårlig. Flytt mappen til en enklere sti (f.eks. C:\NMM) og kjør på nytt."

$P_PYTHON_SEARCH    = 'Leter etter Python 3.10 eller nyere...'
$P_PYTHON_FOUND     = "Bruker Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' er Python {1}.{2} -- trenger 3.10+; prøver neste."
$P_PYTHON_MISSING   = 'Python 3.10 eller nyere ble ikke funnet.'
$P_PYTHON_URL       = 'Installer det fra https://www.python.org/downloads/'
$P_PYTHON_PATH      = "VIKTIG: kryss av 'Add python.exe to PATH' under installasjonen."
$P_PYTHON_FAIL      = 'Python 3.10+ er påkrevd.'

$P_VENV_EXISTS      = 'Eksisterende .venv funnet -- gjenbrukes.'
$P_VENV_CREATE      = 'Oppretter virtuelt miljø i .venv...'
$P_VENV_FAIL        = "Kunne ikke opprette virtuelt miljø. Sørg for at Python-modulen 'venv' er tilgjengelig."

$P_PIP_UPGRADE      = 'Oppgraderer pip / setuptools / wheel...'
$P_PIP_WARN         = 'pip-oppgradering returnerte en ikke-null exitkode; fortsetter uansett.'

$P_REQS_INSTALL     = 'Installerer Python-krav (dette kan ta noen minutter)...'
$P_REQS_CPP         = "På Windows trenger chromadb noen ganger 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installer fra https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'kjør deretter install.bat på nytt.'
$P_REQS_FAIL        = 'Installasjon av Python-avhengigheter mislyktes.'
$P_REQS_DONE        = 'Python-pakker installert.'

$P_OLLAMA_SKIP      = 'Hopper over Ollama (-NoOllama-flagg satt). LLM-funksjoner vil bli deaktivert.'
$P_OLLAMA_AUTO      = 'Installerer Ollama automatisk (-Yes-flagg satt).'
$P_OLLAMA_PROMPT1   = 'Ollama aktiverer AI-kommentarer og LLM-drevet trekk-analyse.'
$P_OLLAMA_PROMPT2   = "Standardmodellen ({0}) er ~5 GB og krever nedlasting."
$P_OLLAMA_YN        = '[NMM] Installer Ollama for LLM-funksjoner? [Y/n]'
$P_OLLAMA_SKIP2     = 'Hopper over Ollama. Kjør install.ps1 på nytt (uten -NoOllama) for å legge det til senere.'
$P_OLLAMA_EXISTS    = "Ollama allerede installert ({0})."
$P_OLLAMA_DOWNLOAD  = 'Laster ned Ollama-installasjonsprogram...'
$P_OLLAMA_DL_FAIL   = "Kunne ikke laste ned Ollama-installasjonsprogrammet automatisk: {0}"
$P_OLLAMA_DL_MANUAL = 'Installer det manuelt fra https://ollama.com og kjør install.bat på nytt.'
$P_OLLAMA_RUN       = 'Kjører Ollama-installasjonsprogram (følg instruksjonene)...'
$P_OLLAMA_RUN_FAIL  = "Ollama-installasjonsprogrammet ble ikke fullført: {0}"
$P_OLLAMA_PATH      = 'Ollama ikke funnet i PATH etter installasjon.'
$P_OLLAMA_PATH_HINT = "Start terminalen på nytt og kjør:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installert.'
$P_OLLAMA_START     = 'Starter Ollama-tjenesten...'
$P_OLLAMA_RUNNING   = 'Ollama-tjenesten kjører allerede.'
$P_OLLAMA_TIMEOUT   = 'Ollama-tjenesten svarte ikke i tide. Start den manuelt: ollama serve'

$P_MODEL_CHECK      = "Sjekker modell '{0}'..."
$P_MODEL_EXISTS     = "Modell '{0}' allerede til stede."
$P_MODEL_PULL       = "Laster ned '{0}' -- dette kan ta flere minutter..."
$P_MODEL_PULL_FAIL  = "Modell-nedlasting mislyktes. Kjør manuelt senere: ollama pull {0}"
$P_MODEL_READY      = "Modell '{0}' klar."

$P_DIRS_READY       = 'Datamapper klare.'
$P_SETTINGS_CREATED = 'Standardinnstillinger opprettet i data\settings.json.'
$P_DB_ERROR         = 'Databaseoppsett støtte på en feil -- fortsett via /databases i spillet.'
$P_DB_MISSING       = 'tools\install_dbs.py ikke funnet -- databasenedlasting hoppes over.'

$P_DONE             = 'Installasjon fullført!'
$P_START_HINT       = 'Slik starter du spillet:'
$P_START_CMD        = '    .\run_nmm.bat'
