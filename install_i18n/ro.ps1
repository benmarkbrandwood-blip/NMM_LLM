# install_i18n/ro.ps1 — Romanian messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Instalare     |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Director de instalare: {0}"
$P_PATH_ERROR       = "Calea de instalare conține '[' sau ']' pe care PowerShell le gestionează prost. Mutați folderul într-o cale mai simplă (ex. C:\NMM) și rulați din nou."

$P_PYTHON_SEARCH    = 'Se caută Python 3.10 sau mai recent...'
$P_PYTHON_FOUND     = "Se folosește Python {0}.{1} prin '{2}'."
$P_PYTHON_OLD       = "'{0}' este Python {1}.{2} -- necesar 3.10+; se încearcă următorul."
$P_PYTHON_MISSING   = 'Python 3.10 sau mai recent nu a fost găsit.'
$P_PYTHON_URL       = 'Instalați-l de la https://www.python.org/downloads/'
$P_PYTHON_PATH      = "IMPORTANT: bifați 'Add python.exe to PATH' în timpul instalării."
$P_PYTHON_FAIL      = 'Python 3.10+ este necesar.'

$P_VENV_EXISTS      = '.venv existent găsit -- se reutilizează.'
$P_VENV_CREATE      = 'Se creează mediul virtual în .venv...'
$P_VENV_FAIL        = "Nu s-a putut crea mediul virtual. Asigurați-vă că modulul Python 'venv' este disponibil."

$P_PIP_UPGRADE      = 'Se actualizează pip / setuptools / wheel...'
$P_PIP_WARN         = 'Actualizarea pip a returnat un cod nenul; se continuă oricum.'

$P_REQS_INSTALL     = 'Se instalează cerințele Python (poate dura câteva minute)...'
$P_REQS_CPP         = "Pe Windows, chromadb necesită uneori 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Instalați de la https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'apoi rulați din nou install.bat.'
$P_REQS_FAIL        = 'Instalarea dependențelor Python a eșuat.'
$P_REQS_DONE        = 'Pachete Python instalate.'

$P_OLLAMA_SKIP      = 'Se sare Ollama (flag -NoOllama setat). Funcțiile LLM vor fi dezactivate.'
$P_OLLAMA_AUTO      = 'Se instalează Ollama automat (flag -Yes setat).'
$P_OLLAMA_PROMPT1   = 'Ollama activează comentariile AI și analiza mutărilor cu LLM.'
$P_OLLAMA_PROMPT2   = "Modelul implicit ({0}) are ~5 GB și necesită descărcare."
$P_OLLAMA_YN        = '[NMM] Instalați Ollama pentru funcții LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Se sare Ollama. Rulați din nou install.ps1 (fără -NoOllama) pentru a-l adăuga mai târziu.'
$P_OLLAMA_EXISTS    = "Ollama deja instalat ({0})."
$P_OLLAMA_DOWNLOAD  = 'Se descarcă programul de instalare Ollama...'
$P_OLLAMA_DL_FAIL   = "Nu s-a putut descărca automat programul de instalare Ollama: {0}"
$P_OLLAMA_DL_MANUAL = 'Instalați-l manual de la https://ollama.com apoi rulați din nou install.bat.'
$P_OLLAMA_RUN       = 'Se rulează programul de instalare Ollama (urmați instrucțiunile)...'
$P_OLLAMA_RUN_FAIL  = "Programul de instalare Ollama nu s-a finalizat: {0}"
$P_OLLAMA_PATH      = 'Ollama nu a fost găsit în PATH după instalare.'
$P_OLLAMA_PATH_HINT = "Reporniți terminalul și rulați:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama instalat.'
$P_OLLAMA_START     = 'Se pornește serviciul Ollama...'
$P_OLLAMA_RUNNING   = 'Serviciul Ollama rulează deja.'
$P_OLLAMA_TIMEOUT   = 'Serviciul Ollama nu a răspuns la timp. Porniți-l manual: ollama serve'

$P_MODEL_CHECK      = "Se verifică modelul '{0}'..."
$P_MODEL_EXISTS     = "Modelul '{0}' este deja prezent."
$P_MODEL_PULL       = "Se descarcă '{0}' -- poate dura câteva minute..."
$P_MODEL_PULL_FAIL  = "Descărcarea modelului a eșuat. Rulați manual mai târziu: ollama pull {0}"
$P_MODEL_READY      = "Modelul '{0}' gata."

$P_DIRS_READY       = 'Directoare de date pregătite.'
$P_SETTINGS_CREATED = 'Setări implicite create la data\settings.json.'
$P_DB_ERROR         = 'Configurarea bazei de date a întâmpinat o eroare -- continuați din /databases în joc.'
$P_DB_MISSING       = 'tools\install_dbs.py negăsit -- descărcarea bazei de date omisă.'

$P_DONE             = 'Instalare completă!'
$P_START_HINT       = 'Pentru a porni jocul:'
$P_START_CMD        = '    .\run_nmm.bat'
