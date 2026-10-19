# install_i18n/da.ps1 — Danish messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installering  |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Installationsmappe: {0}"
$P_PATH_ERROR       = "Installationsstien indeholder '[' eller ']' som PowerShell håndterer dårligt. Flyt mappen til en enklere sti (f.eks. C:\NMM) og kør igen."

$P_PYTHON_SEARCH    = 'Søger efter Python 3.10 eller nyere...'
$P_PYTHON_FOUND     = "Bruger Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' er Python {1}.{2} -- kræver 3.10+; prøver næste."
$P_PYTHON_MISSING   = 'Python 3.10 eller nyere blev ikke fundet.'
$P_PYTHON_URL       = 'Installer det fra https://www.python.org/downloads/'
$P_PYTHON_PATH      = "VIGTIGT: sæt flueben ved 'Add python.exe to PATH' under installationen."
$P_PYTHON_FAIL      = 'Python 3.10+ er påkrævet.'

$P_VENV_EXISTS      = 'Eksisterende .venv fundet -- genbruges.'
$P_VENV_CREATE      = 'Opretter virtuelt miljø i .venv...'
$P_VENV_FAIL        = "Kunne ikke oprette virtuelt miljø. Sørg for at Python-modulet 'venv' er tilgængeligt."

$P_PIP_UPGRADE      = 'Opgraderer pip / setuptools / wheel...'
$P_PIP_WARN         = 'pip-opgradering returnerede en ikke-nul exitkode; fortsætter alligevel.'

$P_REQS_INSTALL     = 'Installerer Python-krav (dette kan tage et par minutter)...'
$P_REQS_CPP         = "På Windows har chromadb nogle gange brug for 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installer fra https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'kør derefter install.bat igen.'
$P_REQS_FAIL        = 'Installation af Python-afhængigheder mislykkedes.'
$P_REQS_DONE        = 'Python-pakker installeret.'

$P_OLLAMA_SKIP      = 'Springer Ollama over (-NoOllama-flag sat). LLM-funktioner deaktiveres.'
$P_OLLAMA_AUTO      = 'Installerer Ollama automatisk (-Yes-flag sat).'
$P_OLLAMA_PROMPT1   = 'Ollama aktiverer AI-kommentarer og LLM-drevet trækanalyse.'
$P_OLLAMA_PROMPT2   = "Standardmodellen ({0}) er ~5 GB og kræver download."
$P_OLLAMA_YN        = '[NMM] Installer Ollama til LLM-funktioner? [Y/n]'
$P_OLLAMA_SKIP2     = 'Springer Ollama over. Kør install.ps1 igen (uden -NoOllama) for at tilføje det senere.'
$P_OLLAMA_EXISTS    = "Ollama allerede installeret ({0})."
$P_OLLAMA_DOWNLOAD  = 'Downloader Ollama-installationsprogram...'
$P_OLLAMA_DL_FAIL   = "Kunne ikke automatisk downloade Ollama-installationsprogrammet: {0}"
$P_OLLAMA_DL_MANUAL = 'Installer det manuelt fra https://ollama.com og kør derefter install.bat igen.'
$P_OLLAMA_RUN       = 'Kører Ollama-installationsprogram (følg vejledningen)...'
$P_OLLAMA_RUN_FAIL  = "Ollama-installationsprogrammet blev ikke fuldført: {0}"
$P_OLLAMA_PATH      = 'Ollama ikke fundet i PATH efter installation.'
$P_OLLAMA_PATH_HINT = "Genstart din terminal og kør:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installeret.'
$P_OLLAMA_START     = 'Starter Ollama-tjenesten...'
$P_OLLAMA_RUNNING   = 'Ollama-tjenesten kører allerede.'
$P_OLLAMA_TIMEOUT   = 'Ollama-tjenesten svarede ikke i tide. Start den manuelt: ollama serve'

$P_MODEL_CHECK      = "Tjekker model '{0}'..."
$P_MODEL_EXISTS     = "Model '{0}' allerede til stede."
$P_MODEL_PULL       = "Downloader '{0}' -- dette kan tage adskillige minutter..."
$P_MODEL_PULL_FAIL  = "Model-download mislykkedes. Kør manuelt senere: ollama pull {0}"
$P_MODEL_READY      = "Model '{0}' klar."

$P_DIRS_READY       = 'Datamapper klar.'
$P_SETTINGS_CREATED = 'Standardindstillinger oprettet i data\settings.json.'
$P_DB_ERROR         = 'Databaseopsætning stødte på en fejl -- fortsæt via /databases i spillet.'
$P_DB_MISSING       = 'tools\install_dbs.py ikke fundet -- databasedownload springes over.'

$P_DONE             = 'Installation fuldført!'
$P_START_HINT       = 'Sådan starter du spillet:'
$P_START_CMD        = '    .\run_nmm.bat'
