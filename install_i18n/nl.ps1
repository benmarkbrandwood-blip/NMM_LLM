# install_i18n/nl.ps1 — Dutch messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installatie   |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Installatiemap: {0}"
$P_PATH_ERROR       = "Het installatiepad bevat '[' of ']' wat PowerShell slecht verwerkt. Verplaats de map naar een eenvoudiger pad (bijv. C:\NMM) en voer opnieuw uit."

$P_PYTHON_SEARCH    = 'Zoeken naar Python 3.10 of nieuwer...'
$P_PYTHON_FOUND     = "Python {0}.{1} gebruiken via '{2}'."
$P_PYTHON_OLD       = "'{0}' is Python {1}.{2} -- 3.10+ vereist; volgende wordt geprobeerd."
$P_PYTHON_MISSING   = 'Python 3.10 of nieuwer niet gevonden.'
$P_PYTHON_URL       = 'Installeer het vanaf https://www.python.org/downloads/'
$P_PYTHON_PATH      = "BELANGRIJK: vink 'Add python.exe to PATH' aan tijdens de installatie."
$P_PYTHON_FAIL      = 'Python 3.10+ is vereist.'

$P_VENV_EXISTS      = 'Bestaande .venv gevonden -- wordt hergebruikt.'
$P_VENV_CREATE      = 'Virtuele omgeving aanmaken in .venv...'
$P_VENV_FAIL        = "Kan geen virtuele omgeving aanmaken. Zorg dat de Python-module 'venv' beschikbaar is."

$P_PIP_UPGRADE      = 'pip / setuptools / wheel bijwerken...'
$P_PIP_WARN         = 'pip-update retourneerde een niet-nul exitcode; toch doorgaan.'

$P_REQS_INSTALL     = 'Python-vereisten installeren (dit kan enkele minuten duren)...'
$P_REQS_CPP         = "Op Windows heeft chromadb soms de 'Microsoft C++ Build Tools' nodig."
$P_REQS_CPP_URL     = 'Installeer vanaf https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'voer daarna install.bat opnieuw uit.'
$P_REQS_FAIL        = 'Installatie van Python-afhankelijkheden mislukt.'
$P_REQS_DONE        = 'Python-pakketten geïnstalleerd.'

$P_OLLAMA_SKIP      = 'Ollama overgeslagen (-NoOllama-vlag ingesteld). LLM-functies worden uitgeschakeld.'
$P_OLLAMA_AUTO      = 'Ollama automatisch installeren (-Yes-vlag ingesteld).'
$P_OLLAMA_PROMPT1   ='Ollama schakelt AI-commentaar en LLM-aangedreven zetsanalyse in.'
$P_OLLAMA_PROMPT2   = "Het standaardmodel ({0}) is ~5 GB en vereist een download."
$P_OLLAMA_YN        = '[NMM] Ollama installeren voor LLM-functies? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama overgeslagen. Voer install.ps1 opnieuw uit (zonder -NoOllama) om het later toe te voegen.'
$P_OLLAMA_EXISTS    = "Ollama al geïnstalleerd ({0})."
$P_OLLAMA_DOWNLOAD  = 'Ollama-installatieprogramma downloaden...'
$P_OLLAMA_DL_FAIL   = "Kon het Ollama-installatieprogramma niet automatisch downloaden: {0}"
$P_OLLAMA_DL_MANUAL = 'Installeer het handmatig vanaf https://ollama.com en voer install.bat opnieuw uit.'
$P_OLLAMA_RUN       = 'Ollama-installatieprogramma uitvoeren (volg de aanwijzingen)...'
$P_OLLAMA_RUN_FAIL  = "Ollama-installatieprogramma niet voltooid: {0}"
$P_OLLAMA_PATH      = 'Ollama niet gevonden in PATH na installatie.'
$P_OLLAMA_PATH_HINT = "Herstart uw terminal en voer uit:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama geïnstalleerd.'
$P_OLLAMA_START     = 'Ollama-service starten...'
$P_OLLAMA_RUNNING   = 'Ollama-service is al actief.'
$P_OLLAMA_TIMEOUT   = 'Ollama-service reageerde niet op tijd. Start het handmatig: ollama serve'

$P_MODEL_CHECK      = "Controleren op model '{0}'..."
$P_MODEL_EXISTS     = "Model '{0}' al aanwezig."
$P_MODEL_PULL       = "'{0}' downloaden -- dit kan enkele minuten duren..."
$P_MODEL_PULL_FAIL  = "Model downloaden mislukt. Voer later handmatig uit: ollama pull {0}"
$P_MODEL_READY      = "Model '{0}' gereed."

$P_DIRS_READY       = 'Datamappen gereed.'
$P_SETTINGS_CREATED = 'Standaardinstellingen aangemaakt in data\settings.json.'
$P_DB_ERROR         = 'Database-instelling heeft een fout ondervonden -- ga verder via /databases in het spel.'
$P_DB_MISSING       = 'tools\install_dbs.py niet gevonden -- databasedownload overgeslagen.'

$P_DONE             = 'Installatie voltooid!'
$P_START_HINT       = 'Het spel starten:'
$P_START_CMD        = '    .\run_nmm.bat'
