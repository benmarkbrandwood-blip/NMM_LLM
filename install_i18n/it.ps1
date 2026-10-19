# install_i18n/it.ps1 — Italian messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installazione |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Directory di installazione: {0}"
$P_PATH_ERROR       = "Il percorso di installazione contiene '[' o ']' che PowerShell gestisce male. Sposta la cartella in un percorso più semplice (es. C:\NMM) e riesegui."

$P_PYTHON_SEARCH    = 'Ricerca di Python 3.10 o più recente...'
$P_PYTHON_FOUND     = "Uso di Python {0}.{1} tramite '{2}'."
$P_PYTHON_OLD       = "'{0}' è Python {1}.{2} -- richiesto 3.10+; tentativo successivo."
$P_PYTHON_MISSING   = 'Python 3.10 o più recente non trovato.'
$P_PYTHON_URL       = 'Installalo da https://www.python.org/downloads/'
$P_PYTHON_PATH      = "IMPORTANTE: seleziona 'Add python.exe to PATH' durante l''installazione."
$P_PYTHON_FAIL      = 'È richiesto Python 3.10+.'

$P_VENV_EXISTS      = '.venv esistente trovato -- riutilizzato.'
$P_VENV_CREATE      = 'Creazione dell''ambiente virtuale in .venv...'
$P_VENV_FAIL        = "Impossibile creare l''ambiente virtuale. Assicurati che il modulo 'venv' di Python sia disponibile."

$P_PIP_UPGRADE      = 'Aggiornamento di pip / setuptools / wheel...'
$P_PIP_WARN         = "L''aggiornamento di pip ha restituito un codice non zero; continuazione comunque."

$P_REQS_INSTALL     = 'Installazione dei requisiti Python (potrebbe richiedere alcuni minuti)...'
$P_REQS_CPP         = "Su Windows, chromadb a volte richiede le 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installa da https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'poi riesegui install.bat.'
$P_REQS_FAIL        = "Installazione delle dipendenze Python non riuscita."
$P_REQS_DONE        = 'Pacchetti Python installati.'

$P_OLLAMA_SKIP      = 'Ollama saltato (flag -NoOllama impostato). Le funzioni LLM saranno disabilitate.'
$P_OLLAMA_AUTO      = 'Installazione automatica di Ollama (flag -Yes impostato).'
$P_OLLAMA_PROMPT1   = 'Ollama abilita i commenti IA e l''analisi delle mosse con LLM.'
$P_OLLAMA_PROMPT2   = "Il modello predefinito ({0}) è ~5 GB e richiede un download."
$P_OLLAMA_YN        = '[NMM] Installare Ollama per le funzioni LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama saltato. Riesegui install.ps1 (senza -NoOllama) per aggiungerlo in seguito.'
$P_OLLAMA_EXISTS    = "Ollama già installato ({0})."
$P_OLLAMA_DOWNLOAD  = "Download del programma di installazione di Ollama..."
$P_OLLAMA_DL_FAIL   = "Impossibile scaricare automaticamente il programma di installazione di Ollama: {0}"
$P_OLLAMA_DL_MANUAL = 'Installalo manualmente da https://ollama.com poi riesegui install.bat.'
$P_OLLAMA_RUN       = 'Esecuzione del programma di installazione di Ollama (segui le istruzioni)...'
$P_OLLAMA_RUN_FAIL  = "Il programma di installazione di Ollama non è stato completato: {0}"
$P_OLLAMA_PATH      = 'Ollama non trovato nel PATH dopo l''installazione.'
$P_OLLAMA_PATH_HINT = "Riavvia il terminale, poi esegui:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installato.'
$P_OLLAMA_START     = 'Avvio del servizio Ollama...'
$P_OLLAMA_RUNNING   = 'Il servizio Ollama è già in esecuzione.'
$P_OLLAMA_TIMEOUT   = 'Il servizio Ollama non ha risposto in tempo. Avvialo manualmente: ollama serve'

$P_MODEL_CHECK      = "Verifica del modello '{0}'..."
$P_MODEL_EXISTS     = "Modello '{0}' già presente."
$P_MODEL_PULL       = "Download di '{0}' -- potrebbe richiedere diversi minuti..."
$P_MODEL_PULL_FAIL  = "Download del modello non riuscito. Esegui manualmente in seguito: ollama pull {0}"
$P_MODEL_READY      = "Modello '{0}' pronto."

$P_DIRS_READY       = 'Directory dati pronte.'
$P_SETTINGS_CREATED = 'Impostazioni predefinite create in data\settings.json.'
$P_DB_ERROR         = 'La configurazione del database ha riscontrato un errore -- continua da /databases nel gioco.'
$P_DB_MISSING       = 'tools\install_dbs.py non trovato -- download del database saltato.'

$P_DONE             = 'Installazione completata!'
$P_START_HINT       = 'Per avviare il gioco:'
$P_START_CMD        = '    .\run_nmm.bat'
