# install_i18n/pl.ps1 — Polish messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Instalator    |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Katalog instalacyjny: {0}"
$P_PATH_ERROR       = "Ścieżka instalacji zawiera '[' lub ']', co PowerShell obsługuje słabo. Przenieś folder do prostszej ścieżki (np. C:\NMM) i uruchom ponownie."

$P_PYTHON_SEARCH    = 'Szukanie Pythona 3.10 lub nowszego...'
$P_PYTHON_FOUND     = "Używanie Pythona {0}.{1} przez '{2}'."
$P_PYTHON_OLD       = "'{0}' to Python {1}.{2} -- wymagany 3.10+; próba następnego."
$P_PYTHON_MISSING   = 'Nie znaleziono Pythona 3.10 ani nowszego.'
$P_PYTHON_URL       = 'Zainstaluj ze strony https://www.python.org/downloads/'
$P_PYTHON_PATH      = "WAŻNE: zaznacz 'Add python.exe to PATH' podczas instalacji."
$P_PYTHON_FAIL      = 'Wymagany jest Python 3.10+.'

$P_VENV_EXISTS      = 'Znaleziono istniejące środowisko .venv -- ponowne użycie.'
$P_VENV_CREATE      = 'Tworzenie środowiska wirtualnego w .venv...'
$P_VENV_FAIL        = "Nie udało się utworzyć środowiska wirtualnego. Upewnij się, że moduł Pythona 'venv' jest dostępny."

$P_PIP_UPGRADE      = 'Aktualizacja pip / setuptools / wheel...'
$P_PIP_WARN         = 'Aktualizacja pip zwróciła niezerowy kod wyjścia; kontynuowanie.'

$P_REQS_INSTALL     = 'Instalowanie wymagań Pythona (może to potrwać kilka minut)...'
$P_REQS_CPP         = "W systemie Windows chromadb czasami wymaga pakietu 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Zainstaluj ze strony https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'następnie uruchom ponownie install.bat.'
$P_REQS_FAIL        = 'Instalacja zależności Pythona nie powiodła się.'
$P_REQS_DONE        = 'Pakiety Pythona zainstalowane.'

$P_OLLAMA_SKIP      = 'Pomijanie Ollama (flaga -NoOllama ustawiona). Funkcje LLM zostaną wyłączone.'
$P_OLLAMA_AUTO      = 'Automatyczna instalacja Ollama (flaga -Yes ustawiona).'
$P_OLLAMA_PROMPT1   = 'Ollama umożliwia komentarze AI i analizę ruchów przez LLM.'
$P_OLLAMA_PROMPT2   = "Domyślny model ({0}) ma ~5 GB i wymaga pobrania."
$P_OLLAMA_YN        = '[NMM] Zainstalować Ollama dla funkcji LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Pomijanie Ollama. Uruchom ponownie install.ps1 (bez -NoOllama), aby dodać go później.'
$P_OLLAMA_EXISTS    = "Ollama jest już zainstalowany ({0})."
$P_OLLAMA_DOWNLOAD  = 'Pobieranie instalatora Ollama...'
$P_OLLAMA_DL_FAIL   = "Nie udało się automatycznie pobrać instalatora Ollama: {0}"
$P_OLLAMA_DL_MANUAL = 'Zainstaluj ręcznie ze strony https://ollama.com, a następnie uruchom ponownie install.bat.'
$P_OLLAMA_RUN       = 'Uruchamianie instalatora Ollama (postępuj zgodnie z instrukcjami)...'
$P_OLLAMA_RUN_FAIL  = "Instalator Ollama nie zakończył pracy: {0}"
$P_OLLAMA_PATH      = 'Ollama nie znaleziony w PATH po instalacji.'
$P_OLLAMA_PATH_HINT = "Uruchom ponownie terminal, a następnie wykonaj:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama zainstalowany.'
$P_OLLAMA_START     = 'Uruchamianie usługi Ollama...'
$P_OLLAMA_RUNNING   = 'Usługa Ollama jest już uruchomiona.'
$P_OLLAMA_TIMEOUT   = 'Usługa Ollama nie odpowiedziała w czasie. Uruchom ją ręcznie: ollama serve'

$P_MODEL_CHECK      = "Sprawdzanie modelu '{0}'..."
$P_MODEL_EXISTS     = "Model '{0}' już obecny."
$P_MODEL_PULL       = "Pobieranie '{0}' -- może to potrwać kilka minut..."
$P_MODEL_PULL_FAIL  = "Pobieranie modelu nie powiodło się. Uruchom ręcznie później: ollama pull {0}"
$P_MODEL_READY      = "Model '{0}' gotowy."

$P_DIRS_READY       = 'Katalogi danych gotowe.'
$P_SETTINGS_CREATED = 'Ustawienia domyślne utworzone w data\settings.json.'
$P_DB_ERROR         = 'Konfiguracja bazy danych napotkała błąd -- kontynuuj przez /databases w grze.'
$P_DB_MISSING       = 'tools\install_dbs.py nie znaleziony -- pobieranie bazy danych pominięte.'

$P_DONE             = 'Instalacja zakończona!'
$P_START_HINT       = 'Aby uruchomić grę:'
$P_START_CMD        = '    .\run_nmm.bat'
