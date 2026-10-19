# install_i18n/hu.ps1 — Hungarian messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Telepítő      |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Telepítési könyvtár: {0}"
$P_PATH_ERROR       = "A telepítési útvonal '[' vagy ']' karaktert tartalmaz, amelyet a PowerShell rosszul kezel. Helyezze a mappát egyszerűbb útvonalra (pl. C:\NMM) és futtassa újra."

$P_PYTHON_SEARCH    = 'Python 3.10 vagy újabb keresése...'
$P_PYTHON_FOUND     = "Python {0}.{1} használata '{2}' segítségével."
$P_PYTHON_OLD       = "'{0}' Python {1}.{2} -- 3.10+ szükséges; következő próbálása."
$P_PYTHON_MISSING   = 'Python 3.10 vagy újabb nem található.'
$P_PYTHON_URL       = 'Telepítse innen: https://www.python.org/downloads/'
$P_PYTHON_PATH      = "FONTOS: jelölje be az 'Add python.exe to PATH' opciót a telepítés során."
$P_PYTHON_FAIL      = 'Python 3.10+ szükséges.'

$P_VENV_EXISTS      = 'Meglévő .venv található -- újrafelhasználás.'
$P_VENV_CREATE      = 'Virtuális környezet létrehozása .venv mappában...'
$P_VENV_FAIL        = "Nem sikerült létrehozni a virtuális környezetet. Győződjön meg arról, hogy a Python 'venv' modulja elérhető."

$P_PIP_UPGRADE      = 'pip / setuptools / wheel frissítése...'
$P_PIP_WARN         = 'A pip frissítése nem nulla kilépési kódot adott vissza; folytatás mindenképpen.'

$P_REQS_INSTALL     = 'Python-követelmények telepítése (ez néhány percet vehet igénybe)...'
$P_REQS_CPP         = "Windowson a chromadb néha igényli a 'Microsoft C++ Build Tools' csomagot."
$P_REQS_CPP_URL     = 'Telepítse innen: https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'majd futtassa újra az install.bat fájlt.'
$P_REQS_FAIL        = 'A Python-függőségek telepítése sikertelen.'
$P_REQS_DONE        = 'Python-csomagok telepítve.'

$P_OLLAMA_SKIP      = 'Ollama kihagyása (-NoOllama jelző beállítva). Az LLM-funkciók letiltásra kerülnek.'
$P_OLLAMA_AUTO      = 'Ollama automatikus telepítése (-Yes jelző beállítva).'
$P_OLLAMA_PROMPT1   = 'Az Ollama lehetővé teszi az AI-kommentárokat és az LLM-alapú lépéselemzést.'
$P_OLLAMA_PROMPT2   = "Az alapértelmezett modell ({0}) ~5 GB és letöltést igényel."
$P_OLLAMA_YN        = '[NMM] Telepíti az Ollama-t az LLM-funkciókhoz? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama kihagyása. Futtassa újra az install.ps1 fájlt (NoOllama nélkül) a későbbi hozzáadáshoz.'
$P_OLLAMA_EXISTS    = "Az Ollama már telepítve van ({0})."
$P_OLLAMA_DOWNLOAD  = 'Ollama telepítő letöltése...'
$P_OLLAMA_DL_FAIL   = "Az Ollama telepítő automatikus letöltése sikertelen: {0}"
$P_OLLAMA_DL_MANUAL = 'Telepítse manuálisan a https://ollama.com oldalról, majd futtassa újra az install.bat fájlt.'
$P_OLLAMA_RUN       = 'Ollama telepítő futtatása (kövesse az utasításokat)...'
$P_OLLAMA_RUN_FAIL  = "Az Ollama telepítő nem fejeződött be: {0}"
$P_OLLAMA_PATH      = 'Az Ollama nem található a PATH-ban a telepítés után.'
$P_OLLAMA_PATH_HINT = "Indítsa újra a terminált, majd futtassa:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama telepítve.'
$P_OLLAMA_START     = 'Ollama-szolgáltatás indítása...'
$P_OLLAMA_RUNNING   = 'Az Ollama-szolgáltatás már fut.'
$P_OLLAMA_TIMEOUT   = 'Az Ollama-szolgáltatás nem válaszolt időben. Indítsa el manuálisan: ollama serve'

$P_MODEL_CHECK      = "'{0}' modell ellenőrzése..."
$P_MODEL_EXISTS     = "A '{0}' modell már jelen van."
$P_MODEL_PULL       = "'{0}' letöltése -- ez több percet vehet igénybe..."
$P_MODEL_PULL_FAIL  = "A modell letöltése sikertelen. Futtassa manuálisan később: ollama pull {0}"
$P_MODEL_READY      = "'{0}' modell kész."

$P_DIRS_READY       = 'Adatkönyvtárak készen állnak.'
$P_SETTINGS_CREATED = 'Alapértelmezett beállítások létrehozva itt: data\settings.json.'
$P_DB_ERROR         = 'Az adatbázis-beállítás hibát észlelt -- folytassa a játékon belüli /databases menüpontban.'
$P_DB_MISSING       = 'tools\install_dbs.py nem található -- adatbázis-letöltés kihagyva.'

$P_DONE             = 'Telepítés kész!'
$P_START_HINT       = 'A játék indítása:'
$P_START_CMD        = '    .\run_nmm.bat'
