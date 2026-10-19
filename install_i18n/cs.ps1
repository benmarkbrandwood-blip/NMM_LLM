# install_i18n/cs.ps1 — Czech messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Instalátor    |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Instalační složka: {0}"
$P_PATH_ERROR       = "Instalační cesta obsahuje '[' nebo ']', s čím PowerShell pracuje špatně. Přesuňte složku na jednodušší cestu (např. C:\NMM) a spusťte znovu."

$P_PYTHON_SEARCH    = 'Hledám Python 3.10 nebo novější...'
$P_PYTHON_FOUND     = "Používám Python {0}.{1} přes '{2}'."
$P_PYTHON_OLD       = "'{0}' je Python {1}.{2} -- potřebuji 3.10+; zkouším další."
$P_PYTHON_MISSING   = 'Python 3.10 nebo novější nebyl nalezen.'
$P_PYTHON_URL       = 'Nainstalujte jej z https://www.python.org/downloads/'
$P_PYTHON_PATH      = "DŮLEŽITÉ: zaškrtněte 'Add python.exe to PATH' během instalace."
$P_PYTHON_FAIL      = 'Je vyžadován Python 3.10+.'

$P_VENV_EXISTS      = 'Nalezeno existující .venv -- opakované použití.'
$P_VENV_CREATE      = 'Vytváření virtuálního prostředí v .venv...'
$P_VENV_FAIL        = "Nepodařilo se vytvořit virtuální prostředí. Ujistěte se, že modul Python 'venv' je dostupný."

$P_PIP_UPGRADE      = 'Aktualizace pip / setuptools / wheel...'
$P_PIP_WARN         = 'Aktualizace pip vrátila nenulový exitový kód; pokračuji.'

$P_REQS_INSTALL     = 'Instalace požadavků Pythonu (může trvat několik minut)...'
$P_REQS_CPP         = "Ve Windows chromadb někdy vyžaduje 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Nainstalujte z https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'poté znovu spusťte install.bat.'
$P_REQS_FAIL        = 'Instalace závislostí Pythonu se nezdařila.'
$P_REQS_DONE        = 'Balíčky Pythonu nainstalovány.'

$P_OLLAMA_SKIP      = 'Přeskakuji Ollama (příznak -NoOllama nastaven). Funkce LLM budou zakázány.'
$P_OLLAMA_AUTO      = 'Automatická instalace Ollama (příznak -Yes nastaven).'
$P_OLLAMA_PROMPT1   = 'Ollama umožňuje komentáře AI a analýzu tahů pomocí LLM.'
$P_OLLAMA_PROMPT2   = "Výchozí model ({0}) je ~5 GB a vyžaduje stažení."
$P_OLLAMA_YN        = '[NMM] Nainstalovat Ollama pro funkce LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Přeskakuji Ollama. Spusťte znovu install.ps1 (bez -NoOllama) pro pozdější přidání.'
$P_OLLAMA_EXISTS    = "Ollama je již nainstalován ({0})."
$P_OLLAMA_DOWNLOAD  = 'Stahuji instalátor Ollama...'
$P_OLLAMA_DL_FAIL   = "Nepodařilo se automaticky stáhnout instalátor Ollama: {0}"
$P_OLLAMA_DL_MANUAL = 'Nainstalujte ručně z https://ollama.com a poté znovu spusťte install.bat.'
$P_OLLAMA_RUN       = 'Spouštím instalátor Ollama (postupujte podle pokynů)...'
$P_OLLAMA_RUN_FAIL  = "Instalátor Ollama se nedokončil: {0}"
$P_OLLAMA_PATH      = 'Ollama nebyl po instalaci nalezen v PATH.'
$P_OLLAMA_PATH_HINT = "Restartujte terminál a spusťte:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama nainstalován.'
$P_OLLAMA_START     = 'Spouštím službu Ollama...'
$P_OLLAMA_RUNNING   = 'Služba Ollama již běží.'
$P_OLLAMA_TIMEOUT   = 'Služba Ollama neodpověděla včas. Spusťte ji ručně: ollama serve'

$P_MODEL_CHECK      = "Kontroluji model '{0}'..."
$P_MODEL_EXISTS     = "Model '{0}' je již přítomen."
$P_MODEL_PULL       = "Stahuji '{0}' -- může trvat několik minut..."
$P_MODEL_PULL_FAIL  = "Stažení modelu se nezdařilo. Spusťte ručně později: ollama pull {0}"
$P_MODEL_READY      = "Model '{0}' připraven."

$P_DIRS_READY       = 'Datové složky připraveny.'
$P_SETTINGS_CREATED = 'Výchozí nastavení vytvořeno v data\settings.json.'
$P_DB_ERROR         = 'Nastavení databáze narazilo na chybu -- pokračujte přes /databases ve hře.'
$P_DB_MISSING       = 'tools\install_dbs.py nenalezen -- stažení databáze přeskočeno.'

$P_DONE             = 'Instalace dokončena!'
$P_START_HINT       = 'Spuštění hry:'
$P_START_CMD        = '    .\run_nmm.bat'
