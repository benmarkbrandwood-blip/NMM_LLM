# install_i18n/de.ps1 — German messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installer     |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Installationsverzeichnis: {0}"
$P_PATH_ERROR       = "Der Installationspfad enthält '[' oder ']', was PowerShell schlecht verarbeitet. Verschieben Sie den Ordner an einen einfacheren Pfad (z. B. C:\NMM) und starten Sie erneut."

$P_PYTHON_SEARCH    = 'Suche nach Python 3.10 oder neuer...'
$P_PYTHON_FOUND     = "Python {0}.{1} wird über '{2}' verwendet."
$P_PYTHON_OLD       = "'{0}' ist Python {1}.{2} -- benötigt 3.10+; nächste wird versucht."
$P_PYTHON_MISSING   = 'Python 3.10 oder neuer wurde nicht gefunden.'
$P_PYTHON_URL       = 'Installieren Sie es von https://www.python.org/downloads/'
$P_PYTHON_PATH      = "WICHTIG: Haken bei 'Add python.exe to PATH' während der Installation setzen."
$P_PYTHON_FAIL      = 'Python 3.10+ ist erforderlich.'

$P_VENV_EXISTS      = 'Vorhandene .venv gefunden -- wird wiederverwendet.'
$P_VENV_CREATE      = 'Virtuelle Umgebung in .venv wird erstellt...'
$P_VENV_FAIL        = "Virtuelle Umgebung konnte nicht erstellt werden. Stellen Sie sicher, dass das Python-Modul 'venv' verfügbar ist."

$P_PIP_UPGRADE      = 'pip / setuptools / wheel werden aktualisiert...'
$P_PIP_WARN         = 'pip-Aktualisierung gab einen Fehlercode zurück; wird trotzdem fortgesetzt.'

$P_REQS_INSTALL     = 'Python-Anforderungen werden installiert (dies kann einige Minuten dauern)...'
$P_REQS_CPP         = "Unter Windows benötigt chromadb manchmal die 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installieren Sie von https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'danach install.bat erneut ausführen.'
$P_REQS_FAIL        = 'Installation der Python-Abhängigkeiten fehlgeschlagen.'
$P_REQS_DONE        = 'Python-Pakete installiert.'

$P_OLLAMA_SKIP      = 'Ollama wird übersprungen (-NoOllama-Flag gesetzt). LLM-Funktionen werden deaktiviert.'
$P_OLLAMA_AUTO      = 'Ollama wird automatisch installiert (-Yes-Flag gesetzt).'
$P_OLLAMA_PROMPT1   = 'Ollama ermöglicht KI-Kommentare und LLM-gestützte Zuganalyse.'
$P_OLLAMA_PROMPT2   = "Das Standardmodell ({0}) ist ~5 GB und erfordert einen Download."
$P_OLLAMA_YN        = '[NMM] Ollama für LLM-Funktionen installieren? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama wird übersprungen. Führen Sie install.ps1 erneut aus (ohne -NoOllama), um es später hinzuzufügen.'
$P_OLLAMA_EXISTS    = "Ollama bereits installiert ({0})."
$P_OLLAMA_DOWNLOAD  = 'Ollama-Installationsprogramm wird heruntergeladen...'
$P_OLLAMA_DL_FAIL   = "Ollama-Installationsprogramm konnte nicht automatisch heruntergeladen werden: {0}"
$P_OLLAMA_DL_MANUAL = 'Installieren Sie es manuell von https://ollama.com und führen Sie install.bat erneut aus.'
$P_OLLAMA_RUN       = 'Ollama-Installationsprogramm wird ausgeführt (folgen Sie den Anweisungen)...'
$P_OLLAMA_RUN_FAIL  = "Ollama-Installationsprogramm wurde nicht abgeschlossen: {0}"
$P_OLLAMA_PATH      = 'Ollama nach der Installation nicht im PATH gefunden.'
$P_OLLAMA_PATH_HINT = "Starten Sie Ihr Terminal neu und führen Sie aus:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installiert.'
$P_OLLAMA_START     = 'Ollama-Dienst wird gestartet...'
$P_OLLAMA_RUNNING   = 'Ollama-Dienst läuft bereits.'
$P_OLLAMA_TIMEOUT   = 'Ollama-Dienst hat nicht rechtzeitig geantwortet. Starten Sie ihn manuell: ollama serve'

$P_MODEL_CHECK      = "Modell '{0}' wird geprüft..."
$P_MODEL_EXISTS     = "Modell '{0}' ist bereits vorhanden."
$P_MODEL_PULL       = "'{0}' wird heruntergeladen -- dies kann mehrere Minuten dauern..."
$P_MODEL_PULL_FAIL  = "Modell-Download fehlgeschlagen. Führen Sie später manuell aus: ollama pull {0}"
$P_MODEL_READY      = "Modell '{0}' bereit."

$P_DIRS_READY       = 'Datenverzeichnisse bereit.'
$P_SETTINGS_CREATED = 'Standardeinstellungen erstellt unter data\settings.json.'
$P_DB_ERROR         = 'Datenbankeinrichtung ist auf einen Fehler gestoßen -- weiter unter /databases im Spiel.'
$P_DB_MISSING       = 'tools\install_dbs.py nicht gefunden -- Datenbankdownload wird übersprungen.'

$P_DONE             = 'Installation abgeschlossen!'
$P_START_HINT       = 'So starten Sie das Spiel:'
$P_START_CMD        = '    .\run_nmm.bat'
