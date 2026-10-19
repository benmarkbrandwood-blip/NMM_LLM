# install_i18n/es.ps1 — Spanish messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Instalador    |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Directorio de instalación: {0}"
$P_PATH_ERROR       = "La ruta de instalación contiene '[' o ']' que PowerShell maneja mal. Mueva la carpeta a una ruta más simple (p. ej. C:\NMM) y vuelva a ejecutar."

$P_PYTHON_SEARCH    = 'Buscando Python 3.10 o más reciente...'
$P_PYTHON_FOUND     = "Usando Python {0}.{1} mediante '{2}'."
$P_PYTHON_OLD       = "'{0}' es Python {1}.{2} -- se necesita 3.10+; intentando el siguiente."
$P_PYTHON_MISSING   = 'No se encontró Python 3.10 o más reciente.'
$P_PYTHON_URL       = 'Instálelo desde https://www.python.org/downloads/'
$P_PYTHON_PATH      = "IMPORTANTE: marque 'Add python.exe to PATH' durante la instalación."
$P_PYTHON_FAIL      = 'Se requiere Python 3.10+.'

$P_VENV_EXISTS      = 'Se encontró .venv existente -- reutilizándolo.'
$P_VENV_CREATE      = 'Creando entorno virtual en .venv...'
$P_VENV_FAIL        = "No se pudo crear el entorno virtual. Asegúrese de que el módulo 'venv' de Python esté disponible."

$P_PIP_UPGRADE      = 'Actualizando pip / setuptools / wheel...'
$P_PIP_WARN         = 'La actualización de pip devolvió un código distinto de cero; continuando de todas formas.'

$P_REQS_INSTALL     = 'Instalando requisitos de Python (esto puede tardar unos minutos)...'
$P_REQS_CPP         = "En Windows, chromadb a veces necesita las 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Instale desde https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'luego vuelva a ejecutar install.bat.'
$P_REQS_FAIL        = 'Falló la instalación de dependencias de Python.'
$P_REQS_DONE        = 'Paquetes de Python instalados.'

$P_OLLAMA_SKIP      = 'Omitiendo Ollama (indicador -NoOllama establecido). Las funciones LLM estarán desactivadas.'
$P_OLLAMA_AUTO      = 'Instalando Ollama automáticamente (indicador -Yes establecido).'
$P_OLLAMA_PROMPT1   = 'Ollama activa los comentarios de IA y el análisis de movimientos con LLM.'
$P_OLLAMA_PROMPT2   = "El modelo predeterminado ({0}) tiene ~5 GB y requiere descarga."
$P_OLLAMA_YN        = '[NMM] ¿Instalar Ollama para funciones LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Omitiendo Ollama. Ejecute install.ps1 de nuevo (sin -NoOllama) para añadirlo más tarde.'
$P_OLLAMA_EXISTS    = "Ollama ya instalado ({0})."
$P_OLLAMA_DOWNLOAD  = 'Descargando el instalador de Ollama...'
$P_OLLAMA_DL_FAIL   = "No se pudo descargar el instalador de Ollama automáticamente: {0}"
$P_OLLAMA_DL_MANUAL = 'Instálelo manualmente desde https://ollama.com y luego vuelva a ejecutar install.bat.'
$P_OLLAMA_RUN       = 'Ejecutando el instalador de Ollama (siga las instrucciones)...'
$P_OLLAMA_RUN_FAIL  = "El instalador de Ollama no se completó: {0}"
$P_OLLAMA_PATH      = 'Ollama no encontrado en PATH tras la instalación.'
$P_OLLAMA_PATH_HINT = "Reinicie su terminal y ejecute:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama instalado.'
$P_OLLAMA_START     = 'Iniciando el servicio Ollama...'
$P_OLLAMA_RUNNING   = 'El servicio Ollama ya está en ejecución.'
$P_OLLAMA_TIMEOUT   = 'El servicio Ollama no respondió a tiempo. Inícielo manualmente: ollama serve'

$P_MODEL_CHECK      = "Comprobando el modelo '{0}'..."
$P_MODEL_EXISTS     = "El modelo '{0}' ya está presente."
$P_MODEL_PULL       = "Descargando '{0}' -- esto puede tardar varios minutos..."
$P_MODEL_PULL_FAIL  = "La descarga del modelo falló. Ejecute manualmente más tarde: ollama pull {0}"
$P_MODEL_READY      = "Modelo '{0}' listo."

$P_DIRS_READY       = 'Directorios de datos listos.'
$P_SETTINGS_CREATED = 'Configuración predeterminada creada en data\settings.json.'
$P_DB_ERROR         = 'La configuración de la base de datos encontró un error -- continúe en /databases en el juego.'
$P_DB_MISSING       = 'tools\install_dbs.py no encontrado -- omitiendo la descarga de la base de datos.'

$P_DONE             = '¡Instalación completa!'
$P_START_HINT       = 'Para iniciar el juego:'
$P_START_CMD        = '    .\run_nmm.bat'
