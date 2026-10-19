# install_i18n/ru.ps1 — Russian messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Установка     |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Папка установки: {0}"
$P_PATH_ERROR       = "Путь установки содержит '[' или ']', которые PowerShell обрабатывает плохо. Переместите папку по более простому пути (напр. C:\NMM) и запустите снова."

$P_PYTHON_SEARCH    = 'Поиск Python 3.10 или новее...'
$P_PYTHON_FOUND     = "Используется Python {0}.{1} через '{2}'."
$P_PYTHON_OLD       = "'{0}' — Python {1}.{2} -- нужен 3.10+; пробуем следующий."
$P_PYTHON_MISSING   = 'Python 3.10 или новее не найден.'
$P_PYTHON_URL       = 'Установите его с https://www.python.org/downloads/'
$P_PYTHON_PATH      = "ВАЖНО: отметьте 'Add python.exe to PATH' при установке."
$P_PYTHON_FAIL      = 'Требуется Python 3.10+.'

$P_VENV_EXISTS      = 'Найдено существующее .venv -- используется повторно.'
$P_VENV_CREATE      = 'Создание виртуального окружения в .venv...'
$P_VENV_FAIL        = "Не удалось создать виртуальное окружение. Убедитесь, что модуль Python 'venv' доступен."

$P_PIP_UPGRADE      = 'Обновление pip / setuptools / wheel...'
$P_PIP_WARN         = 'Обновление pip вернуло ненулевой код; продолжаем.'

$P_REQS_INSTALL     = 'Установка зависимостей Python (может занять несколько минут)...'
$P_REQS_CPP         = "В Windows chromadb иногда требует 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Установите с https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'затем снова запустите install.bat.'
$P_REQS_FAIL        = 'Установка зависимостей Python не удалась.'
$P_REQS_DONE        = 'Пакеты Python установлены.'

$P_OLLAMA_SKIP      = 'Ollama пропущен (установлен флаг -NoOllama). Функции LLM будут отключены.'
$P_OLLAMA_AUTO      = 'Автоматическая установка Ollama (установлен флаг -Yes).'
$P_OLLAMA_PROMPT1   = 'Ollama включает комментарии ИИ и анализ ходов на основе LLM.'
$P_OLLAMA_PROMPT2   = "Модель по умолчанию ({0}) ~5 ГБ и требует загрузки."
$P_OLLAMA_YN        = '[NMM] Установить Ollama для функций LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama пропущен. Запустите install.ps1 снова (без -NoOllama), чтобы добавить позже.'
$P_OLLAMA_EXISTS    = "Ollama уже установлен ({0})."
$P_OLLAMA_DOWNLOAD  = 'Загрузка установщика Ollama...'
$P_OLLAMA_DL_FAIL   = "Не удалось автоматически загрузить установщик Ollama: {0}"
$P_OLLAMA_DL_MANUAL = 'Установите вручную с https://ollama.com, затем снова запустите install.bat.'
$P_OLLAMA_RUN       = 'Запуск установщика Ollama (следуйте инструкциям)...'
$P_OLLAMA_RUN_FAIL  = "Установщик Ollama не завершился: {0}"
$P_OLLAMA_PATH      = 'Ollama не найден в PATH после установки.'
$P_OLLAMA_PATH_HINT = "Перезапустите терминал и выполните:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama установлен.'
$P_OLLAMA_START     = 'Запуск службы Ollama...'
$P_OLLAMA_RUNNING   = 'Служба Ollama уже запущена.'
$P_OLLAMA_TIMEOUT   = 'Служба Ollama не ответила вовремя. Запустите вручную: ollama serve'

$P_MODEL_CHECK      = "Проверка модели '{0}'..."
$P_MODEL_EXISTS     = "Модель '{0}' уже присутствует."
$P_MODEL_PULL       = "Загрузка '{0}' -- это может занять несколько минут..."
$P_MODEL_PULL_FAIL  = "Загрузка модели не удалась. Запустите вручную позже: ollama pull {0}"
$P_MODEL_READY      = "Модель '{0}' готова."

$P_DIRS_READY       = 'Папки данных готовы.'
$P_SETTINGS_CREATED = 'Настройки по умолчанию созданы в data\settings.json.'
$P_DB_ERROR         = 'Настройка базы данных завершилась ошибкой -- продолжите через /databases в игре.'
$P_DB_MISSING       = 'tools\install_dbs.py не найден -- загрузка базы данных пропущена.'

$P_DONE             = 'Установка завершена!'
$P_START_HINT       = 'Запуск игры:'
$P_START_CMD        = '    .\run_nmm.bat'
