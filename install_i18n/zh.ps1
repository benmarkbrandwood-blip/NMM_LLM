# install_i18n/zh.ps1 — Chinese (Simplified) messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   九子棋 -- 安装程序                 |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "安装目录：{0}"
$P_PATH_ERROR       = "安装路径包含 '[' 或 ']'，PowerShell 处理不佳。请将文件夹移至更简单的路径（如 C:\NMM）后重新运行。"

$P_PYTHON_SEARCH    = '正在搜索 Python 3.10 或更高版本...'
$P_PYTHON_FOUND     = "通过 '{2}' 使用 Python {0}.{1}。"
$P_PYTHON_OLD       = "'{0}' 是 Python {1}.{2} -- 需要 3.10+；尝试下一个。"
$P_PYTHON_MISSING   = '未找到 Python 3.10 或更高版本。'
$P_PYTHON_URL       = '请从 https://www.python.org/downloads/ 安装'
$P_PYTHON_PATH      = "重要：安装时请勾选 'Add python.exe to PATH'。"
$P_PYTHON_FAIL      = '需要 Python 3.10+。'

$P_VENV_EXISTS      = '已找到现有 .venv -- 正在复用。'
$P_VENV_CREATE      = '正在 .venv 中创建虚拟环境...'
$P_VENV_FAIL        = "无法创建虚拟环境。请确保 Python 的 'venv' 模块可用。"

$P_PIP_UPGRADE      = '正在升级 pip / setuptools / wheel...'
$P_PIP_WARN         = 'pip 升级返回非零退出码；仍继续。'

$P_REQS_INSTALL     = '正在安装 Python 依赖（可能需要几分钟）...'
$P_REQS_CPP         = "在 Windows 上，chromadb 有时需要 'Microsoft C++ Build Tools'。"
$P_REQS_CPP_URL     = '请从 https://visualstudio.microsoft.com/visual-cpp-build-tools/ 安装'
$P_REQS_CPP_RERUN   = '然后重新运行 install.bat。'
$P_REQS_FAIL        = 'Python 依赖安装失败。'
$P_REQS_DONE        = 'Python 包已安装。'

$P_OLLAMA_SKIP      = '跳过 Ollama（已设置 -NoOllama 标志）。LLM 功能将被禁用。'
$P_OLLAMA_AUTO      = '自动安装 Ollama（已设置 -Yes 标志）。'
$P_OLLAMA_PROMPT1   = 'Ollama 提供 AI 解说和 LLM 驱动的走棋分析。'
$P_OLLAMA_PROMPT2   = "默认模型（{0}）约 5 GB，需要下载。"
$P_OLLAMA_YN        = '[NMM] 是否安装 Ollama 以启用 LLM 功能？[Y/n]'
$P_OLLAMA_SKIP2     = '跳过 Ollama。稍后请重新运行 install.ps1（不带 -NoOllama）来添加。'
$P_OLLAMA_EXISTS    = "Ollama 已安装（{0}）。"
$P_OLLAMA_DOWNLOAD  = '正在下载 Ollama 安装程序...'
$P_OLLAMA_DL_FAIL   = "无法自动下载 Ollama 安装程序：{0}"
$P_OLLAMA_DL_MANUAL = '请从 https://ollama.com 手动安装，然后重新运行 install.bat。'
$P_OLLAMA_RUN       = '正在运行 Ollama 安装程序（按提示操作）...'
$P_OLLAMA_RUN_FAIL  = "Ollama 安装程序未完成：{0}"
$P_OLLAMA_PATH      = '安装后在 PATH 中未找到 Ollama。'
$P_OLLAMA_PATH_HINT = "请重启终端，然后运行：  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama 已安装。'
$P_OLLAMA_START     = '正在启动 Ollama 服务...'
$P_OLLAMA_RUNNING   = 'Ollama 服务已在运行。'
$P_OLLAMA_TIMEOUT   = 'Ollama 服务未及时响应。请手动启动：ollama serve'

$P_MODEL_CHECK      = "正在检查模型 '{0}'..."
$P_MODEL_EXISTS     = "模型 '{0}' 已存在。"
$P_MODEL_PULL       = "正在拉取 '{0}' -- 可能需要数分钟..."
$P_MODEL_PULL_FAIL  = "模型拉取失败。请稍后手动运行：ollama pull {0}"
$P_MODEL_READY      = "模型 '{0}' 已就绪。"

$P_DIRS_READY       = '数据目录已就绪。'
$P_SETTINGS_CREATED = '默认设置已创建于 data\settings.json。'
$P_DB_ERROR         = '数据库设置遇到错误 -- 请在游戏中通过 /databases 继续。'
$P_DB_MISSING       = '未找到 tools\install_dbs.py -- 跳过数据库下载。'

$P_DONE             = '安装完成！'
$P_START_HINT       = '启动游戏：'
$P_START_CMD        = '    .\run_nmm.bat'
