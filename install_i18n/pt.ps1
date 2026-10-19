# install_i18n/pt.ps1 — Portuguese messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Instalador    |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Diretório de instalação: {0}"
$P_PATH_ERROR       = "O caminho de instalação contém '[' ou ']' que o PowerShell trata mal. Mova a pasta para um caminho mais simples (ex. C:\NMM) e execute novamente."

$P_PYTHON_SEARCH    = 'A procurar Python 3.10 ou mais recente...'
$P_PYTHON_FOUND     = "A usar Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' é Python {1}.{2} -- necessário 3.10+; tentando o próximo."
$P_PYTHON_MISSING   = 'Python 3.10 ou mais recente não encontrado.'
$P_PYTHON_URL       = 'Instale a partir de https://www.python.org/downloads/'
$P_PYTHON_PATH      = "IMPORTANTE: marque 'Add python.exe to PATH' durante a instalação."
$P_PYTHON_FAIL      = 'Python 3.10+ é necessário.'

$P_VENV_EXISTS      = '.venv existente encontrado -- a reutilizar.'
$P_VENV_CREATE      = 'A criar ambiente virtual em .venv...'
$P_VENV_FAIL        = "Falha ao criar o ambiente virtual. Certifique-se de que o módulo 'venv' do Python está disponível."

$P_PIP_UPGRADE      = 'A atualizar pip / setuptools / wheel...'
$P_PIP_WARN         = 'A atualização do pip retornou um código não nulo; continuando mesmo assim.'

$P_REQS_INSTALL     = 'A instalar requisitos do Python (isto pode demorar alguns minutos)...'
$P_REQS_CPP         = "No Windows, o chromadb às vezes precisa das 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Instale a partir de https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'depois execute install.bat novamente.'
$P_REQS_FAIL        = 'Falha na instalação das dependências Python.'
$P_REQS_DONE        = 'Pacotes Python instalados.'

$P_OLLAMA_SKIP      = 'A ignorar Ollama (flag -NoOllama definida). As funções LLM serão desativadas.'
$P_OLLAMA_AUTO      = 'A instalar Ollama automaticamente (flag -Yes definida).'
$P_OLLAMA_PROMPT1   = 'O Ollama ativa comentários de IA e análise de movimentos com LLM.'
$P_OLLAMA_PROMPT2   = "O modelo padrão ({0}) tem ~5 GB e requer download."
$P_OLLAMA_YN        = '[NMM] Instalar Ollama para funções LLM? [Y/n]'
$P_OLLAMA_SKIP2     = 'A ignorar Ollama. Execute install.ps1 novamente (sem -NoOllama) para adicioná-lo mais tarde.'
$P_OLLAMA_EXISTS    = "Ollama já instalado ({0})."
$P_OLLAMA_DOWNLOAD  = 'A transferir instalador do Ollama...'
$P_OLLAMA_DL_FAIL   = "Não foi possível transferir o instalador do Ollama automaticamente: {0}"
$P_OLLAMA_DL_MANUAL = 'Instale manualmente a partir de https://ollama.com e execute install.bat novamente.'
$P_OLLAMA_RUN       = 'A executar o instalador do Ollama (siga as instruções)...'
$P_OLLAMA_RUN_FAIL  = "O instalador do Ollama não foi concluído: {0}"
$P_OLLAMA_PATH      = 'Ollama não encontrado no PATH após a instalação.'
$P_OLLAMA_PATH_HINT = "Reinicie o terminal e execute:  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama instalado.'
$P_OLLAMA_START     = 'A iniciar o serviço Ollama...'
$P_OLLAMA_RUNNING   = 'O serviço Ollama já está em execução.'
$P_OLLAMA_TIMEOUT   = 'O serviço Ollama não respondeu a tempo. Inicie-o manualmente: ollama serve'

$P_MODEL_CHECK      = "A verificar o modelo '{0}'..."
$P_MODEL_EXISTS     = "Modelo '{0}' já presente."
$P_MODEL_PULL       = "A transferir '{0}' -- isto pode demorar vários minutos..."
$P_MODEL_PULL_FAIL  = "Falha na transferência do modelo. Execute manualmente mais tarde: ollama pull {0}"
$P_MODEL_READY      = "Modelo '{0}' pronto."

$P_DIRS_READY       = 'Diretórios de dados prontos.'
$P_SETTINGS_CREATED = 'Configurações padrão criadas em data\settings.json.'
$P_DB_ERROR         = 'A configuração da base de dados encontrou um erro -- continue em /databases no jogo.'
$P_DB_MISSING       = 'tools\install_dbs.py não encontrado -- transferência da base de dados ignorada.'

$P_DONE             = 'Instalação concluída!'
$P_START_HINT       = 'Para iniciar o jogo:'
$P_START_CMD        = '    .\run_nmm.bat'
