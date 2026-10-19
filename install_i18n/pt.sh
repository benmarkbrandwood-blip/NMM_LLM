#!/usr/bin/env bash
# install_i18n/pt.sh — Portuguese messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="A verificar Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 não encontrado. Instale o python3 e tente novamente."
I_PYTHON_OLD="É necessário Python 3.10+ (encontrado: %s)."

I_VENV_EXISTS="Ambiente virtual existente encontrado em .venv — criação ignorada."
I_VENV_CREATE="A criar ambiente virtual em .venv ..."
I_PIP_UPGRADE="A atualizar o pip..."
I_REQS_INSTALL="A instalar requisitos do Python..."
I_REQS_DONE="Pacotes Python instalados."

I_OLLAMA_EXISTS="Ollama já está instalado (%s)."
I_OLLAMA_INSTALL="A instalar o Ollama..."
I_BREW_MISSING="Homebrew não encontrado. Instale o Ollama manualmente em https://ollama.com/download/mac"
I_BREW_RERUN="Em seguida, execute este script novamente."
I_OLLAMA_DONE="Ollama instalado."
I_OLLAMA_START="A iniciar o serviço Ollama..."
I_OLLAMA_STARTED="Serviço Ollama iniciado (pid %s)."
I_OLLAMA_RUNNING="O serviço Ollama já está em execução."
I_OLLAMA_TIMEOUT="O serviço Ollama não iniciou a tempo."

I_MODEL_CHECK="A verificar o modelo '%s'..."
I_MODEL_EXISTS="O modelo '%s' já está presente."
I_MODEL_PULL="A transferir '%s' — isto pode demorar vários minutos..."
I_MODEL_READY="Modelo '%s' pronto."

I_DIRS_READY="Diretórios de dados prontos."
I_SETTINGS_CREATED="Configurações padrão criadas em data/settings.json."
I_DB_ERROR="A configuração da base de dados encontrou um erro — continue em /databases no jogo."
I_DB_MISSING="tools/install_dbs.py não encontrado — transferência da base de dados ignorada."

I_DONE="Instalação concluída!"
I_START_HINT="Para iniciar o jogo:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Ambiente virtual não encontrado. Execute primeiro ./install.sh."
R_NO_UVICORN="uvicorn não encontrado no venv. Execute primeiro ./install.sh."
R_OLLAMA_START="Ollama não está em execução — a iniciar em segundo plano..."
R_OLLAMA_TIMEOUT="Ollama não respondeu — as funcionalidades LLM serão desativadas."
R_PORT_BUSY="A porta %s já está em uso. A tentar a porta 8080..."
R_SHUTDOWN="A encerrar..."
R_NO_CKPT="Checkpoints do modelo de IA não encontrados — as funcionalidades de IA serão limitadas."
R_DL_CKPT="Transfira-os via ./install.sh ou abra /databases no jogo."
R_STARTING="A iniciar Nine Men's Morris em %s ..."
R_WAITING="À espera que o servidor esteja pronto..."
R_SERVER_SLOW="O servidor demorou demasiado a responder — a abrir o browser mesmo assim."
R_OPEN_BROWSER="A abrir o browser em %s"
R_NO_BROWSER="Não foi possível detetar um browser. Abra %s manualmente."
R_RUNNING="Nine Men's Morris está em execução em %s"
R_STOP_HINT="Prima Ctrl-C para parar."
