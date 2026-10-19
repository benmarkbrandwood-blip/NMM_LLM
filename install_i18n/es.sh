#!/usr/bin/env bash
# install_i18n/es.sh — Spanish messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Comprobando Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 no encontrado. Instale python3 e inténtelo de nuevo."
I_PYTHON_OLD="Se requiere Python 3.10+ (encontrado: %s)."

I_VENV_EXISTS="Se encontró un entorno virtual existente en .venv — omitiendo creación."
I_VENV_CREATE="Creando entorno virtual en .venv ..."
I_PIP_UPGRADE="Actualizando pip..."
I_REQS_INSTALL="Instalando dependencias de Python..."
I_REQS_DONE="Paquetes de Python instalados."

I_OLLAMA_EXISTS="Ollama ya está instalado (%s)."
I_OLLAMA_INSTALL="Instalando Ollama..."
I_BREW_MISSING="Homebrew no encontrado. Instale Ollama manualmente desde https://ollama.com/download/mac"
I_BREW_RERUN="Luego vuelva a ejecutar este script."
I_OLLAMA_DONE="Ollama instalado."
I_OLLAMA_START="Iniciando el servicio de Ollama..."
I_OLLAMA_STARTED="Servicio de Ollama iniciado (pid %s)."
I_OLLAMA_RUNNING="El servicio de Ollama ya está en ejecución."
I_OLLAMA_TIMEOUT="El servicio de Ollama no se inició a tiempo."

I_MODEL_CHECK="Comprobando el modelo '%s'..."
I_MODEL_EXISTS="El modelo '%s' ya está presente."
I_MODEL_PULL="Descargando '%s' — esto puede tardar varios minutos..."
I_MODEL_READY="Modelo '%s' listo."

I_DIRS_READY="Directorios de datos listos."
I_SETTINGS_CREATED="Configuración predeterminada creada en data/settings.json."
I_DB_ERROR="La configuración de la base de datos encontró un error — continúe en /databases en el juego."
I_DB_MISSING="tools/install_dbs.py no encontrado — omitiendo descarga de base de datos."

I_DONE="¡Instalación completada!"
I_START_HINT="Para iniciar el juego:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Entorno virtual no encontrado. Ejecute ./install.sh primero."
R_NO_UVICORN="uvicorn no encontrado en venv. Ejecute ./install.sh primero."
R_OLLAMA_START="Ollama no está en ejecución — iniciándolo en segundo plano..."
R_OLLAMA_TIMEOUT="Ollama no respondió — las funciones LLM estarán desactivadas."
R_PORT_BUSY="El puerto %s ya está en uso. Intentando el puerto 8080..."
R_SHUTDOWN="Cerrando..."
R_NO_CKPT="Puntos de control del modelo de IA no encontrados — las funciones de IA serán limitadas."
R_DL_CKPT="Descárguelos mediante ./install.sh o abra /databases en el juego."
R_STARTING="Iniciando Nine Men's Morris en %s ..."
R_WAITING="Esperando a que el servidor esté listo..."
R_SERVER_SLOW="El servidor tardó demasiado en responder — abriendo el navegador de todas formas."
R_OPEN_BROWSER="Abriendo el navegador en %s"
R_NO_BROWSER="No se pudo detectar un navegador. Abra %s manualmente."
R_RUNNING="Nine Men's Morris está en ejecución en %s"
R_STOP_HINT="Presione Ctrl-C para detener."
