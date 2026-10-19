#!/usr/bin/env bash
# install_i18n/zh.sh — Simplified Chinese messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="正在检查 Python..."
I_PYTHON_OK="Python %s — 正常"
I_PYTHON_MISSING="未找到 Python 3。请安装 python3 后重试。"
I_PYTHON_OLD="需要 Python 3.10 或更高版本（当前版本：%s）。"

I_VENV_EXISTS="在 .venv 中发现已有虚拟环境 — 跳过创建。"
I_VENV_CREATE="正在 .venv 中创建虚拟环境 ..."
I_PIP_UPGRADE="正在升级 pip..."
I_REQS_INSTALL="正在安装 Python 依赖包..."
I_REQS_DONE="Python 包安装完成。"

I_OLLAMA_EXISTS="Ollama 已安装（%s）。"
I_OLLAMA_INSTALL="正在安装 Ollama..."
I_BREW_MISSING="未找到 Homebrew。请从 https://ollama.com/download/mac 手动安装 Ollama"
I_BREW_RERUN="然后重新运行此脚本。"
I_OLLAMA_DONE="Ollama 安装完成。"
I_OLLAMA_START="正在启动 Ollama 服务..."
I_OLLAMA_STARTED="Ollama 服务已启动（pid %s）。"
I_OLLAMA_RUNNING="Ollama 服务已在运行。"
I_OLLAMA_TIMEOUT="Ollama 服务未能在规定时间内启动。"

I_MODEL_CHECK="正在检查模型 '%s'..."
I_MODEL_EXISTS="模型 '%s' 已存在。"
I_MODEL_PULL="正在拉取 '%s' — 这可能需要几分钟..."
I_MODEL_READY="模型 '%s' 就绪。"

I_DIRS_READY="数据目录就绪。"
I_SETTINGS_CREATED="已在 data/settings.json 创建默认设置。"
I_DB_ERROR="数据库设置遇到错误 — 请在游戏中通过 /databases 继续。"
I_DB_MISSING="未找到 tools/install_dbs.py — 跳过数据库下载。"

I_DONE="安装完成！"
I_START_HINT="启动游戏："
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="未找到虚拟环境。请先运行 ./install.sh。"
R_NO_UVICORN="在 venv 中未找到 uvicorn。请先运行 ./install.sh。"
R_OLLAMA_START="Ollama 未运行 — 正在后台启动..."
R_OLLAMA_TIMEOUT="Ollama 未响应 — LLM 功能将被禁用。"
R_PORT_BUSY="端口 %s 已被占用。正在尝试端口 8080..."
R_SHUTDOWN="正在关闭..."
R_NO_CKPT="未找到 AI 模型检查点 — AI 功能将受限。"
R_DL_CKPT="请通过 ./install.sh 下载，或在游戏中打开 /databases。"
R_STARTING="正在 %s 启动 Nine Men's Morris ..."
R_WAITING="正在等待服务器就绪..."
R_SERVER_SLOW="服务器响应超时 — 仍然打开浏览器。"
R_OPEN_BROWSER="正在打开浏览器，地址：%s"
R_NO_BROWSER="无法检测到浏览器。请手动打开 %s。"
R_RUNNING="Nine Men's Morris 正在 %s 运行"
R_STOP_HINT="按 Ctrl-C 停止。"
