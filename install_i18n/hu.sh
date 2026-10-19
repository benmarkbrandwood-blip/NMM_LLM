#!/usr/bin/env bash
# install_i18n/hu.sh — Hungarian messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Python ellenőrzése..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="A Python 3 nem található. Telepítse a python3 csomagot, majd próbálja újra."
I_PYTHON_OLD="Python 3.10+ szükséges (talált: %s)."

I_VENV_EXISTS="Meglévő venv található a .venv mappában — a létrehozás kihagyva."
I_VENV_CREATE="Virtuális környezet létrehozása a .venv mappában ..."
I_PIP_UPGRADE="A pip frissítése..."
I_REQS_INSTALL="Python-függőségek telepítése..."
I_REQS_DONE="Python-csomagok telepítve."

I_OLLAMA_EXISTS="Az Ollama már telepítve van (%s)."
I_OLLAMA_INSTALL="Az Ollama telepítése..."
I_BREW_MISSING="A Homebrew nem található. Telepítse az Ollama-t kézzel innen: https://ollama.com/download/mac"
I_BREW_RERUN="Ezután futtassa újra ezt a szkriptet."
I_OLLAMA_DONE="Az Ollama telepítve."
I_OLLAMA_START="Az Ollama szolgáltatás indítása..."
I_OLLAMA_STARTED="Az Ollama szolgáltatás elindult (pid %s)."
I_OLLAMA_RUNNING="Az Ollama szolgáltatás már fut."
I_OLLAMA_TIMEOUT="Az Ollama szolgáltatás nem indult el időben."

I_MODEL_CHECK="A(z) '%s' modell ellenőrzése..."
I_MODEL_EXISTS="A(z) '%s' modell már jelen van."
I_MODEL_PULL="A(z) '%s' letöltése — ez több percig tarthat..."
I_MODEL_READY="A(z) '%s' modell kész."

I_DIRS_READY="Adatkönyvtárak készen állnak."
I_SETTINGS_CREATED="Az alapértelmezett beállítások létrehozva: data/settings.json."
I_DB_ERROR="Az adatbázis-beállítás hibába ütközött — folytassa a játékban a /databases oldalon."
I_DB_MISSING="A tools/install_dbs.py nem található — az adatbázis-letöltés kihagyva."

I_DONE="A telepítés kész!"
I_START_HINT="A játék indításához:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="A virtuális környezet nem található. Először futtassa a ./install.sh fájlt."
R_NO_UVICORN="Az uvicorn nem található a venv-ben. Először futtassa a ./install.sh fájlt."
R_OLLAMA_START="Az Ollama nem fut — elindítás a háttérben..."
R_OLLAMA_TIMEOUT="Az Ollama nem válaszolt — az LLM-funkciók le lesznek tiltva."
R_PORT_BUSY="A(z) %s port már használatban van. A 8080-as port kipróbálása..."
R_SHUTDOWN="Leállítás..."
R_NO_CKPT="Az AI-modell ellenőrzőpontjai nem találhatók — az AI-funkciók korlátozottak lesznek."
R_DL_CKPT="Töltse le őket a ./install.sh segítségével, vagy nyissa meg a /databases oldalt a játékban."
R_STARTING="A Nine Men's Morris indítása itt: %s ..."
R_WAITING="Várakozás a szerver készenállására..."
R_SERVER_SLOW="A szerver túl sokáig nem válaszolt — a böngésző megnyitása mindenképpen."
R_OPEN_BROWSER="Böngésző megnyitása itt: %s"
R_NO_BROWSER="Nem sikerült böngészőt észlelni. Nyissa meg manuálisan: %s"
R_RUNNING="A Nine Men's Morris fut itt: %s"
R_STOP_HINT="A leállításhoz nyomja meg a Ctrl-C billentyűkombinációt."
