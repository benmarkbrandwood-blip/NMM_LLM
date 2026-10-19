#!/usr/bin/env bash
# install_i18n/ro.sh — Romanian messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Se verifică Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 nu a fost găsit. Instalați python3 și încercați din nou."
I_PYTHON_OLD="Este necesar Python 3.10+ (găsit: %s)."

I_VENV_EXISTS="S-a găsit un mediu virtual existent în .venv — crearea este omisă."
I_VENV_CREATE="Se creează mediul virtual în .venv ..."
I_PIP_UPGRADE="Se actualizează pip..."
I_REQS_INSTALL="Se instalează cerințele Python..."
I_REQS_DONE="Pachetele Python au fost instalate."

I_OLLAMA_EXISTS="Ollama este deja instalat (%s)."
I_OLLAMA_INSTALL="Se instalează Ollama..."
I_BREW_MISSING="Homebrew nu a fost găsit. Instalați Ollama manual de la https://ollama.com/download/mac"
I_BREW_RERUN="Apoi rulați din nou acest script."
I_OLLAMA_DONE="Ollama instalat."
I_OLLAMA_START="Se pornește serviciul Ollama..."
I_OLLAMA_STARTED="Serviciul Ollama a pornit (pid %s)."
I_OLLAMA_RUNNING="Serviciul Ollama rulează deja."
I_OLLAMA_TIMEOUT="Serviciul Ollama nu a pornit la timp."

I_MODEL_CHECK="Se verifică modelul '%s'..."
I_MODEL_EXISTS="Modelul '%s' este deja prezent."
I_MODEL_PULL="Se descarcă '%s' — aceasta poate dura câteva minute..."
I_MODEL_READY="Modelul '%s' este gata."

I_DIRS_READY="Directoarele de date sunt gata."
I_SETTINGS_CREATED="Setările implicite au fost create în data/settings.json."
I_DB_ERROR="Configurarea bazei de date a întâmpinat o eroare — continuați la /databases în joc."
I_DB_MISSING="tools/install_dbs.py nu a fost găsit — descărcarea bazei de date este omisă."

I_DONE="Instalarea este finalizată!"
I_START_HINT="Pentru a porni jocul:"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Mediul virtual nu a fost găsit. Rulați mai întâi ./install.sh."
R_NO_UVICORN="uvicorn nu a fost găsit în venv. Rulați mai întâi ./install.sh."
R_OLLAMA_START="Ollama nu rulează — se pornește în fundal..."
R_OLLAMA_TIMEOUT="Ollama nu a răspuns — funcțiile LLM vor fi dezactivate."
R_PORT_BUSY="Portul %s este deja utilizat. Se încearcă portul 8080..."
R_SHUTDOWN="Se închide..."
R_NO_CKPT="Punctele de control ale modelului AI nu au fost găsite — funcțiile AI vor fi limitate."
R_DL_CKPT="Descărcați-le prin ./install.sh sau deschideți /databases în joc."
R_STARTING="Se pornește Nine Men's Morris la %s ..."
R_WAITING="Se așteaptă ca serverul să fie gata..."
R_SERVER_SLOW="Serverul a durat prea mult să răspundă — se deschide browserul oricum."
R_OPEN_BROWSER="Se deschide browserul la %s"
R_NO_BROWSER="Nu s-a putut detecta un browser. Deschideți %s manual."
R_RUNNING="Nine Men's Morris rulează la %s"
R_STOP_HINT="Apăsați Ctrl-C pentru a opri."
