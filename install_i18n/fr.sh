#!/usr/bin/env bash
# install_i18n/fr.sh — French messages for install.sh and run_nmm.sh
# Format: plain strings; use printf "$VAR" args for %s substitution.

# ── install.sh ────────────────────────────────────────────────────────────────
I_TITLE_1="  ╔══════════════════════════════════════╗"
I_TITLE_2="  ║   Nine Men's Morris — Installer      ║"
I_TITLE_3="  ╚══════════════════════════════════════╝"

I_PYTHON_CHECK="Vérification de Python..."
I_PYTHON_OK="Python %s — OK"
I_PYTHON_MISSING="Python 3 introuvable. Installez python3 et réessayez."
I_PYTHON_OLD="Python 3.10+ requis (trouvé : %s)."

I_VENV_EXISTS="Environnement virtuel existant trouvé dans .venv — création ignorée."
I_VENV_CREATE="Création de l'environnement virtuel dans .venv ..."
I_PIP_UPGRADE="Mise à jour de pip..."
I_REQS_INSTALL="Installation des dépendances Python..."
I_REQS_DONE="Paquets Python installés."

I_OLLAMA_EXISTS="Ollama est déjà installé (%s)."
I_OLLAMA_INSTALL="Installation d'Ollama..."
I_BREW_MISSING="Homebrew introuvable. Installez Ollama manuellement depuis https://ollama.com/download/mac"
I_BREW_RERUN="Relancez ensuite ce script."
I_OLLAMA_DONE="Ollama installé."
I_OLLAMA_START="Démarrage du service Ollama..."
I_OLLAMA_STARTED="Service Ollama démarré (pid %s)."
I_OLLAMA_RUNNING="Le service Ollama est déjà en cours d'exécution."
I_OLLAMA_TIMEOUT="Le service Ollama n'a pas démarré à temps."

I_MODEL_CHECK="Vérification du modèle '%s'..."
I_MODEL_EXISTS="Le modèle '%s' est déjà présent."
I_MODEL_PULL="Téléchargement de '%s' — cela peut prendre plusieurs minutes..."
I_MODEL_READY="Modèle '%s' prêt."

I_DIRS_READY="Répertoires de données prêts."
I_SETTINGS_CREATED="Paramètres par défaut créés dans data/settings.json."
I_DB_ERROR="La configuration de la base de données a rencontré une erreur — continuer via /databases dans le jeu."
I_DB_MISSING="tools/install_dbs.py introuvable — téléchargement de la base de données ignoré."

I_DONE="Installation terminée !"
I_START_HINT="Pour lancer le jeu :"
I_START_CMD="  ./run_nmm.sh"

# ── run_nmm.sh ────────────────────────────────────────────────────────────────
R_NO_VENV="Environnement virtuel introuvable. Lancez d'abord ./install.sh."
R_NO_UVICORN="uvicorn introuvable dans venv. Lancez d'abord ./install.sh."
R_OLLAMA_START="Ollama ne tourne pas — démarrage en arrière-plan..."
R_OLLAMA_TIMEOUT="Ollama n'a pas répondu — les fonctionnalités LLM seront désactivées."
R_PORT_BUSY="Le port %s est déjà utilisé. Tentative sur le port 8080..."
R_SHUTDOWN="Arrêt en cours..."
R_NO_CKPT="Points de contrôle du modèle IA introuvables — les fonctionnalités IA seront limitées."
R_DL_CKPT="Téléchargez-les via ./install.sh ou ouvrez /databases dans le jeu."
R_STARTING="Démarrage de Nine Men's Morris à %s ..."
R_WAITING="En attente que le serveur soit prêt..."
R_SERVER_SLOW="Le serveur a mis trop de temps à répondre — ouverture du navigateur quand même."
R_OPEN_BROWSER="Ouverture du navigateur à %s"
R_NO_BROWSER="Impossible de détecter un navigateur. Ouvrez %s manuellement."
R_RUNNING="Nine Men's Morris fonctionne à %s"
R_STOP_HINT="Appuyez sur Ctrl-C pour arrêter."
