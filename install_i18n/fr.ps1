# install_i18n/fr.ps1 — French messages for install.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── install.ps1 ───────────────────────────────────────────────────────────────
$P_TITLE_1 = "  +======================================+"
$P_TITLE_2 = "  |   Nine Men's Morris -- Installateur  |"
$P_TITLE_3 = "  +======================================+"

$P_INSTALL_DIR      = "Répertoire d''installation : {0}"
$P_PATH_ERROR       = "Le chemin d''installation contient '[' ou ']' que PowerShell gère mal. Déplacez le dossier vers un chemin plus simple (ex. C:\NMM) et relancez."

$P_PYTHON_SEARCH    = 'Recherche de Python 3.10 ou plus récent...'
$P_PYTHON_FOUND     = "Utilisation de Python {0}.{1} via '{2}'."
$P_PYTHON_OLD       = "'{0}' est Python {1}.{2} -- besoin de 3.10+ ; tentative suivante."
$P_PYTHON_MISSING   = 'Python 3.10 ou plus récent introuvable.'
$P_PYTHON_URL       = 'Installez-le depuis https://www.python.org/downloads/'
$P_PYTHON_PATH      = "IMPORTANT : cochez 'Add python.exe to PATH' lors de l''installation."
$P_PYTHON_FAIL      = 'Python 3.10+ est requis.'

$P_VENV_EXISTS      = 'Environnement .venv existant trouvé -- réutilisation.'
$P_VENV_CREATE      = 'Création de l''environnement virtuel dans .venv...'
$P_VENV_FAIL        = "Impossible de créer l''environnement virtuel. Assurez-vous que le module 'venv' de Python est disponible."

$P_PIP_UPGRADE      = 'Mise à jour de pip / setuptools / wheel...'
$P_PIP_WARN         = 'La mise à jour de pip a retourné un code non nul ; poursuite quand même.'

$P_REQS_INSTALL     = "Installation des dépendances Python (cela peut prendre quelques minutes)..."
$P_REQS_CPP         = "Sous Windows, chromadb nécessite parfois les 'Microsoft C++ Build Tools'."
$P_REQS_CPP_URL     = 'Installez depuis https://visualstudio.microsoft.com/visual-cpp-build-tools/'
$P_REQS_CPP_RERUN   = 'puis relancez install.bat.'
$P_REQS_FAIL        = "L''installation des dépendances Python a échoué."
$P_REQS_DONE        = 'Paquets Python installés.'

$P_OLLAMA_SKIP      = 'Ollama ignoré (indicateur -NoOllama défini). Les fonctions LLM seront désactivées.'
$P_OLLAMA_AUTO      = 'Installation automatique d''Ollama (indicateur -Yes défini).'
$P_OLLAMA_PROMPT1   = 'Ollama active les commentaires IA et l''analyse de coups LLM.'
$P_OLLAMA_PROMPT2   = "Le modèle par défaut ({0}) fait ~5 Go et nécessite un téléchargement."
$P_OLLAMA_YN        = '[NMM] Installer Ollama pour les fonctions LLM ? [Y/n]'
$P_OLLAMA_SKIP2     = 'Ollama ignoré. Relancez install.ps1 (sans -NoOllama) pour l''ajouter plus tard.'
$P_OLLAMA_EXISTS    = "Ollama déjà installé ({0})."
$P_OLLAMA_DOWNLOAD  = "Téléchargement du programme d''installation d''Ollama..."
$P_OLLAMA_DL_FAIL   = "Impossible de télécharger automatiquement le programme d''installation d''Ollama : {0}"
$P_OLLAMA_DL_MANUAL = 'Installez-le manuellement depuis https://ollama.com puis relancez install.bat.'
$P_OLLAMA_RUN       = "Exécution du programme d''installation d''Ollama (suivez les instructions)..."
$P_OLLAMA_RUN_FAIL  = "Le programme d''installation d''Ollama n''a pas abouti : {0}"
$P_OLLAMA_PATH      = 'Ollama introuvable dans PATH après installation.'
$P_OLLAMA_PATH_HINT = "Redémarrez votre terminal, puis exécutez :  ollama pull {0}"
$P_OLLAMA_DONE      = 'Ollama installé.'
$P_OLLAMA_START     = 'Démarrage du service Ollama...'
$P_OLLAMA_RUNNING   = 'Le service Ollama est déjà en cours d''exécution.'
$P_OLLAMA_TIMEOUT   = 'Le service Ollama n''a pas répondu à temps. Démarrez-le manuellement : ollama serve'

$P_MODEL_CHECK      = "Vérification du modèle '{0}'..."
$P_MODEL_EXISTS     = "Modèle '{0}' déjà présent."
$P_MODEL_PULL       = "Téléchargement de '{0}' -- cela peut prendre plusieurs minutes..."
$P_MODEL_PULL_FAIL  = "Échec du téléchargement du modèle. Exécutez manuellement plus tard : ollama pull {0}"
$P_MODEL_READY      = "Modèle '{0}' prêt."

$P_DIRS_READY       = 'Répertoires de données prêts.'
$P_SETTINGS_CREATED = 'Paramètres par défaut créés dans data\settings.json.'
$P_DB_ERROR         = 'La configuration de la base de données a rencontré une erreur -- continuez via /databases dans le jeu.'
$P_DB_MISSING       = 'tools\install_dbs.py introuvable -- téléchargement de la base de données ignoré.'

$P_DONE             = 'Installation terminée !'
$P_START_HINT       = 'Pour démarrer le jeu :'
$P_START_CMD        = '    .\run_nmm.bat'
