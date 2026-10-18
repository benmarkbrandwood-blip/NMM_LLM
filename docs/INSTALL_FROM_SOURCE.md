# Installing from Source (Git)

This guide is for developers or technically comfortable users who want to track
the latest commits, submit issues with exact version info, or contribute to the project.

---

## Prerequisites

- **Git** — [git-scm.com](https://git-scm.com/downloads)
- **Python 3.10+** — [python.org](https://www.python.org/downloads/) (Windows: tick *Add python.exe to PATH*)
- Optional: **Ollama** for LLM commentary — [ollama.com](https://ollama.com)

---

## Clone and install

### Linux / macOS

```bash
git clone https://github.com/benmarkbrandwood-blip/NMM_LLM.git
cd NMM_LLM
./install.sh
```

`install.sh` will:
- Create a Python venv at `.venv/`
- Install all Python dependencies from `requirements.txt`
- Install Ollama if not already present
- Pull the configured LLM model (`llama3.1:8b` by default)
- Prompt you to download AI models and game databases

Then start the game:

```bash
./run_nmm.sh
```

### Windows

Open a terminal (PowerShell or Command Prompt) in the cloned folder:

```
git clone https://github.com/benmarkbrandwood-blip/NMM_LLM.git
cd NMM_LLM
install.bat
```

Then start the game:

```
run_nmm.bat
```

Or double-click either `.bat` file from Explorer.

---

## Download AI models and databases

On first run, open `http://127.0.0.1:8000` and click **Databases** in the header.
Download at minimum the **AI models** bundle (~6 MB). The installer also prompts
for this — you can skip it there and do it in-browser instead.

| Bundle | Size | What it enables |
|--------|------|-----------------|
| AI models & checkpoints | ~6 MB | All AI features — **required** |
| Endgame tables (lite) | ~50 MB | Strong endgame play |
| Full-game database | ~50 MB | Opening + midgame position guidance |
| Human DB (lite/full) | 179–387 MB | Opening explorer, human-move overlay |
| Specialist DB v5 | ~111 MB | Specialist AI training data |

---

## Keeping up to date

```bash
git pull
./install.sh   # re-runs safely; skips steps already done
```

`install.sh` is idempotent — it won't reinstall Ollama or recreate the venv if they already exist.

For database updates: if a bundle's SHA changes in `data/db_manifest.json` after a pull,
the **Databases** page will show an **Update available** badge. Download the new bundle there.

---

## Optional: neural AI training

PyTorch is only needed if you want to retrain the learned AI. It's not required for playing.

```bash
# Linux / macOS
source .venv/bin/activate
pip install -r requirements_learned_ai.txt

# Windows
.venv\Scripts\activate
pip install -r requirements_learned_ai.txt
```

See [`docs/AI_INTERNALS.md`](AI_INTERNALS.md) for training documentation.

---

## Malom perfect database (optional)

The Malom database by Gábor Danner & Gábor E. Gévay (University of Szeged) provides
exact Win/Draw/Loss for every legal position. Download via BitTorrent:

- Torrent: [Malom Standard Ultra-strong v1.1.0](https://www.inf.u-szeged.hu/~danner/mills/dl.php?file=Malom_Standard_Ultra-strong_1.1.0.torrent) (12 GB compressed · 78 GB extracted)
- Extract, then set the path in the game at **Databases → Malom Database Path**

---

## Running tests

```bash
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  (Windows)
python -m unittest discover tests/ -v
```

---

## Reporting issues

[GitHub Issues](https://github.com/benmarkbrandwood-blip/NMM_LLM/issues) — please include
your OS, Python version, and the contents of `data/logs/server.log` if the server crashed.
