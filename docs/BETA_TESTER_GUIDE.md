# Nine Men's Morris — Beta Tester Setup Guide

Thanks for agreeing to try this out! Here's how to get the game running in a few minutes.

---

## Step 1 — Download the game

You can get it from either of these sources (same file, pick whichever works for you):

- **GitHub:** https://github.com/benmarkbrandwood-blip/NMM_LLM/releases/tag/v1.0.0
  — click **nmm_source_v1.0.0.zip** under Assets

- **Internet Archive:** https://archive.org/download/nine-mens-morris-ai
  — click **nmm_source_v1.0.0.zip**

Extract the zip to any folder on your computer.

---

## Step 2 — Install

You'll need **Python 3.10 or newer** installed first. Get it from https://www.python.org/downloads/ — on Windows, tick *"Add python.exe to PATH"* during install.

Once Python is installed, run the installer from the extracted folder:

**Windows** — double-click `install.bat`

**macOS** — double-click `install.command`
*(first time: right-click → Open to allow it past Gatekeeper)*

**Linux** — open a terminal in the folder and run `./install.sh`

The installer sets up everything automatically — it will also ask if you want to install Ollama for AI commentary (optional, ~5 GB download).

---

## Step 3 — Start the game

**Windows** — double-click `run_nmm.bat`

**macOS** — double-click `run_nmm.command`

**Linux** — run `./run_nmm.sh` in a terminal

Your browser will open automatically at `http://127.0.0.1:8000`.

---

## Step 4 — Download the AI models (required)

On first run, click **Databases** in the top menu and download at minimum the **AI models** bundle (~6 MB). The game won't have an AI opponent without this.

The endgame tables and full-game database (recommended, ~100 MB total) give the AI much stronger play — grab those too if you have the space.

---

## Feedback

Please send any bugs, crashes, or feedback to [your contact details here]. If the server crashes, the log at `data/logs/server.log` inside the game folder is the most useful thing to include.
