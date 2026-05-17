# Nine Men's Morris — AI-Powered Web Game

A browser-based Nine Men's Morris game with a classical minimax engine, an Ollama-powered LLM commentary system, a curated opening book, and a fully tunable AI personality system.

![board](https://img.shields.io/badge/game-Nine%20Men's%20Morris-c8a96e?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Quick Start

```bash
git clone <repo-url>
cd NMM_ollama
./install.sh      # one-time setup: venv + Ollama + model
./run_nmm.sh      # starts server and opens browser
```

`install.sh` will:
- Create a Python venv at `.venv/`
- Install all Python dependencies
- Install [Ollama](https://ollama.com) if not already present
- Pull the configured LLM model (`llama3.1:8b` by default)

`run_nmm.sh` will:
- Start Ollama if it isn't running
- Launch the FastAPI server (`uvicorn`)
- Open `http://127.0.0.1:8000` in your browser automatically

---

## Requirements

- **Python 3.10+**
- **Linux / macOS** (WSL2 supported for Windows)
- **curl** (for Ollama install)
- ~5 GB disk for the default LLM model

---

## Features

### Game engine
- Full Nine Men's Morris rules: placement, movement, flying, mill detection, captures
- 10 difficulty levels — levels 1–8 use fixed minimax depth (2–9 ply); levels 9–10 use iterative deepening (20 s / 45 s time budget)
- Undo (rewinds both your move and the AI's response)
- Human vs AI, Human vs Human, or Random colour selection
- Draw by threefold repetition, 50-move rule, and mutual agreement
- **Force Capture** toggle — makes the AI capture aggressively, disabling the fly-sacrifice heuristic

### AI engine
- **Negamax + alpha-beta** with phase-aware heuristics:
  - Closed mills, blocked pieces, piece count, two-configurations, double-mill pivots, win configuration
  - Mobility and immediate mill threats (phase-weighted)
  - Mill-cycle readiness (feeder mills), fork threats, herding/encirclement
  - Cross/cardinal node positional bonus (3-neighbour nodes score higher than 2-neighbour corners)
  - Fly-phase asymmetry bonus (prefer reaching 3 pieces before opponent in 4v4 endgame)
- **Tactical urgency layer** — delta-based bonuses applied at move-selection level (not inside negamax, avoiding sign-inversion):
  - Closing a mill
  - Building or disrupting **cycling mill setups** — pairs of 2-configs whose empty closing squares are adjacent, letting a pivot piece shuttle between them and force a capture every two turns
  - Blocking an opponent's immediately closeable mill
  - Dismantling opponent 2-configurations
  - Creating **feeder diamond structures** — four pieces all adjacent to one key empty square, forming two simultaneous mill threats; if one anchor is captured another slides in to close the remaining mill
  - **Mill wrapping** — occupying exit squares of opponent closed mills so their pivot has nowhere to slide; rewards accepting an opponent mill in exchange for surrounding it
  - Controlling cardinal (cross-node) squares; early-game scatter placement
- **Deadline-aware search** — checks the clock every 4 096 nodes; always returns the best partial result found so far on timeout
- **Auto-force-move** — when the AI exceeds its expected thinking time the browser countdown fires `force_move` automatically; a server-side safety net fires 5 s later if the client message is lost
- **AI resignation** — if the human's position strength exceeds 0.95 (tanh-normalised) for 3 consecutive AI turns, the AI concedes with a farewell message

### Opening book
- Curated opening lines with UCB1-scored selection; learns win/loss/draw outcomes per opening
- **Opening Recogniser** — detects rotated and mirrored variants via full D4 dihedral symmetry (4 rotations × 4 reflections)
- Novel openings discovered during play are saved with `needs_llm_name=True` and named on the next LLM-enabled run

### Endgame
- **Endgame Recogniser** — detects named endgame phases (active / deep), zugzwang risk, and mill-cycle patterns
- Extra search depth added automatically in endgame positions

### LLM commentary (MillsAI)
- Consults a locally running Ollama model for move opinions, position commentary, and post-game session summaries
- Reads the last 10 games (with full move sequences) before each new game
- Remembers bad moves via ChromaDB vector store
- Comments on mill formations, strong moves (score ≥ 0.75), and poor human moves (capped to avoid spam)
- Asks periodic strategic questions to invite the player to think ahead
- **Player chat** — type a message at any point and MillsAI responds in context
- All LLM move recommendations are validated against the legal move list

### AI Tuning & Personalities
- **13 configurable weight sliders** accessible via the **AI Tuning** header button (panel stays open during play); settings persist across sessions via the **Save settings** button:

  | Group | Slider | Default | What it rewards |
  |-------|--------|---------|-----------------|
  | Tactical | Mill closure urgency | 500 | Closing one of the AI's own mills this move |
  | Tactical | Cycling mill setup | 800 | Building two 2-configs whose empty closing squares are adjacent — a single pivot piece shuttles between them, forcing a capture every two turns. Also penalises disrupting the opponent's cycling setups. |
  | Tactical | Block immediate mill threat | 400 | Neutralising an opponent 2-config that could be closed next turn |
  | Tactical | Disrupt opponent 2-configs | 450 | Breaking up any opponent 2-piece mill setup, even if not immediately closeable |
  | Tactical | Feeder diamond creation | 300 | Building a diamond / fork structure — four pieces all adjacent to one key empty square, forming two simultaneous mill threats. If one anchor is captured, another piece slides in to close the remaining mill. |
  | Tactical | Mill wrapping | 250 | Occupying exit squares around opponent closed mills so their pivot piece has nowhere useful to slide. High values let the AI accept an opponent mill if it can surround it. |
  | Tactical | Block cardinal mills | 400 | Occupying or evicting pieces from cross-node (midpoint) squares that border 3 adjacencies |
  | Tactical | Early spread placement | 100 | Placing pieces not adjacent to existing own pieces in the first 6 placements |
  | Positional | Positional weight % | 100 | Overall multiplier on all non-tactical positional scoring (100 = normal) |
  | Positional | Mill count weight % | 100 | How much each closed mill contributes to the static evaluation |
  | Positional | Mobility weight % | 100 | How much having more legal moves than the opponent is valued |
  | Positional | Blocked pieces weight % | 100 | Bonus for having opponent pieces with no legal moves |
  | Behaviour | Make mistakes % | 0 | Probability (%) of playing a deliberately bad move each turn |

- **6 personality presets** — select from the dropdown to pre-fill all sliders; dragging any slider switches to Custom:

  | Preset | Character |
  |--------|-----------|
  | **Balanced** | All defaults |
  | **Aggressive — The Crusher** | Hunts mills and cycling setups relentlessly, ignores wrapping defense, clusters pieces |
  | **Defensive — The Blocker** | Smothers every opponent threat, builds resilient diamond structures, wraps opponent mills |
  | **Positional — The Strategist** | Spreads across cross nodes, builds long-term cycling structures, thinks ahead |
  | **Scholar — The Bookworm** | Methodical opening placement, balanced diamond and wrapping awareness, solid all-round |
  | **Chaos — The Trickster** | Scatters randomly, ignores strategy, 45 % blunder rate |

### Web interface
- SVG board with coordinate labels (a–g, 1–7)
- Real-time game strength graph showing White/Black advantage across all moves
- **Countdown timer** — status bar counts down remaining expected think time; fires Force Move automatically when it reaches zero
- **Force Move button** (animated gold pulse) — visible while AI is thinking; interrupts search immediately if you don't want to wait
- Colour-coded hints: green = legal placements, yellow = selectable pieces, red = capturable pieces
- Optimistic board rendering — your move appears instantly before the server confirms
- Mill highlight on capture; **Hint** system (3 per game) with LLM explanation
- Commentary feed with speaker labels (GameAI / MillsAI / Game)
- **AI resignation overlay** — distinct result screen when the AI concedes

### Self-play training
```bash
python tools/self_play.py --no-llm --games 100 --white 6 --black 6 --swap --parallel 4
```

| Flag | Description |
|------|-------------|
| `--games N` | Number of games |
| `--white D` / `--black D` | AI difficulty 1–10 per side |
| `--blunder P` | Blunder probability for White (0–1) |
| `--swap` | Alternate which side plays White each game |
| `--parallel N` | Run N games simultaneously (fast mode only) |
| `--no-llm` | Skip all LLM calls — fast mode |
| `--name-openings` | Use LLM to name novel openings discovered during the run |
| `--summary` | Ask LLM for a batch summary after all games finish |

---

## Board Coordinate System

```
a7 ——— d7 ——— g7
|       |       |
|  b6 — d6 — f6  |
|  |    |    |  |
|  |  c5-d5-e5  |
a4-b4-c4    e4-f4-g4
|  |  c3-d3-e3  |
|  |    |    |  |
|  b2 — d2 — f2  |
|       |       |
a1 ——— d1 ——— g1
```

24 valid positions on three concentric squares connected by cross-lines.  
**Cross/cardinal nodes** (midpoints of each side, 3 neighbours): `a4 d7 g4 d1 b4 d6 f4 d2 c4 d5 e4 d3`  
**Corner nodes** (corners of squares, 2 neighbours): all remaining 12 positions.

---

## Configuration

Edit `data/settings.json` to change the Ollama model, URL, and LLM behaviour thresholds.

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_model` | `llama3.1:8b` | Ollama model to use |
| `ollama_url` | `http://localhost:11434` | Ollama server address |
| `poor_move_threshold` | `0.3` | Score drop that triggers an LLM comment on a human move |
| `max_poor_move_comments_per_game` | `5` | Cap on poor-move comments per game |
| `endgame_active_threshold` | `11` | Total pieces on board to enter endgame mode |
| `endgame_deep_threshold` | `8` | Total pieces to enter deep-endgame mode |

---

## Project Structure

```
NMM_ollama/
├── game/                   # Core engine: board, rules, game engine
├── ai/
│   ├── game_ai.py          # Negamax + alpha-beta, blunder mode, weights
│   ├── heuristics.py       # Phase-aware evaluation + HeuristicWeights dataclass
│   ├── mills_llm.py        # Ollama LLM interface
│   ├── coordinator.py      # AI deliberation, commentary, resignation tracking
│   ├── opening_book.py     # Opening library + UCB1 selection
│   ├── opening_recognizer.py  # D4 symmetry-aware recognition
│   └── endgame_recognizer.py  # Phase detection, zugzwang, mill-cycle patterns
├── web/
│   ├── app.py              # FastAPI + WebSocket server, session management
│   ├── static/
│   │   ├── game.js         # Game controller, personality presets, weight sliders
│   │   ├── board.js        # SVG board renderer
│   │   └── style.css       # Dark wood theme
│   └── templates/index.html
├── tools/
│   └── self_play.py        # AI vs AI training loop
├── data/
│   ├── settings.json       # Runtime configuration
│   ├── openings/           # Opening book JSON
│   ├── games/              # Game records (JSONL)
│   ├── chroma/             # ChromaDB vector store
│   └── session_memory/     # LLM session narratives
├── tests/                  # unittest test suite (160+ tests)
├── install.sh              # One-time installer
├── run_nmm.sh              # Launch script
└── requirements.txt
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m unittest discover tests/ -v
```

---

## Changing the LLM Model

```bash
ollama pull mistral        # or any other Ollama model
```

Then update `data/settings.json`:
```json
{ "ollama_model": "mistral" }
```

The game uses the new model from the next game start.

---

## License

MIT
