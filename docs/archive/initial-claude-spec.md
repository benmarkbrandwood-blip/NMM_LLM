> Historical archive: this was the initial Claude-oriented product
> specification. It is not current agent guidance or acceptance evidence.
> Start from the repository-root `AGENTS.md` and current handover instead.

# Nine Men's Morris — AI-Powered Game with Dual AI & Ollama Integration

## Project Overview

Build a desktop Nine Men's Morris game in Python featuring:
- A graphical board rendered with Pygame (embedded in a Tkinter window)
- A classical minimax/alpha-beta Game AI (`GameAI`) that plays the computer's pieces
- A second "Mills AI" (`MillsLLM`) that consults a locally running Ollama model to reason about strategy
- Both AIs collaborate and their dialogue is displayed live in an on-screen chat panel
- A **formal Opening Knowledge System** that learns, stores, recognises, and names opening lines from curated strategy literature and played games
- An **Endgame Recognition System** that detects named endgame phases (7v4, 6v4, 4v4, 4v3, 3v3), recognises book-defined winning arrangements and draw patterns, switches GameAI to endgame-tuned evaluation, and lets MillsLLM discuss the endgame by name with phase-specific strategy drawn from the source book
- MillsLLM proactively comments on human moves it judges to be poor, using the GameAI evaluation score as a quality signal
- A **Post-Game Debrief System** where both AIs replay and annotate any game (live or imported), identify the critical turning point, and explain where the losing side went wrong
- Full game recording, bad-move memory, pattern recognition, and post-session LLM memory
- Persistent vector memory via ChromaDB for position recall and pattern retrieval
- Export of games in standard Nine Men's Morris notation

---

## Board Coordinate System

The board has 24 named positions using the standard algebraic grid overlay:

```
a7 --- d7 --- g7
|      |      |
b6 - c6 - d6 - e6 - f6
|    |         |    |
c5 - d5      d5 - e5
|    |         |    |
b4 - c4 - d4 - e4 - f4
|      |      |
a1 --- d1 --- g1
```

Full 24-node coordinate list:
```
Outer ring:  a7, d7, g7, g4, g1, d1, a1, a4
Middle ring: b6, d6, f6, f4, f2, d2, b2, b4
Inner ring:  c5, d5, e5, e3, e1(?), d3, c3, c4  [varies by variant]
```

Use the coordinate scheme from the export example in the brief:
- Columns: a, b, c, d, e, f, g
- Rows: 1–7 (odd rows only on outer ring, staggered on inner)

Refer to the export example notation for canonical position names:
`d2 d6 f4 b4 f2 f6 b2 b6 c3 c5 e3 d3 d1 e5 d5 d7 a4 c4 a4 a7 e4 e5 g7 d7 g4 d1`

The board graph (adjacency list) must be hardcoded as the source of truth for legal moves.

---

## Move Notation (Export Format)

Export format mirrors the example provided:

```
1. d2 d6
2. f4 b4
3. f2 f6
4. b2xf6 f6
5. b6 c3
...
10. c5-c4 a4-a7
11. e5-e4 d5-e5
12. d7-g7 a7-d7
13. g7-g4xd1 e5-d5xb6
14. d2-d1 *
```

### Notation rules:
- **Placement phase**: `<dest>` — e.g. `d2` (White places at d2), `d6` (Black places at d6)
- **Move phase**: `<src>-<dest>` — e.g. `c5-c4`
- **Mill + capture**: `<src>-<dest>x<captured>` or `<dest>x<captured>` (placement + capture)
- `*` = game over / end marker
- Each line is `<move_no>. <white_move> <black_move>`

---

## Architecture Overview

```
nine_mens_morris/
├── claude.md                  # This file
├── main.py                    # Entry point, Tkinter root + Pygame embed
├── game/
│   ├── __init__.py
│   ├── board.py               # Board state, adjacency, mill detection
│   ├── rules.py               # Legal move generation, phase logic
│   ├── game_engine.py         # Game loop, turn management, win conditions
│   ├── notation.py            # Move encoding / decoding / export
│   └── game_importer.py       # Parse exported notation files into game records
├── ai/
│   ├── __init__.py
│   ├── game_ai.py             # GameAI: minimax + alpha-beta pruning
│   ├── heuristics.py          # Evaluation functions for GameAI
│   ├── mills_llm.py           # MillsLLM: Ollama interface + prompt templates
│   ├── coordinator.py         # AI dialogue coordinator (GameAI ↔ MillsLLM)
│   ├── memory_manager.py      # ChromaDB vector store + session JSONL log
│   ├── opening_book.py        # Structured opening knowledge store (curated + learned)
│   ├── opening_recognizer.py  # Real-time opening detection and confidence scoring
│   ├── endgame_recognizer.py  # Named endgame phase detection (7v4, 6v4, 4v3, 3v3 …)
│   └── debrief_engine.py      # Post-game analysis: critical moments, narrative, best-move replay
├── ui/
│   ├── __init__.py
│   ├── board_renderer.py      # Pygame canvas: board drawing, piece rendering
│   ├── chat_panel.py          # Tkinter text widget for AI dialogue
│   ├── settings_panel.py      # Settings: difficulty, think time, colour choice
│   ├── opening_panel.py       # Opening name, confidence, and book-line overlay controls
│   ├── replay_panel.py        # Debrief window: replay controls, score graph, critical markers
│   └── main_window.py         # Tkinter root, layout manager, event wiring
├── data/
│   ├── games/                 # JSONL game records (one file per session)
│   ├── chroma_db/             # Persistent ChromaDB vector store
│   ├── session_memory/        # LLM narrative memory (markdown summaries)
│   ├── debriefs/              # Saved debrief reports (Markdown + JSON)
│   ├── endgames/
│   │   └── endgame_patterns.json  # Named endgame arrangements from strategy literature
│   └── openings/
│       ├── book_openings.json # Curated lines from strategy literature — READ ONLY
│       └── openings.json      # All openings: book + learned variants — writable
├── tools/
│   └── import_openings.py     # CLI tool to import and validate curated book lines
├── tests/
│   └── test_board.py
└── requirements.txt
```

---

## Module Specifications

### `game/board.py`

```python
POSITIONS = [
    "a7","d7","g7","g4","g1","d1","a1","a4",  # outer ring
    "b6","d6","f6","f4","f2","d2","b2","b4",  # middle ring
    "c5","d5","e5","e3","e1","d3","c3","c4",  # inner ring
]

ADJACENCY: dict[str, list[str]]  # hardcoded legal neighbours for each position

MILLS: list[tuple[str, str, str]]  # all 16 possible mills

class BoardState:
    positions: dict[str, str]   # pos -> "W" | "B" | ""
    phase: str                  # "place" | "move" | "fly"
    pieces_on_board: dict       # {"W": int, "B": int}
    pieces_placed: dict         # {"W": int, "B": int}
    pieces_captured: dict       # {"W": int, "B": int}

    def is_mill(self, pos: str, color: str) -> bool: ...
    def legal_placements(self, color: str) -> list[str]: ...
    def legal_moves(self, color: str) -> list[tuple[str, str]]: ...
    def legal_captures(self, color: str) -> list[str]: ...
    def apply_move(self, move: dict) -> "BoardState": ...
    def to_fen_string(self) -> str: ...
    def to_display_grid(self) -> str: ...
```

### `game/rules.py`

- `get_game_phase(board: BoardState, color: str) -> str`
- `is_terminal(board: BoardState) -> tuple[bool, str | None]`  → `(terminal, winner)`
- `is_blocked(board: BoardState, color: str) -> bool`
- `can_fly(board: BoardState, color: str) -> bool`  → True if ≤3 pieces remain

### `game/notation.py`

```python
def encode_move(move: dict, phase: str) -> str: ...
def export_pgn_style(game_record: list[dict]) -> str: ...
def parse_move_string(s: str) -> dict: ...
```

---

## Opening Knowledge System

The Opening Knowledge System is a **first-class, structured subsystem** separate from the
narrative vector memory in `memory_manager.py`. Opening knowledge is symbolic and relational:
named lines, branching trees, and outcome statistics. It is not interchangeable with
ChromaDB semantic search.

### Design Principles

| Concern | Where it lives |
|---|---|
| Named openings, move trees, outcome stats | `opening_book.py` + `openings.json` |
| Real-time ply-by-ply recognition | `opening_recognizer.py` |
| Board-position semantic similarity | `memory_manager.py` (ChromaDB) |
| Session narratives and bad moves | `memory_manager.py` (ChromaDB) |

### Opening Data Schema (`openings.json` and `book_openings.json`)

Each opening is a JSON object:

```json
{
  "opening_id": "mill-rush-white",
  "name": "Mill Rush",
  "aliases": ["Corner Attack", "Flank Mill"],
  "family": "Mill Rush",
  "side": "W",
  "seed_source": "book",
  "line_moves": ["d2", "d6", "f4", "b4", "f2", "f6", "b2", "b6"],
  "branch_moves": [
    {
      "branch_id": "mill-rush-white-flank",
      "deviation_ply": 5,
      "deviation_move": "b2",
      "name": "Mill Rush — Flank Variant",
      "line_continuation": ["b2", "b6", "b4"],
      "strategic_notes": "Trades centre control for rapid flank mill threat.",
      "seed_source": "book",
      "outcome_stats": { "W": 0, "B": 0, "D": 0 }
    }
  ],
  "opening_fen_signatures": [
    { "ply": 4, "fen": "..." },
    { "ply": 6, "fen": "..." },
    { "ply": 8, "fen": "..." }
  ],
  "strategic_notes": "Aims to close a mill at f2-f4-f6 while contesting d6 and b6.",
  "common_blunders": [
    "Playing b6 on ply 3 before securing f6 leaves the centre exposed to d5 counter."
  ],
  "recommended_responses": {
    "B": ["d5", "e3", "c5"]
  },
  "outcome_stats": { "W": 0, "B": 0, "D": 0 },
  "confidence": 1.0,
  "tags": ["aggressive", "flank", "early-mill", "placement"],
  "source_reference": "Chapter 3, §2 — Flank Opening Families"
}
```

**Field glossary:**

| Field | Description |
|---|---|
| `opening_id` | Unique slug, machine-readable |
| `name` | Human-readable opening name |
| `aliases` | Alternative names from different sources |
| `family` | Parent family name for grouping related lines |
| `side` | `"W"`, `"B"`, or `"both"` |
| `seed_source` | `"book"` (curated), `"human"` (human-taught), `"learned"` (discovered from games) |
| `line_moves` | Canonical move sequence in alternating W/B notation |
| `branch_moves` | List of known deviations and their continuations |
| `opening_fen_signatures` | Board FEN snapshots at key plies for transposition detection |
| `strategic_notes` | Plain-language purpose of the opening |
| `common_blunders` | Known poor responses within this opening |
| `recommended_responses` | Book replies for the opposing side |
| `outcome_stats` | Win/loss/draw counts from actual games |
| `confidence` | 1.0 for book lines; lower for learned variants |
| `tags` | `aggressive`, `defensive`, `parallel`, `perpendicular`, `flank`, `centre`, etc. |
| `source_reference` | Citation within the source material |

**Invariant:** `book_openings.json` is **read-only at runtime**. The engine writes only to
`openings.json`. When `openings.json` is first created it is seeded from `book_openings.json`.
This preserves the distinction between curated knowledge and learned experience.

---

### `ai/opening_book.py` — Opening Knowledge Store

```python
from dataclasses import dataclass, field

@dataclass
class BranchMove:
    branch_id: str
    deviation_ply: int
    deviation_move: str
    name: str
    line_continuation: list[str]
    strategic_notes: str
    seed_source: str             # "book" | "human" | "learned"
    outcome_stats: dict          # {"W": int, "B": int, "D": int}

@dataclass
class Opening:
    opening_id: str
    name: str
    aliases: list[str]
    family: str
    side: str
    seed_source: str
    line_moves: list[str]
    branch_moves: list[BranchMove]
    opening_fen_signatures: list[dict]   # [{"ply": int, "fen": str}]
    strategic_notes: str
    common_blunders: list[str]
    recommended_responses: dict          # {"W": [...], "B": [...]}
    outcome_stats: dict
    confidence: float
    tags: list[str]
    source_reference: str = ""

class OpeningBook:
    def __init__(
        self,
        book_path: str = "data/openings/book_openings.json",
        openings_path: str = "data/openings/openings.json",
    ): ...

    def load(self) -> None:
        """
        Load book_openings.json (read-only).
        Seed openings.json from book if it does not yet exist.
        Merge both into the in-memory index.
        """

    def get_by_id(self, opening_id: str) -> Opening | None: ...
    def get_by_name(self, name: str) -> list[Opening]: ...
    def get_by_family(self, family: str) -> list[Opening]: ...
    def get_by_tag(self, tag: str) -> list[Opening]: ...
    def get_by_seed_source(self, source: str) -> list[Opening]: ...

    def save_opening(self, opening: Opening) -> None:
        """
        Write a new or updated opening to openings.json.
        Never writes to book_openings.json.
        Raises ValueError if seed_source == "book" and the opening does not
        already exist (prevents accidental pollution of learned data).
        """

    def update_outcome_stats(self, opening_id: str, winner: str) -> None:
        """Increment W/B/D counter and persist to openings.json."""

    def record_deviation(
        self,
        opening_id: str,
        ply: int,
        move_played: str,
        board_fen: str,
    ) -> BranchMove | None:
        """
        If the deviation already exists as a branch, return it.
        If it is novel, create a new BranchMove with seed_source="learned",
        confidence proportional to how many times this deviation has been seen,
        and save it. Returns the (possibly new) BranchMove.
        """

    def save_novel_opening(
        self,
        move_sequence: list[str],
        board_fen_signatures: list[dict],
        outcome: str | None = None,
    ) -> Opening:
        """
        Called at game end when no opening was recognised.
        Creates a new Opening with seed_source="learned", confidence=0.3,
        auto-generates an opening_id, and saves to openings.json.
        Returns the created Opening.
        """
```

---

### `ai/opening_recognizer.py` — Real-Time Opening Detection

```python
from dataclasses import dataclass

@dataclass
class RecognitionResult:
    opening_id: str | None
    name: str | None
    family: str | None
    confidence: float            # 0.0–1.0
    status: str                  # "exact" | "probable" | "transposition" | "novel"
    matched_ply: int             # how many plies of the known line have been matched
    deviation_ply: int | None    # ply at which a deviation was first detected
    deviation_move: str | None   # what was played at the deviation point
    book_move: str | None        # what the book recommends at this ply
    branch_name: str | None      # matched branch, if any
    strategic_notes: str
    common_blunders: list[str]
    tags: list[str]

class OpeningRecognizer:
    def __init__(self, book: OpeningBook): ...

    move_sequence: list[str]     # all moves played so far this game, alternating W/B
    current_result: RecognitionResult

    def update(self, move_notation: str, board: BoardState) -> RecognitionResult:
        """
        Called after every placement-phase move (both sides).
        Appends move_notation to move_sequence, then re-evaluates recognition.
        Returns the updated RecognitionResult.

        Recognition algorithm:
        1. EXACT PREFIX MATCH
           Scan all openings whose line_moves starts with move_sequence.
           If exactly one match and sequence length >= 2: status="exact", confidence=1.0.
           If multiple matches remain: status="probable", confidence based on remaining candidates.

        2. DEVIATION DETECTION
           If no exact match but move_sequence[:-1] matched a known line on the previous ply:
           - Record deviation_ply and deviation_move.
           - Check branch_moves of the last matched opening for this deviation move.
           - If found: status="probable" on the branch, confidence=branch.confidence.
           - If not found: status="novel" for this branch, confidence=0.2.

        3. TRANSPOSITION DETECTION (FEN SIGNATURE MATCH)
           Compute board.to_fen_string() after the current ply.
           Compare against all opening_fen_signatures at the matching ply depth.
           If a FEN match is found despite move-order difference:
           - status="transposition", confidence=0.7.

        4. NOVEL
           If no match at all and ply >= 4:
           - status="novel", confidence=0.0, all name fields None.

        Recognition is only active during the placement phase (first 9 moves per side).
        After placement ends, current_result is frozen for the rest of the game.
        """

    def get_next_book_move(self) -> str | None:
        """
        Return the book's recommended next move for the current line,
        or None if status is "novel" or recognition is exhausted.
        """

    def get_current_result(self) -> RecognitionResult: ...

    def reset(self) -> None:
        """Called at game start."""
```

---

### `tools/import_openings.py` — Book Import CLI

A standalone command-line tool for curating book openings. It is **never called by the
game at runtime**; it is run once (or whenever new book material is added) to populate
`book_openings.json`.

```
Usage:
  python tools/import_openings.py --input raw_openings.json --validate --output data/openings/book_openings.json

Flags:
  --input       Path to raw JSON or CSV of opening lines
  --validate    Run legality checks: all moves must be legal in sequence, no duplicate IDs
  --output      Write validated openings to this path
  --dry-run     Print validation results without writing
  --merge       Merge into existing file rather than overwriting

Validation checks:
  - Each move in line_moves and branch_moves is a legal Nine Men's Morris placement
  - opening_id values are unique
  - side is "W", "B", or "both"
  - seed_source is "book"
  - confidence is 1.0
  - At least one opening_fen_signature is provided
```

---

## AI Systems

### `ai/game_ai.py` — GameAI (Classical Engine)

Implements **minimax with alpha-beta pruning** for the computer player.

```python
class GameAI:
    difficulty: int
    think_time: float
    color: str

    def choose_move(
        self,
        board: BoardState,
        recognition: RecognitionResult | None = None,
    ) -> dict:
        """
        During the placement phase, if recognition is provided:
        1. Retrieve the book's recommended next move via recognition.get_next_book_move().
        2. If the book move is legal, apply an opening_book_bonus to its minimax score.
           Bonus magnitude: settings.opening_book_bonus (default 0.2).
        3. If a candidate move appears in recognition.common_blunders,
           apply an opening_blunder_penalty (default -0.3).
        4. Once recognition.status == "novel" or the placement phase ends,
           revert to pure minimax with no opening influence.
        Book guidance is advisory: if minimax scores another move >= 0.5 higher than
        the book move after bonuses, GameAI may override.
        """

    def score_move(self, board: BoardState, move: dict) -> float:
        """
        Evaluate a single move from the current board.
        Returns the minimax score of the board state after applying the move.
        Used to assess the quality of the human's last move.
        """

    def _minimax(self, board, depth, alpha, beta, maximising) -> float: ...
    def _quiesce(self, board, alpha, beta) -> float: ...
```

**Depth table:**
| Difficulty | Search Depth |
|---|---|
| 1 (Easy) | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |
| 5 (Hard) | 6+ (iterative deepening) |

### `ai/heuristics.py` — Evaluation Functions

Score a `BoardState` from White's perspective. Based on the Kukreja heuristic:

```
score = w1 * closed_mills
      + w2 * blocked_opponent_pieces
      + w3 * piece_count_diff
      + w4 * open_mills_potential
      + w5 * double_mills
      + w6 * winning_config
```

Coefficients `w1..w6` vary by game phase (place / move / fly).

### `ai/mills_llm.py` — MillsLLM (Ollama Interface)

```python
from ollama import chat

class MillsLLM:
    model: str
    conversation_history: list[dict]
    narrative_memory: str
    bad_moves_context: list[dict]

    def ask_for_move_opinion(
        self,
        board: BoardState,
        candidate_moves: list[dict],
        game_ai_suggestion: dict,
        recognition: RecognitionResult | None = None,
    ) -> str:
        """
        Builds a prompt including:
        - ASCII board display
        - candidate moves
        - GameAI's preferred move
        - retrieved similar positions from ChromaDB
        - bad moves taught by human
        - opening recognition context (if available)
        Returns LLM's reasoning text.
        """

    def evaluate_human_move(
        self,
        board_before: BoardState,
        human_move: dict,
        score_before: float,
        score_after: float,
        score_drop_threshold: float,
        recognition: RecognitionResult | None = None,
    ) -> str | None:
        """
        Called after the human makes a move.
        If score drop >= threshold, ask the LLM to explain the strategic error.
        Opening context is included so the LLM can frame the comment in terms of
        the recognised opening line (e.g. "In the Mill Rush, this square is critical…").
        Returns the comment string, or None if the move was acceptable.
        """

    def record_human_feedback(self, board: BoardState, move: dict, reason: str): ...
    def generate_question_for_human(self, board: BoardState) -> str | None: ...
    def summarise_session(self, game_records: list[dict]) -> str: ...

    def debrief_game(
        self,
        report: "DebriefReport",
    ) -> str:
        """
        Given a completed DebriefReport (with critical_moments and evaluation scores),
        generate a narrative debrief: who won, where the loser went wrong, what
        the turning point was, and what a stronger line would have looked like.
        Returns a multi-paragraph markdown string suitable for display and saving.
        """

    def debrief_position(
        self,
        board: BoardState,
        ply: int,
        move_played: dict,
        best_move: dict,
        score_played: float,
        score_best: float,
        is_critical: bool,
        opening_name: str | None,
        context: str,
    ) -> str:
        """
        Generate a focused 2–4 sentence commentary on a single position during replay.
        Used when the user navigates to a specific ply in the replay panel.
        If is_critical=True, the explanation should describe why this was a turning point.
        """
```

**Prompt template — AI turn (with opening context):**

```
System: You are MillsAI, an expert Nine Men's Morris strategist advising the game engine.
        Reason about the board and suggest the best move for {color}.
        Never suggest illegal moves. Use algebraic notation (a1-g7).
        Avoid moves marked as bad in memory.

User:
  Board (FEN): {board.to_fen_string()}
  Board (visual):
  {board.to_display_grid()}

  Phase: {phase}
  Candidate moves: {moves_list}
  GameAI suggests: {game_ai_suggestion}

  --- Opening Context ---
  Recognised opening: {recognition.name or "Unknown / Novel"}
  Opening family:     {recognition.family or "—"}
  Recognition status: {recognition.status} (confidence {recognition.confidence:.0%})
  Book move at this ply: {recognition.book_move or "none / exhausted"}
  Strategic purpose:  {recognition.strategic_notes or "—"}
  Common blunders in this opening: {recognition.common_blunders or "none on record"}
  Tags: {recognition.tags}
  -----------------------

  Retrieved bad moves for similar positions: {bad_moves_context}
  Narrative memory excerpt: {narrative_memory[-500:]}

  What move do you recommend and why?
  If the board is in a named opening, refer to it by name.
  Do you have a question for the human?
```

**Prompt template — human poor-move comment (with opening context):**

```
System: You are MillsAI, an expert Nine Men's Morris strategist watching the human play.
        Comment briefly and helpfully on a likely strategic error. 1–3 sentences.
        Be encouraging, not condescending. Use algebraic notation (a1-g7).
        If a named opening is in progress, refer to it by name.

User:
  Board before move (FEN): {board_before.to_fen_string()}
  Board before (visual):
  {board_before.to_display_grid()}

  Human played: {human_move_notation}

  Board after move (visual):
  {board_after.to_display_grid()}

  Score delta: {score_delta:+.2f} (threshold: -{threshold:.2f})
  GameAI's preferred alternative: {best_ai_move_notation} (score: {best_score:+.2f})

  --- Opening Context ---
  Recognised opening: {recognition.name or "Unknown / Novel"}
  Status: {recognition.status} (confidence {recognition.confidence:.0%})
  Deviation: human played {recognition.deviation_move} at ply {recognition.deviation_ply};
             book recommends {recognition.book_move}.
  Strategic purpose: {recognition.strategic_notes or "—"}
  Common blunders: {recognition.common_blunders or "none on record"}
  -----------------------

  Retrieved bad moves for this position: {bad_moves_context}
  Narrative memory: {narrative_memory[-300:]}

  Explain the strategic risk. If you are not confident, return an empty string.
```

**Prompt template — opening deviation question:**

```
System: You are MillsAI. Ask the human one brief, curious question about why they
        deviated from a known opening line. Do not lecture; ask as if genuinely curious.

User:
  Opening: {recognition.name} ({recognition.status})
  Book line expected: {recognition.book_move} at ply {recognition.deviation_ply}
  Human played: {recognition.deviation_move}
  Strategic context: {recognition.strategic_notes}

  Generate a single concise question for the human about their deviation.
  Example: "You deviated from the Mill Rush at ply 5 — was that a deliberate counter
  to my d6, or are you trying a different approach to the centre?"
```

### `ai/coordinator.py` — AI Dialogue Coordinator

```python
class Coordinator:
    game_ai: GameAI
    mills_llm: MillsLLM
    recognizer: OpeningRecognizer
    dialogue_log: list[str]

    def deliberate(self, board: BoardState) -> dict:
        """
        1. GameAI generates ranked candidate moves (passes recognition to choose_move)
        2. MillsLLM opines on candidates (receives recognition in ask_for_move_opinion)
        3. Both agree or GameAI overrides on timeout
        4. Final move returned; dialogue_log updated
        5. MillsLLM may append a question for the human
        """

    def react_to_human_move(
        self,
        board_before: BoardState,
        board_after: BoardState,
        human_move: dict,
    ) -> None:
        """
        1. Update recognizer with human's move notation
        2. GameAI scores the move (score_before and score_after)
        3. If score drop >= poor_move_threshold: MillsLLM.evaluate_human_move()
           (passes recognition so comment can reference the opening by name)
        4. If recognition shows a deviation from a book line and llm_question_frequency
           permits: MillsLLM generates a deviation question for the human
        5. Comment and question emitted to chat panel; recorded in game record
        """

    def on_game_start(self) -> None:
        """Reset recognizer; inject opening pattern analysis from prior games."""

    def emit(self, speaker: str, text: str, tag: str = "normal"): ...
```

**Sample dialogue — named opening recognised:**

```
[GameAI]   Evaluating 16 candidates. Book move for Mill Rush: f4 (score +0.76 + bonus)
[MillsLLM] We're in the Mill Rush. f4 is the standard continuation — it prepares the
           f2-f4-f6 mill while contesting b4. I agree with GameAI.
[MillsLLM] Human, you played c3 rather than the expected d5 block. Are you going for
           an inner ring counter, or did you have a different plan?
```

**Sample dialogue — human poor move with opening context:**

```
[MillsLLM] In the Mill Rush, b6 at this ply is a known weak response — it cedes
           the centre and lets me close f2-f4-f6 unopposed. d5 would have been the
           stronger counter here.
```

### `ai/memory_manager.py` — Memory & Learning

```python
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

class MemoryManager:
    chroma_client: chromadb.PersistentClient
    positions_collection: chromadb.Collection
    games_collection: chromadb.Collection

    def store_bad_move(self, board_fen, move, reason, full_board_ascii): ...
    def retrieve_similar_positions(self, board_fen, n_results=5) -> list[dict]: ...
    def save_game_record(self, record: dict): ...
    def load_recent_games(self, n=10) -> list[dict]: ...
    def save_session_narrative(self, text: str): ...
    def retrieve_relevant_narratives(self, query, n=3) -> list[str]: ...
    def analyse_patterns(self, recent_games) -> dict: ...
```

**Note:** `MemoryManager` does **not** store or retrieve opening lines. All structured
opening knowledge lives exclusively in `opening_book.py` + `openings.json`.

**Game record schema (JSONL):**
```json
{
  "session_id": "uuid",
  "date": "2026-05-16T10:32:00",
  "human_color": "W",
  "winner": "B",
  "recognised_opening_id": "mill-rush-white",
  "recognised_opening_name": "Mill Rush",
  "opening_recognition_status": "exact",
  "opening_deviation_ply": null,
  "moves": [
    {
      "turn": 1,
      "color": "W",
      "type": "place",
      "from": null,
      "to": "d2",
      "capture": null,
      "notation": "d2",
      "board_fen_before": "...",
      "game_ai_score": 0.12,
      "llm_opinion": "Solid central placement.",
      "human_feedback": null,
      "llm_poor_move_comment": null,
      "score_delta": null,
      "opening_recognition": {
        "status": "exact",
        "name": "Mill Rush",
        "confidence": 1.0,
        "book_move": "d2",
        "deviation": false
      },
      "endgame_phase": null,
      "endgame_pattern": null
    }
  ],
  "bad_moves_taught": [],
  "llm_summary": "Human played Mill Rush opening but deviated at ply 5..."
}
```

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  [Settings ▼]  Thinking: ████░░ 3s   Difficulty: ●●●○○  [Export] [Debrief] [Import & Analyse]│
├───────────────────────────────┬─────────────────────────────────┤
│                               │  Opening                        │
│      NINE MEN'S MORRIS        │  Mill Rush (exact, 100%)        │
│                               │  [Show book line ☐]             │
│   (Pygame canvas 500×500px)   ├─────────────────────────────────┤
│                               │  AI Discussion                  │
│   Board drawn with:           │  ┌─────────────────────────┐   │
│   - 3 concentric squares      │  │[GameAI] Book: f4 +bonus │   │
│   - connecting lines          │  │[MillsLLM] Mill Rush — f4│   │
│   - coloured piece circles    │  │[MillsLLM ⚠] b6 is weak │   │
│   - highlighted valid moves   │  └─────────────────────────┘   │
│   - last move indicator       │                                 │
│   - book-line ghost piece     │  Human input:                   │
│                               │  ┌─────────────────────────┐   │
│                               │  │ Type move feedback...   │   │
│   Status: White's turn        │  └─────────────────────────┘   │
│   Phase:  Placement (5/9)     │  [Send Feedback]  [Bad Move!]   │
│                               │                                 │
│                               │  White ●: 7   Black ○: 6       │
└───────────────────────────────┴─────────────────────────────────┘
```

### UI Modules

#### `ui/board_renderer.py` (Pygame)
- Draw three concentric squares and connecting lines on a 500×500 canvas
- Render pieces as filled circles (white/black with border)
- Highlight legal move targets on hover (yellow tint)
- Animate piece placement (fade-in) and capture (fade-out)
- Mark mills with a coloured overlay line
- Accept click events → return selected position string
- **Book-line ghost piece**: when "Show book line" is enabled, render the book's
  recommended next move as a translucent ghost piece on the board

#### `ui/opening_panel.py` (Tkinter)
A compact panel displayed above the AI chat panel, showing:
- **Opening name** and recognition status (e.g. "Mill Rush — exact, 100%")
- "Unknown (novel)" if no opening matched
- A **"Show book line" checkbox**: when ticked, `board_renderer` overlays the ghost piece
- A **"Why this opening?" button**: triggers `MillsLLM` to generate a brief
  explanation of the opening's strategic purpose in the chat panel
- Updates after every placement-phase move via the `dialogue_queue`

#### `ui/chat_panel.py` (Tkinter)
- Scrollable `Text` widget, read-only
- Color-tagged speakers:
  - `[GameAI]` — blue
  - `[MillsLLM]` — green
  - `[MillsLLM ⚠]` — amber (poor-move comments and opening deviation questions)
  - `[Human]` — orange
- Auto-scrolls to bottom on new message
- Human feedback entry + "Send Feedback" button
- "Bad Move!" button marks the last AI move as bad; opens a dialog for reason entry

#### `ui/settings_panel.py` (Tkinter)
- **Thinking Time**: slider 1–10 seconds (default 3s)
- **Difficulty**: 1–5 radio buttons (default 3)
- **Colour Choice**: White / Black / Random radio buttons
- **Ollama Model**: text entry (default `llama3.2`)
- **Ollama URL**: text entry (default `http://localhost:11434`)
- **LLM Move Commentary**: toggle on/off (default on)
- **Commentary Sensitivity**: slider for score-drop threshold 0.1–1.0 (default 0.3)
- **Opening Book**: toggle on/off (default on)
- **Opening Book Bonus**: slider 0.0–0.5 (default 0.2)
- Settings persisted to `data/settings.json`

#### `ui/replay_panel.py` (Tkinter Toplevel + Pygame)
- Opens as a separate `tk.Toplevel` window; does not block the main game window
- Left pane: read-only Pygame board at the selected ply; last-move highlight; best-move ghost piece
- Right pane: `[DebriefAI]` commentary panel with its own scroll history
- Bottom: score graph (line chart); navigation controls; auto-play toggle
- Critical moment positions marked with ★ on the graph and in the move drop-down
- "Analyse Position" and "Export Debrief" buttons
- Progress bar shown while `DebriefEngine.analyse()` is running in background

#### `ui/main_window.py`
- Tkinter root window
- Left pane: Pygame canvas via `os.environ['SDL_WINDOWID']` embed
- Right pane: opening panel → chat panel → piece counts → status bar
- Top bar: settings toggle, thinking meter, difficulty indicator, export button,
  "Debrief" button (reopens last live debrief), "Import & Analyse" button (file picker)

---

## Human Feedback & Bad Move Teaching

### Workflow
1. After any AI move, the human may click **"Bad Move!"**
2. A dialog opens: *"Why was this a bad move?"* (free text)
3. The system calls `MemoryManager.store_bad_move()`
4. `MillsLLM.narrative_memory` is updated
5. On subsequent turns, `retrieve_similar_positions()` results are injected into the LLM prompt

### Human Feedback via Chat Box
- If the message begins with "!" it is treated as a game rule clarification
- Otherwise appended to `conversation_history` and the LLM may respond

---

## LLM Commentary on Human Moves

### Trigger Conditions
The LLM generates a proactive comment when **all** hold:
1. `settings.llm_move_commentary` is enabled
2. The human has just confirmed a move
3. Score delta ≤ `-settings.poor_move_threshold`
4. The LLM thread is not already busy

### Opening-Aware Commentary
When a named opening is recognised, comments reference it explicitly:
- "In the Mill Rush, …"
- "This deviates from the standard Mill Rush continuation at ply 5 — …"

If the human deviated from a book line and `llm_question_frequency` permits, the
coordinator also triggers an **opening deviation question** (separate from the poor-move
comment) so the LLM can ask why the human diverged.

### Confidence Gating
- LLM prompt instructs: return empty string if not confident
- Empty response suppresses output entirely

### Frequency Limiting
- Max `max_poor_move_comments_per_game` (default 5) proactive comments per game
- Minimum 2 turns between consecutive proactive comments

---

## Opening Knowledge Integration Points

| Component | Role |
|---|---|
| `OpeningBook` | Load, query, persist structured opening data |
| `OpeningRecognizer` | Update after each placement move; maintain `RecognitionResult` |
| `GameAI.choose_move` | Receive `RecognitionResult`; apply book bonus/blunder penalty |
| `Coordinator.deliberate` | Pass recognition to both GameAI and MillsLLM |
| `Coordinator.react_to_human_move` | Update recognizer; detect deviations; trigger deviation question |
| `MillsLLM` prompts | Include opening context block in all AI-turn and coaching prompts |
| `ui/opening_panel.py` | Display recognised opening name and confidence live |
| `ui/board_renderer.py` | Render ghost piece for book-line preview |
| `MemoryManager.save_game_record` | Record `recognised_opening_id` and per-move `opening_recognition` |
| `OpeningBook.update_outcome_stats` | Called at game end with the winner |
| `OpeningBook.save_novel_opening` | Called at game end if no opening was recognised |
| Export (`notation.py`) | Optionally annotate moves with book deviation markers |

---

## Endgame Recognition System

The Endgame Recognition System is a **first-class, book-grounded subsystem** that detects
which named endgame phase the game has entered, identifies specific named winning or drawing
arrangements, adjusts the AI's evaluation function accordingly, and supplies MillsLLM with
phase-specific strategic guidance drawn directly from the source book.

It activates once the total number of pieces on the board drops to a recognised endgame
threshold and remains active for the rest of the game.

---

### Endgame Phases and Book Assessment

Derived from the strategy source material (chapters 16.1–16.7 and section 8.5):

| Label | Piece counts | Book assessment | Winning strategy |
|---|---|---|---|
| `7v4` | 7 vs 4 | Stronger side usually wins | Create three simultaneous open mills; restrict weaker side to two squares |
| `7v3` | 7 vs 3 | Stronger side wins | Force weaker side out of blocking positions; zugzwang |
| `6v4` | 6 vs 4 | Hard; may draw at expert level | Two simultaneous open mills; or allow opponent to reduce to 3 |
| `6v3` | 6 vs 3 | Stronger side usually wins | Herding; restrict to two squares |
| `5v4` | 5 vs 4 | Stronger side mild advantage | Steer toward mill pairs |
| `5v3` | 5 vs 3 | Stronger side advantage | Force reduction |
| `4v4` | 4 vs 4 | Likely draw at expert level | Race to reach 3 pieces first |
| `4v3` | 4 vs 3 | 4-side: force mill; 3-side: pin | 4-side needs potential double mill; 3-side must pin fourth piece |
| `3v3` | 3 vs 3 | Initiative decides; draw possible | First to 3 pieces attacks; second must block two mill lines simultaneously |

**Endgame begins** when total pieces ≤ 11 (i.e., average ≤ 5.5 per side).
Below 8 total pieces the system is in **deep endgame** and all book patterns are fully active.

---

### Named Endgame Patterns

Derived from the book's named figures and arrangements:

**7v4 and 6v4 Arrangements (6 each — Figures 239–250):**
- `7v4-arrangement-{1..6}`: Six board configurations where the stronger side has three
  simultaneous open mills that the four remaining pieces cannot simultaneously block.
  Also applicable to 6v4 (same arrangements, book notes the structural similarity).
- Recognition: FEN-signature matching at the start of the weaker player's turn.
- Strategic note: "Restrict the opponent's pieces to two squares; remove the piece that
  most blocks the three-mill arrangement."

**3v3 Patterns:**
- `3v3-infinite-loop`: A draw configuration where neither side can force a win.
  Characterised by both sides cycling through the same positions. Detection: position
  repeated three times (threefold repetition).
- `3v3-skipping-side`: A winning pattern where the stronger side escapes blocking by
  moving to the opposite side of the board and establishing dual potential mills at
  cardinal lines. Detectable by specific piece-position signatures.

**Forced Zugzwang:**
- `zugzwang-detected`: Any position where the current player's every legal move
  worsens their evaluation by ≥ `settings.endgame_zugzwang_threshold`.
  Applies most commonly in 3v3, 4v3, and 7v4 positions.

**Herding:**
- `herding-active`: The weaker side has all their pieces restricted to two or fewer
  distinct squares (adjacency-connected cluster ≤ 2). The stronger side is close to
  a forced win.

---

### Endgame Pattern Data Schema (`data/endgames/endgame_patterns.json`)

```json
{
  "pattern_id": "7v4-arrangement-1",
  "name": "7v4 Arrangement 1",
  "phase_label": "7v4",
  "source_reference": "Figure 239, Chapter 16.3",
  "assessment": "winning_stronger",
  "board_fen_signatures": ["...fen_at_this_arrangement..."],
  "stronger_side_strategy": "Remove the piece blocking the third potential mill, then alternate between two open mills. The opponent cannot fill both simultaneously.",
  "weaker_side_strategy": "Keep pieces as spread as possible; avoid being herded to one corner.",
  "tags": ["arrangement", "three-open-mills", "7v4", "6v4"],
  "zugzwang": false,
  "herding": false
}
```

**Field glossary:**

| Field | Description |
|---|---|
| `pattern_id` | Machine slug |
| `name` | Human-readable pattern name |
| `phase_label` | `"7v4"`, `"6v4"`, `"3v3"`, etc. |
| `source_reference` | Figure / chapter from the strategy book |
| `assessment` | `"winning_stronger"` \| `"winning_weaker"` \| `"draw"` \| `"zugzwang"` |
| `board_fen_signatures` | FEN strings used for recognition matching |
| `stronger_side_strategy` | Plain-language strategy from the book |
| `weaker_side_strategy` | Plain-language defence / drawing strategy |
| `tags` | `arrangement`, `herding`, `zugzwang`, `three-open-mills`, `infinite-loop`, etc. |
| `zugzwang` | True if this pattern represents a forced zugzwang position |
| `herding` | True if the weaker side is restricted to ≤2 squares |

---

### `ai/endgame_recognizer.py` — Endgame State Detection

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EndgameState:
    active: bool                      # True once endgame thresholds are crossed
    phase_label: str                  # "7v4", "6v4", "4v4", "4v3", "3v3", "midgame", etc.
    stronger_side: Optional[str]      # "W", "B", or None if equal
    weaker_side: Optional[str]
    piece_counts: dict                # {"W": int, "B": int}
    total_pieces: int

    # Assessment
    assessment: str                   # "winning" | "drawing" | "losing" | "unknown"
                                      # from board.turn's perspective

    # Named pattern match
    named_pattern: Optional[str]      # e.g. "7v4-arrangement-1", "3v3-infinite-loop"
    pattern_confidence: float         # 0.0–1.0
    pattern_source: str               # figure / chapter reference

    # Structural metrics (computed each call)
    open_mills: dict                  # {"W": int, "B": int}
    potential_mills: dict             # {"W": int, "B": int} (one piece away from mill)
    feeder_pieces: dict               # {"W": int, "B": int} (support an open mill)
    mobility: dict                    # {"W": int, "B": int} (total legal moves available)
    restricted_pieces: dict           # {"W": int, "B": int} (pieces with ≤2 empty neighbors)
    zugzwang_detected: bool
    herding_active: bool              # weaker side in ≤2 square cluster

    # Book strategy for current phase
    stronger_strategy: str
    weaker_strategy: str
    phase_notes: str                  # general endgame notes for this phase label

class EndgameRecognizer:
    def __init__(self, patterns_path: str = "data/endgames/endgame_patterns.json"): ...

    def update(self, board: BoardState) -> EndgameState:
        """
        Called after every move.  Computes the full EndgameState.

        Algorithm:
        1. PHASE LABEL
           Determine stronger/weaker side from piece counts.
           Look up phase_label from ENDGAME_THRESHOLDS table.
           If total pieces > 11, return EndgameState(active=False, phase_label="midgame").

        2. STRUCTURAL METRICS
           - open_mills: count mills of each colour that are complete
           - potential_mills: count lines where colour has 2 pieces and 1 empty
           - feeder_pieces: pieces adjacent to an open mill of own colour
           - mobility: len(board.legal_moves(color)) for each color
           - restricted_pieces: own pieces where all adjacents are occupied or own colour

        3. ZUGZWANG DETECTION
           For each legal move of board.turn, tentatively apply it and score.
           If every resulting score is worse than current: zugzwang_detected = True.
           Only runs in deep endgame (total_pieces ≤ 8) to avoid performance cost.

        4. HERDING DETECTION
           Find the connected cluster sizes of the weaker side's pieces.
           herding_active = True if max cluster ≥ all pieces (all in one cluster)
           AND all pieces are within 2 squares of each other.

        5. NAMED PATTERN RECOGNITION
           Compare board.to_fen_string() against endgame_patterns[phase_label].
           If FEN matches: named_pattern = pattern_id, confidence = 1.0.
           If not exact: compute structural similarity score against pattern tags
           (open_mills count matches, herding matches, zugzwang matches).
           Assign confidence proportionally.

        6. ASSESSMENT
           Combine phase label default + named pattern assessment + structural metrics
           to produce 'winning' / 'drawing' / 'losing' / 'unknown' from board.turn's view.
        """

    def get_current_state(self) -> EndgameState: ...

    def reset(self) -> None:
        """Called at game start."""
```

**Phase threshold table (hardcoded):**

```python
ENDGAME_THRESHOLDS = {
    # (stronger_count, weaker_count): (label, default_assessment)
    (7, 4): ("7v4",  "winning_stronger"),
    (7, 3): ("7v3",  "winning_stronger"),
    (6, 4): ("6v4",  "unclear"),
    (6, 3): ("6v3",  "winning_stronger"),
    (5, 4): ("5v4",  "slight_stronger"),
    (5, 3): ("5v3",  "winning_stronger"),
    (4, 4): ("4v4",  "draw_likely"),
    (4, 3): ("4v3",  "unclear"),
    (3, 3): ("3v3",  "initiative_decides"),
}
```

---

### `ai/heuristics.py` — Endgame Evaluation Updates

The existing Kukreja heuristic is tuned for midgame play. Endgame requires a separate
coefficient set and additional terms.

```python
def endgame_score(board: BoardState, color: str, state: EndgameState) -> float:
    """
    Phase-specific evaluation. Called by GameAI.choose_move when
    endgame_state.active is True.

    Additional terms beyond the Kukreja base:
      + w_mobility    * (mobility[color] - mobility[opponent])
      + w_feeder      * (feeder_pieces[color] - feeder_pieces[opponent])
      + w_restrict    * restricted_pieces[opponent]   (opponent pieces trapped)
      - w_restrict    * restricted_pieces[color]      (own pieces trapped — bad)
      + w_zugzwang    * 5.0  (if zugzwang_detected and opponent is to move)
      - w_zugzwang    * 5.0  (if zugzwang_detected and own side is to move)
      + w_herding     * 3.0  (if herding_active and color is stronger_side)
      - w_herding     * 3.0  (if herding_active and color is weaker_side)

    Coefficients by phase:
    """

ENDGAME_WEIGHTS = {
    "7v4": {"mobility": 0.3, "feeder": 0.4, "restrict": 0.8, "zugzwang": 1.0, "herding": 1.2},
    "6v4": {"mobility": 0.4, "feeder": 0.5, "restrict": 1.0, "zugzwang": 1.0, "herding": 1.5},
    "5v4": {"mobility": 0.4, "feeder": 0.4, "restrict": 0.6, "zugzwang": 0.8, "herding": 1.0},
    "4v4": {"mobility": 0.6, "feeder": 0.5, "restrict": 0.5, "zugzwang": 1.0, "herding": 0.5},
    "4v3": {"mobility": 0.8, "feeder": 0.6, "restrict": 0.8, "zugzwang": 1.2, "herding": 1.0},
    "3v3": {"mobility": 1.2, "feeder": 0.8, "restrict": 1.0, "zugzwang": 1.5, "herding": 1.0},
}
```

**Corner vs. cardinal value inversion (fly phase):**
In the fly phase (3v3, 4v3) the book notes that corner pieces have the same value as
cardinal-point pieces — the standard position-value table is flat. `endgame_score` should
use flat position weights when `phase_label in ("3v3", "4v3", "3v4")`.

---

### `ai/game_ai.py` — Endgame Integration

```python
def choose_move(
    self,
    board: BoardState,
    recognition: RecognitionResult | None = None,
    endgame_state: EndgameState | None = None,
) -> dict:
    """
    Extended signature.  Endgame branch:
    - If endgame_state.active:
        - Use endgame_score() instead of the Kukreja base as the leaf evaluator
        - In 3v3 / 4v3: increase search depth by +1 (mobility calculation benefits
          from deeper look-ahead)
        - If a named pattern matches a known win with confidence >= 0.9:
            score the pattern's recommended move directly (+pattern_bonus) and
            return early if it scores > all minimax alternatives
    - Otherwise: normal Kukreja evaluation with opening book bonuses.
    """
```

---

### `ai/coordinator.py` — Endgame Integration

```python
class Coordinator:
    ...
    recognizer: OpeningRecognizer
    endgame_recognizer: EndgameRecognizer   # new

    def deliberate(self, board: BoardState) -> dict:
        """
        Now also calls endgame_recognizer.update(board) and passes the resulting
        EndgameState to both game_ai.choose_move() and mills_llm.ask_for_move_opinion().
        """

    def react_to_human_move(self, board_before, board_after, human_move):
        """
        Also calls endgame_recognizer.update(board_after) and includes
        EndgameState in the evaluate_human_move() call so coaching comments
        can reference the endgame phase by name.
        """

    def on_game_start(self):
        """
        Reset both recognizer and endgame_recognizer.
        """
```

---

### `ai/mills_llm.py` — Endgame Context in Prompts

All MillsLLM prompts gain an **Endgame Context** block (analogous to the Opening Context
block), populated when `endgame_state.active` is True.

**Endgame context block (inserted into all prompts when active):**

```
  --- Endgame Context ---
  Phase:             {state.phase_label}  (e.g. "6v4")
  Piece counts:      W={state.piece_counts['W']}  B={state.piece_counts['B']}
  Stronger side:     {state.stronger_side or "equal"}
  Assessment:        {state.assessment}  (from {board.turn}'s perspective)
  Named pattern:     {state.named_pattern or "None recognised"}
  Pattern source:    {state.pattern_source or "—"}
  Pattern confidence:{state.pattern_confidence:.0%}
  Open mills:        W={state.open_mills['W']}  B={state.open_mills['B']}
  Potential mills:   W={state.potential_mills['W']}  B={state.potential_mills['B']}
  Feeder pieces:     W={state.feeder_pieces['W']}  B={state.feeder_pieces['B']}
  Mobility:          W={state.mobility['W']}  B={state.mobility['B']}
  Zugzwang:          {state.zugzwang_detected}
  Herding active:    {state.herding_active}
  Stronger strategy: {state.stronger_strategy}
  Weaker strategy:   {state.weaker_strategy}
  -----------------------
```

**Sample LLM output enabled by this context:**

```
[MillsLLM] We've entered a 6v4 endgame. White needs to create two simultaneous
           open mills — the opponent can't fill both. Keep the pieces on f2, d2,
           and b2 as feeders. Don't chase the last piece yet.

[MillsLLM] This position matches the 6v4 Arrangement 3 from the book — your best
           move is to open the mill at b2-d2-f2 and force Black to fill one side,
           then close the other.
```

---

### `ui/opening_panel.py` — Endgame Display

The existing opening panel is extended (or a second row added) to show the endgame state
when active:

```
┌──────────────────────────────────────────┐
│  Opening: Mill Rush (exact, 100%)        │
│  [Show book line ☐]                      │
├──────────────────────────────────────────┤
│  Endgame: 6v4 — White advantage          │
│  Pattern: Arrangement 3 (92%)            │
│  Zugzwang: No   Herding: No              │
└──────────────────────────────────────────┘
```

- Opening row hidden once endgame is active (placement phase is over).
- Endgame row shown from piece-count threshold onwards.
- Assessment colour: green (winning), amber (unclear/draw), red (losing) from human's view.

---

### Endgame Integration Points

| Component | Role |
|---|---|
| `EndgameRecognizer` | Update after every move; compute full `EndgameState` |
| `Coordinator.deliberate` | Pass `EndgameState` to `GameAI` and `MillsLLM` |
| `Coordinator.react_to_human_move` | Include endgame context in coaching comments |
| `GameAI.choose_move` | Switch to `endgame_score()` when `endgame_state.active` |
| `heuristics.endgame_score` | Phase-weighted evaluation with zugzwang / herding / mobility terms |
| `MillsLLM` prompts | Endgame context block added when active |
| `ui/opening_panel.py` | Extended row showing phase label, named pattern, assessment |
| `DebriefEngine` | Include `EndgameState` per ply in `DebriefReport` for richer commentary |
| Game record | Add `endgame_phase` and `endgame_pattern` fields to per-move entries |

---

## Pattern Recognition

```python
def analyse_patterns(self, recent_games: list[dict]) -> dict:
    """
    Returns:
    - most_common_opening_positions (top 5 first 6 moves)
    - human_weakness_positions
    - human_preferred_placements
    - successful_mill_setups
    - most_played_openings (from recognised_opening_id fields)
    - opening_win_rates (per opening family, from recent game records)
    """
```

---

## Session End & Memory Consolidation

At game/session end:

1. **GameAI** logs final score and pattern analysis delta
2. **OpeningBook** `update_outcome_stats` called with the game winner
3. If no opening was recognised and placement sequence has ≥ 6 moves,
   `OpeningBook.save_novel_opening` creates a new learned opening entry
4. **MillsLLM** `summarise_session` generates narrative including opening used:
   ```markdown
   ## Session 2026-05-16
   Human played White. Computer won in 24 moves.
   Opening: Mill Rush — human deviated at ply 5 (c3 instead of d5).
   Notable: Human taught us that e3-e5 is weak when b4 is contested.
   Key patterns: human delays inner ring placement.
   ```
5. Summary saved to `data/session_memory/` and embedded into ChromaDB
6. `MillsLLM.narrative_memory` reset; `conversation_history` cleared

---

## Post-Game Debrief System

The debrief system lets both AIs jointly replay and annotate any game — the just-finished
live game or an externally imported file — stepping forward and backward through positions,
identifying the critical turning point, and explaining in plain language where the losing
side went wrong.

The debrief opens in a dedicated `Toplevel` window so it does not disturb the main game UI.
It is available immediately at game end and can also be opened manually at any time via the
top bar "Debrief" button.

---

### `game/game_importer.py` — Notation File Parser

```python
class GameImporter:
    def load_from_file(self, path: str) -> dict:
        """
        Parse a .txt or .pgn-style export file produced by notation.py.
        Reconstructs the full game_record dict (same schema as JSONL records)
        by replaying each move through game_engine to rebuild board states.

        Accepts:
        - The standard two-column format (1. d2 d6  2. f4 b4 ...)
        - Annotated format with {?} deviation markers
        - The [Opening: ...] header comment if present

        Returns a game_record dict with board_fen_before populated for each move
        and source="imported".  game_ai_score, llm_opinion fields will be null
        until DebriefEngine.analyse() is run.

        Raises ValueError with a descriptive message if any move in the file is illegal.
        """

    def load_from_record(self, record: dict) -> dict:
        """
        Accepts an existing JSONL game record (already has board_fen_before).
        Returns it as-is, ready for DebriefEngine. Used for live-game debrief.
        """
```

---

### `ai/debrief_engine.py` — Post-Game Analysis Engine

```python
from dataclasses import dataclass, field

@dataclass
class CriticalMoment:
    ply: int
    color: str               # "W" or "B" — the side that moved
    move_played: str         # notation string
    board_fen_before: str
    score_before: float      # evaluation before the move (from mover's perspective)
    score_after: float       # evaluation after the move (from mover's perspective)
    delta: float             # score_after - score_before (negative = self-harm)
    best_move: str           # what GameAI recommends instead
    score_if_best: float     # evaluation if best_move had been played
    opportunity_lost: float  # score_if_best - score_after
    is_turning_point: bool   # True for the single worst delta for the loser
    llm_comment: str = ""    # filled in by MillsLLM.debrief_position()

@dataclass
class DebriefReport:
    game_record: dict
    source: str                         # "live" or "imported"
    board_states: list[BoardState]      # one per half-move (ply)
    evaluation_scores: list[float]      # GameAI score at each ply (White perspective)
    critical_moments: list[CriticalMoment]
    turning_point: CriticalMoment | None   # the single worst blunder for the loser
    opening_name: str | None
    loser: str | None                   # "W", "B", or None for draw
    winner: str | None
    overall_narrative: str = ""         # filled in by MillsLLM.debrief_game()
    analysed: bool = False

class DebriefEngine:
    def __init__(self, game_ai: GameAI, mills_llm: MillsLLM, opening_book: OpeningBook): ...

    def analyse(
        self,
        game_record: dict,
        source: str = "live",
        critical_threshold: float = 0.4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DebriefReport:
        """
        Full analysis pipeline. Runs in the Debrief Thread.

        1. RECONSTRUCT board states
           Replay each move from board_fen_before fields (or re-derive from move list).
           Produces a BoardState for every ply (0 = start, n = after move n).

        2. EVALUATE every position
           Call GameAI.score_move() on every played move.
           Also compute the best available move and its score at each ply using
           GameAI.choose_move() (at debrief_analysis_depth from settings).
           Store as evaluation_scores[].

        3. IDENTIFY critical moments
           For each ply, compute delta = score_after - score_before (mover's view).
           Any delta <= -critical_threshold is a CriticalMoment.
           Sort CriticalMoments by opportunity_lost descending.
           The top CriticalMoment where color == loser is the turning_point.

        4. MARK the turning point
           Set CriticalMoment.is_turning_point = True on the worst blunder for the loser.

        5. GENERATE overall narrative (async LLM call)
           Call MillsLLM.debrief_game(report) to produce the narrative.
           Store in report.overall_narrative.

        Returns the completed DebriefReport (narrative may still be pending).
        progress_callback(current_ply, total_plies) is called during evaluation
        so the UI can show a progress bar.
        """

    def analyse_position_on_demand(
        self,
        report: DebriefReport,
        ply: int,
    ) -> str:
        """
        Called when the user navigates to a specific ply in the replay panel.
        If the ply's CriticalMoment already has an llm_comment, return it.
        Otherwise call MillsLLM.debrief_position() and cache the result.
        """
```

---

### `ui/replay_panel.py` — Debrief & Replay Window

Opens as a `tk.Toplevel`. Embeds a second Pygame surface (or reuses the same renderer
in a frozen/read-only mode) to display board positions.

```
┌──────────────────────────────────────────────────────────────────┐
│  Post-Game Debrief — Mill Rush  (White wins, 28 moves)   [Close] │
├──────────────────────────────────────────┬───────────────────────┤
│                                          │  Analysis             │
│         Board at ply 14 / 28             │  ┌─────────────────┐ │
│                                          │  │[DebriefAI]      │ │
│      (Pygame canvas — read-only)         │  │ Move 14 was the │ │
│                                          │  │ turning point.  │ │
│      ★ = critical moment overlay         │  │ Black played    │ │
│                                          │  │ c3 but d5 was   │ │
│                                          │  │ essential...    │ │
│                                          │  └─────────────────┘ │
│                                          │                       │
│                                          │  [Analyse Position]   │
│                                          │  [Export Debrief]     │
├──────────────────────────────────────────┴───────────────────────┤
│  Score graph (White positive ↑, Black positive ↓)                │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  +1 ┤     .─.                                           │    │
│  │   0 ┤──────   ────────────────.   .─────────────────── │    │
│  │  -1 ┤                          `─'  ★                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│         Move:  1   5   10   ★14   20   25   28                   │
├──────────────────────────────────────────────────────────────────┤
│  [|◀ Start]  [◀◀ -5]  [◀ Prev]  Move 14 ▼  [Next ▶]  [+5 ▶▶]  [End ▶|] │
│  [Jump to turning point ★]   Speed: ○ Step  ○ Auto 1s  ○ Auto 0.5s      │
└──────────────────────────────────────────────────────────────────┘
```

**Features:**

- **Score graph**: Matplotlib-rendered (or custom Pygame canvas) line chart of
  `evaluation_scores[]`. White-positive up, Black-positive down. Critical moments
  marked with a ★ symbol on the graph. Clicking a ★ on the graph jumps to that ply.
- **Board display**: Read-only Pygame board showing the position at the selected ply.
  The move played at that ply is highlighted (last-move indicator). The best alternative
  move (from `CriticalMoment.best_move`) is shown as a ghost piece.
- **Critical moment overlay**: Positions with `is_turning_point=True` display a ★ badge.
- **Navigation controls**:
  - |◀ Start, ◀◀ −5, ◀ Prev, Next ▶, +5 ▶▶, End ▶|
  - Drop-down move selector for direct jump
  - "Jump to turning point ★" button
  - Auto-play mode with 0.5 s or 1 s step interval
- **Analysis panel**: read-only chat-style text widget showing `[DebriefAI]` commentary.
  Uses its own `conversation_history` (does not share with the live-game chat).
  - Overall narrative shown at the top (produced by `MillsLLM.debrief_game`)
  - Per-position commentary loaded on demand when user navigates to a ply
    (produced by `MillsLLM.debrief_position`)
- **"Analyse Position" button**: manually triggers `debrief_position` for the current ply
  even if it is not a critical moment.
- **"Export Debrief" button**: saves the full report to `data/debriefs/` as both
  a Markdown narrative and a JSON data file.

---

### Debrief LLM Prompt Templates

**Overall game narrative prompt (`MillsLLM.debrief_game`):**

```
System: You are MillsAI, a Nine Men's Morris coach reviewing a completed game.
        Write a structured post-game debrief. Be analytical and specific.
        Use algebraic notation. Reference the opening by name if known.
        Structure your response as:
          ## Result
          ## Opening
          ## Key Turning Point
          ## Where the loser went wrong
          ## What a stronger line looked like
          ## Lessons

User:
  Game: {loser} lost to {winner} in {total_plies} moves.
  Opening: {opening_name or "No recognised opening"}
  Opening recognition: {opening_status}

  Evaluation scores (White perspective, one per ply):
  {evaluation_scores}

  Critical moments (score drop > {threshold}):
  {critical_moments_summary}

  Turning point: ply {turning_point.ply}, {turning_point.color} played
    {turning_point.move_played} (delta {turning_point.delta:+.2f}).
    Best move was {turning_point.best_move} (opportunity lost: {turning_point.opportunity_lost:+.2f}).

  Full move list:
  {pgn_style_moves}

  Write the debrief. Be specific about positions using algebraic notation.
```

**Per-position commentary prompt (`MillsLLM.debrief_position`):**

```
System: You are MillsAI coaching a player through a game replay.
        Comment on this specific position in 2–4 sentences.
        If it is a critical moment, explain clearly why.
        Refer to the opening by name if relevant.

User:
  Ply: {ply} of {total_plies}
  Board (visual):
  {board.to_display_grid()}

  Move played: {move_played_notation} by {color}
  Score before: {score_before:+.2f}  Score after: {score_after:+.2f}  Delta: {delta:+.2f}
  Best alternative: {best_move_notation} (would have scored {score_if_best:+.2f})
  Is turning point: {is_turning_point}
  Opening context: {opening_name or "—"}
  Game context: {context}   {# e.g. "3 moves after opening deviation"}

  Explain what happened here and what the better plan was.
```

---

### Debrief Data Schema

Saved to `data/debriefs/debrief_YYYYMMDD_HHMMSS.json`:

```json
{
  "debrief_id": "uuid",
  "created": "2026-05-16T11:00:00",
  "source": "live",
  "session_id": "uuid",
  "winner": "W",
  "loser": "B",
  "total_plies": 28,
  "opening_name": "Mill Rush",
  "evaluation_scores": [0.0, 0.12, 0.08, ...],
  "turning_point_ply": 14,
  "critical_moments": [
    {
      "ply": 14,
      "color": "B",
      "move_played": "c3",
      "score_before": -0.21,
      "score_after": -0.74,
      "delta": -0.53,
      "best_move": "d5",
      "score_if_best": -0.19,
      "opportunity_lost": 0.55,
      "is_turning_point": true,
      "llm_comment": "Black's c3 abandons the centre..."
    }
  ],
  "overall_narrative": "## Result\nWhite won...",
  "pgn_moves": "1. d2 d6\n2. f4 b4\n..."
}
```

Saved to `data/debriefs/debrief_YYYYMMDD_HHMMSS.md` (the `overall_narrative` as a
standalone Markdown file for easy reading).

---

### Triggering the Debrief

| Trigger | Behaviour |
|---|---|
| Game ends naturally | A dialog appears: "Game over — open debrief?" (Yes / Later) |
| "Debrief" button (top bar) | Opens replay panel for the most recent live game record |
| "Import & Analyse" (top bar) | File picker → `GameImporter.load_from_file()` → `DebriefEngine.analyse()` → opens replay panel |
| Automatic (if `auto_open_debrief` is set) | Replay panel opens immediately at game end without prompting |

`DebriefEngine.analyse()` runs in the Debrief Thread. The replay panel opens immediately,
showing a progress bar while analysis completes. Navigation is available from ply 0
before analysis finishes; commentary is shown only for plies that have been evaluated.

---

## Export Functionality

Export button opens a dialog:
- **Format**: `.mmg` (custom), `.txt`, `.pgn-style`
- **Filename**: auto-generated as `mills_YYYYMMDD_HHMMSS.txt`
- **Annotate deviations**: optional flag to mark book-line deviations with `{!}` or `{?}`

**Annotated output example:**
```
[Opening: Mill Rush]
1. d2 d6
2. f4 b4
3. f2 f6
4. b2xf6 f6
5. b6{?} c3    {? = deviation from book line: book recommended d5}
...
14. d2-d1 *
```

---

## Settings & Configuration

```json
{
  "think_time_seconds": 3,
  "difficulty": 3,
  "human_color": "random",
  "ollama_model": "llama3.2",
  "ollama_url": "http://localhost:11434",
  "llm_question_frequency": 0.3,
  "max_chat_history_turns": 20,
  "chroma_bad_move_results": 5,
  "enable_pattern_analysis": true,
  "llm_move_commentary": true,
  "poor_move_threshold": 0.3,
  "max_poor_move_comments_per_game": 5,
  "opening_book_enabled": true,
  "opening_book_bonus": 0.2,
  "opening_blunder_penalty": 0.3,
  "show_opening_panel": true,
  "endgame_recognition_enabled": true,
  "endgame_active_threshold": 11,
  "endgame_deep_threshold": 8,
  "endgame_zugzwang_threshold": 0.4,
  "endgame_pattern_bonus": 0.3,
  "annotate_deviations_on_export": true,
  "auto_open_debrief": false,
  "debrief_analysis_depth": 4,
  "debrief_critical_threshold": 0.4,
  "debrief_window_width": 1300,
  "debrief_window_height": 750,
  "window_width": 1200,
  "window_height": 700
}
```

---

## Dependencies (`requirements.txt`)

```
pygame>=2.5.0
ollama>=0.3.0
chromadb>=0.5.0
Pillow>=10.0.0
matplotlib>=3.8.0
tkpygame>=0.2.0
```

Core stdlib: `tkinter`, `threading`, `json`, `uuid`, `datetime`, `os`, `pathlib`

---

## Threading Model

```
Main Thread:        Tkinter mainloop + Pygame event loop
AI Thread:          Coordinator.deliberate() and react_to_human_move()
LLM Thread:         MillsLLM Ollama HTTP calls (live game + debrief commentary)
Memory Thread:      ChromaDB writes (MemoryManager)
Opening Thread:     OpeningBook.save_novel_opening / update_outcome_stats (at game end)
Debrief Thread:     DebriefEngine.analyse() — position reconstruction + GameAI re-evaluation
```

Communication via `queue.Queue`:
- `ai_move_queue`: Coordinator → Main Thread (final move)
- `dialogue_queue`: Coordinator → Chat Panel + Opening Panel (live updates)
- `feedback_queue`: Human → MillsLLM (bad-move feedback)
- `debrief_progress_queue`: DebriefEngine → ReplayPanel (ply count for progress bar)
- `debrief_result_queue`: DebriefEngine → ReplayPanel (completed DebriefReport)

The Debrief Thread is only active during analysis. It shares the `GameAI` instance with
the AI Thread but acquires a lock before calling minimax, so live-game analysis is never
interrupted mid-search.

---

## Development Stages

Each stage ends with a testable milestone. Do not begin a stage until the
acceptance criteria of the previous stage pass.

---

### Stage 1 — Core Game Engine
**Goal:** Fully correct, console-testable game with no UI or AI.

| # | Module | Key deliverables |
|---|---|---|
| 1.1 | `game/board.py` | `POSITIONS`, `ADJACENCY`, `MILLS`; `BoardState` with FEN and ASCII display |
| 1.2 | `game/rules.py` | Legal moves for all three phases; terminal detection; fly/block checks |
| 1.3 | `game/notation.py` | `encode_move`, `export_pgn_style`, `parse_move_string` |
| 1.4 | `game/game_engine.py` | Turn management, phase transitions, win conditions |
| 1.5 | `tests/test_board.py` | Unit tests: adjacency, mill detection, legal move counts, notation round-trip |

**Acceptance criteria:**
- All 16 mills correctly detected.
- `legal_moves` returns zero moves for a blocked player.
- A complete game replayed from notation produces the same final board state.

---

### Stage 2 — Classical AI (Console Playable) ✅ COMPLETE
**Goal:** Human vs. GameAI in the terminal; no LLM, no UI.

| # | Module | Key deliverables |
|---|---|---|
| 2.1 | `ai/heuristics.py` | Phase-weighted evaluation; all six Kukreja terms; `endgame_score` placeholder |
| 2.2 | `ai/game_ai.py` | Negamax + alpha-beta; `choose_move`; `score_move`; **blunder mode** |
| 2.3 | `main.py` | Console harness supporting human vs AI and human vs human with `--difficulty`, `--human`, `--blunder`, `--hvh` CLI flags |

**Blunder mode** (`blunder_probability` parameter on `GameAI`): when set > 0, the AI occasionally plays a deliberately poor move drawn from the bottom quartile of scored moves. `GameAI.last_was_blunder` flag lets the Coordinator announce the mistake to MillsLLM so it can invite the human to identify the correct response. Run with `python main.py --blunder 0.3` for ~30% mistake rate.

**Acceptance criteria (all passing, 17 tests in `tests/test_ai.py`):**
- Difficulty 3 picks a legal, non-trivially bad move within 3 seconds.
- `score_move` returns a lower score for a known blunder than for the optimal move.
- `blunder_probability=1.0` always plays a bad move; `0.0` never does.

---

### Stage 3 — Memory & LLM Layer
**Goal:** ChromaDB and Ollama responding; no UI yet.

| # | Module | Key deliverables |
|---|---|---|
| 3.1 | `ai/memory_manager.py` | ChromaDB init; `store_bad_move`, `retrieve_similar_positions`, `save_game_record`; **seeded strategy knowledge** |
| 3.2 | `ai/mills_llm.py` | `ask_for_move_opinion`, `evaluate_human_move`, `record_human_feedback`, `summarise_session` |
| 3.3 | `ai/coordinator.py` | `deliberate`, `react_to_human_move`; dialogue log as plain text; blunder announcement |

#### Seeded Strategy Knowledge (ChromaDB)

On first startup, `MemoryManager` pre-loads a collection of midgame strategy entries from the strategy book so MillsLLM can retrieve contextually relevant advice when commenting on moves. These are stored in a dedicated `strategy_knowledge` ChromaDB collection (separate from bad-move memory) and are never modified at runtime.

**Seeded entries — Midgame Tactics (Chapter 14):**

| ID | Topic | Core concept |
|----|-------|-------------|
| `midgame_01` | Mill hierarchy (priorities) | Double mill with feeders > double mill > mill + feeder > two independent mills > single mill. Use this to decide which piece to capture and when to make a mill. |
| `midgame_02` | Cannon fodder | Form mills that are purely targets to force the opponent to remove a specific piece rather than the one they'd prefer. Useful when blocking a double-mill setup. |
| `midgame_03` | When to allow a mill | Sometimes letting the opponent form a mill is better than blocking it — especially when blocking would over-commit pieces and reduce your mobility. |
| `midgame_04` | Mill redundancy | An opponent's mill can be made redundant when you have an open mill threatening to close. They are forced to stay put while you manoeuvre. Patience is required. |
| `midgame_05` | Mill abandonment | Abandoning your own mill frees your pieces while the opponent must stay in position to hold the mill closed. Gains positional advantage at the cost of the mill. |
| `midgame_06` | Sacrificial mills | Give up a potential mill to disable the opponent's double-mill setup. Often forces them into a series of captures that ultimately leaves you with the upper hand. |
| `midgame_07` | Cardinal point abandonment | Vacating a cardinal point (midpoint of a ring side) can force opponent pieces into worse positions, link three of your own pieces, or bait a forced response. Never abandon without a reason. |
| `midgame_08` | Feeder pieces | Pieces adjacent to an open mill that can close it on the next move. A mill with a feeder is worth significantly more than a standalone mill — plan for feeders before closing. |

**Seeded entries — Blunder-mode learning:**

| ID | Topic | Core concept |
|----|-------|-------------|
| `blunder_01` | Capitalising on unprotected pieces | When the opponent's piece is unprotected (no adjacent own pieces), prioritise moves that threaten or capture it before they can retreat. |
| `blunder_02` | Identifying deliberate AI mistakes | When the AI flags `last_was_blunder`, look for: an open mill that could have been blocked, a piece left isolated, a cardinal point vacated without reason, or a missed capture opportunity. |

**Retrieval in MillsLLM:** when `react_to_human_move` or `deliberate` is called, embed the current board FEN and query `strategy_knowledge` for the top-3 nearest entries. Include retrieved excerpts in the LLM system prompt as background context.

**Blunder announcement in Coordinator:** when `ai.last_was_blunder` is True after `choose_move`, the Coordinator calls `mills_llm.announce_blunder(board, move)` which generates a prompt like: *"I just made a mistake there — can you spot what I should have done instead?"* This is displayed in the chat panel.

**Acceptance criteria:**
- `evaluate_human_move` returns non-empty for delta −0.5, empty for delta −0.05.
- `store_bad_move` → `retrieve_similar_positions` on the same FEN returns the stored entry.
- `deliberate` completes within `think_time` + 2 s and returns a legal move.
- On first startup, `strategy_knowledge` collection contains exactly 10 entries (8 midgame + 2 blunder).
- A query for "I abandoned my mill to gain mobility" returns `midgame_05` as top result.

---

### Stage 4 — Opening Knowledge System
**Goal:** Curated openings loaded, recognised in real time, and influencing AI play.

| # | Module | Key deliverables |
|---|---|---|
| 4.1 | `data/openings/book_openings.json` | At least 10 named opening families from strategy literature, validated by import tool |
| 4.2 | `tools/import_openings.py` | CLI importer with `--validate` and `--dry-run`; all book lines pass legality checks |
| 4.3 | `ai/opening_book.py` | `OpeningBook`: load, query, `save_opening`, `update_outcome_stats`, `record_deviation`, `save_novel_opening` |
| 4.4 | `ai/opening_recognizer.py` | `OpeningRecognizer`: exact prefix, deviation, FEN transposition, confidence scoring |
| 4.5 | `GameAI` integration | `choose_move` accepts `RecognitionResult`; book bonus and blunder penalty applied |
| 4.6 | `MillsLLM` integration | All relevant prompts include opening context block |
| 4.7 | `Coordinator` integration | `react_to_human_move` updates recognizer; deviation question triggered |

**Acceptance criteria:**
- Given the first 6 moves of a known opening, `OpeningRecognizer` returns `status="exact"`.
- Given the first 5 moves of a known opening followed by a book-listed blunder,
  `status="deviation"` is returned and `deviation_ply` is correct.
- Given a transposed move order that produces the same board after 4 plies,
  `status="transposition"` is returned.
- In a console test, GameAI with recognition enabled chooses the book move (or a move
  within 0.2 of it) for at least the first 3 plies of a known opening at difficulty 3.
- At game end, `update_outcome_stats` correctly increments the winner's counter.
- A novel 8-move placement sequence is saved to `openings.json` with `seed_source="learned"`.

---

### Stage 5 — Endgame Recognition System
**Goal:** Named endgame phases detected in real time; GameAI switches to endgame evaluation; MillsLLM discusses the endgame by name.

| # | Module | Key deliverables |
|---|---|---|
| 5.1 | `data/endgames/endgame_patterns.json` | All named patterns from the strategy book: 7v4 Arrangements 1–6, 6v4 Arrangements 1–6, 3v3 Infinite Loop, 3v3 Skipping Side, validated FEN signatures |
| 5.2 | `ai/endgame_recognizer.py` | `EndgameRecognizer.update()`: phase label, structural metrics (open/potential mills, feeders, mobility, restricted pieces), zugzwang detection, herding detection, FEN pattern match |
| 5.3 | `ai/heuristics.py` extension | `endgame_score()`: phase-weighted coefficients per `ENDGAME_WEIGHTS`; flat position table in fly phase; zugzwang, herding, mobility, feeder terms |
| 5.4 | `ai/game_ai.py` extension | `choose_move` accepts `EndgameState`; routes to `endgame_score()` when active; depth +1 for 3v3/4v3; pattern-bonus early return |
| 5.5 | `ai/coordinator.py` extension | `on_game_start` resets `endgame_recognizer`; `deliberate` and `react_to_human_move` pass `EndgameState` to both AI and LLM |
| 5.6 | `ai/mills_llm.py` extension | Endgame context block inserted in all prompts when `endgame_state.active` |

**Acceptance criteria:**
- A board with 7 White and 4 Black pieces triggers `phase_label="7v4"` and `active=True`.
- A board with 12 total pieces returns `active=False`.
- A board position matching a stored FEN signature returns the correct `named_pattern` with `confidence=1.0`.
- Zugzwang detection returns `True` for a hand-constructed position where every legal move worsens White's score by ≥ `endgame_zugzwang_threshold`.
- Herding detection returns `True` when all Black pieces are within a 2-square cluster.
- In a console test, `GameAI.choose_move` at difficulty 3 produces a different (endgame-appropriate) move in a 4v3 position when `endgame_state.active=True` versus using only Kukreja evaluation.
- `MillsLLM` prompt for a 6v4 position contains the endgame context block with correct `phase_label`, `assessment`, and `stronger_strategy`.
- All 14 named patterns load correctly from `endgame_patterns.json` without validation errors.

---

### Stage 6 — Pygame Board Renderer
**Goal:** Visual board with click input and book-line ghost piece.

| # | Module | Key deliverables |
|---|---|---|
| 6.1 | `ui/board_renderer.py` | Three squares; connecting lines; piece circles; hover highlights; mill overlay; click → position string |
| 6.2 | Ghost piece | `render_book_move_ghost(pos)` draws translucent piece for book-line preview |
| 6.3 | Standalone test | `board_renderer.py` as `__main__` with hard-coded `BoardState` |

**Acceptance criteria:**
- All 24 positions clickable and return correct algebraic label.
- Mill positions show coloured overlay.
- Ghost piece renders at the correct position when enabled.

---

### Stage 7 — Full UI Assembly
**Goal:** Complete Tkinter window with embedded Pygame, all panels visible.

| # | Module | Key deliverables |
|---|---|---|
| 7.1 | `ui/opening_panel.py` | Opening name + endgame row (phase label, named pattern, assessment colour); "Show book line" checkbox; "Why this opening?" button |
| 7.2 | `ui/chat_panel.py` | All speaker tags; amber `[MillsLLM ⚠]` for coaching; feedback entry; "Bad Move!" button |
| 7.3 | `ui/settings_panel.py` | All controls including opening book and endgame recognition settings |
| 7.4 | `ui/main_window.py` | Layout: opening/endgame panel above chat; Pygame embed; queue wiring; progress bar |

**Acceptance criteria:**
- Opening panel updates after every placement move.
- Endgame row appears (and opening row hides) once piece count drops to ≤ 11.
- Assessment coloured correctly: green / amber / red from human's perspective.
- "Show book line" checkbox toggles ghost piece via `dialogue_queue`.
- Chat panel renders all four speaker colours correctly.
- Settings persist across restart.

---

### Stage 8 — Integration & Full Game Loop
**Goal:** End-to-end playable game with all systems connected.

| # | Task | Key deliverables |
|---|---|---|
| 8.1 | `main.py` | Entry point; spawn threads; wire queues; launch Tkinter mainloop |
| 8.2 | Human move commentary | `react_to_human_move` live in game; opening-aware and endgame-aware comments appear in chat |
| 8.3 | Deviation questions | LLM asks human about book-line deviations at appropriate frequency |
| 8.4 | Export | Export dialog; annotated `.txt` and `.pgn-style` output |
| 8.5 | Session end | `update_outcome_stats`, `save_novel_opening`, `summarise_session` called on game over |
| 8.6 | Pattern injection | `analyse_patterns` + opening win rates + endgame phase frequency injected into LLM system prompt at session start |

**Acceptance criteria:**
- A complete game produces a valid, annotated export file with `{?}` markers on deviations.
- At least one opening-aware LLM comment appears when the human deviates from a book line.
- At least one endgame-aware LLM comment names the endgame phase in a game that reaches ≤ 11 pieces.
- Session narrative mentions both the recognised opening and the endgame phase by name.
- Novel opening saved to `openings.json` after a game with no recognised opening.

---

### Stage 9 — Post-Game Debrief System
**Goal:** Fully functional debrief and import pipeline, with replay, score graph, and LLM narrative.

| # | Module | Key deliverables |
|---|---|---|
| 9.1 | `game/game_importer.py` | Parse `.txt`/`.pgn-style` files; reconstruct all board states; raise `ValueError` on illegal moves |
| 9.2 | `ai/debrief_engine.py` | `DebriefEngine.analyse()`: board reconstruction, full position evaluation, critical moment detection, turning point identification |
| 9.3 | `MillsLLM` debrief methods | `debrief_game()` and `debrief_position()` with correct prompt templates |
| 9.4 | `ui/replay_panel.py` | Navigation controls; score graph (Matplotlib); board at each ply; `[DebriefAI]` commentary; progress bar during analysis |
| 9.5 | Game-end integration | "Open debrief?" dialog at game end; `auto_open_debrief` setting; "Debrief" and "Import & Analyse" buttons in main window |
| 9.6 | Debrief export | Save to `data/debriefs/` as JSON + Markdown; "Export Debrief" button in replay panel |

**Acceptance criteria:**
- A complete exported game file (`.txt`) can be imported and all board states reconstructed without error.
- An illegal move in an import file produces a clear `ValueError` with the offending ply and notation.
- `DebriefEngine.analyse()` identifies the correct turning point (verified against a hand-annotated game where the losing move is known).
- The score graph renders with ★ markers at every `CriticalMoment`.
- Clicking a ★ on the graph jumps the board to that ply.
- "Jump to turning point" navigates to `turning_point.ply`.
- `debrief_game()` narrative mentions the losing side, the turning point ply, and the better alternative move.
- The replay panel can be opened while a new game is in progress without crashing.
- Debrief thread releases the GameAI lock promptly; live game is not delayed.

---

### Stage 10 — Testing, Hardening & Polish
**Goal:** Stable, shippable v1.

| # | Task | Key deliverables |
|---|---|---|
| 10.1 | Extended tests | Edge cases: fly phase, blocked player, commentary frequency cap, novel opening saving, import of malformed files, endgame phase transitions |
| 10.2 | Opening quality | Play through all book openings; verify recognizer achieves `status="exact"` for canonical lines |
| 10.3 | Endgame quality | Step through all 14 named endgame patterns; verify `named_pattern` matches and `assessment` is correct; verify zugzwang detection fires on known positions |
| 10.4 | Error handling | Ollama offline → graceful fallback; ChromaDB write failure logged but non-fatal; `book_openings.json` or `endgame_patterns.json` missing → clear startup error; import of illegal notation → clear error dialog |
| 10.5 | UI polish | Piece animations; last-move indicator; thinking meter accuracy; opening/endgame panel styling; debrief window resize behaviour; endgame assessment colour transitions |
| 10.6 | Commentary quality | Manually play 3 games that reach each of 7v4, 6v4, 4v3, 3v3; verify LLM names the phase and quotes book strategy |
| 10.7 | Debrief quality | Import 3 known games with annotated blunders; verify `turning_point.ply` matches; verify endgame phase shown correctly per ply in replay |
| 10.8 | Settings persistence | All settings survive restart; opening/endgame sliders take effect immediately |

**Acceptance criteria:**
- Starting with Ollama offline produces a working game with a visible warning.
- Opening recognizer achieves `status="exact"` for all canonical lines in `book_openings.json`.
- All 14 endgame patterns recognised at `confidence=1.0` from their FEN signatures.
- Commentary frequency cap of 5 per game enforced.
- Debrief turning point is correct for all 3 test games.
- No unhandled `Exception` propagates to the user during a normal game or debrief.

---

## Key Design Decisions & Notes

- **Why Tkinter + Pygame?** Tkinter provides native widgets; Pygame provides smooth board rendering. Embed via `SDL_WINDOWID`.
- **Why ChromaDB?** Fully local, uses Ollama embeddings, persists across sessions.
- **Why a separate Opening Knowledge System from ChromaDB?** Opening lines are symbolic, branching, and relational. ChromaDB semantic search cannot reliably distinguish "same move sequence" from "similar position" — exact prefix matching does. The two systems are complementary: ChromaDB handles narrative/pattern memory; `OpeningBook` handles structured move trees.
- **`book_openings.json` is never written at runtime.** This preserves the integrity of curated knowledge. Only `openings.json` is modified. At first run, it is seeded from the book file.
- **Opening recognition stops after the placement phase.** Move-phase positions are too diverse for reliable prefix matching; heuristic evaluation handles those.
- **GameAI is always the final arbiter.** Book bonuses are advisory; a large minimax advantage will still override the book move.
- **The opening deviation question is distinct from the poor-move comment.** A deviation is not necessarily a blunder — the LLM asks why, rather than scolding. The poor-move comment is only triggered by a significant score drop.
- **LLM memory is prompt-based, not weight-based.** Ollama is stateless; `conversation_history` is passed on every call.
- **Difficulty 5 uses iterative deepening** with a hard time cutoff.
- **The `*` end marker** signals resign / block / reduction. Precede with `# Black wins by reduction`.
- **Debrief analysis uses `debrief_analysis_depth`, not `difficulty`.** Debrief re-evaluation
  runs at a fixed depth (default 4) regardless of the game's difficulty setting so analysis
  is consistent across sessions and fast enough not to block the UI. The Debrief Thread
  acquires a shared lock with the AI Thread so the two cannot search simultaneously.
- **The debrief window is non-modal.** It opens as a `Toplevel` so the user can play a
  new game while reviewing a previous one. Each debrief keeps its own LLM
  `conversation_history` so it does not pollute live-game chat context.
- **Game import reconstructs board state from notation, not from FEN.** This means
  imported games are fully validated for legality move by move, and the import fails
  loudly on illegal notation rather than silently producing a corrupt board.

---

## Future Extensions (Out of Scope v1)

- Opening book editor UI (create/edit named openings in-app)
- Opening book export (share curated lines with other users)
- Opening ECO-style classification codes for Nine Men's Morris
- Network play / remote Ollama endpoint
- Voice input for human feedback
- In-app debrief annotation editor (add personal notes to critical moments)
- Tournament mode (multiple AIs)
- Mobile port via Kivy
