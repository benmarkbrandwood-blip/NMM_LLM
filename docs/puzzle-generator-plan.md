# Puzzle Generator Plan for NMM_LLM

Create a new program called **Puzzle Generator** for the NMM_LLM project. The program should use the existing Nine Men's Morris endgame databases in `data/endgame/*.wdl` to generate endgame puzzles that a user can solve through the GUI.

The generator should let the user choose:
- A specific `.wdl` database file from `data/endgame/*.wdl`, or a random one.
- The winning side: `W`, `B`, or `random`.
- The target puzzle depth: `3`, `4`, or random from those two options.

The goal is to generate a position where the chosen side has a **forced win in exactly 3 or 4 moves**, and where the puzzle is interesting rather than trivial.

The existing `EndgameDB` implementation already provides a strong base for this work because it is built around exact board states, side-to-move, and D4 symmetry normalization for canonical indexing [file:68]. It also works with endgame positions only, specifically post-placement states with 11 or fewer total pieces, which is the right scope for this feature [file:68].

## Objectives

The program should:

1. Load a user-selected endgame database from `data/endgame/*.wdl`.
2. Find positions where White or Black has a forced win in exactly 3 or 4 moves.
3. Prefer difficult positions, especially bottleneck positions where only one move wins.
4. Export the puzzle in a format that can be consumed by the NMM_LLM GUI.
5. Validate the user’s attempted solution move by move.
6. Support both single-puzzle generation and batch generation.

## High-level architecture

Implement the project in five layers:

1. **Database adapter**
   - Loads a selected `.wdl` file.
   - Exposes position lookup by board state and side to move.
   - Returns WDL information and, if available, distance-to-win information.
   - Supports probing all legal child positions after each legal move.

2. **Puzzle search engine**
   - Enumerates candidate positions from the selected database.
   - Filters positions by winning side and target win depth.
   - Scores positions for puzzle quality and hardness.
   - Deduplicates positions under symmetry.

3. **Puzzle data model and serialization**
   - Represents puzzles as structured Python data.
   - Exports puzzles as JSON.
   - Supports import of saved puzzle files for replay and GUI loading.

4. **GUI bridge**
   - Converts generated puzzles into the same board/state format used by NMM_LLM.
   - Loads puzzles into the board view.
   - Validates user moves against the stored solution or against live tablebase truth.

5. **Testing and verification**
   - Confirms each generated puzzle is actually correct.
   - Confirms the move uniqueness requirements for hard puzzles.
   - Confirms JSON output can be loaded by the GUI layer.

## Puzzle definition

A valid puzzle should have the following properties:

- The side to move is clearly defined.
- The chosen winning side has a forced win.
- The forced win must be in exactly 3 or 4 moves, depending on user choice.
- The position should be legal and loadable by the existing project board representation.
- The solution line should be stored so the GUI can validate user play.

The displayed objective should read like:
- **White to move and win in 3**
- **Black to move and win in 4**

## Hard puzzle selection

To make the generated problems genuinely interesting, the program should prefer positions that are hard for a human to solve.

The best practical definition of “hard” for a first version is:
- There is exactly **one** move that preserves the forced win.
- Most or all other legal moves reduce the result to a draw or a loss.
- Some wrong moves may even throw the win away immediately.
- The position has enough legal moves that the correct move is not obvious.
- The winning move is ideally quiet or positional rather than an immediate obvious capture.

This bottleneck idea is the most reliable way to measure puzzle quality from tablebase truth rather than from guesswork. The existing endgame code is already state-centric and symmetry-aware, so that same exact-position approach should be reused here [file:68].

## Hardness scoring

Implement a first-pass hardness score such as:

- +4 if exactly one legal move preserves the win.
- +2 if every other move draws or loses.
- +1 if at least one wrong move flips the result to opponent win.
- +1 if the legal move count is 6 or more.
- +1 if the winning move is not an immediate capture or obvious mill-forming move.
- +1 if the opponent also has a narrow best-defense set in the main line.

This score does not need to be perfect. It only needs to rank candidate positions well enough to choose stronger puzzles first.

## Core search algorithm

The search engine should work as follows:

1. Read all candidate positions from the selected database.
2. For each position:
   - Determine side to move.
   - Determine the WDL result for that side.
   - Keep only positions where the requested winning side is winning.
   - Keep only positions with win distance exactly 3 or 4, if that information exists directly.
3. Generate all legal moves from that position.
4. Probe every child position in the database.
5. Determine which moves preserve the forced win.
6. Reject the position unless exactly one move preserves the target result, if generating hard puzzles.
7. Score the surviving positions with the hardness function.
8. Deduplicate equivalent positions under symmetry.
9. Return one random puzzle from the top-ranked set, or save many in batch mode.

## If the database only stores WDL

If the `.wdl` file contains only win/draw/loss information and not exact distance-to-win, the program should derive win-in-3 or win-in-4 using recursive minimax search over legal moves.

Suggested recurrence:

- Win in 1: there exists a move such that the opponent is immediately in a losing state.
- Win in 2: there exists a move such that for every opponent reply, the side to move has a win in 1.
- Win in 3: there exists a move such that for every opponent reply, the side to move has a win in 2.
- Win in 4: same pattern extended one level further.

Because these are endgame tablebase positions, the search depth is small and should be feasible.

## Data model

Define a puzzle dataclass like this:

```python
from dataclasses import dataclass

@dataclass
class Puzzle:
    puzzle_id: str
    source_db: str
    board_fen: str
    canonical_key: str
    side_to_move: str
    winning_side: str
    target_win_in: int
    best_move: str
    solution_line: list[str]
    opponent_best_defense_line: list[str]
    legal_moves: list[str]
    losing_moves: list[str]
    drawing_moves: list[str]
    hardness_score: float
    tags: list[str]
```

Recommended tags:
- `win-in-3`
- `win-in-4`
- `unique-move`
- `bottleneck`
- `endgame`
- `high-branching`
- `quiet-move`

## JSON output format

Export puzzles as JSON so the GUI can load them directly.

Example schema:

```json
{
  "id": "egdb_4v3_00127",
  "source_db": "data/endgame/4v3.wdl",
  "board_fen": "...",
  "side_to_move": "W",
  "winning_side": "W",
  "goal": "White to move and win in 3",
  "target_win_in": 3,
  "best_move": "g4-f4",
  "solution_line": ["g4-f4", "d2-d3", "f4-f6"],
  "hardness_score": 8.0,
  "tags": ["endgame", "unique-move", "bottleneck"]
}
```

Save generated files under:

- `data/puzzles/endgame/` for generated puzzle JSON.
- Optionally `data/puzzles/endgame/cache/` for intermediate search results.

## GUI integration requirements

The program should be designed so it can interface cleanly with the NMM_LLM GUI.

Required GUI behaviors:

- Add a **Puzzle Generator** or **Endgame Puzzle** entry point in the GUI.
- Let the user choose a database, winning side, and target depth.
- Load the selected puzzle position onto the board.
- Show a goal label such as “White to move and win in 3.”
- Allow the user to make moves on the board normally.
- After each move, validate whether the forced win is still preserved.
- If the user finds the correct move, either:
  - auto-play the opponent’s best defense, or
  - show the opponent’s move and continue interactively.
- If the user makes a move that throws away the win, fail the puzzle immediately.
- Support hint and reveal actions.
- Optionally support post-failure analysis mode.

The existing endgame database logic is already keyed around board strings, side-to-move, and move notation fields such as `board_fen_before` and `notation`, so the same primitives should be used as the interface contract between the puzzle generator and the GUI [file:68].

## Deduplication rules

The generator should avoid producing many copies of the same puzzle in rotated or reflected form.

Implement deduplication by:
- Canonicalizing positions under D4 symmetry.
- Using a stable canonical key for puzzle storage.
- Rejecting positions that are equivalent under symmetry to a puzzle already selected.
- Optionally rejecting puzzles with identical first move and equivalent solution tree.

This is especially important because the existing endgame database builder already uses D4 symmetry normalization for exact-state indexing, so the puzzle generator should be consistent with that logic [file:68].

## Failure handling and fallback strategy

If no puzzles are found under the strict requirements, degrade gracefully in this order:

1. Allow positions with more than one winning move.
2. Widen the depth filter from exactly 3 or 4 to a small range such as 2–5.
3. Relax the “hard puzzle” requirement but keep correctness.
4. Report clearly that no suitable puzzle was found for that database and choice.

Do not fabricate exact win depth if the database does not support it directly and the recursive search has not confirmed it.

## Suggested implementation steps

1. Inspect the existing NMM_LLM project for:
   - board serialization format,
   - move notation format,
   - GUI hooks for loading a board state.

2. Build a `.wdl` adapter module:
   - file loader,
   - position probe,
   - child probe,
   - optional distance-to-win support.

3. Build the puzzle search engine:
   - candidate enumeration,
   - win-depth filtering,
   - legal move analysis,
   - hardness scoring,
   - symmetry deduplication.

4. Build puzzle export/import:
   - `Puzzle` dataclass,
   - JSON encoder/decoder,
   - filesystem storage under `data/puzzles/endgame/`.

5. Build GUI integration:
   - load puzzle JSON,
   - render board,
   - validate moves,
   - show hints and reveal line.

6. Add tests:
   - each puzzle is actually winning in the claimed number of moves,
   - “unique move” puzzles really have only one winning first move,
   - symmetric duplicates are not emitted,
   - GUI can load exported puzzles.

## Suggested file layout

A possible layout is:

```text
nmm_llm/
├── tools/
│   ├── puzzle_generator.py
│   ├── puzzle_search.py
│   ├── wdl_adapter.py
│   └── puzzle_models.py
├── data/
│   ├── endgame/
│   │   └── *.wdl
│   └── puzzles/
│       └── endgame/
│           └── *.json
└── gui/
    └── puzzle_mode.py
```

## Acceptance criteria

The first version should be considered complete when it can:

- Let the user choose a `.wdl` endgame database file.
- Let the user choose White, Black, or random as the winning side.
- Generate at least one correct puzzle that is a forced win in 3 or 4 moves.
- Prefer puzzles with a unique winning move.
- Export the puzzle as JSON.
- Load the puzzle into the GUI and validate user moves correctly.
- Avoid simple symmetry duplicates.

## Claude implementation prompt

Use the following implementation prompt:

> Create a new program called `Puzzle Generator` for the NMM_LLM project. It must use a user-selected endgame database from `data/endgame/*.wdl` to generate Nine Men’s Morris endgame puzzles. The user chooses a DB file, the winning side (`W`, `B`, or random), and a target depth of 3 or 4 moves. The generator must find positions where the chosen side has a forced win in exactly 3 or 4 moves. Prefer hard puzzles, especially bottleneck positions where there is exactly one winning move and most other legal moves draw or lose, ideally with some wrong moves handing victory to the opponent.
>
> Reuse the project’s existing board/state and notation formats, and reuse symmetry/canonicalization utilities where possible. The program should deduplicate symmetric equivalents of the same puzzle. It should export puzzles as JSON and be designed to plug into the NMM_LLM GUI, where the user can load a puzzle, make moves, and have the program validate whether the solution is still on the forced-win line.
>
> Implement it in layers: (1) `.wdl` database adapter, (2) puzzle search and hardness scoring, (3) JSON export/import, (4) GUI integration hooks, (5) tests. If the DB does not expose exact win distance, derive win-in-3 or win-in-4 by recursive search over legal moves using minimax and tablebase outcomes.
>
> Include a concrete hardness score, a `Puzzle` dataclass, and a JSON schema. Add batch generation support to save puzzles under `data/puzzles/endgame/`. Add tests that verify each generated puzzle is truly winning in the claimed number of moves and, for hard puzzles, that the first move is unique.

## Notes

The current endgame database builder is a good fit for this because it already treats exact endgame states as canonical database items, uses symmetry normalization, and is focused on compact endgame positions rather than full-game states [file:68]. That should make the puzzle generator reliable, deduplicated, and efficient if it reuses the same state and symmetry concepts [file:68].
