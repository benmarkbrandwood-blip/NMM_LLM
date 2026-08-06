# Puzzle Makeover Plan

**Created:** 2026-08-05  
**Scope:** Diagnosis, redesign notes, and implementation plan for the puzzle generation and inference pipeline. No code yet.

---

## 1. Which Script Is Fastest? (Malom vs endgame DB)

There are three puzzle generators:

| Script | DB used | Method |
|---|---|---|
| `tools/puzzle_generator.py` | `data/endgame/*.wdl` (local retrograde) | True minimax + WDL pruning |
| `tools/malom_puzzle_generator.py` | Malom perfect DB | Greedy extraction |
| `tools/placement_puzzle_generator.py` | Malom perfect DB | Greedy extraction |

**Only the midgame and placement generators use the Malom DB.** The endgame generator uses the project's own local `.wdl` retrograde tables. Those tables require loading multiple files, and the minimax search is inherently more expensive per candidate.

Between midgame and placement, **placement is slower per puzzle** because it rejects all positions with mills (narrow sampling range) and has strict piece-count constraints (`nX_unplaced ∈ [2,6]`). Midgame has a wider random draw over `(nW, nB) ∈ [4,7]²`.

**Therefore: the midgame Malom generator (`malom_puzzle_generator.py`) is the fastest.**

> **Action item**: Add a bench step before coding — run each generator for 60s with `--count 9999` and compare puzzles-per-second. Record the result in this file to confirm the assumption.

**The largest speed lever is multiprocessing, not per-puzzle micro-optimisation.** Puzzle attempts are entirely independent. A pool of N workers gives near-linear throughput. Malom hash state is per-process, so each worker pays the ~20s prewarm once and then amortises it over hundreds of puzzles. This dominates any other optimisation and must be in the unified generator from the start.

---

## 2. Core Diagnosis: The Greedy vs Minimax Gap

### 2a. Endgame generator — correct

`ai/puzzle_search.py` uses a true minimax:
- `find_win_depth` computes the *minimum* number of winning-side moves to force the win.
- On the opponent's turn it iterates **all** replies; if any reply escapes, the path is rejected.
- The `_pv` (principal variation) function picks the **hardest opponent defense** (maximises depth `best_d`).
- It verifies `depth == target` *and* that `target - 1` fails — no shallower win exists.
- **This is rigorous and correct.**

### 2b. Malom midgame and placement generators — greedy, not verified

`ai/malom_puzzle_search.py`'s `extract_solution_line` is **greedy, not minimax**:

1. **Winning side picks lowest opponent `dtw`** — this is not proven to be the minimum-depth path. `actual_win_in` is the greedy path length, which may exceed the true minimum. A puzzle labelled "win in 6" might actually be "win in 4" if a different first move is chosen.

2. **Opponent picks highest `dtw`** — the Malom docs themselves say that midgame `dtw` is a "relative quality hint," not exact distance-to-mate. So "longest path to loss for the defender" is **not actually guaranteed**. The current system might pick a suboptimal defender response and then show a longer solution line than the opponent would actually choose well.

3. **`max_depth=9` is hardcoded** in `extract_solution_line` — this prevents win-in-10 puzzles from ever being emitted.

4. **`score_hardness` in midgame** classifies first-level winning moves by Malom WDL == "L" at *any* depth — but "losing" for the opponent doesn't mean losing within the same depth budget as the stated `target_win_in`. So `max_winning_moves` filtering is semantically different from the endgame version, which uses `_search(child, target-1, memo)` to verify within budget.

**This is the central problem:** the defending side is not necessarily choosing optimal moves, and the solution line is not proven to be the minimum forced-win path.

---

## 3. Full Problem List

### P1 — Greedy extraction is not a proof of forced win depth (midgame + placement)
- `extract_solution_line` produces a *sample path*, not the shortest/verified forced win.
- Fix: port the endgame minimax structure onto Malom WDL queries for midgame/placement.
- For the discovery phase (sampling thousands of candidates) the greedy pass is fine as a pre-filter; a minimax verification pass then confirms depth exactly.

### P2 — "Defender must make the best moves" not guaranteed (midgame + placement)
- Malom dtw is a hint, not exact DTM. Opponent "hardest defense" by dtw is an approximation.
- Fix: during minimax verification, use true worst-case opponent: iterate **all** opponent replies, reject any position where any reply escapes. Among forced-losing replies, choose the one that maximises the verified minimax depth (not dtw).
- **This is what the user described as "longest path to loss for the losing side."**

### P3 — `max_winning_moves` default is 2, semantics differ across systems
- All three CLI scripts default to `--max-winning-moves 2`. The user wants exactly 1 forced pathway.
- Change the unified generator's default to `1`.
- Endgame: counts moves winning within budget. Malom: counts moves where child WDL=="L" at any depth. These are not equivalent. The unified generator must align them by verifying within the stated depth budget.

### P4 — Depth hard-capped at 7 in all three layers
- CLI scripts: `choices=[0, 4, 5, 6, 7]`.
- API validators (`web/app.py` placement route `:1830`): `if depth not in (0, 4, 5, 6, 7)`.
- UI dropdowns (`puzzles.html` endgame: 3–7, midgame/placement: 4–7).
- `extract_solution_line`: `max_depth=9` internal cap.
- Fix: extend all four layers to allow 8, 9, 10.

### P5 — dtw pre-filter windows exclude deep wins and waste time on shallow wins
- Midgame: `_DTW_PREFILTER_MIN=5`, `_DTW_PREFILTER_MAX=17` — likely excludes win-in-8+ candidates.
- Placement: `_PLACEMENT_DTW_MIN=3`, `_PLACEMENT_DTW_MAX=25`.
- Using one wide window for all target depths means both false positives (candidates that fail minimax and waste verification time) and false negatives (deep candidates excluded by too-low MAX).
- Fix: **adaptive per-depth dtw windows**. During the bench step (§1), sample ~1000 positions at each target depth and record the dtw distribution of those that pass minimax verification. Set `[MIN, MAX]` per depth from those empirical ranges. This both tightens the pre-filter and correctly covers depth 8–10.

### P6 — `prewarm_hash_cache` caps at 7, blocks large midgame pieces
- Positions with 8–9 pieces per side are valid but incur a ~1.7s hash init per new `(W, B)` pair.
- If we want to allow 8–9 piece positions (richer midgame), prewarm must be extended.
- Note: wider prewarm adds ~5–15s startup cost but pays off across 3000 samples.

### P7 — Endgame `load_puzzle_db` only loads tables within `max_depth` capture distance
- `load_puzzle_db(max_depth=depth)` loads tables within `depth` captures.
- For win-in-10 with up to 10 captures, this needs to cover a wider range.
- Check memory impact of loading larger table sets.

### P8 — `best-response` endpoint routes to Malom greedy, not the heuristic engine
- `/api/puzzles/malom/best-response` picks the move with highest dtw for the active side.
- **The intended free-play behaviour is:** when the user diverges from the solution line, they continue playing against the project's heuristic engine (negamax in `ai/coordinator.py`), not against Malom dtw-ranking.
- Fix: remove the Malom dtw call in the best-response endpoint entirely. Route to the same engine path used during normal gameplay, at a configurable difficulty depth. This makes the "Continue vs AI" mode consistent with the rest of the game, and removes the Malom approximation from free-play.
- The Malom DB continues to be used for **puzzle generation and validation only** — never as the inference engine for the opponent during free-play.

### P9 — Divergence / free-play is against the heuristic engine (not the puzzle DB)
- When the user plays a move that wins but is not on the stored solution line, the UI shows "Wrong move" dialog with option to "Continue vs AI."
- **Design intent:** after diverging, the user plays against the project's normal heuristic AI at a fixed difficulty. The puzzle generator finds the forced-win position; if the user finds a different winning path the AI will play its best, and if the user cannot convert it, the AI may escape.
- This is already architecturally correct (the free-play path calls the engine via the same game session), but currently breaks by calling `/api/puzzles/malom/best-response` (dtw greedy). Fix is in P8.
- The UI copy ("you may still find a win") is accurate for this design. No UI change needed.

---

## 4. Plan: Unified Generator Script

Create `tools/unified_puzzle_generator.py` that handles all three puzzle types (endgame, midgame, placement) via a `--type` argument. This replaces the need to maintain three separate CLIs.

### 4a. Architecture

```
tools/unified_puzzle_generator.py
  --type endgame|midgame|placement
  --depth 3–10 (or 0 = random within type-appropriate range)
  --side W|B|random
  --max-winning-moves N (default: 1)
  --count N (0 = forever)
  --attempts N
  --out DIR

ai/puzzle_search.py           (endgame, keep as-is — already correct)
ai/malom_puzzle_search.py     (midgame/placement — fix greedy → minimax-verified)
```

The unified CLI reads settings.json for malom_db_path, pre-warms the appropriate hash cache for the type and depth, delegates to the appropriate backend, and writes to:
- `data/puzzles/endgame/` for `--type endgame`
- `data/puzzles/malom/` for `--type midgame`
- `data/puzzles/placement/` for `--type placement`

### 4b. Why keep three separate backends

The endgame backend is built on a different DB format (`.wdl` retrograde tables, local minimax). The midgame/placement backends are Malom-based with different position-sampling strategies. Unifying the CLI is sufficient; the backends stay as separate modules.

---

## 5. Fix Plan: Minimax Verification for Malom Puzzles

### Strategy: Greedy Pre-filter → Minimax Verify

Full minimax over midgame branching (~15–25) to depth 10 is expensive in Python, even with Malom pruning. Pure minimax without memoisation over 3000 candidates would be prohibitive.

Chosen approach:

1. **Greedy pre-filter** (keep current `extract_solution_line` logic): fast, samples thousands of positions, identifies approximate candidates. Widen dtw window as needed for depth 8–10.

2. **Minimax verification pass** (new function, modelled on `ai/puzzle_search.py`): run only on candidates that passed all pre-filters. Uses Malom WDL as oracle:
   - On winning side's turn: iterate all moves; pick the one(s) that keep WDL=="L" for opponent and return minimum minimax depth.
   - On opponent's turn: iterate **all** replies; if any reply leads to WDL!="W" for the winning side → reject the position entirely (not a forced win). Among forced-losing replies, recursively find the one that maximises the minimax depth.
   - Memoisation keyed on (fen, budget) as in the endgame version.
   - `find_malom_win_depth(db, board, ws, max_depth)` → returns exact min-depth or None.

3. **Depth-exact check**: after verification, confirm `find_malom_win_depth(budget=target) == target` and `find_malom_win_depth(budget=target-1) is None`. This ensures the puzzle is exactly win-in-N, not win-in-less.

4. **Solution line extraction**: rebuild `_pv` equivalent for Malom — winner picks move with minimum verified depth, opponent picks move with maximum verified depth (true worst-case defense). No dtw used here.

5. **`max_winning_moves` fix**: count only moves where `find_malom_win_depth(budget=target-1)` returns non-None. Consistent with endgame semantics.

### Feasibility of depth-10

- Midgame branching ~15–25. Budget=10 winning-side moves = up to 20 plies.
- With aggressive Malom WDL pruning (discard any opponent escape immediately; only recurse on "L" moves for winner) the effective branching is dramatically reduced.
- Memoisation on (fen, budget) covers transpositions.
- Expect 0.5–5s per verification call for deep positions. This is fine for a generator that runs offline.
- Prewarm hash cache to max_pieces=9 for depth-10 midgame to avoid per-query init cost.

---

## 6. Batch Mode and Parallelism

### 6a. Batch matrix job

The intended workflow is: generate N puzzles for each cell of a configuration matrix:

```
(type, side, depth) × K puzzles
```

Example matrix for a full batch run:
- `type`: endgame, midgame, placement
- `side`: W, B
- `depth`: 3–7 endgame; 4–10 midgame and placement
- `K`: e.g. 50 puzzles per cell

A batch config file (`data/puzzles/batch_config.json`) specifies the matrix and K. The unified generator reads it when run in `--batch` mode, iterates over cells, skips cells already meeting quota (count existing cache files for that type/side/depth), and resumes from checkpoint if interrupted.

### 6b. Parallelism: multiprocessing pool

The unified generator spawns a process pool of N workers (default: `cpu_count - 1`). Each worker:
1. Pre-warms the Malom hash cache once for its target type (hash state is per-process).
2. Runs an inner loop calling the appropriate `generate_*` function.
3. Writes puzzle JSON atomically (temp file + rename) to avoid concurrent write corruption.
4. Reports back puzzle ID + stats to the parent for progress logging.

Workers are assigned a `(type, side, depth)` cell from the matrix; the parent refills the queue as cells reach quota.

This is the primary throughput lever — 8 workers on the midgame generator should produce ~8× the puzzles/sec of a single process, dominating any per-puzzle optimisation.

### 6c. Balanced output

At serve time, `api_malom_puzzle_random` and `api_puzzle_random` already filter by `(side, depth)` before random selection. With a balanced cache (equal K per cell) the user sees roughly equal puzzle difficulty distribution. No change to serve-time logic needed.

### 6d. Hardness floor

Add `--min-hardness N` to the unified generator. Reject any puzzle with `hardness_score < N` before writing. This prevents easy puzzles (e.g. win-in-3 with obvious capture) from diluting the cache. Default: 3.0.

---

## 7. Symmetry Deduplication

D4 symmetry (4 rotations × 2 reflections = 8 transforms) means the same position can appear up to 8 times in different orientations. With batch generation producing hundreds of puzzles, rotational duplicates will accumulate.

### Implementation

1. Compute a **canonical FEN** under D4: apply all 8 board transforms, take the lexicographically smallest FEN string as the canonical key.
2. Store the canonical key in the puzzle JSON (`canonical_fen` field).
3. At generation time: before writing a new puzzle, check if any existing cached puzzle shares the same `canonical_fen`. If so, skip.
4. At serve time: no change needed — dedup happens at write time.

### Existing cache

The existing 500 malom + 114 placement + 45 endgame puzzles were not generated with canonical dedup. A one-off cleanup pass can:
1. Compute `canonical_fen` for all cached puzzles.
2. For each duplicate group, keep the highest `hardness_score` member, delete the rest.

This is optional but recommended before a large batch run.

---

## 8. Existing Cache Re-verification

The 500 midgame and 114 placement puzzles in cache were generated with the greedy extractor. Their `target_win_in` values and `solution_line` fields are **not guaranteed** to reflect:
- The minimum forced-win depth (may be labelled deeper than the actual minimum).
- Optimal opponent defense (the solution line may show moves the opponent would not choose well).

### Decision: run a re-verification pass

A one-off script (`tools/verify_puzzle_cache.py`) should:
1. Load each cached puzzle JSON.
2. Run `find_malom_win_depth(board, ws, target)` and `find_malom_win_depth(board, ws, target-1)`.
3. If actual minimum depth differs from stored `target_win_in`: update the field (or discard if outside the 4–10 range).
4. Rebuild `solution_line` using the new `get_malom_solution_line` (correct worst-case defender).
5. Recheck `max_winning_moves=1` — discard if more than one move wins within the (corrected) budget.
6. Write a corrected JSON back (or tag with `"greedy-verified": true` for any that cannot be re-verified due to Malom sector coverage gaps).

Endgame puzzles do not need re-verification — they were generated with the correct minimax from the start.

---

## 9. Inference Wiring Changes Required

All changes span four layers: search engine, CLI, server API, and UI.

### 9a. `ai/malom_puzzle_search.py`

- Add `find_malom_win_depth(db, board, ws, max_depth, memo)` — true minimax with Malom WDL oracle, modelled on `ai/puzzle_search.py:_search_inner`.
- Add `get_malom_solution_line(db, board, ws, depth, memo)` — PV extraction with worst-case defender, modelled on `ai/puzzle_search.py:_pv`.
- Update `generate_malom_puzzle` and `generate_malom_placement_puzzle`: replace greedy extraction + hardcoded `max_depth=9` with (a) greedy pre-filter, then (b) minimax verification.
- Use adaptive per-depth dtw windows (set from bench step) rather than single global constants.
- Widen `prewarm_hash_cache` call to `max_pieces=9` in generators that allow depth-10 or large piece counts.
- Fix `score_hardness`: count a move as "winning within budget" only if `find_malom_win_depth(budget=target-1)` is not None.

### 9b. `tools/malom_puzzle_generator.py` and `tools/placement_puzzle_generator.py`

- Extend `--depth` choices from `[0, 4, 5, 6, 7]` to `[0, 4, 5, 6, 7, 8, 9, 10]`.
- Change `--max-winning-moves` default from `2` to `1`.
- These can remain as-is for now if the unified generator is built instead; mark as deprecated once unified is ready.

### 9c. `tools/unified_puzzle_generator.py` (new)

- `--type endgame|midgame|placement`, `--depth 3–10`, `--side W|B|random`.
- `--max-winning-moves` default: 1. `--min-hardness` default: 3.0.
- `--workers N` for multiprocessing pool (default: `cpu_count - 1`).
- `--batch data/puzzles/batch_config.json` for matrix mode.
- Route to correct backend based on `--type`.
- Pre-warm hash cache to `max_pieces=9` for Malom types.
- Atomic JSON write (temp file + rename) for safe parallel output.

### 9d. `web/app.py` server routes

- `api_malom_puzzle_random`: extend depth validator to include 8, 9, 10.
- `api_placement_puzzle_random`: same.
- `api_puzzle_random`: extend depth choices for endgame to include 8, 9, 10; update `load_puzzle_db` call to `max_depth=10`.
- `/api/puzzles/malom/best-response`: **remove Malom dtw routing; route to heuristic engine instead** (see P8). The endpoint signature stays the same; the implementation changes to invoke the negamax search at a configured difficulty.
- Consider adding `/api/puzzles/generate?type=X&depth=Y&side=Z` as a unified generation endpoint.

### 9e. `web/templates/puzzles.html` UI

- **Endgame depth dropdown**: add options `Win in 8`, `Win in 9`, `Win in 10`.
- **Midgame depth dropdown**: add options `Win in 8`, `Win in 9`, `Win in 10`.
- **Placement depth dropdown**: add options `Win in 8`, `Win in 9`, `Win in 10`.
- Update the subtitle copy for midgame to mention "4–10 moves" once depth-10 is supported.
- The wrong-move dialog copy ("you may still find a win") is already correct. No change needed.
- Consider adding a "# of winning moves" indicator in the info panel to help users understand puzzle difficulty tier (1 = only move wins, 2 = two solutions, etc.).

---

## 10. Depth-10 Specific Notes

- **Endgame**: `load_puzzle_db` must be called with `max_depth=10`. This loads tables for up to 10 captures in each direction. Check combined file size — if total exceeds ~2GB it may need a tiered loading strategy.
- **Malom midgame**: win-in-10 in midgame typically means 8–9 pieces per side. `prewarm_hash_cache(9)` required. Expected prewarm cost: ~12–20s total (one-time at CLI start).
- **Malom placement**: placement with depth-10 means many pieces still unplaced; verify `nX_unplaced` range allows it without hitting the `nB_on < 3` guard.
- **Solution line display**: the existing `formatSolutionLine` in the UI handles arbitrary line length — no change needed there.
- **Goal label**: `"{side} to move and win in {N}"` is already generic — no change needed.
- **`target_win_in` field**: stored as int, passed to validators as query param — naturally extends to 10.

---

## 11. Endgame PV Tie-break Note

In `ai/puzzle_search.py:_pv`, when the opponent has multiple moves with equal verified depth, the first move encountered wins (no tiebreak). This is acceptable for now — ties are rare and the solution line is pedagogically valid either way. No change needed unless a specific case surfaces where a capture is chosen over a quiet move with the same depth (which would reduce the puzzle's instructional value).

---

## 12. Testing Requirements

Once code is written (separate task):

1. For each generated puzzle: `find_malom_win_depth(target) == target` and `find_malom_win_depth(target-1) is None`.
2. For unique-move puzzles (`max_winning_moves=1`): exactly one move survives the within-budget win check.
3. For opponent defense: all opponent replies in the verified minimax lose (no escape exists).
4. Regression: existing 500 cached malom + 114 placement + 45 endgame puzzles still load and validate correctly.
5. Depth-10 endgame: at least one puzzle generated and correctly validated by `api_puzzle_validate`.
6. Depth-10 midgame: at least one puzzle generated and correctly validated by `api_malom_puzzle_validate`.

---

## 13. Implementation Order

1. **Bench step**: time all three generators (60s each), record puzzles/sec. At each depth 4–7, sample the dtw distribution of verified positions to set adaptive pre-filter windows. (~1h)
2. **Fix `ai/malom_puzzle_search.py`**: add `find_malom_win_depth` and `get_malom_solution_line` (minimax-verified, worst-case defender). Update both `generate_*` functions to use greedy pre-filter → minimax verify. Fix `score_hardness` semantics.
3. **Adaptive dtw windows**: plug in per-depth `[MIN, MAX]` constants from bench step.
4. **Widen prewarm**: `max_pieces=9`, extend for depth-10.
5. **Write `tools/unified_puzzle_generator.py`**: multiprocessing pool, batch matrix mode, atomic writes, hardness floor, depth 3–10.
6. **Existing cache re-verification**: write and run `tools/verify_puzzle_cache.py`. Discard or correct all greedy-generated midgame and placement puzzles.
7. **Symmetry dedup**: implement D4 canonical FEN, add to write path and to verify script.
8. **Update `web/app.py`**: extend depth validators (all three routes), fix `/api/puzzles/malom/best-response` to route to heuristic engine.
9. **Update `web/templates/puzzles.html`**: extend dropdowns to include 8, 9, 10.
10. **Run tests** (see §12).
