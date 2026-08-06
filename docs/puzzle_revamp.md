# Puzzle Interface Revamp Plan

**Date:** 2026-08-06  
**Status:** Planning — not yet scheduled

---

## Current State

Three puzzle modes (Endgame / Midgame / Placement) share a common board renderer but have near-duplicate loading/metadata functions. The page works but has accumulated several rough edges as features were added piecemeal.

---

## Priority 1 — State Machine (Architectural)

The current state is spread across 7+ boolean/nullable globals:

```
freePlayMode, _opponentThinking, pendingMove, _wrongMoveNotation,
board.capMode, solved, failed, setupMode
```

Replace with a single `puzzlePhase` enum:

```
'idle' | 'user_turn' | 'user_selecting_cap' | 'opponent_thinking'
| 'free_user_turn' | 'free_opp_thinking' | 'solved' | 'failed'
| 'setup' | 'playing_solution'
```

Benefits: impossible state combinations are prevented by construction; debugging is trivial (`console.log(puzzlePhase)`); `onNodeClick` becomes a single switch rather than cascaded guards.

**Effort:** 1–2 days. High value. Do this before adding more features.

---

## Priority 2 — Unified Puzzle Loader

Three nearly-identical `_loadPuzzle` / `_loadMalomPuzzle` / `_loadPlacementPuzzle` functions differ only in endpoint URL, button ID, and one metadata field. Collapse to:

```javascript
async function _loadAnyPuzzle(mode, params) { ... }
```

Callers pass `{ mode, side, depth, db }` and the function routes the URL. Similarly the three metadata `innerHTML` blocks are identical — one `_renderMeta(puzzle)` helper.

Also collapse the three server-side random endpoints — they share the same cache-read / pick / increment / fallback pattern. A single `_puzzle_cache_random(cache_dir, filter_fn)` function would remove ~100 lines.

**Effort:** Half day. Low risk. Worth doing alongside state machine work.

---

## Priority 3 — Error UX

Current error handling dumps every failure into `setStatus()`. Users can't tell "empty cache" from "server down" from "Malom DB not loaded."

**Changes:**
- Add a dedicated `showError(title, body, actions=[])` dialog (reuse wrong-move-overlay structure)
- Distinguish:
  - `503 + "Malom DB not available"` → "Configure Malom DB path in Settings" with link
  - `503 + "No puzzles found"` → "Run the puzzle generator CLI to create puzzles"
  - Network failure → "Server not reachable — is the server running?"
- Wrong-move dialog: differentiate "not on solution path" vs "illegal move" (server returns 400 for illegal)

**Effort:** Half day.

---

## Priority 4 — Visual Feedback During Async

**Missing feedback:**
- Board has no visual "frozen" state during `_opponentThinking` — add a 30% opacity overlay or cursor change
- Loading puzzles: "Loading…" button text is the only feedback — add a spinner to the status area
- After delete: board resets instantly with no animation — add brief "Deleted." flash

**Effort:** 1–2 hours CSS/JS.

---

## Priority 5 — Mode Switching Without Losing Progress

Currently switching modes clears all puzzle state without warning. If user is mid-puzzle, they lose progress.

**Change:** If `!solved && !failed && movesDone.length > 0`, show a one-line confirmation inline: "Switch mode? Current puzzle progress will be lost." with Confirm / Cancel. Not a modal — just a banner under the mode buttons.

**Effort:** 2 hours.

---

## Priority 6 — Capture UX Improvements

- Highlight the mill that was just formed (flash the 3 squares amber) before entering capture mode — currently users may not see why capture was triggered
- Show remaining opponent piece count prominently during capture selection ("3 opponent pieces — capture one to win")
- When opponent has exactly 3 pieces and user forms a mill: label the capture prompt "Capture any piece to win the game"

**Effort:** 2–3 hours.

---

## Priority 7 — Setup / Entry Panel

- When "Enter puzzle" is clicked with a loaded puzzle, seed `setupWinningSide` from `puzzle.winning_side` (currently hardcodes 'W')
- Add notation help tooltip near the solution input field
- Validate that solution line has correct parity (odd number of moves if winning side moves first)
- Clear `setupGrid` properly when switching puzzle modes while setup panel is open

**Effort:** 2 hours.

---

## Priority 8 — Play Count & Puzzle Rotation

*Already implemented (2026-08-06):*
- `play_count` field incremented on serve; preferred lowest-count puzzle when picking random
- Play count shown in metadata panel

**Not yet implemented:**
- Filter option: "New only" checkbox to restrict to `play_count === 0`
- Reset play counts button (in Tools page or puzzle admin)
- Show "Played X times" next to puzzle number when X > 1

---

## Priority 9 — Placement Phase Specific

- `setupUserTurn(null)` in placement loading doesn't pass `legal_dests` — empty squares should be highlighted on load, same as they are for endgame. Fix: compute empty squares and pass them.
- Placement FEN detection (`isPlacementPhase`) relies on magic index 2/3 of FEN split — add a named parser function so FEN format changes are caught in one place.

**Effort:** 1 hour.

---

## Out of Scope (for this revamp)

- Keyboard navigation (accessibility) — separate accessibility pass
- Move undo — complex state management, defer
- Tutorial/onboarding — separate UX project
- Real-time multiplayer puzzle races — future feature

---

## Suggested Order

1. State machine refactor (unblocks everything else cleanly)
2. Unified loader (reduces surface for errors)
3. Error UX (immediate quality improvement)
4. Visual feedback + mode-switch warning (polish)
5. Capture UX + setup fixes (correctness)
6. Remaining play-count features

Do not start new features until Priority 1–3 are done.
