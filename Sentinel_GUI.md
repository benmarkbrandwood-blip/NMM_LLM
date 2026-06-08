# Sentinel GUI — Full Implementation Spec

**Branch:** `feat/sentinel-overlay`  
**Clone from:** `main` (then merge `feat/sentinel-overlay` on top)  
**Do NOT merge to main.**  
**Do NOT run the full test suite.** Syntax checks only.

---

## Overview of what to build

Two independent but visually integrated features:

### Feature 1: Trajectory Overlay in the Scores bar
A new button in the existing Scores bar called **"DB Traj"** and **"Sentinel"** that overlays green/red circles (placement phase) or green/red arrows (move/fly phase) on the board showing:
- **DB Lines** (already exists — `diag-btn-db`): perfect database WDL per move (green=win, red=loss)
- **Sentinel** (NEW): the trained SentinelNet's move quality scores (green=high quality, red=low quality)

The user can select **one at a time** (DB Lines or Sentinel, not both simultaneously — toggling one turns off the other). The existing `diagDB` boolean and `renderDiagDB()` function in `board.js` are the integration point.

### Feature 2: Difficulty-gated visibility and GameAI advisor
- **Human overlay visibility**: tapers off as difficulty increases (levels 1–8 see overlay, level 9+ never see DB/Sentinel overlay)
- **GameAI sentinel advisor activation**: from difficulty 3 upward, the sentinel (or DB if no checkpoint) advises the GameAI, with probability scaling from 10% at level 3 to 100% at level 10

---

## Existing code to understand before writing anything

Read these files thoroughly:

- `web/static/board.js` — `renderDiagDB(moves, opts)` draws the DB overlay. `_dbGroup` is the SVG group. Existing color logic: green `#4caf50` = win/positive, red `#e05050` = loss/negative. The `dbColor(delta, egFlag)` helper maps `eg_flag` ("W"/"L"/"D") and `db_delta` (float) to colors.
- `web/static/game.js` — `diagDB` (bool), `diagTraj` (bool), `diagEnabled` (bool), `_diagRender()`, `_diagOnReceive(msg)`. The diagnostic data flow: user clicks a chip → sets bool → calls `_diagRequestAll()` → server responds with `{type:"diagnostic", moves:[...]}` → `_diagOnReceive()` → `_diagRender()` → `board.renderDiagDB()`. Also `diagStatic`, `diagNegamax` booleans for the Scores/Negamax chips.
- `web/templates/index.html` — the Scores bar at line ~142. Existing chips: `diag-btn-static`, `diag-btn-negamax`, `diag-btn-traj`, `diag-btn-db`. The sentinel checkbox at line ~310: `chk-sentinel`, `sel-sentinel-mode`. The `sentinel-advisory` badge div at line ~64.
- `web/app.py`:
  - `_sentinel_advisor` global (line ~171): loaded at startup from `learned_ai/sentinel/checkpoints/best.pt`, graceful if missing.
  - `_sentinel_payload(adv)` (line ~1006): serialises `SentinelAdvice` into the `ai_move` message's `"sentinel"` key.
  - `get_diagnostic` handler (line ~1964): builds `moves_out` list with per-move `db_delta`, `eg_flag`, `traj_freq`. This is where sentinel scores must also be merged in.
  - `new_game` handler (line ~1608): reads `use_sentinel` and `sentinel_mode` from the message, calls `game_ai.set_sentinel()`.
  - `sentinel_status` endpoint (line ~624): returns `{"available": bool, "checkpoint": str}`.
- `ai/game_ai.py`:
  - `set_sentinel(sentinel, mode)` — attaches the advisor.
  - `_apply_sentinel_intervention(board, move, moves)` — called after heuristic search, calls `self._sentinel_advise()`, stores `self.last_sentinel_advice`.
  - `_sentinel_score_adjust()` and `_sentinel_reconsider()` — intervention modes.
  - `difficulty` attribute on `GameAI` instance.
- `learned_ai/sentinel/infer.py` — `SentinelAdvisor.advise(board_state, candidates, player, played_move_idx)` — the inference call. Returns `SentinelAdvice` with `move_scores: list[float]` (one per candidate, same order), `opportunity_gap`, `advisory_message`, `player`.

---

## Board position index

The 24 board positions and their names (needed for overlay):
```
a1 a4 a7   b2 b4 b6   c3 c4 c5
d1 d2 d3   d5 d6 d7
e3 e4 e5   f2 f4 f6   g1 g4 g7
```
These are the keys used in `mv.from` and `mv.to` in every move dict.

---

## Part 1: Sentinel scores in the diagnostic overlay

### 1a. web/app.py — enrich `get_diagnostic` with sentinel scores

**Location:** The `get_diagnostic` handler, just after the existing Malom/endgame probe block (~line 2110), where `db_deltas`, `eg_flags`, and `traj_freqs` are merged into `moves_out`.

Add sentinel scoring after the existing DB merge loop:

```python
# ── Sentinel overlay: score each legal move with SentinelNet ─────────────
sentinel_scores: dict = {}   # notation → float [0,1]
if _sentinel_advisor is not None and _sentinel_advisor.is_loaded():
    try:
        # Build the candidate list in the same order as moves_out
        # Each entry has {"from": str|None, "to": str, "capture": str|None}
        candidates = [
            {"from": mv_e.get("from"), "to": mv_e.get("to"), "capture": mv_e.get("capture")}
            for mv_e in moves_out
        ]
        if candidates:
            player = color   # "W" or "B" — diag_board.turn
            sent_advice = await asyncio.to_thread(
                _sentinel_advisor.advise,
                diag_board, candidates, player, 0
            )
            if sent_advice is not None:
                for i, mv_e in enumerate(moves_out):
                    if i < len(sent_advice.move_scores):
                        sentinel_scores[_diag_ntn(mv_e)] = round(sent_advice.move_scores[i], 3)
    except Exception as _se:
        log.debug("Sentinel diagnostic scoring failed: %s", _se)

# Merge sentinel scores into moves_out
for mv_e in moves_out:
    ntn = _diag_ntn(mv_e)
    mv_e["sentinel_score"] = sentinel_scores.get(ntn)  # float [0,1] or None
```

Now `moves_out` entries have a `sentinel_score` field (float or None).

### 1b. web/static/board.js — extend `renderDiagDB` to handle sentinel scores

The `renderDiagDB(moves, opts)` function currently handles `db_delta` and `eg_flag` for color, and `traj_freq` for frequency labels. Add a `showSentinel` option:

**Add to the function signature / opts:**
```js
const showSentinel = opts.showSentinel || false;
```

**Extend `dbColor` helper** to also accept a sentinel score:
```js
const dbColor = (delta, egFlag, sentinelScore) => {
  // Sentinel score takes priority if showSentinel
  if (showSentinel && sentinelScore != null) {
    if (sentinelScore >= 0.65) return "#4caf50";  // green: good move
    if (sentinelScore <= 0.35) return "#e05050";  // red: bad move
    return null;   // neutral — don't show
  }
  // existing DB logic
  if (egFlag === "W") return "#4caf50";
  if (egFlag === "L") return "#e05050";
  if (egFlag === "D") return "#888";
  if (delta == null)  return null;
  if (delta > 0.1)    return "#4caf50";
  if (delta < -0.1)   return "#e05050";
  return null;
};
```

**Update every call to `dbColor` in the function** to pass `mv.sentinel_score`:
```js
const col = dbColor(
  showDB ? mv.db_delta : null,
  showDB ? mv.eg_flag  : null,
  showSentinel ? mv.sentinel_score : null
);
```

### 1c. web/static/game.js — add Sentinel chip and wire it up

**State variable:** Add alongside `diagDB`:
```js
let diagSentinel = false;   // show Sentinel AI move quality overlay
```

**In `_diagRender()`**, pass `showSentinel` to `renderDiagDB`:
```js
if ((diagTraj || diagDB || diagSentinel) && dbSource && dbSource.moves) {
  board.renderDiagDB(dbSource.moves, {
    phase:        curPhase,
    selectedSrc:  board.selected,
    showTraj:     diagTraj,
    showDB:       diagDB,
    showSentinel: diagSentinel,
  });
} else {
  board._dbGroup.innerHTML = "";
}
```

**Update the mode label** to include "Sentinel" when active:
```js
if (diagSentinel) modeLabel.push("Sentinel");
```

**Mutual exclusion logic** — DB Lines and Sentinel are mutually exclusive (user picks one):
```js
// When Sentinel turns on, DB turns off; when DB turns on, Sentinel turns off
```
Handle this in the chip click handlers (see Part 2 below).

---

## Part 2: HTML — new Sentinel chip in the Scores bar

### 2a. web/templates/index.html — add Sentinel chip

**Find** the existing DB section in `diag-controls` (~line 152):
```html
<span class="diag-sep">|</span>
<span class="diag-section-label">DB:</span>
<button id="diag-btn-traj"  class="diag-chip" ...>Traj</button>
<button id="diag-btn-db"    class="diag-chip" ...>DB Lines</button>
```

**Add after `diag-btn-db`:**
```html
<button id="diag-btn-sentinel" class="diag-chip" title="Sentinel AI: move quality overlay (green=good, red=bad) — mutually exclusive with DB Lines">Sentinel</button>
```

**Also add** a small status indicator next to it (shown only when sentinel unavailable):
```html
<span id="diag-sentinel-status" style="font-size:.75rem;color:#888;display:none" title="Sentinel checkpoint not loaded">(no model)</span>
```

---

## Part 3: JS — chip click handlers and mutual exclusion

### 3a. web/static/game.js — add Sentinel chip handler

**Find** the existing chip handlers (look for `diag-btn-db` click handler). Add alongside them:

Note: `sentinel_score` is part of the same `get_diagnostic` response as `db_delta`/`eg_flag`. Call `_diagRequestStatic()` when turning sentinel ON to ensure fresh data; otherwise just re-render from cache.

```js
// Sentinel chip — mutually exclusive with DB Lines
$("diag-btn-sentinel") && $("diag-btn-sentinel").addEventListener("click", () => {
  diagSentinel = !diagSentinel;
  if (diagSentinel) {
    diagDB = false;
    $("diag-btn-db") && $("diag-btn-db").classList.remove("diag-chip-active");
    _diagRequestStatic();  // ensure fresh sentinel scores from server
  }
  $("diag-btn-sentinel").classList.toggle("diag-chip-active", diagSentinel);
  _diagRender();
});
```

**Extend the DB Lines chip handler** to also turn off Sentinel:
```js
// Find the existing diag-btn-db click handler and add:
if (diagDB) {
  diagSentinel = false;
  $("diag-btn-sentinel") && $("diag-btn-sentinel").classList.remove("diag-chip-active");
}
```

### 3b. Sentinel availability check on page load

On connection/startup, query the sentinel status endpoint and update the chip:
```js
// After WebSocket connects (or on DOMContentLoaded):
fetch("/api/sentinel_status")
  .then(r => r.json())
  .then(s => {
    const chip   = $("diag-btn-sentinel");
    const status = $("diag-sentinel-status");
    if (!s.available) {
      chip  && (chip.disabled = true, chip.title = "Sentinel model not loaded");
      status && (status.style.display = "inline");
    }
  })
  .catch(() => {});
```

---

## Part 4: Difficulty-gated overlay visibility

### 4a. Rule

| Difficulty | Human sees DB/Sentinel overlay | GameAI advised by Sentinel |
|---|---|---|
| 1 | Full (100%) | Never |
| 2 | Full (100%) | Never |
| 3 | Full (100%) | 10% of moves |
| 4 | Full (100%) | 22% of moves |
| 5 | Full (100%) | 33% of moves |
| 6 | Partial (75%) | 50% of moves |
| 7 | Partial (50%) | 65% of moves |
| 8 | Partial (25%) | 80% of moves |
| 9 | None (0%) | 90% of moves |
| 10 | None (0%) | 100% of moves |

**Human overlay visibility** = fraction of move entries shown (randomly sampled), not a global toggle. At 75%: show 75% of the move dots/arrows (randomly pick which to hide). At 0%: `board._dbGroup.innerHTML = ""` always.

**Sentinel activation probability** = probability that a given call to `_apply_sentinel_intervention` actually runs. Below the threshold, return the original move immediately.

### 4b. web/static/game.js — track current difficulty and gate overlay

Add a state variable:
```js
let currentDifficulty = 3;   // updated from server state messages
```

Update it in the `"state"` message handler (where `gameState` is set from `msg`):
```js
case "state":
  // ... existing state handling ...
  if (msg.difficulty != null) currentDifficulty = msg.difficulty;
  break;
```

Also update from `new_game_ack` or wherever difficulty is confirmed after new_game.

**Gate `_diagRender()` to apply visibility fraction:**

```js
// Overlay visibility by difficulty
function _overlayVisibilityFraction(diff) {
  if (diff <= 5) return 1.0;
  if (diff === 6) return 0.75;
  if (diff === 7) return 0.5;
  if (diff === 8) return 0.25;
  return 0.0;  // 9, 10
}
```

In `_diagRender()`, before calling `board.renderDiagDB(...)`:
```js
const visFrac = _overlayVisibilityFraction(currentDifficulty);
if (visFrac === 0.0) {
  board._dbGroup.innerHTML = "";
  return;  // skip DB/Sentinel overlay entirely
}
```

Pass `visFrac` as an opt to `renderDiagDB` so it can randomly thin the entries:
```js
board.renderDiagDB(dbSource.moves, {
  ...existingOpts,
  visibilityFraction: visFrac,
});
```

**In `board.js` `renderDiagDB`**, apply the fraction using a **deterministic hash** (not `Math.random()`) so the set of shown dots/arrows is stable across redraws:
```js
const visFrac = opts.visibilityFraction != null ? opts.visibilityFraction : 1.0;

// Deterministic per-move inclusion — stable across redraws
function _mvHash(mv) {
  const s = (mv.from || "") + (mv.to || "");
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}
for (const mv of moves) {
  if (visFrac < 1.0 && (_mvHash(mv) % 100) >= Math.round(visFrac * 100)) continue;
  // ... existing rendering code ...
}
```

### 4c. web/app.py — auto-attach sentinel to GameAI based on difficulty

**Location:** The `new_game` handler, after `game_ai` is constructed (~line 1806).

Currently, sentinel is only attached when `use_sentinel=True` from the client message. Replace/extend this with automatic difficulty-based activation:

```python
# Sentinel activation — automatic from difficulty 3+, or explicit from UI
SENTINEL_PROB_BY_DIFF = {
    1: 0.0, 2: 0.0, 3: 0.10, 4: 0.22, 5: 0.33,
    6: 0.50, 7: 0.65, 8: 0.80, 9: 0.90, 10: 1.0,
}
_sent_prob = SENTINEL_PROB_BY_DIFF.get(eff_diff, 0.0)

# Explicit UI toggle always wins; auto-attach if probability permits
_should_attach_sentinel = (
    (use_sentinel and _sentinel_advisor is not None and _sentinel_advisor.is_loaded())
    or
    (_sent_prob > 0.0 and _sentinel_advisor is not None and _sentinel_advisor.is_loaded())
)

if _should_attach_sentinel:
    # sentinel_mode from UI if user explicitly enabled sentinel, else "score_adjust" by default
    _sent_mode = sentinel_mode if use_sentinel else "score_adjust"
    game_ai.set_sentinel(_sentinel_advisor, mode=_sent_mode)
    game_ai._sentinel_activation_prob = _sent_prob   # stored for per-move gating
    log.info("Sentinel attached (diff=%d, prob=%.0f%%, mode=%s)", eff_diff, _sent_prob*100, _sent_mode)
```

**Note:** `_sentinel_activation_prob` is a new attribute you will add to `GameAI`. Store it in `set_sentinel()` or directly after construction.

### 4d. ai/game_ai.py — per-move probability gate in `_apply_sentinel_intervention`

Add `self._sentinel_activation_prob: float = 1.0` to `__init__` (default 1.0 = always).

At the TOP of `_apply_sentinel_intervention`, before any sentinel work:
```python
def _apply_sentinel_intervention(self, board, move, moves):
    if self.sentinel is None:
        return move
    # Probability gate — at lower difficulties, only fire sometimes
    import random as _random
    if _random.random() > self._sentinel_activation_prob:
        return move   # skip this time
    # ... rest of existing code ...
```

---

## Part 5: DB fallback when no Sentinel checkpoint

When `_sentinel_advisor` is None (checkpoint missing), the difficulty-based advisor falls back to using the Malom DB directly for move quality in `score_adjust` mode:

**In `ai/game_ai.py`**, add a `_db_score_adjust` path. This is called instead of sentinel when sentinel is unavailable but difficulty >= 3:

The `GameAI` already has access to `fullgame_db` and `endgame_solved_db`. Add a method:

```python
def _db_score_adjust(self, board, move, moves):
    """Fallback: use Malom DB WDL to score candidates when sentinel is unavailable."""
    if not (self._fullgame_db and self._fullgame_db.is_available()) and \
       not self._endgame_solved_db:
        return move
    try:
        best_move = move
        best_wdl_score = -1  # W=2, D=1, L=0
        wdl_map = {"W": 2, "D": 1, "L": 0}
        for m in moves:
            after = board.apply_move(m)
            # Query from opponent's perspective (after move, it's opponent's turn)
            # so flip: if opponent's result is L, this move is a W for us
            res = None
            if self._endgame_solved_db:
                res = self._endgame_solved_db.query(after)
            if res is None and self._fullgame_db and self._fullgame_db.is_available():
                res = self._fullgame_db.query(after)
            if res:
                # res is from the mover's POV after the move (opponent's turn)
                # So from our POV: flip W↔L
                flipped = {"W": "L", "L": "W", "D": "D"}.get(res, None)
                score = wdl_map.get(flipped, -1)
                if score > best_wdl_score:
                    best_wdl_score = score
                    best_move = m
        return best_move
    except Exception:
        return move
```

Wire this in `_apply_sentinel_intervention`:
```python
# If sentinel unavailable but DB is, use DB fallback for score_adjust
if self.sentinel is None and self.sentinel_mode == "score_adjust":
    return self._db_score_adjust(board, move, moves)
```

**In `web/app.py`**, set `sentinel_mode = "score_adjust"` for the auto-attach path when no sentinel:
```python
if _sent_prob > 0.0 and _sentinel_advisor is None:
    # No sentinel checkpoint — use DB fallback mode
    game_ai.sentinel_mode = "score_adjust"  # tells _apply_sentinel_intervention to try DB fallback
    game_ai._sentinel_activation_prob = _sent_prob
    log.info("Sentinel unavailable — DB fallback mode (diff=%d, prob=%.0f%%)", eff_diff, _sent_prob*100)
```

---

## Part 6: Sentinel badge enrichment — ALREADY DONE

The `ai_move` sentinel badge handler in `game.js` is already fully implemented (lines 990–1036). It uses colour-coded icons, quality%, gap%, and a separate intervention commentary line. **Do not change it.** The existing implementation uses the correct `SentinelAdvice` fields: `player`, `played_move_quality`, `opportunity_gap`, `advisory_message`, `intervention`, `intervention_detail`.

---

## Part 7: Sentinel checkbox visibility gating

The existing sentinel checkbox in the settings panel (`chk-sentinel`, `row-sentinel`) should be hidden at difficulty 9–10 (since it's always forced on there) and shown at all other levels. Also show an explanatory label at difficulty 3–8 indicating it's auto-activated.

```js
function _updateSentinelUI(diff) {
  const row = $("row-sentinel");
  if (!row) return;
  if (diff >= 9) {
    row.style.display = "none";   // always on, don't confuse user
  } else if (diff >= 3) {
    row.style.display = "";
    const prob = [0,0,0,10,22,33,50,65,80][diff] || 0;
    const lbl = row.querySelector("label");
    if (lbl) lbl.title = `Auto-activates ${prob}% of moves at this difficulty`;
  } else {
    row.style.display = "none";   // not active at diff 1-2
  }
}
```

Call `_updateSentinelUI(currentDifficulty)` whenever difficulty changes.

---

## Summary of files to change

| File | Changes |
|---|---|
| `web/app.py` | `get_diagnostic`: add sentinel scoring per move. `new_game`/`setup_game`: auto-attach sentinel by difficulty, add `SENTINEL_PROB_BY_DIFF`. |
| `ai/game_ai.py` | Add `_sentinel_activation_prob` attr. Add probability gate at top of `_apply_sentinel_intervention`. Add `_db_score_adjust()` fallback. |
| `web/static/board.js` | Extend `renderDiagDB`: add `showSentinel` opt, update `dbColor` helper, add `visibilityFraction` random thinning. |
| `web/static/game.js` | Add `diagSentinel` bool, add Sentinel chip handler (mutually exclusive with diagDB), pass `showSentinel` + `visibilityFraction` to `renderDiagDB`, add `currentDifficulty` tracking, add `_overlayVisibilityFraction()`, fetch sentinel status on load, extend sentinel badge handler. |
| `web/templates/index.html` | Add `diag-btn-sentinel` chip + `diag-sentinel-status` span after `diag-btn-db`. |

---

## Syntax checks (no full test suite)

```bash
python -m py_compile web/app.py     && echo "app.py OK"
python -m py_compile ai/game_ai.py  && echo "game_ai.py OK"
node --check web/static/game.js     && echo "game.js OK"
node --check web/static/board.js    && echo "board.js OK"
```

---

## Commit and push

```bash
git add web/app.py ai/game_ai.py web/static/board.js web/static/game.js web/templates/index.html
git commit -m "feat(sentinel-gui): overlay chip, difficulty-gated visibility, auto-advisor activation"
git push origin feat/sentinel-overlay
```

**DO NOT merge to main. DO NOT run the full test suite.**

---

## Key facts to avoid mistakes

1. `board.js` uses SVG — all drawing is via `_el("circle", {...})` and `_el("line", {...})` helpers. Do not use canvas or DOM elements.
2. `$("id")` in `game.js` is `getElementById` shorthand — defined at the top of the file.
3. `moves_out` entries in `get_diagnostic` always have `"from"`, `"to"`, `"capture"` keys (any can be None). The `_diag_ntn()` helper converts these to notation strings for dict keying.
4. The `SentinelAdvisor.advise()` call is CPU-bound — wrap in `asyncio.to_thread()` in `app.py`.
5. `sentinel_score` is `None` when sentinel unavailable — the `dbColor` helper must treat `None` as "no signal" (return null, don't draw).
6. The `diagDB` and `diagSentinel` booleans must be mutually exclusive — toggling one must clear the other.
7. `currentDifficulty` in game.js must be seeded from the initial state message, not just from new_game. Check the `"state"` message handler for where `gameState` is populated.
8. `_sentinel_activation_prob` defaults to `1.0` (always fires) so existing `use_sentinel=True` games work identically.
9. The DB fallback in `_db_score_adjust` must query the position AFTER the move is applied, then flip W↔L (because after applying a move it's the opponent's turn, so their "L" = our "W").
10. Do NOT change the board model, D4 symmetry logic, or move generation in any way.
