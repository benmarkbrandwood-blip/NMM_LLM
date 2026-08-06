# insert_human_pred_overlay

This document defines a complete implementation plan for integrating `human_move_policy_net_v2_candidate.npz` as a **fallback human-move predictor** in the live diagnostic overlay and the Game Explorer. All insertion points, function names, and front-end hooks were verified against the current repository layout and source files in `web/static/board.js`, `web/static/game.js`, `web/static/explorer.js`, and the repository structure exposed in the attached repository listing.[cite:5]

## Background

The current live diagnostic overlay already supports multiple data channels on legal moves: trajectory frequency labels rendered as `T:XX%`, sentinel probability labels rendered as `S:XX%`, overseer probability labels rendered as `O:XX%`, and DB/Malom outcome arrows and halos driven by `db_delta`, `eg_flag`, and `eg_dtw` inside `renderDiagDB()` in `web/static/board.js`.[cite:5] The existing UI also exposes a dedicated trajectory toggle (`diag-btn-traj`) and the main client render path already passes `showTraj`, `showSentinel`, and `showOverseer` into `board.renderDiagDB()` from `_diagRender()` in `web/static/game.js`.[cite:5]

The new policy net should not compete with HumanDB trajectory data; it should only activate when **all** legal moves in a position lack real trajectory coverage, meaning no move has `traj_freq > 0`.[cite:5] In those fallback positions, the backend should infer likely human play from `human_move_policy_net_v2_candidate.npz`, return per-move `pred_human_prob`, expose `has_traj_data: false`, and let the front end switch terminology from **HumanDB best** to **Pred Human** while reusing the existing trajectory-oriented overlay affordances.[cite:5]

## Files changed

| File | Purpose |
|---|---|
| `ai/human_pred_net.py` | New loader/inference wrapper for `human_move_policy_net_v2_candidate.npz`.[cite:5] |
| `ai/value_net.py` or shared feature module | Reuse or extract the existing 79-feature encoder so policy-net inputs exactly match training expectations.[cite:5] |
| `web/app.py` | Startup load, diagnostic fallback inference, explorer fallback inference, response schema additions.[cite:5] |
| `web/static/board.js` | Add `predLabel()` and render `P:XX%` blue text in all relevant branches.[cite:5] |
| `web/static/game.js` | Pass `showPredHuman` and `hasTrajData`; swap top-right label text/color when fallback is active.[cite:5] |
| `web/static/explorer.js` | Add predicted-human designation logic and equal-slice pie overlay rendering when multiple best markers coincide.[cite:5] |
| `web/static/style.css` | Add shared Pred Human label/legend styling if needed.[cite:5] |
| `web/templates/index.html` | Change `Traj` chip label to `Traj/Pred`; optionally rename Overseer chip to `AI choice` if desired.[cite:5] |

## Policy net module

> **Implementation note (2026-08-01):** Do NOT create `ai/human_pred_net.py`.
> `ai/human_move_policy_advisor.py` already exists with the correct implementation:
> 4-layer architecture (w0/b0 through w3/b3), band-aware 82-dim input (79 board
> features + 3 Elo band one-hot), and `try_load()` helper.  The plan’s proposed
> `HumanMovePolicyNet` class was incorrect on three counts: 3 layers vs actual 4,
> keys `w1..w3` vs actual `w0..w3`, and no band handling.
> Use `HumanMovePolicyAdvisor.probs(board, legal_moves, elo_band)` — default
> band is `"middle"` for the overlay.

## Feature extraction

> **Implementation note (2026-08-01):** Do NOT create `ai/feature_encoding.py`.
> `board_to_features` from `ai/value_net` is already wrapped inside
> `HumanMovePolicyAdvisor._successor_features()` and correctly applies the
> 82-dim encoding (79 board features + 3-way Elo band one-hot).

## Backend startup

Load the model once at app startup rather than per request. The repository’s server entry point is `web/app.py`, and that is the correct location to initialize a process-wide predictor object used by both the WebSocket diagnostic flow and explorer endpoints.[cite:5]

### Insertion point

In `web/app.py`, near existing AI/database startup initialization, add imports and model-path resolution immediately after other `ai.*` imports and before the FastAPI app begins serving requests.[cite:5]

### Actual code block (implemented 2026-08-01)

```python
from ai.human_move_policy_advisor import try_load as _try_load_hmpa
_human_move_policy_path = _ROOT / "data" / "human_move_policy_net_v2_candidate.npz"
_human_move_policy_advisor = _try_load_hmpa(_human_move_policy_path)
if _human_move_policy_advisor is not None:
    _hmpa_kb = round(_human_move_policy_path.stat().st_size / 1024, 1)
    log.info("HumanMovePolicyAdvisor: loaded from %s (%s KB)", _human_move_policy_path, _hmpa_kb)
else:
    log.info("HumanMovePolicyAdvisor: not found at %s — pred-human overlay disabled", _human_move_policy_path)
```

### Optional hardening

Log one startup line indicating whether the net loaded successfully and from which path, because this feature should fail safe rather than crash the server if the `.npz` file is absent.[cite:5]

## Diagnostic WebSocket flow

The live game path already uses `_diagOnReceive()` and `_diagRender()` on the client side, so the backend should keep its response contract simple: always return legal moves, and additionally return `has_traj_data` plus `pred_human_prob` only when fallback prediction is needed.[cite:5]

### Insertion point

In `web/app.py`, locate the `get_diagnostic` WebSocket message handler where move dictionaries are assembled with existing fields like `traj_freq`, `db_delta`, `eg_flag`, `eg_dtw`, `sentinel_score`, and `overseer_prob`.[cite:5] Insert the fallback logic **after** all legal moves have been computed and trajectory frequencies have been attached, but **before** the payload is emitted to the socket.[cite:5]

### Required algorithm

1. Compute or preserve the legal move list as normal.[cite:5]
2. Determine `has_traj_data = any((mv.get("traj_freq") or 0) > 0 for mv in moves)`.[cite:5]
3. If `has_traj_data` is `True`, set `pred_human_prob = None` for all moves or omit the field entirely.[cite:5]
4. If `has_traj_data` is `False` and `HUMAN_POLICY_NET` is loaded, encode one feature row per move, run inference, and attach `pred_human_prob: float` to each move dict.[cite:5]
5. Emit top-level `has_traj_data` in the WebSocket response in all cases so the front end can swap wording deterministically.[cite:5]

### Example backend code

```python
has_traj_data = any((mv.get("traj_freq") or 0) > 0 for mv in moves)

if not has_traj_data and HUMAN_POLICY_NET is not None:
    feature_rows = [
        encode_move_policy_features(game, mv, game.to_move)
        for mv in moves
    ]
    probs = HUMAN_POLICY_NET.predict_probs(feature_rows)
    for mv, prob in zip(moves, probs):
        mv["pred_human_prob"] = float(prob)
else:
    for mv in moves:
        mv["pred_human_prob"] = None

payload["has_traj_data"] = has_traj_data
payload["moves"] = moves
```

### Contract invariant

The fallback is **position-wide**, not per move.[cite:5] Mixed states such as one move using `traj_freq` and another using `pred_human_prob` in the same position should be treated as invalid and must never be returned.[cite:5]

## Explorer backend flow

The explorer must receive the same fallback signal so it can rename the designation from HumanDB to Pred Human and compute the best predicted move when no trajectory data exists.[cite:5] This should be implemented in the explorer endpoint that currently returns move annotations for the SVG/node overlay in `web/static/explorer.js`.[cite:5]

### Insertion point

In the explorer response builder inside `web/app.py`, add the same `has_traj_data` computation after move scoring and trajectory annotation are complete.[cite:5]

### Required additions

- Add top-level `has_traj_data: bool`.[cite:5]
- Attach `pred_human_prob` to each move when `has_traj_data == false`.[cite:5]
- Compute `is_pred_human_best` on the move(s) with maximal `pred_human_prob` in fallback mode.[cite:5]
- Ensure `is_human_best` remains reserved for real trajectory-backed HumanDB selection only.[cite:5]

### Example logic

```python
has_traj_data = any((mv.get("traj_freq") or 0) > 0 for mv in moves)

if not has_traj_data and HUMAN_POLICY_NET is not None and moves:
    probs = HUMAN_POLICY_NET.predict_probs([
        encode_move_policy_features(game, mv, game.to_move)
        for mv in moves
    ])
    best = float(np.max(probs))
    for mv, prob in zip(moves, probs):
        p = float(prob)
        mv["pred_human_prob"] = p
        mv["is_pred_human_best"] = abs(p - best) < 1e-9
        mv["is_human_best"] = False
else:
    for mv in moves:
        mv["pred_human_prob"] = None
        mv["is_pred_human_best"] = False
```

## `game.js` changes

The client already has the correct central render point: `_diagRender()` at approximately the `board.renderDiagDB()` call site, and `_diagOnReceive()` where incoming diagnostic metadata is normalized before redraw.[cite:5]

### `_diagRender()` insertion

In `web/static/game.js`, locate this existing call pattern inside `_diagRender()`:

```javascript
board.renderDiagDB(dbSource.moves, {
  phase: currentPhase,
  selectedSrc: board.selected,
  showTraj: diagTraj,
  showDB: diagDB,
  showSentinel: diagSentinel,
  showOverseer: diagOverseer,
  visibilityFraction: diagVisibilityFraction,
});
```

Extend it to pass the two new flags:

```javascript
board.renderDiagDB(dbSource.moves, {
  phase: currentPhase,
  selectedSrc: board.selected,
  showTraj: diagTraj,
  showPredHuman: diagTraj,
  hasTrajData: dbSource.has_traj_data ?? true,
  showDB: diagDB,
  showSentinel: diagSentinel,
  showOverseer: diagOverseer,
  visibilityFraction: diagVisibilityFraction,
});
```

This preserves the existing user toggle model: the same chip that exposes trajectory information also exposes fallback predicted-human information when trajectory coverage is absent.[cite:5]

### `_diagOnReceive()` insertion

In `web/static/game.js`, inside `_diagOnReceive(msg)`, after the incoming payload is cached and before `_diagRender()` is called, update the top-right TO MOVE legend entry.[cite:5]

### Example code

```javascript
const humanLabel = document.getElementById("to-move-human-label");
if (humanLabel) {
  const hasTrajData = msg.has_traj_data !== false;
  humanLabel.textContent = hasTrajData ? "HumanDB best" : "Pred Human";
  humanLabel.style.color = hasTrajData ? "#a86fdf" : "#5591c7";
}
```

This change should happen on every diagnostic refresh so the label always matches the current position rather than becoming stale across moves or replay jumps.[cite:5]

## `board.js` changes

`renderDiagDB()` in `web/static/board.js` is the correct and only place to add the blue fallback label because it already handles placement/capture overlays, selected-piece move overlays, and unselected per-source overlays for other data channels.[cite:5]

### New options

At the top of `renderDiagDB()`, immediately after the existing option extraction:

```javascript
const showPredHuman = opts.showPredHuman || false;
const hasTrajData   = opts.hasTrajData !== false;
```

### New helper

Insert this directly after `overseerLabel()` and before `dtwLabel()`:

```javascript
const predLabel = (prob) => {
  if (!showPredHuman || hasTrajData || prob == null) return null;
  const pct = Math.round(prob * 100);
  if (pct < 1) return null;
  return `P:${pct}%`;
};
```

### Placement / capture branch

Inside the `phase === "place" || phase === "capture"` branch, add:

```javascript
const plbl = predLabel(mv.pred_human_prob);
```

Then change the label gate from:

```javascript
if (slbl || olbl || dlbl || freq > 0) {
```

to:

```javascript
if (slbl || olbl || dlbl || plbl || freq > 0) {
```

Insert the Pred Human text block immediately after the trajectory `T:XX%` block and before DTW/Sentinel/Overseer so the human-oriented overlays stay grouped together:

```javascript
if (plbl) {
  const t = _el("text", { x, y: ty, "font-size":"9", fill:"#5591c7",
    "text-anchor":"middle", "font-family":"monospace",
    stroke:"white", "stroke-width":"2.5", "stroke-linejoin":"round",
    "paint-order":"stroke" });
  t.textContent = plbl;
  this._dbGroup.appendChild(t);
  ty -= 11;
}
```

### Movement selected branch

Inside the selected-source movement section, add:

```javascript
const plbl = selSrc ? predLabel(mv.pred_human_prob) : null;
```

Change the gate from:

```javascript
if (slbl || olbl || dlbl || freq > 0) {
```

to:

```javascript
if (slbl || olbl || dlbl || plbl || freq > 0) {
```

Insert the blue label immediately after the purple trajectory label and before DTW:

```javascript
if (plbl) {
  const t = _el("text", { x: x + 1, y: ty, "font-size":"8", fill:"#5591c7",
    "text-anchor":"middle", "font-family":"monospace",
    stroke:"white", "stroke-width":"2.5", "stroke-linejoin":"round",
    "paint-order":"stroke" });
  t.textContent = plbl;
  this._dbGroup.appendChild(t);
  ty -= 10;
}
```

### Movement unselected per-source branch

Add a new block parallel to the existing per-source sentinel and overseer aggregations. This is required because, when no piece is selected, `renderDiagDB()` shows best-per-source summaries above each movable piece rather than per-destination labels.[cite:5]

Insert the new block **after** the existing per-source sentinel label block and **before** the per-source overseer block so label stacking stays consistent:

```javascript
if (!selSrc && showPredHuman && !hasTrajData) {
  const srcBestPred = new Map();
  const srcHasSent = new Set();
  for (const mv of moves) {
    if (!mv.from) continue;
    if (mv.pred_human_prob != null) {
      const prev = srcBestPred.get(mv.from);
      if (prev == null || mv.pred_human_prob > prev)
        srcBestPred.set(mv.from, mv.pred_human_prob);
    }
    if (showSentinel && mv.sentinel_score != null) {
      srcHasSent.add(mv.from);
    }
  }
  for (const [src, prob] of srcBestPred) {
    const plbl = predLabel(prob);
    if (!plbl) continue;
    const [x, y] = nodeXY(src);
    const sentOffset = srcHasSent.has(src) ? -11 : 0;
    const t = _el("text", { x, y: y - PIECE_R - 3 + sentOffset,
      "font-size":"10", "font-weight":"bold",
      fill:"#5591c7", "text-anchor":"middle", "font-family":"monospace",
      stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
      "paint-order":"stroke" });
    t.textContent = plbl;
    this._dbGroup.appendChild(t);
  }
}
```

### Notes on stacking

The current implementation uses `ty -= 11` or `ty -= 10` to stack labels vertically, and the new Pred Human label should obey the same spacing discipline so overlay density remains predictable.[cite:5] Because `P:XX%` conceptually replaces HumanDB fallback information, it should appear in the same family as `T:XX%` rather than after sentinel/overseer labels.[cite:5]

## `index.html` changes

The trajectory chip label should be updated to communicate dual behavior in a compact way.[cite:5]

### Insertion point

In `web/templates/index.html`, locate the button or chip element with id `diag-btn-traj` and change its visible text from:

```html
Traj
```

to:

```html
Traj/Pred
```

If the current UI copy still uses `Overseer`, an optional parallel copy edit is to rename that chip to `AI choice`, but that is separate from the core policy-net feature and should only be included if you want the plan to bundle UI terminology cleanup.[cite:5]

## Explorer rendering

The repository contains a dedicated `web/static/explorer.js`, so explorer logic should be implemented there rather than inline in the template.[cite:5] This page needs two upgrades: fallback designation renaming and multi-color node rendering when multiple best categories share the same square.[cite:5]

### Explorer data rules

For each explorer position:

- Use **HumanDB best** only when `has_traj_data === true` and `is_human_best === true`.[cite:5]
- Use **Pred Human** only when `has_traj_data === false` and `is_pred_human_best === true`.[cite:5]
- Continue to use existing Sentinel best and Heuristic best markers unchanged.[cite:5]

### Designation color constants

Add or centralize constants near explorer overlay rendering helpers:

```javascript
const DESIGNATION_COLORS = {
  human: "#a86fdf",
  predHuman: "#5591c7",
  sentinel: "#66bb6a",
  heuristic: "#f5a623",
};
```

Use the repository’s requested blue `#5591c7` for predicted-human fallback and preserve the existing purple `#a86fdf` for true HumanDB.[cite:5]

### Pie-slice helper

Add a small SVG arc-path helper in `explorer.js`:

```javascript
function pieArcPath(cx, cy, r, startAngle, endAngle) {
  const rad = deg => (deg - 90) * Math.PI / 180;
  const x1 = cx + r * Math.cos(rad(startAngle));
  const y1 = cy + r * Math.sin(rad(startAngle));
  const x2 = cx + r * Math.cos(rad(endAngle));
  const y2 = cy + r * Math.sin(rad(endAngle));
  const largeArc = (endAngle - startAngle) > 180 ? 1 : 0;
  return [
    `M ${cx} ${cy}`,
    `L ${x1} ${y1}`,
    `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
    "Z"
  ].join(" ");
}
```

### Composite designation renderer

Create a renderer that accepts a list of categories occupying the same node and either draws a solid circle for one category or equal pie slices for two or three categories:

```javascript
function drawDesignationCircle(svgGroup, cx, cy, r, designations) {
  if (!designations || !designations.length) return;

  if (designations.length === 1) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", r);
    circle.setAttribute("fill", DESIGNATION_COLORS[designations[0]]);
    circle.setAttribute("stroke", "#1a1208");
    circle.setAttribute("stroke-width", "1.5");
    svgGroup.appendChild(circle);
    return;
  }

  const sliceAngle = 360 / designations.length;
  designations.forEach((key, idx) => {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pieArcPath(cx, cy, r, idx * sliceAngle, (idx + 1) * sliceAngle));
    path.setAttribute("fill", DESIGNATION_COLORS[key]);
    svgGroup.appendChild(path);
  });

  const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  ring.setAttribute("cx", cx);
  ring.setAttribute("cy", cy);
  ring.setAttribute("r", r);
  ring.setAttribute("fill", "none");
  ring.setAttribute("stroke", "#1a1208");
  ring.setAttribute("stroke-width", "1.5");
  svgGroup.appendChild(ring);
}
```

### Explorer aggregation logic

Where explorer currently determines node markers, replace one-marker-per-node logic with an accumulator such as:

```javascript
const nodeDesignations = new Map();

for (const mv of moves) {
  const key = mv.to || mv.from;
  if (!key) continue;
  if (!nodeDesignations.has(key)) nodeDesignations.set(key, new Set());
  const set = nodeDesignations.get(key);

  if (hasTrajData && mv.is_human_best) set.add("human");
  if (!hasTrajData && mv.is_pred_human_best) set.add("predHuman");
  if (mv.is_sentinel_best) set.add("sentinel");
  if (mv.is_heuristic_best) set.add("heuristic");
}

for (const [pos, set] of nodeDesignations.entries()) {
  const [cx, cy] = nodeXY(pos);
  drawDesignationCircle(overlayGroup, cx, cy, 8, Array.from(set));
}
```

That yields the requested 50/50 and 33.33/33.33/33.33 visual split when categories collide on the same move destination.[cite:5]

### Explorer legend update

Any textual legend in explorer should also switch from HumanDB to Pred Human when `has_traj_data === false` for the current position.[cite:5] This can be done either dynamically or by rendering both legend entries and hiding one, but dynamic text is cleaner and avoids contradictory labels.[cite:5]

## CSS additions

Most live overlay text styling in `board.js` is inline SVG attribute styling rather than class-based CSS, so CSS needs are modest.[cite:5] The main CSS additions are for explorer legends or any HTML labels introduced for Pred Human.[cite:5]

### Suggested CSS

Add to `web/static/style.css`:

```css
.pred-human-label {
  color: #5591c7;
}

.legend-dot-pred-human {
  background: #5591c7;
}

.legend-dot-humandb {
  background: #a86fdf;
}
```

If explorer legend markers are SVG-based rather than HTML-based, keep the color constants in JS and omit CSS duplication.[cite:5]

## Rollout sequence

1. Create `ai/human_pred_net.py` and validate `.npz` loading against the actual stored key names.[cite:5]
2. Extract or reuse the exact 79-feature encoder from the existing value-net/training path; add shape assertion tests.[cite:5]
3. Load the predictor once in `web/app.py` startup and log success/failure without crashing on absence.[cite:5]
4. Add fallback inference and `has_traj_data` emission to the `get_diagnostic` WebSocket flow.[cite:5]
5. Add the same fallback inference and `is_pred_human_best` emission to the explorer endpoint.[cite:5]
6. Update `web/templates/index.html` chip text from `Traj` to `Traj/Pred`.[cite:5]
7. Update `web/static/game.js` to pass `showPredHuman`/`hasTrajData` and to swap `#to-move-human-label` text and color dynamically.[cite:5]
8. Update `web/static/board.js` with `predLabel()` and rendering in placement, selected movement, and unselected per-source paths.[cite:5]
9. Update `web/static/explorer.js` to support Pred Human designation, node aggregation, and equal-slice pie rendering for overlapping best markers.[cite:5]

## Invariants

- The policy net is a **fallback only** and is never shown in a position where any move has `traj_freq > 0`.[cite:5]
- `has_traj_data` is computed once per position and governs both label text and overlay type.[cite:5]
- Move ordering must remain stable from legal move generation through feature encoding, inference, and payload serialization.[cite:5]
- `pred_human_prob` must sum to approximately 1.0 across all legal moves in fallback positions.[cite:5]
- Real HumanDB labels remain purple; predicted-human labels are blue.[cite:5]
- Explorer should never show both HumanDB and Pred Human for the same position.[cite:5]
- Overlapping designation circles must preserve all active categories rather than letting the last one drawn win.[cite:5]
- Front-end fallback wording must update on every diagnostic refresh, not only on initial page load.[cite:5]
- Absence of the `.npz` file must degrade gracefully: `has_traj_data` can still be false, but `pred_human_prob` should stay null and no crash should occur.[cite:5]
- Any future training refresh of the model must not require front-end changes as long as the output remains per-move probabilities.[cite:5]

## Verification checklist

- [ ] Confirm `human_move_policy_net_v2_candidate.npz` key names (`w1`, `b1`, etc.) match the loader implementation.[cite:5]
- [ ] Confirm the shared encoder returns exactly 79 features for every legal move.[cite:5]
- [ ] Confirm a position with real HumanDB data still shows `T:XX%` and never shows `P:XX%`.[cite:5]
- [ ] Confirm a position with zero trajectory coverage shows `P:XX%` blue labels and top-right `Pred Human` text.[cite:5]
- [ ] Confirm placement overlays, selected movement overlays, and unselected source summaries all render predicted-human data correctly.[cite:5]
- [ ] Confirm `diag-btn-traj` now reads `Traj/Pred` and still toggles the overlay correctly.[cite:5]
- [ ] Confirm explorer uses purple HumanDB markers only when `has_traj_data === true`.[cite:5]
- [ ] Confirm explorer uses blue Pred Human markers only when `has_traj_data === false`.[cite:5]
- [ ] Confirm two-way and three-way marker overlaps render as equal pie slices rather than a single solid circle.[cite:5]
- [ ] Confirm server behavior remains stable when the policy-net file is missing or malformed, with a logged warning instead of an application crash.[cite:5]
