# Post-Game Assessment Plan

> **Status: Future direction — not yet prioritised.**
> This document is a preliminary design, not an authoritative implementation plan.
> It will need a review pass against the live codebase before any code is written.
>
> **Last audited: 2026-09-19** — full rewrite; stale references removed.
> Horizon-effect (short/long node budget) approach retired; see Open Questions §6.

---

## Motivation

After a game ends, NMM_LLM can evaluate every move without the time pressure of live play.
Several diagnostic signals are available that each measure move quality from a different
perspective — heuristic search, learned move quality (Sentinel), blunder-zone density (GapNet),
historical trajectory data, human behaviour, and the Generalist AI policy. Comparing the move
actually played against what was best available under each signal gives a *regret* per move.
The ply with the largest regret is the **turning point** — the moment the game was most likely
decided. Malom confirms whether a turning-point candidate was objectively damaging. The LLM
then produces a short commentary grounded in those facts.

---

## Diagnostic Signals

### 1. Heuristic search (`score_move()` in clean mode)

A clean `GameAI` instance (no HumanPref, no Sentinel adjustment, no trajectory bonuses —
all default to `None`) re-scores the position after each move. Per ply:

- `heuristic_score_white`: absolute heuristic evaluation of the board *after* the move,
  normalised to White's perspective (positive = White winning). This is the **curve value**
  used for game-arc visualisation and turning-point detection. For White plies use the raw
  `evaluate()` result; for Black plies negate it.
- `score_played`: heuristic score of the move actually made (mover's perspective, as
  returned by `score_move()`).
- `score_best`: heuristic score of the top-ranked alternative.
- `best_alt`: the move that would have ranked first.
- `r_h = score_best − score_played` (heuristic regret, ≥ 0).

The sequence `h(t) = heuristic_score_white` over all plies is the **heuristic score
curve**. The ply with the steepest drop in `h(t)` is a turning-point candidate under
this signal.

"Clean" means: construct `GameAI` with default args (no optional advisor args passed);
do not call `set_sentinel()`. This is the existing default construction path — no new
flag required. Confirmed in `ai/game_ai.py`: `human_pref_net=None` (line 458),
`self.sentinel = None` (line 553), `self._trajectory_db = None` (line 544) are all
default. A freshly constructed `GameAI` with no extra arguments is already clean.

### 2. Sentinel (`SentinelAdvisor.advise()`)

`SentinelAdvisor.advise(board_state, candidates, player, played_move_idx)` scores all
candidate moves in one batched forward pass. Returns a `SentinelAdvice` with per-move
quality scores.

Per ply:
- `sentinel_score_white`: Sentinel quality score for the move made, normalised to
  White's perspective (negate for Black plies). This is the **curve value** — the
  sequence `s(t) = sentinel_score_white` over all plies is the **Sentinel score curve**.
  The ply with the steepest drop in `s(t)` is a turning-point candidate under this signal.
- `sentinel_played`: raw Sentinel quality score from the mover's perspective (as returned
  by `SentinelAdvice.played_move_quality`).
- `sentinel_best`: highest Sentinel score among all candidates.
- `r_s = sentinel_best − sentinel_played` (`opportunity_gap` in `SentinelAdvice`).

Sentinel can be called post-hoc for arbitrary positions — it is not restricted to live
play. Pass the full legal move list; use the index of the played move as
`played_move_idx`.

### 3. GapNet blunder-zone density (`gap_net.predict()`)

`gap_net.predict(board, color)` returns a tanh value in (−1, +1); converted to a
blunder-zone density in (0, 1) via `(raw + 1) / 2`. Near 1 = humans frequently blunder
from this position; near 0 = humans typically play well here.

Per ply:
- `blunder_zone_score`: blunder-zone density for the board *before* the move, from the
  mover's perspective. Always in [0, 1]; `None` when the model is not loaded.

Unlike the move-level signals, GapNet measures the *position* rather than the move. It
answers "was this a known trap?" rather than "was this move poor?" Used alongside `r_h`
and `r_s`, it distinguishes two qualitatively different poor moves:

- High `blunder_zone_score` + high `r_h`: *the player fell into a known trap* — a
  position that trips most players.
- Low `blunder_zone_score` + high `r_h`: *an anomalous error in a safe-looking position*
  — more surprising, and arguably more telling about the player's level.
- High `blunder_zone_score` + low `r_h`: *good defensive play* — navigated a known trap
  successfully (worth highlighting positively in LLM commentary).

In `_detect_turning_point`, `blunder_zone_score` acts as a tiebreaker once Stage 5
upgrades the hierarchy: two plies with equal `drop(t)` prefer the one with higher
blunder density (the known trap, not noise).

GapNet is a *contextual* signal. It does not adjudicate move quality independently.

### 4. Trajectory signal (`TrajectoryDB.query()`)

`TrajectoryDB.query(board, current_color)` returns a `{notation: delta}` dict for every
candidate move at the current position. Delta is a confidence-weighted win-rate offset
(range −0.5 to +0.5; positive = historically good for `current_color`). Returns `{}` when
fewer than `min_samples` games cover the position.

Per ply (where data is available):
- `traj_delta_played`: the delta for the move actually made.
- `traj_delta_best`: the highest delta among all candidates.
- `r_t = traj_delta_best − traj_delta_played` (trajectory regret).
- `traj_n`: total sample count at this position (confidence indicator).

A large `r_t` with high `traj_n` means the played move diverges from historically
winning lines.

### 5. Human preference (`HumanMovePolicyAdvisor`, teacher v4)

`HumanMovePolicyAdvisor.probs(board, legal_moves, elo_band)` returns a probability
distribution over legal moves for a player at the given Elo band.

Per ply:
- `policy_prob`: probability that a player at this band would have played the move made.
- `policy_top_move`: the move with the highest probability, and its probability.
- `policy_prob_source`: `"empirical"` if `n > 50` events at this position in
  `human_db_candidate_new.sqlite:moves_elo_bins`; `"learned"` otherwise.
- `policy_support_n`: the n used to decide source.
- A move is flagged `unconventional` if `policy_prob` falls below a configurable
  threshold (calibrate during validation — do not fix a number before seeing data).

Human preference is a *descriptive* signal. High human frequency does not override a
heuristic or Malom-confirmed downgrade.

### 6. Generalist AI policy

`GeneralistPolicyAdvisor.probs(board, legal_moves)` (or equivalent interface on the
scaffolded generalist) returns a probability distribution over legal moves reflecting
what the strongest trained AI would play. Unlike human preference, this signal asks
"what would the best available AI do?"

Per ply:
- `generalist_policy_prob`: probability the generalist would play the move made.
- `generalist_top_move`: the generalist's preferred alternative.
- `generalist_value_after`: optional value-head win probability after the move,
  White-normalised — a third score curve if the value head is exposed.

**Own-game caveat:** when the generalist played one side of the game, its scores for
its own moves are trivially high and carry no signal. This field should be tagged
`generalist_self_assessed: true` in that case and treated as uninformative. The signal
is most useful for human-vs-human games or for assessing the human's side of a
human-vs-AI game.

Generalist policy is a *descriptive* signal at the same tier as human preference. It
does not override Malom, Sentinel, or heuristic adjudication.

---

## Score Curves and Turning Point

### Score curves

Two score curves are traced over every ply of the game, both normalised to White's
perspective so they form a consistent series regardless of who is moving:

- **Heuristic curve** `h(t)`: `heuristic_score_white` at each ply. Derived from the
  clean `score_move()` call; always available.
- **Sentinel curve** `s(t)`: `sentinel_score_white` at each ply. Available when the
  Sentinel model is loaded.

The curves show the game arc — a sustained downward trend for White means the position
deteriorated. A sudden steep drop identifies the moment the game changed. Both curves
are stored in `PostGameAnnotation.heuristic_curve` and `PostGameAnnotation.sentinel_curve`
for LLM context and (later) UI rendering.

### Turning point and regret

Per-signal regret is computed at each ply where data is available:

| Signal | Regret | Source |
|--------|--------|--------|
| `r_h` | `score_best − score_played` | Heuristic search |
| `r_s` | `sentinel_best − sentinel_played` | Sentinel |
| `r_t` | `traj_delta_best − traj_delta_played` | Trajectory DB |

Quality adjudication uses the following fallback hierarchy — signals are not independent
votes and must not be summed:

1. **Malom WDL** (when available): objective authority. A `win_to_draw`, `win_to_loss`,
   or `draw_to_loss` transition is the definitive classification. Severity order:
   `win_to_loss` > `draw_to_loss` > `win_to_draw`.
2. **Sentinel + Heuristic agreement** (Malom unavailable): when both `r_s` and `r_h`
   indicate a significant gap at the same ply, that ply is flagged `poor_candidate`
   with elevated confidence. Neither signal alone is sufficient at this tier.
3. **Heuristic alone** (Sentinel unavailable or disagrees): `r_h` above the calibrated
   threshold flags a ply as `poor_candidate` with lower confidence.

**Turning point detection:**

- **With Malom**: `argmax_t` over severity class, ties broken by `r_h`.
- **Without Malom**: `argmax_t drop(t)` where `drop(t)` is the steepest combined step
  in the curves. If both curves are available, use:
  `drop(t) = α × Δh(t) + β × Δs(t)` where `Δh(t) = h(t-1) − h(t)` (positive = White
  losing ground) and `α`, `β` are calibration parameters (leave open until validation).
  If only `h(t)` is available, use `drop(t) = Δh(t)`.
- `turning_point_oracle`: `"malom_full"` | `"retrograde_wdl"` | `"sentinel+heuristic"` |
  `"heuristic"` — records which path was used.

**Poor-move threshold**: flag `poor_candidate` if `r_h` exceeds threshold AND/OR `r_s`
corroborates (both above threshold). Malom confirmation upgrades to `confirmed_poor`.
Record `oracle_source` for every Malom lookup.

Exclude positions where `wdl_before` is already a loss for the mover. Fail closed —
never substitute a neutral default when Malom abstains.

---

## LLM Synthesis

The LLM's role is prose synthesis over already-decided facts. It does not adjudicate
move quality.

Extend `MillsLLM.debrief_game()` to accept a `PostGameAnnotation` object. Build a
structured prompt with these sections:

**GAME FACTS**: result, winner/loser colour, total plies, opening name (if known).

**SCORE TREND**: a brief characterisation of the game arc derived from the heuristic
and Sentinel curves — e.g. "White led throughout", "Black recovered after ply 20",
"position was balanced until ply 34". Summarise the curve shape in words; do not quote
raw score numbers in the LLM prompt.

**TURNING POINT**: ply number, move played, best alternative (`best_alt` from the
heuristic scorer), Malom downgrade class if confirmed (W/D/L — never a decimal), or
heuristic/sentinel regret if Malom abstained, oracle source (`turning_point_oracle`).

**OTHER POOR MOVES** (optional, if more than one `confirmed_poor`): ply, move, downgrade
class or heuristic regret. Keep brief — one line per move.

**HUMAN CONFORMANCE** (optional, post-validation): for `unconventional` moves, the
population's preferred move and its frequency. Label source as empirical or learned.

The LLM output should be 3–5 sentences: one on the game's overall character, one
focused on the turning point, and optionally one on notable patterns.

**Hard prompt constraints:**
- Malom outputs are W, D, or L. Never write a decimal Malom figure (e.g., "0.72").
- Label any continuous figure (policy_prob, sentinel score) explicitly as a model
  probability or empirical win rate.
- Do not invent move quality claims not present in the fact block.

---

## Data Classes

### `MoveAnnotation` (per ply)

```
ply                   int
color                 "W" | "B"
phase                 "placement" | "movement" | "fly"
move_played           str              # notation
best_alt              str | None       # top heuristic alternative

# Heuristic signal
heuristic_score_white float            # absolute eval after move, White-normalised (curve value)
score_played          float            # heuristic score of move made (mover's perspective)
score_best            float            # heuristic score of best alternative
r_h                   float            # heuristic regret (score_best − score_played)

# Sentinel signal
sentinel_score_white  float | None     # Sentinel score after move, White-normalised (curve value)
sentinel_played       float | None     # raw Sentinel quality for move made (mover's perspective)
sentinel_best         float | None     # highest Sentinel score among candidates
r_s                   float | None     # sentinel regret (sentinel_best − sentinel_played)

# GapNet blunder-zone density
blunder_zone_score    float | None     # (raw+1)/2 for board BEFORE move, mover's perspective [0,1]

# Trajectory signal
traj_delta_played     float | None
traj_delta_best       float | None
r_t                   float | None     # trajectory regret
traj_n                int | None

# Malom adjudication
wdl_before            "W"|"D"|"L"|None # mover's perspective, pre-move
wdl_after             "W"|"D"|"L"|None # mover's perspective, post-move
oracle_source         "malom_full" | "retrograde_wdl" | "none"
abstained_reason      str | None
quality               "confirmed_poor" | "poor_candidate" | "clean"

# Human policy
policy_prob           float | None
policy_top_move       str | None
policy_top_prob       float | None
policy_prob_source    "empirical" | "learned" | None
policy_support_n      int | None
is_unconventional     bool

# Generalist AI policy
generalist_policy_prob  float | None     # probability generalist plays this move
generalist_top_move     str | None       # generalist's preferred alternative
generalist_value_after  float | None     # value-head win prob after move, White-normalised
generalist_self_assessed bool            # True when generalist played this side (uninformative)
```

### `PostGameAnnotation` (game level)

```
moves                 list[MoveAnnotation]
heuristic_curve       list[float]      # heuristic_score_white at each ply (index = ply)
sentinel_curve        list[float|None] # sentinel_score_white at each ply; None if unavailable
turning_point_ply     int | None
turning_point_quality str              # WDL class, or "r_h:<value>", or "r_h+r_s:<value>"
turning_point_oracle  str              # "malom_full"|"retrograde_wdl"|"sentinel+heuristic"|"heuristic"
opening_name          str | None
```

---

## Implementation Prerequisites

Two gaps must be resolved before code is written:

**1. `debrief_game()` interface extension.**
`MillsLLM.debrief_game()` currently receives only winner, loser, opening_name, move count.
It must be extended to accept a `PostGameAnnotation` and build the structured prompt
sections from it. This is the only mandatory change to existing interfaces.

**2. Sentinel called post-hoc.**
Sentinel's `advise()` requires a list of candidate move dicts in the same format the
heuristic search produces. In post-game replay, the board states are available (game record),
but the candidate list must be regenerated by calling `board.get_legal_moves()` for each
ply. Confirm this produces the correct format before wiring the assessment loop.

No Rust changes are required. No new training is required.

---

## Validation

Before any UI work, validate on 5–10 games (mix of human-vs-human and human-vs-AI):

- Does the turning point correspond to the moment a human observer would identify as
  the game-deciding move?
- Does the LLM commentary stay grounded in the fact block — no invented quality claims?
- What is the per-game wall-clock time? (Determines scheduling approach — see below.)
- What `r_h` threshold produces a good signal-to-noise ratio for `poor_candidate`?

Gate all further work on this review.

---

## Scheduling (Open Decision)

Per-game assessment wall-clock time is unknown until validation. Options:

- **Background WebSocket push**: trigger asynchronously on game end; push annotation
  markers to the client as each ply completes.
- **Pre-compute at game end**: run synchronously before the post-game screen appears;
  adds latency to the transition.
- **On-demand per ply**: assess a ply only when the user clicks; cheap, no summary view.

Choose after validation establishes actual latency on representative hardware.

---

## Dependencies and Build Order

| Component | Requires | New training? |
|-----------|----------|---------------|
| Clean scorer (GameAI default construction) | Verification only | No |
| Per-ply heuristic + sentinel + trajectory scoring | Clean scorer | No |
| Malom adjudication | Per-ply scoring + Malom DB mount | No |
| Human policy signals | `HumanMovePolicyAdvisor` checkpoint | No |
| Turning point detection | Per-ply scoring | No |
| `MoveAnnotation` / `PostGameAnnotation` dataclasses | — | No |
| `debrief_game()` extension | Dataclasses | No |
| LLM synthesis + prompt validation | Extended debrief | No |
| Validation (5–10 games) | All above + user review | No |
| UI | Validation + scheduling decision | No |

---

## Staged Implementation Plan

Each stage has a test gate; later stages must not begin until the prior gate passes.
The `r_h` values used below are **normalised** (0.0 = optimal, 1.0 = worst), where
`r_h = 1.0 − score_played` since `score_best = 1.0` by construction.

### Stage 1 — Data classes + clean scorer + heuristic loop *(implement now)*

**Goal:** replay a game record, score every ply with the clean heuristic, build
`PostGameAnnotation` with `heuristic_curve` and heuristic-only turning-point detection.

**New code:**
- `GameAI.assess_position(board)` — thin wrapper around `_score_all()` with no
  deadline; returns `[(move, raw_int_score), ...]` sorted best-first. Caps search
  depth via a configurable `max_assess_depth` (default 4) to bound wall-clock cost.
- `ai/post_game_assessor.py` — `MoveAnnotation`, `PostGameAnnotation`, `PostGameAssessor`.
  Sentinel/Malom/Trajectory/Policy fields default to `None`.

**Test gate:** `tests/test_post_game_assessor.py`
- `assessor = PostGameAssessor(difficulty=1, depth=3)`; 6-ply synthetic game record.
- `heuristic_curve` has one entry per ply.
- All `r_h` are ≥ 0 and ≤ 1.
- `turning_point_oracle == "heuristic"`.

### Stage 2 — Sentinel + GapNet blunder-zone integration *(complete)*

**Goal:** populate the Sentinel per-ply move-quality scores and the GapNet per-ply
position blunder-zone density.

**New code:**
- `PostGameAssessor.__init__` gains optional `sentinel` and `gap_net` args.
- Per-ply Sentinel call: `sentinel.advise(board, candidates, color, played_idx)`;
  populates `sentinel_played`, `sentinel_best`, `r_s`, `sentinel_score_white` (White-
  normalised: negate for Black plies); skips gracefully when `sentinel` is `None`.
- Per-ply GapNet call: `gap_net.predict(board, color)`; convert to [0,1] via
  `(raw+1)/2`; store as `blunder_zone_score`; skips when `gap_net` is `None`.

**Test gate:** mock both advisors; assert sentinel fields and `blunder_zone_score`
populated when present; assert all remain `None` when advisors absent; assert
`blunder_zone_score` is always in [0, 1] when populated.

### Stage 3 — Malom adjudication

**Goal:** look up `wdl_before` and `wdl_after` for every ply; classify
`quality` as `confirmed_poor` / `poor_candidate` / `clean`; flag `oracle_source`.

**New code:** `PostGameAssessor.__init__` gains optional `malom_db` arg. Per-ply: look
up position before and after the move; apply the signal-hierarchy classification;
populate `oracle_source`, `abstained_reason`, `quality`.

**Test gate:** use a known Malom-covered position (or a mock); assert `wdl_before`
and `wdl_after` are `"W"/"D"/"L"` or `None`; assert `quality == "confirmed_poor"` for
a verified `win_to_loss` transition.

### Stage 4 — Trajectory, Human policy, and Generalist AI signals

**Goal:** populate trajectory, human policy, and generalist AI policy fields per ply.

**New code:** `PostGameAssessor.__init__` gains `trajectory_db`, `policy_advisor`, and
`generalist_advisor` optional args. Per-ply trajectory query (field is `None` when `{}`
returned). Per-ply human policy probs call; `is_unconventional` left uncalibrated until
Stage 5 validation. Per-ply generalist policy probs; `generalist_self_assessed` flagged
when the generalist's `color` matches the ply's mover and the game was AI-played on
that side.

**Test gate:** mock all three advisors; assert trajectory fields populated at covered
positions; assert `traj_n` is `None` when coverage below threshold; assert policy and
generalist fields populated when advisors present; assert `generalist_self_assessed`
is set correctly.

### Stage 5 — Full turning-point hierarchy + poor-move thresholds

**Goal:** implement the full Malom → Sentinel+Heuristic → Heuristic turning-point
selection and calibrate `poor_candidate` thresholds from real game data.

**New code:** `_detect_turning_point` upgraded to the three-tier hierarchy (§ Score
Curves and Turning Point). `turning_point_oracle` field set correctly for each path.
`poor_candidate` flagging with calibrated `r_h` and `r_s` thresholds from validation.

**Test gate:** three tests — one Malom-covered game (oracle `"malom_full"`), one
Sentinel+Heuristic game (oracle `"sentinel+heuristic"`), one heuristic-only game
(oracle `"heuristic"`). Validate `turning_point_quality` format matches its oracle.

### Stage 6 — LLM synthesis

**Goal:** extend `debrief_game()` to accept a `PostGameAnnotation` and produce
the structured 3–5 sentence commentary.

**New code:** `MillsLLM.debrief_game()` extended to accept `PostGameAnnotation` as
well as the existing thin `DebriefReport`. Builds the four-section prompt (Game Facts,
Score Trend, Turning Point, Other Poor Moves) following the hard constraints in §LLM
Synthesis (no decimal Malom figures, no invented quality claims).

**Test gate (offline):** pass a `PostGameAnnotation` with known fields to a mocked
`_build_debrief_prompt()`; assert all four sections are present; assert no decimal
figures appear in the Turning Point section when oracle is `"malom_full"`.

### Stage 7 — UI integration

**Goal:** pivotal moves highlighted in the move-replay slider; LLM commentary
displayed in the MillsAI chat panel after the game ends.

**New code:**
- `web/app.py` and `/api/debrief` endpoint: on game end, run `PostGameAssessor.assess()`
  asynchronously; return `PostGameAnnotation` JSON alongside the existing `DebriefReport`.
- Frontend replay component: read `turning_point_ply` and `quality == "confirmed_poor"`
  or `"poor_candidate"` moves from the annotation; add highlight markers to the move
  timeline.
- MillsAI chat panel: after game end, display the LLM debrief text (from Stage 6) in
  the existing chat window beneath a "Game analysis" header.

**Test gate:** manual validation on 5–10 games (§Validation). Gate further polish on
that review.

---

## Open Questions

1. **`r_h` and `r_s` thresholds for `poor_candidate`.** What regret gap is meaningful
   for each signal? Calibrate from validation data; do not fix numbers before seeing
   actual distributions.

2. **Curve combination weights (`α`, `β`).** The `drop(t) = α × Δh + β × Δs` formula
   for Malom-absent turning-point detection requires calibration. Start with `α = β = 1`
   (equal weight) and adjust based on which signal better predicts the human-identified
   turning point during validation.

3. **Unconventional move threshold.** What `policy_prob` level marks a move as
   unconventional? Calibrate from validation data alongside the regret thresholds.

3. **Sentinel post-hoc candidate format.** Confirm that `board.get_legal_moves()` returns
   move dicts in the format `SentinelAdvisor.advise()` expects (same structure as the
   heuristic search output). If not, a thin adapter is needed.

4. **Trajectory coverage.** `TrajectoryDB.query()` returns `{}` when fewer than
   `min_samples=3` games cover a position. In practice this will be most positions.
   Decide whether a low-coverage trajectory signal is worth including in the prompt, or
   whether it should be silently omitted.

5. **LLM prompt validation.** Test the structured prompt against `llama3.1:8b` before
   wiring any UI. Confirm it respects discrete W/D/L attribution and does not invent
   continuous figures. Gate LLM synthesis on this test.

6. **Horizon effect detection (future, currently blocked).** Short/long node-budget
   comparison would add an explanation for *why* a move looked good shallowly. Blocked
   by: Rust backend does not support `node_limit`; PV is not returned by
   `py_search_root_scored()`. Defer until those infrastructure gaps are addressed. If
   pursued, it adds an explanatory layer on top of this plan — it does not replace it.

7. **`n > 50` empirical threshold.** Set independently from any training sweep — the
   assessor's source-selection threshold and the training class-balance threshold serve
   different purposes.
