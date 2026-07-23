# Gen 2b Training Script — Plan

## 1. Motivation

The v2a training run has surfaced several structural issues that cannot be safely
patched into `scripts/train_s_gen_v2a.py` without changing the semantics of the
existing checkpoints and logs. `train_s_gen_v2b.py` is a clean-slate rewrite that
fixes those issues while remaining compatible with the same feature extractor,
model class, opponent scheduling, and learner architecture.

**Concerns being addressed (grouped):**

| # | Concern | Root cause in v2a |
|---|---|---|
| A | Advancement reload discontinuity | Loads `latest.pt` (or `best{prev}.pt`) at advancement; in-memory model diverges from disk between log ticks |
| B | Rehearsal chained advancement | Rehearsal expands lower-difficulty opponents, so post-advance scores are inflated by a temporarily easier mixture; a passing gate can chain into another advance |
| C | Termination classification collapses infrastructure failures into W/D/L | `is_terminal()` returns only material/no-legal-move; missing learner action becomes a loss, opponent exception becomes a win |
| D | `DRAW_LONG` conflates rules-based draws with `max_ply` truncation | 776/776 draws in one v2a log occur at exactly `ply=60`; the draw rate on plots is really "unresolved within 60 plies" |
| E | Temperature keeps cooling while stuck at one difficulty | `_compute_temperature` uses global `game_count / max_games`; a plateau at diff 1 cools to near-greedy without any advancement pressure |
| F | Checkpoint restart discards curriculum evidence | `latest.pt` preserves weights + partial recovery flags but not optimizer, RNG, advance history, `games_at_level`, `frozen_target_age`, rehearsal cooldown, or temp/entropy boosts |
| G | No visibility into how games end | Plots show only W/D/L; cannot tell truncation vs. rules-based draw, or material vs. blocking wins |
| H | No regression tests | Every fix requires re-running a full training experiment to validate |

Section 3 onwards walks through each fix.

---

## 2. Scope and naming

**In scope:**
- New file `scripts/train_s_gen_v2b.py`, based on v2a but with the specific
  behavioural changes listed below. Shared helpers stay in `learned_ai/training/`
  and can be imported by both scripts.
- New termination classification helpers in a shared module.
- New regression test file `tests/test_train_s_gen_v2b.py`.
- Plot script extension to render termination-reason breakdown alongside W/D/L.

**Out of scope:**
- Repetition detection and 50-move rule in `game/board.py`. These are proposed as
  a separate change (Section 4.4); v2b will *log* max-ply truncation as its own
  termination class so those rules can be added later without altering v2b's
  training signal contract.
- Changes to the scaffolded feature builder, model architecture, opponent
  scheduling, or A2C/PPO update logic.
- Changes to `train_s_gen_v2a.py` itself. v2a continues as-is for legacy runs.

**Naming:**

| Artifact | v2b path |
|---|---|
| Training script | `scripts/train_s_gen_v2b.py` |
| Default checkpoint dir | `learned_ai/checkpoints/scaffolded/s_gen_v2b/` |
| Regression tests | `tests/test_train_s_gen_v2b.py` |
| Termination helper | `learned_ai/training/termination.py` |
| Plot script | `tools/plot_specialist_training.py` (existing, extended) |

v2a and v2b keep separate checkpoint directories. There is no auto-migration of
v2a checkpoints into v2b.

---

## 3. Advancement mechanics

### 3.1 No reload on advancement (fix A)

At the moment of advancement, do not reload any checkpoint. The in-memory model
just achieved the advancement condition — those weights are the correct starting
point for the new level. `latest.pt` may be stale because it was written at the
last log tick.

**Rule:** at advancement, snapshot the current in-memory model to
`best{prev_diff}.pt` (milestone), then continue with the same in-memory weights.

The optimizer, frozen opponent, and win histories are still reset as they are
now — that behaviour is independent of the model-weight decision.

Regression test: `test_advance_preserves_in_memory_weights` — construct a fake
run, force an advancement condition, verify that immediately after the advance
the model's parameters are byte-identical to what they were the tick before.

### 3.2 Advancement cooldown during rehearsal (fix B)

When difficulty advances:
1. Set `advance_cooldown_batches = args.advance_rehearsal_batches + N`
   where `N` is a small trailing gate (e.g. 8 batches) so a fresh sample is
   accumulated before the next advance check.
2. Continue normal training and logging.
3. While `advance_cooldown_batches > 0`:
   - Decrement by 1 per outer batch.
   - Do **not** append game outcomes to `level_heuristic_history` for
     advancement purposes. (They still go into `win_history_heuristic` for
     display / recovery decisions.)
   - Do not call `_sanmill_check_advance`.
4. When `advance_cooldown_batches` reaches 0, clear `level_heuristic_history`
   completely so the next advancement decision is based only on fresh
   post-cooldown samples.

**Persistence.** `advance_cooldown_batches` is saved in the checkpoint and
restored on resume. A restart during the cooldown must not silently reset the
counter to zero.

Regression tests:
- `test_cooldown_blocks_advance` — populate `level_heuristic_history` with a
  passing sample, set `advance_cooldown_batches > 0`, verify no advance.
- `test_cooldown_clears_history_at_zero` — verify `level_heuristic_history` is
  empty at the exact tick the cooldown expires.
- `test_cooldown_persists_across_reload` — save a checkpoint with a nonzero
  cooldown, load it, verify the cooldown is unchanged.
- `test_cooldown_batch_size_invariant` — with `batch_games ∈ {1, 6, 12}`, the
  cooldown expires after the same number of *batches*, not games.

### 3.3 Advancement statistic interpretation (concern D numerical example)

The v2a run showed match score ≈ 0.51 when 80% of games are truncated draws and
55% of decisive games are wins. A 0.55 gate is unreachable in that regime. Two
independent decisions:

- **Termination classification** (Section 4) will separate truncated draws from
  rules-based draws, so the score fed into `check_advance` reflects games with
  a real outcome.
- **Cooldown + fresh sample** (3.2) ensures a subsequent advance is not driven
  by the transiently easier post-advance opponent mixture.

If, after those fixes, decisive-game outcomes still cannot reach the gate,
consider a curriculum-transition mode: an operator command that advances the
level with a note in the log ("operator-forced advance") rather than treating a
statistically impossible gate as evidence of readiness. Not implemented in v2b
by default; hooked for future use.

---

## 4. Termination classification (fixes C, D, G)

### 4.1 Termination reasons

`learned_ai/training/termination.py` defines an enum:

```python
class TerminationReason(str, Enum):
    WIN_FEWER_THAN_THREE   = "win_lt3"          # opponent reduced below 3
    WIN_NO_LEGAL_MOVE      = "win_blocked"      # opponent has no legal move
    LOSS_FEWER_THAN_THREE  = "loss_lt3"
    LOSS_NO_LEGAL_MOVE     = "loss_blocked"
    DRAW_REPETITION        = "draw_rep"         # rules-based (not yet detected)
    DRAW_50_MOVE           = "draw_50"          # rules-based (not yet detected)
    DRAW_MAX_PLY_TRUNCATED = "draw_trunc"       # rollout hit max_ply
    INFRA_LEARNER_FAILURE  = "infra_learner"    # policy/encoder raised or produced no action
    INFRA_OPPONENT_FAILURE = "infra_opponent"   # opponent raised or produced no action
```

Every rollout returns a `(outcome, TerminationReason)` pair.

### 4.2 Infrastructure failure handling

The current v2a code paths that can silently convert an exception or missing
action into W/D/L:

| Code path in v2a | Current bad behaviour | v2b handling |
|---|---|---|
| Learner's `select_action` returns `None` or raises | Treated as a game loss for the learner | Marked `INFRA_LEARNER_FAILURE`, excluded from W/D/L histories, excluded from advancement, logged as its own row |
| Opponent's `select_move` raises | Rollout may terminate with learner-win outcome even though position is non-terminal | Marked `INFRA_OPPONENT_FAILURE`, excluded from W/D/L, excluded from advancement, logged as its own row |

**Contract:** `INFRA_*` never enters `win_history`, `win_history_heuristic`, or
`level_heuristic_history`. A run with a persistent infra failure will show
those termination classes clearly in the plot rather than being masked as
apparent losses or wins.

Regression tests:
- `test_learner_missing_action_marked_infra` — mock the learner to return None,
  verify the rollout is classified `INFRA_LEARNER_FAILURE` and never lands in
  advancement history.
- `test_opponent_exception_marked_infra` — mock opponent to raise, same check.
- `test_infra_excluded_from_advance_stat` — put N infra rows into a rollout
  batch and verify `_sanmill_check_advance` sees the same window as a run
  without them.

### 4.3 Max-ply truncation as its own class

The current `DRAW_LONG` value is retained as a reward tag for the RL update but
the *termination reason* is separated. `DRAW_MAX_PLY_TRUNCATED` is:

- Counted in the rolling termination-mix plot (Section 5).
- Included in W/D/L histories with reward `DRAW_LONG` (same as v2a) so the RL
  update signal is unchanged.
- Reported per-log-tick with an absolute count.

An open decision (Section 8): whether truncated draws should be excluded from
`level_heuristic_history` entirely. v2b implements a `--truncation-in-advance`
flag defaulted to `include`, with `exclude` and `half_point` alternatives.

### 4.4 Repetition + 50-move rule (deferred)

Repetition detection and 50-move-no-capture are called out but not implemented
in v2b's initial cut. `TerminationReason.DRAW_REPETITION` and `DRAW_50_MOVE`
are reserved; when someone later wires them into `game/board.py`, the trainer
will already recognise them.

---

## 5. Temperature schedule (fix E)

Current `_compute_temperature(game_count, max_games, ...)` uses global game
progress, so time spent stuck at one difficulty keeps cooling the exploration.

**v2b behaviour:**

- Track a floor `TEMP_LEVEL_FLOOR = 0.65` (configurable via `--temp-floor`).
  The scheduled temperature never drops below this floor while
  `advance_cooldown_batches > 0` or while `games_at_level > 500` without an
  advance.
- On advancement: reset the cooling clock. New level begins at `temp_start`
  and cools with `games_at_level`, not global `game_count`.
- Existing `temp_boost` / `entropy_boost` are unchanged.

Regression tests:
- `test_temp_floor_holds_during_cooldown` — set cooldown > 0, run 500 iters,
  verify temperature never dropped below the floor.
- `test_temp_resets_on_advance` — trigger advance, verify next batch uses
  the level-start temperature not the pre-advance temperature.
- `test_temp_persistence` — save checkpoint, load, verify same effective
  temperature.

---

## 6. Checkpoint state persistence (fix F)

The v2a `latest.pt` stores model weights, some recovery fields, and metadata.
It does **not** store enough to make a resume trajectory-equivalent to a run
that continues without restart.

**v2b checkpoint schema** — every field below is required on load:

```python
ckpt = {
    # Model
    "model": model.state_dict(),
    "model_config": model.get_config(),
    "stage": STAGE_TAG,

    # Optimizer
    "optimizer": opt.state_dict(),

    # Curriculum
    "game_count": game_count,
    "difficulty": difficulty,
    "games_at_level": games_at_level,
    "advance_cooldown_batches": advance_cooldown_batches,
    "advance_rehearsal_remaining": advance_rehearsal_remaining,
    "level_heuristic_history": list(level_heuristic_history),
    "win_history": list(win_history),
    "win_history_heuristic": list(win_history_heuristic),

    # Frozen opponent
    "frozen_target_age": games_since_target_update,
    # frozen_opp weights are not persisted; refresh() is called on resume.

    # Exploration / recovery
    "temperature":               temperature,
    "temp_boost":                temp_boost,
    "entropy_boost":              entropy_boost,
    "hot_explore_remaining":     hot_explore_remaining,
    "hot_explore_triggered":     hot_explore_triggered,
    "recovery_grace":            recovery_grace,
    "recovery_state":            _current_recovery_state,
    "recovery_baseline_win_rate": recovery_baseline_win_rate,

    # RNG
    "rng_python": random.getstate(),
    "rng_numpy":  np.random.get_state(),
    "rng_torch":  torch.get_rng_state(),

    # Termination stats
    "termination_history": list(termination_history),
}
```

Regression tests:
- `test_checkpoint_round_trip_all_fields` — for each field above, verify save →
  load restores the same value.
- `test_resume_deterministic_step` — with fixed RNG state, run 1 batch,
  checkpoint, load, run 1 batch, verify the second batch's game outcomes are
  identical to a control run that never restarted.
- `test_missing_field_uses_safe_default` — load a checkpoint missing one of the
  new fields (v2b loading a hypothetical partial file); verify it warns and
  uses documented defaults rather than crashing.

Frozen opponent weights are deliberately excluded — refreshing to the current
model on resume is close enough for training and avoids doubling the file size.

---

## 7. Termination stats + plot (concern G)

### 7.1 Trainer side

- Maintain a `termination_history: deque[tuple[game_count, TerminationReason]]`
  with `maxlen=200`.
- Append on every rollout that produces a valid termination (including
  infra failures for observability).
- At each log tick, compute rolling percentages over the last 50
  *non-infra* terminations and write them into the diag row.

Fields added to `GameDiag`:

```python
term_win_lt3_pct:   float
term_win_blocked_pct: float
term_loss_lt3_pct:  float
term_loss_blocked_pct: float
term_draw_rep_pct:  float
term_draw_50_pct:   float
term_draw_trunc_pct: float
term_infra_learner_count: int  # absolute, not %; expected to be zero
term_infra_opponent_count: int # absolute, not zero-expected
```

### 7.2 Plot side

Extend `tools/plot_specialist_training.py` with a new row (Row 6) after the
retro reward row:

- Stacked-area plot of the seven percentage terms over training games.
- Colours: green for wins, red for losses, grey for draws (subshaded by
  reason).
- Two absolute-count series (`infra_*`) drawn on the same panel with a
  distinct marker style; a nonzero infra count for more than 50 games in a row
  is annotated in red so infrastructure problems are visible at a glance.

Regression test `test_termination_diag_populated` — build a synthetic
`GameDiag` from a scripted rollout set and verify the percentages sum to 100
(within float epsilon).

---

## 8. Regression test suite

Consolidated list. Every test is unit-scale; none requires an actual training
run.

| Test | Verifies |
|---|---|
| test_advance_preserves_in_memory_weights | § 3.1 — no reload on advance |
| test_cooldown_blocks_advance | § 3.2 |
| test_cooldown_clears_history_at_zero | § 3.2 |
| test_cooldown_persists_across_reload | § 3.2 |
| test_cooldown_batch_size_invariant | § 3.2 |
| test_learner_missing_action_marked_infra | § 4.2 |
| test_opponent_exception_marked_infra | § 4.2 |
| test_infra_excluded_from_advance_stat | § 4.2 |
| test_termination_reason_all_paths | § 4.1 — enum coverage on each `is_terminal` branch |
| test_max_ply_marks_truncation | § 4.3 — rollout hitting max_ply gets `DRAW_MAX_PLY_TRUNCATED` |
| test_temp_floor_holds_during_cooldown | § 5 |
| test_temp_resets_on_advance | § 5 |
| test_temp_persistence | § 5 |
| test_checkpoint_round_trip_all_fields | § 6 |
| test_resume_deterministic_step | § 6 |
| test_missing_field_uses_safe_default | § 6 |
| test_termination_diag_populated | § 7 |
| test_termination_percentages_exclude_infra | § 7 — infra never counted in %s |

**CI hook:** these tests live in `tests/test_train_s_gen_v2b.py` and are
picked up by the existing pytest suite. No changes needed to CI config.

---

## 9. Depth logging (concern F, informational)

`GameAI.difficulty` is clamped to [1, 10] internally and `max_search_depth`
caps at 19. The depth actually reached in a search depends on time budget,
hardware, and position. v2b will log per-rollout `max_depth_reached` (already
computed by the search) into `GameDiag.max_depth_reached_mean`. The plot
overlay is optional; the persisted stat lets us audit whether higher
difficulty is actually producing deeper play.

No regression test — this is an observational field.

---

## 10. Migration path

1. Land `learned_ai/training/termination.py` and regression tests first, with
   both v2a and v2b importing it. Termination classification is the largest
   correctness fix and belongs in a shared module.
2. Land `scripts/train_s_gen_v2b.py` as a copy-modify of v2a with sections 3–7
   applied. Keep v2a untouched to preserve legacy runs.
3. Land the plot script row extension.
4. Regression tests run green before any training run is started with v2b.
5. First v2b training run uses the same command-line profile as the current
   v2a run except for `--out-dir learned_ai/checkpoints/scaffolded/s_gen_v2b`
   and any new v2b-specific flags (`--advance-rehearsal-batches`,
   `--temp-floor`, `--truncation-in-advance`).

There is no auto-migration of v2a checkpoints. A fresh v2b run starts from
either scratch or an explicit `--resume` pointing at a v2a checkpoint (with
the v2b loader tolerating missing v2b-only fields via safe defaults).

---

## 11. Open decisions before implementation

The plan cannot fix without user direction:

1. **Truncated-draw treatment.** Include, exclude, or half-point in advancement
   statistics? Default proposed: `include` (matches v2a behaviour). Alternative
   flags implemented so it can be changed without further code changes.
2. **Repetition + 50-move-rule detection.** Deferred to a separate change on
   `game/board.py`. v2b logs `DRAW_MAX_PLY_TRUNCATED` cleanly so this can be
   added without altering v2b's semantics.
3. **Curriculum-transition operator command.** Should there be a `SIGUSR1`
   or file-based signal that forces an advance regardless of the statistic?
   Not proposed for the first v2b cut.
4. **Temperature floor value.** Proposed 0.65 as a middle-ground; user may
   want a stricter or looser default.
5. **Rehearsal cooldown length.** Proposed `advance_rehearsal_batches + 8`.
   User may want a different tail.

Each of these is captured in the plan; decisions during implementation should
land as commit-visible edits to this document.

---

## 12. Deliverables checklist

- [ ] `learned_ai/training/termination.py`
- [ ] `scripts/train_s_gen_v2b.py`
- [ ] `tools/plot_specialist_training.py` — termination row
- [ ] `tests/test_train_s_gen_v2b.py`
- [ ] `docs/gen_2b_plan.md` (this file)
- [ ] Sign-off on open decisions in Section 11
