# Discussion Plan — Gen v2b, Sentinel v2, ValueNet v2, HumanPrefNet, GapNet v2

Distilled from `Sanmill discussion 26-07-2026.odt`.  Every entry pairs a
concrete concern with a specific proposed change.  Nothing here is scheduled
yet — this doc is the shared queue we work off before writing code.

## Priority ordering across the whole retrain cycle

Ordering suggested in the discussion (roughly cheapest → most invasive):

1. **PPO temperature consistency** — see Gen 2b §T.
2. **Rules-based draw tracking and truncation treatment** — Gen 2b §D.
3. **Complete exact-resume state** — Gen 2b §R.
4. **Separation of curriculum and strength statistics** — Gen 2b §A.
5. **Win-only learning-rate rule** — Gen 2b §L.
6. **Clearer opponent-type logging** — Gen 2b §O.
7. **Sentinel/ValueNet/HumanPref split fixes** — one canonical position- (or
   game-) level manifest reused everywhere; see the per-artifact sections.
8. **GapNet v2 target semantics + rebuild** — before any GapNet promotion.
9. **Sanmill referee prototype** — larger, run alongside the above.

---

## Gen v2b trainer

### §A. Advancement statistics polluted by soft-transition mixture
- **Concern**: advancement window includes rehearsal / blended-opponent
  outcomes, so the score that gates the level jump is not a pure signal.
- **Proposed**:
  - Pause advancement checks during the rehearsal / soft-transition window,
    OR keep a **separate advancement history** that only records **primary
    games against the current-difficulty reference opponent**.
  - Log per-opponent strata (retries, confirmation, lower/higher diff,
    blunder, blended, frozen) as separate curriculum statistics so which
    stratum drives an advance is visible.
  - Perform advancement only after the complete parallel batch has been
    processed; start the new difficulty on the next clean batch boundary.
  - Regression test: results pre-generated before an advance cannot be
    attributed to the new difficulty.

### §T. Temperature schedule and boost
- **Concerns**:
  1. Temperature schedule is *global*: cools even when the model is stuck at
     one difficulty, killing exploration on plateaus.
  2. Temperature boost at advancement decays *per outer batch*; batch size
     silently changes curriculum timing, and the boost is not persisted
     across restarts.
- **Proposed**:
  - Add a **persisted minimum-temperature floor**, a **per-level
    temperature schedule**, or explicitly pause annealing during a
    curriculum plateau.
  - Express boost durations in **primary games**, not batches.
  - Log scheduled temperature, boost, and effective temperature as three
    separate fields.
  - Persist temperature boost + cooldown in `latest.pt` so restarts do not
    erase them.

### §D. Draws vs truncations
- **Concerns**:
  - Trainer conflates max-ply truncation with a real draw.
  - Rules-based draws (repetition, 50-move / 100-ply no-capture) are not
    implemented in Python.
  - Max-ply limit (100 or 120) is *not* equivalent to the 100-ply
    no-capture rule.
- **Proposed**:
  - Log **separate termination reasons**: win by material, win by blocking,
    repetition draw, 50-move draw, max-ply truncation.
  - Distinguish total-ply truncation from the rules-based no-capture
    counter and repetition draws.
  - Decide later how each termination class contributes to advancement /
    scoring — logging first, semantics after.
  - Longer-term: use **Sanmill** as the authoritative rule engine to own
    counters, repetition state, terminal detection, and outcome reasons
    (see §S).

### §R. Exact-resume state
- **Concern**: restart of a training run does not restore full state.
  Adam optimizer, Python/NumPy/PyTorch RNGs, advancement histories,
  games-at-level counters, frozen-target age, rehearsal cooldown,
  temperature state, entropy boost, etc. all reset on resume.
- **Proposed**:
  - Extend `latest.pt` (or a companion sidecar) to persist optimizer state,
    all RNG states, every curriculum counter, cooldowns, temperature state,
    entropy boost, and any other history field a decision path reads.
  - Regression test proves exact-resume: run N batches, checkpoint, restart,
    run N more, and verify the trajectory matches an unbroken run.
  - This is required to run multi-week trainings on a laptop that must
    survive suspend/resume.

### §L. Learning-rate rule
- **Concern**: current LR-adaptation is "win-only" — LR is reduced when win
  rate drops but never re-raised.  Locks the run into low-LR mode after any
  transient dip.
- **Proposed**: replace with a symmetric adjustment (both directions), or
  scale by a smoothed win-rate delta rather than a threshold.

### §O. Opponent-type logging
- **Concern**: current logs mash together very different opponent classes.
- **Proposed**: emit one field per rollout naming the opponent class
  explicitly (`vs_heuristic_current`, `vs_heuristic_easy`,
  `vs_heuristic_hard`, `vs_heuristic_blunder`, `vs_heuristic_blend`,
  `vs_frozen`, `vs_humanlike`, …).  Downstream plots and stats can then
  strat-split cleanly.

### §I. Infrastructure vs game outcomes
- **Concern**: learner-encoder / policy exceptions and opponent-engine
  exceptions can currently land as apparent Ws or Ls.  These contaminate
  RL rewards **and** the SpecialistDB W/L records.
- **Proposed**:
  - Classify termination reasons explicitly: `infra_learner`,
    `infra_opponent`, `truncation`, and each *valid* game outcome.
  - **Never** feed infra failures into RL reward, W/L history, or
    advancement statistics.
  - Regression tests for each path.

### §Δ. Difficulty realisation
- **Concern**: at levels above 10 the difficulty index is capped internally
  and only *search time* varies.  We do not log realised search depth /
  nodes, so "diff 12" vs "diff 15" is opaque.
- **Proposed**: log actual per-move budget, depth reached, and node count
  as GameDiag fields.  Use those to (a) understand the diff-10 plateau,
  and (b) build a more accurate curriculum.

---

## Sanmill authoritative referee (§S)

Larger, cross-cutting.  Called out because several §D / §I / §Δ items
converge here.

- **Concern**: Python trainer re-implements some Mill rules.  Risk of
  divergence from Sanmill's authoritative rules and history.
- **Proposed prototype**: persistent Sanmill process that owns
  - legal-move generation
  - rule history
  - repetition + no-capture counters
  - terminal detection
  - outcome-reason attribution

  Trainer talks to it via a tested bridge translating our move dicts to
  Sanmill tokens.  Result: rules stay consistent between training and any
  evaluation that reuses Sanmill.

- **Scope caveat**: this is a substantial engineering effort; treat as a
  parallel track rather than a prerequisite for the fixes above.

---

## Sentinel v2

### §S1. Split leakage
- **Concern**: train/validation split is done at move-example level, so
  positions and even successive plies from the same game appear on both
  sides.  The "held-out" JSONL source is not truly disjoint.
- **Proposed**:
  - Split at the **position** (or **whole-game**) level *before* expanding
    to candidate moves.
  - Freeze the split as a manifest file (e.g., `split_v2.json` listing game
    IDs or state-key hashes) and use the *same* manifest at both training
    and evaluation time.
  - Persist dataset composition + provenance alongside the `.npz` so the
    exact recipe is reproducible.

### §S2. `RUN_STAGE1=1` does not reproduce the original heuristic Stage 1
- **Concern**: the wrapper's `RUN_STAGE1=1` path reuses the Malom-labelled
  combined `.npz`, which is a *different* training regime than the pure
  heuristic-labelled JSONL replay the rollback doc describes.
- **Proposed**:
  - Treat **Recipe A** (the explicit `train_sentinel.py` command in
    `docs/sentinel_stage1_rollback.md`) as the canonical heuristic Stage 1.
  - Remove the wrapper-equivalence claim.
  - Keep rollback outputs in clearly experimental directories; do not
    overwrite production `best.pt` from them.

### §S3. Stage-1 anchoring is a hypothesis
- **Concern**: "Stage 1 anchors the trunk, Stage 2 can't move" is a plausible
  reading of the plateau but not yet proven.
- **Proposed**: clean A/B experiment
  - Arm A: heuristic warm-start (Stage 1) → Stage 2
  - Arm B: direct Malom training from random init
  - Equal compute budgets, properly held-out evaluation (see §S1).
- **Success**: whichever arm produces the better held-out score wins.  If
  they're within noise, note that Stage 1's plateau claim was overstated.

### §S4. Benchmark configuration is inconclusive
- **Concern**: v1 vs v2 sentinel game-bench differences are small over the
  limited number of games run.  Promotion criteria are not tight enough.
- **Proposed**:
  - Treat current bench results as **observational**, not promotion-grade.
  - Refine bench design: more games, multiple seeds, and a written
    promotion threshold (e.g., "beats baseline by ≥ Xpp across seeds, with
    z-test").
  - Only promote after evidence is reproducible.

---

## ValueNet v2

### §V1. Evaluator bug — same random init reported for v1 and v2
- **Concern**: `eval_value_net_v2.py` instantiates a fresh `ValueNet` and
  calls `ValueNet.load(...)` without assigning the return value.  Both v1
  and v2 metrics are actually measuring the same random initial model.
- **Proposed**:
  - Fix loader to `net = ValueNet.load(str(net_path), ...)`.
  - Regression test with two obviously different checkpoints (e.g., all-zero
    weights vs all-ones) to prove the loader is being respected.
  - Re-run v1 vs v2 evaluation.
  - Verify existing checkpoints on disk are untouched via SHA-256 before
    rerunning.

### §V2. Training / evaluation splits are inconsistent
- **Concern**: even after the loader bug is fixed, the split rule inside
  `ValueNet.train` differs from the split rule inside `eval_value_net_v2.py`
  — so "held-out MSE" can still reuse training positions.
- **Proposed**: define **one** frozen manifest (position-level, matching
  §S1) and have both `ValueNet.train` and `eval_value_net_v2.py` obey it.

---

## HumanPrefNet

### §H1. Pair construction ignores frequency and leaks positions
- **Concerns**:
  - Pairs are built from every recorded move without using the
    `total_frequency` field — a move played once and a move played 1000
    times count the same.
  - Pair-level train/eval splitting lets the same position appear in both
    sides, and can produce contradictory `AB` / `BA` pairs.
- **Proposed**:
  - Split by **state** or **game** *before* pair expansion.
  - Incorporate move frequencies into the loss — either as a per-pair
    weight or as a **weighted listwise categorical** target over the legal
    moves at a position (which is closer to what human-choice modelling
    actually wants).
  - Use a state- or trajectory-based held-out slice; align with §S1's
    manifest.

### §H2. Spearman below planned threshold
- **Concern**: current HumanPrefNet Spearman r is much lower than the
  plan's threshold, and the high-Elo / prune-bench checks are still not run.
- **Proposed**: treat current metrics as preliminary.  Add specific
  diagnostics:
  - Per-position ranking accuracy on multi-move positions.
  - Elo-strata top-1 accuracy on the highest-Elo subset only.
  - Prune-bench: AI with HumanPrefNet as a move-filter (drop bottom 10%
    ranked moves) vs unpruned AI.
- Once the diagnostic split is right (§H1), decide whether to adjust
  architecture, loss, or data weighting.

### §H3. Evaluation slice not truly held out
- Same fix as §S1 / §V2 — one shared manifest, applied before pair
  expansion.

### §H4. Integration test attribution
- **Concern**: the encouraging integration-test game does not isolate
  HumanPrefNet's contribution — the stack includes ValueNet, GapNet,
  Sentinel, DBs, and the Star Square filter, and many moves match Malom
  regardless.
- **Proposed**: preserve the game as observational evidence; add controlled
  ablations later
  - Pure HumanPrefNet (VN / GapNet / Sentinel off)
  - HumanPref sampling vs argmax
  - Full stack
  - All with fixed seeds + positions.
- Log the move source per ply (Malom / HumanPref / VN / Sentinel / variance)
  so the ablation output is diagnosable.

---

## GapNet v2

### §G1. Dataset does not encode the target we claimed
- **Concern**: `docs/retrain_v2_plan.md` says the v2 GapNet dataset encodes
  "HumanPref vs Malom disagreement".  Currently it uses a **Sentinel +
  heuristic composite** as the score, does not query per-move Malom over
  all legal candidates, and its "blunder" condition does not require a
  *better* alternative to exist.
- **Proposed** (either / or):
  - Document the composite-score interpretation explicitly, and drop the
    "HumanPref vs Malom disagreement" claim; OR
  - Rebuild the dataset by querying **all legal candidate moves through
    corrected Malom**, explicitly requiring an available *better* move,
    and defining `y = malom_quality(best_available) − malom_quality(played)`.
- **Gate**: pause any GapNet v2 promotion until this is decided and
  implemented.

### §G2. Observational bench evidence to preserve
Recorded 26-07-2026, worth noting in the tracker:

```
scripts/bench_sentinel.py --games 500 --difficulty 2 --time-budget 3.0 \
  --white-gap-net --gap-net-path data/gap_net.npz \
  --black-humanlike --humanlike-blend 100 \
  --human-pref-path data/human_pref_net.npz

500/500  A:185  B:59  D:256
Config A: White[d2+gap_net]           37.0 %
Config B: Black[d2+humanlike100%]     11.8 %
Draws                                 51.2 %
A edge +25.2 pp
```

This is what GapNet-vs-HumanPref-proxy looks like today.  Preserve for
comparison after any GapNet dataset rebuild.

---

## Move-source, filters, board representation

### §M1. Star Square filter semantics unclear
- **Concern**: it is not clear whether Star Square is a hard restriction
  or a soft preference, and Malom's fast path can bypass the filter in
  some positions.
- **Proposed**:
  - Decide explicitly: hard restriction or soft preference?
  - Make Malom / other fast paths respect the intended filter.
  - Log the **final move source** per move (Malom, HumanPref, ValueNet,
    Sentinel, variance, book) *and* whether the Star Square filter fired.

### §M2. Board index ordering differs from Sanmill
- **Concern**: NMMLLM uses outer-to-inner index numbering; Sanmill uses
  inner-to-outer.  Some features treat indices as *numeric* rather than
  *topological*, which may hurt learning.
- **Proposed**:
  - Keep current ordering now for compatibility.
  - Plan a **separate experiment branch** with a topology-aware
    representation (adjacency, mill-line membership, symmetry) rather than
    raw index values.  Fresh checkpoints, not a mid-stream swap.

---

## What to defer

- **Sanmill authoritative referee (§S)** — treat as a parallel prototype,
  not a blocker for the higher-priority fixes.
- **Board-representation experiment (§M2)** — a fresh branch, after the
  correctness fixes above land.
- **Round-robin population training** — mentioned as a possible path around
  the difficulty-10 plateau, worth revisiting only after clean §S / §V /
  §H / §G v2 artifacts exist.

## Observational only — do not act on yet

- Current v2b `s_gen_v2b/...` disposable training run.  Keep as evidence,
  do not modify semantics mid-run.
- Current v1-vs-v2 sentinel game-bench numbers (see §S4).
- The `gap_net vs humanlike100%` 500-game bench above (§G2).

---

## Working-order summary (one-liner per line)

```
[Gen 2b]        §T  temperature-schedule consistency (floor + per-level + no batch decay)
[Gen 2b]        §D  termination-reason logging (win_lt3, win_blocked, draw_rep, draw_50, draw_trunc)
[Gen 2b]        §R  exact-resume state (optimizer, RNGs, curriculum counters, cooldowns)
[Gen 2b]        §A  curriculum vs strength stats separation + per-opponent strata
[Gen 2b]        §L  symmetric LR rule
[Gen 2b]        §O  opponent-type logging
[Split]         §S1 §V2 §H1 §H3  one shared position/game-level manifest
[ValueNet]      §V1 fix `net = ValueNet.load(...)` + regression test
[GapNet]        §G1 decide target semantics; pause promotion until fixed
[Sentinel]      §S2 canonical Recipe-A rollback path; drop wrapper-equivalence claim
[Sentinel]      §S3 A/B: Stage-1 warm-start vs direct Malom
[Sentinel]      §S4 tighter benchmark + promotion thresholds
[Star Square]   §M1 hard-vs-soft decision + per-move source log
[Prototype]     §S  Sanmill referee (parallel track)
[Deferred]      §M2 topology-aware board rep
```
