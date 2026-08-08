# Sanmill Fixed-Resource Curriculum v1

## Status and authority

Status: `smoke_passed_managed_plan_required`

Parent experiment: `dev-v4-sanmill-refereed-fresh-v1`

On 8 August 2026, after exact-resume parity passed, the product owner
delegated the remaining technical launch decisions to Codex through entry into
the retained run. This document freezes the resource curriculum chosen under
that delegation. It authorizes implementation, focused verification, one
isolated five-game schedule smoke, evidence publication, and preparation of a
managed long-run plan. It does not allow a failed gate to be weakened or a
failed smoke to be reused.

## Decision

The first retained Sanmill lineage will use a deterministic, game-indexed
resource curriculum. It will not use outcome-dependent promotion and will not
reuse the local-GameAI `legacy-score` gate.

The fixed node ceilings and inclusive zero-based game ranges are:

| Level | Node ceiling | Global scheduled games | Games |
| ---: | ---: | ---: | ---: |
| 1 | 1,000 | 0-499 | 500 |
| 2 | 5,000 | 500-999 | 500 |
| 3 | 25,000 | 1,000-1,499 | 500 |
| 4 | 100,000 | 1,500-2,499 | 1,000 |
| 5 | 500,000 | 2,500-4,999 | 2,500 |

The schedule identity is therefore the ordered pair:

```text
node ceilings: 1,000,5,000,25,000,100,000,500,000
stage games:   500,500,500,1,000,2,500
```

The level for a game is derived solely from its global scheduled game index.
It is not inferred from a rolling score, checkpoint filename, segment number,
wall-clock time, or mutable counter. Exact resume at a stage boundary must
therefore choose the same level as an uninterrupted process.

## Interpretation

These levels are measured search-work ceilings, not certified Sanmill skill
levels. The completed node calibration showed stable fixed-work execution and
position-dependent depth separation, while the no-update integrated probe
covered all five ceilings through the production referee/opponent route. It
did not prove monotonic strength.

Moving to the next level means access to a more expensive learning
environment. It does not claim that the learner has beaten or mastered the
previous level. Training outcomes remain diagnostics; later frozen held-out
evaluation owns any strength claim.

The allocation deliberately reaches 500,000 nodes halfway through the
5,000-game game budget. This prevents a 12-active-hour stop from spending the
entire run in inexpensive introductory work, while reserving half the game
budget for the largest calibrated ceiling. No ceiling above 500,000 nodes is
used because none was calibrated.

## Transition semantics

A fixed-resource transition changes only the current Sanmill node ceiling and
the logged curriculum level. It must preserve:

- learner parameters and Adam state;
- frozen-target parameters and age;
- Python, NumPy, Torch, and configuration RNG state;
- temperature and the global annealing schedule;
- accumulated rollout/update state and training histories;
- SpecialistDB identity and contents; and
- exact-resume lineage and checkpoint continuity.

It must not invoke recovery, resurrection, rehearsal, learning-rate or
entropy boosts, temperature resets, score gates, checkpoint reloads, or
history clearing. Lower-level blending is disabled: each scheduled game uses
the one ceiling assigned to its global index when it reaches the Sanmill
opponent stratum.

## Required implementation evidence

Before a retained plan may be published, tests must prove:

1. strict parsing and validation of the node and duration values;
2. exact boundary selection at games 0, 499, 500, 999, 1,000, 1,499,
   1,500, 2,499, 2,500, and 4,999;
3. rejection of missing, non-positive, length-mismatched, or non-covering
   schedules;
4. identical next-level selection after exact resume at a boundary;
5. absence of score-based advancement and transition resets; and
6. continued rejection of the fixed-resource options on the local/GameAI
   route.

The one-run schedule smoke must use a separate empty
`sector-corrected-v1` SpecialistDB and output directory. It compresses the
five stages to one game each, sets the frozen-target ratio to zero so every
level invokes Sanmill search, and retains Sanmill as referee. It is route and
schedule evidence only, not learning or strength evidence.

## Retained-run envelope after a passing smoke

The managed plan may use only the already approved product envelope:

- random fresh initialization; seed 42; A2C; `batch_games=1`;
- at most 5,000 completed games or 43,200 active seconds, whichever occurs
  first;
- 250-game process segments with verified exact resume;
- `max_ply=120` and `sim_ply_depth=5`;
- frozen-target/Sanmill mix `0.60 / 0.40`;
- the five-level schedule frozen above;
- global temperature `0.90` linearly annealed to `0.20` by 80% progress;
- target refresh every 50 games and checkpoint/log cadence every 50 games;
- no Sentinel, ValueNet, GapNet, S1A warm-start, S1B refresher, imitation
  mixing, opening forcing, branch rollouts, recovery, or resurrection; and
- a new isolated output tree and new empty `sector-corrected-v1`
  SpecialistDB.

The supervisor must quarantine on non-finite updates, rule or mirror drift,
Sanmill protocol/search failure, checkpoint failure, data identity change,
resume mismatch, or evidence-chain failure. It must not stop early because of
training win rate, draw rate, or an apparent graph trend.
