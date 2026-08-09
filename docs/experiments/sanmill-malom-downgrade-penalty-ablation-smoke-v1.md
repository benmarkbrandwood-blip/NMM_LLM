# Sanmill Malom-Downgrade Penalty Ablation Smoke v1

## Status and decision

Status: `designed_unlaunched_needs_product_authorization`

Experiment family:
`dev-v4-sanmill-malom-downgrade-penalty-ablation-smoke-v1`

The first six-arm experiment compared an unconditional Mill bonus with a
Malom-preserving Mill bonus. Its immutable result was `inconclusive`: two of
three seed pairs favoured the corrected arm, but the median absolute reduction
in the preregistered Mill-only downgrade rate was only 0.396 percentage points,
below the frozen five-point gate. That result is not reinterpreted, pooled with
this experiment or extended after inspection.

This successor changes one learning factor. The control retains
`malom-preserving-only`. The treatment retains the same Mill rule and adds an
asymmetric penalty to every learner action that loses exact Malom WDL rank:

```text
exact quality  0:  0.00
exact quality -1: -0.25
exact quality -2: -0.50
```

There is no positive generic Malom reward. This avoids recreating the earlier
safe-draw incentive. The experiment asks whether this denser signal reduces
later exact-WDL downgrades consistently enough to justify a separately frozen
longer successor.

The complete machine-readable contract is
[`sanmill-malom-downgrade-penalty-ablation-smoke-v1.json`](sanmill-malom-downgrade-penalty-ablation-smoke-v1.json).

## Observed facts

- The earlier treatment changed reward on only 64 Mill-downgrade actions over
  its 1,500 corrected-arm games.
- Reconstructing the same logs found 2,137 learner actions with any exact WDL
  downgrade. This is a post-hoc exposure count, not a training result for the
  new penalty.
- The frozen held-out first-downgrade cohort contains 19 states: 16
  Mill-forming and three non-Mill actions, across placement, movement and
  flying and across Book, HumanDB and Perfect DB strata.
- A source-only calculation applies the new reward to all 19 states and changes
  the total immediate reward from 0.00 to -4.75. It loads no candidate, creates
  no optimizer, plays no game, changes no action and updates no weight.
- Seeds 45, 46 and 47 produce matched same-seed schedules. Across both arms
  they contain 1,230 Sanmill-opponent games, with both learner colours present
  in every pair.

The source-only calculation becomes admissible preparation evidence only after
it is generated once from a clean published `dev`. Readiness independently
reconstructs its summary from every state row and verifies its source cohort,
canonical identity, implementation bytes, Git lineage and claim boundary.

## Hypothesis

A fixed penalty on every selected action that loses one or two exact Malom WDL
ranks will reduce the tail all-action downgrade rate more consistently than
Mill-only bonus suppression, without violating the existing fixed policy-health
safety boundary.

## Supporting evidence

The new signal is mechanically denser and covers non-Mill mistakes. Its scale
is bounded: one-rank loss matches the historical `0.25` Mill bonus and two-rank
loss is twice that amount. Existing reward modes, resume configuration and
historical result remain unchanged. Focused rollout, rank-weighting,
fail-closed exact-quality, observability, result-accounting and readiness tests
cover the new route.

An optimistic independent-action approximation suggests that distinguishing an
eight-percent rate from six percent requires roughly 2,554 actions per arm.
The preceding runs had about 3,000 known learner actions in each 200-game tail,
so a two-percentage-point engineering gate is measurable at the planned scale.
This calculation ignores within-game and within-seed dependence and is not a
formal confidence claim.

## Counterevidence and limits

- The old treatment's effect was seed-sensitive and disappeared under some
  whole-run sensitivities.
- The 19-state probe proves reward wiring only; it cannot prove learning or
  causal policy improvement.
- Three seed pairs are sufficient for a bounded consistency check, not broad
  population inference.
- The 29-state policy-health corpus is inspected development data, not held-out
  validation. Ordinary supervised train/validation curves do not exist for
  this RL experiment.
- Training W/D/L, including results against Sanmill, is diagnostic only. It is
  neither the primary endpoint nor strength-promotion evidence.
- Penalising exact suboptimal actions may reduce exploration or merely move
  errors between phases. Phase, opponent, colour, termination and entropy
  reports are therefore mandatory.

## Frozen six-arm design

Seeds 45, 46 and 47 each have a fresh control and treatment arm. The launch
order alternates treatment position within seed blocks:

1. seed 45 control;
2. seed 45 penalty;
3. seed 46 penalty;
4. seed 46 control;
5. seed 47 control;
6. seed 47 penalty.

Every arm starts from random weights and a byte-identical closed, empty
`sector-corrected-v1` SpecialistDB. Same-seed arms share learner colour,
opponent source, retry-ply and Torch schedules. Only experiment identifiers,
isolated paths and `mill_bonus_mode` may differ.

Each arm runs one 500-game segment on the unchanged 5,000-game global schedule,
so it remains at the observed 1,000-node Sanmill level. Runs are sequential.
The complete envelope is at most 3,000 games, six active wall hours and
73,800,000 requested Sanmill nodes. No second segment or automatic extension is
permitted.

All other choices match the preceding controlled experiment: A2C, one game per
batch, learning rate `0.0001`, gamma `0.99`, entropy coefficient `0.01`, update
interval 64 steps, 60% frozen target and 40% Sanmill, 120 logical-ply cap,
depth-5 minimal rollouts, temperature `0.90` to `0.20`, no branches or recovery,
and no PPO, Sentinel, ValueNet, GapNet, imitation, S1B refresher or opening
forcing.

## Decision rule

The primary endpoint uses games 301 through 500:

```text
learner actions losing one or two exact Malom WDL ranks
-------------------------------------------------------
all learner actions with exact Malom support
```

Every arm must have at least 2,000 known tail actions. The result supports the
penalty only when all treatment arms pass safety, at least two of three pairs
favour treatment, no seed is harmed by more than two percentage points and the
median paired absolute reduction is at least two percentage points. Any other
finite, identity-valid result is `inconclusive`; the threshold cannot be lowered
and the run cannot be extended after results are visible.

Report every seed pair before summaries. Also report whole-run sensitivity,
one- and two-rank counts, placement/movement/flying rates, opponent source,
learner colour, termination reason, losses, entropy, temperature, selection
probability, reward components and the fixed policy-health diagnostic.

A supporting result permits only the design of a separately authorized longer
successor followed by newly independent held-out evaluation. An inconclusive
result ends reward-only escalation and redirects diagnosis toward the policy,
curriculum or auxiliary-loss architecture.

## Preparation and authority boundary

After the source is published, generate the immutable no-update probe once:

```powershell
.\.venv\Scripts\python.exe scripts\probe_malom_downgrade_penalty.py --write
```

Then audit source readiness and prepare six isolated plans without launching:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_mill_bonus_ablation.py `
  --contract docs\experiments\sanmill-malom-downgrade-penalty-ablation-smoke-v1.json `
  --report out\malom-downgrade-penalty-ablation-smoke-v1\readiness.json

.\.venv\Scripts\python.exe scripts\prepare_mill_bonus_ablation.py `
  --contract docs\experiments\sanmill-malom-downgrade-penalty-ablation-smoke-v1.json `
  --report out\malom-downgrade-penalty-ablation-smoke-v1\readiness.json `
  --prepare
```

Preparation must return `ready_for_product_authorization` with no arm output or
authorization already present. The source contract itself authorizes zero
segments, no publication and no promotion. A separate product authorization is
required before the six sequential arms may run.
