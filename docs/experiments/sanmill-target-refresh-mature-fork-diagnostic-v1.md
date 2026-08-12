# Mature target-refresh fork diagnostic v1

Status: `designed_unlaunched_needs_publication`

Plan identity:
`7a0bd214c353d67bf52d3fb5c8d8c2184f4e6c647d49910a117539415cb2c0c0`

The authoritative machine-readable contract is
[`sanmill-target-refresh-mature-fork-diagnostic-v1.json`](sanmill-target-refresh-mature-fork-diagnostic-v1.json).
This document explains its scientific boundary. Neither file authorizes a
launch.

## Objective

Determine whether a frozen-opponent refresh becomes useful after the learner
has consumed 8,192 transitions beyond the original game-50 fork. The
treatment refreshes the frozen opponent exactly once at that mature boundary.
The paired control retains the same stale game-50 opponent. Both arms then
receive exactly the same further transition budget, temperature schedule,
learning rate, fixed-node Sanmill work, data-reading mode and measurement
work.

This is a mechanism diagnostic. It is not held-out strength evidence and
cannot promote a model, publish a model, prove that repeated refreshes are
safe, or start retained or long training.

## Observed facts

The completed attempt-003 direct cross-play compared the game-50
`refresh-once` and `no-refresh` policies without training or database writes.
Across 288 games, no-refresh scored `178 W / 12 D / 98 L`; the paired mean
score effect was `+0.2777778` in the no-refresh direction. All three seeds and
all three phases supported that direction. This establishes that the early
refresh was harmful under the tested schedule, not that a frozen opponent
should remain stale forever.

The schedule-isolation experiment also produced one exact no-refresh
checkpoint per seed at 8,192 consumed post-fork transitions. Their current
facts are:

| Seed | Game | Updates | Total consumed | Pending | Target age | DB positions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 67 | 439 | 146 | 9,344 | 20 | 439 | 16,849 |
| 68 | 327 | 150 | 9,600 | 40 | 327 | 17,031 |
| 69 | 518 | 143 | 9,152 | 17 | 518 | 15,821 |

All three checkpoints have role `transition_diagnostic_candidate`, preserve a
stale target that differs from the mature policy, and bind separate closed
`sector-corrected-v1` SpecialistDB files. Their exact file, payload,
configuration and database identities are in the JSON contract.

## Hypothesis

The game-50 refresh may have been harmful because it replaced a simple stable
opponent before the learner had formed a useful policy. A later one-time
refresh may expose the learner to a stronger, more relevant opponent and
improve subsequent learning. This is falsifiable: the mature-refresh policy
must beat the stale-control policy under a fixed paired development measure,
not merely differ in parameters or training curves.

## Supporting evidence

- The early direct effect is large, consistent across three seeds and visible
  in placement, movement and flying starts.
- Policy separation appeared only late in the preceding equal-transition
  diagnostic, so a later boundary is scientifically distinct from game 50.
- The selected source checkpoints all exist at the same exact consumed
  transition boundary and retain complete Adam, RNG, model, target, data and
  trainer state.

## Counterevidence and confounders

- The preceding fixed-anchor Sanmill measurement was severely floor-limited;
  a zero win rate cannot distinguish two weak policies reliably.
- A single mature refresh cannot prove that periodic refresh at the same
  interval is safe or optimal.
- The source checkpoints have different game and update counts even though
  post-fork consumed transitions match. The new experiment therefore fixes
  only subsequent exposure and interprets results as conditional on those
  exact source states.
- The three source checkpoints store slightly different game-edge temperature
  values. The common fork explicitly normalizes the schedule scalar to
  `0.8379808850090307`, the prior transition-indexed schedule value at exactly
  8,192 transitions.
- Each source contains 17–40 pending pre-boundary samples. Carrying those
  samples across the treatment would contaminate the causal boundary. They
  are discarded identically before either branch is created and are reported
  explicitly.

## Common-fork integrity

For each seed, preparation creates one shared untreated mature fork. Only the
following neutral state is allowed to change:

1. clear the pending transition queue;
2. reset the target-refresh state to a captured but untreated mature fork;
3. set the schedule temperature to the exact mature-boundary origin.

The learner model, frozen target, Adam state, RNG state, data cursor,
curriculum, counters and every non-allowlisted recovery field must remain
equal. The source checkpoint is never overwritten. The two arm checkpoints
are descriptor-only branches with an identical payload. The corresponding
SpecialistDB files are byte-identical per-seed clones before either arm runs.

## Frozen training comparison

Each seed has two conditions in this exact order:

1. `refresh-mature`: copy the mature learner into the frozen opponent once;
2. `stale-control`: retain the old game-50 frozen opponent.

After treatment, no further target refresh is allowed. Each arm uses:

- A2C, batch size one and exact 64-transition update batches;
- 8,192 consumed transitions after the mature fork;
- fixed learning rate `0.0001`;
- transition-indexed temperature from `0.8379808850090307` toward `0.20`
  across the remaining 98,112-transition horizon;
- 60% frozen-opponent and 40% Sanmill games;
- fixed Sanmill work of 1,000 nodes per search;
- 120 logical plies per training game;
- theoretical-only SpecialistDB reads;
- `malom-preserving-only` Mill reward;
- no policy auxiliary, downgrade penalty, imitation, PPO, recovery,
  Sentinel, ValueNet or GapNet.

The training order is seed 67 pair, seed 68 pair, then seed 69 pair. Only one
process may be active at a time.

## Frozen development measurements

The result must inspect raw and complete-window training curves, all three
seeds, the fixed hyperparameters and database identities, and metrics split by
phase, opponent, colour and termination reason. Training W/D/L is context,
not the selection statistic.

At 4,096 and 8,192 post-mature transitions, the two policies are compared on
the same frozen 64-state placement/movement/flying corpus. At temperatures
`1.0` and `0.2`, the report includes full-action top-1 agreement,
Jensen–Shannon divergence, total variation, Malom-preserving probability mass
and rank movement.

At the final boundary, the policies also play 144 colour-swapped pairs, 288
CPU no-update games, over twelve audited phase-covered histories and four
replicates. The same-seed mature common-fork policy supplies shared lookahead
features. Sanmill is only the strict portable referee; it does not select a
move. Max-ply truncation is reported separately from rules draws.

The direct contrast is `refresh-mature minus stale-control`. A material result
requires an aggregate paired score effect of at least `0.0833333`, the same
minimum effect in at least two seeds, no opposite seed worse than `0.0416667`,
and truncation no higher than 25%. Policy divergence must be persistent from
4,096 through 8,192 transitions in at least two seeds when it is used as
supporting mechanism evidence. No outcome automatically selects a long run.

## Resources and stop conditions

The complete ceiling is 3,600 training games, 49,152 consumed transitions,
four active hours and 288 later no-update games. Each arm is capped at 600
games and 0.6 active hours. The requested Sanmill ceiling is bounded by
172,800,000 nodes across all training games.

Any source, checkpoint, database, plan, runtime, rules, corpus or
implementation drift stops the whole sequence. The same applies to a pair
difference outside the declared treatment, non-finite state, label or sidecar
drift, Sanmill disagreement, policy-health failure, resource exhaustion, or
an incomplete measurement ledger. Failure consumes the one-shot sequence;
there is no automatic retry, resume, extension, held-out evaluation,
promotion, publication or long-training fallback.

## Next gate

After the contract and implementation are ordinarily pushed, the preparer may
create the three common forks, six database clones, six managed plans and six
technical preflights. It must not create authorization files or launch a
trainer. A clean parent readiness identity and one explicit product
authorization for the complete aggregate envelope are required before the
six-arm sequence can run once.
