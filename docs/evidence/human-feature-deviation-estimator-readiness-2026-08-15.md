# Human feature deviation estimator readiness

Date: 15 August 2026

Status: `B_not_ready_fail_closed`

## Outcome first

The conditional-choice implementation is numerically healthy, but the frozen
research-confirmation execution gate is not met.  Research confirmation
remains unopened.

The only failed readiness item is the preregistered power requirement for a
true `0.01`-nat average-unique-player log-loss improvement.  The exploratory
paired player SD is `0.140557`; its player-bootstrap 95% interval is
`[0.132147, 0.149679]`.  Readiness must use the conservative upper endpoint,
`0.149679`, which exceeds the `0.078770` maximum compatible with 80% planning
power at 487 confirmation players.

This is a current-corpus limitation under the frozen estimand, threshold, and
player split.  It is not an optimizer or access-control failure.  At the same
SD upper bound, the minimum detectable effect at 487 players is `0.019002`
nats and the required player count for a `0.01`-nat effect is 1,759.  The
fixed confirmation arm contains only 487 structurally participating players.

The exploratory point improvement is much larger, `0.160192` nats.  That does
not repair the failed minimum-effect power gate and is not confirmation
evidence.  Thresholds cannot be changed after seeing it.

F0-H0 remains stopped.  This result makes no human-trap, causal, product,
playing-strength, `A_allow`, promotion, training, deployment, publication, or
release claim.

## Identity and frozen ordering

The first numerical contract was committed and pushed before estimator code
or new result access:

- v1 plan identity:
  `0ed6030a41efc91724fa5ab453a5d126d1ab55923182629451e067dd6cdfcbb3`;
- v1 plan file SHA-256:
  `d88b43d651374410b4d9e32ffde5485bab065f9ecc4c14ab8c390fd178d56caf`;
  and
- freeze commit: `44e14c45880e52489b151d1000c205a1d4918ec7`.

Its outcome-blind greedy affinity assignment failed its own structural gate.
Only 286 players appeared in same-fold sampled games, below the frozen 800
minimum.  The five folds contributed 9, 662, 10, 1,280, and 4 games.  No raw
game, human action, feature, Malom label, or outcome had been read.

The v1 plan remains immutable.  A separately sealed v2 changed only the
player-to-fold assignment:

- v2 plan identity:
  `246eef0ae2c42cb6606b8fa28a2ebb323f7ef65b7a80bdc09e4e456fba43eb87`;
- v2 plan file SHA-256:
  `223ac31c52c789465b7f3457aebc25bdbbd45e1cb51240b4fc83e2bf8399a2a5`;
  and
- correction commit: `a2afa1e464e86359b226161e5b7cc40a7189aa61`.

All numerical, feature, sample-budget, uncertainty, power, D-to-L, access, and
claim sections are inherited byte-for-byte from v1.  V2 uses whole Louvain
communities at resolution 2.0 and a frozen structural balancing rule.

The corrected folds and sample were then frozen before outcome access:

- structure identity:
  `b2ab654856a13d17fbc5256b6395c078e6cd13db9114da390daf720d952e6ae4`;
- structure file SHA-256:
  `b4664e6eb3e4cd32779f83a88da60a4b0734b67380348a0b1e30d6ceb0c53391`;
- sample identity:
  `92538491c86bfdb60e1129226fb8f101f25d0accbab65feccae723632ce2fb90`;
  and
- structure and numerical-core commit:
  `3b990fb66ae3c8060fee4c3241fe65e9969e47ab`.

The complete execution implementation was committed and pushed at
`54ae724af16c7efe9f0c5eea8f2e88edff106aba` before the sample was opened.
The completed result identity is
`0df4a8bcfab8636048c8b005945a1d4bd719b23f377c06d25a6d6e5b745d0ec2`;
its file SHA-256 is
`b53eb80f18cbac197954f99dcae926bea7eedb2a2fff1b75484e43148c83a52d`.

## Preflight reconciliation

Every sealed identity was independently recomputed before implementation.

| Artifact | Declared and recomputed identity | File SHA-256 |
| --- | --- | --- |
| screen v2 | `5919b9666d66c568898797e3b2089a71a71bc289696291d29c7aec6dd91e0935` | `c7be5ee600e1a52657108ff923e4a28831451b563ab57bd152edbc77d1731289` |
| task-named split v2 | `b3cdff57ccdbe4148d661c10cbbb6c6515b76ccd76e0cead2d39c2ab912b3acb` | `75925b87fac6695d918792e23a25f10497d1098082c8441a622494ff129ee2f0` |
| selected split v3 | `8187ffa06cc73f4e052b7481f06dc3629a23feace63e086c7075c74c17940028` | `8b98ec4ebd0876d2d5d56e1cfc35e99259724eace18ab09337a7f8e4a31425d3` |
| F0-D0 manifest | `bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7` | `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6` |
| B2 membership | `06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b` | `06c3be92c87927d506dc36eb908aec3064220f4ead2ebb3b5ff3dfb7bf5032cb` |

The F0-D0 corpus identity is
`4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29`.

The task named `human-feature-deviation-train-split-v2.json`, but that file
is the preserved rejected cut-minimizing split with only 137 confirmation
games.  The frozen screen-v2 lineage explicitly binds the selected
activity-balanced split stored as `train-split-v3.json`.  The identity chain,
not the ambiguous version label, controls this result.

The existing final-test guard was exercised with two real failure tests.  It
raised before the mocked raw reader, decision loader, or feature producer was
called.  Prior split and exploration records also independently reported zero
research-confirmation and official protected-partition result reads.

## Frozen feature and numerical contract

The exact ten fields are:

1. `source_degree`;
2. `destination_degree`;
3. `capture_degree`;
4. `closes_mill`;
5. `opponent_immediate_mill_destinations_removed`;
6. `creates_mill_fork`;
7. `new_own_potential_mills`;
8. `own_mobility_delta`;
9. `opponent_mobility_reduction`; and
10. `captured_opponent_threat_lines`.

The first three are the nested geometry control.  The full model uses all
ten.  Exact definitions are frozen in the v1 plan and verified against
`extended_action_feature_scores`; no feature was added after exploration.
Malom tier or `A_pos` membership is never a predictor.

The model is a no-intercept conditional multinomial logit.  It minimizes the
average of each actor player's average choice-set negative log likelihood,
plus `0.5 * 0.01 * ||beta||^2`.  Every coefficient is penalized.  Fold-local
standardization sees training alternatives only.  Complete atomic actions
are sorted by `(from, to, capture)`; duplicate, missing, nonfinite, empty, or
over-768 inventories fail closed.  Single-action choices abstain from both
specifications and count against a signed one-percent ceiling.

The deterministic L-BFGS optimizer uses Armijo backtracking, 100 iterations,
a `1e-7` primary gradient tolerance, and frozen separation diagnostics.  The
focused suite first failed because the implementation module did not exist,
then passed only after these behaviors were implemented.

## Outcome-blind fold structure

V2 assigns whole graph communities to five folds.  The assigned player counts
are 204, 226, 226, 226, and 226.  The 20,264 research-exploration games split
into 8,271 same-fold and 11,993 cross-fold games.  The latter, 59.18%, are
discarded entirely from fitting and scoring.

Each fold contributes exactly 1,280 sampled games:

| Fold | Assigned players | Sample players | Games | Decisions | Kish | Gini |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 204 | 169 | 1,280 | 62,220 | 15.75 | 0.836 |
| 1 | 226 | 186 | 1,280 | 60,860 | 26.48 | 0.788 |
| 2 | 226 | 212 | 1,280 | 56,224 | 35.46 | 0.774 |
| 3 | 226 | 198 | 1,280 | 55,264 | 37.04 | 0.759 |
| 4 | 226 | 215 | 1,280 | 57,624 | 29.06 | 0.802 |

The union contains all 980 structurally participating exploration players.
Every player's complete sampled games occur in only one held-out fold.

## Expanded exploration and oracle accounting

The predeclared projection for 6,400 games was 302,776 decisions, 4,017,835
queries, and 697.5 seconds.  Hard limits were 400,000 decisions, 6,000,000
queries, and 1,800 active seconds.

The completed pass measured:

- 6,400 games and 292,192 decisions;
- 979 observed actor players;
- 3,869,797 read-only Malom queries in 314.896 seconds;
- 13.244 queries per decision and 12,289 queries per second;
- 100% positional-label coverage and zero oracle abstentions;
- 1,166 single-action choices, or 0.399%; and
- `|A_pos| > 1` on 88.5798% of decisions.

The modifiable share is close to both earlier exploratory observations,
88.39% and 88.609%.  It is reported as a consistency diagnostic, not a rerun
or relaxation of the stopped F0-H0 state-frequency screen.

The structure counted 980 player incidences; 979 appeared as decision actors
after strict replay.  The readiness floor was 800, so this difference does
not change the decision.

## Optimizer and cross-fit diagnostics

All ten fold/specification fits converged.  Geometry fits took 10 iterations;
full fits took 22 or 23.  Final geometry gradient infinity norms were between
`1.03e-8` and `2.71e-8`; full norms were between `1.38e-7` and `7.21e-7` and
met the frozen secondary convergence rule.

Observed-information condition numbers were 10.86 to 11.09 for geometry and
37.62 to 38.11 for the full model.  Maximum absolute coefficients stayed
below 0.90, minimum information eigenvalues stayed above 0.038, and no chosen
probability met the near-separation threshold.  Fold zero was fitted twice;
both specifications reproduced coefficients and objectives exactly.

There was no implementation or design failure in the completed v2 run.

## Paired log-loss variance and power

Average-unique-player exploratory log loss was 2.146903 nats for geometry and
1.986711 for the full panel.  Their paired improvement was 0.160192 nats,
with player-bootstrap interval `[0.151550, 0.168783]`.

The player SD was 0.140557, with 95% bootstrap interval
`[0.132147, 0.149679]`.  The conservative upper endpoint controls readiness.

The planning coefficient was independently recomputed as

`(1.9599639845 + 0.8416212336) / sqrt(N)`.

At `N=487` it is 0.126951944, and the maximum SD for a `0.01`-nat effect is
0.078769963.  The conservative projected minimum detectable effect is
0.01900209 nats.  The gate fails.

At the required Kish sensitivity `N=58.9108356`, the coefficient is
0.365011246, the SD ceiling is 0.027396416, and the minimum detectable effect
is 0.05463466 nats.  It also fails, so changing between the two declared
planning views does not flip the decision.

At the SD upper bound, 1,759 independent players are required to power the
frozen `0.01`-nat effect.  This is 3.61 times the 487-player confirmation
structure.  Alternatively, a minimum effect of at least 0.019002 nats would
be required at 487 players, but adopting that value would be a new problem
and cannot be done after this result.

## D-to-L estimator and precision

The binding frozen text is not “95% half-width at most two points.”  It is:
the top-minus-bottom risk-quintile D-to-L difference must be at least two
percentage points and its lower 95% bound must also be at least two points.
The handoff's half-width wording is materially different and was not used.

Fold-specific 20th and 80th percentile boundaries came only from that fold's
training players.  Each player had equal total parent-D weight.  Zero-event
players remained zero; no pseudo-count created an event.

Across 199,234 out-of-fold parent-D decisions there were 10,416 D-to-L events
from 769 players.  Another 210 of 979 parent-D players had zero events, or
21.45%.  The top and bottom groups covered 897 and 979 players respectively.

The exploratory top-minus-bottom contrast was 0.379851, interval
`[0.355157, 0.404032]`.  Its conservative projection at 487 players has
half-width 0.035013 and lower bound 0.344838.  The Kish sensitivity lower
bound is 0.279183.  Both exceed the binding 0.02 floor.

The full-panel D-to-L Brier improvement over geometry was 0.035231, interval
`[0.033020, 0.037443]`.  Its projected lower bounds are 0.032094 at 487 and
0.026213 under Kish.  Both are positive.  Thus the D-to-L support and
precision path passes this exploratory readiness calibration; it does not
override the failed log-loss power gate.

## Readiness decision and next authority

The conjunctive decision is `B_not_ready_fail_closed`.

- implementation or design failures: none;
- current-corpus intrinsic failures under the frozen contract:
  `paired_log_loss_player_SD`; and
- research-confirmation result reads: zero.

The current 487-player protected arm must not be opened under this plan.  A
later product decision would have to choose between obtaining at least 1,759
comparable independent confirmation players, or defining a genuinely new
minimum-effect question of at least about 0.019 nats.  Neither change is
authorized here, and neither may be described as a repair of this frozen
result.

## Claim boundary and inherited bias

Every oracle statement is positional-only `A_pos`, never full-history
`A_allow`.  The source domain is the observed PlayOK-like platform.  UI
orientation, time control, and exact source rules variant are unavailable.

History recovery is nonrandom: 1,751 excluded games contain only 35 draws,
while 92,789 retained games contain 26,157.  Another 54,923 games have no
independently verifiable terminal basis.  The result cannot be transported to
product UI users or a new population.

The access ledger records zero research-confirmation, official selection,
official confirmation, official final-test, HumanDB, source-pool `2eb04f54`,
database-write, game, search, strategy-model, or training operations.

## Verification

Task-scope Ruff passes for the evaluation module, both execution scripts, and
both estimator-readiness test files.  The complete feature-deviation focused
group passes 36 tests.  The B2 freeze and final-test guard file passes eight
tests.  The mandatory Malom, DB-teacher, and label-provenance suite passes 103
tests plus 498 parameterized subtests.
