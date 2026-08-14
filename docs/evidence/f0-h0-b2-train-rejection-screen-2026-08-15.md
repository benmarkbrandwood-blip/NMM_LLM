# F0-H0 B2 train-only rejection screen evidence

Date: 2026-08-15

Status: `stop_condition_triggered`

Decision source: corrected v2 result identity
`8bd2da62785e9c8cda0a055e98213959cbdf8f88aa860384171f00f4f39c6bdc`

## Decision

The F0-H0 rejection screen triggers the governing stop condition.  The
current observed PlayOK-like train corpus does not meet the frozen independent
support and concentration contract for optional human-target specialization.
E0, F0-H1, T0-H-pilot, trap-reward work, and training changes remain closed.
No substitute corpus may be introduced to bypass this result.

The stop is not caused by a lack of positional choice.  Of 429,523 sampled
human decisions, 380,923, or 88.69%, have more than one positional-tier-
preserving move in `A_pos`.  It is also not caused by Malom coverage or by
the state-level estimability precheck.  Those gates pass.  The decisive
failures are:

- only 20.16% of decisions belong to a `ring16` class supported by at least
  five independent player keys and ten independent games, against the frozen
  80% floor; and
- player and supported-choice evidence is too concentrated.  Player Gini is
  0.804 against a maximum of 0.75, Kish effective support is 177.7 against a
  minimum of 500, and the top 5% and 10% player shares exceed their limits.

This is a positional-only `A_pos` result.  It is not an `A_allow` result, a
full-rule safety proof, a causal inducement estimate, an approval of a later
gate, or evidence about product UI users or a new population.

## Frozen inputs and technical correction

The original pre-statistic plan remains unchanged:

- plan identity:
  `dd87175dc950cbcde4b0b44cd5d4a8da0b039dcbd3cacaf198ba43ec00de0bdc`;
- plan file SHA-256:
  `a5c9c09bba136f9b794c369aded06df6c0980fe9849320cfdaf6bacca22a5dc1`;
- threshold-object identity:
  `006acb8b1efe4d13b8e3fcae9d30739d9c0f98346130db0e0b795204924c536f`;
- frozen train-sample identity:
  `124f568c60c4c3e475987ac7f2187b758c617cf70a7ed1b052b2301265138cd1`.

The first execution produced historical result identity
`84434066d5c4fc58fd82585a32c5bff9ef69ecfe4badb2a0c3289b4b0fb7068b`
and file SHA-256
`e03e6e63a788e53f76479c48995d3c02657231c74a78d6396a35cb57ce403297`.
Review found a deterministic estimator defect: when a transition had zero
observed events, unequal Jeffreys pseudo-count denominators could manufacture
a positive action contrast.  A focused regression test reproduced the
failure before the correction.  The historical plan and result are retained
unchanged but are not the decision source.

The corrected technical replay was frozen separately:

- v2 plan identity:
  `a6972c3dae62ec249ccf6ea7bc7bf46132288a15db41b1c33b347b75615a9d0c`;
- v2 plan file SHA-256:
  `d14ea75eba7f073c45451e26e5fa38293120918c1c7ca1bac7115634343e7f42`;
- v2 result file SHA-256:
  `9f2d1d8e85a358ffd5db4c13cef46e7fb9d41e3c73d89e3d2a4c3bc227cbe809`.

The v2 plan copies the v1 sample and complete threshold object exactly.  It
changes no gate, seed, membership, or resampling rule.  It binds only the
zero-event correction.  Independent comparison confirms that v1 and v2 have
identical sample, bases, support, reachability, concentration, four-A,
mechanism, product-scope, secondary, gate, access, and prohibited-operation
records.  Only plan lineage and the defective zero-event four-B rows change.

## Scope and access boundary

The already frozen 10,000-game cost sample contains:

| Partition | Frozen sample games | Content opened |
| --- | ---: | ---: |
| train | 9,113 | 9,113 |
| selection | 887 | 0 |
| confirmation | 0 | 0 |
| final-test | 0 | 0 |

No resampling occurred.  The train intersection contains 429,523 decisions
from 1,314 player keys.  The full train membership remains 36,949 games,
1,742,416 decisions, and 2,216 player keys; its game content was not opened
outside the frozen sample.  Strict factual-result counts use only F0-D0
metadata for the full train membership.

The screen issued 5,558,466 read-only queries against the
`sector-corrected-v1` Malom snapshot.  All 429,523 decisions were covered and
zero were imputed.  The result contains the SHA-256 and size of every one of
the 9,113 raw input files, the Malom manifest and component inventory, the
ruleset, and all frozen lineage inputs.

The access audit records zero selection, confirmation, or final-test content
reads; zero source-pool `2eb04f54` reads or consumption; and zero games,
searches, model loads, training updates, database writes, or rebuilds.

## Dimension 1: independent support

Support requires at least five independent player keys and ten independent
games.  Unsupported classes abstain.

| State unit | Classes | Supported classes | Supported class share | Supported decision share | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact positional FEN | 335,932 | 786 | 0.234% | 12.14% | sensitivity |
| `ring16` positional class | 271,762 | 1,705 | 0.627% | 20.16% | fail, floor 80% |

For exact states, the median and 90th-percentile independent game support are
both one; the 95th percentile is two and the 99th is three.  For `ring16`,
the corresponding values are one, two, three, and seven.  Aggregation raises
coverage but does not approach the signed 80% decision floor.

## Dimension 2: modifiable positional-state reachability

`A_pos` cardinality is greater than one for 380,923 of 429,523 decisions:
88.685%, with a whole-game clustered 95% interval of 88.541% to 88.830%.
The cardinality median is 5; the 75th, 90th, 95th, and 99th percentiles are
11, 20, 23, and 50.  Only 48,600 decisions have cardinality one.

| Stratum | Modifiable / total | Fraction |
| --- | ---: | ---: |
| placement | 140,536 / 162,377 | 86.55% |
| movement | 226,817 / 253,091 | 89.62% |
| flying | 13,570 / 14,055 | 96.55% |
| White actor | 194,140 / 216,851 | 89.53% |
| Black actor | 186,783 / 212,672 | 87.83% |
| parent W | 57,571 / 74,578 | 77.20% |
| parent D | 265,493 / 295,683 | 89.79% |
| parent L | 57,859 / 59,262 | 97.63% |

Support-qualified and modifiable decisions total 83,627, or 19.47%, with a
whole-game clustered interval of 19.19% to 19.75%.  Every one of the 9,113
sample games contains at least one such decision.  All four preregistered
reachability gates pass.  The game therefore offers substantial positional
choice; the rejection is about reliable human evidence for those choices.

## Dimension 3: concentration

| Unit | Gini | Kish effective units | Top 1% | Top 5% | Top 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| players | 0.804 | 177.7 | 17.92% | 51.80% | 72.95% |
| games | 0.292 | 6,917.9 | 3.42% | 12.79% | 22.76% |
| exact states | 0.211 | 1,667.1 | 15.53% | 22.77% | 29.61% |
| `ring16` classes | 0.349 | 944.7 | 22.36% | 32.99% | 39.94% |

For support-qualified modifiable decisions, the top 5% of players contribute
51.23%, also above the frozen 50% maximum.  There are 1,314 contributing
players, so the failure is not an absence of player IDs; it is unequal
evidence weight.  Six of eight concentration gates fail.  The only passing
player gates are the 25% top-1 maximum and the minimum of 100 supported
modifiable players.

## Dimension 4: upper bound and estimability

### Four-A state-level estimability

The preregistered `k=20`, `m=5` precheck passes:

| State unit | Repeated classes | Classes with >=2 safe actions at `m` | Covered decisions | Games | Players |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact positional FEN | 325 | 275 | 43,900 (10.22%) | 9,070 | 1,313 |
| `ring16` class | 504 | 394 | 62,306 (14.51%) | 9,070 | 1,313 |

This means the upper-bound calculation is empirically defined on a supported
subset.  It does not repair the failed 80% general support gate and does not
permit unsupported states to inherit a global probability.

### Four-B state-conditioned association

There are 7,827 games with a first positional theory downgrade, all `D->L`.
Of them, 6,777 follow a modifiable preserving predecessor and 1,691 follow a
support-qualified one.  The latter is 18.56% of all 9,113 sample games, with
a fixed-source engineering interval of 17.77% to 19.37%.

For `D->L`, the uncorrected weighted within-class safe-action max-minus-min is
4.36%.  Deterministic whole-game two-fold cross-fitting, Jeffreys shrinkage,
and a 2,000-replicate `ring16`-class bootstrap give a corrected point of
5.63% and interval 4.04% to 7.52%.  This is an observational within-class
association on reused train evidence, not a causal trap effect and not a
comparison with a frozen reference policy.

`W->D` and `W->L` each have zero observed events across 62,306 eligible
exposures.  Their action-contrast point is zero.  The zero-event exposure-rate
Wilson upper bound is 0.0062%; no action contrast is asserted for either
transition.  These corrected rows replace the prior-generated v1 artefact.

### Four-C strict factual-result basis

The full train strict-outcome subset contains 15,135 games and 1,539 player
keys:

| Recorded and independently replayed outcome | Games | Share |
| --- | ---: | ---: |
| White win | 3,214 | 21.24% |
| Black win | 3,786 | 25.01% |
| Draw | 8,135 | 53.75% |

These factual outcomes are reported separately.  They are not extrapolated
into the 9,113-game mechanism sample and do not establish conversion after a
human error.

## Gate ledger

Sixteen of 23 frozen gates pass.  Seven fail:

1. `independent_support.minimum_ring16_supported_decision_fraction`;
2. `concentration.maximum_player_gini`;
3. `concentration.maximum_player_top_5_percent_share`;
4. `concentration.maximum_player_top_10_percent_share`;
5. `concentration.minimum_player_kish_effective_units`;
6. `concentration.maximum_supported_modifiable_player_top_5_share`; and
7. `concentration.maximum_ring16_state_top_1_percent_share`.

All modifiable-reachability, four-A estimability, product upper-bound, and
oracle-coverage gates pass.  Because the contract is conjunctive, the seven
support/concentration failures trigger the stop condition.

## Required biases and claim limits

History selection is non-random: the excluded 1,751 games contain only 35
draws, while the 92,789 retained-history games contain 26,157 draws.  Another
54,923 games have no independently verifiable terminal basis.  Zero observed
result disagreement does not validate those games.

UI orientation, time control, exact source rules variant, explicit import
batch, and upstream source-file identity remain absent.  The result is limited
to the observed PlayOK-like source domain.  The known state-convergence
addendum remains binding: state overlap is diagnostic, while game and player
membership own contamination and generalization claims.

The design's stop condition for overly concentrated independent support is
therefore met.  The correct action is to stop this optional human-target
specialization path for the frozen corpus and definition.  This result grants
no authority to open selection, confirmation, or final-test; run E0 or any
later gate; change rewards; train; or search for replacement data.

## Verification

The result verifier reproduces the v2 plan lineage, result identity, and file
SHA-256.  Focused tests cover sealed plans/results, exact v1/v2 threshold and
sample equality, train-only access, the real protected-partition failure,
support and estimability rules, zero-event correction, and decisive evidence
values.  Task-scope Ruff and the mandatory Malom, DB-teacher, and label-
provenance test group are part of the completion check.
