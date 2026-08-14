# Retained-v4 product-route and specialist-lineage audit

Date: 2026-08-14

Status: `completed_zero_game_read_only_audit`

Source commit before this documentation change:
`7f073e4c5ad8b86ca78b0f669c75daef52a2e115`

This audit closes the deployment question for the current retained-v4
candidate. It loaded models and queried existing read-only evidence, but it
played no game, trained no model, updated no database, copied no checkpoint,
and did not read any of the 108 unconsumed held-out source-pool records.

## Decision

The exact `retained-v4-no-refresh` route remains the preferred research
candidate and retained-v3 remains its frozen comparator. Retained-v4 is **not
promoted** into the current product Generalist slot.

The promotion gate for placing the retained-v4 weights behind the current
`GeneralistAgent` product route is `fatal_stop`: the product route is not the
route that was evaluated, and the mismatch is now demonstrated by inference
rather than inferred only from code. This verdict does not invalidate the
research checkpoint or its completed named-route held-out result.

There is no launch command. No new evaluation or training is authorized.

## Observed product routing

The current product has two different neural-player paths:

- a session starts with both `use_overseer_player` and
  `use_generalist_player` false;
- difficulty 9 or 10 automatically activates the three-specialist router;
- the specialist route takes priority if specialist and Generalist modes are
  both requested; and
- the Generalist route runs only after an explicit Generalist checkbox choice
  and only when specialist mode is inactive.

Consequently, ordinary difficulty 1--8 play does not use a neural player,
while ordinary difficulty 9--10 play uses the three specialists. The current
Generalist slot is an explicit opt-in research surface, not the default
high-difficulty product opponent.

The current Generalist and frozen training routes have the same 134-float
input width, so ordinary tensor-shape validation cannot detect their semantic
difference.

| Component or behavior | Product `GeneralistAgent` on this host | Frozen retained-v4 route |
| --- | --- | --- |
| Policy weights | retained-v4 policy for this audit | retained-v4 policy |
| Sentinel | loaded and consumed | disabled |
| ValueNet | phase ValueNet loaded and consumed | disabled |
| GapNet | loaded and consumed | disabled |
| HumanDB | active HumanDB | same identity, read-only |
| SpecialistDB | unavailable because `data/specialist_db.sqlite` is absent | read-only final v4 snapshot, SHA-256 `3d69d1ac...` |
| Malom | assigned through `set_db()`, but `score_moves()` passes `db=None` and does not consume it | strict Malom used as the lookahead endgame database |
| Learner continuation | heuristic product lookahead | frozen target model, age 5,000 games |
| Rollout evaluation | product heuristic | `training_rollout_evaluate` |
| SpecialistDB minimum samples | encoder default 5 when a DB exists | explicitly 3 |
| Dependency failure | non-strict zero/neutral fallback is possible | `strict=True`, fail closed |

The frozen bundle is
`817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f`.
It binds HumanDB, SpecialistDB, Malom, the policy model, the frozen target,
12-ply lookahead, five-ply target continuation, and terminal order
`rules-then-malom-v1`.

## Same-position reproduction

### Inputs and method

The audit selected the first four unique completed start positions in ledger
order for each of placement, movement, and flying from the already completed
1,012-game held-out ledger:

- ledger SHA-256:
  `7dd72447a4e918e5ee816604f18b5e3497a002a2969d1ca22119043a8df8fc45`;
- deterministic 12-position selection identity:
  `1a3de933b496e8d7fb0b69404681d70b45be517bb6f6a54f963994782901e6fc`;
- remaining source-pool records read: 0; and
- games played: 0.

The frozen bundle verifier reused `capture_model_canary` and
`verify_model_canary`. Both the policy and target reproduced their stored
policy and value outputs with maximum absolute difference 0.0. The audit then
fed the same legal moves and the same retained-v4 policy weights through:

1. the current product component route, using read-only HumanDB access for
   query equivalence; and
2. `TrainingAlignedPolicy` with its exact bound resources and strict encoder.

For each position, probabilities were aligned by the full
`(from, to, capture)` move key. Total variation is one half of the absolute
probability difference summed across all legal moves.

### Results

| Phase | Start | Same argmax | Total variation |
| --- | --- | ---: | ---: |
| placement | `late-import-heldout-001` | no | 47.88% |
| placement | `late-import-heldout-004` | no | 73.54% |
| placement | `late-import-heldout-007` | no | 54.86% |
| placement | `late-import-heldout-010` | no | 65.23% |
| movement | `late-import-heldout-002` | yes | 57.55% |
| movement | `late-import-heldout-005` | yes | 9.00% |
| movement | `late-import-heldout-008` | no | 62.69% |
| movement | `late-import-heldout-011` | yes | 20.78% |
| flying | `late-import-heldout-003` | yes | 2.41% |
| flying | `late-import-heldout-006` | yes | 75.42% |
| flying | `late-import-heldout-009` | no | 13.57% |
| flying | `late-import-heldout-012` | no | 60.22% |

Aggregate observations:

- argmax agreement: 5/12;
- argmax disagreement: 7/12;
- mean total variation: 45.26%;
- range: 2.41% to 75.42%;
- placement argmax agreement: 0/4; and
- 8,796 of 34,572 aligned feature cells differed by more than `1e-7`.

The canary result rules out a damaged or different bare model as the cause.
The feature and action results directly falsify route equivalence. The 1,012
held-out games apply to the frozen training-aligned route and cannot be
transferred to the current product `GeneralistAgent` route.

## Active three-specialist inventory

All three active `best.pt` files retain the hashes recorded in the July main
integration audit, load successfully as 134/80 `ScaffoldedPolicyNet` models,
and contain finite weights.

| Stage | SHA-256 | Games | Difficulty | Best rate | LR | Temperature |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `s_open_v2` | `D020E1442676E16CDCED6C91DAC958817C3A22A283CC293D6E19930A87703701` | 14,350 | 3 | 0.225 | 0.0000642857 | 0.64891 |
| `s_mid_v2` | `A587AB995224A1D43C99FD2F42E4BFF9C060AC6DA55EDCDDB43A39FC07EF26D2` | 20,750 | 7 | 0.1388889 | 0.00005 | 0.537015 |
| `s_end_v2` | `5DE51A1AFD5794374D4394CCE2950957A23F02504B5C5952A062D91414B94BE8` | 17,600 | 3 | 0.15 | 0.00005 | 0.592035 |

Each payload's `source_checkpoint` is a self-referential maintainer path of
the form
`/home/.../NMM_ollama/learned_ai/checkpoints/scaffolded/<stage>/best.pt`.
The files contain model weights/configuration and the table fields above, but
not an optimizer, RNG state, target model, data cursor, database identity,
Malom-label version, source commit, run manifest, or exact-resume envelope.

The current blobs first appear in reachable Git history in commit
`1335536e21e3673e83bec333eb7e5fac339d690d`, but that records repository
import time, not the training run that produced them. Therefore it is not
possible to prove from the checkpoint or repository history whether their
training consumed the corrected Malom decoder and corrected persisted labels.
The correct durable conclusion is **lineage untraceable**, not a claim that a
post-correction retraining occurred.

## Sanmill as a separate product option

The retained-v4 and retained-v3 fixed held-out score rates against the pinned
500,000-node Sanmill endpoint were 42.7866% and 41.1067%, respectively. This
supports the retained-v4 research selection but does not make either route a
replacement for the opponent it scored below 50% against.

Sanmill and this repository have compatible AGPL-family licensing, the strict
bridge already has smoke and long-run evidence, and node ceilings provide a
natural technical difficulty control. The locally pinned training binary is
5,641,216 bytes; the separate prefix-replay binary is 4,109,312 bytes. Thus
the earlier informal 3.7 MB size is not an exact identity for the active
training runtime.

Direct Sanmill integration is a plausible future product design if the only
need is a stronger high-difficulty opponent. It is not authorized or selected
by this audit; packaging, device latency, UI, redistribution, and failure
behavior still require their own product contract.

## Hypotheses and limits

### Supported

- The current product Generalist route is not deployment-equivalent to the
  retained-v4 route.
- The current default routing gives the Generalist slot little direct product
  exposure.
- The active specialists are byte-identical to the previously audited legacy
  artifacts and are loadable.
- Their corrected-label training lineage cannot be established.

### Not established

- The 12-position reproduction is not a strength comparison between the two
  routes; it is a semantic-compatibility falsification.
- The retained-v4 advantage is not refresh causality, Elo, or population
  superiority.
- Sanmill has not been accepted as a shipped product component.
- The active specialists are not proved incorrect merely because their
  lineage is incomplete.

## Deferred code findings

No code was changed under this audit. The following defects remain queued for
a separately authorized implementation:

1. `specialist_router.py` says `(k, 122)` and 15-ply lookahead, while the live
   schema is 62 base features plus 12 times 6 lookahead signals, or 134.
2. `web/app.py` says alpha-beta supplies top-K specialist candidates, while the
   code passes every legal move.
3. Specialist and Generalist scoring failures return `None`; the web path
   falls back to the coordinator after stderr/log output without a persistent
   user-visible degraded-mode state.

The third item conflicts with the repository's fail-visible component policy
and should be treated as a product reliability fix, not as part of model
promotion.
