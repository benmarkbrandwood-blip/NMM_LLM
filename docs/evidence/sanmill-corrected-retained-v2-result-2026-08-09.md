# Sanmill-Corrected Retained v2 Completion Evidence — 9 August 2026

## Result

The authorized `managed-sanmill-corrected-retained-v2` plan reached its
bounded terminal condition:

- controller state: `completed`;
- completed games: 5,000 of 5,000;
- accepted process segments: 20 of 20;
- recorded active controller time: 1.563432 hours, below the 12-hour limit;
- final segment-completion event: sequence 46,
  `30bc9d9014d6cfd7331db6fe03cc38581d3670b02a4c51bbc8cfae4b8a4ce28f`;
  and
- final plan-completion event: sequence 47,
  `fa720f7972f256f2888c691755ae3ecb8d2a79f8b2f42ae2de3b8f1debfc6a82`.

No controller lock remains. All 20 mandatory fixed-state policy-health gates
passed before their segment became an accepted resume parent.

This was not an uninterrupted, single-implementation execution. A verified
Sanmill response-interpretation defect stopped segment 13 after game 3,200,
and a later host-side command timeout interrupted segment 18 before its first
local checkpoint. Both incidents were handled through fail-closed,
ledger-recorded recovery described below. The final state is an auditable
managed continuation, not a claim of bit-for-bit equality with a hypothetical
failure-free run.

## Frozen authority and final implementation identity

| Item | Identity |
| --- | --- |
| Plan ID | `managed-sanmill-corrected-retained-v2` |
| Plan semantic SHA-256 | `d498cd1b9a32d8e6ebb9cf6da4c38c884f0f4584a32538244779c67e143685e8` |
| Plan file SHA-256 | `810e43350b6f440e56eb5f2d242797664686ec2e29592df421b051a2ba24fd6d` |
| Authorization file SHA-256 | `1c51fd0aa53fe38bfd3f2fa458719728485b47757dfb4b0f15cb3ab0a27d39a1` |
| Plan source commit | `ff7e360d3660c142768b1247d2d17ad79ceb435f` |
| Final runtime commit | `4973e321d17a5b6f3cb697d66ccdc9701d4d8a30` |
| Experiment ID | `dev-v4-sanmill-corrected-retained-v2` |
| Experiment digest | `sha256:e51a80eb936426fbdf81f219a29458afeed05188dd8f0d7e336fb50a00c179fa` |
| Resume-config SHA-256 | `b8530c75d7f411402ab16c3f4610f3832b952bd3acaf88d9742a1e917fa37df9` |
| Training ruleset semantic digest | `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| Sanmill source commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| MIF release | `mif-suite-1.0` at `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a` |
| MIF Suite JCS digest | `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |

The immutable plan retained its original semantic identity during recovery.
Only published descendant implementation commits were admitted:

1. `7049416b297f1a280510d81dfe38a05fb5ce1964` accepts a successful,
   searched Sanmill response that expands zero new nodes because its
   persistent transposition table satisfies every search call;
2. `627913996738874707842e91ce39c29f56f4185e` adds evidence-bound
   recovery for a failed managed segment; and
3. `4973e321d17a5b6f3cb697d66ccdc9701d4d8a30` permits a host-interrupted
   segment with no local checkpoint to fall back only to its last accepted
   parent boundary.

## Complete lineage-log accounting

The active `segments/` directories contain 4,800 final-lineage game rows.
The other 200 accepted rows, games 3,001 through 3,200, are preserved in:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/
managed-sanmill-corrected-retained-v2/quarantine/
segment-0013.failed-20260809T051546Z/train_log.jsonl
```

Those rows are not discarded work: the segment-13 recovery event explicitly
binds `resume_game_count=3200`, and the recovery checkpoint restores the
model, optimiser, scheduler, scaler, RNG and trainer state saved after game
3,200. The failed action after that boundary never produced a counted game.

Combining that accepted prefix with the active segment logs gives:

- exactly 5,000 rows;
- game numbers exactly 1 through 5,000;
- no missing or duplicate game number; and
- 5,000 unique `game_id` values.

The interrupted first attempt at segment 18 produced no accepted game row.
The completed replacement segment supplies games 4,251 through 4,500.

The machine-local read-only dashboard was adjusted after completion to use
the recovery ledger when assembling this split prefix. Before that local
display-only adjustment it showed 4,800 rows and a discontinuity. The
controller ledger, checkpoint envelopes and hashes in this document are the
acceptance evidence; the dashboard is not.

## Training-outcome diagnostics

The trainer encodes a learner win as `1.5`, a learner loss as `-1.0`, and
rules draws or max-ply truncations as draw-class outcomes. The aggregate is:

| Opponent / learner colour | Games | Wins | Draws | Losses | Score rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 5,000 | 93 | 2,036 | 2,871 | 22.22% |
| Frozen target, all | 2,987 | 81 | 1,175 | 1,731 | 22.38% |
| Frozen target, learner White | 1,489 | 46 | 596 | 847 | 23.10% |
| Frozen target, learner Black | 1,498 | 35 | 579 | 884 | 21.66% |
| Sanmill search, all | 2,013 | 12 | 861 | 1,140 | 21.98% |
| Sanmill search, learner White | 1,012 | 5 | 426 | 581 | 21.54% |
| Sanmill search, learner Black | 1,001 | 7 | 435 | 559 | 22.43% |

For the final 200 chronological games, the aggregate score rate was 47.50%:
49.58% over 119 frozen-target games and 44.44% over 81 Sanmill-search games.
This is a training-tail diagnostic, not a frozen evaluation.

Sanmill-search outcomes by curriculum resource level were:

| Level | Node ceiling | Games | Wins | Draws | Losses | Score rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 207 | 0 | 0 | 207 | 0.00% |
| 2 | 5,000 | 197 | 0 | 0 | 197 | 0.00% |
| 3 | 25,000 | 191 | 0 | 0 | 191 | 0.00% |
| 4 | 100,000 | 393 | 0 | 1 | 392 | 0.13% |
| 5 | 500,000 | 1,025 | 12 | 860 | 153 | 43.12% |

The resource rows are confounded with training time: each later level saw a
different learner. They therefore cannot be read as a controlled comparison
of Sanmill strength levels or as evidence that a larger node ceiling is an
easier opponent.

Termination reasons across all 5,000 games were:

| Termination reason | Games |
| --- | ---: |
| Loss by fewer than three pieces | 1,662 |
| Loss by no legal moves | 1,302 |
| Threefold-repetition draw | 1,363 |
| 120-ply truncation | 494 |
| Fifty-move draw | 179 |

The 494 truncations are not rules-based draws. They remain separately visible
and must not be used to infer the natural draw rate of Nine Men's Morris.

## Mandatory policy-health gates

Every boundary evaluated the same 29 fixed Malom-critical states. The direct
lookahead argmax value-preserving rate was exactly 1.0 in all 20 reports. The
candidate thresholds were a preserving rate of at least 0.50 and a mean
preserving-minus-downgrading logit margin of at least -0.10.

| Segment | Game | Candidate preserving rate | Mean logit margin |
| ---: | ---: | ---: | ---: |
| 1 | 250 | 0.586207 | +0.000103 |
| 2 | 500 | 0.862069 | +0.002509 |
| 3 | 750 | 0.965517 | +0.023941 |
| 4 | 1,000 | 0.862069 | +0.028236 |
| 5 | 1,250 | 0.862069 | +0.045834 |
| 6 | 1,500 | 0.896552 | +0.131919 |
| 7 | 1,750 | 0.896552 | +0.190771 |
| 8 | 2,000 | 0.896552 | +0.316419 |
| 9 | 2,250 | 0.896552 | +0.521179 |
| 10 | 2,500 | 0.965517 | +0.963225 |
| 11 | 2,750 | 1.000000 | +2.253489 |
| 12 | 3,000 | 1.000000 | +3.132290 |
| 13 | 3,250 | 1.000000 | +3.500421 |
| 14 | 3,500 | 1.000000 | +3.671555 |
| 15 | 3,750 | 1.000000 | +3.688865 |
| 16 | 4,000 | 0.965517 | +3.043150 |
| 17 | 4,250 | 0.965517 | +2.294683 |
| 18 | 4,500 | 0.965517 | +2.561997 |
| 19 | 4,750 | 0.965517 | +2.353618 |
| 20 | 5,000 | 0.965517 | +2.398080 |

The final report SHA-256 is
`69ec46ef55933b7ba7cf1682268985c15a5f5c189e120d2bf1ba5a0d76e6aab8`.
The minimum candidate rate and margin occurred at the first boundary and
still passed. These results reject the previously observed policy-direction
collapse on this fixed corpus; they do not measure general playing strength.

## Incident 1: cached Sanmill search at segment 13

After the periodic game-3,200 checkpoint, Sanmill returned a complete legal
`d7-g7` logical turn. It reported 30 search calls and completed depth 30, but
`total_nodes=0` because its persistent transposition table supplied every
iteration without a new node expansion. The old parser incorrectly required
positive nodes even though the fixed node count is a ceiling.

The exact response failed before commit `7049416` and passed after it. The
repair retains positive-search-call, node-sum, budget, action-shape, legality
and state-replay checks. Its tracked reproduction record is
[`sanmill-zero-expansion-recovery-2026-08-09.json`](../experiments/sanmill-zero-expansion-recovery-2026-08-09.json),
SHA-256
`5c54c46cb0cf30b4f669347165589b45fa0f5fdce9f90545f7bf37ba80cd124a`.

The original failure event SHA-256 is
`6e80cca071b7b216f7d26d63c403c33707eb21e1fba78d3b41264e4e6d7a5657`.
The verified-recovery event SHA-256 is
`6e0342e8165a27c247b49def749ae1d8f7a90e9eae700c55a4e3089eb3e352b0`.

The game-3,200 checkpoint recorded SpecialistDB SHA-256
`8e86835a77ceb1033e56d43e593cc03e864dcdcdfa8c3d17ed734906c492654f`.
The live same-lineage database had already advanced to
`2d00329d852bcd671f4e6ab41c6d6ceb4dbd82a5faaa557352073b59112308a5`
by the time the trainer failed. Recovery restored the complete game-3,200
checkpoint state but preserved and rebound that trusted live database. This
is why the continuation is not bit-for-bit counterfactual parity.

The replacement segment completed game 3,250 and passed its policy-health
gate at a 1.0 candidate preserving rate and +3.500421 margin.

## Incident 2: host interruption before segment-18 checkpoint

An outer 30-minute command limit terminated the foreground controller just
after segment 18 began. This was not a trainer exception. The interrupted
attempt had no `train_log.jsonl` and no segment-local `latest.pt`, so it could
not be treated as completed work.

Commit `4973e32` made this specific host-interruption case fall back only to
the previously accepted game-4,250 checkpoint. It quarantined the partial
directory, backed up the live database, rebound the database and experiment
identities, and recorded recovery event SHA-256
`72de13fcfa9e35bd26c1b2545944a66dd1f9ea12e7dac5c899a32ac04ae5dc29`.
Technical failures without a failed-segment checkpoint remain fail-closed.

The replacement segment then completed games 4,251 through 4,500 and passed
its policy-health gate.

## Final checkpoint and SpecialistDB

`checkpoint_tool.py verify` accepted:

- checkpoint
  `learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/managed-sanmill-corrected-retained-v2/segments/segment-0020/latest.pt`;
- checkpoint ID
  `managed-sanmill-corrected-retained-v2-segment-0020:checkpoint:00000006`;
- save reason `final`;
- payload SHA-256
  `8b4017ce856012fa3c4d578c56c5f32a6d5ebae97b9f17c6cbd2c5228146de19`;
  and
- file SHA-256
  `df00861a5ced53b6c9b16ed89f2762d41a82f1d74fce970b5d0bdf6adba4ac4d`.

The descriptor records the final SpecialistDB identity
`ea2df42d6df837588e1a2d87e37bd025c2b612f87695aa9ae16da064aebf62a8`,
which equals the closed database file hash. Read-only SQLite verification
reported `quick_check=ok` and:

- 150,890 positions;
- 4,825 positions meeting the database's five-sample query threshold;
- 35,330 trusted `sector-corrected-v1` Malom labels;
- 1,862 winning lines;
- zero preferred or promoted plays;
- empirical position totals of 50,957 wins, 148,338 draws and 50,957 losses;
  and
- lineage root `managed-sanmill-corrected-retained-v2-segment-0001`.

The WAL was empty. A read-only audit left a normal shared-memory sidecar; the
database file hash remained stable.

Additional terminal artefact identities are:

| Artefact | SHA-256 |
| --- | --- |
| Controller ledger | `e86b1f541bc8a2c5359fc35621e86cf1dec561c44579fadead889151c13732ab` |
| Final segment run events | `cd181ca777d0d2ef9c189bb2e0b0501da5d3125cc72b1e837994d23bc70575e4` |
| Final policy-health report | `69ec46ef55933b7ba7cf1682268985c15a5f5c189e120d2bf1ba5a0d76e6aab8` |

The final segment did not create `best.pt`; its final report accurately says
that only `latest.pt` exists. This is expected when that process segment does
not satisfy its local best-checkpoint condition and is not a missing-output
error.

## Verification performed after completion

The final verification used a repository-local ignored pytest base directory
because the host's default `pytest-of-user` directory is not readable.

- 127 focused Sanmill parser, manager, preflight, checkpoint and policy-health
  tests passed;
- four historical tests tied to the moving Sanmill checkout were explicitly
  deselected after a separate run confirmed that they fail closed because
  current Sanmill changed protected paths relative to their older smoke-v2
  pin;
- 103 mandatory Malom, Sentinel DB-teacher and label-provenance tests passed,
  including 498 parameterized subtests;
- Ruff passed for the changed Python implementation and tests;
- `checkpoint_tool.py verify` passed for the final checkpoint; and
- `git diff --check` passed.

The four moving-checkout failures do not test the isolated
`sanmill-training-a6623f8` runtime used by this run. They remain part of the
known complete-suite baseline and are not presented as passes. No new full
`tests/` all-pass claim is made here.

The repeated `HumanDB Malom labels are disabled` stderr notice is expected:
the imported HumanDB's human frequencies and outcomes remain available while
its unversioned historical Malom columns are masked. It did not stop or alter
the run.

## Claim boundary and next gate

This evidence establishes bounded completion, checkpoint/database integrity,
recovery traceability, finite training updates and survival of the fixed
anti-collapse gate. It does not establish playing strength, promotion,
publication readiness, MIF full conformance, or superiority to Sanmill,
GameAI, a maintainer model or any other baseline.

The next experiment must be a separately frozen, held-out evaluation. It
must bind the retained candidate checkpoint, a compatible frozen baseline,
reviewed starting positions, colour swapping, fixed resource limits, rule
identity, adjudication and an acceptance rule before any candidate result is
viewed.
