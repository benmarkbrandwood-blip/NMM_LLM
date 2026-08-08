# Fresh Sanmill-Refereed Smoke 002 Result — 8 August 2026

## Decision

`passed_integration`

The separately authorised run `sanmill-refereed-fresh-v1-smoke-002`
completed from scratch at clean, published `dev` commit
`894360d11fc309e5aa58e3289d1c6817831a553c`. Before launch,
`dev == origin/dev`, the final exact-command preflight returned
`ready_for_smoke`, and both `errors` and `unresolved_decisions` were empty.

The run completed both scheduled games, performed one finite A2C update,
published a verified version-2 `latest.pt`, and ended with a valid
hash-linked `completed` lifecycle event. The one-run authorisation is
consumed. This is integration and evidence-chain proof only: it is not a
playing-strength result, a throughput benchmark, a node-ladder decision, an
advancement decision, or authority for another smoke or a long run.

## Frozen launch contract

| Field | Value |
| --- | --- |
| Experiment | `dev-v4-sanmill-refereed-fresh-v1` |
| Run | `sanmill-refereed-fresh-v1-smoke-002` |
| Git | `894360d11fc309e5aa58e3289d1c6817831a553c`; clean; `dev == origin/dev` |
| Start | `fresh`; no checkpoint and no automatic resume |
| Algorithm | A2C; PPO and recovery disabled |
| Components | Sentinel, ValueNet, GapNet, S1A warm-start, imitation mixing, S1B refresher, trainer-side opening forcing, and branches explicitly disabled |
| Referee | pinned Sanmill for every complete logical turn |
| Opponents | one Sanmill-search game and one frozen-target game; Sanmill refereed both |
| Search | one fixed 1,000-node ceiling; one thread; no wall-clock limit; no random failure fallback |
| Rollout | two games; `batch_games=1`; depth-5 simulation; `max_ply=120` |
| Schedule | seed 42; temperature start 0.90; frozen-target ratio 0.60; advancement disabled |
| Rules | `nmm-training-core@2`, semantic digest `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| Experiment digest | `sha256:fcfe99652bc2c9123b5a3615a1052371af309df96db68fc88df180289c26f929` |
| MIF evidence | tag `mif-suite-1.0`, Suite JCS `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |

The exact command was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --launch smoke `
  --run-id sanmill-refereed-fresh-v1-smoke-002 `
  --paths-config data\training_paths.local.json `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --start-mode fresh `
  --out-dir out\sanmill-refereed-fresh-v1-smoke-002 `
  --specialist-db data\specialist_db.sanmill_refereed_fresh_v1.smoke-002.sqlite `
  --referee-engine sanmill `
  --opponent-engine sanmill `
  --sanmill-node-ladder 1000 `
  --curriculum-advance-policy disabled `
  --diff-start 1 --diff-max 1 `
  --self-play-ratio 0.60 `
  --seed 42 `
  --max-games 2 --segment-games 2 `
  --max-ply 120 `
  --batch-games 1 `
  --sim-ply-depth 5 `
  --temp-start 0.90 `
  --log-every 1 `
  --max-branches-per-game 0 `
  --minimal-rollouts `
  --no-recovery `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-s1b-refresher `
  --no-imitation-mix --no-opening-forcing
```

## Pre-launch evidence

The final preflight ran after the three repair/evidence commits had been
published. It verified:

- absent output and a separate empty SpecialistDB;
- `quick_check=ok`, `malom_label_version=sector-corrected-v1`, no lineage,
  and 0/0/0 rows in the three training tables;
- pinned Sanmill commit
  `a6623f88959f7453594df274fbe1f128af7ff55e`, tree
  `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`, release binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`,
  and strict-referee semantic digest
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`;
- byte-equivalent first-turn probes from two fresh Sanmill processes; and
- all selected and disabled components in the resolved configuration.

The preflight config SHA-256 was
`a3d7c1cbb20288a860d0fb106f19b0e338392b3219b823ddb7acb687d46217f2`.
The launch manifest includes the launch and run ID and therefore has distinct
config SHA-256
`e765218fee8826d8707d116cc35c61796f1f9059997d4097c2bc2bb6e00bcf06`.
The exact-resume compatibility config SHA-256 is
`2f5cf0d27895d5956b687d782b29e2afea673b25282cea29903f1d060d268f82`.

The corresponding repair verification had already reported 182 trainer,
launch, checkpoint, resume, bridge, and referee tests passed with six
documented historical moving-checkout tests deselected. The mandatory Malom
and label-provenance group reported 103 tests and 498 parameterised subtests
passed. No code changed between those results, publication, final preflight,
and launch.

## Runtime result

The process exited with code zero. The lifecycle interval from
`training_started` to `training_completed` was 24.829461 seconds; the whole
invocation, including its embedded preflight, took about 31.2 seconds.

| Game | Stratum | Learner | Logical plies | Terminal reason | Learner outcome | Sanmill search |
| ---: | --- | --- | ---: | --- | ---: | --- |
| 1 | Sanmill opponent | Black | 43 | `lose_no_legal_moves` | -1.0 | 22 calls, 19,423 actual nodes |
| 2 | frozen target | Black | 52 | `lose_fewer_than_three` | +1.5 | none; Sanmill still refereed |

Terminal reason names describe the losing condition reported by the referee;
they do not imply that the learner was the losing side in both rows. The
second row is a learner win under the trainer's existing weighted return
encoding. Two games carry no strength inference.

The 1,000-node value is a per-search ceiling. The Sanmill game averaged about
882.86 actual nodes over its 22 calls because some positions completed before
the ceiling. Consequently neither 19,423 total nodes nor the elapsed time can
be extrapolated linearly into a retained-run throughput estimate.

The final flush consumed 47 learner steps and recorded one finite optimiser
update:

| Field | Value |
| --- | ---: |
| Policy loss | 0.4442588687 |
| Value loss | 0.8320805430 |
| Entropy | 2.4063904285 |
| Learning rate | 0.00005 |
| `game_count` | 2 |
| `batch_count` | 2 |
| `update_count` | 1 |

The final stored temperature is 0.4625 because this disposable two-game run
evaluates the global schedule against `max_games=2`. It is a smoke-only
schedule consequence, not evidence for retained-run cooling.

The HumanDB warning is expected fail-closed behaviour: its historical Malom
columns lack the required version and remain masked, while documented human
frequencies and outcomes remain available.

## Persisted evidence

The canonical manifest identity is
`0444eb8ac0cfd299864d04211009de9f7c4fd119c5621ff944cf29dbddba2890`.
The lifecycle ledger reloads with the ordered, hash-linked states
`preflight_passed`, `running`, and `completed`.

| Ignored local artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| `run-manifest.json` | 6,947 | `5e9290fe9d529d9a336d8f4d86dac7acfe35131519cad03f097b97f39e97f13b` |
| `run-events.jsonl` | 1,441 | `5afa9a47ce2a9adf39bab2c3258715499e7b5f14a9d1d0a39501480cbe7ab841` |
| `train_log.jsonl` | 2,608 | `51e78a28ff3f2fa7317de19726d2d49c0a6862ce20ed016aa740f230988b046e` |
| `update_log.jsonl` | 169 | `18857d9e279a99196b7ef7e576f99fe5fee6460f7f893126871b150ab721f5ff` |
| `latest.pt` | 2,119,050 | `bc71a10e1c535cfeef16bc711dc106918e7ae326eff96c598cb3b22a3f654e9b` |
| `latest.prev0.pt` | 1,994,259 | `039667d7acb85513d1af215f25ad3363253cbe829aa63669381449e826a59490` |
| `latest.prev1.pt` | 1,435,742 | `ba99da85d80b9d6b9d14d6d95e8b5f9168588fabdd477ea91a01c3470ed1fd3c` |
| final SpecialistDB | 61,440 | `896604e40e7525698b6bc9bfcc16aad23583a0543ec7bf94ded2c385c980e00c` |

All three checkpoint envelopes pass payload integrity and finite-value
validation. The final descriptor records
`sanmill-refereed-fresh-v1-smoke-002:checkpoint:00000003`, `role=latest`,
`save_reason=final`, payload SHA-256
`e6852e019d102e0d0c69ca586b0a15f4dae54e1f9aad70d8b079840e7696b950`,
and the expected run, experiment, and resume-config identities. Optimiser and
component RNG state are present. `best.pt` is correctly absent because the
run did not meet the documented minimum of ten reference-opponent games
(internally retained as the heuristic-history stratum) at a logging checkpoint
with an improved win rate.

The final SpecialistDB passes `quick_check`, retains
`malom_label_version=sector-corrected-v1`, and binds lineage root
`sanmill-refereed-fresh-v1-smoke-002`. It contains 94 positions, zero persisted
Malom labels, one winning line, and zero preferred plays. It is completed
smoke evidence and must not be reused as an empty input for another fresh run.

## Remaining gate

The current long-run verdict is `needs_decision`. This run closes the bounded
training integration gate, but leaves the following deliberately unfrozen:

1. a representative fixed-position, fixed-node throughput envelope;
2. the number and values of Sanmill curriculum levels;
3. the advancement statistic and its treatment of rule draws and truncations;
4. the frozen-target/Sanmill-opponent ratio;
5. process reuse and timeout limits; and
6. retained-run segment, checkpoint, game, and wall-time budgets.

The next safe action is a non-training Sanmill node-throughput calibration.
It must use fixed replayable positions across placement, movement, flying,
and compulsory-removal boundaries, record requested and actual nodes plus
completed depth and latency, separate cold-start from warm-process results,
and never load or update a candidate model.
