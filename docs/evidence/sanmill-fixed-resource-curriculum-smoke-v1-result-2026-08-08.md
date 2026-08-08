# Sanmill Fixed-Resource Curriculum Smoke v1 Result — 8 August 2026

## Verdict

`passed_resource_schedule_route`

The one authorized launch completed all five games at clean published commit
`598c8780dd8b8a80a120988b8d7d0e9daa737058`. It exercised every calibrated
Sanmill node ceiling in the frozen order, produced five finite periodic A2C
updates, published a verified final checkpoint, and ended with a complete
three-event lifecycle chain. The launch authority is consumed.

This is resource-schedule and update-route evidence. All five random fresh
learner games lost to Sanmill; that small, deliberately changing-work sample
is neither an anomaly nor a strength estimate.

## Launch identity

| Field | Value |
| --- | --- |
| Run ID | `sanmill-fixed-resource-curriculum-smoke-v1-20260808-001` |
| NMM_LLM commit | `598c8780dd8b8a80a120988b8d7d0e9daa737058`; clean; `dev == origin/dev` |
| Launch config SHA-256 | `e564297e6f079763aaab158d352213522399d77a74bd42e62c47efece89508a5` |
| Resume-config SHA-256 | `47d3b691575ccfda59f3cff6d84b8b888795f487e41b146274b38d0d38dd02c3` |
| Experiment digest | `sha256:53700537c55cc49f58223e7d6d4ae209c18b3bbf050ac79d5ca15eafc3ad2be9` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Strict-referee digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| Ruleset semantic digest | `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| MIF Suite | `mif-suite-1.0`; JCS `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |

The trainer started at `2026-08-08T10:15:01.596807Z` and recorded completion
at `2026-08-08T10:15:38.559924Z`. The containing command, including its
read-only preflight, returned exit code zero after about 44 wall-clock seconds.

## Scheduled route result

| Global game | Level | Node ceiling | Search calls | Actual nodes | Mean completed depth | Logical plies | Terminal reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 1,000 | 22 | 19,423 | 3.045 | 43 | `lose_no_legal_moves` |
| 1 | 2 | 5,000 | 10 | 31,373 | 4.100 | 19 | `lose_no_legal_moves` |
| 2 | 3 | 25,000 | 17 | 327,103 | 5.471 | 33 | `lose_no_legal_moves` |
| 3 | 4 | 100,000 | 12 | 806,059 | 6.833 | 24 | `lose_no_legal_moves` |
| 4 | 5 | 500,000 | 15 | 5,501,363 | 8.667 | 29 | `lose_no_legal_moves` |

Every row was `vs_sanmill`; no frozen or local-GameAI opponent was sampled.
Each aggregate actual-node count was positive and did not exceed its ceiling
times its search-call count. Sanmill remained referee for every learner and
opponent turn. No game reached the 120-ply experiment truncation.

## Update and checkpoint result

Five `periodic` updates were recorded at games 1 through 5. Their policy loss,
value loss, entropy, learning rate, and batch size were finite. Policy losses
ranged from `0.2879684865` to `0.9108259082`; value losses from `0.4042743444`
to `0.4527369738`; entropy from `1.9286332130` to `2.6988327503`; and batch
sizes from 9 to 21 steps. These ranges are integrity facts, not optimization
quality thresholds.

The final checkpoint verification returned:

```text
checkpoint_id:
sanmill-fixed-resource-curriculum-smoke-v1-20260808-001:checkpoint:00000006

payload_sha256:
53846d7bf59582480ec41588d72d8c93fb42198f6c15aa2710a0412235e8515e

status: verified
```

Its trainer state records game count 5, level 5, and update count 5. All 14
model tensors and all 42 nested optimizer-state tensors were finite. The
checkpoint binds the frozen resume-config, experiment, ruleset, MIF, Sanmill,
HumanDB, Malom, and final SpecialistDB identities.

## Data and evidence identities

| Artefact | SHA-256 |
| --- | --- |
| `latest.pt` | `bb6046dc05ec7b7703284df2a64549965640ae8fb0bab7eecfbae70ab45a8a5a` |
| `train_log.jsonl` | `09ef324c027fb0cef8f8cfb063fa24ddf5eb60c3e966921825a9a146731438df` |
| `update_log.jsonl` | `fd430861c8189b586e9dfb72858bdea1aa2d134bcc4e0595c79866dd9466cc4c` |
| `run-manifest.json` | `cf48c7ae6680a73a3153912e57092178c6b0f901503ea0e027f80a4de5087a56` |
| `run-events.jsonl` | `7f565eefcc0119bdde7f1493021a560a49c50406849eeba303152546d83053ec` |
| final SpecialistDB | `380994ea0fe4b9aff897b2847558fc5ab65d9d6c9ca791328b5d04d4537b181b` |

The final SpecialistDB has `quick_check=ok`, retains
`sector-corrected-v1`, is bound to this run ID, and contains 139 positions,
49 trusted Malom labels, zero winning lines, and zero preferred plays.

The lifecycle chain is exactly:

```text
preflight_passed -> training_started -> training_completed
```

with final event SHA-256
`9fad084f0f1c682da574dce78a47d3bd8c90c55658487acbe31316f8135779bd`.

## Decision

The fixed-resource implementation has passed its required bounded route
smoke. The next gate is to generate and audit an immutable managed plan for
the already frozen 5,000-game / 12-active-hour / 250-game-segment retained
lineage. This smoke output and database remain disposable and must not be used
as that lineage's checkpoint or database.
