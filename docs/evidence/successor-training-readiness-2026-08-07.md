# Successor Training Readiness — 7 August 2026

## Verdict

`needs_decision`

The implementation and local assets are ready for a new bounded smoke, but no
smoke or long-run launch is authorized. A long run must not start before the
smoke reaches a real policy update and its evidence is audited.

This record does not authorize a push, plan, authorization file, smoke,
training segment, evaluation, or promotion.

## Fixed Technical Boundary

- Fresh random initialization; no maintainer or historical checkpoint.
- A2C; PPO disabled.
- Sentinel, ValueNet, GapNet, S1A warm-start, RL imitation mixing, S1B
  refresher, and opening forcing disabled explicitly.
- One process, one CUDA device, `batch_games=1`.
- Heuristic opponents use one thread and a fixed 500,000-node ceiling; no
  wall-clock search budget.
- MIF release tag `mif-suite-1.0`, release commit
  `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, and Suite JCS SHA-256
  `81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`.
- Ruleset `nmm-training-core@2`, semantic digest
  `52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`.
- Rule draws are distinct from `max_ply` experiment truncation.

## Gate Audit

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Git branch | `dev`, tracked worktree clean, local commits not yet on `origin/dev` | Clean and remotely recoverable launch commit | `needs_decision` for push |
| Maintainer `main` | Fetched and audited through `bc46b51e`; no wholesale cherry-pick accepted | Every adopted change independently justified and tested | pass |
| Protocol identity | Final immutable MIF release is bound by run manifest code | Final release tag, commit, Suite, evidence, and release-manifest hashes | pass |
| Rules identity | MRS v2 digest matches trainer behavior | Exact supported MRS; no floating prose-only rules | pass |
| Malom | 512 components, 83,582,223,577 bytes, manifest identity verified | Corrected complete source and matching manifest | pass |
| HumanDB | Present; `quick_check=ok`; historical Malom columns masked | Frequencies/outcomes only, no unversioned labels | pass |
| Previous SpecialistDB | 132,182 positions; bound to completed managed run | Must never seed another fresh run | pass by rejection |
| Successor SpecialistDB | New ignored 45,056-byte DB; zero rows; `sector-corrected-v1`; no lineage | Empty, trusted, isolated | pass |
| Output | Proposed smoke directory absent | New isolated directory | pass |
| Components | All excluded components disabled explicitly | No missing component treated as implicit disablement | pass |
| Checkpoint lineage | `start_mode=fresh`; no resume flags | No historical or automatic resume | pass |
| Hardware | PyTorch 2.8.0+cu129, one RTX 4090, native core available | One CUDA device and native fixed-node search | pass |
| Storage | About 319.9 GB free on the data volume | Enough for smoke and successor evidence | pass |
| Focused tests | 132 readiness tests plus 103 Malom/provenance tests and 498 subtests | Modified subsystems and mandatory provenance suite green | pass |
| Smoke evidence | Not launched | At least one real update; imitation absent; identities and output audited | pending authorization |
| Long-run plan | Not created | New objective, game/wall envelope, truncation, commit, DB, output, and authorization | pending smoke and product decision |

## Read-only Preflight Actually Run

The following exact command was reviewed and executed read-only. The
`successor-readiness-only-v1` experiment name is a probe identity, not a frozen
product objective. The value 120 is a proposed smoke ceiling only; it was not
selected by timeout or recorded as a long-run decision.

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --preflight smoke `
  --experiment-id successor-readiness-only-v1 `
  --start-mode fresh `
  --paths-config data\training_paths.local.json `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2_successor_readiness_v1 `
  --specialist-db data\specialist_db.successor_readiness_v1.sqlite `
  --ruleset-manifest data\rulesets\nmm-training-core@2.json `
  --no-sentinel --no-value-net --no-gap-net `
  --no-s1a-warmstart --no-imitation-mix --no-s1b-refresher `
  --no-opening-forcing `
  --max-games 2 --segment-games 2 --batch-games 1 `
  --max-ply 120 --max-ply-branch 120 `
  --max-branches-per-game 0 --sim-ply-depth 5 `
  --heuristic-node-budget 500000 `
  --seed 42 --temp-start 0.90 `
  --self-play-ratio 0.50 --update-target-every 50 --log-every 50
```

The preflight returned `ready_for_smoke`, zero errors, SpecialistDB counts
0/0/0, isolated absent output, and no checkpoint source. It did not create an
output directory, model, run manifest, authorization, or training event.

## Product Decisions Still Required

The recommended next objective is a fresh, rules-corrected v4 successor smoke
whose sole purpose is to prove the hardened route performs a real update and
persists the final MIF, ruleset, experiment, component, and database lineage.
The recommended smoke envelope is two games, `max_ply=120`, and a two-hour
external safety limit. This is deliberately not long-training authorization.

After the smoke passes, the owner must separately choose the long-run game or
wall-time envelope and authorize launch. A 5,000-game successor is the most
direct comparison with the completed baseline; the completed 5,000-game run
used 1.9069 hours at the historical 60-ply cap, so a 12-hour ceiling would be
a conservative initial envelope for a 120-ply successor. That estimate is an
inference and must be replaced by the new smoke's measured throughput before a
plan is frozen.

The current local `dev` must also be pushed before retained training so the
exact launch commit is remotely recoverable. Push and launch require explicit
owner approval; neither is inferred from this readiness work.

## External Rule-parity Follow-up

Sanmill strict UCI currently appears to count only stable moving states reached
after a move, while `stable-moving-v1` counts a supplied stable moving origin.
This does not block the self-contained NMM_LLM smoke, but formal Sanmill-referee
parity remains open. See
[`main-integration-audit-2026-08-07.md`](main-integration-audit-2026-08-07.md)
for the exact action-8/action-9 observation.
