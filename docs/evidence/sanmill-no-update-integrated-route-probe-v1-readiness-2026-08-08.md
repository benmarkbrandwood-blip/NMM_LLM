# Sanmill No-Update Integrated-Route Probe v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

This verdict applies only to the bounded 36-game, no-update route probe. It is
not a training-smoke verdict, long-run readiness, strength evidence, or launch
authority. The machine preflight returned `ready_for_authorized_probe` and
explicitly returned `launch_authorized=false`.

The preflight ran from clean, published implementation commit
`70fcd3c4d3e065b326faa3cbbd96f33775302ddc`, with
`dev == origin/dev`. This readiness record is a later documentation-only
change. Before launch, that evidence commit must be published and the same
preflight must pass again from the final clean published `dev` tip. The owner
must then provide separate one-run authority; silence or a timeout cannot
supply it.

## Gate result

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Git source | clean published `dev` at `70fcd3c`; tree `9fd30fe`; upstream equal | clean implementation already on `origin/dev` | pass |
| Plan | identity `7aa079d`; raw SHA-256 `b422ea4`; 36 immutable schedule entries | tracked content-addressed plan | pass |
| Start and learning | fresh seed-42 model; no checkpoint; no optimiser; `requires_grad=false` | no update-capable state | pass |
| Model identity | learner and frozen target both `15106b6`; no mutation during preflight | deterministic fresh initialization | pass |
| HumanDB | closed immutable snapshot; SHA-256 `97be715`; `quick_check=ok`; no sidecars | read-only human-frequency source | pass |
| SpecialistDB | empty corrected snapshot; SHA-256 `b4d522d`; `quick_check=ok`; no sidecars | read-only empty `sector-corrected-v1` source | pass |
| Malom | 512 components; manifest `f4c52b0`; anchor `5078bf8`; corrected trust | exact corrected read-only source | pass |
| Rules | `nmm-training-core@2`; semantic digest `52f6ad2`; document digest `1dfdf57` | frozen training rules | pass |
| Sanmill | commit `a6623f8`; tree `17b9b0f`; binary `5fbf3cb`; runtime identity `705eabc` | exact clean strict runtime | pass |
| Referee | `mif-stable-moving-v1`; semantic digest `1b2b88c`; strict failure and no fallback | authoritative complete-history route | pass |
| Device | CUDA 12.9; PyTorch 2.8.0; RTX 4090; 25,756,696,576 bytes | intended CUDA device available | pass |
| Route check | two logical plies; Sanmill count two; zero search calls; max-ply truncation | unscheduled no-search production route | pass |
| Bounded work | 36 games; 4,320 plies; 1,800 searches; 227,160,000 requested-node maximum | plan hard bounds | pass |
| Launch | `launch_authorized=false` | explicit later one-run authority | pending |

Short hashes in the table are display abbreviations of the full identities
below; they are not alternate identities.

## Frozen identities

| Field | Value |
| --- | --- |
| NMM_LLM implementation commit | `70fcd3c4d3e065b326faa3cbbd96f33775302ddc` |
| NMM_LLM tree | `9fd30fea151b4cd177472966a96dcb362317e6f1` |
| Plan identity | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` |
| Plan raw SHA-256 | `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Fresh model state | `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00` |
| HumanDB SHA-256 | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c` |
| SpecialistDB SHA-256 | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f` |
| Malom manifest SHA-256 | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Malom anchor SHA-256 | `5078bf84505fe2845a4af7c36907efa2d66b2eb76f149ce12faa248117405b68` |
| Rules semantic digest | `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| Rules document digest | `sha256:1dfdf5777f36866a53a942c1addd21857d3b72eede8ea2bf4fe1beedfbe878f2` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Referee semantic digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The HumanDB snapshot contains 94,429 games, 2,152,889 positions, and
2,516,356 moves. Its historical Malom columns remain disabled because it has
no trusted label-version metadata; only documented human frequencies and
outcomes are usable. The empty SpecialistDB has zero positions, preferred
plays, and winning lines, and declares
`malom_label_version=sector-corrected-v1`. Neither database had a WAL or SHM
sidecar before or after preflight.

## Exact preflight and reviewed launch command

The passing read-only command was:

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route.py `
  --preflight `
  --plan docs\experiments\sanmill-no-update-integrated-route-probe-v1.json `
  --paths-config data\training_paths.local.json
```

The next launch command is reviewed but not authorized or executed:

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route.py `
  --launch probe `
  --plan docs\experiments\sanmill-no-update-integrated-route-probe-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-probe-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-probe-v1-20260808-001.json
```

The proposed output and its `.partial` and failure variants were absent during
this audit. Absence must be checked again immediately before any authorized
launch.

## Verification

The published source passed:

- 62 focused probe, rollout-control, referee, checkpoint, resume,
  temperature, update, and calibration tests;
- 103 mandatory Malom, DB-teacher, and label-provenance tests, with 498
  parameterized subtests;
- Ruff over the new probe, runner, focused tests, and adjacent clean training
  seams; and
- `git diff --check`.

The first focused invocation used pytest's inaccessible machine-global temp
root and therefore produced setup-only `WinError 5` errors for `tmp_path`
tests; no assertion failed. The unchanged suite passed after assigning a new
ignored repository-local `--basetemp` and disabling pytest's unwritable cache.
No test was skipped, deleted, or weakened.

Source inspection and a focused guard confirm that the probe imports neither
the MIF reference runner nor copied gameplay logic. It calls the production
NMM_LLM rollout, Sanmill referee, and Sanmill opponent seams.

## Remaining gates

1. Publish this documentation-only evidence commit by ordinary fast-forward.
2. Repeat the exact preflight from that final clean published commit.
3. Obtain explicit authority for exactly the reviewed one-run command.

Until all three occur, the 36 planned games remain unconsumed. Training,
candidate evaluation, node-ladder adoption, advancement design, and long-run
launch remain unauthorized.
