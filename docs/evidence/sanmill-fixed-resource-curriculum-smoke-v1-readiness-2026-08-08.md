# Sanmill Fixed-Resource Curriculum Smoke v1 Readiness — 8 August 2026

## Verdict

`ready_for_one_authorized_smoke`

The exact read-only command in the
[smoke contract](../experiments/sanmill-fixed-resource-curriculum-smoke-v1.md)
returned `ready_for_smoke` with `errors=[]` and
`unresolved_decisions=[]` at clean published commit
`63121b6437601bf01c6cca2b03f5a7d421453c9c`. The output directory remained
absent and the isolated SpecialistDB remained byte-identical to its initial
state.

The delegated technical authority permits this readiness record to be
published and the command to run once only after the same preflight passes
again from the clean commit containing this record. A failure consumes that
authority and stops the route.

## Frozen identities

| Field | Identity |
| --- | --- |
| NMM_LLM implementation | `502248aa0b55ae0e9924f6e604d9d1ce91501a1b` |
| NMM_LLM readiness source | `63121b6437601bf01c6cca2b03f5a7d421453c9c`; clean; `dev == origin/dev` |
| Preflight config SHA-256 | `11b376a60196013483a0b73d43870e03108a24408e0542b71a860a5ecb3fdd8d` |
| Resume-config SHA-256 | `47d3b691575ccfda59f3cff6d84b8b888795f487e41b146274b38d0d38dd02c3` |
| Experiment digest | `sha256:8efe1982f8ff32813e11b3b638bff0ebb36db2df61e42ed8c359d9db9d102424` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Strict-referee digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| Ruleset semantic digest | `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` |
| HumanDB content identity | `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31` |
| Malom dataset identity | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Initial SpecialistDB SHA-256 | `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d` |
| MIF Suite tag | `mif-suite-1.0` |
| MIF Suite JCS SHA-256 | `sha256:81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f` |

The runtime probe used two fresh Sanmill processes and returned identical
semantic first-turn observations at 1,000 nodes. Strict failure policy,
origin-counted `mif-stable-moving-v1`, one thread, disabled shuffling, fixed
seed, fixed nodes, and all prohibited DB/patch/random fallback sources were
confirmed.

## Configuration review

The resolved command binds:

- fresh random initialization, seed 42, A2C, and `batch_games=1`;
- five total and segment games, one per resource stage;
- levels `1,2,3,4,5` mapped to
  `1,000,5,000,25,000,100,000,500,000` nodes;
- `self_play_ratio=0`, so all five games exercise Sanmill search while
  Sanmill referees every turn;
- `max_ply=120`, simulation depth 5, and `update_every=8`;
- no branch, recovery, Sentinel, ValueNet, GapNet, warm-start, refresher,
  imitation mix, or opening forcing; and
- a new unbound, empty, trusted SpecialistDB and absent output directory.

The global-game schedule and node ladder participate in the resume-config
hash. The implementation accepts a preceding level only at an exact stage
boundary and fails closed on any other persisted-level disagreement. The
fixed-resource policy never samples a lower node level.

## Verification

The current implementation passed:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_generalist_managed_controls.py `
  tests\test_generalist_preflight.py `
  tests\test_generalist_launch.py `
  tests\test_generalist_run_manifest.py `
  tests\test_manage_generalist_run.py `
  tests\test_managed_generalist.py `
  tests\test_checkpoint_envelope.py `
  tests\test_resume_parity.py `
  tests\test_train_s_gen_v2_checkpoint_v2.py `
  tests\test_train_s_gen_v2_checkpoints.py `
  tests\test_train_s_gen_v2_curriculum.py `
  tests\test_train_s_gen_v2_game_identity.py `
  tests\test_train_s_gen_v2_observability.py `
  tests\test_train_s_gen_v2_segment_stop.py `
  tests\test_train_s_gen_v2_updates.py `
  tests\test_sanmill_training_referee.py -q
```

Result: `143 passed`.

The mandatory Malom and label-provenance command passed `103` tests and `498`
parameterized subtests. Ruff passed on the changed preflight and test files;
the trainer passed the focused undefined-name check. The trainer retains its
documented pre-existing whole-file style baseline, which was not rewritten as
part of this semantic change. `git diff --check` passed.

## Remaining gate

After this record is committed and pushed, rerun the exact read-only command
from the new clean published commit. Only another `ready_for_smoke` verdict
with unchanged input identities authorizes replacing `--preflight smoke` with
the contract's one `--launch smoke` invocation. Success must be audited and
committed before a managed retained plan is created.
