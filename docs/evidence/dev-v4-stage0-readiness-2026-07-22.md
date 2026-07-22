# Dev v4 Stage-0 Readiness Audit — 22 July 2026

Verdict: **ready_for_long_run**, scoped only to the fixed 212-game Stage-0
evaluation. This is not a training or promotion-readiness verdict.

On 23 July 2026 (Asia/Shanghai), after receiving this readiness result, the
product owner explicitly instructed: `启动 Stage-0 评测` (start the Stage-0
evaluation). That authorizes the exact freeze and run below. It does not
authorize training, a different evaluation contract, promotion, or
publication.

Execution update: the authorized run completed on 23 July 2026. See the
[result evidence](dev-v4-stage0-result-2026-07-23.md). The readiness verdict is
retained as the pre-execution gate, not as a second launch authorization.

## Reviewed launch contract

The proposed action is one deterministic Stage-0 diagnostic on CPU:

- evaluation ID: `dev-v4-formal-paired-eval-v1`;
- 106 unique owner-accepted placement starts;
- one colour-swapped pair per start, for 212 games;
- candidate bundle
  `ab2c8f38570c14ec839d2e516732b22e1c811bf5911843b5eafee9cbaf3fb483`;
- scratch-init control bundle
  `058145238ac03f006779689e14af35eac1d3128921d4b2cddfb698d457fdd86f`;
- `policy-argmax-v1`, zero lookahead rollouts, `max_ply=200`, seed 42;
- Sentinel, ValueNet, GapNet, HumanDB override, and SpecialistDB override
  disabled;
- no training, checkpoint mutation, promotion, or publication.

The candidate bundle was exported from the final version-2 envelope of
`managed-v4-baseline-v1` segment 20. Its frozen training commit is
`9ee3543195255456b2b3832f8371a8f64d25a6af`, and its plan SHA-256 is
`3f696e60c508a972dc42c79f630e90ad20e870001190321a13f0c3a12a4251c1`.
The control is the separately recorded architecture-matched
`scratch-init-v1` bundle; no maintainer-`main` checkpoint enters either side.

CPU is the reviewed device because both bundles verify there, this route has
no search workload, and CPU avoids unnecessary contention with other model
work. A different device requires a new freeze identity and readiness check.

## Gate evidence

The audit ran from repository root `NMM_LLM` on branch `dev`. Its clean code
snapshot was commit `b92d62e23179559085163bf51df8e349796dac33`; the branch was
0 commits behind and 15 commits ahead of `origin/dev`. Being locally ahead is
not an evaluation-integrity failure and does not imply push authorization.
Follow-up commit `381e80e` then closed legacy-unbound execution while retaining
historical loading and recomputation; it does not change the audited corpus,
bundles, or output paths.

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Repository state | Correct repository root; Git tree clean at the audit snapshot | One clean repository, no outer repository | pass |
| Experiment definition | Stage-0 purpose, route, control, fixed workload, non-claims, interval rule, and owner-reviewed corpus are recorded | No unresolved technical experiment parameter | pass |
| Corpus | 106 entries, 106 exact-unique entries; canonical SHA-256 `04bc5782ab79ebeba34d0ff91bcd40fe05e823d539b16ba234b5eedcd123bb9d`; list-file SHA-256 `80c8502f9b4d13ef4aa749200f6b59530de8b5d4e28d8fb10222a15dd40398d7` | One pair per unique start and no deterministic reuse | pass |
| Candidate bundle | Identity matches the contract; model identity `8cf60796a4398fb8588c2d9cc13deedabcb199df15a30e0d1812f71ac0489597`; policy/value canary maximum absolute differences both 0 | Strict manifest, hashes, load, and canary verification | pass |
| Baseline bundle | Identity matches the contract; model identity `93537d0a68f450ea5cbb7cd29e330a2b2253e00330ffee9cff49aa787eba025d`; policy/value canary maximum absolute differences both 0 | Strict manifest, hashes, load, and canary verification | pass |
| Runtime binding | `nmm.paired-runtime.v1`; clean Git commit, Windows/PyTorch identity, CPU identity, float32, `policy-argmax-v1`, disabled components, and zeroed 72-feature lookahead block | Freeze and run must have an identical, explicit runtime identity | pass |
| Output isolation | `stage0-spec.json`, final ledger, partial ledger, and reserved result target are absent; checkpoint evaluation paths are ignored | New isolated targets with no overwrite | pass |
| Focused verification | 28 candidate-lifecycle, corpus, paired-runner, and model-bundle tests passed | All modified evaluation paths green | pass |
| Broader baseline | 927 tests and 498 subtests passed at `08e8c33`; subsequent evaluation changes have focused coverage | No unexplained pre-existing collection/runtime failure | pass |
| Product authorization | Explicit owner instruction on 23 July 2026 to start Stage 0 | Explicit authorization after readiness review | pass |

An ignored `prep-summary.json` predates the final 106-position corpus and
contains rejected-draft paths. It is historical preparation only and is not
launch evidence. None of the reviewed output targets exists.

The runtime identity includes the current Git commit. A prospective spec
identity produced during a no-write probe is therefore intentionally not
recorded as durable evidence: committing this audit changes `HEAD`. The
freeze command below rebuilds and validates the identity from the then-current
clean commit before it verifies bundles or writes the immutable spec.

## Failure and quarantine boundary

There are no permitted silent exceptions. An unclean tree, code/runtime
identity drift, device mismatch, malformed specification, bundle mismatch,
canary failure, legacy unbound runtime, duplicate or reused start, existing
final output, malformed partial ledger, or broken evidence hash chain stops the
run. A valid partial ledger may resume only under the same frozen
specification and runtime. Failed or suspect evidence remains separate for
diagnosis and must not be promoted.

## Reviewed and executed commands

Freeze, after one final clean-state check:

```powershell
.\.venv\Scripts\python.exe scripts\paired_evaluation.py freeze `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\candidate-bundle `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\baseline-bundle `
  docs\experiments\dev-v4-formal-paired-eval-v1-start-positions.json `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-spec.json `
  --evaluation-id dev-v4-formal-paired-eval-v1 `
  --pairs 106 `
  --seed 42 `
  --max-ply 200 `
  --acceptance-margin 0 `
  --rejection-margin 0 `
  --device cpu
```

Run only the spec produced by that command:

```powershell
.\.venv\Scripts\python.exe scripts\paired_evaluation.py run `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-spec.json `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\candidate-bundle `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\baseline-bundle `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-games.jsonl `
  --device cpu
```

The immutable JSONL ledger is the primary evidence. Its result can be
recomputed without loading either model:

```powershell
.\.venv\Scripts\python.exe scripts\paired_evaluation.py recompute `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-spec.json `
  learned_ai\checkpoints\evaluation\dev-v4-formal-paired-eval-v1\stage0-games.jsonl
```

## Authorization boundary

There are no unresolved launch decisions for this exact CPU Stage-0 run. The
authorization is consumed by one immutable specification and its resumable
212-game ledger. Any change to corpus, bundle, route, device, workload,
threshold, or output identity requires a new readiness review and explicit
authorization.
