# Repository Working Agreement

## Start Here

The directory containing this file is the only Git repository and must be the
primary Codex workspace. Its parent directory is a data container, not a
repository. Do not initialise an outer repository there.

Before changing code or Git history:

1. Run `git rev-parse --show-toplevel` and confirm that it returns this
   repository.
2. Read the current handover in
   [`docs/handoff/windows-training-2026-07-20.md`](docs/handoff/windows-training-2026-07-20.md).
3. Read [`docs/local-training-layout.md`](docs/local-training-layout.md)
   before moving, regenerating, or deleting any training data.
4. Inspect `git status --short --branch` and the local/remote commit graph.

## Authority and Evidence

- Keep durable repository rules in this file. Put current machine and run state
  in the handover, machine-local data contracts in the local-layout document,
  frozen run decisions in `docs/experiments/`, and target architecture in the
  relevant design or plan document.
- Verify prose against current code, configuration, artefacts, and tests. When
  they disagree, stop and reconcile the document that owns the disputed fact;
  do not silently choose the more convenient account.
- For training changes, preserve a traceable chain from requirement to code,
  focused test, and runtime evidence. A screenshot, narrative, or generated
  report is supporting context, not acceptance evidence by itself.

## Standing Delegated Training Authorization

- The product owner may explicitly authorize one bounded, preregistered
  sequence instead of answering once per plan or arm. Record that standing
  delegation in the owning experiment document or current handover. It must
  identify the objective or plan family, aggregate game and wall-time bounds,
  permitted order and operations, claim boundary, stop conditions, prohibited
  actions, and expiry or revocation condition.
- When a standing delegation exactly covers an immutable child plan, the Agent
  may create its per-plan `authorization.json` just in time and launch it after
  all technical gates pass. Record `product-owner-delegated-agent` as the
  operator and cite the standing delegation in the decision note. Do not ask
  the product owner to approve each seed, arm, segment, node count, learning
  rate, or other technical choice already inside that envelope.
- A failure remains fail closed. Diagnose it autonomously, but do not retry a
  counted run unless the standing delegation explicitly permits a
  semantics-identical recovery and the recovery path is proven safe.
- Standing delegation never silently expands its scope. A new objective,
  larger aggregate game or wall-time budget, long training not explicitly
  named by the grant, held-out evaluation, model promotion, publication,
  release, destructive cleanup, Git history rewrite, or external side effect
  still requires the applicable explicit authority.
- General requests for autonomy are not an unlimited training grant. If the
  required objective or aggregate resource envelope is absent, ask once for
  the parent decision, never once per technical child plan. The product owner
  may revoke a standing delegation at any time.

## Git Safety

- Do not use a blind `git pull` to resolve rewritten but patch-equivalent
  commits. Establish the commit graph and patch equivalence first.
- Do not push, force-push, merge, rebase, or rewrite history unless the user
  explicitly authorises that operation.
- The product owner granted standing authority on 11 August 2026 for Codex to
  make an ordinary fast-forward push of its own verified commits from local
  `dev` to `origin/dev` without asking again. Before relying on this grant,
  fetch the remote, confirm the active branch is `dev`, confirm `origin/dev`
  is an ancestor of local `dev`, preserve unrelated changes, and verify the
  commits being published. This grant does not cover force-push, merge,
  rebase, amend, or any other history rewrite, and the owner may revoke it at
  any time.
- Preserve unrelated user changes and ignored local training artefacts.
- Keep one independently justified fix per commit.
- Write commit subjects and bodies in English. Unless the user asks for a
  different format, wrap commit-message body lines at 72 ASCII characters.

## Local Paths and Large Data

- Machine-specific paths belong in `data/training_paths.local.json`. This
  file is ignored and must not be committed.
- Keep the Malom tablebase and source archives outside the repository.
- Keep imported SQLite databases, recursive game records, endgame tables,
  generated checkpoints, and backup snapshots ignored unless the repository
  already contains an explicit tracked exception.
- Never overwrite or relabel the isolated legacy SpecialistDB. Its location
  and checksum are recorded in the local-layout document.

## Malom and Training Safety

- Treat project rules and the repository's independently tested semantics as
  authoritative. Sanmill is a useful reference implementation, not a reason
  to bypass analysis of this codebase.
- Treat machine-local handover notes and screenshots as historical context,
  not as authoritative facts or acceptance evidence. Verify their claims
  against current code, configuration, artefacts, and tests before acting.
- Persisted Malom labels are trusted only when their metadata version is
  `sector-corrected-v1`.
- The imported HumanDB's human frequencies and outcomes are usable, but its
  unversioned historical Malom columns are not training labels.
- Historical specialist, generalist, Sentinel, value-net, and gap-net
  artefacts must retain their recorded provenance. Do not silently describe
  them as retrained after the decoder correction.
- Do not start a long training run until its checkpoint lineage, output
  directory, database paths, and smoke-test result have been recorded.
- In particular, review the known `--auto-resume-best` path issue documented
  in the handover before starting the generalist trainer.

## Change and Diagnosis Discipline

- Treat answer, review, and diagnosis requests as read-only unless the user
  also asks for implementation.
- Before fixing a defect, establish the smallest deterministic reproduction or
  focused test that is capable of failing for the reported reason.
- Keep diagnostic hypotheses falsifiable, record decisive evidence, and remove
  temporary instrumentation after the cause is understood.
- Never obtain a green result by deleting, skipping, or weakening tests;
  swallowing required errors; substituting empty or mock data; or turning a
  required component failure into a neutral/default value.
- A component is absent only when the selected experiment explicitly disables
  it. Missing files, incompatible checkpoints, and unavailable required data
  are stop conditions, not implicit disablement.

## Project Skills

- Use `.agents/skills/verify-training-readiness` when preparing, reviewing,
  resuming, smoke-testing, or starting a training run. It is a preflight and
  evidence workflow; invoking it does not itself authorise a training launch.
  Before requesting authority, check whether a recorded standing delegation
  already covers the exact immutable plan.
- Long-running training still requires an explicit user request and the launch
  gate recorded by the owning experiment document.

## Proportionate Verification

For Malom, DB-teacher, or label-provenance changes, run at least:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py -q
```

Also run focused tests for every modified subsystem. Do not present the full
test suite as a clean baseline until its known missing/stale internal-interface
collection failures have been resolved.
