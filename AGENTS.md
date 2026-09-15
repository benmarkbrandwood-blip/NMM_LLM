# Repository Working Agreement

## Start Here

The directory containing this file is the only Git repository and must be the
primary Codex workspace. Its parent directory is a data container, not a
repository. Do not initialise an outer repository there.

Confirm the workspace with `git rev-parse --show-toplevel` at session start
and after a workspace switch. Inspect `git status --short --branch` before
changes; inspect the local/remote graph before Git publication or history work.

Read the documents relevant to the task:

- For training, resume, or provenance work, start with the
  [handover status index](docs/handoff/windows-training-2026-07-20.md#current-status-and-decision-index)
  and the selected run's sections. Read its complete owning experiment contract
  before preparing or executing a launch. Historical experiments are not defaults.
- For training input/output contracts and before moving, regenerating, or
  deleting training data, read the applicable inventory, trust-boundary, and
  data-handling sections of
  [`docs/local-training-layout.md`](docs/local-training-layout.md).
- For ordinary code or documentation edits, read the affected module's rules
  and contracts. Full historical handover reading is not a prerequisite.

## Authority and Evidence

- Keep durable repository rules in this file. Put current machine and run state
  in the handover, machine-local data contracts in the local-layout document,
  frozen run decisions in `docs/experiments/`, and target architecture in the
  relevant design or plan document.
- Verify relevant prose against current code, configuration, artefacts, and
  tests. When they disagree, block the affected launch, data mutation, or
  publication until reconciled; continue authorized investigation and local
  repairs. In a read-only review, report the discrepancy without editing.
  Never silently choose the more convenient account or alter a frozen contract
  or acceptance threshold to pass a gate.
- Machine-local narratives and screenshots are historical context, not runtime
  proof. Distinguish those observations from explicitly recorded owner grants.
- For training changes, preserve a traceable chain from requirement to code,
  focused test, and runtime evidence. A screenshot, narrative, or generated
  report is supporting context, not acceptance evidence by itself.

## Standing Delegated Training Authorization

- Launch only when execution is requested or an authorized execution sequence
  is still active, the current instruction does not require read-only work or
  a pause, valid authority covers the operation, and technical gates pass.
  A standing grant alone does not turn a review into execution.
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
  explicitly authorises that operation, including the standing ordinary-push
  grant below. Apply an existing valid grant without requesting it again.
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
- Persisted Malom labels are trusted only when their metadata version is
  `sector-corrected-v1`.
- The imported HumanDB's human frequencies and outcomes are usable, but its
  unversioned historical Malom columns are not training labels.
- Historical specialist, generalist, Sentinel, value-net, and gap-net
  artefacts must retain their recorded provenance. Do not silently describe
  them as retrained after the decoder correction.
- Do not start a long training run until its checkpoint lineage, output
  directory, database paths, and smoke-test result have been recorded.
- Contract-backed Generalist launches prohibit `--auto-resume-best`. A fresh
  launch has no resume checkpoint; weights-only or exact-resume modes require
  an explicit compatible checkpoint and verified output/lineage binding.
  Historical weights-only artefacts cannot satisfy an exact-resume gate.

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
  are launch blockers, not evidence of implicit disablement. Diagnose within the
  authorized scope; fixing code does not authorize retrying counted work.
- For implementation requests, continue through the requested implementation,
  relevant verification, and repairs caused by the change. A first patch or a
  passing preflight is not task completion. Safe local verification with
  disposable fixtures and no production access needs no approval at each step.
- If a protected operation is blocked, complete independent authorized work
  and report the remaining blocker. Ask only for an unresolved user-owned
  decision that prevents progress, after checking existing authority. Never
  treat elapsed time or silence as approval, or resume an explicitly paused
  objective without the owner's instruction.

## Project Skills

- Use `.agents/skills/verify-training-readiness` for a specific run's launch or
  resume readiness, training anomaly diagnosis, or scientific result review.
  Routine progress queries, documentation edits, and ordinary Git work do not
  trigger it. Select its task-specific reference rather than all workflows.
- Long training requires explicit coverage in a direct authorization or an
  existing valid standing grant, plus the owning experiment's launch gate.
  Skill invocation itself grants no execution authority.

## Proportionate Verification

For Malom, DB-teacher, or label-provenance changes, run at least:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py -q
```

Run focused tests for affected behavior and all required experiment gates.
Read-only audits may inspect or run existing safe tests; add regression tests
when implementation is authorized and the defect or behavior needs them.
For documentation-only edits, validate links, metadata, and instruction
consistency without requiring unrelated runtime suites. Once relevant checks
pass, repeat or expand them only for new changes, failures, or unresolved risks.
Report the scope, commit, and limitations of verification. Claim a clean full
suite only after it actually collects and runs cleanly; historical successes
or failures are not a current baseline.
