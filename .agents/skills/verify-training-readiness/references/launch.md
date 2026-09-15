# Launch and Resume Readiness

Use this workflow for a specific proposed launch or resume. A read-only review
reports findings and missing decisions without creating plans, authorization
files, or training outputs. Preparation and execution writes must be within
the user's requested scope and repository authority.

## Bind the Selected Contract

Read the complete owning experiment contract and relevant current handover
sections. Confirm the repository root, commit, worktree state, and local/remote
graph required by that contract. Inspect the target entry point and its tests;
historical command examples are evidence to verify, not reusable launch commands.

Determine the selected start mode and smoke or long-run purpose from the task
and contract. For the Generalist, distinguish fresh, weights-only, and
exact-resume semantics. Resolve routine technical choices inside an existing
delegation; ask only when an unresolved owner decision changes lineage or scope.

## Resolve Inputs Read-Only

- Derive precedence from the selected entry point. Report effective logical
  keys and their sources: CLI, environment, local config, or code default.
  Do not copy machine-specific absolute values into tracked files or ordinary
  chat output. Read the applicable local-layout inventory and data rules.
- Check required input existence, type, size, and lineage identity. Hash
  databases or checkpoints when the owning contract requires it; do not hash
  multi-gigabyte files speculatively. Required resource checks must fit the
  authorized operation budget.
- Open SQLite audit inputs read-only using the documented reader and snapshot
  requirements. Perform required integrity checks and report relevant metadata
  and counts. Apply AGENTS.md's persisted-label trust boundary.
- Verify Malom through the path the target process will resolve. A missing
  required tablebase cannot be replaced by a substitute or an implicit disable.
- Inspect the selected resume checkpoint: stage, counters, source lineage,
  feature/config compatibility, and experiment-specific metadata. A console
  message does not establish file existence or exact-resume compatibility.
- For fresh runs, verify absent resume flags and a dedicated output directory
  without historical checkpoints or logs. For resumes, verify the intended
  output/checkpoint binding and the exact-resume state required by the contract.

## Freeze and Verify the Launch

For authorized preparation, record the command, commit, seed, device, logical
output/database paths, component switches, update algorithm, opponent and
temperature schedules, game/batch budgets, concurrency, checkpoint and monitoring
cadence, and stop criteria. On review, verify these records without rewriting
them. A missing component is not an explicit disable decision.

Every consequential long-run choice must be frozen before launch. Do not invent
a value to pass a gate or change an immutable plan in place. Bounded disposable
smoke choices must remain within their authority and state which long-run
decisions they do not establish.

Apply AGENTS.md's proportionate verification rules and every test required by
the selected experiment. Check path resolution, resume logic, scheduling, and
component switches where relevant. Existing safe tests can establish review
evidence; add regression tests only for authorized implementation. Report
unrelated failures separately without claiming a clean full suite.

## Apply Existing Authority

Use AGENTS.md's standing-delegation policy as the authority source. Verify the
exact plan identity, allowed operations/order, budget and aggregate consumption,
prerequisite children, expiry/revocation, and unconsumed authorization.

Within active authorized execution, an absent leaf authorization.json is not
by itself an owner decision when a valid standing grant covers the child. After
technical gates pass, create the ordinary plan-bound authorization just in time
with operator product-owner-delegated-agent and cite the grant. Do not ask again
for technical details already inside its envelope.

When actual execution requires an unresolved parent objective, resource envelope,
or scope expansion, finish independent preparation and present one concrete
decision with the proposed bounds. Silence or a timeout is not authorization.
For read-only reviews, report authority as missing without demanding approval
for an execution the user did not request. Git publication uses the separate
AGENTS.md Git grant; training authority does not cover it.

On an anomaly, stop the affected sequence and diagnose within scope. Preserve
attempt consumption and immutable evidence. Recovery of counted work requires
explicit coverage and proof of semantics-identical safety, even after a fix.

## Report the Gate and Continue the Task

Use the selected runner's documented verdict contract. Where these existing
readiness labels apply, keep their meanings:

- `ready_for_smoke`: bounded-smoke gates pass; no long-run approval implied.
- `ready_for_long_run`: required tests and launch gates pass, with consequential
  long-run choices frozen.
- `needs_decision`: a user-owned lineage or experiment choice remains unresolved.
- `fatal_stop`: required data, provenance, compatibility, output isolation,
  tests, or repository state fails the documented contract.

Summarize gate, observed, expected, and result, plus authority status,
unresolved decisions, and the reviewed command if one exists. Never invent a
launch command for a blocked proposal. Do not change persisted runner schemas
or verdicts merely to fit this presentation.

A review ends with that evidence. Active execution continues after a passing
gate only when current task scope and valid authority also permit it. A failed
gate blocks the protected operation, not independent authorized diagnosis or
repairs; it does not authorize a counted retry.
