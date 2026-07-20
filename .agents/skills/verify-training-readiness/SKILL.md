---
name: verify-training-readiness
description: Verify whether an NMM_LLM training smoke, long run, or resume is safe to launch by checking Git state, experiment decisions, resolved paths, data and checkpoint provenance, output isolation, focused tests, and the exact launch contract. Use when preparing, reviewing, resuming, diagnosing, smoke-testing, or starting a training run, or when asked whether training is ready.
---

# Verify Training Readiness

## Overview

Audit the selected training run against repository reality and emit a concise,
evidence-backed verdict. Keep the audit read-only unless the user separately
authorises a smoke or training launch.

## Establish the Contract

1. Read `AGENTS.md`, the current Windows handover, and
   `docs/local-training-layout.md` completely.
2. Read the experiment document that owns the proposed run. For the current
   corrected Generalist baseline, use
   `docs/experiments/dev-v4-malom-corrected-baseline.md`.
3. Confirm the repository root, current commit, worktree state, and local/remote
   graph. Do not assume that a documented commit or clean tree is still current.
4. Inspect the actual target entry point and tests. Treat command examples and
   historical notes as claims to verify, not executable truth.
5. Classify the request as `fresh` or `resume`, and as `smoke` or `long_run`.
   If that intent is ambiguous and changes lineage, report `needs_decision`.

## Resolve Inputs Read-Only

- Derive configuration precedence from the selected entry point. Report the
  effective logical keys and which source supplied each value: CLI, environment,
  local path config, or code default. Do not copy host-specific absolute paths
  into tracked files or ordinary chat output.
- Check required files for existence, type, size, and identity where lineage
  matters. Hash source databases or checkpoints when the owning documentation
  requires it; do not hash multi-gigabyte files speculatively.
- Open SQLite inputs through a read-only URI for audit queries. Run integrity
  checks and report schema or metadata versions plus relevant row counts.
- Trust persisted Malom labels only when metadata is
  `malom_label_version=sector-corrected-v1`. HumanDB frequencies and outcomes
  remain usable when documented, but unversioned historical Malom columns do
  not become labels.
- Verify Malom availability through the path the target process will actually
  resolve. Do not create a substitute database or silently accept a missing
  tablebase.
- For a resume, inspect the exact checkpoint that will be selected, including
  stage, counters, source checkpoint, feature/config compatibility, and any
  experiment-specific metadata. A console message is not proof that a file
  exists.
- For a fresh run, verify that resume flags are absent and the dedicated output
  directory has no historical checkpoints or logs. For a resume, verify that
  the output directory and checkpoint lineage intentionally match.

## Freeze the Launch Contract

Record the exact command, commit, seed, device, output and database logical
paths, enabled and disabled components, update algorithm, opponent schedule,
temperature schedule, game and batch budgets, concurrency, checkpoint cadence,
monitoring cadence, and stop criteria. A missing component is not equivalent to
an explicit `--no-*` decision.

For a long run, every experiment-owned choice must already be frozen in its
experiment document. Do not infer a consequential value just to make the gate
pass. A smoke may use bounded disposable values, but it must state which
long-run decisions it does not approve.

## Run Proportionate Verification

1. Run the mandatory Malom, DB-teacher, and provenance tests from `AGENTS.md`
   whenever those seams are involved.
2. Add focused tests for the selected trainer's path resolution, resume logic,
   scheduling, and component switches as applicable.
3. When diagnosing a defect, first establish a deterministic reproduction or
   focused test capable of going red for that defect.
4. Report known unrelated collection failures separately. Never describe the
   full suite as clean when it did not collect or run cleanly.
5. Do not delete, skip, weaken, or rewrite assertions merely to obtain green.

## Issue the Verdict

Use exactly one verdict:

- `ready_for_smoke`: all bounded-smoke gates pass; no long-run approval implied.
- `ready_for_long_run`: all tests and experiment-specific launch gates pass and
  every consequential long-run choice is frozen.
- `needs_decision`: evidence is sound but a user-owned lineage or experiment
  choice remains unresolved.
- `fatal_stop`: required data, provenance, compatibility, output isolation,
  tests, or repository state fails the documented contract.

Summarise evidence in a compact table with `gate`, `observed`, `expected`, and
`result`, followed by unresolved decisions and the reviewed exact command.
Separate facts from inferences. Do not turn an audit request into a launch. If
the user explicitly requested launch, proceed only after reporting a passing
gate; otherwise stop with the failed or unresolved condition.
