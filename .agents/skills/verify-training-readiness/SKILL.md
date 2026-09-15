---
name: verify-training-readiness
description: Check launch or resume readiness for a specific NMM_LLM run, or review its training anomalies and scientific results. Excludes routine progress queries, documentation edits, and ordinary Git work.
---

# Verify Training Readiness

Assess the selected run using its actual contract and artifacts. Repository
policy lives in [AGENTS.md](../../../AGENTS.md), including authorization,
data protection, label trust, Git publication, and required verification.

## Choose the Workflow

- **Launch or resume readiness:** read [launch.md](references/launch.md).
  Use it for preflight review and for the gate inside an authorized execution
  task. Readiness review alone does not launch anything.
- **Training anomaly or scientific result:** read
  [analysis.md](references/analysis.md). Use existing evidence to answer the
  question; do not demand a new launch plan or approval to complete analysis.
- **Routine progress:** answer from the selected run's existing status or logs.
  Do not load either reference unless an anomaly or scientific claim needs it.

For a combined request, read each reference only when that part becomes
relevant. Select the experiment from the user's requested run and the
[handover status index](../../../docs/handoff/windows-training-2026-07-20.md#current-status-and-decision-index),
not from an old example. Read relevant status sections and the owning contract;
do not load the entire historical handover by default.

## Scope and Completion

A current read-only or pause instruction takes precedence over any standing
execution grant. For active execution, apply AGENTS.md's authority checks;
report passing preflight in a progress update and continue without another
confirmation when all technical and authority gates pass.

Fail closed for the affected protected operation. Continue independent work
already authorized, including diagnosis and requested local repairs. Do not
weaken contracts, ignore required data, or retry counted work without coverage.
Use existing safe tests for reviews; new code or tests require implementation
scope. A ready verdict is not completion of a requested implementation or run.

Finish an analysis with the supported answer and evidence limits. Finish an
execution task with its required outcome and verification, or the precise
remaining blocker after completing independent authorized work.
