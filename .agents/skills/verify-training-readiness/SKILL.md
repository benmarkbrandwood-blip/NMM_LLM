---
name: verify-training-readiness
description: Verify and analyze an NMM_LLM training smoke, long run, resume, or completed result by checking Git state, experiment decisions, resolved paths, data and checkpoint provenance, output isolation, focused tests, the exact launch contract, learning curves, seeds, hyperparameters, baselines, ablations, and disaggregated metrics. Use when preparing, reviewing, resuming, diagnosing, smoke-testing, starting, monitoring, or interpreting a training run, or when asked whether training is ready, healthy, successful, or worth promoting.
---

# Verify Training Readiness

## Overview

Audit the selected training run against repository reality and emit a concise,
evidence-backed verdict. Keep the audit read-only unless the user separately
authorises a smoke or training launch directly, or a recorded standing
delegation already covers the exact immutable plan.

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
6. Locate any direct authorization or standing delegated authorization. A
   standing grant is valid only when its objective or plan family, aggregate
   resource envelope, allowed operations and order, claim boundary, stop and
   prohibited conditions, and expiry are all recorded and cover the exact
   child plan.

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

## Resolve Launch Authority Without Repeated Prompts

- Technical readiness and launch authority are separate checks.
- If a direct plan authorization exists, verify its plan hash and bounds.
- If a valid standing delegation covers the exact plan, do not return
  `needs_decision` merely because the leaf `authorization.json` is absent. Run
  preflight, verify prerequisite children and aggregate consumption, create the
  ordinary plan-bound authorization just in time with
  `product-owner-delegated-agent`, and continue without asking again.
- Do not ask the product owner to choose seeds, arms, learning rates, node
  budgets, transition counts, checkpoint cadence, or other technical details
  already inside the recorded envelope.
- On an anomaly, stop the sequence and diagnose it. Do not automatically retry
  counted work unless the grant explicitly covers a proven
  semantics-identical recovery.
- Ask once, without a timeout or default, only when the parent objective or
  aggregate resource envelope is missing or when the requested action expands
  scope. Long training, held-out evaluation, promotion, publication, release,
  destructive cleanup, and Git history rewriting require explicit coverage.
- A user's request for autonomy is not by itself evidence of unbounded
  authority; bind the concrete grant before relying on it.

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

## Analyse Learning Evidence

When diagnosing training behaviour or interpreting a result, do not infer a
learning conclusion from one smoothed curve or one seed. Assemble the widest
comparable evidence that is actually available:

- inspect raw and smoothed train and validation curves together, with their
  windows, sample counts, segment boundaries and axes stated;
- compare multiple fixed seeds and report individual runs plus centre,
  dispersion and outliers instead of silently pooling them;
- bind exact hyperparameters and schedules, including optimiser, learning
  rate, temperature, entropy, batch/update cadence, rollout horizon, opponent
  mix, search budget and enabled components;
- bind dataset, split, ruleset, label-schema and database versions and check
  for leakage, identity drift or changed class balance;
- compare against a frozen, compatible baseline under the same rules,
  starting positions, colours, work budget and adjudication;
- use controlled ablations to isolate claimed causes, changing one relevant
  factor at a time unless an interaction experiment is explicitly designed;
  and
- report per-class metrics with support counts and, when applicable,
  per-phase, per-opponent, per-colour and per-termination results. Include
  macro and micro summaries when imbalance makes the distinction material.

Treat missing evidence as an explicit gap. If ordinary supervised validation
does not exist for an RL run, use a frozen held-out corpus or evaluation as a
separately named validation-like measure; never invent or relabel a training
metric as validation. Distinguish raw observations from rolling-window or
plotting artefacts, and distinguish rules draws from max-ply truncations.

Structure every substantive diagnosis under these separate labels:

1. **Observed facts / 观察事实** — directly measured values with artefact,
   version, seed and scope identities;
2. **Hypotheses / 假设** — falsifiable explanations, ranked when several fit;
3. **Supporting evidence / 支持证据** — evidence that raises each hypothesis;
4. **Counterevidence / 反证** — conflicting results, alternative explanations
   and confounders; and
5. **Next validation experiments / 下一步验证实验** — the smallest decisive
   experiment, including control, changed variable, seeds, data version,
   metrics, acceptance rule and resource bound.

Mark predictions explicitly. Do not present curve correlation, a training-tail
improvement, a single-seed result, or an anti-collapse gate as causal evidence
or playing-strength promotion.

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
the user explicitly requested launch, or a valid standing delegation covers
the exact plan, proceed only after reporting a passing gate. Otherwise stop
with the failed or unresolved condition. Never request the same product
decision again for each child plan inside one valid standing delegation.
