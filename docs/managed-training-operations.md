# Managed Training Operations

## Purpose

This workflow lets a non-ML product owner work with an autonomous Agent without
becoming the manual operator for technical training choices. It applies to the
current corrected v4 Generalist baseline on one Windows machine and one CUDA
GPU. It does not introduce distributed training, a C++ trainer, or a daemon.

The central rule is separation of authority:

- the Agent owns technical configuration, preflight, diagnostics, bounded
  recovery, evidence validation, and quarantine;
- the product owner owns the objective, aggregate resource envelope, the
  initial direct or standing launch authorization, any later resource
  expansion, and publication or promotion;
- creating or editing a technical plan never authorizes training;
- a technical failure does not become an ML question for the product owner.

This document distinguishes a direct one-plan authorization from a standing
delegation over a finite preregistered plan family. Both remain separate from
technical readiness.

## Product Decisions

The Agent should ask the product owner only when at least one of these product
boundaries changes:

1. whether to pursue the stated training objective;
2. the maximum game count or wall-clock resource envelope;
3. whether to authorize one launch or a finite preregistered sequence;
4. whether to create a successor plan after the authorized resources are
   exhausted;
5. whether evidence may support publication, model promotion, or a release
   claim.

Questions about learning rate, node budget, temperature, checkpoint cadence,
data-loader structure, CUDA diagnostics, or exact-resume compatibility belong
to the Agent. The Agent must choose conservatively, record the choice, test it,
and stop fail-closed if evidence is inadequate.

## Standing Delegated Authorization

A product owner who does not want to operate individual technical arms may
authorize a parent sequence once. The durable record must bind all of these:

- the objective and owning experiment or plan-family identity;
- the complete allowed plan set or deterministic eligibility rule and order;
- aggregate game, active wall-time, and hardware bounds;
- whether preparation, per-plan authorization, launch, and proven
  semantics-identical recovery are allowed;
- the claim boundary and actions that remain forbidden; and
- an expiry condition and the fact that the owner may revoke the grant.

For a covered child plan, the Agent performs readiness just in time, verifies
that prior required children completed cleanly, and writes the ordinary
plan-bound `authorization.json` with
`authorized-by=product-owner-delegated-agent`. Its decision note cites the
standing grant and the exact child scope. The Agent then launches without a
second product prompt. A missing leaf authorization file is therefore an
operation for the Agent, not an unresolved product decision, when and only when
a valid standing delegation covers that exact plan.

The delegation does not authorize a retry merely because a process failed. An
anomaly stops the sequence for Agent diagnosis. Automatic continuation is
allowed only after the evidence shows the next action is still inside the same
immutable semantics and the grant explicitly covers that recovery. Otherwise
the Agent preserves and quarantines the evidence before requesting any truly
new product scope.

Long training, held-out evaluation, resource expansion, promotion,
publication, release, destructive cleanup, external writes, and Git history
rewrites remain outside a standing preparatory grant unless each is explicitly
named. Ordinary Git push authority is also governed separately by
`AGENTS.md`.

## Durable Contracts

Each managed objective has a dedicated ignored control directory containing:

| File or directory | Meaning |
| --- | --- |
| `plan.json` | Immutable Git, semantic, path-config, game, segment, and wall-time bounds |
| `authorization.json` | Separate product authorization bound to the exact plan SHA-256 |
| `controller-events.jsonl` | Append-only, hash-chained supervisor history |
| `controller.lock` | Exclusive ownership while one supervisor is active |
| `segments/segment-NNNN/` | One isolated trainer run contract, ledger, logs, and checkpoints |

`plan.json` also forbids publication and promotion. Those actions require a
later evidence-specific decision and cannot be smuggled into a training launch.

The local `data/training_paths.local.json` remains ignored and
machine-specific. The plan records its file identity, while trainer preflight
resolves and verifies the actual Malom, HumanDB, SpecialistDB, and output paths.

## Completed Managed v4 Technical Default

The completed 5,000-game plan used this conservative default:

- A2C, with PPO disabled;
- fresh random initialization;
- Sentinel, ValueNet, and GapNet disabled;
- S1A warm-start and ongoing imitation mixing disabled independently;
- 50% frozen-target and 50% heuristic opponents;
- target refresh every 50 games;
- full rollout, `sim_ply_depth=5`, no branch rollouts, and `max_ply=60`;
- 500,000 single-threaded native search nodes per heuristic move, under the
  contract in [`fixed-node-heuristic-search.md`](fixed-node-heuristic-search.md);
- temperature `0.90` to `0.20` over the existing 80% schedule;
- seed 42 and `batch_games=1`;
- 5,000 games in 250-game process segments;
- `latest.pt` and diagnostic publication every 50 games.

These values are historical lineage, not playing-strength evidence or a
reusable successor default. In particular, `max_ply=60` was experiment
truncation rather than a rules draw. Current plan preparation requires an
explicit objective, a new experiment ID, and an explicit logical-ply ceiling;
it cannot silently copy those three boundaries. Every actual run freezes the
complete selection in a new immutable plan. Changing a frozen plan requires a
new plan and a new product authorization.

## Lifecycle

### 1. Agent prepares a plan

From a clean committed worktree, the Agent runs a command equivalent to:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir <new-ignored-control-directory> `
  --max-wall-hours <product-resource-limit> `
  --objective <product-approved-objective> `
  --experiment-id <new-experiment-id> `
  --seed <explicit-trainer-seed> `
  --max-ply <explicit-experiment-truncation>
```

Preparation validates the technical configuration and records the current Git
commit and local path-config hash. It writes no model checkpoint and starts no
training. Its status is `awaiting_product_authorization`.

### 2. Resolve direct or standing authorization

The Agent presents a short product view containing the objective, maximum game
count, wall-time bound, and claim boundary. If the product owner explicitly
approves, the Agent records that decision with a command equivalent to:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py authorize `
  --plan <control-directory>\plan.json `
  --authorization <control-directory>\authorization.json `
  --authorized-by product-owner `
  --decision-note <recorded-product-decision>
```

The Agent must not create this authorization merely because a plan exists. It
first checks for a recorded standing delegation. If that grant covers the
exact plan, the Agent creates the same plan-bound file just in time and records
the delegated operator and parent decision. If no valid grant exists, the
Agent asks once for the parent objective and resource envelope, waits for an
explicit answer, and never uses a response timeout or a default selection.

### 3. Agent launches bounded work

For one segment:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py run-next `
  --plan <control-directory>\plan.json `
  --authorization <control-directory>\authorization.json
```

For all safe segments within the same authorization:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py run-authorized `
  --plan <control-directory>\plan.json `
  --authorization <control-directory>\authorization.json
```

The first segment is always `fresh`. Every later segment is a new isolated run
whose only permitted parent is the verified `latest.pt` from the immediately
preceding completed segment. Continuation uses explicit `exact-resume`; no
directory scan or “best checkpoint” heuristic selects the source.

The supervisor invokes Python with an argument list and no shell. It enforces
a single-controller lock and supplies the trainer only the remaining authorized
wall time. It validates the child run ledger, checkpoint envelope, experiment
identity, semantic hash, and expected game count before scheduling another
segment.

### 3b. Host-reboot recovery

If the host reboots while a segment is marked `running`, do not treat the
incomplete segment as completed. From a clean worktree that is the frozen plan
commit or a descendant that only adds recovery tooling, run:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py recover-interrupted `
  --plan <control-directory>\plan.json `
  --authorization <control-directory>\authorization.json
```

This clears a stale controller lock when its PID is dead, quarantines the
incomplete segment directory, backs up the live SpecialistDB, publishes a
recovery checkpoint whose SpecialistDB identity matches the live database, and
records `managed_segment_interrupted` with `reason_code=host_reboot`. Then
relaunch with `run-authorized` or `run-next`. The restarted segment
exact-resumes from the recovery checkpoint and must still finish at the
original segment game bound.

### 4. Agent reports product status

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py status `
  --plan <control-directory>\plan.json `
  --authorization <control-directory>\authorization.json
```

The top-level response is intentionally small:

- `state` and a plain-language `summary`;
- whether a product decision is required;
- completed and maximum games;
- completed segments and consumed wall time;
- nested technical identities for Agent diagnosis.

## Stop and Escalation Policy

The supervisor stops automatically for a dirty or different Git commit,
changed path configuration, missing authorization, output reuse, incompatible
resume state, checkpoint corruption, wrong game count, non-zero trainer exit,
timeout, or broken evidence chain.

Technical stops use `stopped_for_agent_review` and do not ask the product owner
to diagnose ML or infrastructure details. The Agent investigates using the run
contracts, event ledgers, focused tests, and the training-readiness workflow.
The Agent may continue only when a direct authorization or recorded standing
delegation still covers the same immutable semantics and the recovery path is
proven safe.

`resource_limit_reached` is different: increasing the wall-time or game budget
changes product scope. The Agent must present the evidence and request a new
explicit product decision. A successor plan and authorization are required;
the old files remain immutable.

Neither training completion nor a favorable training metric authorizes model
promotion. Candidate evaluation and any accepted/release claim remain separate
P1 work under a frozen evaluation contract.

## Current Readiness Boundary

The managed contracts, authorization gate, fixed-node search control, explicit
imitation controls, segment command construction, lock ownership, and
product-status behavior have focused automated coverage. The Rust node cap was
also exercised through the rebuilt local PyO3 extension.

This infrastructure does not by itself make the long run ready. Before launch,
the Agent must invoke the repository's training-readiness workflow, run a new
disposable smoke that reaches at least one RL update, validate the intended
empty corrected SpecialistDB and output isolation, and record the exact launch
plan. A long run still requires an explicit product launch authorization or a
standing delegation that explicitly names that exact long-run scope. If the
current standing grant covers the plan, the Agent must not ask again. If it
does not, the readiness question must wait for an explicit product answer; no
timeout or default selection may create the objective, aggregate resource
envelope, truncation ceiling, or authorization.
