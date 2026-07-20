# Managed Training Operations

## Purpose

This workflow lets a non-ML product owner work with an autonomous Agent without
becoming the manual operator for technical training choices. It applies to the
current corrected v4 Generalist baseline on one Windows machine and one CUDA
GPU. It does not introduce distributed training, a C++ trainer, or a daemon.

The central rule is separation of authority:

- the Agent owns technical configuration, preflight, diagnostics, bounded
  recovery, evidence validation, and quarantine;
- the product owner owns the objective, total resource envelope, launch
  authorization, any later resource expansion, and publication or promotion;
- creating or editing a technical plan never authorizes training;
- a technical failure does not become an ML question for the product owner.

No managed plan or launch authorization existed when this document was added,
and no training was started as part of the infrastructure change.

## Product Decisions

The Agent should ask the product owner only when at least one of these product
boundaries changes:

1. whether to pursue the stated training objective;
2. the maximum game count or wall-clock resource envelope;
3. whether to authorize the first smoke or long-run launch;
4. whether to create a successor plan after the authorized resources are
   exhausted;
5. whether evidence may support publication, model promotion, or a release
   claim.

Questions about learning rate, node budget, temperature, checkpoint cadence,
data-loader structure, CUDA diagnostics, or exact-resume compatibility belong
to the Agent. The Agent must choose conservatively, record the choice, test it,
and stop fail-closed if evidence is inadequate.

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

## Managed v4 Technical Default

The supervisor currently prepares this conservative default:

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

These are technical defaults, not playing-strength evidence. The Agent may
revise them before results are visible, but every actual run must freeze the
complete selection in a new immutable plan. Changing a frozen plan requires a
new plan and a new product authorization.

## Lifecycle

### 1. Agent prepares a plan

From a clean committed worktree, the Agent runs a command equivalent to:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir <new-ignored-control-directory> `
  --max-wall-hours <product-resource-limit>
```

Preparation validates the technical configuration and records the current Git
commit and local path-config hash. It writes no model checkpoint and starts no
training. Its status is `awaiting_product_authorization`.

### 2. Product owner authorizes or rejects

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

The Agent must not create this authorization merely because a plan exists.
When asking for this decision through Codex, it must wait for an explicit user
answer and must not use a response timeout.

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
The Agent may continue only when an existing authorization still covers the
same immutable semantics and the recovery path is proven safe.

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
plan. A long run still requires an explicit product launch authorization.
