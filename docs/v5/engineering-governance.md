# v5 Engineering Governance

Status: governing planning, review, contract, and failure-severity
specification.

The purpose of governance is to protect semantic changes and retained
evidence. It must not require a constitutional-scale approval process before
the project can learn whether its core idea is feasible.

## Risk Classes

| Class | Examples | Required control |
| --- | --- | --- |
| `analysis_only` | Read-only F0 audit, inventory, benchmark with no candidate result, report generation | Bounded question, inputs, commands, output directory, and evidence limits; self-review plus reproducible checks |
| `standard` | Reversible refactor, ordinary training instrumentation, performance work, test-fixture repair with unchanged semantics | Scoped plan, failure-first test, review, verification, and rollback |
| `critical` | Rules, oracle/comparator, gold data, labels/splits, teacher objective/reward, runtime authorisation, release thresholds/claims | Full plan, independent adversarial review, fault injection, immutable execution lock, and complete traceability |

F0 `analysis_only` work may precede a complete machine-readable governance
stack. It cannot modify databases, labels, models, runtime decisions, or
release artifacts.

## Normative Authority

Passing code is not the highest source of truth. Use this order:

1. signed product and rules-variant decisions plus the primary normative
   sources they identify;
2. the modular specification that owns the disputed semantic field;
3. independent reference implementations or hand-derived boundary results;
4. executable conformance tests;
5. production or research implementation;
6. experimental and operational observations.

A conformance test is accepted only inside its declared rules version and
support domain. If new evidence shows that its expected result encoded the
wrong rule, preserve the old test result as history, correct the owning
specification and independent expectation, and invalidate dependent evidence.
Do not preserve an incorrect semantic result merely because it was previously
marked accepted.

Raw inputs, commands, hashes, logs, and results are immutable evidence.
Interpretations, review dispositions, and pass/fail status are append-only but
corrigible: a later record may supersede them while preserving the earlier
record and reason.

## Review Policy

Heterogeneous model or human review is required for critical rules/oracle/gold,
label/split, reward/objective, authorisation, and release changes.

For standard work, same-model self-review plus independent executable CI is
permitted when:

- the change is reversible;
- no semantic contract or public interface changes;
- tests have independent expected results;
- the candidate cannot be promoted directly.

If heterogeneous review is unavailable, critical work may remain an
experimental draft but cannot produce accepted gold, labels, or release
evidence. Different model agreement is supporting review, not proof.

## Preapproved Automatic-Repair Boundary

An approved execution batch may automatically repair a newly exposed defect
without restarting review only when every condition holds:

- same declared module and file set;
- driven by an existing failing test or deterministic verifier;
- no change to rules, oracle meaning, data/schema, objective, threshold,
  candidate pool, public API, product claim, or support domain;
- bounded diff and side effects;
- no new dependency or external write;
- the required focused and cumulative gates rerun.

Any semantic expansion stops the batch and requires a revised plan. A repair
must not be called mechanical merely to avoid review.

## Minimal Executable Contract

`experiment-card.yaml`, the historical `training_contract.yaml`, or an
experiment-specific equivalent contains only machine-enforceable hard
constraints:

- variant, code/data/rule/oracle/model hashes;
- dependency DAG and accepted prerequisite artifacts;
- dataset read/write/access permissions;
- active objective and candidate-product mode;
- permitted/forbidden runtime authority and fallback;
- error policy;
- resource and run budgets;
- release thresholds and support domain;
- exact-resume and output-isolation requirements;
- validation commands and required artifacts.

Research rationale, statistical derivations, formulas, report prose, and
review discussion remain in Markdown or analysis code. A semantic statement
must not be duplicated in Markdown, YAML, implementation, and tests with four
independent owners. The contract references the owning specification and
encodes only the values needed to reject an invalid execution.

The four execution-level sources of truth are:

- `product-decision.md` for user, persona, deployment, and value decisions;
- `oracle-and-rules-spec.md` plus its signed variant record for deterministic
  semantics;
- one `experiment-card.yaml` for the actual variables and permissions of a
  bounded run; and
- one `release-decision.yaml` for the applicable release gates and result.

The research, human-data, runtime, and governance documents remain design
owners and reference material. An executor reads this entry point, the product
decision, the one owning modular specification, and the selected card. It does
not need to load every design note for a bounded change that cannot affect
their semantics.

## Failure Severity

Use one of these non-overlapping levels:

| Level | Meaning | Example action |
| --- | --- | --- |
| `input_reject` | One record/request is outside the declared valid input | Quarantine record; continue only if the batch contract permits |
| `job_fail` | Current bounded job cannot produce a trustworthy artifact | Exit non-zero; preserve reproduction evidence |
| `experiment_block` | Evidence, support, resources, or prerequisites are inadequate | Do not start/continue the dependent experiment |
| `release_block` | Candidate evidence cannot support promotion | Retain current product and artifacts |
| `runtime_unavailable` | A proof-bearing product cannot authorise this decision in budget or support domain | Follow the pre-signed product adjudication; no hidden fallback |
| `fatal_safety_fault` | False authorisation, illegal/unauthorised move, oracle/rule contradiction, or corrupted authority state | Stop the affected runtime and quarantine the artifact/version |

Planning conflicts, a second confirmation access, or an over-budget pilot are
not named fatal safety faults. Conversely, a false authorisation must not be
downgraded to an ordinary stage block.

## Implementation Plan Contents

A critical plan records:

- objective and non-objectives;
- base commit and dependency artifacts;
- actual files/APIs/assets inspected;
- requirement-to-symbol-to-test-to-artifact traceability;
- data flow, perspective, action atomicity, and migration;
- failure-first tests and independent expected-result provenance;
- commands, side effects, output isolation, budgets, and rollback;
- open questions whose answer could change semantics;
- review findings and dispositions.

A standard plan may omit sections irrelevant to its bounded change. An
analysis-only plan is a compact audit card. Templates scale with risk; they do
not require empty boilerplate merely for schema compliance.

## Evidence and Verification

Accepted evidence consists of recomputable artifacts:

- immutable inputs and identities;
- command, environment, and exit status;
- focused tests and applicable cumulative tests;
- raw result plus independent recomputation;
- support domain and known unknowns;
- review and approval identity for critical work.

Screenshots and plots aid diagnosis. They do not create pass status.

Gold expected results must come from an independent reference or hand-derived
boundary specification, not the implementation under test. A test cannot be
made green by skipping, weakening, using empty data, swallowing required
errors, or substituting a mock on a formal integration path.

## Data and Run Authority

Documentation work, read-only audits, and implementation do not authorise:

- database activation or destructive migration;
- training or evaluation launch;
- checkpoint promotion;
- publication or product deployment;
- merge, rebase, force-push, or history rewrite.

Each action uses its owning experiment or operational approval. Training
preflight follows the repository's readiness workflow and records checkpoint
lineage, output directory, database paths, Git state, component flags, and
exact command.

## Agent and Human Responsibilities

Agents may make routine technical decisions inside an approved scope. They
must stop for:

- a product objective or target-population choice;
- a new training/evaluation launch;
- a semantic rule/oracle ambiguity;
- legal/privacy authority not inferable from accepted records;
- a support-domain or claims change;
- material external coordination.

The product owner is not asked to choose low-level technical details that code
and tests can determine. The agent supplies evidence and a recommendation
rather than transferring unresolved engineering diagnosis to a nontechnical
approver.

## Versioning

A change to any of these creates a new semantic version and invalidates
downstream evidence as declared by the dependency graph:

- rules/history/atomic-action semantics;
- oracle/comparator or asset identity;
- human target/estimand/split;
- teacher objective or candidate set;
- runtime product mode, pack/proof authority, or support domain;
- release endpoint, margin, or claim.

Pure editorial clarification and link repair do not invalidate model evidence
when they change no owning semantics. Every such change is still reviewed for
accidental normative drift.
