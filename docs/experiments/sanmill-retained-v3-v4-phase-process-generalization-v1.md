# Retained-v3/v4 phase-process generalization v1

Status: `draft_source_ready_runner_plan_readiness_and_authority_absent`

This is the recommended successor design after the completed reused-corpus
passivity diagnostic. It authorizes no evaluation game, candidate load,
training, update, database or checkpoint write, promotion, publication or
release. The machine-readable plan is intentionally absent until the runtime
can execute this contract exactly.

## Objective and claim boundary

Describe whether the named retained-v4 final route has a higher probability
than retained-v3 of remaining strict-rules ongoing through 108 additional
logical plies after a frozen phase-history start. The result may describe only
these two named routes on this fixed, previously project-visible corpus.

It cannot support a playing-strength, equivalence, held-out, target-refresh
causal, promotion or publication claim. Eventual W/D/L is secondary and
descriptive.

## Frozen source corpus

The source artifact is
[phase-process corpus v1](sanmill-retained-v3-v4-phase-process-corpus-v1.json),
identity
`3be3d76c34511e0f78d0f5bfe4a338c415c393306a955538bb85823e9d62c080`,
file SHA-256
`8353ff3e52465bf99f7cf468a9cbcb4681a673ac2cebcdae00c253df8a22670b`,
and records identity
`8fdf3adf60857543a440aee4b354938bec32a6f6f667effa381774abadf7d95d`.

It contains 39 standard-start-replayable, current-referee-nonterminal starts:
18 placement, 14 movement and seven flying. It excludes the prior 12-start
phase development measurement and has zero exact/D4 overlap with the previous
64 opening starts, HumanDB, retained-v3 SpecialistDB and retained-v4
SpecialistDB at the frozen start. The full source audit is in the
[readiness evidence](../evidence/sanmill-retained-v3-v4-phase-process-corpus-readiness-2026-08-13.md).

## Proposed immutable protocol

| Field | Required value |
| --- | --- |
| Starts | all 39 frozen records, no subset selection after candidate load |
| Candidate colours | White and Black for every start |
| Candidates | retained-v3 refresh-50 final route and retained-v4 no-refresh final route |
| Games | 39 × 2 colours × 2 candidates = 156 |
| Pairing order | adjacent v3 then v4 inside each start/colour unit |
| Candidate selection | deterministic CPU float32 policy argmax over each exact training-aligned route |
| Opponent | pinned strict Sanmill, one thread, MTD(f), IDS, shuffle off, seed 42 |
| Opponent work | at most 500,000 nodes per complete logical turn; no wall-clock move limit |
| Start | replay the complete variable-length history and verify FEN, history hash, clocks and non-terminal state |
| Observation window | 108 additional complete logical plies after the frozen start |
| Completion | continue to strict rules terminal or 1,536-post-start invalid safety cap |
| Retry | none; explicitly authorized exact-spec missing-suffix resume only |

The primary unit is one start. For each start, first compute the v4-minus-v3
survival difference for candidate White and candidate Black, then average the
two colours. The primary estimate is the mean of those 39 start values with a
two-sided normal engineering interval using `z=1.96`.

- lower bound above zero and half-width at most 0.10:
  `v4_higher_108_post_start_ply_survival`;
- upper bound below zero and half-width at most 0.10:
  `v3_higher_108_post_start_ply_survival`;
- interval crossing zero with half-width at most 0.10: `inconclusive`;
- half-width above 0.10: `inconclusive_precision` regardless of direction.

The 39-start count gives an estimated 9.3651pp half-width when the previous
development corpus's start-level standard deviation is reused. That is a
planning input, not a promised precision result and not permission to add
starts after launch.

## Secondary evidence

For every candidate and phase, report with denominators:

- survival through 108 post-start plies by candidate colour and source phase;
- start, horizon and terminal no-capture counts, plus horizon-minus-start
  change for games that reach the window;
- repetition count/history length at start, horizon and terminal;
- termination reason, total and post-start length, and invalid safety caps;
- eventual strict W/D/L as descriptive evidence only;
- Malom query coverage and exact W/D/L preserving/downgrading candidate turns;
- immediate W/D/L-preserving capture opportunities and missed selections; and
- complete Malom-order opportunity exposure and normalized ordinal regret,
  with any conditional metric clearly labelled exploratory unless separately
  preregistered.

The web page must show the primary start-clustered interval, phase support,
the no-capture/repetition trajectories, termination causes and every metric's
denominator. Help text must state that continuation survival is not a draw or
strength measure, Malom is history-free, the corpus is not held out, and
incomplete caps are never converted into draws.

## Proposed resource envelope

- at most 156 games;
- at most two active evaluator hours;
- one evaluator and one strict Sanmill process at a time;
- at most 1,536 post-start logical plies per game;
- at most 119,808 Sanmill search turns and a summed per-turn ceiling of
  59,904,000,000 nodes across the theoretical safety envelope;
- no automatic retry, semantic recovery, resource extension or result-based
  early stop.

## Remaining freeze gates

1. Create successor-owned, sidecar-free, read-only copies of both exact route
   bundles and both candidate SpecialistDB snapshots; bind their byte
   identities without changing the completed diagnostic namespace.
2. Implement variable-history replay, the relative 108-ply horizon,
   start-clustered reporting and the required web explanations.
3. Add focused schedule, ledger, report, exact-resume, strict-prefix and web
   tests; run the mandatory Malom/DB/provenance gate because both route
   databases are in scope.
4. Freeze one machine-readable plan at a clean implementation commit and run
   final source readiness against fresh ignored output paths.
5. Obtain one explicit product authorization bound to that exact plan and
   readiness identity before opening the first game.
