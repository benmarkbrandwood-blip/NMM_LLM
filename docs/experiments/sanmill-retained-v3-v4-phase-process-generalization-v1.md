# Retained-v3/v4 phase-process generalization v1

Status: `source_ready_0ff79e39_awaiting_product_authorization`

This is the recommended successor design after the completed reused-corpus
passivity diagnostic. It authorizes no evaluation game, policy decision,
training, update, database or checkpoint mutation, promotion, publication or
release. Candidate artefacts have now been opened only to verify successor-
owned byte-exact snapshots and deterministic synthetic canaries.

The canonical [machine plan](sanmill-retained-v3-v4-phase-process-generalization-v1.json)
has identity
`4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256`
and file SHA-256
`09245e5f66af3d18ba2818d1dfac70b4c7eec8d63c9388d501b32846dfccf9d3`.
It binds implementation commit `5a318a063b561b12bafe5e72e44ff6fdc9426f1e`
and is tracked by plan commit `117a5be8086af04ba0b311f44a23cdc9804a7284`.
It is not an authorization.

The earlier unlaunched identity `7dfd8e32...` was superseded before any
authorization or runtime artifact existed. The only reason was to bind the
plan to the readiness-identity hardening that rejects post-plan runtime-code
changes while allowing the four named status documents to record evidence.

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

## Frozen immutable protocol

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

## Successor-owned input snapshots

The ignored input root is
`learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-phase-process-generalization-v1/inputs`.
It was populated once from the completed diagnostic inputs, marked read-only,
and audited without selecting a move. Snapshot identity is
`b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac`;
the canonical manifest file SHA-256 is
`cda9456e0234a9532ddfb1b90e3a78bb6a35ef788c0eddfca607e9f33cb1942a`.

The v3 and v4 route identities remain respectively `b6d7ecf6...` and
`817d2e36...`; their ordered file-list identities are `97c6413a...` and
`f701206d...`. The sidecar-free SpecialistDB main-file SHA-256 values remain
`82d7fbcd...` and `3d69d1ac...`. Runtime planning must bind only these
successor paths, not the completed plan-`035c68f8` paths.

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

## Frozen resource envelope

- at most 156 games;
- at most two active evaluator hours;
- one evaluator and one strict Sanmill process at a time;
- at most 1,536 post-start logical plies per game;
- at most 119,808 Sanmill search turns and a summed per-turn ceiling of
  59,904,000,000 nodes across the theoretical safety envelope;
- no automatic retry, semantic recovery, resource extension or result-based
  early stop.

## Source readiness

Two complete fresh preflights on published clean `dev` produced the same
source-readiness identity
`0ff79e398233c7ed9fcdec4cc5cd406837330140a3c1cec720e11eaa274ae365`.
All ten technical gates passed: repository and plan, absent fresh outputs,
successor inputs, corpus and schedule, both candidate routes, pinned Sanmill,
all 39 strict history replays, process ownership, focused tests, mandatory
Malom/DB/provenance tests and Ruff. The only failed gate was the deliberately
separate product authorization.

Both preflights requested zero corpus candidate moves and played zero games.
The HumanDB loader's warning that its unversioned historical Malom columns are
disabled is expected and consistent with the frozen data contract; corrected
Malom labels come only from the separately identified tablebase.

## Remaining launch gates

1. Completed: create and byte-audit successor-owned, sidecar-free, read-only
   copies of both exact route bundles and SpecialistDB snapshots without
   changing the completed diagnostic namespace.
2. Completed at source commit `f8070d1`: implement variable-history replay,
   the relative 108-ply horizon, start-clustered reporting, live partial
   recomputation and the required web explanations.
3. Completed at commits `60bb10e`, `728fa25`, `32e8843` and `5a318a0`:
   implement the
   fail-closed runner, exact missing-suffix resume, identity-bound zero-game
   safe-capture/full-order follow-up, non-skippable plan-freezer tests, and a
   stable source identity that fails on post-plan runtime changes.
4. Completed and published: freeze machine plan `4c85ff33` at plan commit
   `117a5be`; run every technical gate twice and obtain stable source readiness
   `0ff79e39`.
5. Obtain one explicit product authorization bound to that plan and source-
   readiness identity before opening the first game.
