# Baseline-v1 attempt-002 instrumentation audit

Date: 2026-08-16

This is a zero-game, zero-search implementation audit.  It does not contain a
candidate result and does not consume the once-only formal measurement.  The
frozen attempt supplement is
`e92c2095d2a0bbcaf892fb8e6be707a9b8ca71d73b332bf9d7ce5dcae0f51d29`.

## Disposition

The point Malom proxy has been removed.  Query accounting now attaches an
optional observer to the real `MalomDB.query_value` implementation.  Every
single-state, move-quality, all-moves, trajectory, training-lookahead, and
direct `A_pos` route reaches that one implementation.  This reduces the
resource-counting interception surface to one method and preserves the real
public signatures of `MalomDB` and `ExternalSolvedDB`.

The generic surface audit passes.  It found no remaining transparent proxy,
no post-construction Malom delegate rebinding, and no callable-shape mismatch.
This is readiness evidence only.  Runtime return shapes and every exercised
path must still pass the all-surface non-evidence rehearsal before preflight.

## Why single-throat counting is safer

The failed implementation wrapped only `query()` and delegated every other
attribute through `__getattr__`.  Its explicit method called
`ResourceLedger.add_malom()` without the required `count`.  More importantly,
the transparent delegation meant `query_state`, `query_move_quality`,
`query_all_moves`, and `query_trajectory` could bypass the counter while still
appearing API-compatible.

The new observer is invoked immediately before the unique physical Malom
lookup.  The observer reserves one query against the ceiling before any table
read.  Terminal short circuits in `ExternalSolvedDB` do not reach the throat
and therefore count zero.  The retained-v4 route receives the observer at
construction; no policy or lookahead member is replaced later.

## Complete instrumentation inventory

The audit treats the following as the complete attempt-owned interception and
adaptation surface.

| Area | Methods | Attributes or result fields |
| --- | --- | --- |
| Malom access | `MalomDB.query_value`; delegated entry points `MalomDB.query`, `ExternalSolvedDB.query_state`, `query`, `query_move_quality`, `query_all_moves`, `query_trajectory` | `MalomDB._query_observer` |
| Sanmill and strict referee | `SanmillUciSession.search_logical_turn`, `SanmillTrainingGame.apply_nmm_move`, `assert_current_board`, `search_and_apply`, `_checked_search_result`, `_checked_position_state`, `_strict_terminal_outcome` | `session`, `state`, `history` |
| Candidate routes | `TrainingAlignedPolicy.score_moves`, `_RetainedV4Scorer.score`, `_ProductSpecialistScorer.score`, `ModelPolicySet.scorer`, `_select_scored_move`, `_candidate_choice` | `TrainingAlignedPolicy.malom`, `lookahead_advisor`, `SpecialistRouter._gameai` |
| Result packaging | `validate_game_record`, `_finalize_game`, `compact_game`, `append_game_record`, `load_game_records` | `winner`, `outcome_reason`, `candidate_score`, `termination_class` |
| Durable resources | `ResourceLedger.add_engine`, `add_malom`, `record`, `append_resource_checkpoint`, `load_resource_checkpoints`, `verify_resource_game_alignment`, `write_json_atomic`, `write_sealed_json` | `engine_searches`, `malom_queries`, `active_seconds` |
| Protected-data guard | `EstimatorAccess.assert_allowed`, `derive`, `load_decisions` | official and research partition maps, allowed sessions, successful and denied access logs |
| Canaries | `_PoisonGameAI.__getattribute__`, `audit_specialist_gameai_dependency`, `audit_instrumentation_surface` | `SpecialistRouter._gameai` |

The audited public surfaces also freeze every public method and property on
`MalomDB`, `ExternalSolvedDB`, `ResourceLedger`, `TrainingAlignedPolicy`, both
candidate scorer adapters, `ModelPolicySet`, `SanmillUciSession`,
`SanmillTrainingGame`, and `EstimatorAccess`.  Adding a public interception
method without updating the contract makes the audit fail.

## Signature and return-shape checks

The generic audit binds the real call shape for all intercepted callable
paths.  It covers positional arguments, keyword-only arguments, defaults, and
the declared return annotation.  In particular it checks:

- the optional `query_observer` constructor keyword on both Malom adapters;
- single-state, move-quality, batch all-moves, and trajectory queries;
- engine node-budget and optional-depth arguments;
- placement, movement, and flying moves through the same board argument type;
- strict-referee apply, assertion, search, position-state, and terminal result
  shapes;
- candidate scoring, allowed-subset selection, and final result packaging;
- resource increments, snapshots, checkpoint append/recovery/alignment, and
  atomic or sealed writes; and
- protected-access denial before a content producer or decision loader runs.

There are no remaining mismatches in the static audit.  Runtime canaries are
predeclared to assert concrete return shapes and exact query deltas for every
Malom entry point.  The terminal canary separately proves that a rules-terminal
board returns a WDL without a table query.

## Tests that retain discrimination

The original red-then-green regression remains as
`test_counting_malom_proxy_records_each_completed_query`; it now proves that
two real `MalomDB` entry calls reserve exactly two queries without using a
proxy.  It is supplemented by three generic tests:

- the complete surface and all real call signatures pass;
- replacing an existing method with an incompatible signature is rejected;
- adding an unregistered public query method is rejected.

The crash regression writes and fsyncs the resource checkpoint before the game
record and proves the completed-game resource state remains recoverable after
the simulated interruption.  The protected-data regression proves the content
producer is never invoked for selection, confirmation, final-test, or
research-confirmation sessions.

At freeze time the focused baseline and route-policy suite reports 16 passing
tests, and task-scope Ruff passes.  The attempt-002 preflight will rerun these
tests, the mandatory Malom/DB-teacher/provenance group, and Ruff from the clean
authorization-bound tree.

## Rehearsal coverage frozen before execution

The new non-evidence rehearsal contains 24 live games: three source phases by
four frozen arms by both candidate colors.  Three additional real strict
referee replays cover a threefold draw, a fifty-move draw, and a decisive
terminal.  The source records are outside the 254 formal starts.  This covers
both free and `A_pos`-constrained selection, all three phases, both colors,
draw and decisive packaging, all Malom query forms, durable journals, and the
protected-data guard.

No mock, stub, neutral fallback, or default result is accepted for rehearsal
coverage.  If this rehearsal fails because of another instrumentation or
contract mismatch, the frozen rule is to stop and produce a structural tooling
analysis; no further point repair is allowed.

## Boundaries

The failed attempt-001 authorization `3f30c558...` is void for the repaired
implementation.  Its 23 Malom queries and 8.057233 seconds remain charged to
the unchanged aggregate envelope.  Its output namespace is checked byte for
byte before the new rehearsal and is never reused.

This audit authorizes no training, checkpoint change, database write,
promotion, deployment, publication, release, or model claim.  All safety
statements remain positional-only `A_pos`; they are not `A_allow`.
