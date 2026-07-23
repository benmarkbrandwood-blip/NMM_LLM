# Sanmill Strict UCI Bridge Smoke v1

Date: 23 July 2026

Status: **completed and passed under the book-off bridge contract; not a
formal evaluation specification and not authority for candidate-versus-
baseline games**.

Related:

- [next-evaluation decision brief](dev-v4-training-aligned-evaluation-v1-decision-brief.md)
- [local path registry contract](../local-training-layout.md)
- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)
- [completed bridge result](../evidence/sanmill-strict-uci-bridge-smoke-2026-07-23.md)

## Decision and purpose

The in-repository `GameAI` is deferred as the formal playing-strength
baseline. Its compact position and current game lifecycle do not yet preserve
the complete no-capture and repetition history needed for an authoritative
referee.

The next bounded task is to validate a strict subprocess bridge to one exact
Sanmill revision. Sanmill owns the standard-NMM rule state, complete action
history, legal-action enumeration, and terminal adjudication. NMM_LLM mirrors
stable positions only to detect coordinate, move-codec, and ordinary legality
divergence. A passing mirror does not transfer rule authority back to the
current NMM_LLM `GameEngine`.

The pinned source revision for this smoke is:

```text
6f080c5a6d15919bf0a45fa5528c45d4487a2b8f
```

The checkout is found through the ignored `sanmill_checkout` entry in
`data/training_paths.local.json`. No host-specific checkout path is committed.
The audit must reject a different `HEAD`, any tracked or untracked source-tree
change, an absent binary, or a binary without the required fail-closed build
marker.

## Strict engine contract

The bridge must start `target/release/tgf[.exe] mill uci`, complete the UCI
handshake, verify every required option is advertised, and apply these values
before search:

| Area | Required value |
| --- | --- |
| Search | `Algorithm=2` (MTD(f)); `go nodes N` with automatic phase depth |
| Concurrency | `Threads=1`; `UseLazySmp=false` |
| Randomness | `Shuffling=false`; `SearchShuffleSeed=42` |
| Clock | `MoveTimeMs=0`; no wall-clock search command |
| Optional data | HumanDB, perfect database, and patch/trap behavior disabled |
| Opening policy | `DrawOnHumanExperience=true`; `DeveloperMode=false` |
| Opening book | requested for a later baseline, but unavailable and disabled here |
| Search heuristics | mobility enabled; `FocusOnBlockingPaths=false`; lazy AI disabled |
| Standard pieces | 9 pieces; flying at 3; no diagonal lines |
| Variant captures | all non-standard capture variants disabled |
| Standard draws | `NMoveRule=100`; `EndgameNMoveRule=100`; threefold enabled |

All other variant values that distinguish standard Nine Men's Morris from a
Sanmill variant are also sent explicitly. Environment variables beginning
with `TGF_` are removed from the child environment so an ambient eval-weight,
database, patch, trap, SMP, or tracing setting cannot alter the run.

Before every search, the bridge queries Sanmill's exported FEN, legal actions,
repetition summary, and pinned `d` diagnostic. The diagnostic's `winner` and
`outcome_reason` fields are the authoritative game result; the bridge validates
them against the exported game-over phase and empty legal-action set. It does
not infer a winner from the compact NMM_LLM position.

`nodes N` is the deterministic work ceiling, not a promise that every position
will consume exactly `N` nodes. No positive explicit depth is sent. This is
intentional: in normal, non-developer mode, `DrawOnHumanExperience=true`
selects Sanmill's phase-aware depth table, including reduced early-placement
depth intended to avoid over-aggressive early Mill formation. The bridge
records the ceiling, selected depth, and reported node count. This smoke tests
that policy's deterministic integration; it does not establish its playing-
strength benefit. The node ceiling for a later formal evaluation remains a
product decision and is not frozen here.

## Opening-book correction and remaining interface gate

The owner wants Sanmill's opening book enabled for a later formal baseline,
but it was not enabled in this smoke. At the pinned revision the provider is
still in the Flutter layer; `tgf mill uci` advertises no opening-book option
and does not load it.

The data defect found during readiness review is closed in the pinned local
Sanmill checkout. Two independent Sanmill commits removed the occupied-`c3`
recommendation, added authoritative whole-asset legality coverage, and removed
one duplicate `c5` recommendation that otherwise altered the provider's
rank-biased sampling weight:

| Field | Audited value |
| --- | --- |
| Repair commits | `69d379a1a4e23395a45706df60f63282da20e85f`, then `6f080c5a6d15919bf0a45fa5528c45d4487a2b8f` |
| Asset | `src/ui/flutter_app/assets/opening_books/nmm/opening_book.json` |
| Asset SHA-256 | `cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5` |
| Oracle entries | 109 |
| Unique recommendations | 437 checked; 0 illegal; 0 duplicate |
| Removed-key SHA-256 | `904777ade504367c4e62446f105f1b125aaea7d6bec217984518025d8df3b0d1` |
| Removed-key presence | false |

The historical 106-position owner-reviewed corpus retains its original source
provenance. It must not be relabelled as if it had been generated from this
corrected live asset.

The remaining gate is no longer an opening-book data defect. Before formal
book use, Sanmill must expose a deterministic fail-closed UCI or referee
interface, and NMM_LLM must freeze and test the paired-opening policy against
the exact published source, binary, and asset identities. No illegal action,
database failure, malformed response, or unexplained source transition may
fall through silently to another policy.

## Provisional paired-opening diversity smoke

Engine search shuffling remains disabled. Opening diversity will instead be
introduced by a separate, seeded referee prefix sampler, so different pair
identifiers can receive different openings while any individual pair remains
exactly reproducible.

The current proposal for a later infrastructure smoke, not a frozen formal
ratio, is 75% corrected-book-derived prefixes and 25% prefixes sampled directly
from the StrictSteps perfect database's tied-best legal actions. The latter
prefix is exactly **eight logical player moves in total: four by each side,
equivalent to four full rounds, not eight rounds**. A move that forms a Mill
and its required staged removal are one logical move even though the UCI
protocol represents them as two action tokens. Implementations must therefore
count completed side-to-move changes, not raw UCI tokens. A suitable explicit
parameter name is `opening_logical_plies=8`.

The experiment identity, pair identifier, start identifier, and frozen seed
must determine the prefix. Both colour-swapped games in a pair must replay the
same complete prefix before candidate and baseline take control. MTD(f) resumes
after the prefix, still with `Shuffling=false`, one thread, fixed nodes, and a
fixed search seed. A perfect-database miss, query error, non-StrictSteps result,
or illegal selected action is a hard stop. Book coverage and book-miss
semantics must likewise be made explicit before the proposal is frozen; there
must be no hidden fallback.

Sanmill's existing `crates/tgf-cli/tests/head_to_head.rs` contains a useful
deterministic StrictSteps opening sampler, but its raw-action counter cannot be
reused unchanged because a staged removal would incorrectly consume another
opening ply.

## Random-fallback prohibition

At the pinned revision, Sanmill's optimized UCI path normally tries a depth-4
search and then `random_search` if the main search returns no move for an
ongoing position. `Shuffling=false` does not disable that failure fallback.

For this smoke the binary must therefore be built with optimization and Rust
debug assertions enabled:

```powershell
cargo --config profile.release.debug-assertions=true build --release -p tgf-cli
```

The assertion is immediately before the fallback chain. The bridge verifies
that the assertion text is present in the compiled binary. If the main search
fails, the worker assertion prevents the fallback from running; absence of a
valid `bestmove` then becomes a timeout/process failure. The bridge never
substitutes a random or locally selected move. An illegal, malformed, absent,
or unexplained `bestmove none` is a hard failure.

The pinned assertion also fires if `go` is incorrectly sent to a decisive
terminal position, because both that legitimate no-move state and an ongoing
search failure produce `MOVE_NONE` before the fallback. The bridge therefore
queries the authoritative terminal fields before every search and refuses to
send `go` after game over. A dedicated draw probe may exercise Sanmill's
zero-node `bestmove draw` short-circuit, but normal match control stops on the
authoritative terminal snapshot.

## Rule and codec checks

The validation report must include all of the following:

1. the pinned commit, clean-tree result, binary SHA-256, build command, UCI
   identity, advertised required options, and exact option contract;
2. focused Sanmill rule tests for the 100-ply no-capture rule, capture reset,
   full-history threefold repetition, standard legal actions, and immediate
   terminal import;
3. black-box UCI probes showing authoritative winner/reason values for a
   no-capture draw, threefold draw, capture reset, and fewer-than-three loss;
4. two fresh-process replays with identical options, seed, node budget, and
   initial history, whose moves, stable FENs, terminal result, and reported
   nodes are byte-for-byte equal after timing fields are excluded;
5. a Sanmill-versus-Sanmill bridge smoke of at most 60 complete player turns.
   A mill-forming turn includes its staged `x<square>` removal; both UCI
   searches and their node counts are recorded;
6. at every stable state, equality between Sanmill's primary legal-action set
   and the base move set projected from NMM_LLM's atomic legal moves, plus
   equality of the removal set while a Sanmill removal is pending; and
7. fixed-node timing samples for representative placement, movement, and
   flying positions. They inform, but do not select, the later formal budget.

Sanmill remains authoritative when a historical rule outcome is involved.
The compact NMM_LLM FEN contains no repetition window and no no-capture
counter. Consequently, a future formal start cannot be represented only by
that FEN if its prior history is meant to count. It must carry a replayable
Sanmill action prefix, or explicitly freeze reset-at-start semantics after
domain review.

## Stop conditions and claim boundary

Stop and publish a failed or blocked report on any source/binary identity
drift, unsupported option, process failure, protocol timeout, random-fallback
marker failure, illegal move, rule-probe failure, nondeterministic semantic
replay, legal-set divergence, malformed evidence, or resource anomaly.

The 60-turn ceiling is only an infrastructure and performance bound. Reaching
it is not a rules draw and says nothing about playing strength. This task must
not load the candidate bundle, run candidate-versus-Sanmill games, compute a
promotion interval, freeze the 64-position review corpus, or authorize a
formal evaluation.
