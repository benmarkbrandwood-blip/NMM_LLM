# Dev v4 Formal Paired Evaluation v1 — Blocked Freeze Record

## Status and claim boundary

Evaluation ID: `dev-v4-formal-paired-eval-v1`

Status: **fatal stop; freeze and run are not authorized**.

The managed training plan `managed-v4-baseline-v1` completed 5,000 games and
20 segments on 21 July 2026 (UTC). That is infrastructure and lineage evidence
only. It is not playing-strength evidence and does not authorize promotion.

The initial product decisions to explore `policy-argmax-v1`, use a
scratch-init comparison, and construct a corpus authorized preparation only.
Expert review has since established hard blockers and materially narrowed the
claim. The earlier 64-start freeze defaults and the one-endpoint-per-named-line
alternative are superseded.

No `EvaluationSpec` may be frozen, no paired games may be run, and no
promotion or publication decision may be made from this experiment until the
prerequisites in this document are complete and the product owner gives a new
explicit authorization.

Related contracts:

- [expert decision record](dev-v4-formal-paired-eval-v1-decision-brief.md)
- [corpus review record](dev-v4-formal-paired-eval-v1-corpus-review.md)
- [training experiment](dev-v4-malom-corrected-baseline.md)
- [managed operations](../managed-training-operations.md)
- [evaluation and promotion design](../v4-infrastructure-hardening-plan.md)

## Candidate and baseline artifacts

| Field | Value |
| --- | --- |
| Candidate source | `learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected/managed_v4_baseline_v1/segments/segment-0020/latest.pt` |
| Envelope | `checkpoint-envelope-v2` (`NMMCKP2`) |
| Training plan ID | `managed-v4-baseline-v1` |
| Training plan SHA-256 | `3f696e60c508a972dc42c79f630e90ad20e870001190321a13f0c3a12a4251c1` |
| Frozen training commit | `9ee3543195255456b2b3832f8371a8f64d25a6af` |
| Candidate bundle identity | `ab2c8f38570c14ec839d2e516732b22e1c811bf5911843b5eafee9cbaf3fb483` |
| Scratch-init bundle identity | `058145238ac03f006779689e14af35eac1d3128921d4b2cddfb698d457fdd86f` |

Both bundles passed CPU verification, including zero canary difference. Their
verification does not clear the runner or corpus blockers.

Artifact root:

`learned_ai/checkpoints/evaluation/dev-v4-formal-paired-eval-v1/`

## Expert-reviewed decision

Subject to repair, regeneration, review, and a new authorization, the proposed
Stage-0 contract is:

| Field | Proposed post-repair Stage-0 contract |
| --- | --- |
| Purpose | Training-signal diagnostic under a deterministic feature ablation |
| Protocol | Fixed-N paired colour swap; schema `nmm.paired-evaluation.v1` |
| Starts | 109 unique playable NMM FENs projected from 110 raw Sanmill move-oracle keys |
| Phase coverage | Placement 109 / movement 0 / flying 0 |
| Pairs / games | `109` pairs / `218` games; exactly one pair per unique start |
| Seed | `42`, recorded for provenance but not used by deterministic move selection |
| `max_ply` | `200`; overflow scored as a draw after draw lifecycle is repaired |
| Route name | `policy-argmax-v1` |
| Work budget | `{"lookahead_rollouts_per_move": 0}` |
| Components | Sentinel, ValueNet, GapNet, HumanDB override, and SpecialistDB override absent |
| Opponent | Verified architecture-matched `scratch-init-v1` bundle |
| Rules | `nmm-v4-corrected` |
| Result rule | Lower interval bound `> 0`: accept; upper bound `< 0`: reject; otherwise inconclusive |

This table is a review target, not a frozen spec or launch authorization. No
final Oracle corpus file or corpus SHA-256 exists yet.

## Why this is Stage 0, not a formal strength gate

Training passed a real `LookaheadAdvisor` into the feature encoder. The
current paired route passes `lookahead_advisor=None`, which zeroes the
72-feature lookahead block. This is an input-route ablation, not merely a
smaller search budget.

The route can test whether the trained policy retains a detectable signal
against scratch initialization under the same ablation. It cannot fairly
attribute an inconclusive or rejected result to lack of learning, because the
candidate is evaluated off its training input route.

The scratch bundle is likewise a training-gain control. Beating random
initialization does not establish useful product strength.

The proposed Oracle corpus is also narrow:

- all starts are placement phase, with 0–16 pieces placed;
- there are no movement or flying starts;
- 28 of 109 Oracle orbits overlap a Sanmill named-line trajectory;
- 23 of 109 overlap within the first eight plies;
- 100 of 107 Sanmill named lines exactly match the local training opening
  pool, and all 11 curated lines match.

Sanmill describes the move Oracle as separately engine-derived, so it is not
correct to call it a direct training-book export. The measured overlap and
early-placement distribution nevertheless mean it is not demonstrated
held-out or training-disjoint. It must be described as a
source-overlapping, in-distribution-adjacent convenience corpus.

## Fatal blockers

| Blocker | Evidence and required disposition |
| --- | --- |
| Engine-level draws crash the runner | Repetition and 50-move draws can set `finished=True, winner=None`. The runner does not exit and the next move raises `ValueError: Game is already over.` Repair and focused regression tests are mandatory. |
| A crash can strand the ledger | The games ledger is opened with exclusive creation. Define and test partial-ledger cleanup, quarantine, or explicit resume/restart semantics. |
| Deterministic start reuse falsifies the nominal sample size | Pure argmax plus modulo start selection repeats identical pairs. Set `pairs == unique starts`; never reuse starts to narrow the interval. |
| Named-line endpoints are ambiguous | 49 of 107 lines have 2–42 legal endpoints because removal choices are omitted; one line fails replay and one endpoint is terminal. Do not freeze synthetic one-per-line endpoints. |
| The 64-start draft is invalidated | It has 64 FENs but 63 symmetry orbits and was an arbitrary narrow slice. Preserve only as rejected historical evidence. |
| Oracle facts were overstated | 110 raw keys project to 109 unique playable FENs, all placement phase. Generate and review the exact replacement corpus. |
| Route is not training-aligned | The 72 lookahead features are zeroed at evaluation. Keep the claim at Stage 0 and build a separately frozen aligned evaluator for strength. |
| Freeze state was not reproducible at review time | The audit began with the branch ahead of `origin/dev` and relevant documents and draft artifacts uncommitted. Recheck the live state and freeze only from a clean, tracked commit. |

The existing focused bundle/evaluation/lifecycle tests pass, but they do not
exercise the engine-level draw transition or ledger retry behavior and
therefore do not clear the fatal stop.

## Statistical interpretation

Move choice is deterministic. For a fixed bundle, route, and start, a repeated
colour-swapped pair contains no new information. The earlier 64-start /
256-pair proposal copied every result four times and would have understated
uncertainty.

For Stage 0, use one pair for each of the 109 unique starts. The normal
interval over pair-score differences is an engineering summary of variation
across this fixed convenience corpus. It is not automatically a population
confidence interval; any stronger interpretation requires separately justified
sampling assumptions.

An inconclusive result remains inconclusive. A later v2 may be separately
preregistered and frozen with additional unique starts or a route-aligned
evaluator. It must have its own corpus hash and analysis contract, and its
observations must not be pooled with v1 as though they were one prespecified
sample.

## Completed preparation

1. The 5,000-game managed run completed with recorded checkpoint lineage.
2. Candidate and scratch-init bundles were exported and CPU-verified.
3. The historical 64-position draft was generated and audited.
4. Named-line endpoint ambiguity, Oracle projection, phase coverage, corpus
   overlap, runner determinism, draw lifecycle, and feature-route mismatch
   were independently checked.

## Mandatory sequence before any freeze or run

1. Repair draw completion and ledger restart/recovery behavior.
2. Add focused tests for repetition draw, 50-move draw, and partial-ledger
   retry semantics.
3. Generate an Oracle-only list of exactly 109 unique playable positions.
4. Validate exact uniqueness, 16-way orbit uniqueness, playability, phase
   counts, provenance, and overlap; record a new
   `start_positions_sha256`.
5. Complete owner review of the exact list.
6. Record the final route, corpus, bundle identities, work budget, interval
   interpretation, and non-claims in a clean tracked commit.
7. Re-run the focused verification required by the readiness workflow.
8. Request a new explicit product authorization for freeze and run.

Only after those steps may an immutable `EvaluationSpec` be created. The
current document intentionally contains no approved freeze or run command.

## Explicit non-claims

Regardless of a future Stage-0 result:

- training metrics are not promotion evidence;
- the random three-position infrastructure smoke is not a formal corpus;
- the rejected 64-position draft is not a frozen corpus;
- the Oracle corpus is not held-out evidence;
- `policy-argmax-v1` is not training-route-aligned strength evidence;
- an accepted Stage-0 result is not release or publication approval;
- `promotion_allowed=false` remains in force.
