# Dev v4 Formal Paired Evaluation v1 — Stage-0 Execution Record

## Status and claim boundary

Evaluation ID: `dev-v4-formal-paired-eval-v1`

Status: **completed; protocol decision `accepted`; no strength or promotion
claim**.

The managed training plan `managed-v4-baseline-v1` completed 5,000 games and
20 segments on 21 July 2026 (UTC). That is infrastructure and lineage evidence
only. It is not playing-strength evidence and does not authorize promotion.

The initial product decisions to explore `policy-argmax-v1`, use a
scratch-init comparison, and construct a corpus authorized preparation only.
Expert review has since established hard blockers and materially narrowed the
claim. The earlier 64-start freeze defaults and the one-endpoint-per-named-line
alternative are superseded.

The paired-runner prerequisites identified by that review are now repaired:
engine-level repetition and 50-move draws end the game, while an interrupted
run retains a validated `.partial` ledger and resumes only its missing games.
Malformed or mismatched partial evidence fails closed. The focused evaluation
suite passes all 15 tests. The owner reviewed the 107 generated candidates,
requested removal of original review position 101, and accepted the remaining
106. The regenerated review package carried
`owner_review_complete_not_frozen` before the selected list was bound into the
immutable Stage-0 spec.

A read-only audit from clean commit `b92d62e` reverified the corpus and both
bundles, confirmed isolated output targets, and constructed the complete
Stage-0 specification in memory without writing it. New specifications bind
the clean Git commit, selected device, platform and PyTorch identity, float32
precision, route, disabled components, and zeroed lookahead block; execution
fails closed on drift. Legacy unbound specifications remain readable and
recomputable but cannot create new game evidence. The combined focused
readiness suite passes 28 tests. On 23 July 2026, the product owner explicitly
instructed Codex to start the Stage-0 evaluation. The technical and product
launch gates therefore passed for this contract only. The immutable spec was
frozen from commit `a45b44e`, and all 212 games completed with a verified final
ledger.

The one-run authorization has been consumed. It does not authorize training,
a rerun, a changed evaluation, or any promotion or publication decision.

Related contracts:

- [expert decision record](dev-v4-formal-paired-eval-v1-decision-brief.md)
- [corpus review record](dev-v4-formal-paired-eval-v1-corpus-review.md)
- [training experiment](dev-v4-malom-corrected-baseline.md)
- [managed operations](../managed-training-operations.md)
- [evaluation and promotion design](../v4-infrastructure-hardening-plan.md)
- [readiness evidence](../evidence/dev-v4-stage0-readiness-2026-07-22.md)
- [completed result evidence](../evidence/dev-v4-stage0-result-2026-07-23.md)

## Completed Stage-0 result

The frozen spec identity is
`26f80c14d70320aa025c85319791c625e821babb2e542095aeb4711d4c11d48b`.
Independent ledger recomputation reports 193 candidate wins, 8 draws, and 11
losses over 212 games. The pair-score-difference mean is
`0.8584905660377359`, with the frozen interval
`[0.7972174156720373, 0.9197637164034345]`; the protocol decision is
`accepted`.

The final ledger SHA-256 is
`6800d95ca8e968e2b7ded3f02b87451e1a84c7f47a068a72e36d53bcd1978848`,
and the result identity is
`0280a4a8eee8ee39506dcef0816a3265bc64229985699c63a3420e34978bbc99`.
The `.partial` file was atomically removed. This is only positive
training-signal evidence under the documented ablation and random-init
control; it is not formal playing-strength evidence.

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
verification alone did not clear the corpus or freeze gates; the later
readiness audit reverified them as part of the complete contract. The later
explicit authorization was consumed by the completed run.

Artifact root:

`learned_ai/checkpoints/evaluation/dev-v4-formal-paired-eval-v1/`

## Expert-reviewed decision

After the final clean-state and absent-output checks passed, the completed
Stage-0 contract was:

| Field | Completed Stage-0 contract |
| --- | --- |
| Purpose | Training-signal diagnostic under a deterministic feature ablation |
| Protocol | Fixed-N paired colour swap; schema `nmm.paired-evaluation.v1` |
| Starts | 106 owner-accepted unique playable stable NMM FENs selected from 107 FENs projected from 108 `action=p` keys; original review position 101 excluded; two `action=r` keys retained only as successor provenance |
| Phase coverage | Placement 106 / movement 0 / flying 0 |
| Pairs / games | `106` pairs / `212` games; exactly one pair per unique start |
| Seed | `42`, recorded for provenance but not used by deterministic move selection |
| `max_ply` | `200`; overflow scored as a draw |
| Route name | `policy-argmax-v1` |
| Work budget | `{"lookahead_rollouts_per_move": 0}` |
| Components | Sentinel, ValueNet, GapNet, HumanDB override, and SpecialistDB override absent |
| Opponent | Verified architecture-matched `scratch-init-v1` bundle |
| Rules | `nmm-v4-corrected` |
| Result rule | Lower interval bound `> 0`: accept; upper bound `< 0`: reject; otherwise inconclusive |

This table records the consumed contract; it is not authority for a rerun. The
[106-FEN list](dev-v4-formal-paired-eval-v1-start-positions.json) and
[audit/PNG record](dev-v4-formal-paired-eval-v1-corpus-review.md) are bound by
the immutable Stage-0 spec identity recorded above.

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
- 28 of 106 selected Oracle orbits overlap a Sanmill named-line trajectory;
- 23 of 106 overlap within the first eight plies;
- one of 439 source move recommendations is illegal after projection; its
  playable source candidate is the owner-excluded original position 101;
- 100 of 107 Sanmill named lines exactly match the local training opening
  pool, and all 11 curated lines match.

Sanmill describes the move Oracle as separately engine-derived, so it is not
correct to call it a direct training-book export. The measured overlap and
early-placement distribution nevertheless mean it is not demonstrated
held-out or training-disjoint. It must be described as a
source-overlapping, in-distribution-adjacent convenience corpus.

## Prerequisite and blocker status

| Item | Status | Evidence and required disposition |
| --- | --- | --- |
| Engine-level draws crash the runner | Cleared in the runner | The runner now exits on `engine.finished`, preserves `winner=None`, and records the engine's repetition or 50-move draw reason. Both paths have focused regressions. |
| A crash can strand the ledger | Cleared in the runner | Games are written and fsynced to `<output>.partial`. A same-spec, ordered, hash-valid prefix resumes only missing games; a complete ledger is recomputed and atomically published. Malformed evidence is retained and rejected. |
| Deterministic start reuse falsifies the nominal sample size | Cleared in contract and code | Pure argmax plus modulo start selection repeats identical pairs. The specification now rejects duplicate starts and any pair count above the number of unique starts; Stage 0 uses exactly 106 pairs. |
| Named-line endpoints are ambiguous | Rejected as a corpus source | 49 of 107 lines have 2–42 legal endpoints because removal choices are omitted; one line fails replay and one endpoint is terminal. Do not freeze synthetic one-per-line endpoints. |
| The 64-start draft is invalidated | Rejected historical evidence | It has 64 FENs but 63 symmetry orbits and was an arbitrary narrow slice. Preserve only as rejected historical evidence. |
| Oracle facts were overstated | Corrected and owner-reviewed artifact generated | Of 110 raw keys, 108 are stable `action=p` keys yielding 107 exact/ring16-unique candidates. The owner excluded original position 101 and accepted 106; two `action=r` keys are pending removals whose stable successors duplicate selected starts. |
| Owner corpus review | Cleared | The owner completed all 107 candidates, recommended removing original 101, and accepted the other 106. The exclusion and its source identity are part of the reproducible artifact. |
| Route is not training-aligned | Stage-0 claim boundary recorded | The 72 lookahead features are zeroed at evaluation. Keep the claim at Stage 0 and build a separately frozen aligned evaluator for strength. |
| Freeze state was not reproducible at review time | Cleared by read-only audit | Commit `b92d62e` was clean when corpus, bundles, targets, runtime identity, and an in-memory specification were reverified. The freeze command repeats the clean check and binds its then-current commit. |
| Runtime identity and route were descriptive only | Cleared in contract and code | New specifications bind clean Git, CPU/CUDA identity, platform, PyTorch, float32, route, disabled components, and the zeroed lookahead block. A run fails before model loading or evidence writes on drift; a legacy unbound specification cannot run. |
| Product launch decision | Cleared | The product owner explicitly instructed Codex to start this exact Stage-0 evaluation on 23 July 2026. |

The focused command
`python -m pytest tests/test_paired_evaluation.py -q` now reports `15 passed`,
covering both engine-level draw transitions, valid partial-ledger resume and
atomic publication, fail-closed malformed partial evidence, deterministic
start reuse, and runtime binding. The combined candidate-lifecycle, corpus,
paired-runner, and bundle command reports `28 passed`. This clears the
technical runner and freeze prerequisites. The separate product authorization
was later recorded, consumed by one run, and cannot be reused.

## Statistical interpretation

Move choice is deterministic. For a fixed bundle, route, and start, a repeated
colour-swapped pair contains no new information. The earlier 64-start /
256-pair proposal copied every result four times and would have understated
uncertainty.

For Stage 0, use one pair for each of the 106 unique starts. The normal
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
5. The owner-reviewed 106-position replacement, freeze-compatible list, 106
   individual PNGs, nine contact sheets, and hash manifest were regenerated and
   audited after excluding original review position 101.
6. The specification now prevents deterministic start reuse and binds the
   clean code, device, runtime, route, component, and feature contracts.
7. The read-only readiness audit reverified both bundles, the corpus, output
   isolation, runtime identity, and the complete in-memory specification.
8. The authorized spec was frozen, all 212 games completed, and independent
   recomputation reproduced the runner's result and evidence identities.

## Completed execution and post-run boundary

The mandatory sequence completed without a stop condition:

1. The post-authorization tree was clean and every target was absent.
2. One immutable specification was frozen on CPU.
3. Its 212-game ledger completed and was atomically published.
4. Independent recomputation reproduced the same result.

No parameter substitution, second run, promotion, or publication is
authorized. Preserve the spec and ledger together under their recorded
identities.

## Explicit non-claims

Regardless of a future Stage-0 result:

- training metrics are not promotion evidence;
- the random three-position infrastructure smoke is not a formal corpus;
- the rejected 64-position draft is not a frozen corpus;
- the Oracle corpus is not held-out evidence;
- `policy-argmax-v1` is not training-route-aligned strength evidence;
- an accepted Stage-0 result is not release or publication approval;
- `promotion_allowed=false` remains in force.
