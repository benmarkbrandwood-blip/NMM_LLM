# Sanmill-corrected retained-v2 held-out evaluation v1

## Status and purpose

Plan ID: `sanmill-corrected-retained-v2-heldout-eval-v1`

Evaluation ID: `dev-v4-sanmill-corrected-retained-v2-heldout-v1`

Status: `frozen_awaiting_runner_and_final_preflight`

Plan identity:
`212076e9423b671b83783efef411db3b4a56c8c67ae36a463d381d6939d4d982`

This contract freezes the first independent post-training match for the
completed retained-v2 candidate. It measures the candidate against the same
fixed 500,000-node Sanmill teacher used at the endpoint of the training
curriculum. It does not reopen training and does not itself authorize model
promotion or publication.

The machine-readable contract is
[`sanmill-corrected-retained-v2-heldout-eval-v1.json`](sanmill-corrected-retained-v2-heldout-eval-v1.json).
Its separate authorization must bind the final plan identity, file hash and
plan commit before a candidate game may start.

## Why this is independent, and its limit

The 64 exact twelve-ply histories were source-frozen on 1 August, before this
candidate was trained. They were not selected from candidate outputs and were
not used by the 29-state segment health gate. Each history is replayed exactly
and used once as a colour-role-swapped pair.

“Held out” does not mean that every final board is absent from every
trainer-visible data source. The source-only
[exposure audit](../evidence/sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.md)
found 30 HumanDB D4 matches and one final SpecialistDB D4 match. The complete
64-start result is therefore the operational full-route benchmark. The same
ledger also has a frozen 34-start sensitivity subset with no HumanDB or final
SpecialistDB D4 match. Both are fixed now; no result-dependent exclusion is
allowed.

The old 29-position phase corpus is not a held-out strength set because it was
queried after every training segment and controlled whether a segment could
become a resume parent. It remains anti-collapse evidence only and is excluded
from this match.

## Candidate

The candidate is the final version-2 checkpoint from the completed
`managed-sanmill-corrected-retained-v2` lineage:

- checkpoint SHA-256
  `df00861a5ced53b6c9b16ed89f2762d41a82f1d74fce970b5d0bdf6adba4ac4d`;
- checkpoint payload SHA-256
  `8b4017ce856012fa3c4d578c56c5f32a6d5ebae97b9f17c6cbd2c5228146de19`;
- route-bundle identity
  `c2652119b64a2808ebcd5e7dc661873f3f897065b7d529bd9e261328f0981f23`;
- final policy model identity
  `8347d1e3927847f6b67360af89eb8f66f22bda6716ba03eeacedad288dcb89ea`;
- frozen-target model identity
  `e17442f2c0e689b56069f15f3b57d6164a1691f386c9f6ffdb4d89564fff6e1`;
  and
- frozen-target age: 50 games.

The route is deterministic CPU float32 policy argmax over the exact
`s-gen-v2-training-aligned-v1` 134-feature input. It uses the final frozen
target for learner-side lookahead continuation, HumanDB top frequency with
historical heuristic fallback for opponent continuation, corrected Malom
terminal early exit, and the final SpecialistDB counterfactual features.
Sentinel, ValueNet and GapNet remain disabled. All resources open read-only.

The final checkpoint's HumanDB, SpecialistDB, Malom, rules, MIF and Sanmill
identities must match before the model is loaded. The candidate has no encoded
repetition/no-progress history input; this is a documented architecture fact,
not permission to bypass the history-bearing referee.

## Baseline and rules owner

Sanmill is both the opponent implementation and the sole match referee. The
isolated runtime remains pinned to:

- commit `a6623f88959f7453594df274fbe1f128af7ff55e`;
- tree `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`;
- binary SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`;
- runtime identity
  `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`;
  and
- strict-referee semantic digest
  `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.

Search uses one thread, MTD(f), IDS, shuffling off, seed 42, no wall-clock
limit and a 500,000-node ceiling for every complete logical turn. No explicit
depth is sent, so the already tested `DrawOnHumanExperience=true` placement
depth policy remains active. Actual nodes and completed depth are recorded per
turn. A ceiling is not represented as exact node consumption.

Strict failure policy forbids Perfect DB, opening-book, HumanDB, patch/trap,
depth-4 or random failure substitution. The fixed 500,000-node choice is the
measured final training-teacher configuration. An uncalibrated 1,000,000-node
opponent or a wall-clock budget would answer a different question and is not
silently substituted.

MIF Suite tag `mif-suite-1.0`, its release commit and Suite digest, training
rules semantic digest, and Sanmill strict-referee identity are all part of the
evaluation specification.

## Corpus and paired workload

The executable corpus identity is
`417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.
It contains 22 Book, 21 genuine frozen-sample HumanDB and 21 StrictSteps
Perfect DB histories. Every record is twelve complete logical plies, `[6, 6]`
by side, and the 64 final FENs and D4 structures are unique.

For every record:

1. start a fresh strict Sanmill process;
2. replay and verify the complete prefix history and final history SHA-256;
3. play one game with the candidate controlling White;
4. start another fresh process and replay the identical history; and
5. play one game with the candidate controlling Black.

This produces exactly 64 independent pair units and 128 deterministic games.
Repeating a pair would add no information and is forbidden. There is no
runtime prefix fallback or source resampling.

Rules draws are whatever the pinned strict referee declares, including
threefold repetition and the 100-logical-ply no-progress rule. The safety cap
is 1,536 post-prefix logical plies. Reaching it makes the evaluation incomplete
and invalid; it is never scored as a draw. Candidate errors, illegal actions,
history divergence, non-finite tensors and Sanmill failures likewise stop
without manufacturing a game result.

## Frozen analysis

Candidate game score is 1 for a win, 0.5 for a rules draw and 0 for a loss.
For each start, the pair-score difference is:

```text
candidate score as White + candidate score as Black - 1
```

The primary summary is the mean across 64 unique starts with the existing
two-sided 95% normal engineering interval (`z=1.96`). It is a fixed-corpus
variation summary, not a population-confidence claim.

- lower bound above 0: `candidate_ahead`;
- upper bound below 0: `candidate_behind`;
- otherwise: `inconclusive`; and
- any missing, truncated or integrity-invalid game: `invalid`.

The neutral point is 50%, not 55%. A separate statement that the candidate is
stronger than this Sanmill baseline is allowed only for
`candidate_ahead`. An inconclusive result stays inconclusive.

The report must also give support counts and results by Book/HumanDB/Perfect
DB source, candidate colour, termination reason, and 34-start strict
independence membership. The strict subset uses the same already completed
games; it is not an extra sample. Raw game records, actual nodes, depths and
turn counts remain available beside all summaries.

Only one retained training seed exists. Runtime seed 42 and deterministic
resampling do not create additional trained seeds. The result cannot establish
cross-seed stability, and training curves or the final 200-game tail must be
reported separately from this held-out ledger. No causal claim or ablation
claim is part of this v1 match.

## Resource and restart boundary

- at most 128 completed games;
- at most six active evaluator hours;
- one CPU evaluator process and one strict Sanmill process at a time; and
- an ordered, hash-chained partial ledger fsynced after every game.

A host interruption may resume only the missing suffix of the exact same
specification after validating the complete partial prefix. It is not a new
run and may not overwrite or replay completed games. Resource exhaustion or a
semantic failure stops for diagnosis. There is no result-based early stop.

## Required implementation and final preflight

No candidate game may start until a focused runner implements and tests:

- exact full-history prefix replay in two fresh processes per pair;
- candidate/Baseline role swapping without colour or history mutation;
- strict state agreement after every complete logical turn;
- rules-only terminal scoring and non-draw safety-cap handling;
- actual-node/depth and termination-reason evidence;
- atomic, no-overwrite, hash-chained ledger resume;
- independent recomputation of complete, per-stratum and strict-subset
  results; and
- fail-closed candidate, database, runtime, corpus and specification identity
  checks.

The final read-only preflight must additionally show a clean published
`dev == origin/dev`, the exact plan and authorization, verified ignored bundle,
read-only database identities, pinned Sanmill runtime, absent output targets,
no competing trainer/evaluator process, passing focused tests, and the
mandatory Malom/label-provenance suite. It may run canaries and synthetic
protocol probes but must not consume a corpus game.

## Claim and authorization boundary

This contract freezes the protocol before any candidate game is viewed. The
separate product-owner authorization may permit exactly one complete run plus
safe same-spec resume after the gates above pass. It cannot authorize changed
starts, thresholds, baseline work, route, components, rules, adjudication or
resource expansion.

Completion will be Candidate interoperability and fixed-corpus strength
evidence only. It is not automatic promotion, publication, a multi-seed
result, a full MIF conformance claim, or authority to retrain.
