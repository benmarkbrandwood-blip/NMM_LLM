# Retained-v3/v4 high-precision held-out score v1

Status: `frozen_awaiting_product_authorization`

The product owner selected the 253-start / 1,012-game fixed-width option with
a target 95% engineering half-width of at most 1.5 percentage points. This
document records that decision and the resulting immutable evaluation plan.
It is not launch authority. No candidate game has been played under this plan.

The canonical [machine plan](sanmill-retained-v3-v4-heldout-score-v1.json) has
identity
`6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714`
and file SHA-256
`7e523265bc8d0b6ae5d861919033a86af2bbdb9238d54991f57d57d8161a0f24`.
It binds implementation commit
`5eb142383f710c17377deedc8b1cfcc5287daa02` and is tracked by plan commit
`592950062515f697b52c8e8d355aa091da4cc839`. The plan JSON is frozen and
must not be rewritten to record later status.

## Product interpretation

The selected budget asks a narrow question: on this fixed, candidate-blind,
training-disjoint prefix, what is the mean retained-v4 minus retained-v3 game
score difference against the same pinned Sanmill opponent?

The target `1.5pp` is an interval-width target, not an equivalence margin and
not a promise that the result will separate the candidates. If the interval
crosses zero, the result is `inconclusive`. If the observed start-level
variation makes the half-width exceed 1.5pp, the result is
`inconclusive_precision`. Neither outcome may be relabelled as equivalence.

## Frozen held-out prefix

The source pool identity is
`2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7`,
with records identity
`4e5f9ecf7508a995b74af6a36bcf966c89d9141940770ebb21c3629446830a31`.
The plan takes exactly the first 253 records in its frozen master order. The
ordered prefix-record identity is
`99951a691c106a86aa5e4affc16ced2b63866e2dd589379527d068b022003c7b`.
It contains 99 placement, 98 movement and 56 flying starts.

The records were selected without loading either candidate policy or reading
candidate outcomes. They are source-game-unique and `ring16`-unique, absent
under the recorded exposure checks from both retained routes' active training
stores, and strictly replayable with complete rule history. The order and
253-record prefix must not change after this product choice.

## Primary score contract

| Field | Frozen value |
| --- | --- |
| Independent units | 253 starts |
| Candidate colours | White and Black for every start |
| Candidates per colour unit | retained-v3 and retained-v4 |
| Total games | 253 × 2 × 2 = 1,012 |
| Score | win 1, draw 0.5, loss 0 |
| Start value | average the v4-minus-v3 score difference over both colours |
| Primary estimate | mean of the 253 start values |
| Interval | two-sided normal engineering interval, `z=1.96` |
| Maximum target half-width | 0.015 |

The directional decision is allowed only when every game reaches a strict
rules terminal, the half-width is at most 1.5pp, and the interval lies wholly
above or wholly below zero. Crossing zero is `inconclusive`. Exceeding the
width target is `inconclusive_precision`. Any safety-cap game makes the
primary result `inconclusive_incomplete_safety_cap`; a cap is never a draw.

The interval describes these two named routes on this fixed held-out corpus.
It is not a population variance guarantee, general Elo or universal playing-
strength claim. The routes differ in seed, source revision, target age and
accumulated SpecialistDB, so the result cannot identify a refresh effect.

## Protocol and secondary evidence

- replay each complete variable-length history and verify FEN, history hash,
  rule clocks, logical ply and nonterminal status before loading a candidate;
- use deterministic CPU float32 policy argmax over each exact training-aligned
  route;
- pair adjacent v3 then v4 games inside every start/colour unit;
- use pinned strict Sanmill with one thread, MTD(f), IDS, shuffle off, seed 42
  and at most 500,000 nodes per complete logical turn;
- continue to a strict rules terminal or the invalid 1,536-post-start safety
  cap; do not stop or extend based on interim results; and
- permit only separately authorized, same-spec, missing-suffix exact resume
  after a host interruption. There is no automatic retry or semantic-failure
  recovery.

Survival through 108 post-start plies, no-capture and repetition state,
length, termination reason, phase, Malom coverage and move deltas remain
secondary process descriptions. They may explain a score result but may not
replace the frozen paired-score primary after outcomes are known. Malom is
history-free and cannot replace the strict referee.

## Frozen inputs and resource ceiling

The ignored successor input root is
`learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-heldout-score-v1/inputs`.
Its snapshot identity is
`1ceb11ce5cec1ff44a9c1f03d69a961f191bdd951d7fe31ef078bb40bf2874c3`;
the canonical manifest SHA-256 is
`2918d2de63001135de33c1296656d49aefb8f182e721472c5dfc643378004b92`.
The route-bundle identities remain `b6d7ecf6...` and `817d2e36...`, and the
sidecar-free SpecialistDB SHA-256 values remain `82d7fbcd...` and
`3d69d1ac...`. These successor-owned copies are read-only evaluation inputs,
not new training lineages.

The immutable ceiling is:

- at most 1,012 games and four active evaluator hours;
- at most 777,216 Sanmill search turns and a summed theoretical node ceiling
  of 388,608,000,000;
- one evaluator and one strict Sanmill process at a time;
- no training, policy or database update, expansion, automatic retry,
  semantic recovery or result-based early stop; and
- no equivalence, Elo, population, refresh-causal, promotion, publication or
  release action or claim.

## Source readiness and launch gate

Two complete fresh preflights on published clean `dev` produced the same
source-readiness identity
`f233c991aa66a8699fac8952fd0c758a5fabb09de7a0e66ba3043635934b2b08`.
All technical gates passed, including the exact plan, fresh output namespace,
successor inputs, frozen prefix and schedule, candidate routes, pinned
Sanmill, all 253 strict history replays, process ownership, focused tests,
mandatory Malom/provenance tests and Ruff. Both preflights requested zero
candidate moves from the held-out corpus and played zero games.

Product authorization bound to the exact plan and source-readiness identities
is absent. Therefore the current verdict is `needs_decision`: technically
ready for a plan-bound launch decision, but not authorized and not launched.
See the
[readiness evidence](../evidence/sanmill-retained-v3-v4-heldout-score-readiness-2026-08-14.md).
