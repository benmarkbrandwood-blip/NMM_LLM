# Retained-v3/v4 phase-process generalization v1 result — 14 August 2026

Status: `completed_process_generalization_inconclusive_not_held_out`

Decision status: `needs_decision`

This is the completion record for the named retained-v3 and no-refresh-v4
routes on the frozen 39-start phase-process corpus. The corpus was previously
visible to the project. This result is therefore fixed-corpus process evidence,
not held-out evidence, a playing-strength result, a target-refresh causal
experiment, an equivalence result, or authority for another game, training,
promotion, publication or release.

## Authority, execution and identities

| Item | Recorded value |
| --- | --- |
| Frozen plan identity | `4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256` |
| Frozen plan file SHA-256 | `09245e5f66af3d18ba2818d1dfac70b4c7eec8d63c9388d501b32846dfccf9d3` |
| Product-authorized source readiness | `0ff79e398233c7ed9fcdec4cc5cd406837330140a3c1cec720e11eaa274ae365` |
| Post-authorization launch readiness | `aeea1625ce38607aea4c08a0c0f363ba0c1d29c7584d4f498f562b6300f58677` |
| Authorization identity | `ceedd13ae9abcce8b8f9e5103488057114408e887e5bd946b95041fc781faebb` (`product-owner-direct`) |
| Runtime spec identity | `bb349a96df3e8445d3687c7c24dc474fe595d63aa890085ed6c6b2a94574fe72` |
| Launch identity | `827820192f3e694d374b94c918f6410404437410715587dcd788d951bb5e4dc3` |
| Primary report identity | `6007af186b9a7ce908416f4578ebc31c0c19fc27733c32ed44751bb39cc3c812` |
| Mechanism report identity | `afcdff218b8dfd47bb17c3f0438a9e3fd9e1298547a9c205e1949c5e26c97562` |
| Completion identity | `48ac2ad4c6abc79b69c7de597ad46a5197949b4dcdee1f962e621d1be2fc57c8` |
| Ledger SHA-256 | `45506e5cedf5ab9bdcba9dd687349869b639fb8bd46fd8990cbaf4bb79ef3211` |
| Ledger tail record SHA-256 | `0c8ee2062fb1f02a6bfdf293875bcd6060585277ccab6588a14daf8baa8101a7` |
| Primary report file SHA-256 | `1bf9980698f95a8e47fbabc78f152a9788d105992eecb48f286c2111e70c0b23` |
| Mechanism report file SHA-256 | `f2388702d82e28c95b6da101da326d5b0fc792c86d29946d695cb148873bfd96` |
| Completion file SHA-256 | `978bb873b2979cc9d2f215761a4879667465b9fbddc844fb0017f59328b01fef` |

The launch ran once from clean published `dev` at
`52a3240ee824e3c608450027c596bc3332ea3748`. It opened the first fixed-corpus
game at `2026-08-13T17:00:22.572527Z` and completed at
`2026-08-13T17:06:58.362540Z`. All 156 planned games completed. Evaluator
active time was 399.619311 seconds (0.1110 hours), below the two-hour ceiling.
There was no retry, resume, semantic recovery, extension, training, optimizer
update, database or checkpoint mutation, held-out or strength claim,
promotion, publication or release. The authorization is consumed.

## Predeclared primary result

The primary endpoint was whether a game remained strict-rules ongoing through
108 additional logical plies after its frozen history start. Continuation
survival is neither a draw nor a strength metric.

| Candidate | Survived / games | Survival rate |
| --- | ---: | ---: |
| retained-v3, refresh every 50 | 17 / 78 | 21.7949% |
| retained-v4, no refresh | 15 / 78 | 19.2308% |

The predeclared start-clustered v4-minus-v3 difference was `-0.0256410`
(-2.5641 percentage points). Its fixed-corpus engineering interval was
`[-0.0607070, +0.0094249]`, with half-width `0.0350659`. Precision was within
the frozen 0.10 maximum, but the interval crosses zero, so the frozen decision
is `inconclusive`.

This does not reproduce the earlier reused-opening-corpus direction of higher
v4 survival. The point estimate is in the opposite direction, but the interval
does not establish higher v3 survival either. The justified result is
inconclusive process generalization for these named routes on this corpus.

## Phase strata and terminal outcomes

Phase results are descriptive; no phase-specific directional gate was frozen.

| Frozen start phase | Games per candidate | v3 survived | v4 survived |
| --- | ---: | ---: | ---: |
| Placement | 36 | 11 (30.5556%) | 10 (27.7778%) |
| Movement | 28 | 6 (21.4286%) | 5 (17.8571%) |
| Flying | 14 | 0 (0%) | 0 (0%) |

All 156 games reached a strict rules terminal and none reached the
1,536-post-start safety cap. Eventual W/D/L is secondary and descriptive.

| Candidate | W / D / L | Score | Fifty-move | Threefold | Fewer than three | No legal moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retained-v3 | 11 / 45 / 22 | 42.9487% | 18 | 27 | 17 | 16 |
| retained-v4 | 12 / 44 / 22 | 43.5897% | 14 | 30 | 19 | 15 |

Post-start length averaged 50.5000 plies for v3 and 45.8718 for v4; medians
were 29 and 24.5, and maxima were 233 and 213. The paired restricted-length
difference was `-0.0030132`, interval `[-0.0065535, +0.0005272]`, and is
inconclusive. Unlike the earlier reused-opening corpus, this corpus did not
show more v4 fifty-move terminals: counts were 14 for v4 and 18 for v3.

At the relative horizon, history-free Malom classified the 17 surviving v3
positions as one theoretical win and 16 draws, and all 15 surviving v4
positions as draws. Neither set contained a theoretical loss. The survivor
sets differ because the routes selected different paths, and the counts are
small. This is not a paired or history-aware outcome comparison.

## Zero-new-game mechanism report

The completion controller reaggregated the same 156-row ledger into the
mechanism report; `new_games=0`. All 1,963 queryable v3 candidate turns and all
1,783 queryable v4 candidate turns selected a coarse Malom-WDL-preserving
action. V3 selected a preserving capture on all 156 observed safe-capture
opportunities; v4 did so on all 155. No missed-safe-capture mechanism was
observed.

The start-clustered exploratory board-revisit-share difference was
`+2.1487pp` for v4 minus v3, interval `[+0.2565pp, +4.0408pp]`. This metric was
explicitly labelled `exploratory_no_directional_gate`; the interval is useful
for designing a future preregistered mechanism test but is not a post-hoc
directional or causal decision.

The start-clustered full-order regret difference was `+1.7401pp`, interval
`[-0.9551pp, +4.4353pp]`. The safe-capture-opportunity-share difference was
`-0.5988pp`, interval `[-2.2431pp, +1.0455pp]`. Both cross zero. Opportunity
exposure also differs between routes, so conditional denominators must not be
read as interchangeable route quality measures.

## Verification and preservation

The primary and mechanism reports were independently recomputed from the
completed ledger and matched their persisted JSON exactly. A post-completion
input audit reproduced snapshot identity
`b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac`.
Both route identities and both SpecialistDB main-file hashes were unchanged;
the databases remained read-only and sidecar-free. No `failure.json` or
evaluator lock remained, and the repository remained clean after execution.

Preserve the entire ignored output namespace at
`learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-phase-process-generalization-v1`.
Do not rerun, extend, repair, resume or relabel it held out. The frozen plan
JSON remains unchanged so its identity continues to bind the consumed direct
authorization.

## Decision and next validation

The preregistered process-generalization question is answered as
`inconclusive`; adding starts or changing the endpoint after seeing this result
is prohibited. No further evaluation or training is authorized by this
record.

The next owner decision must choose a different claim before any new budget is
requested:

1. A playing-strength relation requires genuinely held-out starts, paired
   score as the primary endpoint, start-level clustering, and a frozen
   fixed-width, minimum-effect or equivalence target. Survival remains a
   process endpoint, not a substitute for score.
2. A target-refresh causal claim requires same-source, same-seed,
   equal-transition refresh/no-refresh training pairs across multiple seeds at
   the late maturity of interest. Neither retained route can supply causality.
3. The exploratory revisit-share direction may motivate a separately frozen
   mechanism diagnostic on a new corpus. It cannot be promoted from this
   result without a new plan, readiness record and explicit authority.
