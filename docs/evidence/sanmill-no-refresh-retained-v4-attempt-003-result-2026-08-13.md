# Sanmill no-refresh retained-v4 attempt-003 result — 13 August 2026

Status: `completed_research_baseline_not_promoted`

## Observed facts

### Authority, execution and lineage

| Item | Recorded value |
| --- | --- |
| Plan ID | `managed-sanmill-no-refresh-retained-v4-seed70-attempt-003` |
| Plan semantic identity | `1702726f3686928fac59b9608ed85f8cf1a6d799b4cff83513dd1f553adfb2d2` |
| Readiness identity | `77cc65ad9814daacd13221b26ffe3a022fd9d60948dd00a6502a4f0aea42dd31` |
| Training source | `662fe160e3ef1c3dc3f16cdcd40dd9a3e12b145b` |
| Seed | `70` |
| Target schedule | `target-refresh-every=5001`; no refresh inside 5,000 games |
| Controller result | `completed`; 5,000/5,000 games; 20/20 accepted segments |
| Active time | 1.9478 hours under the 12-hour ceiling |
| Final controller event | sequence 42, `managed_plan_completed`, 13 August 2026 06:49:14Z |

The authorization was consumed once. There was no automatic retry, failed
segment recovery, resource extension, held-out game, promotion, publication or
release. The controller ledger contains 43 hash-linked records, sequence 0
through 42. The 20 accepted `train_log.jsonl` files contain exactly 5,000
unique consecutive game numbers. The 1,811 update rows are finite, and all 20
policy-health gates passed.

`target_age` is exactly 1 through 5,000. It appears as 1 only in game 1 and is
5,000 in the final game. This proves that the requested no-refresh treatment
executed; it does not prove that no-refresh improved transferable play.

### Opponent arms and termination reasons

The two training opponent arms have incompatible meanings and must be read
separately. Win rate counts wins only. Score rate is ordinary match scoring,
`(wins + 0.5 * draws) / games`; it is not the scalar training reward. A
max-ply truncation is incomplete even though the legacy training outcome field
places it in the draw bucket.

| Opponent arm | Games | W / D / L | Win rate | Logged score | Rules draws | 120-ply truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen initial target | 2,980 | 2,721 / 259 / 0 | 91.31% | 95.65% | 145 | 114 |
| Sanmill search | 2,020 | 23 / 1,185 / 812 | 1.14% | 30.47% | 501 | 684 |
| Mixed total, diagnostic only | 5,000 | 2,744 / 1,444 / 812 | 54.88% | 69.32% | 646 | 798 |

The 69.32% aggregate is listed last deliberately. It is dominated by the
planned 60/40 opponent mixture and an increasingly easy frozen-target arm; it
is not a strength statistic.

The latest same-source windows are:

| Opponent arm | Source-window games | W / D / L | Win rate | Logged score | Rules draws | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen initial target | 200 | 189 / 11 / 0 | 94.50% | 97.25% | 11 | 0 |
| Sanmill search | 200 | 9 / 191 / 0 | 4.50% | 52.25% | 68 | 123 |

These are source-local windows, not the last 200 chronological games. The
trainer manifest records `rolling_win=40`; its misleadingly named legacy
`win_rate_200` field therefore means the latest 40 mixed-source games. The
final value is 27 wins in 40 games, or 67.5%. The final 200 chronological games
are a third window: 116 wins, 84 draws and no losses. None of the three windows
may be substituted for another.

Sanmill results by sequential fixed-node stage are:

| Level | Nodes | Games | W / D / L | Score | Rules draws | Truncations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 198 | 0 / 1 / 197 | 0.25% | 0 | 1 |
| 2 | 5,000 | 176 | 0 / 0 / 176 | 0.00% | 0 | 0 |
| 3 | 25,000 | 210 | 1 / 3 / 206 | 1.19% | 2 | 1 |
| 4 | 100,000 | 385 | 7 / 152 / 226 | 21.56% | 75 | 77 |
| 5 | 500,000 | 1,051 | 15 / 1,029 / 7 | 50.38% | 424 | 605 |

Levels are fully confounded with training age, temperature, target age and
database growth. A useful within-level observation remains at 100,000 nodes:
segments 7 through 10 recorded `1/4/85`, `0/11/83`, `1/50/42` and `5/87/16`.
The fixed opponent budget rules out node count as the cause of that local
transition, but model updates, temperature and SpecialistDB state still
changed. It is behavior-change evidence, not a weight-only capability test.

### Descriptive retained-v3 comparison

Retained v3 used seed 58 and source `3f400135`; attempt-003 used seed 70 and
source `662fe160`. The following is descriptive, not a causal refresh
comparison:

| 500,000-node Sanmill stage | Games | W / D / L | Score | Rules draws | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3, refresh every 50 | 1,004 | 24 / 964 / 16 | 50.40% | 523 | 441 (43.9%) |
| v4, no refresh | 1,051 | 15 / 1,029 / 7 | 50.38% | 424 | 605 (57.6%) |

The logged scores are essentially equal while v4 has fewer wins, fewer losses
and substantially more truncations. In each run's latest 200 same-source
Sanmill games, v3 is `4/193/3` (50.25% score) and v4 is `9/191/0` (52.25%).
Those small counts do not overcome the seed, source, trajectory, temperature
and database confounders.

### Learning diagnostics

The fixed 64-state policy-health corpus passed at every accepted boundary. At
game 5,000, direct and candidate critical-state preservation are both 1.0,
the preserving-minus-downgrading mean logit margin is `+5.129311`, and
temperature-1 entropy is `2.083567`. Temperature-1 entropy also fell from
about 2.976 at the first boundary, so policy weights sharpened in addition to
the separate exploration-temperature anneal.

Per-game Malom move diagnostics are not monotonic. There are 1,914 games with
a positive downgrade rate, the last at game 3,180. Preserving rate is 0.8182
at game 1, 0.5833 at game 500 and 0.6667 at game 1,000. Only the fixed-corpus
boundary gate reaches and retains perfect critical argmax preservation.

The learning rate is `1e-4` at game 50 and `5e-5` from game 51. The trainer's
reward constants favor a win (`+1.5`) over both a rules draw (`-0.15`) and a
max-ply truncation (`-0.25`). The logs contain no gradient-norm series, so they
cannot support a claim that gradient capacity was exhausted or that the
reward objective itself prefers a draw.

### Final artifacts and retrospective-audit limit

`checkpoint_tool.py verify` accepted the final checkpoint:

- path:
  `learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/managed-sanmill-no-refresh-retained-v4-seed70-attempt-003/segments/segment-0020/latest.pt`;
- checkpoint ID:
  `managed-sanmill-no-refresh-retained-v4-seed70-attempt-003-segment-0020:checkpoint:00000006`;
- payload SHA-256:
  `ed7932bc7c11b1aa41274ea0de7bd08902812b1188ca4739b6d0d8dc15e46727`;
- file SHA-256:
  `295b268e697255908f9c7517f4697ca251a10ec0f13d922cbcbab2260fb5105d`;
- game count 5,000, update count 1,811, target age 5,000 and temperature 0.20.

The final SpecialistDB main file is 32,600,064 bytes with SHA-256
`3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed`.
Immutable read-only inspection reports `quick_check=ok`, metadata
`sector-corrected-v1`, 242,006 positions, 4,185 winning lines and zero
preferred plays. A later post-completion reader created an empty WAL and a
32,768-byte SHM sidecar at 15:44 local time; the main-file identity remained
unchanged. Preserve those sidecars rather than deleting them to manufacture a
sidecar-free claim.

The training logs do not persist the final board, complete move history,
repetition count or no-capture count for a truncated game. The 798 historical
cap states therefore cannot be queried retrospectively without a behavioral
replay. A Malom query would in any case describe the theoretical value of a
board placement and side to move; it cannot reproduce strict history-dependent
threefold or no-progress adjudication.

## Hypotheses

1. The fixed-100,000-node transition and late low loss rate are consistent
   with learning loss avoidance. They do not yet establish win conversion.
2. Keeping the initial target stale plausibly made the frozen arm progressively
   uninformative and reduced pressure to improve against a changing peer. This
   could contribute to the higher late truncation share, but the v3/v4
   comparison cannot identify that cause.
3. The 57.6% Sanmill truncation rate at level 5 may reflect a more passive
   policy, unresolved winning/losing positions, or both. It is an
   incompleteness signal, not proof of theoretical draws.

## Supporting evidence

- At a fixed 100,000-node budget, losses fall from 85 to 16 across four
  consecutive segments while rules draws and truncations increase.
- The final same-source Sanmill window contains no loss, but 123 of 200 games
  are truncated and only nine are wins.
- The policy-health gate rules out catastrophic Malom-value direction collapse
  on its 29 critical development states, and temperature-1 entropy shows
  genuine weight sharpening.
- The stale target's age reaches 5,000 and its arm reaches 94.5% recent win
  rate, confirming that the treatment created a very easy internal opponent.

## Counterevidence and confounders

- One seed per retained condition and different source commits prevent causal
  attribution to refresh cadence.
- The frozen-arm result is a designed stale-opponent artifact, not evidence
  that no-refresh avoided instability or increased strength.
- The 29-state corpus was repeatedly inspected during development and is not
  held out. Per-game preserving rates contradict any claim that preservation
  rose monotonically from the first game.
- Temperature, training age, database state and node stage change together.
  Entropy cannot be interpreted without both scheduled and temperature-1
  views.
- Max-ply truncations are incomplete. Assigning them half a point is a logged
  diagnostic convention, not a strict-referee outcome.
- Malom has no repetition or no-capture history and cannot relabel a capped
  game as its actual result. Existing logs also lack the state needed for such
  a retrospective board query.
- The reward mapping explicitly prefers wins, and no gradient-norm evidence
  supports an exhausted-learning claim.

## Next validation experiments

The immediate safe work was the tracked read-only dashboard correction. It now
reads `rolling_win` from every segment manifest, labels the mixed 40-game
number as a training diagnostic, shows frozen and Sanmill recent windows
separately, and exposes rules draws and max-ply truncations by source and node
level with bilingual help.

The next scientific decision must separate two objectives:

1. **Passivity/mechanism diagnosis.** A no-update evaluator may compare the
   frozen v3 and v4 final checkpoints on identical, colour-swapped starts. It
   should use the strict complete-history referee, continue to a safety cap
   above the rules window, keep a cap as invalid, and record paired game
   length, rules termination, cap incidence and action-level Malom-preserving
   trajectories. It may snapshot Malom W/D/L at logical ply 120 only as a
   secondary theoretical-state diagnostic; it must never overwrite the game
   result.
2. **Playing-strength relation.** Paired W/D/L score remains the primary
   outcome if the claim concerns strength. The observed decisive-game shares
   at 500,000 nodes are only 2.09% for v4 and 3.98% for v3, so 128 games per
   checkpoint would expect roughly 3 and 5 decisive games. A 16-game pilot is
   also too small to estimate that rare-event rate reliably. Before spending
   held-out authority, freeze the target minimum detectable paired-score
   effect, power or precision requirement, maximum sample size, interim rule,
   disjoint development and confirmatory corpora, and treatment of invalid
   caps. Process metrics can power the mechanism question but cannot replace
   W/D/L for a strength claim.

The historical training logs cannot supply the proposed cap-state snapshots;
the evaluator must capture them prospectively. No held-out or replay games are
authorized by the completed long-run grant.

| Launch gate | State |
| --- | --- |
| v3/v4 checkpoint identities | passed |
| Attempt-003 completion and metric recomputation | passed |
| Retrospective cap-state evidence | unavailable; state/history not logged |
| Evaluation objective and primary estimand | not frozen |
| Power/precision rule and bounded game count | not frozen |
| New corpus and exposure audit | not frozen |
| Exact prospective evaluator and focused tests | not implemented |
| Separate held-out/development launch authority | absent |

Verdict: `needs_decision`.

This verdict permits read-only analysis, dashboard work, evaluator design and
preflight preparation. It does not permit evaluation games, training, resume,
promotion, publication or release.
