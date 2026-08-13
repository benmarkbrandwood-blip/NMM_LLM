# Sanmill retained-v3 versus no-refresh-v4 evaluation decision brief

Status: `needs_decision_no_games_authorized`

This is a source-only planning record. It authorizes no development game,
held-out game, model update, database write, checkpoint write, promotion or
publication.

## Observed facts

The proposed 64-start, colour-swapped design would produce 128 games per
checkpoint and 256 games total. In the sequential 500,000-node training stage,
the result counts were:

| Candidate | Games | Wins + losses | Decisive rate | Wilson 95% interval |
| --- | ---: | ---: | ---: | ---: |
| retained-v3, refresh every 50 | 1,004 | 40 | 3.984% | 2.939%–5.380% |
| retained-v4, no refresh | 1,051 | 22 | 2.093% | 1.386%–3.149% |

These rates are confounded training observations, not evaluation-rate
estimates. Treating them only as plug-in planning values gives:

| Games per checkpoint | Expected decisive v3 | Expected decisive v4 | Probability of zero v3 | Probability of zero v4 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.64 | 0.33 | 52.2% | 71.3% |
| 128 | 5.10 | 2.68 | 0.55% | 6.67% |
| 256 | 10.20 | 5.36 | 0.003% | 0.44% |
| 512 | 20.40 | 10.72 | <0.001% | 0.002% |

At the v4 plug-in rate, 1,234 games are needed to have a 90% chance of seeing
at least 20 decisive results. The analogous v3 count is 647. These are event
yield calculations, not paired-effect power and not recommended budgets.
They show why 16 games cannot estimate the rare-event rate and why 128 games
per checkpoint is unlikely to distinguish effects near two percentage points.

The calculations are reproducible with:

```powershell
.\.venv\Scripts\python.exe tools\estimate_decisive_event_feasibility.py `
  --label v4-no-refresh-l5 --observed-events 22 --observed-games 1051

.\.venv\Scripts\python.exe tools\estimate_decisive_event_feasibility.py `
  --label v3-refresh50-l5 --observed-events 40 --observed-games 1004
```

The existing training logs cannot answer what the 120-ply cap states were.
They store neither final board/history nor repetition and no-capture state.
Malom also omits those strict-referee history counters. Cap-state W/D/L must
therefore be captured prospectively and may be used only as a theoretical
position diagnostic.

## Hypotheses

1. If the objective is **passivity/mechanism diagnosis**, paired game length,
   cap incidence, rules termination and Malom-preserving trajectories may be
   more sensitive than W/D/L to a v3/v4 behavior difference.
2. If the objective is **playing-strength relation**, W/D/L pair score must
   remain primary. Process metrics can explain a result but cannot replace it.
3. Deterministic argmax evaluation could have a different decisive rate from
   temperature-sampled training, but a 16-game pilot is too small to estimate
   whether that difference is material.

## Supporting evidence

- The two candidates have almost equal logged 500,000-node stage scores,
  50.40% and 50.38%, while their truncation shares differ materially,
  43.9% and 57.6%.
- The 128-game plug-in yields are only about five and three decisive games.
- Under the existing strict protocol, a 1,536-post-prefix-ply cap is invalid,
  not a draw; this preserves honest result semantics.
- Colour-swapped paired starts and identical Sanmill work remain the correct
  controls for both objectives.

## Counterevidence and confounders

- v3 and v4 use different seeds and source commits. Their direct comparison
  cannot identify the causal effect of refresh cadence.
- Training decisive rates may not transfer to deterministic evaluation.
- A fixed-corpus engineering interval is not population inference. A formal
  power claim needs a defined sampling universe or a preregistered finite-
  corpus precision rule.
- We do not yet know the variance of paired process differences or the rate of
  discordant paired scores. Decisive counts alone cannot choose a final sample
  size.
- Reusing the already inspected 64-start corpus could support development
  diagnosis but would not create new held-out evidence.

## Next validation experiments

Freeze one of two different contracts; do not combine their claims after
seeing results.

### Option A: passivity/mechanism development study

- Same frozen v3/v4 `latest.pt` identities.
- Same colour-swapped start histories, candidate inference route, 500,000-node
  Sanmill configuration and strict complete-history referee.
- Prospective snapshots at logical ply 120, including full rule history and a
  separately labelled Malom theoretical W/D/L.
- Continue each game to the existing 1,536-post-prefix-ply safety cap; cap is
  invalid, never scored as a draw.
- Primary estimands may be paired cap incidence, logical-ply distribution and
  preserving-to-downgrading trajectory events. W/D/L is secondary and no
  strength claim is allowed.
- A development corpus may be reused only if labelled non-held-out. Its size
  still needs a variance/precision rule and resource bound.

### Option B: playing-strength evaluation

- Pair score is primary; process metrics and ply-120 Malom snapshots are
  explanatory secondary endpoints.
- Freeze a practically relevant minimum paired-score effect and an error or
  fixed-width precision rule before choosing sample size.
- Use a new exposure-audited confirmatory corpus. If a development pilot is
  needed for discordance and decisive-rate estimation, keep it disjoint and
  prohibit its results from changing the confirmatory effect threshold.
- Freeze the maximum number of starts/games, active-time ceiling, invalid-cap
  rule and any interim stopping rule. Repeating deterministic starts adds no
  information.

For the causal refresh question, neither option is sufficient. That requires
same-source, same-seed, equal-transition refresh/no-refresh pairs across
multiple seeds.

| Launch gate | State |
| --- | --- |
| Checkpoint and source identities | known |
| Objective: mechanism or strength | product decision required |
| Primary estimand and material threshold | not frozen |
| Variance/discordance basis | unavailable |
| Corpus membership and exposure audit | not frozen |
| Maximum game/time/node envelope | not frozen |
| Prospective cap-state evaluator | not implemented |
| Focused referee/evaluator tests | not run |
| Separate launch authority | absent |

Verdict: `needs_decision`.
