# SpecialistDB read calibration transition diagnosis

Status: development diagnosis only. This record is not held-out evidence,
playing-strength evidence, or launch authority.

## Evidence identity

- Training source: `a7529045ecb1323f0bc2a548642fea7238ec7a35`.
- Frozen read-calibration plan identity:
  `36a1feb6bc9e403890f7c3b6b6f3444a97a9cd721272b760a2b25d0f8091459b`.
- Readiness identity:
  `ee68e2d90d069bd65643d0e02ecb4c408fb522a35d6e19bd6246b1cb1b640b6f`.
- Raw result identity:
  `90da60538e782c85b5871e35eec4895e44fe76003309b3ad13c417c8868f86de`.
- Raw result SHA-256:
  `e8a8f3aac6076697b9a31c7532880976f4222801d002f988d5f65bf78c8344e9`.
- Raw result location:
  `out/specialist-db-training-read-calibration-v1/result.json` (ignored,
  machine-local evidence).

The authorized calibration completed all six 250-game arms, 1,500 games,
405 optimizer updates, and all six policy-health gates. Its read-mode verdict
was eligible only for later design and selected no read mode.

## Observed facts

Every arm reached the same coupled transition after game 50:

- game 50 recorded `target_age=50` and learning rate `0.0001`;
- before game 51 the frozen opponent was refreshed, so game 51 recorded
  `target_age=1`; and
- the periodic learning-rate rule simultaneously changed the rate to
  `0.00005`, because the rolling full-resource Sanmill win rate was zero.

The learner's score against the frozen-model opponent changed as follows. A
score is `(wins + 0.5 * draws) / games`; it is an observed training statistic,
not a strength estimate.

| Arm | Games 1-50 W/D/L | Score | Games 51-100 W/D/L | Score |
| --- | ---: | ---: | ---: | ---: |
| seed61 full | 8/6/21 | 0.314 | 0/0/34 | 0.000 |
| seed61 theoretical-only | 8/6/21 | 0.314 | 0/0/34 | 0.000 |
| seed62 full | 7/21/1 | 0.603 | 0/17/15 | 0.266 |
| seed62 theoretical-only | 6/21/2 | 0.569 | 0/15/17 | 0.234 |
| seed63 full | 28/3/1 | 0.922 | 0/12/23 | 0.171 |
| seed63 theoretical-only | 27/4/1 | 0.906 | 0/2/33 | 0.029 |

The decline persisted in later windows. Sanmill itself was unbeaten by the
learner before the first transition, so the same observed data caused the
historical adaptive rule to use its minimum multiplier of 0.5.

The current opponent paths also differ in action selection. The learner
samples from its temperature-scaled policy distribution. The frozen-model
opponent selects the maximum-logit legal action from a copied policy snapshot.
Refreshing therefore replaces the initial frozen snapshot with the learner's
current weights while also changing sampled learner versus deterministic
opponent behaviour. This is a code fact, not proof that refresh is harmful.

## Hypotheses

1. Refreshing the frozen opponent at game 50 makes that opponent materially
   harder because it uses the learner's newer weights deterministically.
2. Halving the learning rate at the same boundary slows adaptation to that
   changed opponent.
3. The two changes interact; neither factor alone may explain the decline.

## Supporting evidence

- The boundary and the decline recur across three seeds and both SpecialistDB
  read modes.
- Target age and learning rate both change at exactly the first affected
  boundary.
- The control flow refreshes the target before scheduling game 51 and applies
  learning-rate adaptation after logging game 50.
- The read-mode comparison does not explain the shared seed61 collapse or the
  shared timing across all six arms.

## Counterevidence and limits

- The learner did not beat the 1,000-node Sanmill opponent before refresh, so
  weak policy quality can explain part of the poor outcomes independently of
  either transition.
- Training W/D/L is endogenous: both learner and frozen opponent change with
  training. It is not held-out strength evidence.
- A deterministic frozen opponent can expose policy weaknesses without being
  a training defect.
- The existing runs changed both factors together. They cannot identify a
  refresh main effect, a learning-rate main effect, or their interaction.
- SpecialistDB contents evolve within each arm. A new causal diagnostic must
  hold its read projection fixed and use isolated byte-identical databases.

## Required next validation

Run a short, preregistered 2x2 paired diagnostic with two fresh seeds:

1. refresh at game 50 plus historical adaptive learning rate;
2. refresh at game 50 plus fixed learning rate;
3. no refresh within the experiment plus historical adaptive learning rate;
4. no refresh within the experiment plus fixed learning rate.

All four arms within a seed must be identical through game 50. Only games
51-100 may contribute to the factorial contrasts. This diagnostic must finish
before any held-out comparison or long training is considered.
