# Sanmill Malom-Downgrade Penalty Ablation Result — 9 August 2026

## Decision

The frozen six-arm experiment is complete and its preregistered verdict is
`inconclusive`.  The result is finite, identity-valid and safe, but it does not
meet the material-effect gate and must not be extended or reinterpreted after
inspection.

The immutable ignored result is
`out/malom-downgrade-penalty-ablation-smoke-v1/result.json`:

- result identity:
  `1118b9ace643f2cdaa14c88bf48676d2460c589d184299cf23d564e0311a915d`;
- file SHA-256:
  `c26d4ebe890fdc06e88be639b63979b10b964fd1322d03e8ee80c14e2ba49020`;
- plan identity:
  `3e6c4d356934c4d35ba298ce9b6430446a96f97354b52daf052c198142b5df21`;
- readiness identity:
  `9655aa5be1246aeade04487b0a8bba6c8de6df48fa98d0366990d3c3baca3eeb`;
- readiness SHA-256:
  `d1bf4e204b5ec30781ca1ae9b49900aa278e7423bb5dd440edabd7661bd50818`;
  and
- clean published training and analysis source:
  `fb9b7036e4a08e92331491ed67d3bfdcdfc7bf2f`.

The publisher revalidated every arm's plan, authorization, preflight, run
manifest, lifecycle, complete 500-game log, update log, checkpoint,
SpecialistDB and policy-health report before creating the result with
exclusive-create semantics.  The focused publisher and contract tests report
`12 passed`.

## Observed facts

The primary endpoint is the exact-WDL downgrade rate over learner actions in
games 301 through 500.  Positive reduction means the treatment is better.

| Seed | Control | Penalty treatment | Control minus treatment |
| ---: | ---: | ---: | ---: |
| 45 | 245 / 4,148 = 5.9065% | 242 / 4,027 = 6.0094% | -0.1030 pp |
| 46 | 265 / 3,159 = 8.3887% | 262 / 3,370 = 7.7745% | +0.6142 pp |
| 47 | 275 / 3,152 = 8.7246% | 264 / 3,474 = 7.5993% | +1.1253 pp |

All six arms exceed the frozen 2,000-action minimum, all three treatment arms
pass policy-health safety, two of three pairs favour treatment and no seed is
harmed by two percentage points.  The median reduction is only
`0.006142498987866604` (0.6142 percentage points), below the frozen
`0.02` requirement.  The gate therefore returns `inconclusive`.

Whole-run rates move in the favourable direction for all three treatment
arms, but whole-run sensitivity is not the preregistered endpoint:

| Seed | Control whole run | Treatment whole run | Reduction |
| ---: | ---: | ---: | ---: |
| 45 | 808 / 11,027 = 7.3275% | 805 / 11,211 = 7.1804% | +0.1470 pp |
| 46 | 654 / 8,313 = 7.8672% | 651 / 8,657 = 7.5199% | +0.3473 pp |
| 47 | 965 / 9,238 = 10.4460% | 626 / 9,547 = 6.5570% | +3.8889 pp |

The tail signal is highly phase-concentrated.  Placement has exactly 1,800
supported learner actions in every arm and downgrade rates between 13.0% and
14.44%.  Movement rates range from 0.234% to 1.075%; flying is zero in four
arms and below 2% in the other two.  Treatment reduces the placement rate in
all three pairs, but only by 0.17, 0.33 and 0.72 percentage points.

The opponent split does not support a strong Sanmill-facing effect.  Tail
downgrade rates against Sanmill remain about 9.94% to 10.55%; treatment
reductions are only 0.15, 0.15 and 0.48 percentage points.  The larger
seed-46 and seed-47 reductions occur mainly against the frozen target.  Both
learner colours are represented and seed 45 is slightly adverse for Black,
so no colour-specific general benefit is established.

All 771 optimizer updates are finite across the six arms.  There is no
ordinary supervised validation curve.  Final 50-game entropy remains between
2.108 and 2.249 and chosen probability between 0.179 and 0.200; no collapse is
visible.  Every treatment policy selects a value-preserving action on all 29
inspected development states, with preserving-minus-downgrading logit margins
of +0.122, +0.060 and +0.186.  These are safety diagnostics, not held-out
strength results.

Training W/D/L remains secondary.  No arm wins a game against the 1,000-node
Sanmill opponent.  Treatment increases frozen-opponent score in all three
pairs primarily by producing more draws.  Threefold terminations increase
from 68 to 80, 10 to 25 and 23 to 44 within the matched seed pairs.  This is
consistent with safer play, but it is not evidence of improved winning
strength.

Runtime identity remains MIF tag `mif-suite-1.0` at release commit
`a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, rules semantic digest
`sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`,
and Sanmill commit
`a6623f88959f7453594df274fbe1f128af7ff55e` with binary SHA-256
`5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`.

## Hypothesis

The scalar downgrade penalty reaches the policy only through the one-step A2C
advantage.  Its effect is therefore diluted by outcome reward, bootstrap
variance and credit assignment.  A direct auxiliary objective that transfers
probability mass from exact-WDL-downgrading actions to the complete set of
exact-WDL-preserving actions may provide a denser and less seed-sensitive
training signal without prescribing one arbitrary move inside the preserving
set.

## Supporting evidence

- The treatment changes hundreds of immediate rewards and produces positive
  policy-health margins, yet the preregistered tail effect remains small.
- Downgrades are concentrated in placement, where delayed game outcomes are a
  particularly indirect teacher and the existing heuristic/Sentinel reward
  routes are disabled by the frozen experiment.
- The treatment effect against Sanmill is much smaller than its effect against
  the frozen target, so increasing the same scalar is unlikely to close the
  opponent-facing gap.
- Existing corrected Malom queries can label every learner action, and the
  trainer already fails closed when exact support is absent.

## Counterevidence and limits

- Three seed pairs are a bounded engineering screen, not a population-level
  estimate.
- Seed 47 has a large favourable whole-run sensitivity, so a delayed scalar
  effect cannot be ruled out; the frozen tail gate nevertheless forbids
  extending this run to find out after seeing the result.
- A preserving-set auxiliary may increase passive drawing because the initial
  standard position is game-theoretically drawn.  It must be measured against
  Sanmill, repetition terminations, entropy and practical match score rather
  than assumed beneficial.
- The 29-state policy-health corpus is inspected development data.  It may
  verify wiring and anti-collapse only; it cannot become the successor's
  independent strength evaluation.
- Exact WDL does not order equally preserving moves by practical winning
  chance or full draw-liveness history.  The auxiliary must leave those moves
  available to the ordinary policy objective.

## Next validation experiment

Reward-only escalation ends here.  The next smallest falsifiable experiment
is an online preserving-set policy auxiliary:

1. retain `malom-preserving-only` reward in both arms;
2. change only whether a bounded auxiliary loss maximizes total policy mass on
   all exact-WDL-preserving legal actions;
3. keep the target uniform over the preserving set rather than selecting one
   Oracle move;
4. first prove no-update label coverage, finite gradients, exact zero loss
   when every move preserves WDL, and a positive preserving-mass step on fixed
   development states;
5. then use fresh paired seeds, the same Sanmill referee/opponent schedule and
   the same per-seed, phase, opponent, colour and termination reporting; and
6. reject the route if it merely increases repetition draws, loses entropy,
   lacks a consistent Sanmill-facing downgrade reduction, or crosses the
   existing policy-health boundary.

No training continuation, promotion, strength claim or threshold change is
authorized by this evidence.
