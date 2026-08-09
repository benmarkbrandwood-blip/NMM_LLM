# Sanmill Malom policy-auxiliary calibration result — 9 August 2026

## Decision

The frozen four-arm optimizer-integration calibration completed without an
infrastructure or policy-health failure.  Its preregistered verdict is
`inconclusive_recalibration_required`: no non-zero coefficient is selected.

The immutable local result is
`out/malom-policy-auxiliary-calibration-smoke-v1/result.json`:

- result identity:
  `d11384f661192db84662a6e43e85cdd6eb299672724178a83a02858b0b12113f`;
- file SHA-256:
  `bdfbfbb4b68dca011277d76604e385a2a5918dd818dca963c50600528fb64761`;
- file size: 459,253 bytes;
- plan identity:
  `bdee5fc858b065203d61edbd199e4e77be32262c3fb75a72172e4f7489542aba`;
- readiness identity:
  `62a17abe2ecbb05aac07e7b1fa7f42d7be70cf9cca707d8a598bd98564b15bc0`;
- training source commit:
  `2b04d5af09468c191409cfa70b3699411dd9ed85`; and
- clean result-analysis source commit:
  `1e06f556daee000a6aa2a42b84f0ade802e49dfc`.

The result publisher revalidated every arm's plan, authorization, preflight,
run manifest, lifecycle, complete 100-game log, optimizer log, checkpoint,
SpecialistDB and policy-health report before publishing the result with
exclusive-create semantics.  The four arms used 584.13 seconds of active
training time in total, below the frozen two-hour cap.  No continuation,
extension, promotion or publication was run.

## Observed facts

All arms used fresh seed 51, A2C, base learning rate `1e-4`, update batches of
64 steps, `max_ply=120`, the same deterministic 100-game schedule and isolated
empty `sector-corrected-v1` SpecialistDB copies.  Each arm made 15 updates at
the base rate, then 11 at the inherited `0.5` adaptive floor (`5e-5`) after
the first 50-game boundary because the recent heuristic-opponent win rate was
zero.  This schedule was identical across arms; the sole intended arm
difference remained the preserving-set auxiliary coefficient.

| Arm | Coefficient | Fixed-state mass gain over control change | Scaled auxiliary / absolute policy loss | Repetition draw rate | Gate result |
| --- | ---: | ---: | ---: | ---: | --- |
| control-c000 | 0.00 | — | 0.000 | 16% | reference |
| low-c003 | 0.03 | 0.0000996 | 0.183 | 16% | below mass threshold |
| medium-c010 | 0.10 | 0.0002352 | 0.517 | 15% | below mass threshold |
| high-c030 | 0.30 | 0.0002903 | 1.625 | below mass threshold and loss-dominant |

The frozen detectable-mass threshold was 0.001 and the maximum loss-scale
ratio was 1.0.  All three active arms had complete update-label coverage,
finite updates, informative support and passing policy-health and entropy
checks.  None reached the mass threshold.  The highest coefficient also
crossed the loss-scale limit, so increasing the raw coefficient is not an
acceptable continuation of this design.

The auxiliary signal moved the fixed development diagnostic monotonically in
the intended direction:

| Coefficient | Final preserving mass | Training change | Preserving-minus-downgrading logit margin |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.3676082 | +0.0000251 | +0.0001485 |
| 0.03 | 0.3677078 | +0.0001247 | +0.0007708 |
| 0.10 | 0.3678435 | +0.0002603 | +0.0016252 |
| 0.30 | 0.3678985 | +0.0003154 | +0.0019711 |

This continuous diagnostic did not translate into a detectable discrete
action-quality or match result in 100 games.  The control, low and high arms
each finished 5 wins, 20 draws and 75 losses; medium finished 5 wins, 19 draws
and 76 losses.  Every arm was 0 wins, 0 draws and 48 losses against Sanmill.
For those Sanmill games, all arms recorded the same 70 placement, one movement
and zero flying WDL-downgrading actions.  Whole-run downgrade counts were also
identical at 172 placement, 85 movement and zero flying actions; only the
number of supported movement and flying actions varied slightly with game
length.

The active arms labelled every optimizer step.  Informative support was
455/1,967 for coefficient 0.03, 460/1,948 for 0.10 and 448/1,925 for 0.30.
Thus the small effect is not explained by missing labels.  All four policies
still selected a preserving action on all 29 inspected critical states.  The
additional entropy drop relative to control was at most
`2.253800630569458e-7`, so no collapse is visible.

The run has raw and complete-window 50-game training curves plus 26 finite
optimizer updates per arm.  Ordinary RL has no supervised validation curve.
The fixed 64-position policy-health corpus is inspected development evidence,
not held-out validation or strength evidence.

Runtime identity remained MIF tag `mif-suite-1.0` at release commit
`a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, rules semantic digest
`sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`,
and Sanmill commit
`a6623f88959f7453594df274fbe1f128af7ff55e` with binary SHA-256
`5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`.

## Hypothesis

The preserving-set loss has the correct direction but its raw loss magnitude
is a poor proxy for its optimizer influence.  It is averaged only over
informative states, while the A2C policy, entropy and value gradients are
combined over different supports and then jointly clipped before one Adam
step.  Consequently, choosing a coefficient from loss values alone can reach
the frozen loss-dominance limit before producing a useful preserving-mass
update.

## Supporting evidence

- The fixed-state mass and logit margin increase monotonically with the
  coefficient, establishing a real directional effect.
- Active-arm label coverage is complete and 23.1–23.6% of batch steps are
  informative, excluding sparse or missing supervision as the immediate
  explanation.
- The previous no-update probe measured a finite but small unscaled auxiliary
  gradient L2 norm of approximately 0.0015 and found that a direct float32
  step at the production learning rate was below observable resolution.
- Coefficient 0.30 makes the scaled loss 1.625 times the median absolute A2C
  policy loss, yet gains only 0.0002903 over control.  More raw coefficient is
  therefore an inefficient and already disallowed remedy.
- Identical Sanmill-facing W/D/L and downgrade counts show that the continuous
  movement was too small to alter the observed behavior in this calibration.

## Counterevidence and limits

- Only one fresh seed was used.  The monotonic fixed-state response is useful
  engineering evidence, not a learning-effect distribution.
- The run is intentionally short.  A delayed effect is possible, but the
  frozen contract forbids extending these arms after seeing their result.
- Loss-scale comparison does not measure gradient norm, gradient alignment,
  clipping contribution or Adam's effective parameter update.  The proposed
  mechanism is therefore still a falsifiable hypothesis.
- The arms share a seed and deterministic schedule, so their nearly identical
  discrete trajectories improve paired sensitivity but are not independent
  match samples.
- The fixed diagnostic corpus is development data.  It cannot establish
  generalization, playing strength or promotion readiness.

## Next validation experiment

Do not lower the 0.001 threshold, extend these arms or select coefficient 0.30
post hoc.  Before any new training, add a disposable, no-update gradient
interaction audit on production-shaped A2C batches.  It must report separately
for the policy, entropy, value and preserving-set objectives:

1. unscaled and coefficient-scaled gradient L2 norms;
2. pairwise cosine alignment, especially auxiliary versus policy;
3. the joint pre-clip norm, clipping scale and each objective's contribution;
4. Adam-equivalent first-step parameter and fixed-state probability deltas;
5. informative support by placement, movement and flying phase; and
6. multiple fresh seeds with the same data and runtime identities.

If that audit confirms loss/gradient-scale mismatch, freeze a new normalization
rule against a target gradient contribution before preparing another bounded
calibration.  Any new rule must be deterministic, persisted in checkpoints and
fail closed on non-finite or zero reference gradients.  No new training or
long-run inclusion is authorized by this result.
