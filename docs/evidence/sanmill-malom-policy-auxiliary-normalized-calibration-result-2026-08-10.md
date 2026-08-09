# Sanmill Malom policy-auxiliary normalized calibration result — 10 August 2026

## Decision

The authorized three-seed, six-arm optimizer-integration calibration completed
without an infrastructure, identity, numerical, checkpoint, database, referee,
policy-health, or resource failure. Its preregistered verdict is
`inconclusive_stop_and_redesign`. Target ratio `0.25` is not selected for a
retained or long training run.

The immutable local result is
`out/malom-policy-auxiliary-normalized-calibration-v1/result.json`:

- result identity:
  `669124f2803609fe87fabc15c38a798711e78541ed1a39614cf44837a51a58ac`;
- file SHA-256:
  `0d59bc587d66006255020e5ab3b7faab2f8b9c693a1c139686a475d0e93828bb`;
- file size: 1,004,023 bytes;
- readiness identity:
  `a5fb75eda17b4609902294f424300cb45f964440852ecfe4a008f1ea70733637`;
- plan identity:
  `1b6f8d05047c4de9d6603d9ae1f26714cb1a23b3b96749e76136387a5f0b53ab`;
  and
- clean published training source commit:
  `83c5cf658bf6edfc75ece6785a87587685eee7a1`.

The frozen result publisher revalidated all six plans, authorizations,
preflights, manifests, event chains, complete 100-game logs, optimizer logs,
checkpoints, isolated SpecialistDBs, and policy-health reports before creating
the result with exclusive-create semantics. Total active training time was
685.61 seconds, or 0.1905 hours, below the two-hour cap. The 270 Sanmill games
made 4,386 search calls, requested 4,386,000 nodes, and expanded 3,773,778
nodes. No retry, extension, resume, promotion, model publication, or long run
was started.

## Observed facts

All six arms used fresh A2C initialization, the same frozen schedule within
each seed, learning rate `1e-4` with the inherited `5e-5` floor after game 50,
temperature `0.90`, `max_ply=120`, 60% frozen-target opponents, 40% pinned
1,000-node Sanmill opponents, and separate empty `sector-corrected-v1`
SpecialistDB copies. Sentinel, ValueNet, GapNet, imitation, opening forcing,
branching, PPO, and recovery remained disabled.

Every arm completed 100 games. The seed-55 and seed-56 arms made 31 optimizer
updates each; the seed-57 arms made 26 because their complete trajectories
contained fewer learner steps. All updates were finite. All six final
checkpoints verified, all policy-health gates passed, and every SpecialistDB
reported `quick_check=ok` with the correct label version and lineage root.

Every normalized update had complete exact-WDL labels and an informative
subset. The applied auxiliary-to-ordinary policy-head gradient ratio was
`0.25` within floating-point error on all 88 treatment updates. No update hit
the coefficient cap. The treatment arms labelled 100% of 6,704 learner steps;
1,544 were informative. Their median effective coefficients were 0.01144,
0.01182, and 0.00790 for seeds 55, 56, and 57 respectively.

| Seed | Paired fixed-state preserving-mass gain | Whole-run downgrade-rate change | Last-50 downgrade-rate change | Repetition-rate change | Pair gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 55 | +0.0000111 | -0.00135 | -0.00425 | +0.11 | repetition gate failed |
| 56 | +0.0000371 | -0.01589 | -0.04571 | -0.02 | safe but below mass gate |
| 57 | +0.0000298 | -0.00153 | -0.00214 | +0.02 | safe but below mass gate |

All three fixed-state paired gains were positive, but their median was only
`0.0000298`, about 3% of the frozen `0.001` minimum. The effect remained
positive in placement, movement, and flying fixed-state slices, but it was
small in every slice. Fixed-state entropy was unchanged at the displayed
precision and no collapse was observed.

The selected-action diagnostic also moved in the intended direction in all
three seeds. Most of the observable change came from seed 56 placement states.
Seed 55 and seed 57 selected nearly the same actions as their controls, so
their aggregate downgrade-rate changes were small. Flying support was thin and
cannot support a phase-specific conclusion.

Training W/D/L was not a selection metric. It nevertheless shows why the raw
repetition-rate gate must not be interpreted as a strength result:

| Seed | Control W/D/L | Treatment W/D/L | Control frozen score | Treatment frozen score | Sanmill score, both arms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 55 | 24/15/61 | 24/26/50 | 0.516 | 0.607 | 0.000 |
| 56 | 26/9/65 | 26/7/67 | 0.476 | 0.460 | 0.026 |
| 57 | 17/3/80 | 17/5/78 | 0.440 | 0.464 | 0.000 |

For seed 55 the eleven-point repetition increase coincided with eleven fewer
losses against the frozen target, not fewer wins. That may be defensive
improvement or short-run path noise; it is not evidence of pathology or
strength. The frozen safety rule still fails and must not be changed after the
result. A future contract must distinguish loss-to-draw repetition from
unproductive repetition before collecting new data.

The complete-window curves show modestly lower treatment entropy and lower
downgrade rates, but not a stable reward or W/D/L improvement across all three
seeds. There is no supervised validation curve in this ordinary RL experiment.
The fixed 29-state diagnostic remains inspected development evidence, not
held-out validation.

## Hypothesis

The normalization correctly controls instantaneous gradient scale, but the
chosen target produces too little cumulative policy movement for the frozen
100-game detection rule. Prior disposable Adam replay found one-step
preserving-mass deltas between roughly one and four millionths. The current
26–31 update arms produced paired fixed-state gains of roughly one to four
times `10^-5`, which is consistent with that measured step scale rather than
with a broken implementation.

The result therefore does not show that exact-WDL supervision has the wrong
direction. It shows that a 0.25 pre-Adam gradient ratio, 26–31 updates, and an
absolute probability-mass gate of 0.001 do not form a sensitive bounded
selection experiment. Increasing the target or extending these observed arms
post hoc would violate the contract and could allow the auxiliary objective to
dominate A2C.

## Supporting evidence

- All 88 treatment updates reconciled to the target ratio, remained finite,
  and avoided the coefficient cap, excluding a normalization implementation
  failure.
- All three paired fixed-state gains and all three selected-action
  downgrade-rate changes had the intended sign.
- The treatment fixed-state logit margin exceeded its paired control in every
  seed and every phase.
- Treatment entropy stayed close to control, and every policy-health check
  passed, excluding visible collapse at target 0.25.
- The observed cumulative gain is the scale predicted by the earlier
  disposable one-step Adam measurements.

## Counterevidence and limits

- The median fixed-state gain missed the preregistered threshold by a factor of
  about 34, so directional consistency alone is insufficient to adopt the
  mechanism.
- Seed 56 improved placement downgrade rate materially, while seed 55 and 57
  barely changed their discrete action trajectories. The behavioral effect is
  not robust yet.
- None of the six arms beat the 1,000-node Sanmill opponent, and only seed 56
  drew two Sanmill games in both conditions. This is short fresh-run context,
  not a strength test.
- The treatment-control reward and frozen-opponent W/D/L differences disagree
  by seed. They cannot substitute for the failed mechanism gate.
- The fixed corpus is development data and has no independent train/validation
  split. No claim of generalization, playing strength, promotion, or long-run
  readiness follows.

## Next validation experiment

Do not lower the completed experiment's thresholds, extend an arm, select
target 0.25, or start retained training.

First run a no-game, no-persistent-update target-response audit on the three
persisted treatment final-flush batches. From the exact same pre-update model
and Adam state, compare auxiliary-off with normalized target ratios 0.25,
0.50, and 1.00 under the existing coefficient cap. It must:

1. reproduce the real target-0.25 final update within the existing replay
   tolerance;
2. report effective coefficient, policy-gradient ratio, cosine, joint norm,
   clipping, Adam parameter delta, entropy, and informative preserving-mass
   delta for every seed and target;
3. retain phase support and identify any missing flying evidence;
4. leave every checkpoint, optimizer, database, and tracked source byte
   unchanged; and
5. make no normalization, training, promotion, or strength decision itself.

If stronger targets produce a monotonic, bounded response without excessive
parameter or entropy movement, freeze a new independent-seed learning
calibration with a redesigned endpoint and a repetition classification that
separates rescued draws from stalling. If the response saturates or becomes
unstable, abandon gradient-ratio escalation and design a separate
KL-constrained teacher or safe-action sampling mechanism. Either path requires
a new contract and explicit launch authorization.
