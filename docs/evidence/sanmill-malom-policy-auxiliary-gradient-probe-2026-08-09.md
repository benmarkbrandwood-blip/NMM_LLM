# Malom preserving-set policy auxiliary probe — 9 August 2026

## Outcome

The exact-WDL preserving-set policy objective is wired correctly enough to
prepare a small optimizer-integration smoke. It is **not** ready for a retained
run or a new six-arm effectiveness experiment yet.

The accepted raw report is
`out/diagnostics/malom-policy-auxiliary-gradient-probe-v2.json`:

- source commit: `b2ccecf2a7518adc99d1e1b8c38887ab3938b8ec`;
- probe identity:
  `5ea60e2955a7a9b878ec4119648ed91ddcffd94687bb3c0976571e96048daa9c`;
- file SHA-256:
  `ad1e6e3ee7596a872d3129e623d377e083c439f6bcbee23705a10bf8ced1b003`;
- file size: 34,974 bytes.

The earlier local v1 report is superseded diagnostic output. Its direct
float32 parameter step was smaller than observable parameter resolution at the
production learning rate. The v2 probe therefore measures the analytic
directional derivative and retains the rounded direct step as counterevidence.

## Observed facts

The probe reconstructed fresh policies for seeds 48, 49 and 50. For every
seed, it used the same inspected 64-position placement/movement/flying
development corpus and the production feature route, with Sentinel, ValueNet
and GapNet absent and Malom used only for training labels.

| Fact | Result |
|---|---:|
| Positions per seed | 64 |
| Complete legal actions per seed | 1,583 |
| Exact-WDL-preserving actions | 1,168 |
| Exact-WDL-downgrading actions | 415 |
| Informative positions containing both classes | 29 |
| Positions where every legal action preserves WDL | 35 |
| Root values | 18 win / 24 draw / 22 loss |
| Informative placement / movement / flying positions | 15 / 8 / 6 |

All action sets matched by complete `{from,to,capture}` identity. There were no
unknown values, missing or duplicate actions, positive minimax deltas, or empty
preserving sets.

The fresh models assigned only 0.36758309–0.36758315 mean probability to the
preserving set on the 29 informative positions. Their mean preserving-set loss
was 1.5265350–1.5265355. This is expected from the near-uniform small-gain fresh
initialization and proves that the objective has a non-trivial signal.

The unscaled auxiliary gradient was finite for all six policy parameter tensors
in every seed. Its L2 norm was 0.001503–0.001568. Moving down that loss gradient
had a positive analytic derivative for preserving probability in every seed,
and the descent direction aligned with the preserving-probability gradient at
cosine 0.9702–0.9865.

The first label pass took 2.70 seconds for 1,583 actions because it included
cold database/cache cost. The following two passes took 0.0420 and 0.0438
seconds. This does not identify the label route as the likely training
throughput bottleneck, but the real rollout smoke must still measure it.

The HumanDB snapshot, SpecialistDB snapshot, Malom `std.secval`, original model
parameters and tracked worktree were byte-identical after the probe. No
checkpoint, optimizer state or database was written.

## Hypothesis

A direct loss on total probability assigned to **all** exact-WDL-preserving
actions should supply a clearer policy gradient than a scalar penalty applied
only after the learner samples a downgrade. It should preserve choice among
tied safe actions instead of imitating one arbitrary Oracle action.

## Supporting evidence

1. Full action coverage proves the label can be attached to the same variable
   legal-action matrix consumed by the policy.
2. All-safe states contribute zero supervised preference, so the objective
   cannot invent a ranking that Malom WDL does not support.
3. The measured descent direction increases preserving probability for all
   three fresh initializations.
4. Placement supplies 15 of the 29 informative positions. That targets the
   phase that dominated the previous downgrade-rate failure.
5. Warm label throughput is high relative to complete Sanmill-refereed rollout
   cost.

## Counterevidence and limits

1. This corpus is inspected development data, not held-out validation or
   strength evidence.
2. The three fresh policies are almost uniform by construction. Their nearly
   identical metrics are not three independent learned outcomes.
3. An isolated SGD direction does not determine the trajectory produced by
   Adam when A2C, entropy and value gradients are mixed.
4. At learning rate `1e-4`, the direct float32 parameter step for coefficients
   0.03, 0.10 and 0.30 rounded to no measurable probability change. The
   analytic derivative proves direction, not useful effect size.
5. Exact WDL does not rank practical winning chances, speed, liveness or human
   plausibility among preserving actions. The auxiliary may still increase
   repetition draws or reduce useful exploration.
6. There are no train/validation curves, learned checkpoints, baseline games,
   opponent-source results or ablation outcomes in this probe.

## Next validation experiment

Prepare, but do not yet launch, one optimizer-integration calibration with four
fresh, schedule-paired arms using seed 51:

| Arm | Reward contract | Auxiliary coefficient |
|---|---|---:|
| control | `malom-preserving-only` | 0.00 |
| low | `malom-preserving-only` | 0.03 |
| medium | `malom-preserving-only` | 0.10 |
| high | `malom-preserving-only` | 0.30 |

Each arm should be limited to 100 games at the 1,000-node Sanmill level, 120
logical ply, the same fresh initialization and immutable opponent schedule, and
one isolated SpecialistDB. The full calibration is capped at 400 games and two
active hours. It is a wiring and scale experiment, not an effectiveness or
promotion experiment.

The result must report raw and 50-game curves, update-level policy/value/entropy
and auxiliary losses, scaled auxiliary-to-absolute-policy-loss ratio, label
support, preserving probability, exact downgrade rate by phase/opponent/colour,
W/D/L, termination reasons, checkpoint integrity and the fixed development
policy-health diagnostic.

Reject an arm for any identity, finite-value, coverage, checkpoint, referee or
database failure. Also reject it if it causes policy collapse, materially
reduces entropy, or mainly converts losses into repetition draws. If several
coefficients remain technically healthy, choose the **lowest** one that shows a
detectable fixed-state preserving-mass improvement without dominating the A2C
policy-loss scale. Only then freeze a new multi-seed control-versus-treatment
effectiveness experiment. Do not extend this calibration after seeing results.
