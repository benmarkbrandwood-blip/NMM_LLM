# Sanmill Malom policy-auxiliary normalized calibration v1

Status: `completed_inconclusive_stop_and_redesign`

Machine-readable contract:
[`sanmill-malom-policy-auxiliary-normalized-calibration-v1.json`](sanmill-malom-policy-auxiliary-normalized-calibration-v1.json)

Plan identity:
`1b6f8d05047c4de9d6603d9ae1f26714cb1a23b3b96749e76136387a5f0b53ab`

Completed result:
[normalized calibration result evidence](../evidence/sanmill-malom-policy-auxiliary-normalized-calibration-result-2026-08-10.md)

Result identity:
`669124f2803609fe87fabc15c38a798711e78541ed1a39614cf44837a51a58ac`

This experiment is a bounded optimizer-integration calibration. It is not a
held-out evaluation, a playing-strength comparison, a promotion decision, or
authorization for a retained or long training run. The six arms have zero
authorized segments. Preparation may create plans, empty isolated databases,
and preflight records, but it must not create an authorization or launch a
trainer.

## Product question

Does scaling the exact Malom policy auxiliary on every informative update to
target 25% of the ordinary policy-head gradient produce a small, repeatable
increase in Malom-preserving policy mass without destabilising ordinary A2C
learning?

The comparison is deliberately against an auxiliary-off control, not against
an earlier fixed coefficient. The earlier four-arm fixed-coefficient study did
not establish a safe or effective coefficient, while the no-update capture
showed that the coefficient required to achieve the same gradient ratio varies
materially between batches.

## Observed facts

- The completed no-update capture used three fresh seeds, 60 games, and 19
  batches. It labelled 1,473 steps, of which 453 were informative because at
  least one legal action downgraded exact WDL.
- For a target policy-head ratio of 0.25, the implied effective coefficient
  ranged from about 0.0481 to 0.2154, with median about 0.1044. No coefficient
  was selected and no optimizer, backward pass, or training update was run.
- The raw capture and its interpretation are bound in the machine-readable
  contract and in the
  [result evidence](../evidence/sanmill-malom-policy-auxiliary-no-update-batch-capture-result-2026-08-10.md).
- The previous fixed-coefficient calibration is insufficient evidence for an
  effectiveness claim. Its result does not justify choosing a fixed scale or
  starting a retained run.
- The Sanmill referee, exact Malom source, HumanDB usage boundary, empty
  `sector-corrected-v1` SpecialistDB template, rules identity, and MIF release
  are held fixed by the contract.

## Hypothesis

Per-batch normalization should reduce the accidental variation in auxiliary
gradient strength seen in the no-update capture. A target ratio of 0.25 is a
mechanism to test, not a proven optimum. The cap of 0.25 limits the coefficient
when the raw auxiliary gradient is small.

The prediction is a positive paired change in fixed-state Malom-preserving
probability mass in at least two of three seeds, without excessive entropy loss,
repetition growth, non-finite values, or violations of the referee, database,
label, checkpoint, and resource contracts.

## Counterevidence and limits

- The no-update capture measured gradients before optimization. It cannot show
  how gradients interact over 100 games or whether policy changes persist.
- A gradient-norm target does not control direction. The auxiliary gradient can
  be orthogonal to or oppose the ordinary policy gradient; cosine is therefore
  recorded on every informative update.
- One hundred games per arm is intentionally small. A null or noisy result is
  evidence to stop or redesign, not evidence that the mechanism can never work.
- The 29-state development corpus is inspected and fixed, but it is not the
  separately frozen held-out strength evaluation. It can support a mechanism
  decision only.
- Training W/D/L is confounded by the changing learner, frozen-target games,
  colour, short horizon, and a small number of Sanmill games. It is reported by
  class but is not the selection metric.

## Six-arm paired design

The frozen order is:

| Order | Seed | Control | Treatment |
|---:|---:|---|---|
| 1-2 | 55 | `fixed`, coefficient 0 | `policy-head-normalized`, target 0.25 |
| 3-4 | 56 | `fixed`, coefficient 0 | `policy-head-normalized`, target 0.25 |
| 5-6 | 57 | `fixed`, coefficient 0 | `policy-head-normalized`, target 0.25 |

Within each seed, the random initialization, game schedule, opponent source,
learner colour, A2C hyperparameters, data identities, and rules identities are
the same. The only training-semantic difference is
`malom_policy_aux_mode`. Arm-specific IDs, output paths, databases, and launch
order are isolation metadata rather than learning factors.

Each arm starts from fresh random weights and a separate byte-identical empty
SpecialistDB. No checkpoint passes between arms and no exact resume is used
between conditions.

## Frozen hyperparameters and resources

- A2C; learning rate `1e-4`; discount `0.99`; entropy coefficient `0.01`.
- One game at a time; update every 64 steps; no branch rollouts; simulated
  lookahead depth 5.
- Temperature starts at 0.90 and reaches 0.20 at 80% of the 5,000-game schedule.
- 60% frozen-target opponents and 40% Sanmill opponents.
- Sanmill fixed-resource ladder
  `1,000 / 5,000 / 25,000 / 100,000 / 500,000` nodes, but the 100-game arm
  bound keeps this calibration entirely at the observed 1,000-node level.
- Maximum 120 logical plies per game.
- Sentinel, ValueNet, GapNet, imitation warm-start, imitation mixing, S1B,
  opening forcing, PPO, recovery, and branching remain disabled.
- Exactly 100 completed games and at most one third of an active hour per arm;
  six arms total at most 600 games and two active hours.
- Arms run one at a time. Any arm failure stops the whole sequence.

The deterministic schedules contain the following 100 games per condition:

| Seed | Frozen, learner black | Frozen, learner white | Sanmill, learner black | Sanmill, learner white |
|---:|---:|---:|---:|---:|
| 55 | 33 | 28 | 23 | 16 |
| 56 | 34 | 28 | 19 | 19 |
| 57 | 20 | 22 | 35 | 23 |

## Normalization semantics

For an informative batch, the treatment computes detached gradient norms for:

1. the ordinary policy-head objective, including policy and entropy terms but
   excluding the value objective; and
2. the unscaled exact-WDL Malom policy auxiliary.

The effective coefficient targets
`0.25 * ordinary_norm / raw_auxiliary_norm` and is capped at 0.25. The
denominator floor is `1e-12`. An informative batch with an auxiliary norm below
that floor fails closed. If the ordinary policy-head norm is below the floor,
the auxiliary coefficient is explicitly zero. A batch with no downgrading
alternative is labelled but applies no auxiliary update.

The control keeps the selected-action Malom quality diagnostics but does not
enumerate the all-action auxiliary labels and applies no auxiliary gradient.
The treatment enumerates the complete legal action set because those labels
are required to compute its loss. This diagnostic-cost difference is recorded;
it is not a second learning signal, and the completed-game rather than
wall-clock comparison is primary.

## Evidence to collect

The result must retain raw and complete-window training curves for policy loss,
value loss, entropy, raw auxiliary loss, effective coefficient, applied ratio,
and gradient cosine. It must also report the 29-state preserving mass and
entropy before and after each arm.

Metrics must be disaggregated by seed and condition, board phase, opponent
source, learner colour, and termination reason. Exact selected-action downgrade
rates must be reported by phase and opponent source. W/D/L must remain separated
by opponent source and colour rather than collapsed into one headline rate.

There is no validation-loss curve in this calibration because no supervised
train/validation split is being optimized. The fixed-state diagnostic is the
predeclared development comparison; later held-out evaluation remains separate.

The contract also pins the dedicated result analyzer and immutable publisher
by path and SHA-256 before any arm can be authorized. The analyzer rejects
partial normalization diagnostics, non-reconciling norms or ratios, incomplete
phase support, changed runtime identities, incomplete arms and post-training
decision-rule changes. The result publisher writes once into the ignored
experiment directory and refuses overwrite.

After, and only after, all six authorized arms have completed, the frozen
publisher command is:

```powershell
.\.venv\Scripts\python.exe `
  scripts/report_malom_policy_auxiliary_normalized_calibration.py
```

Before completion it must return `not_reportable`; it never starts, resumes or
modifies training.

## Frozen decision rule

For each arm, compare the final candidate with a deterministically reconstructed
scratch model on the fixed 29-state corpus. Within each seed, subtract the
control change from the normalized change. The mechanism is eligible only for a
later effectiveness experiment when all of the following hold:

- all six arms complete with finite updates and unchanged identities;
- every treatment update has complete exact Malom labels;
- the applied gradient ratio never exceeds 0.250001;
- the median paired preserving-mass gain is at least 0.001;
- at least two of the three seed-pair gains are positive;
- fixed-state entropy falls by no more than 0.15 beyond control;
- repetition-draw rate rises by no more than 0.10 beyond control; and
- all referee, database, checkpoint, policy-health, and resource gates pass.

Passing permits only the design of a later effectiveness experiment. It does
not authorize that experiment, a long run, promotion, publication, or a strength
claim. Failure or ambiguity means stop and redesign; the thresholds must not be
changed after seeing results.

## Preparation and launch boundary

The source-only audit command is:

```powershell
.\.venv\Scripts\python.exe `
  scripts/prepare_malom_policy_auxiliary_normalized_calibration.py
```

After all implementation and preparation commits are published and the source
audit is clean, `--prepare` may create six managed plans, six preflight records,
and six isolated database copies. It cannot authorize or run a segment.

This document and its JSON contract do not authorize `authorize`, `run-next`,
training, automatic retry, extension, resume, promotion, or publication.
