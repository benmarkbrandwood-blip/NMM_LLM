# Training Anomalies and Scientific Results

Answer the requested question from the named run's existing evidence. Bind its
run/segment identity, artifact versions, and owning experiment contract; use
current code when diagnosing current behavior and the recorded producing code
when interpreting historical results. Follow AGENTS.md's data and label trust
rules. Audit database inputs read-only and perform only relevant, required
identity checks; avoid speculative large-file hashing.

Do not require fresh/resume classification, a new frozen launch command, or
launch authorization to report existing evidence. A stopped or failed run can
still support a completed analysis. New experiments are proposals unless
execution is separately requested and authorized.

## Operational Diagnosis

For a path, process, or checkpoint error, report the observed failure, decisive
evidence, supported cause or remaining hypotheses, and next action. Use existing
safe tests or a disposable reproduction as appropriate. In a read-only review,
report missing coverage without adding code or instrumentation. If a repair is
requested, establish the failing reproduction and complete the relevant fix and
verification under AGENTS.md; do not stop at proposing the fix.

## Learning, Causality, and Playing Strength

Choose evidence proportional to the claim, subject to all experiment-owned
acceptance requirements. Preserve these scientific boundaries:

- Inspect raw and smoothed training/validation curves with windows, sample
  counts, segment boundaries, and axes. Distinguish measurements from plotting
  artifacts and rules draws from max-ply truncations.
- Report individual fixed seeds, central tendency, dispersion, and outliers
  when comparing learning outcomes. A single seed supports only its observed
  run, not general improvement or robustness.
- Bind exact hyperparameters and schedules: optimizer, learning rate,
  temperature, entropy, batch/update cadence, rollout horizon, opponent mix,
  search budget, and enabled components where applicable.
- Bind dataset, split, ruleset, label schema, and database versions. Check
  leakage, identity drift, and class balance for conclusions they could affect.
- Compare strength against a frozen compatible baseline under the same rules,
  starts, colors, work budget, and adjudication. Use controlled ablations for
  causal claims, changing one relevant factor unless an interaction experiment
  is explicitly designed.
- Report class support and relevant phase, opponent, color, and termination
  breakdowns; include macro/micro summaries when imbalance matters.

Missing evidence limits the conclusion; it does not justify fabricating metrics
or starting additional runs. If an RL run lacks supervised validation, existing
authorized frozen held-out results may be reported as a separately named
validation-like measure. Do not create held-out exposure or relabel training
metrics as validation to fill the gap.

For substantive scientific diagnosis, distinguish observed facts, falsifiable
hypotheses, supporting evidence, counterevidence/confounders, and proposed next
validation. Use separate headings when they clarify the result, not a mandatory
five-section template for every operational error. A proposed decisive experiment
should specify control, changed variable, seeds, data version, metrics,
acceptance rule, and resource bounds; proposing it grants no launch authority.

Mark predictions and uncertainty explicitly. Curve correlation, training-tail
improvement, single-seed results, and anti-collapse gates are not causal proof
or playing-strength promotion. Deliver the supported answer and limitations
without imposing launch verdicts on an analysis-only task.
