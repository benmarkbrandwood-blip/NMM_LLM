# Human Move Policy Net Eval Review

This review interprets the stricter validation evaluation and assesses whether the HumanMovePolicyNet prerequisite gates in the GapNet v3 plan are now passed. The assessment is based on two attached artefacts: `gap_v3_prerequisite_eval_V2.json` and `gap_net_v3_plan.md`.

## Verdict

The HumanMovePolicyNet Phase 4b prerequisite gates for GapNet v3 are **passed** under the current plan definition in `gap_net_v3_plan.md`, with one important boundary: the degradation-calibration results should be treated as informative for later GapNet design and consumption choices, not as a blocker for the HumanMovePolicyNet prerequisite itself, which is also the stance already encoded in `gap_net_v3_plan.md`.

In other words, Stage B can reasonably be treated as closed for the purpose of allowing GapNet v3 prerequisite work to proceed, while Stage C and later work should continue to carry forward the calibration caveats recorded in `gap_v3_prerequisite_eval_V2.json` and in the stricter evaluator-hardening discussion.

## Basis for the decision

The relevant gate language appears in `gap_net_v3_plan.md`, especially the HumanMovePolicyNet prerequisite section and the stage-promotion conditions. The evaluation evidence comes from `gap_v3_prerequisite_eval_V2.json`, which reports the strict validation run with temperature scaling, per-band and per-phase strata, split-integrity checks, zero inference skips, provenance checks, and the revised formal-vs-diagnostic degradation calibration outputs.

This review therefore asks a narrow question: does the strict validation artefact satisfy the documented HumanMovePolicyNet prerequisite gates as currently written in `gap_net_v3_plan.md`? The answer is yes.

## 1. NLL gate against the uniform baseline

`gap_net_v3_plan.md` requires the model to beat the uniform baseline by a substantial margin, specifically framed as an event-weighted NLL improvement requirement across bands.

From `gap_v3_prerequisite_eval_V2.json`:

- Model overall event NLL at `T*=0.7674`: 1.5534.
- Uniform overall event NLL: 2.3474.

That is a relative improvement of about 33.8%, which is comfortably above the plan threshold.

Per Elo band, the model also beats uniform clearly:

- Lower band: 1.5912 vs 2.3182, improvement about 31.4%.
- Middle band: 1.5388 vs 2.3377, improvement about 34.2%.
- Upper band: 1.5531 vs 2.3629, improvement about 34.2%.

This means the NLL gate is not just narrowly satisfied; it is passed with a healthy margin in every band reported in `gap_v3_prerequisite_eval_V2.json`.

## 2. ECE gate after temperature scaling

`gap_net_v3_plan.md` requires acceptable calibration after temperature scaling, with an explicit per-band ECE threshold.

From `gap_v3_prerequisite_eval_V2.json`, at `T*=0.7674`:

- Overall top-label ECE: 0.0166.
- Lower band ECE: 0.0370.
- Middle band ECE: 0.0275.
- Upper band ECE: 0.0223.

All of these are below 0.05.

The comparison against `T=1` is also directionally correct and important:

- At `T=1`, overall ECE is 0.0609.
- At `T*=0.7674`, overall ECE drops to 0.0166.

So temperature scaling is doing the job the plan expects it to do, and the calibration gate is passed according to the criteria laid out in `gap_net_v3_plan.md`.

## 3. Strict evaluation integrity and zero-skip behavior

One of the major concerns in the stricter review was that the evaluator should not be silently skipping failures or repairing bad rows invisibly. `gap_v3_prerequisite_eval_V2.json` reports the relevant skip and integrity counters directly.

The validation run reports:

- attempted samples: 384,837;
- evaluated samples: 384,837;
- skipped board reconstruction: 0;
- skipped legal-count mismatch: 0;
- skipped zero-target samples: 0;
- skipped inference failures: 0.

That is the cleanest possible result for the main policy evaluation path. It means the stricter evaluator contract is functioning without any visible operational failures on the validation corpus used for this prerequisite artefact.

`gap_v3_prerequisite_eval_V2.json` also reports:

- split integrity pass: true;
- validation keys appearing in train: 0.

This satisfies the split-integrity requirement as currently stated in `gap_net_v3_plan.md`.

## 4. Abstention and OOD interpretation

`gap_v3_prerequisite_eval_V2.json` reports an abstention row with zero samples and zero events. That supports the claim that there were no strict-inference failures during the formal validation pass.

The more important point is that the current stricter review already recognises that the older OOD wording was too strong for a state-key split. `gap_net_v3_plan.md` already indicates that this wording should be revised, and the V2 eval effectively replaces that old framing with an explicit split-integrity check. That is the correct interpretation: the relevant gate is the integrity assertion, not a supposedly independent OOD slice.

So on the terms of the current plan, the abstention/integrity side of the gate is passed.

## 5. What the model is good at, and where it is weaker

The evaluation in `gap_v3_prerequisite_eval_V2.json` paints a coherent picture of the policy model.

### By phase

- Placement is materially harder than ordinary movement: placement NLL is 1.8326, while movement NLL is 1.2505.
- Fly is clearly the weakest regime: fly NLL is 2.9263, top-1 coverage is only 22.82%, and top-5 coverage is 56.19%.

This is not a gate failure under `gap_net_v3_plan.md`, but it is an important limitation to carry into downstream GapNet work. If a later stage relies heavily on fly-phase human-likelihood estimates, it should do so knowingly and probably stratify diagnostics by phase.

### By legal-move count

The model is strong in low branching-factor positions and degrades as branching grows:

- `lmc_2-5`: NLL 0.8984, top-1 61.40%, top-5 100.00%.
- `lmc_6-10`: NLL 1.2845, top-1 51.33%.
- `lmc_11-20`: NLL 1.9353, top-1 37.66%.
- `lmc_21+`: NLL 2.0178, top-1 33.95%.

Again, this is not a prerequisite failure, but it matters for interpreting where the policy net is likely to help most: it is strongest when the move set is smaller and human preference is more concentrated.

### By transition class

The model is weakest precisely where human moves are most dangerous from the mover’s objective perspective:

- `draw_to_loss`: NLL 2.3546, top-1 21.43%, ECE 0.213.
- `win_to_draw`: NLL 2.0939, top-1 26.69%, ECE 0.137.
- `win_to_loss`: NLL 2.7316, top-1 15.29%, ECE 0.261.

This is not surprising. These are rare, high-cost transitions. It does not invalidate the prerequisite gate, but it is exactly why later GapNet stages should keep human-policy uncertainty and Malom objective semantics formally distinct, as emphasised in `gap_net_v3_plan.md`.

## 6. Degrade calibration: not a Phase 4b blocker, but important for later GapNet work

This is the area that deserves the most careful wording.

`gap_v3_prerequisite_eval_V2.json` now separates degradation calibration into two views:

1. **Formal degradation calibration**, where all legal moves must have trusted Malom labels.
2. **Diagnostic degradation calibration**, conditioned on the labelled subset only.

That is a materially better design than the earlier mixed-denominator version.

### Formal degradation view

For the formal view at `T*=0.7674`, `gap_v3_prerequisite_eval_V2.json` reports `n=2,617` samples and `skip_partial_labels=382,220`.

The headline numbers are:

- `P(degrade)` macro: pred 0.0051 vs obs 0.0054.
- `P(degrade)` micro: pred 0.0073 vs obs 0.0023.
- `WDL severity` macro: pred 0.0026 vs obs 0.0027.
- `WDL severity` micro: pred 0.0036 vs obs 0.0012.

The macro alignment is quite close. The micro values are more separated, with prediction above observation.

The key point is interpretive: this is no longer the old invalid “macro vs micro” mismatch. The numbers are now defined more carefully, and they show that under the strict formal-label requirement the degradation signal exists on only a very small subset of validation samples.

This does **not** read as a HumanMovePolicyNet prerequisite failure under `gap_net_v3_plan.md`. Instead, it reads as an important caution for later GapNet consumption design: any method that tries to translate predicted human-likely moves into expected regret should use these formal degradation numbers carefully and should not overclaim exploitability from disagreement alone.

### Diagnostic labelled-subset view

The diagnostic view in `gap_v3_prerequisite_eval_V2.json` covers a much larger slice: `n=376,037`.

At `T*=0.7674` it reports:

- `P(degrade)` macro: pred 0.0676 vs obs 0.0692.
- `P(degrade)` micro: pred 0.0536 vs obs 0.0498.
- `WDL severity` macro: pred 0.0351 vs obs 0.0360.
- `WDL severity` micro: pred 0.0276 vs obs 0.0257.

These are actually quite close, but because this diagnostic view is explicitly conditioned on the labelled subset, it should remain diagnostic only. That is already the correct framing in `gap_v3_prerequisite_eval_V2.json`, and it matches the stricter review standard.

### Gate interpretation

Under the current wording and intent of `gap_net_v3_plan.md`, these degradation figures are best interpreted as **Stage C design inputs**, not as a Phase 4b blocker. The policy-net prerequisite is about whether the human move policy is good enough and calibrated enough to be trusted as an input dependency. On that question, the answer is yes.

The degradation results simply say: when GapNet later composes human-likelihood with Malom-based objective loss, it must continue to do so with the fail-closed and semantics-aware discipline already described in `gap_net_v3_plan.md`.

## 7. Empirical reference and KL

The eval correctly labels the in-sample empirical distribution as descriptive only. `gap_v3_prerequisite_eval_V2.json` explicitly says it is non-predictive and excluded from gates and promotion criteria.

That is the correct interpretation. Its near-zero Brier and near-zero ECE are exactly what should happen when a sample’s own observed distribution is scored against itself; they do not constitute baseline evidence.

The more useful descriptive quantity is the reported KL divergence:

- Observed distribution KL to model for positions with at least 10 events: mean KL 0.5224 over 4,673 samples.

This is not a gate in `gap_net_v3_plan.md`, but it is a useful descriptive marker of how far the model remains from the empirical observed distribution on the higher-support subset.

## 8. Provenance and compatibility

`gap_v3_prerequisite_eval_V2.json` also looks materially stronger on provenance than the earlier versions.

It records and checks:

- model SHA-256;
- candidate DB SHA-256;
- Malom label version `sector-corrected-v1`;
- DB quick-check OK;
- dataset metadata SHA-256;
- evaluator script SHA-256;
- evaluator Git commit;
- evaluator dirty flag;
- model and dataset provenance blocks;
- compatibility failures: none;
- provenance OK: true.

This is enough to treat the evaluation artefact as bound to a specific model, dataset, DB, and evaluator state. The only soft caution is that the evaluator Git dirty flag is `true`, which is acceptable for an internal development review but ideally would be cleaned before treating the artefact as a final archival report.

For the purpose of the prerequisite gate in `gap_net_v3_plan.md`, though, the provenance chain is strong enough and the compatibility checks pass.

## 9. Final conclusion on the gates

On the evidence in `gap_v3_prerequisite_eval_V2.json`, and against the gate language currently written in `gap_net_v3_plan.md`, the HumanMovePolicyNet prerequisite gates should be treated as **passed**.

### Passed clearly

- Event-weighted NLL beats uniform by a wide margin overall and in every Elo band.
- Post-temperature top-label ECE is below 0.05 in every band.
- Main-path strict evaluation has zero board, legal-count, zero-target, or inference skips.
- Split integrity passes with zero validation keys found in train.
- Abstention is zero.
- Provenance and compatibility checks pass.

### Passed, but carry forward these caveats

- Fly-phase performance is weak relative to placement and ordinary movement.
- Rare degrading transition classes remain difficult and miscalibrated relative to common classes.
- Formal degradation calibration is computed on a very small fully labelled subset, while the much broader labelled-subset view is diagnostic only.
- The degradation outputs should inform later GapNet design and evaluation, but they do not currently justify blocking the HumanMovePolicyNet prerequisite stage.

## Recommended project interpretation

The practical reading is:

1. Treat Stage B / HumanMovePolicyNet prerequisite as closed.
2. Allow GapNet v3 prerequisite work to proceed to the next authorised stage under `gap_net_v3_plan.md`.
3. Keep the degradation-calibration caveats visible in Stage C and later, especially when defining any expected-regret target or gameplay consumption mode.
4. Continue to avoid consuming the reserved confirmation slice unless and until the relevant later-stage evaluator contract is frozen.

That is the most faithful reading of the current evidence and the current written plan.
