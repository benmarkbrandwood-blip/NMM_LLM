# Human Move Policy Eval Hardening Plan

This document rewrites the evaluation-hardening plan for the human move policy net with a stricter formal-evidence standard. It is based on the current `tools/eval_human_move_policy_net.py` implementation and the follow-up review comments about strict inference, Malom-label semantics, macro/micro degradation calibration, temperature handling, baseline interpretation, split diagnostics, provenance compatibility, and holdout policy.[cite:33]

The purpose of this plan is to make the evaluator fail closed, bind it to compatible artefacts, remove ambiguous or self-referential evidence, and freeze a defensible validation-only contract before any reserved confirmation slice is consumed.[cite:33] It does **not** assume policy-model retraining is immediately required, but it does make any “no retraining required” conclusion conditional on a strict dataset and feature-bank audit.[cite:33]

## Objective

The evaluator should become a formal evidence path, not a best-effort diagnostic script.[cite:33] That means the script must stop silently recovering from inference failures, stop treating missing labels as non-degrading zeroes, stop mixing macro predictions with micro observations, stop presenting a self-scored empirical distribution as a predictive baseline, and stop overstating the evidentiary status of the reserved 5% confirmation slice.[cite:33]

Stage C should remain blocked until Stage A’s versioned Malom regret API and golden corpus pass, and until this hardened Stage B evaluation contract is implemented and rerun on validation only.[cite:33] Overlay exploration may continue in parallel because disagreement among AI policy, heuristic preference, and predicted human policy is diagnostically useful and is not itself an error condition.[cite:33]

## Files to edit

| File | Required role |
|---|---|
| `tools/eval_human_move_policy_net.py` | Main evaluation contract: strict inference usage, degrade semantics, baseline naming, split diagnostics, provenance, reporting.[cite:33] |
| `ai/human_move_policy_advisor.py` | Add strict probability/logit API with no silent fallback path for formal evaluation.[cite:33] |
| `tools/extract_human_move_policy_dataset.py` or a new auditor | Audit successor-feature-bank rows against strict re-encoding across all splits; rebuild dataset only if mismatches/failure rows are found.[cite:33] |
| `tests/test_human_move_policy_eval.py` | Add regression tests for macro/micro correction, partial-label semantics, strict inference, provenance compatibility, and temperature comparisons.[cite:33] |
| Advisor/evaluator tests | Add targeted failure-mode tests for strict paths and fail-closed behavior.[cite:33] |
| `docs/gap_net_v3_plan.md` | Replace superseded ECE/leakage/degradation claims after corrected validation rerun.[cite:33] |
| `docs/human_move_policy_net_plan.md` | Update baseline semantics, calibration language, and split interpretation.[cite:33] |
| `docs/nmm_human_move_policy_eval_report.md` | Mark old conclusions superseded and replace them only after corrected validation evidence exists.[cite:33] |

## 1. Strict inference path

The current evaluator cannot detect every bad row because `HumanMovePolicyAdvisor.probs()` already repairs some failures internally by returning uniform probabilities, and the successor encoder can write zero feature rows on application or encoding failures.[cite:33] By the time `tools/eval_human_move_policy_net.py` sees the returned `probs`, the original failure may already have been hidden.[cite:33]

### Required changes in `ai/human_move_policy_advisor.py`

Add a strict API such as:

- `raw_logits_strict(...)`
- `probs_strict(...)`

The strict path must raise on:

- successor application failure;[cite:33]
- feature-encoding failure;[cite:33]
- non-finite raw logits;[cite:33]
- output shape mismatch;[cite:33]
- negative or non-finite probabilities;[cite:33]
- invalid normalization / zero softmax total.[cite:33]

The formal evaluator must use the strict path only.[cite:33] The existing fallback path may remain available for gameplay overlays or exploratory tools, but it must not be used in temperature fitting or formal validation/confirmation evaluation.[cite:33]

### Required changes in `tools/eval_human_move_policy_net.py`

Replace the current probability call in `_run_eval_loop()` with the strict version, and use the strict raw-logit path inside `_find_temperature()` so Pass 1 does not fit temperature over corrupted rows.[cite:33] The current proposed post-hoc checks after `probs = adv.probs(...)` are insufficient because the fallback may already have hidden the underlying defect.[cite:33]

### Skip policy for formal evaluation

Board reconstruction failure, legal-move-count mismatch, zero-target samples, encoding failures, or any strict inference failure should no longer be silently skipped.[cite:33] For the formal contract, either fail closed immediately or track exact attempted/evaluated/skipped counts and require zero skips before accepting the report.[cite:33]

A recommended structure is:

- `n_attempted_samples`
- `n_evaluated_samples`
- `n_skipped_board_reconstruction`
- `n_skipped_legal_count_mismatch`
- `n_skipped_zero_target`
- `n_skipped_inference_failure`

For a frozen evaluator, all skip counts should be zero or the run should fail.[cite:33]

### Dataset / feature-bank audit

Because the extractor may previously have written zero rows when successor encoding failed, the current extracted feature bank must be audited against strict re-encoding across all splits.[cite:33] If the stored rows match strict regeneration with zero failures, the current model can remain eligible.[cite:33] If any zero-row artefact or feature mismatch is found, the dataset must be rebuilt with fail-closed extraction and the model retrained before any formal Stage B pass claim is made.[cite:33]

This makes “no retraining required” conditional on a completed audit, not a default assumption.[cite:33]

## 2. Malom label integrity and partial-label semantics

The current degradation loop accumulates degrading probability and severity only over labelled moves, but still keeps all human events in the denominator, and unlabelled or `label_inconsistency` moves effectively contribute zero degrade mass and zero regret.[cite:33] That can bias both predicted and observed degradation downward.[cite:33]

### Required semantic change

Missing or inconsistent Malom labels must never be treated as safe zero-regret moves.[cite:33] A `label_inconsistency` should be treated as a data-integrity failure, not as a non-degrading category.[cite:33]

For degradation calibration, choose one of the following and encode it explicitly:

1. **Complete trusted classification requirement** — include a state-band sample only if all legal moves have trusted labels; otherwise skip it with a counted reason.[cite:33]
2. **Labelled-support conditioning** — condition both predicted and observed quantities on the labelled subset only, and report all coverage numbers explicitly.[cite:33]

### Required reporting additions

For each degrade-calibration view, report:

- number of legal moves and labelled legal moves;[cite:33]
- total policy mass covered by labels;[cite:33]
- total human events and labelled human events;[cite:33]
- number of skipped partial-label samples;[cite:33]
- skip reasons, including missing labels and label inconsistencies.[cite:33]

### Rename current “regret” quantity

The current 0/0.5/1 weights are WDL downgrade severity, not the future versioned `malom.query_regret` value.[cite:33] Rename this quantity in code and report output to `wdl_downgrade_severity` or similar so it cannot be confused with the Stage A regret API.[cite:33]

## 3. Complete the macro/micro correction

The current `DegradeCal` compares a macro prediction (`mean_pred_prob_degrade`) against a micro observation (`obs_degrade_freq`) and does the same for the current expected-regret field.[cite:33] That means the reported “approximately 2.7× underprediction” should be withdrawn until corrected metrics are rerun.[cite:33]

### Unit naming

The unit here is a `(state_key, Elo band)` sample, not necessarily a unique position.[cite:33] The code and report should name it accordingly.

### Required matched outputs

For both degradation probability and WDL downgrade severity, report:

- macro prediction vs macro observation, weighting each state-band sample equally;[cite:33]
- micro prediction vs micro observation, weighting each sample by its human event count.[cite:33]

### ECE naming and weighting

ECE should not remain a generic single number beside both macro and micro quantities.[cite:33] Ideally report both:

- state-band-weighted ECE;[cite:33]
- event-weighted ECE.[cite:33]

At minimum, every ECE field must state its weighting explicitly.[cite:33]

### Stratified degradation calibration

Report degradation calibration by:

- Elo band;[cite:33]
- phase.[cite:33]

Phase should use the mover’s actual `get_game_phase()` result and distinguish placement, movement, and flying; the current code folds flying into movement.[cite:33]

## 4. Compare `T=1` and `T*` using the same strict logits

Temperature scaling should continue to be fitted on validation only and later frozen for any confirmation-slice evaluation.[cite:33] But the evaluation should compute strict raw logits once per sample, then derive both `T=1` and `T*` distributions from those same logits so the calibration comparison uses exactly the same rows.[cite:33]

### Required metrics at both temperatures

At minimum report, for validation:

- NLL at `T=1` and `T*`;[cite:33]
- top-label ECE at `T=1` and `T*`;[cite:33]
- macro and micro degradation calibration at `T=1` and `T*`.[cite:33]

Top-k ranking need not be duplicated because scalar temperature does not change ranking.[cite:33]

### Optimizer robustness

The temperature optimizer result must be checked for:

- success flag;[cite:33]
- finite value;[cite:33]
- boundary saturation.[cite:33]

For a formal run, missing SciPy or optimizer failure should not silently fall back to `T=1`.[cite:33] `T=1` should occur only under an explicit `--skip-temperature` contract.[cite:33]

## 5. Baseline and metric semantics

The current `_empirical_probs(targets)` function is self-referential because it builds a distribution from the sample’s own observed counts and scores that same distribution on the same sample.[cite:33] This is descriptive and can still be useful, but it is not a predictive empirical baseline and should not be used as promotion evidence.[cite:33]

### Required low-risk change for this pass

Do **not** implement the earlier exact-state train-only OOF baseline idea in this patch.[cite:33] Because the split is already by `state_key`, validation state keys do not exist in the training partition and exact-state train-only coverage would be effectively zero.[cite:33]

Instead:

- rename the current baseline to `in_sample_empirical_reference`;[cite:33]
- state clearly that it is descriptive and non-predictive;[cite:33]
- remove it from gates and promotion logic;[cite:33]
- rename the KL result to something like `observed_distribution_kl_to_model`.[cite:33]

### Uniform baseline

Uniform NLL and Brier remain valid baselines.[cite:33] Uniform top-k is currently tied to arbitrary first-legal-move ordering, so either remove uniform top-k entirely or report expected random top-k coverage instead.[cite:33]

### Model metric naming corrections

The current model top-1/top-3/top-5 fields are event-weighted coverage of actual human moves, not accuracy against one most-frequent human move per position.[cite:33] The docs and metric names should be corrected accordingly.[cite:33]

The existing Brier score is averaged equally over samples, so it should be labelled as macro multiclass Brier unless an event-weighted variant is added.[cite:33]

## 6. Split diagnostics and session-index compatibility

The current OOD row duplicates the full validation/test set by construction because the split is already by state key.[cite:33] Rather than reporting duplicate metrics under the name OOD, replace it with a split-integrity assertion and report any violation as an error.[cite:33]

### `game_val_only` sensitivity slice

`game_val_only` should be renamed to a diagnostic sensitivity slice and removed from promotion or leakage gates.[cite:33] The current condition excludes train-game membership but can still include states reached by test games, so either require an exclusively val-game mask or name the actual condition precisely.[cite:33]

If the reserved confirmation slice is eventually run, the current loop should not keep reusing the val-game mask without a corresponding test-game diagnostic definition.[cite:33] Either implement a test-appropriate slice or omit game-slice reporting from confirmation evaluation.[cite:33]

### Session-index compatibility checks

If a session index is explicitly requested, the evaluator should fail if it is missing or incompatible rather than warning and continuing.[cite:33] It should validate:

- state-key ordering compatibility;[cite:33]
- array length compatibility;[cite:33]
- embedded dataset metadata hash equality to the current `metadata.npz`.[cite:33]

Recording that `player_split_mask` exists is useful, but the report must explicitly state that it is not consumed by the evaluator.[cite:33]

## 7. Strengthen provenance as a compatibility chain

The evaluator should not just record hashes; it should also compare them for compatibility.[cite:33] The candidate SQLite database should be opened read-only, an integrity or quick check should be run, and the shared Malom provenance helper and version constant should require `sector-corrected-v1`.[cite:33]

### Artefacts to bind

The report should bind and compare:

- model SHA-256;[cite:33]
- candidate DB SHA-256 and Malom label version;[cite:33]
- dataset metadata identity and successor-feature-bank identity;[cite:33]
- session-index SHA-256 and its embedded dataset metadata hash;[cite:33]
- evaluator Git commit, evaluator script hash, and clean/dirty status;[cite:33]
- model-embedded dataset provenance and Elo-band configuration.[cite:33]

### Compatibility checks

Examples of required comparisons:

- candidate DB hash in the model/dataset provenance must equal the DB being evaluated;[cite:33]
- session index must identify the current dataset;[cite:33]
- evaluator artefacts must be finite, present, and mutually compatible.[cite:33]

A list of unrelated hashes is not enough; the evaluator must establish a provenance chain.[cite:33]

## 8. Update owning documents in the same change set

The documentation should be updated alongside the code, not later.[cite:33] Current documents still contain old ECE results, overstate the in-sample empirical reference, treat `game_val_only` as evidence against leakage, and record the 2.7× degradation result as if it were based on a valid denominator comparison.[cite:33]

Before committing the plan or evaluator update, remove or replace any placeholder citations such as `[cite:33]` with real file, function, or commit references.[cite:33]

## 9. Preserve the reserved confirmation slice

The 5% partition should be described as a **reserved development-confirmation holdout**, not as a pristine untouched test set.[cite:33] Those hash buckets were part of the older v1 validation range, so the v2 model did not train on them, but the project has already received information from that region during design and earlier evaluation work.[cite:33]

Continue to run validation only while implementing and reviewing these changes.[cite:33] After the evaluator, tests, interpretation, and thresholds are committed and frozen, rerun validation, inspect the corrected JSON, and only then decide whether to consume the confirmation slice once.[cite:33] It should not be used as final independent publication evidence.[cite:33]

## Recommended implementation order

1. Strict inference, data/label integrity, and provenance compatibility.[cite:33]
2. Macro/micro correction and partial-label semantics.[cite:33]
3. Temperature comparison and metric/baseline corrections.[cite:33]
4. Split diagnostics and report wording.[cite:33]
5. Focused synthetic regression tests.[cite:33]
6. Commit the evaluator contract.[cite:33]
7. Rerun validation only and inspect the corrected JSON.[cite:33]
8. Freeze interpretation and thresholds.[cite:33]
9. Run the reserved confirmation slice separately only if still justified.[cite:33]

## Acceptance criteria

The hardening pass is complete when:

- the formal evaluator uses a strict advisor path and cannot silently replace failures with uniform outputs;[cite:33]
- board/encoding/legal-count/target anomalies are either zero or cause the formal run to fail;[cite:33]
- missing or inconsistent Malom labels never become zero degradation or zero severity by default;[cite:33]
- degradation metrics report matched macro and micro pairs with explicit weighting;[cite:33]
- degradation calibration is available at both `T=1` and `T*` from the same strict logits;[cite:33]
- the self-scored empirical reference is clearly labelled descriptive and non-predictive;[cite:33]
- OOD is replaced by a split-integrity assertion, and `game_val_only` is demoted to a precisely named diagnostic slice;[cite:33]
- provenance establishes compatibility between model, DB, dataset, session index, and evaluator artefacts;[cite:33]
- all relevant docs are updated and old quantitative claims are marked superseded;[cite:33]
- validation is rerun under the frozen evaluator contract before any reserved confirmation-slice execution.[cite:33]
