# Human Move Policy Eval Hardening Plan

## Status

The top-label ECE fix and `DegradeCal` class shipped in commit `2827934`, together with a corrected validation run at `data/gap_v3_prerequisite_eval.json` (T*=0.767). The remaining items in this plan refer to work not yet committed.

## Objective

Make `tools/eval_human_move_policy_net.py` a formal evidence path: fail-closed on inference and encoding failures, bound to compatible artefacts, free of ambiguous baselines, with matched macro/micro calibration metrics and a defensible provenance chain. All changes must complete and validation must be re-run before any reserved confirmation-slice execution.

## Files

| File | Role |
|---|---|
| `tools/eval_human_move_policy_net.py` | Strict inference, degrade semantics, macro/micro correction, baseline renaming, split diagnostics, provenance chain, reporting. |
| `ai/human_move_policy_advisor.py` | Add `raw_logits_strict()` and `probs_strict()` with no silent fallback. |
| `tools/extract_human_move_policy_dataset.py` | Full-corpus audit: re-encode every row in every split; rebuild only if mismatches found. |
| `tests/test_human_move_policy_eval.py` | Regression tests for strict paths, macro/micro correction, partial-label semantics, provenance compatibility. |
| `docs/gap_net_v3_plan.md` | Replace superseded ECE/leakage/degradation claims after corrected rerun. |
| `docs/human_move_policy_net_plan.md` | Update baseline semantics, calibration language, split interpretation. |
| `docs/nmm_human_move_policy_eval_report.md` | Mark old conclusions superseded; replace only after corrected evidence. |

## 1. Strict inference path

**Problem.** `HumanMovePolicyAdvisor.probs()` (`ai/human_move_policy_advisor.py:151`) hides inference failures by returning uniform probabilities when logits are non-finite or the softmax total is zero. `_successor_features()` (line 101) writes zero rows on `apply_move` or `board_to_features` failure. Failures hidden before `_run_eval_loop` sees the result are undetectable by post-hoc checks.

**Required in `ai/human_move_policy_advisor.py`.**

Add two methods:

- `raw_logits_strict(board, legal_moves, elo_band) → np.ndarray` — raises on successor application failure, feature-encoding failure, non-finite raw logits, or output shape mismatch.
- `probs_strict(board, legal_moves, elo_band) → np.ndarray` — calls `raw_logits_strict`, then raises on negative, non-finite, or zero-total softmax output. No silent fallback.

The existing `probs()` fallback path may remain for gameplay overlays only.

**Required in `tools/eval_human_move_policy_net.py`.**

Replace `adv.probs(...)` in `_run_eval_loop` (line 469) with `adv.probs_strict(...)`. Replace `adv.rank(...)` in `_find_temperature` (line 327) with `adv.raw_logits_strict(...)`.

**Skip policy.** Classify each failure type separately. Track:

```
n_attempted_samples
n_evaluated_samples
n_skipped_board_reconstruction   # board_from_state_key returns None
n_skipped_legal_count_mismatch   # get_all_legal_moves count ≠ stored count
n_skipped_zero_target            # all observed counts are zero
n_skipped_inference_failure      # probs_strict raised
```

For the formal gate: `n_skipped_inference_failure` must be zero. Board-reconstruction and legal-count skips combined must be < 0.1% of `n_attempted_samples`. Either violation fails the run.

**Feature-bank audit.** Re-encode every row across all splits using the strict encoder from `extract_human_move_policy_dataset.py`. Zero-row artefacts or any feature mismatch require a full dataset rebuild and model retrain before any formal gate pass is claimed. This audit must complete before §1 acceptance criteria are met.

**Acceptance criteria (§1).**
- `probs_strict` and `raw_logits_strict` exist in `HumanMovePolicyAdvisor`.
- Formal evaluator uses strict paths only; `probs()` is not called in `_run_eval_loop` or `_find_temperature`.
- All six skip counters present in the report.
- A validation run with `n_skipped_inference_failure == 0` recorded.
- Feature-bank audit complete with zero zero-row artefacts or mismatches.

## 2. Malom label integrity and partial-label semantics

**Problem.** Missing or `label_inconsistency` moves contribute zero degrade mass and zero regret but stay in the denominator, biasing both predicted and observed calibration downward. The field names `pred_regret` / `regret_ece` conflict with the future Stage A `malom.query_regret` API.

**Two-tier rule.**

*Formal gate / promotion evidence (Option 1):* include a `(state_key, band)` sample only if all legal moves have trusted Malom labels. Partial-label skips are counted under `n_skipped_partial_labels`. For a formal gate pass, this count must be zero.

*Exploratory / diagnostic report (Option 2):* condition both predicted and observed quantities on the labelled subset only. Clearly label this view `"diagnostic_only": true` in the report; never use it in promotion criteria or gates.

**Rename throughout code and report output:**

| Old name | New name |
|---|---|
| `pred_regret` | `pred_wdl_severity` |
| `obs_regret` | `obs_wdl_severity` |
| `regret_ece` | `wdl_severity_ece` |
| `mean_pred_regret` | `mean_pred_wdl_severity` |

**Required reporting additions (per degrade-calibration view):**
- `n_legal_moves`, `n_labelled_legal_moves`
- `total_policy_mass_covered` (sum of `probs[j]` for labelled moves)
- `n_human_events_total`, `n_human_events_labelled`
- `n_skipped_partial_labels` and skip reason breakdown

**Acceptance criteria (§2).**
- Formal degrade calibration uses only samples with 100% trusted labels.
- Exploratory view present and marked `"diagnostic_only": true`.
- No `regret` in any field name; all renamed to `wdl_severity`.
- Reporting additions present in both views.

## 3. Macro/micro correction

**Problem.** `DegradeCal.finalize()` (line 257) compares `mean_pred_prob_degrade` (position-weighted) against `obs_degrade_freq` (event-weighted). The denominators differ. The "~2.7× underprediction" observation in `gap_net_v3_plan.md` §16 is not valid until corrected metrics are rerun.

**Unit.** The unit throughout the report is a `(state_key, Elo band)` sample — not a unique board position. Name it accordingly in code and output.

**Required matched outputs.** For both degrade probability and WDL severity:

- **Macro** (sample-weighted): `macro_pred` = mean of per-sample predicted values; `macro_obs` = mean of per-sample observed frequencies (`obs_degrade / obs_total` per sample).
- **Micro** (event-weighted): `micro_pred` = Σ(`pred × obs_total`) / Σ(`obs_total`); `micro_obs` = Σ(`obs_degrade`) / Σ(`obs_total`).

**ECE naming.** Report both `macro_degrade_ece` (sample-weighted) and `micro_degrade_ece` (weighted by `obs_total`). Every ECE field must name its weighting in the field name.

**Stratified degradation.** Report degrade calibration by Elo band and by phase. Phase must use `board.get_game_phase()` and distinguish placement, movement, and flying separately — the current code folds flying into movement.

**Acceptance criteria (§3).**
- Four paired outputs per view: `macro_pred`, `macro_obs`, `micro_pred`, `micro_obs`.
- No single `obs_degrade_freq` without a `macro_obs_degrade_freq` alongside it.
- ECE field names state their weighting.
- Degrade calibration reported per band and per phase including fly.

## 4. Temperature: single-pass strict logits, T=1 and T* comparison

**Changes to `_find_temperature`.** Collect raw logits once per sample using `adv.raw_logits_strict(...)`. Optimizer failure (non-success flag, non-finite result, or boundary saturation at T=0.1 or T=10.0) must raise `RuntimeError` — not silently fall back to T=1.0. `T=1.0` occurs only under an explicit `--skip-temperature` contract.

**Pass 2.** Derive both T=1 and T* distributions from the same collected strict logits. Report, for validation at both temperatures:
- NLL at T=1 and T*
- Top-label ECE at T=1 and T*
- Macro and micro degrade calibration at T=1 and T*

Top-k ranking need not be duplicated (scalar temperature does not change ranking).

**Acceptance criteria (§4).**
- `_find_temperature` raises if scipy absent, optimizer fails, or T* saturates at a boundary.
- Report contains `t1` and `t_star` sub-keys under every calibration metric.
- Logit collection is a single pass; T=1 and T* results derived from identical rows.

## 5. Baseline and metric semantics

**In-sample empirical reference.** `_empirical_probs` (`tools/eval_human_move_policy_net.py:274`) builds a distribution from the sample's own observed counts and scores it on those same counts. This is descriptive, not predictive.

- Rename report key: `baseline_empirical` → `in_sample_empirical_reference`.
- Add `"descriptive_only": true` field.
- Remove from gates and promotion logic.
- Rename KL result: `empirical_kl_supported` → `observed_distribution_kl_to_model`.

**Uniform top-k.** `uni_t1 = notations[0]` (`_run_eval_loop` line ~536) is arbitrary. Either remove uniform top-k or report expected random top-k coverage (1/n_legal averaged over samples).

**Model metric naming.** `top1`/`top3`/`top5` are event-weighted human-move coverage by the model's top-k set, not accuracy against one most-frequent human move per position. Rename to `human_event_top1_coverage` etc., or add an explicit schema note in the report.

**Brier.** The current Brier is macro (averaged equally over samples). Label it `macro_multiclass_brier`.

**Acceptance criteria (§5).**
- `baseline_empirical` → `in_sample_empirical_reference` in all report keys.
- `"descriptive_only": true` present; key absent from gates.
- Uniform top-k removed or reports expected random coverage.
- Model top-k fields clearly named as event-weighted coverage.

## 6. Split diagnostics

**OOD tautology.** The v2 split assigns each `state_key` to exactly one partition, so 100% of val `state_key`s are by construction absent from training. The `ood` stratum is identical to the full val set. Replace `ood` with a split-integrity assertion: verify that no val `state_key` appears in the train partition and report this as `"split_integrity_check": {"pass": bool, "n_val_keys_in_train": int}`. Remove the `ood` metrics stratum.

**`game_val_only`.** The current condition (`(mask & MASK_TRAIN) == 0 and (mask & MASK_VAL) != 0`) excludes train-game state keys but may include states reached by test games. Rename to `val_game_exclusive_stratum` and document the exact condition. If the session index does not cover test-game membership, report `"approximate_condition": true` alongside the stratum.

**Session-index compatibility.** If `--session-index` is passed, raise `RuntimeError` (do not warn) if the file is absent or if any of these checks fail:
- Array length matches `n_state_keys` in dataset metadata.
- State-key ordering compatible (spot-check first/last 100).
- Embedded `metadata_hash` equals the current `metadata.npz` hash.

Record that `player_split_mask` exists but is not consumed by the evaluator.

**Acceptance criteria (§6).**
- `ood` stratum removed; `split_integrity_check` present.
- `game_val_only` renamed with exact condition documented.
- Session-index incompatibility raises, not warns.

## 7. Provenance compatibility chain

Record and compare (not just record) the following artefacts:

| Artefact | Field | Compatibility check |
|---|---|---|
| Model SHA-256 | `model_sha256` | |
| Candidate DB SHA-256 | `candidate_db_sha256` | Must equal hash embedded in model/dataset provenance |
| Malom label version | `malom_label_version` | Must be `sector-corrected-v1` |
| Dataset metadata hash | `dataset_metadata_hash` | Must match `metadata.npz` hash on disk |
| Session index SHA-256 + embedded dataset hash | `session_index_sha256` | Embedded hash must equal `dataset_metadata_hash` |
| Evaluator git commit + dirty flag | `evaluator_git_commit`, `evaluator_git_dirty` | |
| Evaluator script SHA-256 | `evaluator_script_sha256` | |
| Model-embedded dataset provenance hash | — | Must equal `dataset_metadata_hash` |

Open the candidate DB read-only (`sqlite3.connect(f"file:{path}?mode=ro", uri=True)`) and run `PRAGMA quick_check` at startup.

A list of hashes is not enough: include `"provenance_ok": bool` and a list of any compatibility failures. `provenance_ok: false` should fail the run.

**Acceptance criteria (§7).**
- `provenance_ok` field present; `false` causes the run to fail.
- DB opened read-only with quick-check.
- Malom label version verified against `sector-corrected-v1`.
- Git commit + dirty flag and script SHA both present in report.

## 8. Update owning documents

Update `docs/gap_net_v3_plan.md`, `docs/human_move_policy_net_plan.md`, and `docs/nmm_human_move_policy_eval_report.md` in the same commit that freezes the evaluator contract. Mark all quantitative claims that relied on the broken ECE metric or mismatched macro/micro denominator as superseded. Replace them only after the corrected validation rerun exists.

## 9. Reserved confirmation slice

The 5% hash-bucket holdout is a **reserved development-confirmation holdout**, not a pristine independent test set. Those buckets were within the older v1 validation range; the v2 model did not train on them, but the project has already received information from that region during design and earlier evaluation work.

Do not run the confirmation slice until: (a) the strict evaluator contract is committed and frozen, (b) validation is re-run and the corrected JSON inspected, and (c) thresholds are set from validation numbers only. Consume the confirmation slice at most once. It should not be presented as final independent publication evidence.

## Implementation order and phase gates

| Phase | Work | Gate before proceeding |
|---|---|---|
| 1 | §1: strict advisor API + skip counters + feature-bank audit | Zero inference failures on validation; audit complete with zero mismatches |
| 2 | §2: two-tier degrade semantics + rename wdl_severity | Formal degrade view uses complete-label samples only; exploratory view marked diagnostic |
| 3 | §3: matched macro/micro + stratified degrade cal | Macro and micro pairs reported; fly phase distinguished |
| 4 | §4: single-pass strict logits + T=1 vs T* report | Both temperature metrics present; boundary saturation raises |
| 5 | §5 + §6: baseline rename + split diagnostics | OOD removed; `in_sample_empirical_reference` labelled descriptive |
| 6 | §7: provenance chain | `provenance_ok: true` on a clean working tree |
| 7 | §8: document updates | All superseded claims marked |
| 8 | Rerun validation only | Corrected JSON inspected and thresholds frozen |
| 9 | Confirmation slice (if still justified) | Single execution; result recorded once |

## Acceptance criteria

The hardening pass is complete when:

- The formal evaluator uses `probs_strict` / `raw_logits_strict` and cannot silently replace failures with uniform outputs.
- All six skip counters present; `n_skipped_inference_failure == 0` on validation.
- Feature-bank audit complete with zero zero-row artefacts.
- Partial-label samples excluded from the formal degrade gate; exploratory view marked `diagnostic_only`.
- No `regret` field names remain; all renamed to `wdl_severity`.
- Macro and micro degrade calibration pairs reported with explicit weighting; fly phase separated.
- T=1 and T* metrics derived from identical strict logits; optimizer failure and boundary saturation raise.
- `in_sample_empirical_reference` labelled descriptive and excluded from gates.
- OOD replaced by `split_integrity_check`; `val_game_exclusive_stratum` named with exact condition.
- `provenance_ok: true`; DB opened read-only; git commit and script SHA both recorded.
- Owning docs updated and old quantitative claims marked superseded.
- Validation rerun and inspected under the frozen evaluator before any confirmation-slice execution.
