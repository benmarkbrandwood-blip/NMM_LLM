# Normalized Malom policy-auxiliary target-response audit v1

Status: `completed_stop_gradient_ratio_escalation`

Machine-readable plan:
[`sanmill-malom-policy-auxiliary-normalized-target-response-audit-v1.json`](sanmill-malom-policy-auxiliary-normalized-target-response-audit-v1.json)

This is a read-only optimizer-mechanism audit. It creates no games, performs
no update on a persisted model or optimizer, changes no database, and makes no
playing-strength, promotion, target-selection, or launch claim.

## Question

The completed normalized-0.25 calibration produced a positive paired
fixed-state preserving-mass change in all three seeds, but its median gain was
only `2.9817801611209394e-05`, well below the frozen `0.001` threshold. Was
that small response caused by choosing a target that is too weak, or does the
one-step optimizer response saturate or become unsafe as the target rises?

## Frozen inputs

The audit uses only the final-flush batch from each persisted normalized
treatment arm, in seed order 55, 56, 57. Each comparison starts from the same
pre-update model and Adam state saved at game 100. The real final checkpoint
is used only to prove that the disposable target-0.25 replay matches the
production update.

All checkpoint, update-log, and completed-result byte identities are fixed in
the JSON plan. The implementation is fixed at commit
`fb779091789581ce3d5264f202e96ce555917a9b` and by source-file SHA-256.

## Frozen comparison

For every seed, compare target policy-head ratios `0.25`, `0.50`, and `1.00`.
The coefficient is derived from the ordinary policy-plus-entropy gradient and
the raw exact-WDL preserving-set gradient, then capped at `0.25`. Each target
gets one production-equivalent Adam update on independent deep copies.

Record:

- labelled and informative phase support;
- raw and applied gradient norms, cosine, joint norm, and clipping;
- uncapped and applied coefficients and the realized ratio;
- parameter movement relative to auxiliary-off;
- informative preserving probability before and after the update;
- post-update entropy and baseline-to-treatment policy KL; and
- exact replay residuals for the real target-0.25 update.

The three final batches contain placement-informative examples but no
movement- or flying-informative examples. The audit must preserve and report
that limitation; it cannot support a phase-generalization claim.

## Decision boundary

An escalation mechanism may be designed only if all three seeds show a
non-decreasing preserving-mass response across the ordered targets, the
target-1.00 median exceeds the target-0.25 median, target 0.50 is uncapped in
all seeds, mean policy KL is at most `1e-4`, absolute informative entropy
movement is at most `0.01`, target-0.25 replay is within `1e-6`, and no source
byte changes.

Passing that screen permits preparation of a new independent-seed learning
calibration only. It does not select a setting for retained or long training.
A target-1.00 cap is a measured ceiling, not automatically a failure.

If the response is non-monotonic, already saturated, or outside the safety
bounds, stop increasing the gradient ratio. The next comparison should then
keep auxiliary-off A2C as the control and separately design either a
KL-constrained teacher update or a safe-action sampling mechanism. Completed
calibration thresholds must not be lowered after observing the result.

## Execution

After this plan and its source implementation are committed with a clean
tracked worktree, execute the tool once on CPU with the exact source commit.
The ignored output is created exclusively and must never be overwritten.

```powershell
.\.venv\Scripts\python.exe `
  tools\audit_malom_policy_auxiliary_normalized_target_response.py `
  --plan docs\experiments\sanmill-malom-policy-auxiliary-normalized-target-response-audit-v1.json `
  --output out\malom-policy-auxiliary-normalized-target-response-audit-v1\result.json `
  --expected-source-commit <frozen-plan-commit>
```

## Completed outcome

The one permitted audit completed without mutating its persisted inputs. The
target-0.25 production update replayed in all three seeds and every bounded
response check passed. Seed 56 nevertheless failed the frozen monotonicity
gate: its already tiny preserving-mass delta became more negative as the
target increased. The preregistered verdict is therefore
`stop_gradient_ratio_escalation`.

Preserve the detailed
[result evidence](../evidence/sanmill-malom-policy-auxiliary-normalized-target-response-2026-08-10.md).
No normalized target is selected for retained training, and this audit grants
no authority for another calibration or long run.
