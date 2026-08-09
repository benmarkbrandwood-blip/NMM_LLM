# Normalized Malom auxiliary target-response audit — 10 August 2026

## Decision

The frozen no-game audit completed once from a clean tracked source. Its
preregistered verdict is `stop_gradient_ratio_escalation`. Normalized target
ratios `0.25`, `0.50`, and `1.00` are not selected for retained or long
training, and the completed calibration thresholds are unchanged.

The immutable local artefacts are:

- raw audit:
  `out/malom-policy-auxiliary-normalized-target-response-audit-v1/result.json`;
- raw audit identity:
  `819d84d2ed7bb943260aa0627c22db0c0b94944ea2c058ee3fba3116a49f2fa4`;
- raw audit SHA-256:
  `559cc184dd682d1be7c007898dabbfa4ef4d595296bfb954912e1272565e827b`;
- raw audit size: 85,309 bytes;
- decision:
  `out/malom-policy-auxiliary-normalized-target-response-audit-v1/decision.json`;
- decision identity:
  `6f6359df371be56e5b5f25c2a31287363e4d31fe58f87422f4cd46767e6249fc`;
- decision SHA-256:
  `41a8c23985f7290fe63ea550fceceaa080fe893855bfb986c1ded32772445fbf`;
- decision size: 6,217 bytes; and
- clean decision-source commit:
  `32004a5d39cafed3aef9f458dc15c9e9342772c0`.

No game was created. The tool made production-equivalent Adam updates only on
disposable deep copies. It did not mutate a persisted checkpoint, optimizer,
database, calibration result, or tracked source file.

## Observed facts

The target-0.25 update replayed the persisted production update within the
frozen tolerance for all three seeds. Every response was finite, target 0.50
was uncapped in every seed, and no update crossed the policy-KL or entropy
safety bounds. Seed 55 reached the coefficient cap only at target 1.00.

| Seed | Target 0.25 mass delta | Target 0.50 mass delta | Target 1.00 mass delta | Monotonic |
| ---: | ---: | ---: | ---: | :---: |
| 55 | `+0.000000656` | `+0.000001431` | `+0.000002980` | yes |
| 56 | `-0.000000018` | `-0.000000063` | `-0.000000238` | no |
| 57 | `+0.000000320` | `+0.000000664` | `+0.000001268` | yes |

The median preserving-mass deltas were `0.000000320`, `0.000000664`, and
`0.000001268` for targets 0.25, 0.50, and 1.00. The largest policy KL was
`1.5908163675248943e-7`; the largest absolute informative-entropy movement was
`4.76837158203125e-7`.

The effective coefficient required for target 0.25 varied from `0.00712` to
`0.06320` across seeds. The final-flush batches supplied informative placement
examples but no informative movement or flying examples. This audit therefore
does not establish phase-general behavior.

## Hypothesis

Pre-Adam policy-head gradient normalization is not a reliable controller of
the desired post-update functional change on these production batches. Adam
state, gradient alignment, and batch composition can make the same nominal
ratio produce different signs and very small probability changes.

## Supporting evidence

- The target ratio and production replay checks passed, excluding a simple
  coefficient or update-route implementation error.
- The required coefficient varied by almost one order of magnitude, while the
  post-update preserving-mass response remained near float32 resolution.
- Increasing the target produced a larger median response, but seed 56 moved
  further in the opposite direction at each step.
- No clipping, entropy excursion, KL excursion, or non-finite value explains
  the failed monotonicity gate.

## Counterevidence and limits

- Seed 55 and seed 57 were monotonic, and the median response rose roughly with
  the target. The result does not prove that normalized supervision can never
  work.
- Seed 56's negative deltas are extremely small. They are evidence of absent
  robust monotonicity, not evidence of material policy damage.
- Only one production batch per seed was audited, and those batches did not
  contain informative movement or flying examples.
- This is optimizer-mechanism evidence, not a training, validation, held-out
  strength, promotion, or playing-strength result.

## Next validation experiment

Do not increase the normalized gradient ratio, extend the completed arms, or
adopt the auxiliary in retained training. The next retained baseline should
keep the policy auxiliary off and use only independently justified training
semantics.

If direct Malom policy supervision is revisited later, it must be a separately
frozen experiment. A KL-constrained teacher update or safe-action sampling
mechanism would need independent seeds, explicit phase support, auxiliary-off
controls, fixed data and runtime identities, per-phase and per-opponent
metrics, and a held-out strength protocol. It is not a prerequisite for the
next corrected baseline.
