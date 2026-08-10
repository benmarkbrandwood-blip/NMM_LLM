# SpecialistDB Placement-Coverage Policy Audit v2 Result

## Outcome

Status: `material_on_placement_coverage_corpus`

Report:
`specialist-db-policy-mechanism-placement-coverage-v2-policy-audit-2026-08-10.json`

Report SHA-256:
`b7b2894fd91c590f4ad4858d55acb37af7a9ca39e1e09083d897e04f1684e759`

Evidence ID:
`62ceccc88a401cbe46c7c473c199a4b598aa12c0f6ed2633b7ecef206e8d519a`

Execution commit:
`979cc8280ff9056bf3eadc730fbf8cb8fe855d7b`

## Observed facts

- The audit evaluated 100 coverage-positive placement states and 2,047 legal
  successors through the retained-v3 production-aligned CPU encoder.
- There were 688 empirical projections, 196 theoretical labels and 117
  theoretical-versus-empirical modal disagreements across legal successors.
- `full` had 694 usable action projections in all 100 states.
  `empirical_disabled` had 196 theoretical projections in 77 states.
- Removing empirical evidence changed the policy argmax at indices 36, 37 and
  38. This exactly met the frozen three-change material threshold.
- There was no critical preserving-versus-downgrading crossing. Both modes
  chose a corrected-Malom-preserving action in all 100 states.
- Mean scheduled-temperature total variation was 0.01743, below the frozen
  0.05 threshold. The maximum was 0.11377.
- At temperature 1, mean and maximum total variation were 0.01089 and 0.02686.
- The three changes occurred in very early placement states after one piece
  per side. In all three, `full` and `malom_disabled` agreed, while
  `empirical_disabled` and `all_disabled` selected the same alternative. This
  localizes the observed argmax effect to empirical evidence rather than the
  theoretical fallback.
- The checkpoint and SpecialistDB SHA-256 values were unchanged before and
  after execution, and the audit snapshot remained sidecar-free.

## Hypothesis

Empirical cumulative SpecialistDB features can materially alter the final
policy's early placement preferences, especially at the final scheduled
temperature of 0.20. The mechanism result does not establish whether that
influence improves or harms learning or playing strength.

## Supporting evidence

The three changed states had broad empirical coverage and no theoretical
projection in the two selected actions. Removing empirical reads therefore
removed the discriminating database features. The effect survived as an
argmax change and reached total-variation distances of about 0.086 to 0.111 in
those states.

## Counterevidence and claim limits

- The result is exactly at one trigger boundary rather than well beyond it.
- Average distribution movement remained below its independent material
  threshold.
- No selected action lost corrected-Malom value.
- All tested states were placement states drawn from inspected development
  histories; movement and flying remain unmeasured.
- This no-update audit cannot estimate causal training effects, seed
  variability, performance against Sanmill or held-out strength.

## Next validation experiment

Prepare a paired three-seed, single-factor training calibration comparing the
current `full` SpecialistDB projection with `empirical_disabled` reads. Pair
seeds, opponent schedule, initialization, data identities, temperature,
updates, resource limits and evaluation. Give every arm its own fresh isolated
SpecialistDB. Do not start the calibration until the trainer switch, focused
tests, immutable plan, database identities and final readiness audit are
complete and a separate launch authorization is recorded.
