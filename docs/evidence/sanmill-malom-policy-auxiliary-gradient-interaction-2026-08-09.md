# Malom policy-auxiliary gradient interaction audit — 9 August 2026

## Outcome

The post-calibration audit confirms that raw auxiliary-loss magnitude is not a
reliable normalization variable.  The coefficient must not be increased or
selected from the completed four-arm run.  No normalization rule is selected
yet because only two production batches were recoverable.

The accepted ignored report is
`out/diagnostics/malom-policy-auxiliary-gradient-interaction-v2.json`:

- audit identity:
  `9d338f3b0e3c2a8d4a94a480a814ebc1137c5d3ffe60f1141865ceedd1b715e0`;
- file SHA-256:
  `2576a9323170a5f590e2751ec58f1ef391fc0658cc8be3fbc77bbef748ff98be`;
- file size: 14,430 bytes;
- audit implementation commit:
  `eb665b84b59c74a4e50947a34f06dfa1fb195368`;
- v2 plan SHA-256:
  `72342ff9014efc6778e66277b0777fe3adaee07054d365a1b76eadafddbb29f3`;
  and
- source calibration result identity:
  `d11384f661192db84662a6e43e85cdd6eb299672724178a83a02858b0b12113f`.

The audit ran no games and made no persistent optimizer or model update.  It
loaded the medium and high arms' periodic game-100 checkpoints, which preserve
the exact pending steps immediately before each final flush.  It replayed the
production A2C update only on disposable model and Adam copies and compared the
result with the logged update and final checkpoint.

## Observed facts

| Metric | Coefficient 0.10 | Coefficient 0.30 |
| --- | ---: | ---: |
| Persisted batch steps | 60 | 58 |
| Informative exact-WDL steps | 12 | 7 |
| Informative phase | placement only | placement only |
| Actual Adam learning rate | 0.00005 | 0.00005 |
| Ordinary policy-plus-entropy gradient L2 | 0.0071171 | 0.0008193 |
| Applied auxiliary gradient L2 | 0.0049218 | 0.0218464 |
| Auxiliary / ordinary policy-head ratio | 0.6915 | 26.6658 |
| Auxiliary / ordinary policy-head cosine | +0.9972 | -0.1284 |
| Full pre-clip gradient L2 | 0.6286 | 0.6344 |
| Global clip scale | 1.0 | 1.0 |
| Treatment-minus-baseline Adam delta L2 | 0.0005406 | 0.0009689 |
| One-step informative preserving-mass delta | +0.000000779 | +0.000003237 |

The value-head applied gradient was about 0.63 in both batches and dominated
the full-model norm.  This does not suppress the policy auxiliary: the policy
and value networks have disjoint parameters, their gradient dot products are
zero, and the joint norm remained below the clipping threshold.  The relevant
comparison is therefore within the policy head, not against the full-model
gradient.

The same nominal coefficient has no stable relationship to the ordinary
policy gradient.  At 0.10, the auxiliary was 69% as large as the ordinary
policy-head gradient and almost perfectly aligned with it.  At 0.30, the
ordinary policy gradient happened to be small, making the auxiliary 26.7 times
larger and slightly conflicting.  Raw loss ratios did not reveal this: their
corresponding calibration values were only 0.517 and 1.625.

Neither update was clipped.  The disposable treatment step increased
preserving mass relative to the same Adam state with coefficient zero, but by
less than four millionths on its own informative batch.  This is consistent
with the small fixed-state movement observed over all 26 calibration updates.

CPU replay matched the GPU log within `6.68e-6` for all scalar losses.  Every
functionally relevant final parameter matched the persisted GPU checkpoint
within `4.66e-10`.  The raw parameter comparison contains a `3.78e-6` to
`4.45e-6` difference in the final shared scalar policy bias.  That bias adds
the same constant to every legal action logit and cancels exactly under
softmax.  The report retains this raw difference but excludes only that proven
invariant from functional replay acceptance.

## Hypothesis

Per-batch gradient normalization against the ordinary **policy-head** gradient
can bound the auxiliary's optimizer influence more meaningfully than a fixed
loss coefficient.  A candidate scale would be proportional to
`||g_policy+entropy|| / ||g_aux||`, with explicit handling for no-informative,
zero-gradient and non-finite batches.

This is a mechanism hypothesis, not a selected algorithm.  The observed ratio
varied by a factor of about 39 between only two terminal batches.  A target
ratio, cap and low-gradient policy cannot be frozen responsibly from that
sample alone.

## Supporting evidence

- Both real batches reconcile with the production final update, so the
  component gradients describe the completed experiment rather than a
  synthetic approximation.
- The policy-head ratio differs radically from the raw loss ratio, directly
  falsifying loss magnitude as a sufficient normalization proxy.
- The 0.30 batch demonstrates the required failure mode: a fixed coefficient
  can dominate a temporarily small ordinary policy gradient.
- The 0.10 batch demonstrates that the auxiliary can align with the A2C update,
  so always treating disagreement as an error would also be unjustified.
- No global clipping occurred and the value head is disjoint, excluding two
  alternative explanations for the small fixed-state effect.

## Counterevidence and limits

- Only two final-flush batches from one seed were recoverable.  The low arm had
  no pending final batch and the zero arm did not collect preserving-set masks.
- Both informative subsets contain only placement states.  Nothing here
  calibrates movement or flying gradients.
- These batches come from different learned policies and trajectories.  Their
  ratio difference combines model state, action support and return variation.
- The audit is post-hoc mechanism evidence.  It has no train/validation curve,
  held-out baseline, multi-seed result or strength interpretation.
- A policy-gradient conflict is not automatically harmful: the exact-WDL
  objective is intended to oppose sampled downgrades in some states.  No
  projection or conflict-removal rule is justified by these two batches.
- The inherited learning-rate adaptation placed both batches at half the base
  rate.  This was common to all arms but limits extrapolation to other stages.

## Next validation experiment

Before implementing a new auxiliary normalization in retained training, add a
bounded no-update batch-capture probe.  Across several fresh seeds it should:

1. collect complete production trajectories and retroactive rewards without
   calling `optimizer.step`;
2. preserve exact Malom masks and phase labels for every learner step;
3. form the same 64-step batches used by A2C;
4. report policy, entropy, value and auxiliary gradient norms and cosines for
   every batch, including batches with no informative states;
5. compare candidate policy-head target ratios without changing model weights;
6. report phase, opponent source, learner colour, termination and reward
   support; and
7. bind seeds, schedules, data, rules, Sanmill, MIF and source identities.

Only that distribution can justify a deterministic target ratio, maximum
scale and low-gradient behavior.  The resulting rule must be checkpointed or
stateless, exact-resume compatible, and fail closed on invalid gradients.  A
separate bounded learning calibration would still be required afterward.  No
training launch, continuation, coefficient selection or long-run inclusion is
authorized by this audit.

