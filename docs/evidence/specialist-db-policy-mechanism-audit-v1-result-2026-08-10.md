# SpecialistDB Policy-Mechanism Audit v1 Result

## Outcome

Evidence status: `coverage_negative_uninformative`

Machine decision under the preregistered numerical rule:
`not_material_on_fixed_corpus`

Report:
`specialist-db-policy-mechanism-audit-v1-2026-08-10.json`

Report SHA-256:
`ac21af257053c8d44cc5caf9b1f6218c9dea057d5cbffe40fd8ff74c65b195b8`

Evidence ID:
`74943b6e65596797fdc368f6d1a47fad0da495c6ff8cefad99ffbc58dedd69c0`

The audit was read-only. The checkpoint and SpecialistDB SHA-256 values were
unchanged before and after execution, and no SQLite WAL, SHM or rollback
journal sidecar was present before or after execution.

## Observed facts

- The fixed development corpus contained 64 positions: 22 placement, 21
  movement and 21 flying positions.
- The audit examined 1,583 complete legal-turn successors: 384 placement, 223
  movement and 976 flying successors.
- None of those successors supplied a usable theoretical or empirical
  SpecialistDB projection in any mode.
- Consequently, all four feature projections were byte-identical for every
  position. There were zero argmax changes, zero Malom-preservation crossings
  and zero total-variation distance.
- The final checkpoint selected a Malom-preserving argmax for all 64
  positions in every projection.

## Hypotheses

The primary remaining hypothesis is that the phase-covered development corpus
does not overlap the cumulative SpecialistDB sufficiently to expose the
mechanism. This is distinct from the hypothesis that the SpecialistDB has no
effect on policy features or choices.

## Supporting evidence

The mechanism can only change an encoded action when the post-move board has a
usable SpecialistDB projection. The measured usable-hit count was exactly
zero, so identical features and policy outputs are the expected consequence
regardless of model sensitivity.

## Counterevidence and claim limits

- The report does not show that empirical SpecialistDB evidence is harmless,
  redundant or causally irrelevant during training.
- It does not estimate the effect of historical database writes on learned
  weights.
- It does not support a strength, promotion or held-out-generalization claim.
- The numerical rule did not trigger, so this result alone does not authorize
  the conditional three-seed calibration.

## Next validation experiment

Build a candidate-blind diagnostic corpus from already frozen complete
Book/HumanDB/PerfectDB histories. Replay and deduplicate their intermediate
states, then retain every state with at least one empirical SpecialistDB
successor at the frozen three-sample threshold. Freeze that source-only corpus
and its coverage evidence before loading the final checkpoint. Re-run the
same four projections on the resulting coverage-positive corpus.

Because the available complete histories cover placement only, the follow-up
can diagnose the placement-phase mechanism but must not be presented as a
movement- or flying-phase conclusion.
