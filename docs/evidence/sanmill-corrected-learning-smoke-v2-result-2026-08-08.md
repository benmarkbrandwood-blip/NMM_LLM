# Sanmill Corrected Learning Smoke v2 Result — 8 August 2026

## Decision

`passed_bounded_learning_health_not_strength`

At clean published commit
`12ab5cfae322d2107ecdc3b5ce89b90d66dcf510`, the one authorized
500-game `run-next` completed at the first fixed-resource boundary. No second
segment was created or started. The smoke passed its finite-update,
identity-chain, checkpoint-integrity, and fixed-state anti-collapse gates.

This result does not establish playing strength. The fresh learner won only
three games, all against the frozen-model stratum, and lost every Sanmill
search game. Its significance is narrower: after 110 real A2C updates, the
policy had not learned the old lineage's inverse-Malom direction.

## Frozen plan and preflight

The managed plan bound:

- plan ID `managed-sanmill-corrected-learning-smoke-v2`;
- canonical plan SHA-256
  `4ed054696f6d2fe904cdcd9165f4c79900c2215018fbeafba67e4bd48977ae8f`;
- raw plan file SHA-256
  `f1b5d992450b4185c9a49171e74a0d5e40e3fab9052340e524072e0b023d6f5c`;
- resume-config SHA-256
  `0f93576624ed877deb0c2b52dd6cce813d6f26616ce236036c5eed7454a8b580`;
- experiment digest
  `sha256:693f75cc5573ef09f8be887cd6fb12557e16c51616454cfb46aed16c27984479`;
  and
- one 500-game segment on the 5,000-game temperature and node schedule.

The authorization file SHA-256 is
`1c0cc8f6ea0438ca63c877f0676d0fe9e7eb1365fef736519bf68a21a7f590b3`.
Its decision note authorizes exactly one `run-next` and forbids
`run-authorized` or continuation before result review.

The final read-only preflight returned `ready_for_long_run`, `errors=[]`, and
`unresolved_decisions=[]`. It bound the corrected Malom dataset, masked
HumanDB labels, empty trusted SpecialistDB, MIF Suite, MRS ruleset, and the
isolated Sanmill runtime. The preflight configuration SHA-256 was
`8111001f4f059844cd90d127686128171c295ee680a270afea98b1660bbd6cfa`.

## Runtime result

The segment completed games 1 through 500 in 214.235 active seconds. It
stopped with temperature 0.8125 at the first 500-game curriculum boundary and
did not enter the 5,000-node level.

The trainer performed 109 normal periodic updates and one final flush. All
logged policy losses, value losses, entropies, learning rates, rewards,
temperatures, and probabilities were finite. The final checkpoint records 500
games and 110 updates.

| Opponent | Learner colour | Win | Draw/truncation | Loss |
| --- | --- | ---: | ---: | ---: |
| Frozen target | Black | 0 | 4 | 141 |
| Frozen target | White | 3 | 3 | 142 |
| Sanmill 1,000 nodes | Black | 0 | 0 | 102 |
| Sanmill 1,000 nodes | White | 0 | 0 | 105 |
| **Total** | both | **3** | **7** | **490** |

Termination reasons were 241 fewer-than-three losses, 252 no-legal-move
losses, six threefold-repetition draws, and one experiment truncation at 120
logical plies. Sanmill was the authoritative referee for every row.

The near-total loss rate is not accepted as strength evidence and is not
ignored. High-temperature sampling against deterministic frozen argmax and a
Sanmill search opponent can create a large early asymmetry. The result does
not yet distinguish that expected pressure from a longer-horizon optimisation
problem. Segment-boundary policy health, rather than W/D/L, is the current
learning-correctness discriminator; held-out frozen evaluation remains the
later strength discriminator.

## Fixed-state policy health

The committed machine-readable report is
[`artifacts/generalist-policy-health-corrected-smoke-v2.json`](artifacts/generalist-policy-health-corrected-smoke-v2.json).
Its SHA-256 is
`386d520b8ae1b6539f03ebe5bbaf7d652abcc3acae9e2307427d6358af3fcf52`
and its evidence ID is
`b1c3065b6b8c4ac9ff14f3fef9a006ca230618973d0e9c88aa2f4f4786c11231`.

| Metric on the 29 Malom-critical states | Frozen gate | Scratch | Game 500 |
| --- | ---: | ---: | ---: |
| Direct lookahead argmax preserves value | exactly 1.0 | not applicable | 1.0 |
| Policy argmax preserves value | at least 0.50 | 0.690 | 0.966 |
| Preserving minus downgrading logit | at least -0.10 | approximately 0 | +0.0044 |
| Scheduled preserving probability mass | diagnostic | 0.368 | 0.369 |
| Scheduled expected Malom move quality | diagnostic | -0.690 | -0.689 |

For comparison, the old game-5,000 policy scored 0.069 and -0.730 on the two
quarantine metrics. The successor therefore passes the prespecified smoke
gate with substantial margin. This says that the corrected route retained the
available signal at game 500; it does not guarantee that it will remain
healthy through game 5,000.

## Checkpoint and database integrity

`checkpoint_tool.py verify` accepted the final envelope:

- checkpoint ID
  `managed-sanmill-corrected-learning-smoke-v2-segment-0001:checkpoint:00000011`;
- payload SHA-256
  `5f151690e87430e5e0aff4266da1c458ed6c23fa6cafa5ed7af79d810b2f2507`;
- checkpoint file SHA-256
  `9df22b9781bd111c7255575aa1695b1f4c7fdbb8b1992620c6f7620f0b2578bf`;
  and
- recorded SpecialistDB SHA-256
  `276f841ea42700dfa697920ba8ec1bca8bc4b35cad5fd4c31d5a3e2d6788a2e6`.

The closed SpecialistDB has that same file hash, passes its trusted-label
check, and contains 13,920 positions, 4,703 trusted Malom labels, 10 winning
lines, no preferred plays, and the correct segment-0001 lineage root.

## Persisted local artefacts

| Artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| Controller events | 2,237 | `637d54538bab65ada318e396758434793f774ea3d0379c6e86b44837cafe7b6f` |
| Run manifest | 7,915 | `2256edef070fd9fb8ecd3643e3455d0e701ff07fae4f77777e770a35dfd5aff8` |
| Run events | 1,504 | `f468166988656b9a8d3964d0a67aa7d538e36806a3045894366bb93bf473b7e7` |
| Training log | 666,790 | `84f7135d8227d49e131d3789fc72da5ea115c9e29594519e3c35ce53219cfd81` |
| Update log | 18,622 | `cbfd31e3f16b8bf33faf727a1da0924e4366676aff3c77a9f834775d24a10be7` |
| Latest checkpoint | 2,120,211 | `9df22b9781bd111c7255575aa1695b1f4c7fdbb8b1992620c6f7620f0b2578bf` |
| SpecialistDB | 1,785,856 | `276f841ea42700dfa697920ba8ec1bca8bc4b35cad5fd4c31d5a3e2d6788a2e6` |
| Policy-health report | 10,694 | `386d520b8ae1b6539f03ebe5bbaf7d652abcc3acae9e2307427d6358af3fcf52` |

## Verification and next gate

Before launch, 188 focused trainer, manager, checkpoint, exact-resume,
Sanmill, and health-audit tests passed. The mandatory Malom, DB-teacher, and
label-provenance suite passed 103 tests and 498 parameterized subtests.

This smoke's one-segment authority is consumed. The local manager may report
that another segment is technically schedulable, but the experiment contract
overrides that generic state: segment 0002 is not authorized.

Before a retained lineage may start:

1. rerun continuous-versus-segmented exact-resume parity on the current
   corrected source, because pending steps now carry bootstrap perspective;
2. make the managed supervisor run the fixed-state health audit at every
   segment boundary and quarantine on the frozen limits; and
3. publish a new fresh 5,000-game plan and readiness record. The smoke
   checkpoint and SpecialistDB must not seed that retained lineage.
