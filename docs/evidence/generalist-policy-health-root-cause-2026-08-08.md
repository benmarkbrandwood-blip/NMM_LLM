# Generalist Policy-Health Root-Cause Evidence — 8 August 2026

## Claim boundary

This record explains why the completed
`managed-sanmill-v4-fresh-v1` lineage must not be resumed. It is fixed-state,
read-only diagnostic evidence, not playing-strength, promotion, or formal
evaluation evidence. The 64-position phase corpus remains a reviewed draft
coverage corpus rather than a frozen strength suite.

The old run completed all 5,000 games without an infrastructure failure, but
finished with 136 wins, 49 draws, and 4,815 losses. Its three Sanmill draws
and 2,010 Sanmill losses showed that the retained policy was not usable. The
aggregate outcome alone did not identify the learning defect, so this audit
tests the policy direction on fixed positions instead.

## Reproduced fixed-state evidence

The machine-readable report is
[`artifacts/generalist-policy-health-old-final-v1.json`](artifacts/generalist-policy-health-old-final-v1.json).
It was generated at clean published NMM_LLM commit
`3a4fdd1b10a559287f6901baf63d5772cf231a22` with:

- report SHA-256
  `4d0feb98b0e117f235ca95db69803882fb6c2139d713141d75280a5feb700e6e`;
- evidence ID
  `1b8cf2b6d85dfa21c05fb7d098fe4249358a21518439c08a3e8602a517f184dc`;
- corpus SHA-256
  `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e`;
- game-5,000 checkpoint and its recorded final frozen target;
- the checkpoint-bound HumanDB, SpecialistDB, and corrected Malom identities;
  and
- the production 134-feature route with simulation depth 5.

The report reconstructs the seed-42 scratch network and evaluates both it and
the game-5,000 policy on identical feature matrices. This separates weight
learning from position, target-network, and data-route differences.

| Metric on 29 Malom-critical positions | Direct lookahead | Scratch policy | Game-5,000 policy |
| --- | ---: | ---: | ---: |
| Argmax preserves current Malom value | 100.0% | 69.0% | 6.9% |
| Best-preserving minus best-downgrading signal/logit | +0.500 | approximately 0 | -0.730 |
| Scheduled preserving probability mass | not applicable | 36.8% | 11.4% |
| Scheduled expected Malom move quality | not applicable | -0.691 | -1.080 |

The direct signal therefore retained the correct direction while the trained
policy learned a strong inverse preference. This reproduces the earlier local
diagnosis with a smaller, committed, identity-bound report.

## Corrected defects

Two independently tested implementation defects can drive or amplify this
failure mode:

1. Commit `4b0420755428d73581108f6e93cd95407b1b72dc` corrects A2C and
   PPO bootstrapping after a complete logical action. The successor board is
   encoded for the opponent to move, so its value must be negated before it is
   added to the current player's return. The prior code added it with the same
   sign.
2. Commit `0fbc9510400c88a493b6e2efdcf7c9e92ae8b150` gives frozen-model
   opponents the same lookahead and SpecialistDB feature construction used by
   the learner. Selection remains deterministic argmax. The prior frozen
   opponent used zero-padded lookahead and no SpecialistDB, so self-play mixed
   two different input semantics under one set of weights.

These defects are established by deterministic code-level reproductions and
focused regression tests. The fixed-state report establishes the old policy's
failure signature; it does not by itself allocate a percentage of causal
responsibility to either defect.

## Data warning interpretation

The audit prints that HumanDB Malom labels are disabled because the imported
database lacks `sector-corrected-v1` Malom metadata. This is expected and
correct: historical HumanDB Malom columns remain masked, while human move
frequencies and empirical outcomes remain available. The report independently
verifies the HumanDB structural identity recorded by the checkpoint.

## Retraining consequence

The old checkpoint and its mutable SpecialistDB are historical evidence only.
They must not be loaded into the successor smoke or retained run. A successor
must start from seed-42 random weights with a new empty, trusted
SpecialistDB. Sanmill remains authoritative referee and search opponent.

Passing a short successor smoke cannot prove strength or guarantee that a
late-training collapse is impossible. It must demonstrate finite updates and
must not already reproduce the inverse-Malom signature. A retained run must
repeat this fixed-state audit at segment boundaries and quarantine the lineage
if the direction crosses the frozen stop boundary.
