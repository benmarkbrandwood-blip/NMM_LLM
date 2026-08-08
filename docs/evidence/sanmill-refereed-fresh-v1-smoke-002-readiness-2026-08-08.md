# Fresh Sanmill-Refereed Smoke 002 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

This is a read-only retry preflight. It does not authorize launch, consume a
run ID, or approve any long-run node ladder or advancement rule.

## Gate result

The reviewed smoke-002 command in the
[experiment contract](../experiments/dev-v4-sanmill-refereed-fresh-v1.md)
ran from clean commit `d7d6e4dbc22f95c79280ef05c93c8eb8e0a03167` and returned no
errors or unresolved decisions.

| Gate | Observed | Required | Result |
| --- | --- | --- | --- |
| Git | clean `dev` at `d7d6e4d` | clean tracked tree | pass |
| Start | fresh; no resume source | random weights | pass |
| Output | proposed smoke-002 directory absent | isolated and absent | pass |
| SpecialistDB | empty, current labels, no lineage | fresh trusted database | pass |
| Sanmill | exact source, tree, binary, profile, and two-process probe | pinned fail-closed runtime | pass |
| Components | PPO, recovery, legacy nets, imitation, and opening forcing off | experiment boundary | pass |
| Work | two games, 1,000 nodes, batch one, advancement disabled | bounded integration only | pass |

The preflight identities are:

| Field | Value |
| --- | --- |
| Config SHA-256 | `a3d7c1cbb20288a860d0fb106f19b0e338392b3219b823ddb7acb687d46217f2` |
| Resume-config SHA-256 | `2f5cf0d27895d5956b687d782b29e2afea673b25282cea29903f1d060d268f82` |
| Experiment digest | `sha256:38915d6904871f434e21534f868ed47afbe07cc6c87efc6bf5b4f894b61bc62d` |
| SpecialistDB content SHA-256 | `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d` |
| SpecialistDB identity | `2d2fcfb4d2dd8ccb8a118dc4900472bbfa027865a1788fc14e8360df889fe2e7` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Sanmill deterministic probe | `0a6c478d75cded748fa397e65831f0cfc0e3c3040248b4741a5704f4f35d03bd` |

The terminal repair verification reports `182 passed, 6 deselected` for the
selected trainer/launch/checkpoint/bridge/referee group and `103 passed, 498
subtests passed` for the mandatory Malom/provenance group. Ruff and
`git diff --check` pass.

Because this evidence file changes the Git commit, the same read-only command
must be rerun once from the final clean evidence commit before the owner is
asked to authorize a launch. A passing rerun still does not itself authorize
smoke-002.
