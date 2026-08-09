# Retained-v2 mill-bonus no-update probe — 9 August 2026

## Outcome

The clean, committed probe replayed all 19 frozen first-WDL-downgrade turns
through the live `sector-corrected-v1` Malom route. Every complete logical turn
was legal, every successor FEN matched the held-out ledger, and every Malom
quality matched the committed downgrade.

The only compared output was the immediate mill-shaping component:

| Finding | Result |
| --- | ---: |
| Frozen downgrade states | 19 |
| Mill-forming downgrade states | 16 |
| Legacy unconditional reward total | +4.0 |
| Malom-preserving-only reward total | 0.0 |
| Disabled reward total | 0.0 |
| Candidate models loaded | 0 |
| Optimizers or weight updates | 0 |
| New games | 0 |

The mode did not select an action, apply a different action or construct a
different successor. It changed only the reward component. Six placement,
four movement and nine flying states were covered; the source strata were six
Book, two HumanDB and eleven PerfectDB states.

## Interpretation

This closes the route-level question behind the proposed correction. Under
the historical implementation, the 16 premature mill-closing turns receive
`16 × 0.25 = +4.0` immediate shaping despite losing corrected WDL. Under
`malom-preserving-only`, those contradictory bonuses are suppressed.

This does not mean the corrected mode is equivalent to disabling mill shaping.
The frozen cohort contains only downgrades. Focused tests separately establish
that a value-preserving mill retains the same `+0.25` bonus in the corrected
mode, whereas `disabled` always returns zero.

## Claim boundary

The result is deterministic reward-route integration evidence, not evidence
that a new policy will learn better. The 19 states are outcome-selected from
one retained candidate and seed. No causal effect, strength improvement,
promotion, conversion behavior or draw reduction is claimed.

The falsifiable next experiment is a matched learning smoke. Legacy and
corrected arms must share the same seed, data versions, fresh initialization,
Sanmill referee/opponent, node curriculum, hyperparameters, update count,
resource envelope and held-out evaluator. The only learning-semantic
difference may be `--mill-bonus-mode`.

## Immutable identities

| Artefact | Identity / SHA-256 |
| --- | --- |
| Probe identity | `8f554f113ca65f05b8733f7e28b1e26177f58283c10b1c6f7d97abd603ef2186` |
| Probe file SHA-256 | `0560c3fe3b89f32e4a9f59778c214167496e404be10ba24b03622fdc5a618f37` |
| Probe implementation commit | `3292107eb5c16faa95623b7bfc07087607b65117` |
| WDL audit | `6bbb4a50aa7999d06679c802cfeb5b913f0f5abf0689aa0291ec55459304b504` |
| Full-oracle audit | `7cfa9ede873ae4fb34d7821472c62bba540f1b509476073062d52b487995cf65` |
| Corrected Malom | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |

The machine-readable evidence is
[`sanmill-corrected-retained-v2-mill-bonus-probe-2026-08-09.json`](sanmill-corrected-retained-v2-mill-bonus-probe-2026-08-09.json).

