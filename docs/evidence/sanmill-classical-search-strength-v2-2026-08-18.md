# Sanmill Classical-Search Strength v2 Result

Date: 2026-08-18

## Verdict

The product's classical difficulty 9 and 10 coordinators did not outperform
the learned routes on the frozen 48-start subset.  Their strict score rates
against the pinned 100,000-node Sanmill runtime were 29.69% and 31.77%.

The motivating stopping premise is false: classical search is not above the
55.73% retained-v4 result.  Both classical budgets are directionally below
retained-v4, the Malom-constrained active specialists, and Malom-safe random.
They are statistically indistinguishable from the 31.25% unconstrained active
specialists on this subset.

Increasing the deterministic work ceiling from 13,887,000 to 18,367,000 nodes
improved the point estimate by only 2.08 percentage points.  Its paired 95%
interval is -0.37 to +4.54 points, so this measurement does not establish that
difficulty 10 is stronger than difficulty 9.

This result does not authorize training, checkpoint changes, promotion,
deployment, or release.

## Frozen identities and runtime

- Timing calibration plan: `71dbfcbe3a160211c14e868c0ff4492679cb2ea2b9c175ee3dfbcd0f14beb1f8`
- Timing calibration result: `84a4c363233dfb3eaaddf510fe357f7e4f9ef6cedc160fdc322cb0a1e0f57c4d`
- Formal v2 plan: `0bbe5145b83e29ba48617d1f1ec32c0e35e2921457557ab98aa52acb0974fa39`
- Formal authorization: `a5e3941a5deaa3502fc5d0b81db63b11050b86a8279622e00a11a91677d33460`
- Corrected result identity: `fe77312f303670c1bb8489f423926ac87fe8cbe702e252d366b452484c0bfe9f`
- Result-identity correction: `26c729995a0d058b7477ab38dcd9d71ab462ba9724ce158b34f588899749d5c3`
- Product source: `origin/main` commit `4e4a7241e9d5427100b46dfe34f5ae384ff9f613`
- Sanmill source: `a6623f88959f7453594df274fbe1f128af7ff55e`
- Sanmill binary SHA-256: `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`
- Malom label version: `sector-corrected-v1`

The exact `origin/main` Python search files and a native extension built from
that tree were loaded under an isolated module name.  Hashes bind those files,
the canonical evolved weights, FullgameDB, EndgameSolvedDB, PhaseValueNet, and
GapNet.  The product route's Malom path is stale on this machine, so candidate
search did not use Malom.  Malom was opened separately and read-only only for
the secondary positional-downgrade labels.

The measured coordinator is the canonical balanced AI-vs-AI product route.
Interactive-session opening and trajectory state was not reconstructed for
frozen midgame starts.  That deviation limits direct equivalence with a live
human session.

## Calibration and deterministic proxy

Twelve blind states, four per source phase, mapped the product wall-clock
settings to fixed work ceilings:

| Product setting | Formal node ceiling | Timed positive-node range |
|---|---:|---:|
| Difficulty 9, nominal 30 seconds | 13,887,000 | 42 to 121,251,840 |
| Difficulty 10, nominal 60 seconds | 18,367,000 | 42 to 234,518,528 |

Two fresh single-thread instances chose the same move at each budget on one
blind state per phase.  A warm Rust transposition table changed elapsed work
and completed depth but not the canary move.  Formal play therefore created a
fresh AI per game and retained the Rust table within that game.

This proxy is deterministic but is not an exact replay of product wall-clock
play.  Product timing uses two search threads; formal fixed-node play uses one.
Many product database paths and depth-14 completions return far before 30 or
60 seconds.  Therefore the fixed ceiling represents typical positive work
from the calibration sample, not a literal per-move wall-clock entitlement.
The direction of the remaining strength bias from single-thread determinism
versus two-thread Lazy SMP is unknown.

## Known-answer gate

Before the classical runtime was loaded, the random-safe arm was rerun on all
48 selected starts in both colors.  All 96 games matched the existing record
exactly in move sequence, terminal reason, strict-history digest,
no-progress clock, and repetition clock.

- Observed identity: `45b4ed7c905fa53cc0813dc0a48f751fbb4e0a990c5515a6c07b18c231c2de68`
- Reference identity: `45b4ed7c905fa53cc0813dc0a48f751fbb4e0a990c5515a6c07b18c231c2de68`
- Differing games: 0

This passed the only hard stopping condition and established the same Sanmill,
referee, color, start, `A_pos`, and scoring behavior as the prior measurement.

## Primary results

All scores below are recomputed on the same 48 starts, with both colors per
start.  Intervals are paired at the start level after averaging colors.

| Arm | W / D / L | Score |
|---|---:|---:|
| Retained-v4 full route | existing same-subset record | 55.73% |
| Retained-v4 constrained to `A_pos` | existing same-subset record | 55.73% |
| Active specialists constrained to `A_pos` | existing same-subset record | 51.56% |
| Malom-safe random | existing same-subset record | 43.23% |
| Active specialists, unconstrained | existing same-subset record | 31.25% |
| Classical difficulty 9, 13,887,000 nodes | 11 / 35 / 50 | 29.69% |
| Classical difficulty 10, 18,367,000 nodes | 13 / 35 / 48 | 31.77% |

### Paired contrasts

The frozen precision gate was a maximum 8.5-percentage-point half-width.

| Contrast | Difference | 95% interval | Half-width | Precision | Direction |
|---|---:|---:|---:|---|---|
| D9 minus retained-v4 | -26.04pp | [-35.26, -16.82]pp | 9.22pp | inadequate | lower |
| D10 minus retained-v4 | -23.96pp | [-32.83, -15.09]pp | 8.87pp | inadequate | lower |
| D9 minus specialists + `A_pos` | -21.88pp | [-28.34, -15.41]pp | 6.46pp | adequate | lower |
| D10 minus specialists + `A_pos` | -19.79pp | [-26.31, -13.27]pp | 6.52pp | adequate | lower |
| D9 minus random-safe | -13.54pp | [-21.39, -5.69]pp | 7.85pp | adequate | lower |
| D10 minus random-safe | -11.46pp | [-19.17, -3.74]pp | 7.71pp | adequate | lower |
| D9 minus free specialists | -1.56pp | [-7.10, +3.98]pp | 5.54pp | adequate | inconclusive |
| D10 minus free specialists | +0.52pp | [-5.40, +6.45]pp | 5.92pp | adequate | inconclusive |

The two retained-v4 contrasts missed the predeclared precision gate by 0.72
and 0.37 points.  They must retain the `precision_inadequate` flag.  Their
intervals nevertheless lie wholly and substantially below zero, so they rule
out the specific premise that classical search is higher than retained-v4;
they do not support a high-precision estimate of the exact deficit.

Retained-v4 free and retained-v4 `A_pos` had identical start-level scores on
this subset, so their two classical contrasts are numerically identical.

## Phase and color diagnostics

Each phase contains 16 starts and 32 games per budget.

| Budget and source phase | W / D / L | Score | Self downgrades / turns |
|---|---:|---:|---:|
| D9 placement | 2 / 4 / 26 | 12.50% | 34 / 544, 6.25% |
| D9 movement | 4 / 7 / 21 | 23.44% | 27 / 509, 5.30% |
| D9 flying | 5 / 24 / 3 | 53.13% | 0 / 637, 0% |
| D10 placement | 3 / 4 / 25 | 15.63% | 35 / 594, 5.89% |
| D10 movement | 5 / 7 / 20 | 26.56% | 21 / 442, 4.75% |
| D10 flying | 5 / 24 / 3 | 53.13% | 0 / 637, 0% |

The weakness is concentrated in placement and movement.  Flying results are
identical at both budgets because most candidate turns use product database or
very shallow paths.  This is not evidence that search depth has no value in
all flying positions; it is the behavior of this exact frozen route and pool.

Color scores were 31.25% as White and 28.13% as Black for difficulty 9, and
34.38% as White and 29.17% as Black for difficulty 10.  The primary estimator
averages both colors within each start and is therefore not driven by this
imbalance.

## Positional self-downgrades

Classical difficulty 9 made 61 positional W/D/L downgrades in 1,690 candidate
turns, 3.61%: 45 D-to-L and 16 W-to-D.  Difficulty 10 made 56 in 1,673 turns,
3.35%: 41 D-to-L, 14 W-to-D, and one W-to-L.

These are `A_pos` diagnostics only.  They do not include repetition or the
no-progress clock and are not full-rule `A_allow` labels.  No candidate move
was filtered by Malom in this measurement.

The zero flying-stage downgrade count is consistent with the product
EndgameSolvedDB path.  Placement and movement still lose positional tier at
roughly 5% to 6% of candidate turns, which helps explain why more raw search
does not close the gap to the safety-constrained routes.

## Terminal reasons and actual work

Both budgets produced 35 draws: nine by the fifty-move rule and 26 by
threefold repetition.  There were no safety-cap completions.  Difficulty 9
ended 13 games by fewer-than-three and 48 by no-legal-moves; difficulty 10
ended 16 and 45 respectively.  Strict rules, not Malom, determined every
terminal result.

| Work metric | Difficulty 9 | Difficulty 10 |
|---|---:|---:|
| Candidate searches | 1,690 | 1,673 |
| Product bypasses | 593, 35.09% | 519, 31.02% |
| Full-ceiling searches | 396, 23.43% | 381, 22.77% |
| Median nodes | 54 | 54 |
| 75th-percentile nodes | 10,422,096 | 12,977,033 |
| 90th-percentile nodes | 13,887,000 | 18,367,000 |
| 75th-percentile elapsed | 2.65 s | 3.27 s |
| 90th-percentile elapsed | 3.80 s | 4.99 s |
| 90th-percentile completed depth | 12 | 13 |

The very low median is real: FullgameDB, EndgameSolvedDB, shallow forced paths,
and other product shortcuts answer many turns without consuming the frozen
ceiling.  Placement and movement consume most deep-search work; flying has
median zero nodes and no full-ceiling search in this subset.

## Resource and access audit

The one execution completed 288 of 288 planned games:

- 96 known-answer games;
- 192 classical-search games;
- 4,734.316340300022 active seconds, including calibration;
- 10,520 Sanmill single-step searches;
- 264,270 Malom read-only queries;
- zero training updates;
- zero checkpoint modifications;
- zero database writes.

All FullgameDB, EndgameSolvedDB, ValueNet, GapNet, and other tracked read-only
resource snapshots were byte-, size-, timestamp-, journal-, WAL-, and SHM-
identical before and after execution.  Official selection, confirmation,
final-test, research-confirmation, and source pool `2eb04f54` read counts were
all zero.

The compact per-game manifest contains all 192 classical and 96 reproduction
records.  Full local ledgers are hash chained:

- classical ledger: 192 records, file SHA-256 `b27903f745bda6bcf4200308b96a978e5acb1d1c4f2230737c597e26f9a11701`, tail `f0a091133401557f0149bbb25ce70f8c0e11d999c43bc1e93f9c8ed7b16b946a`;
- reproduction ledger: 96 records, file SHA-256 `ee46f2e1e3569d4c91a89325f50419caf0e5ea62647649764993e8aa9f75b204`, tail `f389c7bc090f86b51f1c29e6448484661da27c34f2bb5dc3e10f2d120266d29f`.

## Result-identity correction

The first result file was written successfully and reported identity
`1bdeadbff571be3f9cdf5a54e0b82c1abd7981d58fc440497b1f8420bc323b14`.
Independent post-run verification correctly rejected that identity after a
JSON round trip.  Two `completed_depths` histograms had integer keys in memory;
JSON converted them to strings, changing canonical sort order after reload.

The original 317,839-byte file is preserved byte for byte at SHA-256
`1f20252abda2d697e7212c5b3a815994b970d6eee66850505c3b0db1fde5e079`.
No result value, game, query, or search was rerun.  A new manifest reseals the
already serialized semantic payload with verified identity `fe77312f...` and
file SHA-256 `964710c134f4a81593b574e2083f96b773dc7a5671695295df6091896372c1ed`.
The only field difference is `result_identity`.

The generator now serializes depth histogram keys as strings, with a
round-trip regression test.  The machine correction record independently
verifies both raw ledger chains and records zero additional measurement work.

## Interpretation boundary

This is a directional internal result for one exact `origin/main` product
snapshot, two deterministic fixed-node proxies, one pinned Sanmill runtime,
and one reused development start pool.  It is not held-out population
evidence, a live product latency test, a claim about human opponents, proof of
retained-v4 product compatibility, or evidence for promotion or deployment.

The safe routes have an information advantage over unconstrained classical
search because they use Malom positional safety.  That advantage is part of
the product decision being measured, not a claim that the underlying policy
or search alone is stronger.  Conversely, the full retained-v4 route is not
the current product `GeneralistAgent` route and must not be copied into it
without the already documented route-parity work.

The practical conclusion is narrow: on this exact comparable subset, shipping
the existing classical difficulty 9/10 coordinator does not provide a stronger
opponent than the measured safe learned routes, and merely increasing its
calibrated work from difficulty 9 to 10 did not yield a resolved gain.
