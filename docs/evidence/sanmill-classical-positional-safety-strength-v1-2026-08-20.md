# Current-Dev Classical `A_pos` Strength Result

Date: 2026-08-20

## Verdict

On the frozen 48-start subset, the delivered final positional-safety gate had
exactly zero incremental effect on current-`dev` classical play.  At both
difficulty 9 and difficulty 10, filtered and unfiltered play scored 66.67%
with 47 wins, 34 draws, and 15 losses in 96 games.  Every paired game also had
the same move sequence, strict terminal reason, final history, and final
positional label.

The result is more specific than "no resolved improvement": the final gate
inspected all 3,739 classical turns, found every original move already inside
`A_pos`, and intervened zero times.  The primary paired difference was exactly
0.00 percentage points at each difficulty, with a finite-sample engineering
interval of [0.00, 0.00] points.  This is exact identity on this frozen pool,
not a population equivalence claim; no equivalence margin was preregistered.

The explanation is a route interaction.  Current `dev` supplies the validated
Malom adapter to `GameAI`.  Difficulty 9 and 10 enable its database path on
every move, and `GameAI.choose_move` returns a complete-value Malom win or draw
before most classical searches.  The final `ProductPositionalSafetyGate` is
therefore defense in depth under this runtime, rather than the component that
created the observed safe play.

Current-`dev` classical play itself scored 66.67%, substantially above the old
`origin/main` v2 scores of 29.69% and 31.77%.  Those old numbers do not transfer
to `dev`: only 26 of 192 unfiltered records matched semantically, while 166
differed.  This difference is important, but it cannot be attributed solely
to the Malom resolver or any one file because the two source trees differ by
hundreds of commits.

The result does not authorize training, model changes, promotion, deployment,
or release.

## Frozen identities

- Plan: `87aebf86fbaaede73560fc4706dcf54778f2d9bb80ecbd8c89be1ed9060882e5`
- Authorization: `bbc4be1594f238fed030d1929bb5e85db708feca57404b13e768a6005bab6a6e`
- Result: `0e8112776c9fe626c2a84a4710fa9fc09a27cf02789339bfc56fe4ae15665d61`
- Result-file SHA-256: `bfb904a0b3dcae4cf75beea251519f5bce1e23d4dec2dd2273d485d20ca36e81`
- Product source: `776aa48095225149143abff8ea6a6965486f5229`
- Start pool: `385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9`
- Formal 48-start membership: `973a5205411da8448daca1c6ca3df8176ef6ac56fc2cc0b8c20be27787c9c727`
- Sanmill runtime: `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`
- Malom content: `c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544`
- Malom label version: `sector-corrected-v1`

The result manifest round-trips under its canonical seal.  Independent
post-run recomputation also verified both local raw-ledger hashes and the
candidate append chain.

## Known-answer gate

Before any candidate arm, the frozen random-safe arm was rerun on all 48
starts in both colors.  All 96 games matched the existing records exactly in
move sequence, terminal reason, strict-history digest, repetition state, and
no-progress clock.

- Expected identity: `45b4ed7c905fa53cc0813dc0a48f751fbb4e0a990c5515a6c07b18c231c2de68`
- Observed identity: `45b4ed7c905fa53cc0813dc0a48f751fbb4e0a990c5515a6c07b18c231c2de68`
- Differing games: 0
- W / D / L: 3 / 77 / 16, score 43.23%
- Terminal reasons: 28 threefold, 49 fifty-move, 12 no-legal-moves,
  7 fewer-than-three

This passed the end-to-end gate for the start reconstruction, color schedule,
strict referee, `A_pos` queries, pinned Sanmill runtime, and scoring.

## Product route actually measured

Both current-`dev` arms instantiated the same `GameAI`, read-only resources,
internal Malom adapter, start/color schedule, node ceiling, and referee.  The
only primary-arm difference was whether the actual delivered final choke,
`web.app._finalize_product_ai_move`, invoked
`ProductPositionalSafetyGate.constrain`.

When that final gate must replace an unsafe classical move, the delivered
rule is deterministic fixed-depth-2 re-search over the Malom-proved `A_pos`,
followed by canonical `(from, to, capture)` tie breaking.  It is not a root
restriction on the original deep search.  No replacement re-search was
needed in this execution.

The disabled-gate canary returned the original move byte for byte.  Thus the
zero-intervention result was not produced by a disabled gate altering the
unfiltered path.

There is an important product-route boundary.  Human-versus-AI difficulty 9
and 10 still prefers the specialist override when specialist assets load.
This experiment measures the delivered classical fallback and AI-versus-AI
classical path.  It must not be described as the sole human-facing default.

## Primary results

The independent unit is a start after averaging the two colors.  The frozen
precision target was a maximum 4.5-percentage-point half-width.  It was based
on the prior same-start specialist filter variance scaled by the observed
classic-versus-specialist intervention-rate ratio, giving projected half
widths of 3.92 points for difficulty 9 and 3.78 points for difficulty 10.

| Difficulty and arm | W / D / L | Score | Final-gate interventions |
|---|---:|---:|---:|
| D9 unfiltered | 47 / 34 / 15 | 66.67% | not invoked |
| D9 `A_pos` | 47 / 34 / 15 | 66.67% | 0 / 1,869 |
| D10 unfiltered | 47 / 34 / 15 | 66.67% | not invoked |
| D10 `A_pos` | 47 / 34 / 15 | 66.67% | 0 / 1,870 |

| Primary contrast | Difference | 95% interval | Half-width | Precision | Frozen decision |
|---|---:|---:|---:|---|---|
| D9 filtered minus unfiltered | 0.00pp | [0.00, 0.00]pp | 0.00pp | adequate | direction inconclusive |
| D10 filtered minus unfiltered | 0.00pp | [0.00, 0.00]pp | 0.00pp | adequate | direction inconclusive |

All 48 start-level differences were zero at each difficulty.  The frozen rule
uses `direction_inconclusive` whenever the interval does not lie strictly on
one side of zero.  That label is mechanically correct even though the finite
pool result is exactly zero.

## Why the final gate did not intervene

The measurement observed no original positional downgrade in any of the
7,478 current-`dev` classical candidate turns across both filtered and
unfiltered arms.  In the filtered arms, every one of the 3,739 gate decisions
had selection rule `original-already-in-A_pos`; runtime failures, selection
failures, unavailable requests, and restricted-root re-searches were all zero.

This differs from `origin/main` v2, which observed 61 downgrades in 1,690 D9
turns and 56 in 1,673 D10 turns.  The confirmed configuration difference is
that current `dev` passes the validated Malom adapter into `GameAI`, while the
old main product route did not have a usable Malom path.  At difficulty 9 and
10, `GameAI._db_access_prob` is 1.0.  Its Malom fast path compares complete
candidate values and directly returns a winning or drawing move when one is
available.

That path bypassed classical search in 1,772 of 1,869 D9 turns (94.81%) and
1,769 of 1,870 D10 turns (94.60%).  The final gate then independently verified
the same positional tier.  This double checking explains both the zero
interventions and the nonzero but small gate latency.

The result does not prove the final gate can never matter.  It remains a
fail-visible defense for another move source, a changed internal route, or an
internal route that does not return a complete-value Malom result.  It does
prove that it added no move or score change under this exact validated runtime
and pool.

## Main-versus-dev unfiltered comparison

The 192 current-`dev` unfiltered records were compared to the frozen v2 main
records before the filtered arms ran.

- Exact semantic matches: 26 / 192
- Different records: 166 / 192
- Move-sequence differences: 156
- Search-work or label-only differences: 10
- Old main D9 score: 29.69%; current dev D9: 66.67%, +36.98pp
- Old main D10 score: 31.77%; current dev D10: 66.67%, +34.90pp

The disabled-gate canary excluded the new final filter as the cause of the
unfiltered-path difference.  The same-`dev` filtered-minus-unfiltered contrast
therefore remained valid and execution continued as preregistered.

The result is nevertheless confounded for branch comparison.  The new active
internal Malom route is a concrete, behaviorally relevant difference, but the
source trees contain many other changes.  The +36.98 and +34.90 point changes
are not causal estimates for the resolver, the final filter, or any single
commit.

## Phase and terminal diagnostics

All four current-`dev` arms produced the same phase and terminal summaries.
Each source phase contains 16 starts and 32 games per arm.

| Source phase | W / D / L | Score | Candidate turns per filtered arm | Interventions |
|---|---:|---:|---:|---:|
| Placement | 18 / 9 / 5 | 70.31% | 145 | 0 |
| Movement | 10 / 15 / 7 | 54.69% | 1,284 D9 / 1,285 D10 | 0 |
| Flying | 19 / 10 / 3 | 75.00% | 440 | 0 |

Every arm ended 16 games by the fifty-move rule, 18 by threefold repetition,
32 by fewer-than-three, and 30 by no-legal-moves.  There were no safety-cap
completions.  Strict rules, not Malom, determined every terminal result.

`A_pos` is position-only W/D/L preservation.  It does not carry repetition or
no-progress history and must not be called `A_allow` or full-rule safety.

## Work and latency

The fixed work ceilings remained 13,887,000 nodes for D9 and 18,367,000 for
D10.  Each game used a new AI, retained its Rust transposition table within
the game, and used the pinned single-thread deterministic settings.

| Metric | D9 `A_pos` | D10 `A_pos` |
|---|---:|---:|
| Candidate turns | 1,869 | 1,870 |
| Internal DB/search bypasses | 1,772, 94.81% | 1,769, 94.60% |
| Positive-node searches | 97 | 101 |
| Full-ceiling searches | 9 | 8 |
| Median nodes | 0 | 0 |
| 95th-percentile nodes | 2 | 2 |
| 99th-percentile nodes | 2,011,190 | 2,328,435 |
| Maximum nodes | 13,887,000 | 18,367,000 |

The final gate's direct per-turn elapsed time, including its read-only Malom
queries, was small but not zero:

| Gate latency | D9 | D10 |
|---|---:|---:|
| Mean | 0.520 ms | 0.508 ms |
| Median | 0.261 ms | 0.258 ms |
| 95th percentile | 1.504 ms | 1.482 ms |
| 99th percentile | 4.156 ms | 3.901 ms |
| Maximum | 7.365 ms | 7.916 ms |
| Sum across all measured candidate turns | 971.3 ms | 949.9 ms |

Whole-game filtered-minus-unfiltered elapsed differences fluctuated in both
directions despite identical moves.  They are runtime noise, not evidence of
a speedup.  The direct gate timing above is the relevant incremental latency
observation.

## Same-subset context

Existing records were recomputed on the same 48 starts.  These comparisons
are descriptive engineering context, not the primary contrast.

| Arm | Score | Current-dev classical difference |
|---|---:|---:|
| Current-dev D9/D10 classical, free or `A_pos` | 66.67% | reference |
| Retained-v4 full route | 55.73% | +10.94pp, interval [+4.76, +17.11]pp |
| Retained-v4 constrained to `A_pos` | 55.73% | +10.94pp, interval [+4.76, +17.11]pp |
| Active specialists constrained to `A_pos` | 51.56% | +15.10pp, interval [+10.32, +19.89]pp |
| Malom-safe random | 43.23% | +23.44pp, interval [+18.51, +28.36]pp |
| Active specialists, unconstrained | 31.25% | +35.42pp, interval [+26.44, +44.39]pp |

These are exact reused-start comparisons under the recorded engineering
interval, not held-out population evidence.  They do show that the earlier
premise "the learned routes may be stronger than classical" does not hold for
this current-`dev`, internally Malom-enabled classical runtime on this pool.
They do not establish how a human-facing product population would rank the
routes.

## Resource, integrity, and access audit

The one authorized execution completed all 480 planned games:

- 96 known-answer games;
- 192 current-`dev` unfiltered games;
- 192 current-`dev` filtered games;
- 929.8309242000105 active seconds;
- 18,674 Sanmill single-step searches;
- 372,321 Malom read-only queries;
- zero training updates, checkpoint modifications, or database writes.

All limits were respected.  Read-only FullgameDB, EndgameSolvedDB, ValueNet,
GapNet, and Malom snapshots remained unchanged, with no journal, WAL, or SHM
side effects.  Official selection, confirmation, final-test,
research-confirmation, and source-pool `2eb04f54` reads were all zero.

The tracked result manifest contains compact records for all 384 candidate
games and 96 reproduction records.  Full local records are retained in the
ignored execution namespace:

- candidate ledger: 384 hash-chained records, file SHA-256
  `15458a1b2014c1d5121b8a1979b9d37fc6343651c037e471d6404691a2726451`,
  tail `08bce686e35defc5fd9514218d0df9f1302dc554459ef5b2c5e3ef0b7dbac34f`;
- reproduction ledger: 96 individually sealed preserved records, file
  SHA-256
  `1b3c1bfa3bcc99aedb700597e94acc82375ffe104707eb5f973c70510a5ca042`.

Independent post-run recomputation verified the result seal, both raw-file
hashes, all 384 candidate record links, all 96 reproduction wrappers, arm
W/D/L totals, terminal counts, zero downgrade/intervention totals, and exact
filtered/unfiltered trajectories.

## Verification

- 31 focused classical-search, product-safety, frozen-contract, and result
  tests passed.
- 103 Malom, DB-teacher, and label-provenance tests passed, including 498
  generated subtests.
- Ruff passed for every task Python file.

The first Malom provenance invocation pointed `--basetemp` below a missing
local parent and produced 16 fixture-setup errors after 87 tests had passed.
No product or measurement assertion failed.  After creating that local parent,
the identical test group passed 103/103.  The pytest cache warning about the
host's inaccessible default `.pytest_cache` remains non-semantic because the
explicit local basetemp was used successfully.

## Interpretation boundary

This is one internal measurement for exact current-`dev` source, one local
validated `sector-corrected-v1` Malom snapshot, one pinned Sanmill runtime,
two deterministic node ceilings, and a reused 48-start development pool.  It
is not held-out population evidence, a human-opponent result, a live product
latency study, a general equivalence claim, or evidence for training,
promotion, deployment, or release.

The narrow product conclusion is:

1. The final `A_pos` choke added no move or score change in this exact runtime.
2. Current-`dev` classical play was already positionally safe because its
   internal Malom route dominated move selection.
3. The old `origin/main` classical scores cannot be used for current `dev`.
4. On this reused pool, current-`dev` classical scored above every previously
   measured learned route, but that ranking is not a user-population claim.

Machine result:
[`sanmill-classical-positional-safety-strength-v1-manifest-2026-08-20.json`](sanmill-classical-positional-safety-strength-v1-manifest-2026-08-20.json).

Frozen plan:
[`sanmill-classical-positional-safety-strength-v1.json`](../experiments/sanmill-classical-positional-safety-strength-v1.json).
