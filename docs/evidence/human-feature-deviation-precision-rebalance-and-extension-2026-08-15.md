# Human feature deviation: precision, rebalance, and exploration extension

Date: 15 August 2026

## Outcome first

The research confirmation set was not opened.  The v1 confirmation design is
not structurally certified to meet its frozen precision gates.  This is not a
proof that confirmation is mathematically impossible: the decisive
player-level loss variance and D-to-L event support are result variables and
were deliberately not read.

An outcome-blind player rebalance improved the confirmation-side structure,
but not enough to remove that uncertainty.  The selected split has 487
participating confirmation players instead of 290, and its decision-weighted
Kish diagnostic rises from 46.78 to 58.91.  Confirmation remains sealed until
an exploration-only implementation and variance-calibration round passes the
new frozen execution gate.

The 1,024-game exploration extension completed.  It shows that the v1
`creates_double_mill` feature was the wrong operationalization of a common
double-three idea and that `material_balance_after` cannot be separated from
`closes_mill` inside an atomic choice set.  A revised ten-term panel is frozen
in v2.  These are exploratory design findings, not confirmatory evidence.

F0-H0 remains stopped.  This round does not reopen its failed exact-state or
`ring16` estimator and authorizes no E0, F0-H1, T0, reward change, game,
training, promotion, publication, or release.

## Frozen identities

- original feature plan:
  `04177a73ca5b9a1aa8cc8352477f2050759e6a742cee049f1191d3064ae5d662`;
- original player split:
  `fa74650c1afdffeb0d30f334b2b7859538f81b0e502c17a64092bfdcd99a06dd`;
- original 128-game exploration:
  `c489dca91c00569491d2b50a879bd014081e0109e0899afcc2bf2f13d584d7d6`;
- first, rejected cut-only design-round plan:
  `5ab80986109a4a2b7d918a85091701af733b1a6b9167e7f7a1202d49a26ebe6e`;
- rejected cut-only structural result:
  `1e7477f80adb8e60d2bde8a27bf26865c8f35a3590f7390c760ab1538c0d88e7`;
- corrected activity-balanced design-round plan:
  `60e1f0ee63871b76dce17741ebec985660ceb99df3da28edc73dacb48884f347`;
- corrected structural result:
  `56c9a34f72761e061babcd1d6ca959be614ad37d380cb1803cf57997975c4299`;
- selected split:
  `8187ffa06cc73f4e052b7481f06dc3629a23feace63e086c7075c74c17940028`;
- exploration extension:
  `53e010f473a88d4a384b264906a4a8d1826b92fd5f48e4b386b57356ee78c61a`;
- exact recovery record:
  `95b649c2377ec010ae9e9a39a2a54b9b2eef7873ddb8c91b900fd920128b7d83`;
  and
- revised, blocked-before-confirmation plan:
  `5919b9666d66c568898797e3b2089a71a71bc289696291d29c7aec6dd91e0935`.

The machine-readable manifests and split files are the numeric sources.  This
document is interpretation and handoff context.

## 1. Confirmation precision reachability

### Outcome-blind input contract

This step decoded only the F0-D0 manifest and frozen membership documents.
The fields used were session ID, the two source-domain player keys, move
count, official train membership, and old pilot membership.  Per-color
decision counts follow mechanically from move count:

- White decisions: `(move_count + 1) // 2`;
- Black decisions: `move_count // 2`.

It did not open a raw game file and could not see a human action, Malom tier,
tier-loss event, feature value, or recorded/replayed outcome.  The structural
manifest records all those reads as zero.

### Current v1 confirmation structure

The v1 confirmation arm has:

- 551 assigned player keys but only 290 participating keys;
- 2,751 same-arm games and 129,697 decisions;
- player-decision Gini 0.78270;
- decision-weighted Kish effective units 46.7788;
- maximum player share 5.987%; and
- top 1%, 5%, and 10% player shares of 16.19%, 46.45%, and 67.03%.

Assigned players are not independent evaluable clusters when they have no
surviving same-arm game.  Conversely, Kish is a concentration diagnostic and
is not literally the sample size of the v1 equal-player estimand.  Both the
optimistic participating-player count and the conservative Kish proxy are
therefore reported.

For a two-sided 95% interval and 80% planning power, the normal planning
coefficient is

`(z_0.975 + z_0.80) * SD / sqrt(N)`.

At `N=290`, the multiplier on an unknown player-level SD is 0.16451.  A true
0.01-nat improvement can have 80% power with its lower bound above zero only
if that SD is at most 0.06078 nats.  At Kish `N=46.78`, the corresponding
multiplier is 0.40962 and the SD ceiling is 0.02441 nats.

For the frozen D-to-L requirement that the lower bound itself reach two
percentage points, a player-level SD of 0.25 would require a true difference
of 6.11 points at `N=290`, or 12.24 points under the Kish proxy.  At SD 0.50,
the corresponding requirements are 10.23 and 22.48 points.

Log loss is not structurally bounded by the present unfitted implementation,
and D-to-L event counts are prohibited result variables.  Consequently the
v1 thresholds are neither certified reachable nor proven impossible.  The
fail-closed answer to task one is **not certified; do not open confirmation**.

## 2. Player-isolated split rebalancing

### Why the first frozen algorithm was rejected

The first design froze Louvain communities and then minimized cross-player
game weight at fixed nominal player ratios.  That objective was wrong for the
task.  It preferentially placed low-activity peripheral players in the
confirmation arm.  Its nominal 50% confirmation candidate retained only 137
same-arm games, 48 participating players, and Kish 8.99.

That negative result was not overwritten.  Its plan, result, and generated
split remain immutable and are explicitly marked not for exploration or
confirmation.

Before measuring a corrected candidate, v2 froze a different structural
algorithm.  It uses the same 13 Louvain communities, balances player count and
player decision mass inside each community, locks every player observed in
the old 128-game pilot to exploration, performs no cut-only optimization, and
allows the old v1 split to win.  The selection rule is to maximize the smaller
arm's decision-weighted Kish value, then minimize discarded games, then use
the candidate ID.  No result variable is part of the algorithm or rule.

### Corrected candidate measurements

| split | exploration players/games/decisions | confirmation players/games/decisions | Kish E/C | Gini E/C | discarded games |
|---|---:|---:|---:|---:|---:|
| frozen v1 | 1,416 / 19,257 / 905,648 | 290 / 2,751 / 129,697 | 129.75 / 46.78 | .857 / .783 | 14,941 (40.44%) |
| 30% C | 1,425 / 25,927 / 1,243,633 | 226 / 973 / 41,287 | 144.16 / 43.27 | .859 / .716 | 10,049 (27.20%) |
| 40% C | 1,192 / 22,688 / 1,095,173 | 332 / 1,663 / 71,490 | 125.33 / 55.50 | .856 / .743 | 12,598 (34.10%) |
| 50% C | 980 / 20,264 / 958,664 | 487 / 2,543 / 117,802 | 118.64 / 58.91 | .844 / .772 | 14,142 (38.27%) |

“Players” in the table means participating players.  Assigned player counts
for the selected 50% split are exactly 1,108 and 1,108.  Its two player-key
sets are disjoint.  All old pilot players remain in exploration.

The frozen rule selects the 50% split.  It increases confirmation Kish by
25.9% and participating players by 67.9%, while slightly reducing confirmation
same-arm games and decisions.  It is an improvement in independent-player
structure, not a claim of more raw data or sufficient precision.

For the selected split, the optimistic `N=487` planning multiplier is 0.12695
per unit SD, and the 0.01-nat SD ceiling is 0.07877.  The Kish proxy multiplier
is 0.36501, with a ceiling of 0.02740.  Precision remains uncertified because
the missing quantities are model-dependent result variances and event
support.  This is a limitation after rebalancing, not a demonstrated
information-theoretic upper bound for every possible estimator or split.

## 3. Extended exploration

### Frozen budget and execution

The sample retained all 128 old pilot games and added the first 896 eligible
v2 exploration games under a frozen SHA-256 rank.  Its identity is
`5ef1625e39e50367929bd6239f3454939b8e5d5298de971e38f089a2bf242fe1`.

The old pilot implied about 13.27 queries per decision.  The preregistered
linear estimate was 49,120 decisions, 651,704 queries, and 428.81 seconds.
Hard bounds were 75,000 decisions, 1,000,000 queries, and 1,200 seconds.

The completed extension contains:

- 1,024 games and 48,855 decisions from 354 players;
- player-decision Gini 0.63889 and Kish 108.716;
- 632,094 Malom queries in 109.68 seconds;
- 100% positional-label coverage and zero abstentions;
- 43,290 decisions, or 88.609%, with `|A_pos| > 1`; and
- positional chosen-tier losses of 919 W-to-D, 83 W-to-L, and 1,642 D-to-L.

These transition counts describe the exploration sample only.  They are not
confirmation support and cannot be projected into protected membership.

### Double mill versus a double-three setup

The old `creates_double_mill` score still varied on zero of 48,855 complete
choice sets.  The implementation is not algebraically impossible: a focused
fixture proves that placing on the shared square of two prepared lines can
simultaneously close two mills.  It is simply a much narrower event than the
human concept the feature name suggested.

The v2 `creates_mill_fork` proxy asks whether an action leaves at least two
distinct future immediate-mill destinations and creates at least one of them.
It varied in 12,744 choice sets; 12,769 sets contained at least one such legal
alternative.  This is a defensible visible double-three proxy.  It remains an
association feature, not a causal trap label.

### Mill closure and material are not separately identifiable

For every legal action in all 48,855 complete choice sets, the extension
tested

`material_balance_after - closes_mill / 9`.

It was constant within every choice set; the largest floating residual range
was `1.11e-16`.  In this atomic action representation, a capture is available
only after closing a mill, so the two columns differ only by a choice-set
constant and scale.  Conditional choice cannot identify two coefficients.
Ridge regularization would pick a numerical decomposition, not create
scientific separability.  v2 therefore removes `material_balance_after`.

### Feature-panel decision

The v2 panel keeps three geometry controls and uses these seven tactical
terms:

- `closes_mill`;
- `opponent_immediate_mill_destinations_removed`;
- `creates_mill_fork`;
- `new_own_potential_mills`;
- `own_mobility_delta`;
- `opponent_mobility_reduction`; and
- `captured_opponent_threat_lines`.

Their exploratory varying-choice counts are respectively 5,287, 13,450,
12,744, 35,516, 30,732, 30,553, and 4,651.  The three geometry controls also
vary.  This is enough to freeze the dictionary; it is not evidence that a
fitted model will predict unseen players.

The replacement block feature counts opponent immediate-mill destinations
removed, including removal caused by a capture.  Own and opponent mobility
are kept separately.  Destination degree retains the high-connectivity
preference, while capture degree and captured threat lines describe capture
choice.  Flying remains a reporting stratum because phase is constant inside
a choice set.  A last-move locality feature is not admitted: its exact history
contract was not frozen before this exploration, and adding it now would be
post-result feature fishing.

## 4. v2 disposition and why confirmation did not run

The revised plan preserves the v1 substantive effect floors:

- average-unique-player log-loss improvement at least 0.01 nat with its 95%
  lower bound above zero; and
- D-to-L top-versus-bottom risk-quintile difference at least two percentage
  points with its 95% lower bound meeting that floor.

It changes the feature dictionary and player split, and adds an execution
gate.  Before protected confirmation may be opened, a separate
exploration-only round must:

1. implement the frozen conditional-choice estimator and optimizer;
2. freeze numerical probability handling and convergence tests;
3. use player-level cross-fitting on v2 exploration only;
4. show projected 80% power for a 0.01-nat effect at 487 players; and
5. show projected 95% half-width at most two points for the D-to-L contrast
   under the frozen event-support rule.

No statistical model was fitted in this round because model fitting was
explicitly prohibited.  Therefore the player-level variance needed by items
4 and 5 does not exist yet.  The precision gate is not met, even though the
split and v2 preregistration are now frozen.  Since the user's three execution
preconditions are conjunctive, confirmation was correctly not executed.

This answer is more specific than “the corpus is too small.”  Rebalancing
improved the bottleneck, but structure alone cannot decide whether the 0.01
nat target is powered.  If exploration-only calibration estimates a
player-level paired-loss SD above 0.07877 nats at the planned estimator, the
selected confirmation arm cannot reach 80% planning power for that minimum
effect without changing the estimand, threshold, or data.  The frozen rule
forbids relaxing the threshold after that result.

## Failure and recovery record

The first extension process failed before its first Malom query because the
board and database arguments to the existing comparator were reversed.  One
permitted frozen-exploration game had been read and replayed.  No summary
statistic or output manifest existed; research confirmation and every
official holdout remained untouched.

The correction added an explicit, focused-tested board-first wrapper.  A
separate recovery record froze the same sample, features, Malom snapshot, and
budgets before a full restart.  No partial prefix was reused.  The completed
manifest is the only extension result.

## Access and claim boundary

The completed extension records zero reads from research confirmation,
official selection, official confirmation, official final-test, HumanDB, and
source pool `2eb04f54`.  It records zero database writes, games, search
batches, model loads, training, or weight updates.

Every Malom statement is `A_pos` and positional-only.  The tablebase query does
not contain repetition multiplicity or the no-progress clock, so none of this
is an `A_allow` proof.  Findings are limited to the observed PlayOK-like
source domain and cannot be transported to product users, UI orientation,
time controls, exact rule variants, or a new population.

The inherited recovery bias remains material: the excluded 1,751 games have
only 35 draws, whereas the retained 92,789 have 26,157.  Another 54,923 games
lack an independently verifiable terminal basis.  Zero terminal disagreement
in the verifiable subset does not validate those missing outcomes.

## Verification

Task-scope Ruff passes for the new evaluation module, both runners, and all
three feature-deviation test files.  The focused feature-deviation group passes
15 tests.  The required Malom, DB-teacher, and label-provenance group passes
103 tests and 498 parameterized subtests.  Its first invocation was blocked by
the host's inaccessible default pytest temporary root; the identical suite
passed with a fresh repository-local `--basetemp`.  No test was skipped or
weakened.
