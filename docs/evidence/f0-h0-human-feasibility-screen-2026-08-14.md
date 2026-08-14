# F0-H0 read-only rejection screen -- 14 August 2026

## Verdict

**`触发停止条件`**

The mandatory player-isolated four-way split cannot be constructed from the
F0-D0 source-domain corpus.  All 4,994 behavior-eligible player keys and all
92,226 behavior games form one connected player-game component.  Keeping all
games of every player in one partition therefore makes that entire component
indivisible.

The frozen assignment contains 92,226 train games and zero selection,
one-time-confirmation, or final-test games.  This is a prerequisite failure,
not an approximate ratio miss.  F0-H0 stopped at that gate.  No raw human game,
Malom entry, HumanDB membership, model, game, search batch, or training path was
opened or started.

The machine-readable result is
[the F0-H0 result manifest](f0-h0-human-feasibility-screen-manifest-2026-08-14.json).
Its result identity is
`714627f8be20bc45a267c97752171644040fc1273a24f82a570a7cb83512fe82`.

## Frozen chronology and identities

The plan was committed and pushed before the membership calculation:

- repository base:
  `4803dcde127ca8e4a16a8df287de1a7dc76a5e2a`;
- preregistration commit:
  `ba2fbf658cd2922e56544ab11bf3f5d16d78425f`;
- plan identity:
  `95a802625867906ab453ed7a52bbba1e0202b08473b10f897ba81c87fb59d530`;
- split commit:
  `1d37ae52c2df5ddb10c4a429f0f530033cad15fa`;
- split identity:
  `e41da5fbf1a2ba60441273664c6834dafcb54bcb79d541f3349a948f7cac5dd4`.

Before splitting, the runner independently recomputed and matched all three
mandatory F0-D0 identities:

- corpus identity:
  `4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29`;
- manifest identity:
  `bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7`;
- manifest file SHA-256:
  `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6`.

The plan file SHA-256 is `3ec32220b220f019b3a60c8a2e1519eae9933933a28b5ccf16072b06b78e2136`.
The 9,021,902-byte
[membership file](../experiments/f0-h0-human-player-split-membership-v1.json)
has SHA-256
`1ef901a6776bab15b96fa4ab25273223ae2028568f619fb35f6ecd96094b26c4`.

## Split method and proof of failure

The split used only the F0-D0 manifest's behavior eligibility, session IDs,
player keys, move counts, and file identities.  It did not open the canonical
raw game files.

The frozen rule was:

1. create an undirected edge between the two player keys in every eligible
   game;
2. compute complete connected components;
3. treat each component as indivisible so a player and all of that player's
   games cannot cross partitions;
4. assign components in descending game-count order to the partition with the
   smallest normalized post-assignment load, using a frozen hash tie-break.

The graph has exactly one component:

| Item | Measured value |
| --- | ---: |
| Player keys | 4,994 |
| Games | 92,226 |
| Logical plies | 4,394,220 |
| Game share | 100% |
| Provisional cross-partition games | 38,156 |

If any game were moved out of train, at least one endpoint player would occur
in more than one partition.  Repeating that argument along the connected path
places the complete component in the same partition.  Thus a non-empty
player-isolated selection or final-test partition cannot coexist with complete
coverage of these games.

The actual frozen counts are:

| Partition | Games | Logical plies | Player keys |
| --- | ---: | ---: | ---: |
| train | 92,226 | 4,394,220 | 4,994 |
| selection | 0 | 0 | 0 |
| one-time-confirmation | 0 | 0 | 0 |
| final-test | 0 | 0 | 0 |

Membership generation is not exposure of final-test game content.  The frozen
access counters for final-test raw records and statistics are both zero.

## Preregistered dimensions and fail-closed abstention

The plan froze every threshold before the component count was measured.  A
null value below means that the statistic was not admissibly estimated after
the prerequisite failure.  It does not mean zero and is not a neutral default.

| Dimension | Frozen threshold | Measured value | Disposition |
| --- | --- | --- | --- |
| Independent support | at least 30 analysis components; largest share at most 20% | 1 component; largest share 100% | failed |
| State support | coarse class at least 30 players and 100 games; at least 80% supported decisions | not opened | abstained |
| Positional `A_pos` reach | modifiable states at least 5%; supported game-reach LCB at least 10% | 0 Malom queries; distribution not opened | abstained |
| Player decision concentration | top 1% at most 25%; Gini at most 0.75; Kish ESS at least 500 | component concentration measured; decision counts not opened | abstained |
| Product upper bound | conservative factual upper 95% bound at least 1 score point per 100 natural games | not opened | abstained |

The four requested scientific dimensions cannot be completed under the frozen
design once the independent split is empty.  Computing them on the all-train
component would expose exploratory quantities without a selection population
and would not repair the failed player-isolation requirement.  The screen
therefore records the prerequisite observation and explicit abstentions rather
than manufacturing nominal values.

No `A_allow` claim was made.  No `A_pos` value was queried.  W-to-D, W-to-L,
D-to-L, and within-tier comparator regret remain unmeasured rather than being
combined or set to zero.

## Cost decision

The preregistration froze a 64-game, 256-state bounded Malom timing pilot, a
50-million-query and two-hour full-pass ceiling, and an 8,192-whole-game
deterministic fallback sample.  The split gate occurs before this cost gate.
Consequently:

- the cost pilot was not started;
- Malom queries were zero;
- no full-versus-sample decision or sample membership was produced; and
- no cost estimate was needed to authorize an analysis that had already
  stopped.

This preserves the required ordering: no screening statistic was viewed before
the split was frozen, and no partial Malom result was used to change sampling.

## Inherited limitations

The F0-D0 attrition remains non-random.  The 1,751 excluded games contain only
35 draws, while the 92,789 history-recoverable games contain 26,157 draws.
Nothing here converts the retained corpus into a random sample.

UI orientation, time control, exact source rule variant, explicit import batch,
and upstream source-file identity remain absent.  The 54,923 games without an
independently verifiable terminal result remain unverified.  Even absent the
split failure, claims would be restricted to the observed PlayOK-like source
domain and could not be transported to the product UI, other time controls,
other rules variants, or new people.

## Access and operation audit

| Operation | Count |
| --- | ---: |
| Raw human game files opened | 0 |
| HumanDB membership reads | 0 |
| Malom queries | 0 |
| Remaining `2eb04f54` source-pool records read | 0 |
| Games or search batches started | 0 |
| Models loaded | 0 |
| Database writes | 0 |
| Training updates | 0 |

An initial direct-file runner invocation failed during Python import because
the repository root was not on the module path.  It occurred before boundary
loading and produced no output.  The equivalent module entry point then
generated the split.  No retry of a counted run or semantic analysis occurred.

## Verification

The focused test file covers source-base rejection, player leakage rejection,
connected-component assignment, strict replay, `A_pos` versus complete-order
regret, concentration, Wilson intervals, and component-robust interval
behavior.  Before the split it passed as follows:

```text
7 passed in 0.61s
```

Task-scope Ruff also passed.  The split loader subsequently recomputed both the
sealed split identity and file SHA-256 and rechecked complete game coverage,
unique player membership, zero player leakage, and zero protected raw access.
An independent standard-library disjoint-set calculation, without importing
the F0-H0 implementation, again returned one component, 92,226 games,
4,394,220 logical plies, and 4,994 player keys.

## Consequence

The safe-human-trap funnel stops at F0-H0 for this source-domain corpus and the
required player-isolated design.  E0, F0-H1, T0-H-pilot, trap reward changes,
training changes, alternative data substitution, and source-pool access were
not started or authorized by this work.
