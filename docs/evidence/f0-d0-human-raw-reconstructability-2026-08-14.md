# F0-D0 raw human-game reconstructability screen

Date: 2026-08-14

Status: `completed_read_only_f0_d0_partial_recoverability`

Audit implementation commit:
`a2dcf9e6fe427ac4f834458deabf4e8bc6da168a`

Machine-readable evidence:
[F0-D0 manifest](f0-d0-human-raw-reconstructability-manifest-2026-08-14.json)

- file size: 30,378,269 bytes;
- file SHA-256:
  `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6`;
- manifest identity:
  `bf7404d1f090073a1b36635b89d329e7011140d48e4fb3b3076efd7e55b5bca7`;
- current raw-corpus identity:
  `4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29`.

## Decision

F0-D0 is complete. The current raw PlayOK corpus is **partially recoverable
inside its observed source domain**. It is not fully recoverable across all
five required dimensions.

The source records support a player-keyed, strict-history behavior subset of
92,226 games, 4,394,220 logical plies, and 4,994 independent source-scoped
player keys. They also support a smaller strict-outcome subset of 37,866 games
and 3,392 player keys. This is enough to reject the global statement that the
raw histories are wholly unusable. It does not establish transport to the
product UI, a different time control, a specific PlayOK rules variant, or a
new human population.

F0-H0 was not started. No F0-H0 membership, access, estimand, or acceptance
decision is created by this screen. No game, search batch, model load,
training update, database write, migration, rebuild, promotion, publication,
or release occurred. The source-pool artifact `2eb04f54` was not opened, and
none of its remaining 108 records was read or consumed as a pool record.

## Inputs and method

The audit enumerated every `*.jsonl` below `data/human_games`, hashed the exact
bytes, required one JSON object per file, reconciled the filename and embedded
session ID, and compared the resulting session set with `imported.json`.
Byte-identical duplicate exports were retained in the file inventory and
collapsed only for the unique-session replay. A same-session hash conflict
would have stopped the audit.

Each unique record was replayed from `BoardState.new_game()` under tracked
ruleset `nmm-training-core@2`. At every logical ply the audit checked:

- the recorded pre-move FEN;
- side to move, turn number, phase label, and notation;
- one exact match in the complete legal-move set; and
- absence of any recorded move after a strict rules terminal.

`StandardDrawTracker` reconstructed the active stable-position repetition
multiset and the 100-logical-ply no-progress clock. Placement and removal
resets, the threefold threshold, and repetition priority therefore came from
the same independently tested rules used by the project. A zero-move record
has the exact initial history and zero clocks, but contributes no human
decision and is excluded from behavior support.

Player handles were not copied into the manifest. Each source-supplied handle
was converted to a domain-separated SHA-256 key over the source and handle.
This preserves within-corpus grouping without claiming that aliases, account
changes, or identities across platforms can be linked.

The active and archived HumanDB files were opened only with SQLite URI
`mode=ro&immutable=1`. Main files and all present WAL/SHM sidecars were hashed
before and after the query; both databases returned `quick_check=ok`, and no
file identity changed. Aggregate `positions` and `moves` rows were not used to
manufacture game membership.

## Independently reproducible identities

The manifest uses canonical UTF-8 JSON with sorted keys, no insignificant
spaces, and no non-finite numbers. It records all 95,397 input-file paths,
lengths, and SHA-256 values. The 95,397 comprise 95,389 raw JSONL files,
`imported.json`, the strict ruleset, two HumanDB main files, and four present
sidecars.

The corpus identity is SHA-256 over this canonical projection:

```text
{
  schema_version: "nmm.human-raw-corpus.v1",
  raw_files_identity,
  session_source_identity,
  imported_manifest_sha256,
  unique_sessions
}
```

`raw_files_identity` is the canonical ordered list of raw relative path, byte
length, and file SHA-256. `session_source_identity` is the canonical ordered
list of session ID, canonical raw file, that file's SHA-256, and import
timestamp. The compact row field names, enum tables, player-key table, and
presence-mask semantics are included in the manifest itself.

Two final checks passed:

1. the project verifier recomputed the manifest, input, audit-result, player,
   raw-file, session-source, and corpus identities; and
2. a separate Python-standard-library check reopened and rehashed all 95,397
   actual inputs, found zero length/hash mismatches, and independently
   reproduced the raw-file, session-source, corpus, and manifest identities.

The principal nested identities are:

| Identity | SHA-256 |
| --- | --- |
| Raw files | `ca709dae9ae596f73716050cf81a1b641bbaebfb4bf7d9323e19dacecf3eaf9c` |
| Session source | `f9c510794ca0171b7e8ca4aabe7df7cc420d3d79ed31f77f38f6a340537adace` |
| All inputs | `5d543a1d7857b8568ce3cae8c0f07f7f1c3dad1727df17d242ba86d6ce5f24c9` |
| Audit result | `6e534c115bc34be7b2b2f775598b3f27407aef3569bdd41126c00e9286b6a96d` |
| Player-key set | `3fee89671f5ba5a8ce0000b273a682006c85e59e12632f2dba1fb988201ee3bc` |

## Exact inventory reconciliation

### Current raw files and import manifest

The three current counts are not three estimates of one quantity:

- 95,389 is a physical JSONL file-occurrence count: 94,527 top-level files
  plus 862 files under `test_set`;
- 94,540 is the unique session count: the 94,527 top-level sessions plus 13
  sessions present only under `test_set`; and
- the other 849 `test_set` files are byte-identical duplicates of top-level
  sessions. There are zero conflicting duplicate hashes.

`imported.json` contains exactly 94,540 IDs. Its set is exactly equal to the
94,540 unique raw session IDs: both set differences are empty. Its file
SHA-256 is
`90e1e12668b8f9e0ca93365ec499b1b328163f9d10961b89c741637e30327beb`.

### Active HumanDB and 94,429

The active database is not a current unique-game manifest. Its exact retained
state decomposes as follows:

```text
94,983 processed paths
= 94,134 normalized session IDs + 849 duplicate paths

94,423 exact sum(processed_files.games_found)
= 93,575 unique session IDs with a game + 848 duplicate accepted paths

94,429 meta.total_games
= 94,423 processed ledger sum + 6 metadata drift
```

There are 560 processed rows with `games_found=0` and 94,423 with
`games_found=1`. At the session level, 559 have no accepted game. Duplicate
groups are exactly 848 `(1,1)` groups and one `(0,0)` group. All 849 duplicate
groups have matching hashes.

The active processed set lacks exactly 406 current raw sessions and has zero
sessions absent from the current raw set. Thus the apparent difference
between 94,540 current sessions and `meta.total_games=94,429` is exactly:

```text
94,540 - 406 newly unprocessed - 559 zero-game sessions
+ 848 duplicate accepted paths + 6 metadata drift = 94,429
```

The retained database state contains no event ledger capable of assigning the
six-row metadata drift to particular games. The current and SHA-aware builders
write `total_games` and `games_found` from the same per-run counter, so this
state proves that the metadata and processed ledger diverged after or outside
one internally consistent full build, but it does not prove how. The manifest
therefore records the cause as
`not_recoverable_from_retained_database_state`; inventing six game IDs would
violate the fail-closed requirement.

### Archived candidate and 95,221

The archived rebuilt candidate is a different input inventory:

```text
95,785 processed paths
= 94,936 normalized session IDs + 849 duplicate paths

94,936 normalized session IDs
= 94,540 current raw sessions + 396 archived-only sessions

95,221 sum(processed_files.games_found) and meta.total_games
= 94,373 unique session IDs with a game + 848 duplicate accepted paths
```

It contains 564 zero-game paths, corresponding to 563 zero-game session IDs
plus the duplicate copy in the one `(0,0)` group. Its duplicate patterns are
again exactly 848 `(1,1)` and one `(0,0)`. Metadata and processed sum agree;
there is no drift. The manifest lists all 396 archived-only IDs individually.

Therefore 95,221 is exact for the archived candidate's different
95,785-path inventory. It is not an identity or acceptance denominator for
the current 95,389-file / 94,540-session corpus.

## Five-dimension results

| Dimension | Recoverable | Partial | Not recoverable | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Complete history | 92,789 | 0 | 1,751 | 98.1479% replay from the initial state with strict repetition/no-progress state |
| Player | 94,540 | 0 | 0 | both source handles present; 5,036 source-scoped player keys |
| Source | 0 | 94,540 | 0 | platform, source type, current raw file, and import timestamp present; upstream file identity and explicit batch absent |
| Result | 37,866 | 54,923 | 1,751 | strict terminal agreement, nonterminal result without basis, or unavailable because history failed |
| Condition | 0 | 94,540 | 0 | color assignment, date, and both Elo fields present; UI orientation, time control, and exact rules variant absent |

No session is fully recoverable in all five dimensions. “Partial” is not
treated as “recoverable” in the acceptance count.

The usable support classes are:

| Support class | Games | Logical plies | Independent player keys |
| --- | ---: | ---: | ---: |
| All source records | 94,540 | 4,470,985 | 5,036 |
| Strict-history recoverable | 92,789 | 4,394,220 | 5,013 |
| Behavior replay eligible | 92,226 | 4,394,220 | 4,994 |
| Strict-outcome eligible | 37,866 | 2,026,799 | 3,392 |

The difference between strict-history and behavior support is 563 exact
zero-move histories. They represent 0.5955% of unique sessions and contain no
choice to model.

## Result recovery and individual failure attribution

Every raw record contains a recorded result/winner pair, but none preserves a
terminal or draw basis. Independent replay nevertheless reached a strict
terminal in 37,866 games:

- 21,175 threefold repetitions;
- 344 no-progress draws;
- 8,332 fewer-than-three terminals; and
- 8,015 no-legal-move terminals.

All 37,866 recorded outcomes agree with the independent result. There are zero
disagreements, so the manifest's required `result_disagreements` list is
empty.

Another 54,923 histories end in a nonterminal strict state. A resignation,
timeout, disconnection, source-side adjudication, or another cause may explain
an individual case, but the raw schema does not preserve that basis. Those
games are result-partial and carry per-session
`result.terminal_basis_missing`; no cause is imputed.

The remaining 1,751 sessions fail at their first move without one exact strict
legal match. The failure index ranges from zero-based logical ply 4 through
210, with mean 42.8407. Each compact game row records its session, canonical
file link, history status, `history.illegal_or_ambiguous`, first failure ply,
and the derived `result.history_unrecoverable` attribution; the linked input
row supplies the raw file SHA-256. Missing plies, unsupported source variants,
or importer defects are possible hypotheses, but the absent upstream identity
and exact rules variant prevent a unique attribution.

There were no malformed JSON files, multi-record JSONL files, filename/session
mismatches, unassigned files, conflicting duplicate bytes, FEN mismatches,
turn/color/type/notation mismatches, moves after a strict terminal, or
recorded-versus-independent result disagreements. Interrupted and missing-ply
rates cannot be separated from the 1,751 combined strict-legality failures or
the 54,923 missing terminal bases; reporting a narrower rate would be an
estimate, not manifest evidence.

## Selection bias and claim boundary

Player recovery itself selects out no game: all 94,540 records have both
source handles. That supports grouping by the recorded source account, but it
does not prove real-person identity, account continuity, or cross-platform
linkage.

History recovery does select a materially different subset. The 1,751
excluded games contain 797 recorded White wins, 919 Black wins, and only 35
draws, with mean 43.8407 moves and date range 2025-12-12 through 2026-07-14.
The 92,789 recovered histories contain 30,880 White wins, 35,752 Black wins,
and 26,157 draws, with mean 47.3571 moves and date range 2025-12-11 through
2026-07-19. A later behavior or outcome analysis must report this attrition
and cannot describe the recovered subset as a random sample of the source.

The source and condition gaps are universal. Any later result is limited to
the recorded PlayOK-like source domain unless separately observed transport
evidence exists. In particular, UI orientation, time control, exact source
rules, upstream archive file, and import batch cannot be reconstructed from
the aggregate HumanDB or inferred from current filenames.

## Consequence for F0-H0 and reconstruction cost

The compliant path identified in the governing design is feasible for the
recoverable subset:

1. freeze source-game membership before aggregate-state access;
2. keep both byte-identical copies of one session in one component;
3. group and split by the source-scoped player keys;
4. replay the 92,226 behavior games and retain full rule history; and
5. only then join canonical state keys to a versioned database or build a new
   dataset that retains game identity.

The minimum source-domain replay volume is 4,394,220 logical plies across
92,226 behavior games. A strict-outcome analysis has only 37,866 games and
2,026,799 plies. The final audit completed the full inventory, hashing, replay,
and two immutable HumanDB inspections in 164.9 seconds with 16 worker threads;
that is an observed audit cost, not a rebuild or F0-H0 runtime guarantee.

The aggregate HumanDB cannot perform this split because its `positions` and
`moves` tables retain no per-game membership. No database was rebuilt here.
If F0-H0 is separately frozen, it must consume only the manifest-identified
source-domain subset and must not relabel the current aggregate tables as
game- or player-isolated evidence.

F0-D0 alone neither accepts nor rejects human-target specialization. It shows
that history and player support survive at useful scale, while full source,
result, and condition recovery do not. The next gate remains a separately
frozen, read-only, rejection-only F0-H0. Until then, E0, F0-H1,
T0-H-pilot, trap rewards, training changes, and the 108-record source pool
remain untouched.

## Verification

Focused verification passed at the final implementation and evidence state:

```text
python -m pytest -p no:cacheprovider \
  tests/test_human_raw_reconstructability.py \
  tests/test_draw_rules.py -q
14 passed

ruff check \
  learned_ai/evaluation/human_raw_reconstructability.py \
  scripts/audit_human_raw_reconstructability.py \
  tests/test_human_raw_reconstructability.py
All checks passed!
```

`git diff --check` also passed. A full-tree `ruff check .` was run and reported
679 pre-existing violations in unrelated legacy files. Those baseline defects
were not changed, suppressed, or used to weaken the focused gate.

## Hypotheses, counterevidence, and next validation

### Supported observations

- The current raw corpus and import manifest have an exact, independently
  reproducible identity.
- Strict history and source-scoped player grouping are recoverable for most
  records.
- The active HumanDB inventory is stale and path-duplicated, while the 95,221
  count belongs to a distinct archived inventory.
- Source termination basis and three behavior-relevant condition fields are
  universally absent.

### Unresolved hypotheses

- The 1,751 legality failures may reflect missing moves, source variants, or
  historical importer behavior; F0-D0 cannot distinguish them.
- The active HumanDB's six-game metadata drift must have arisen outside one
  internally consistent full builder transaction, but the retained state does
  not identify the event or games.

### Falsifying or limiting evidence

- Zero result disagreements do not make the 54,923 nonterminal outcomes
  independently verified.
- Full player-field presence does not establish real-person or cross-platform
  identity.
- A large source-domain behavior subset does not establish a product-UI or
  future-population effect.

### Next validation boundary

The only permitted successor in the governing order is a separately frozen
F0-H0 planning snapshot with explicit membership, access controls, target
condition, and source-domain claim boundary. This document does not create or
execute that successor. If those prerequisites cannot be met, the design's
human-specialization stop condition applies; no alternative data source is to
be substituted by default.
