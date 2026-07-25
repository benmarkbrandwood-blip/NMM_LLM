# Twelve-Ply HumanDB Source Audit

Date: 2026-07-25

Status: source-only evidence; final corpus remains `needs_decision`.

Machine-readable evidence:
[`sanmill-layered-human-source-audit-2026-07-25.json`](sanmill-layered-human-source-audit-2026-07-25.json)

## Immutable database snapshot

The active `human_db.sqlite` was not treated as immutable because its 32,768-
byte `-shm` sidecar was non-empty. The audit used SQLite's online backup API to
create an isolated point-in-time snapshot. It did not delete either active
sidecar.

The active database, its zero-byte `-wal`, and its 32,768-byte `-shm` remained
present. The sidecar content hashes and database file size were unchanged
across the backup; the SHM modification time changed as expected when SQLite
acquired read-side coordination state.

The ignored snapshot is found through
`human_db_prefix12_snapshot_path` in the machine-local path registry. Its
evidence is:

| Field | Value |
|---|---:|
| Byte length | 738,091,008 |
| SHA-256 | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c` |
| `quick_check` | `ok` |
| Schema version in `meta` | 2 |
| Processed files | 94,983 |
| Recorded games | 94,429 |
| Positions | 2,152,889 |
| Moves | 2,516,356 |

The portable HumanDB source identity is
`ac50bb5ad836909e85442e8e287c4fef6f25e3569b250f882deda0090a3327cb`.
The unversioned historical Malom columns remain excluded as labels.

## Genuine history extraction

The audit recursively read the current `data/human_games` sample. It did not
chain independent per-position frequency queries. Each accepted prefix came
from one JSONL game and passed all of these checks for its first twelve logical
plies:

- the stored pre-move FEN matched the replayed board;
- the stored actor matched the side to move;
- the stored atomic move and mandatory capture matched exactly one legal move;
- the notation matched that legal move; and
- no earlier terminal state occurred.

The source inventory was:

| Source record result | Count |
|---|---:|
| Complete legal twelve-ply record | 93,783 |
| Game shorter than twelve plies | 1,605 |
| Invalid or incomplete history | 1 |
| Total recursive JSONL files | 95,389 |

The one invalid source is `human_ml11792845.jsonl`, file SHA-256
`51383f49aed02701255c6f41f42953dd3c1a3dacc7cbbd3721165f8d7f30382c`.
Its twelfth action places Black at `a1`, completing `a1-d1-g1`, but the source
record ends without the mandatory capture. It was excluded rather than
repaired.

All 93,783 accepted records report `source=playok` and
`source_type=human_vs_human`.

## Duplicate files and the frequency unit

The recursive source tree contains copied records, including held-out material.
The 93,783 accepted files represent 92,939 distinct PlayOK session identifiers:

- 844 sessions have a second source-file copy;
- no session has more than two accepted copies; and
- no duplicate session disagrees about its prefix or result.

Human frequency, result distribution, repeated support, ranking, and
concentration therefore use distinct games, not file copies. Raw file
occurrence counts remain in the ledger as provenance evidence.

## Frequency distribution

The 92,939 distinct eligible games produce 83,002 exact twelve-ply histories.
The distribution is highly sparse:

- 77,828 histories occur in one game;
- 5,174 histories occur in at least two games;
- repeated histories cover 15,111 games;
- the most common history occurs in 76 games;
- the top ten histories cover 355 games, or 0.382%;
- the top 64 histories cover 1,165 games, or 1.254%; and
- the top 1,000 histories cover 6,001 games, or 6.457%.

Thus 93.77% of exact histories are singletons, and 83.74% of distinct eligible
games have a twelve-ply history seen only once in this sample. The data still
provides thousands of genuinely repeated histories, but it does not support a
claim that a small set represents most human openings.

No minimum frequency or HumanDB corpus membership is frozen by this audit.
"Frequent" means only frequent within this current PlayOK sample.

## Book overlap

Among all 83,002 unique HumanDB histories:

- 3 exactly match a complete named-Book action history;
- 36 share an exact final FEN with named Book; and
- 528 share a final `ring16` orbit with named Book.

Perfect DB overlap remains pending until the independent StrictSteps audit is
complete.

## Local full evidence

The complete per-history evidence is deliberately machine-local and ignored:

- source-file manifest: 16,335,924 bytes, SHA-256
  `35c5d9ff4aaa1c2256077d88a6770b36ac977a9c4ad79e5a21ed79d306493e73`;
- 83,002-history ledger: 66,828,965 bytes, SHA-256
  `96d324bd1f8fc85424bf60e7c3188b3004824a227cee7b69c9b00676bf3f76e9`.

Their locations are resolved through
`human_db_prefix12_source_manifest_path` and
`human_db_prefix12_history_ledger_path`. Every ledger row records the complete
logical turns and action tokens, source-file occurrences, distinct-game count,
first/second-player colour roles, distinct-game result distribution, final
NMM FEN, final `ring16` orbit, and Book overlap.

The tracked JSON is 18,873 bytes with file SHA-256
`6a73e5f608945b95239ee673a7d3f2b083f9f8f5941b16cdbb0e20ed3ca55dcf`.
Its internal audit identity is
`15aa81528a93d0777cc6479a285ff41058ae5860383b24d2629038cd779478fa`.

This audit loaded no candidate and played no games.
