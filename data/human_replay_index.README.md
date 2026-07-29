# Human Replay Index

## Purpose

`human_replay_index.sqlite` is an anonymised, reproducible replay sidecar for the
HumanDB source corpus. It maps aggregate HumanDB position/move rows back to
complete game prefixes without rescanning the JSONL corpus.

This is a **candidate index**, not a game-theoretic oracle. Rust/TGF must
replay every selected history and Perfect DB must independently certify its
W/D/L and distance result before publication.

## Provenance

- Built at (UTC): `2026-07-28T02:46:41Z`
- Source root: `I:\Mill_Training\NMM_LLM\data\human_games`
- Source JSONL files: `95389`
- Anonymised games: `94540`
- Logical turns: `4470985`
- Searchable movement roots: `2662650`
- Duplicate game records ignored: `849`
- Invalid game records skipped: `0`
- Database SHA-256: `15cf134ca465d5cd2ef7590eba069b55c249a188b191adc09a21c47d26c1db82`
- Schema version: `1`
- HumanDB state-key model: `sector-corrected-v1`

Player names and account identifiers are never deserialised into the index.
A source game is represented by the SHA-256 of its exact JSONL row, its
relative source file and its line number.

## Database Roles

- **HumanDB** stores aggregate positions, move frequencies and Malom priors.
- **Human Replay Index** stores anonymised ordered histories for provenance.
- **Perfect DB** supplies the final mathematical proof used for publication.

## Schema

### `meta`

Key/value build metadata, including the schema, source, counts and privacy
model above.

### `games`

| Column | Type | Meaning |
| --- | --- | --- |
| `source_sha256` | `TEXT` | Exact source-row SHA-256; primary key |
| `source_file` | `TEXT` | Path relative to the source root |
| `source_line` | `INTEGER` | One-based JSONL line |
| `logical_turn_count` | `INTEGER` | Number of complete logical turns |

### `turns`

| Column | Type | Meaning |
| --- | --- | --- |
| `source_sha256` | `TEXT` | Parent game |
| `logical_ply` | `INTEGER` | One-based complete-turn number |
| `notation` | `TEXT` | Recorded Sanmill full-turn notation |
| `canonical_notation` | `TEXT` | HumanDB D4-frame notation; `NULL` before movement |
| `board_fen` | `TEXT` | Compact position immediately before the turn |
| `state_key` | `TEXT` | HumanDB-compatible key; `NULL` before movement |
| `move_type` | `TEXT` | Source move type |
| `side_to_move` | `INTEGER` | `0` White, `1` Black |
| `placed_white`, `placed_black` | `INTEGER` | Pieces placed so far |
| `white_on_board`, `black_on_board` | `INTEGER` | Current material |

The primary key is (`source_sha256`, `logical_ply`). Lookup indexes cover
(`state_key`, `canonical_notation`) and movement-root material/ply filters.

## Symmetry

HumanDB joins use its eight D4 symmetries. Sanmill's additional outer/inner
ring swap is applied only at puzzle export, yielding 16 presentation
transforms without duplicating replay rows. Puzzle deduplication remains
canonical under all 16 transforms.

## Example Query

```sql
SELECT r.source_sha256, r.logical_ply, r.notation, p.malom_dtw
FROM turns AS r
JOIN human.positions AS p ON p.state_key = r.state_key
JOIN human.moves AS m
  ON m.state_key = r.state_key
 AND m.notation = r.canonical_notation
WHERE p.malom_wdl = 'W'
  AND p.canonical_winning_move IS NOT NULL
  AND r.canonical_notation <> p.canonical_winning_move
  AND m.malom_wdl_after IN ('D', 'W');
```

The HumanDB result is only a mining prior; the production Rust puzzle
generator performs the final replay and Perfect DB check.

## Rebuild

```powershell
cargo run -p tgf-cli --release -- mill replay-index `
  --games-dir "I:\Mill_Training\NMM_LLM\data\human_games" `
  --out "I:\Mill_Training\NMM_LLM\data\human_replay_index.sqlite" `
  --workers 8 `
  --files-per-batch 64
```

The builder uses bounded parallel workers and one SQLite writer, validates
row counts and `PRAGMA integrity_check`, and publishes the database and this
README only after both have been prepared. Existing outputs are never
overwritten automatically. The version-controlled specification is
`docs/HUMAN_REPLAY_INDEX.md`.
