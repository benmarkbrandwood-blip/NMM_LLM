# Fixed-Node Heuristic Search Contract

## Purpose

Managed Generalist training uses a deterministic per-move native node budget
for heuristic opponents (`--heuristic-node-budget`). That path is fail-closed:
Rust search must return a playable move from the same candidate set Python
already filtered. This document records the durable principles behind that
contract and the Sanmill reference used when tightening qsearch.

Sanmill remains a read-only reference checkout. Project rules and the
repository's independently tested NMM semantics stay authoritative. Do not
replace `native/nmm_core` with a Sanmill runtime bridge for this baseline.

## Sanmill Reference

Observed local reference used for the 21 July 2026 alignment:

| Field | Value |
| --- | --- |
| Checkout key | `sanmill_checkout` in ignored `data/training_paths.local.json` |
| Branch | `next` |
| Commit | `6a64010aed7ea4193502ea17c242f68e09fe576a` |
| Quiescence default | `MaxQuiescenceDepth = 0` (stand-pat only at the horizon) |
| Mill qsearch extension | `quiescence_kind_tag = MillActionKind::Remove` |
| Primary sources | `crates/tgf-search/src/searcher/qsearch.rs`, `crates/tgf-search/src/searcher/mod.rs`, Mill self-play audit defaults |

Record the Sanmill commit in any future differential report that cites this
alignment. If the local checkout moves, update this table after verifying the
same defaults still hold.

## Principles

1. **Same candidate set in and out.**  
   Python `GameAI` may narrow legal moves (mandatory mill block, dead-placement
   filters, bans). Native root search must score that allowlist through
   `root_moves`, not the full legal set followed by a post-hoc intersection.
   A miss after filtering is a bug, not a recoverable soft failure.

2. **Fixed work uses stand-pat leaves.**  
   When `node_limit` is set, search uses `fast_eval` (Sanmill
   `MaxQuiescenceDepth=0` analogue). Extended quiet forcing trees must not
   burn a fixed training budget inside the first root move.

3. **Placement is stand-pat even for time-budget extended qsearch.**  
   During `Phase::Place`, qsearch returns stand-pat. Placement-phase
   “reachable two-config” forcing is rejected because almost every developing
   placement qualifies while pieces remain to place.

4. **Non-placement qsearch extends capture/mill-close only.**  
   Align with Sanmill’s Remove-focused quiescence: do not extend quiet
   two-config creators or forced-block quiet moves. Cap tactical recursion
   with `QS_FORCING_CAP`.

5. **Fixed-node aborts must still publish a complete root score list.**  
   If the budget ends mid-root, keep any deeper completed iteration and fill
   missing root candidates with static eval so callers always see every
   allowlisted move.

6. **Time-budget play may keep richer qsearch; fixed-node training may not.**  
   Interactive / time-budget paths may set `use_extended_qsearch` for
   non-placement capture/mill-close extensions. Managed training with
   `--heuristic-node-budget` must remain deterministic and fail-closed under
   principles 1–5.

7. **Sanmill is reference, not authority.**  
   Use Sanmill to choose conservative quiescence defaults and to design
   differential tests. Do not import TGF staged remove actions into the
   training opponent without an explicit, separately authorized experiment.

## Failure Modes These Principles Prevent

| Symptom | Cause addressed |
| --- | --- |
| `exhausted its budget before scoring a move` | Placement/forcing qsearch burned the whole budget on the first root move |
| `returned no allowed move` / allowlist miss | Native scored the full legal set; abort left scores that missed Python’s mandatory block |
| Silent Python fallback hiding search bugs | Fixed-node path must not fall back; completeness and allowlist identity are required |

## Related Code

- `native/nmm_core/src/search.rs` — qsearch policy, root allowlist, completion
- `native/nmm_core/src/lib.rs` — `py_search_root_scored(..., root_moves=..., fast_eval=...)`
- `ai/game_ai.py` — passes filtered `root_moves`; forces `fast_eval` under node budgets
- `docs/managed-training-operations.md` — managed launch uses this contract
- `docs/local-training-layout.md` — Sanmill checkout boundary
