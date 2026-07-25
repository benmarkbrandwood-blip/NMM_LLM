# Sanmill Strict Logical-Turn Bridge Smoke v2

Date: 25 July 2026

Status: **frozen for one bounded infrastructure smoke; not a formal
candidate-versus-baseline evaluation contract**.

This contract supersedes the active bridge mechanics in
[`sanmill-strict-uci-bridge-smoke-v1.md`](sanmill-strict-uci-bridge-smoke-v1.md).
The v1 document and evidence remain immutable historical records of the
earlier assertion-build bridge.

## Purpose and claim boundary

Sanmill remains the authority for complete game history, legal actions,
no-capture and repetition counters, and terminal adjudication. NMM_LLM mirrors
stable board positions only to detect coordinate, codec, and ordinary
legal-action divergence.

This smoke may establish only:

- strict protocol integration;
- rule and history consistency;
- one-budget logical-turn handling;
- cross-process reproducibility; and
- indicative placement, movement, flying, and compound-Mill performance.

It must not load a candidate checkpoint, run candidate-versus-baseline games,
estimate playing strength, decide promotion, or consume a formal-evaluation
authorization.

## Frozen Sanmill identity

The bridge source revision is:

```text
db65eb3e73189d934d615d0f47519d395193c646
```

Its tree is:

```text
b8fa6c0119c2dec4443efc59deab8b7d835e0c88
```

The ordinary Windows release binary is built with:

```powershell
cargo build --release -p tgf-cli
```

The accepted `target/release/tgf.exe` identity is:

| Field | Value |
| --- | --- |
| Size | 4,109,312 bytes |
| SHA-256 | `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc` |

The checkout is resolved from the ignored `sanmill_checkout` entry in
`data/training_paths.local.json`; no host-specific path is committed. The
checkout may be exactly the pinned revision or a descendant only when all
changes after the pin are outside the CLI, rule crates, Rust build files,
bridge protocol document, and audited opening-book asset. The evidence records
both the pinned source revision and observed checkout `HEAD`. Any change
inside the pinned scope is a hard stop and requires a new contract and binary
identity.

## Strict process and option contract

The bridge starts `target/release/tgf[.exe] mill uci`, verifies the engine
identity and required advertised options, removes all inherited `TGF_*`
variables, and applies the complete option set recorded by
`strict_contract_record()`.

The essential values are:

| Area | Required value |
| --- | --- |
| Failure policy | `StrictFailurePolicy=true` |
| Search | `Algorithm=2`; `IDSEnabled=true` |
| Work | `go logical nodes N`; no wall-clock limit |
| Concurrency | `Threads=1`; `UseLazySmp=false` |
| Randomness | `Shuffling=false`; `SearchShuffleSeed=42` |
| Clock | `MoveTimeMs=0` |
| Opening depth | `DrawOnHumanExperience=true`; no explicit depth in normal smoke turns |
| Optional data | Perfect DB and patch/trap sources disabled |
| Opening book | audited but inactive in bridge search |
| Human DB | query interface available but inactive in bridge search |
| Standard rules | 9 pieces, flying at 3, no diagonals, 100-ply no-capture rules, threefold enabled |

`DrawOnHumanExperience` is Sanmill's phase-aware search-depth policy. This
smoke checks that the policy is actually active; it does not claim that the
policy improves strength. The dedicated compound-Mill probe supplies
`depth 8` only to make that regression fixture stable and explicit.

## Versioned machine protocol

All three JSON protocols are frozen at version 1:

- strict errors: `info string sanmill_error {JSON}`;
- complete logical turns: `info string sanmill_logical_turn {JSON}`; and
- authoritative snapshots: `info string sanmill_state {JSON}`.

Every response is parsed as JSON and validated for version, status, required
types, rules identity, action shape, outcome consistency, and node accounting.
Malformed, unknown-version, unavailable-position, or explicit error responses
are hard failures. The standard-NMM rules identity must be:

```text
3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f
```

A failed `position` command is synchronized through its following `readyok`
before the error is returned to the caller. This prevents a later command from
mistaking a stale synchronization response for its own completion.

`go logical nodes N` has one aggregate ceiling for the primary action and any
mandatory removal. The response must satisfy:

```text
primary_nodes + removal_nodes = total_nodes <= node_budget
```

An ordinary placement, movement, or flight contains one action token. A
Mill-forming turn contains the primary token and its `x<square>` removal, but
increments `logical_ply_count` exactly once.

`go logical` does not mutate the UCI position. The caller must append the full
returned action sequence, resend the complete `position ... moves ...`
history, call `statejson`, and verify:

- the replayed FEN equals `resulting_fen`;
- action-token and logical-ply deltas are exact;
- the history SHA-256 changes;
- no removal remains pending; and
- terminal flag, winner, and reason agree.

No Perfect DB, patch/trap, depth-4 recovery, local substitute, or random
fallback is permitted on this logical-turn path. A search failure must arrive
as `sanmill_error` with no `bestmove` or replacement action.

## Required probes

The smoke must pass all of the following:

1. source, binary, licence, option, rules, and opening-book identities;
2. malformed JSON, wrong-version, inconsistent-count, node-overrun, and
   action/model mismatch regression tests;
3. an illegal replayed history that returns the exact strict error and leaves
   the protocol stream usable for a subsequent valid position;
4. authoritative `statejson` checks for the 100-ply no-capture draw,
   full-history threefold draw, capture reset, and fewer-than-three loss;
5. a stable compound turn whose move and mandatory removal share one node
   ceiling and replay to the reported state;
6. at every non-terminal self-play state, equality between Sanmill primary
   actions and the base actions of NMM_LLM atomic moves;
7. two fresh-process self-plays with identical complete semantic records after
   elapsed time is excluded; and
8. fixed-node timing probes for placement, movement, flying, and a compound
   Mill turn.

The self-play ceiling is 60 complete logical turns. It is only a bridge and
performance bound. Reaching it is not a rules draw and must not be reported as
one.

## Authorized command

After the adapter, tests, auditor, and this contract are committed and the
readiness audit passes, the one authorized infrastructure smoke is:

```powershell
.\.venv\Scripts\python.exe scripts\audit_sanmill_uci_bridge.py `
  --node-budget 10000 `
  --max-turns 60 `
  --performance-budgets 1000,10000,100000,500000 `
  --output out\diagnostics\sanmill-strict-uci-bridge-smoke-v2.json
```

The raw output remains under the ignored diagnostics directory until it has
been reviewed for status, identities, path portability, rule evidence,
reproducibility, performance, and the claim boundary. A durable result may be
committed only after that review.

## Stop conditions

Stop without running or publishing a passed result on any relevant source
drift, binary mismatch, dirty pinned source scope, unsupported option,
unexpected rule identity, strict error, protocol timeout, malformed JSON,
illegal action, pending-removal leak, replay mismatch, history/count mismatch,
rule-probe failure, cross-process semantic difference, or unexplained resource
anomaly.
