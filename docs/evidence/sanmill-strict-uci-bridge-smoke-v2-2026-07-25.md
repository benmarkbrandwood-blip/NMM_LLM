# Sanmill Strict Logical-Turn Bridge Smoke v2 Result

Date: 25 July 2026

Status: **passed for bridge, rule, reproducibility, and local performance
evidence only; formal candidate-versus-baseline evaluation remains stopped**.

Related records:

- [frozen v2 contract](../experiments/sanmill-strict-uci-bridge-smoke-v2.md)
- [complete machine evidence](sanmill-strict-uci-bridge-smoke-v2-2026-07-25.json)
- [historical v1 result](sanmill-strict-uci-bridge-smoke-2026-07-23.md)

## Claim boundary

No candidate checkpoint was loaded. No candidate-versus-Sanmill game,
playing-strength interval, promotion decision, or formal evaluation was run.
The 60-turn parameter was an infrastructure ceiling, not a Mill draw rule.

This result establishes only that the pinned release and adapter passed the
frozen strict-protocol, rule-history, complete-logical-turn, deterministic
replay, and representative performance checks on this host.

## Implementation and evidence identity

| Field | Result |
| --- | --- |
| NMM_LLM commit | `70de75bb8247ec6795b69045ac53558161e6c045` |
| NMM_LLM tree | `7a88e4dc6b37981cc95ae5a4114835bc01d6adc0` |
| Sanmill pinned commit | `db65eb3e73189d934d615d0f47519d395193c646` |
| Sanmill pinned tree | `b8fa6c0119c2dec4443efc59deab8b7d835e0c88` |
| Observed Sanmill checkout HEAD | `aa6b0c99ee3fca13b0d34e6f929257959ed51414` |
| Relevant post-pin source changes | none |
| Binary | `target/release/tgf.exe` |
| Binary size | 4,109,312 bytes |
| Binary SHA-256 | `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc` |
| Build command | `cargo build --release -p tgf-cli` |
| Contract identity | `c878e36743a15119104a8669ebac2698ad14e83c8bbf88e5684bb546e2ce0b29` |
| Evidence identity | `b8e31cb621e95ecdf5708145c3c4c3ba43b0fbae863bd93460db1beba96cd188` |
| JSON file SHA-256 | `7ec31fdce3086adced94fe07ac13fadcf3f4040a78d12010b320bf4533002d7a` |
| JSON size | 135,289 bytes |

The observed checkout is a descendant of the pin. Its two post-pin commits
change only Flutter tap handling, a Flutter test, and the application version.
The auditor found no difference in the pinned CLI, rule crates, Rust build
inputs, bridge protocol document, or opening-book asset. The binary hash still
matched the release built from the pinned bridge source.

The committed JSON contains no host-specific absolute path. It records the
ignored path-registry lookup key and repository-relative binary and asset
paths.

## Strict protocol result

The process advertised and accepted the frozen standard-NMM options,
including:

- `StrictFailurePolicy=true`;
- one thread, no lazy SMP, and MTD(f);
- `MoveTimeMs=0`, `IDSEnabled=true`, and fixed node ceilings;
- `Shuffling=false` and search seed 42;
- `DrawOnHumanExperience=true` with no explicit normal-turn depth; and
- Perfect DB, patches, traps, HumanDB search use, and opening-book search use
  disabled.

The active rules identity was
`3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f`.
All strict-error, logical-turn, and state protocols were version 1.

The adapter rejected malformed JSON, wrong versions, inconsistent counters,
model/action disagreement, and node-accounting overflow in focused tests. A
real illegal replay returned
`position_history_illegal_action`; the adapter consumed the associated
`readyok`, then accepted and reported a subsequent valid position. A strict
failure therefore cannot be hidden by a stale synchronization response.

## Rule and complete-turn evidence

| Probe | Result |
| --- | --- |
| Empty-board opening policy | effective and completed depth 1; 52 nodes |
| 100-ply no-capture | terminal draw; `drawFiftyMove`; 0 search nodes |
| Threefold repetition | current count advanced from 2 to 3; terminal `drawThreefoldRepetition`; 0 search nodes |
| Fewer than three | terminal Black win in the fixture; `loseFewerThanThree`; 0 search nodes |
| Compound Mill turn | `d6-d5`, then `xc3`; 11,776 total nodes under one 500,000-node ceiling |
| Capture reset | no-capture count 0; repetition count and history length 0 |

For the compound turn,
`primary_nodes + removal_nodes = total_nodes = 11,776`; the removal was
reconstructed from the completed primary search and therefore consumed no
additional nodes. Replaying both action tokens advanced exactly one logical
ply and produced the exact `resulting_fen`.

At every ongoing stable self-play state, Sanmill's legal primary-action set
equalled the base actions projected from NMM_LLM atomic moves. The logical
response's `{from,to,capture}` object also selected exactly one NMM_LLM atomic
move. Sanmill remained authoritative for all history-dependent outcomes.

## Reproducibility result

Two fresh Sanmill processes started from the standard initial position with a
10,000-node ceiling per complete logical turn. After elapsed time was removed,
the complete records were identical:

```text
ae51a16b726e7227f499f054310fed5fbd4b158d8f1b998a4d8cb65d1f7c27bc
```

Both runs used 65 UCI action tokens for 57 logical turns, including eight
mandatory removals. The final turn was `a4-b4`, then `xf4`. The replayed
history SHA-256 changed, the logical-ply count advanced from 56 to 57, and
both the logical response and `statejson` reported White winning because
Black had fewer than three pieces. The run ended by a Sanmill rule outcome,
not by the 60-turn infrastructure ceiling.

The first full v2 invocation stopped before writing evidence because the
adapter rejected this terminal snapshot: Sanmill's encoded state legitimately
retained `action=remove` after the completed removal changed the phase to
`game_over`. Commit `70de75b` permits only that closed terminal combination;
an ongoing remove action without a pending removal remains an error. The
focused regression suite and a 60-turn diagnostic passed before readiness was
re-established and the recorded smoke was rerun.

## Performance samples

These are single local measurements. They inform later workload design but do
not freeze a formal node budget or provide a cross-machine latency guarantee.

| Position | Node ceiling | Actual nodes | Effective depth | Completed depth | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Placement | 500,000 | 1,080 | 3 | 3 | 0.150 ms |
| Movement | 500,000 | 500,000 | 30 | 18 | 59.657 ms |
| Flying | 500,000 | 500,000 | 30 | 10 | 36.191 ms |
| Compound Mill | 500,000 | 11,776 | 8 | 8 | 13.817 ms |

The placement search stopped below the ceiling because the phase-aware depth
completed. Movement and flying consumed the full shared ceiling. The compound
fixture completed its explicitly frozen depth below the ceiling.

## Opening-book audit

The corrected opening-book asset identity remained unchanged:

| Field | Result |
| --- | --- |
| Oracle entries | 109 |
| Unique recommendations | 437 |
| Illegal recommendations | 0 |
| Duplicate recommendations | 0 |
| Historical invalid recommendation present | false |

This is data-integrity evidence, not evidence that the book was used during
search. Book play and HumanDB query results remained inactive in this smoke.

## Verification

- NMM_LLM focused bridge tests after the terminal-action fix: 41 passed.
- Rule-probe diagnostic: passed.
- Complete 60-turn diagnostic: reached a valid rule terminal at turn 57.
- Readiness was repeated after the fix: evidence source scope clean, output
  absent, no residual `tgf` process, and pinned identities matched.
- Recorded smoke: passed with evidence and semantic identities recomputed
  successfully.
- Absolute-path scan of the committed JSON: no match.

The complete repository suite was not rerun for this bridge-only change; this
record does not replace the previously recorded full-suite baseline.

## Decision

The strict Sanmill logical-turn bridge is suitable for continued
infrastructure and evaluator implementation under the pinned v2 contract.
The formal candidate-versus-baseline evaluation remains unauthorized. Its
baseline configuration, opening-source policy, reviewed starts, workload, and
separate launch gate must still be frozen before any strength games run.
