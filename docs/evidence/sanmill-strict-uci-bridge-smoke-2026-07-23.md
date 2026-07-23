# Sanmill Strict UCI Bridge Smoke Result

Date: 23 July 2026

Status: **bridge validation passed with the opening book disabled; formal
candidate-versus-baseline evaluation remains stopped**.

Related records:

- [authorized bridge contract](../experiments/sanmill-strict-uci-bridge-smoke-v1.md)
- [machine-readable evidence](sanmill-strict-uci-bridge-smoke-2026-07-23.json)
- [next-evaluation decision brief](../experiments/dev-v4-training-aligned-evaluation-v1-decision-brief.md)

## Claim boundary

This run validated source and binary identity, the UCI subprocess boundary,
standard-rule outcomes, deterministic replay, staged capture handling, legal-
action parity, and representative fixed-node performance. It loaded no
candidate checkpoint and ran no candidate-versus-Sanmill game. It is not
playing-strength or promotion evidence and does not authorize formal
evaluation.

## Frozen implementation identity

| Field | Result |
| --- | --- |
| Sanmill commit | `6f080c5a6d15919bf0a45fa5528c45d4487a2b8f` |
| Sanmill tree | `8b52f4d084758414ebc9aa4db239448f69e10bcf` |
| Working tree | clean at inspection |
| Binary | `target/release/tgf.exe` |
| Binary SHA-256 | `b1c816ee40f6cb9a91916ad094e82175ee6c975c7d15c396e672af58a15dc1a6` |
| Binary size | 3,720,192 bytes |
| Build | optimized release with debug assertions enabled |
| License | AGPL-3.0-or-later; pinned `Copying.txt` hash recorded in JSON |
| NMM_LLM bridge source | commit `d692f488583b8f8ec04361cf352fc3968ee1d495`; tree `2786c25c7abdf2372b057aab4a0ad91d4a33635d` |
| Contract identity | `26f175d53d0e76813788f23229336b8f823bc1ae20f65b6b2f04187d0ef6879b` |
| Evidence identity | `723d40acf63d22cc7341ba234fff470e5fc5b8c55bf06053540df8ef0cd85b19` |

The child process used one thread, a 16 MiB hash, MTD(f), shuffling off,
seed 42, no wall-clock limit, and `go nodes N` without a positive explicit
depth. It disabled lazy AI, HumanDB, the perfect database, patches, and traps.
It enabled mobility and the normal non-developer
`DrawOnHumanExperience` phase-depth policy. `FocusOnBlockingPaths` was
explicitly disabled: Sanmill's UCI option advertisement says `default true`,
but the engine rule default, Flutter setting default, and patch fingerprint
all use `false`, and the app's rating logic treats enabling it as weaker play.

All inherited `TGF_*` variables were removed. The binary contained the
required assertion immediately before Sanmill's release-only depth-4/random
fallback chain. The bridge supplied no substitute move on failure.

## Rule and terminal evidence

The bridge read Sanmill's exported FEN, complete replay history, legal actions,
and the pinned `d` diagnostic before every search. The diagnostic's `winner`
and `outcome_reason` fields, not an NMM_LLM inference, supplied the result.

| Probe | Sanmill result |
| --- | --- |
| 100-ply without capture | draw; `drawFiftyMove`; zero-node `bestmove draw` |
| Threefold repetition | draw; `drawThreefoldRepetition`; two prior matching keys retained |
| Fewer than three pieces | Black win; `loseFewerThanThree`; no legal actions |
| Staged Mill capture | `d6-d5`, then pending removal search chose `xd1` |
| Capture reset | no-capture counter reset to 0; repetition root reset; history length 0 |

A decisive terminal position is never searched. In the pinned fail-closed
build, calling `go` there would encounter the same `MOVE_NONE` assertion used
to prevent the release fallback from masking an ongoing-position search bug.
The bridge instead stops on Sanmill's authoritative terminal snapshot. The
dedicated terminal-draw probes are safe because Sanmill short-circuits them to
`bestmove draw` before the `MOVE_NONE` path.

At every ongoing stable state in the replay, Sanmill's primary legal-action
set equalled the NMM_LLM atomic move bases. During the forced Mill, Sanmill's
pending `x<square>` set equalled the captures attached to the corresponding
NMM_LLM atomic move. This is codec and ordinary-legality evidence only;
Sanmill remains authoritative for history-dependent adjudication.

## Determinism smoke

Two fresh Sanmill processes replayed the same standard start with a 10,000-
node ceiling per search and a maximum of 60 complete player turns. A removal,
when present, is the second staged UCI action of the same Mill turn rather than
another player turn.

Both runs produced identical moves, FENs, selected depths, node counts,
terminal winner, and terminal reason after timing fields were removed. Each
ended at turn 55 with a threefold-repetition draw. The common semantic identity
is:

```text
0a61ccb62163096a8429fd56a7027466121ca3b23d7ba67bce37c1b369209b80
```

This proves reproducibility for the tested pinned contract. It does not select
the later formal node ceiling or prove deterministic behavior after any
Sanmill source, compiler, option, or binary change.

## Performance smoke

Each sample began from a cleared hash and used the same strict process
configuration. Times are single local measurements and especially noisy at
small budgets; the JSON retains every sample.

| Phase | Node ceiling | Actual nodes | Selected depth | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| Placement | 500,000 | 865 | 3 | 0.291 ms |
| Movement | 500,000 | 500,000 | 30 | 103.133 ms |
| Flying | 500,000 | 500,000 | 30 | 59.852 ms |

The placement result is expected. `DrawOnHumanExperience` selected depth 3
and finished before reaching the node ceiling. A separate empty-board probe
selected depth 1 and searched 52 nodes under a 100,000-node ceiling; with the
same non-developer configuration but the policy disabled, SkillLevel 30 would
select depth 30. The setting therefore was active rather than merely present
in the option transcript.

The 500,000-node samples show that this host can run the representative
movement and flying searches well below 200 ms. They do not freeze 500,000
nodes as the formal budget and are not a latency guarantee across positions or
machines.

## Opening-book result and remaining gate

The pinned Sanmill source closes both book-data findings in two atomic commits:

1. `69d379a1a4e23395a45706df60f63282da20e85f` removes the recommendation to
   place at occupied `c3` and adds authoritative whole-asset legality tests.
2. `6f080c5a6d15919bf0a45fa5528c45d4487a2b8f` removes the sole duplicate
   `c5` candidate, which otherwise changed rank-biased selection weight.

| Field | Value |
| --- | --- |
| Asset SHA-256 | `cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5` |
| Oracle entries | 109 |
| Unique recommendations checked | 437 |
| Illegal recommendations | 0 |
| Duplicate recommendations | 0 |
| Historical invalid-key SHA-256 | `904777ade504367c4e62446f105f1b125aaea7d6bec217984518025d8df3b0d1` |
| Historical invalid key still present | false |

The audit queried the pinned Sanmill UCI legal-action set for every entry; it
did not trust an independent NMM_LLM-only legality reconstruction. The older
owner-reviewed 106-position corpus keeps its historical provenance and is not
silently regenerated or relabelled from this corrected asset.

Book play was nevertheless disabled in this bridge smoke because `tgf mill
uci` still advertises no opening-book option. The remaining gate is a
deterministic fail-closed UCI or referee interface and a frozen paired-opening
diversity policy. No book miss, illegal action, database error, or malformed
response may silently change the selecting policy.

The provisional next infrastructure smoke, which this evidence does not
authorize, would assign 75% of pair identifiers to corrected-book-derived
prefixes and 25% to StrictSteps perfect-database tied-best prefixes. The latter
contains exactly eight logical player moves in total: four by each side, or
four full rounds, not eight rounds. A Mill-forming move and its required staged
removal count as one logical move despite using two UCI action tokens. Both
colour-swapped games replay the same frozen-seed prefix before MTD(f) resumes
with search shuffling disabled. The ratio, prefix length, book-miss semantics,
and exact database identity remain to be audited and frozen.

## Verification

- Sanmill focused opening-asset tests: 2 passed after both corrections.
- Sanmill: `cargo test --workspace` passed; no pre-existing ignored external or
  slow test was weakened, deleted, or reclassified.
- Sanmill formatting and lint: `./format.sh s` changed zero of 767 checked
  files and its Clippy phase passed.
- NMM_LLM focused strict-bridge tests: 22 passed.
- NMM_LLM complete suite: 984 passed and 498 subtests passed in 3228.82
  seconds (53:48).

## Decision

The strict UCI bridge is suitable for continued infrastructure work under the
pinned, book-off contract. The corrected book data passes authoritative replay,
but the desired opening configuration still cannot be represented by the
current UCI and its paired-diversity policy is not frozen. Formal candidate-
versus-baseline games remain unauthorized. After the opening-interface gate
closes, the formal node budget, reviewed starts, history semantics, match
workload, and launch must still be frozen separately.
