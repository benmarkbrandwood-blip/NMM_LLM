# Sanmill Exact-Resume Parity Smoke v1 Failure — 8 August 2026

## Decision

`fatal_stop_preflight_policy`

The staged parity smoke ran its uninterrupted two-game reference and its
one-game first segment once at clean, published commit
`1858ee5982b497f7e505f16b10c9bba656dd353f`. Both invocations completed with
finite periodic A2C updates, valid checkpoint envelopes, valid lifecycle
chains, and trusted SpecialistDB state.

The required exact-resume preflight then stopped the comparison before
segment 0002 was created or launched. Its only error was:

```text
non-fresh imports require an explicit non-fresh experiment ID
```

The checkpoint's resume-config SHA-256, SpecialistDB content identity,
experiment digest, MIF identity, rules identity, Sanmill identity, feature
schema, and source experiment all matched. The failure was an over-broad
preflight policy: it treated every non-`fresh` start as an imported lineage,
including an integrity-verified `exact-resume` within the same fresh lineage.

This v1 comparison is closed. Segment 0002 was never launched, no parity
verdict was produced, and neither completed checkpoint may be resumed from a
new source commit. All local artefacts remain ignored, isolated diagnostic
evidence.

## Completed work before the stop

| Route | Games | Updates | Update reason | Lifecycle | Checkpoint |
| --- | ---: | ---: | --- | --- | --- |
| Continuous reference | 2 | 2 | `periodic`, `periodic` | `preflight_passed -> running -> completed` | valid v2 final checkpoint at game 2 |
| Segment 0001 | 1 | 1 | `periodic` | `preflight_passed -> running -> completed` | valid v2 final checkpoint at game 1 |
| Segment 0002 | 0 | 0 | not launched | no output | none |

The first update is byte-for-byte identical in the two update logs:

```json
{"game":1,"policy_loss":0.7583574056625366,"value_loss":0.4042743444442749,"entropy":1.928633213043213,"lr":0.0001,"batch_steps":21,"reason":"periodic"}
```

Neither completed route used a final-flush update. The continuous checkpoint
records `game_count=2` and `update_count=2`; the first segmented checkpoint
records `game_count=1` and `update_count=1`.

## Exact failed gate

The exact-resume preflight read checkpoint
`sanmill-refereed-exact-resume-parity-v1-segment-0001:checkpoint:00000002`
and observed:

- checkpoint and requested resume-config SHA-256
  `cfe87f495b142739a99a1f3415a99a3125fda5520142ad940a52033b9560fbda`;
- checkpoint and requested experiment digest
  `sha256:578aff9a08e56353bda38b105a34f882ddf5b683c8ee1a38f244abf9dc993a30`;
- SpecialistDB content SHA-256
  `c79a2f9c4305d98c1dee05299d87aad1addfebdecbbc9a98e1e3abfd8d17702a`;
- lineage root
  `sanmill-refereed-exact-resume-parity-v1-segment-0001`; and
- source run ID matching that lineage root.

The proposed segment-0002 output did not exist. The preflight returned
`fatal_stop` solely because the experiment ID was
`dev-v4-sanmill-refereed-fresh-v1`. No state mutation or training followed.

## Persisted ignored evidence

| Artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| Continuous `run-manifest.json` | 7,127 | `176b5a1d198b56757829418c0b3c1aed55d1a9f86911c45641482680a1eb61b7` |
| Continuous `run-events.jsonl` | 1,486 | `61779bccdd7a6382304e82c1675e29420045241edcdd4a7c751e3785194a9c07` |
| Continuous `train_log.jsonl` | 2,651 | `160d7219e2545c3a935e3d467c1889bb15ec92634bd5e947fd1a476cdc34a0df` |
| Continuous `update_log.jsonl` | 334 | `e4a4307caa0c86bae3ce1ebd816e3ba77962b29a6c1b0dd341cd581340d70bc7` |
| Continuous `latest.pt` | 2,119,095 | `d2de135e56c8ace9ea985ffe2d6a45c2b7f802924c7c72d0feb9bfe17636c7d6` |
| Segment 0001 `run-manifest.json` | 7,136 | `9d0c59cd7bd097c914d581f2c53f2d6a840b430e26c6dfd08934b9fa0452c0e4` |
| Segment 0001 `run-events.jsonl` | 1,492 | `a5326997a7ae3a6b2b3a05b83df57dcf22b0e7907079bb6d6c7027eafb7f00d0` |
| Segment 0001 `train_log.jsonl` | 1,309 | `a331ed970c5164f1ff61140be099471ef662ecf93af96dd29039c1f95d684c58` |
| Segment 0001 `update_log.jsonl` | 167 | `ac000e3cebcd19725bda8cfdad1e4b6124f3bc269dcd64224f19e124b40b417e` |
| Segment 0001 `latest.pt` | 2,119,101 | `4696d514d58f32270d6ecec7bde90d7a8f1953b841d0cda1b2dfb04bc9d7a2c6` |
| Continuous SpecialistDB | 61,440 | `77b826845061d967fb9666d4a0357472c07480781ce9c7f60dd978885398f952` |
| Segmented SpecialistDB | 45,056 | `c79a2f9c4305d98c1dee05299d87aad1addfebdecbbc9a98e1e3abfd8d17702a` |

Both databases pass `quick_check`, retain
`malom_label_version=sector-corrected-v1`, and remain bound only to their v1
diagnostic lineages. The continuous database has 94 positions, 20 persisted
trusted Malom labels, one winning line, and no preferred plays. The segmented
database has 42 positions, 10 trusted Malom labels, no winning lines, and no
preferred plays.

## Reproduction and repair

The existing exact-resume preflight fixture was changed to use the frozen
fresh-lineage experiment ID. It failed before the repair for the same reason
as the runtime preflight. The repair narrows the import guard to
`start_mode=weights-only`; an integrity-verified exact resume continues to be
checked against checkpoint experiment ID, experiment digest, resume config,
mutable database identity, MIF, rules, Sanmill runtime, and feature schema.

Repair commit:

```text
6e820c1 (Allow exact resume within a fresh lineage)
```

Verification after the repair:

```text
2 targeted policy regressions passed
67 focused update/resume/preflight/referee tests passed
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Focused Ruff checks passed
git diff --check passed
```

The regression also proves that `weights-only` imports into either frozen
fresh experiment ID remain rejected. The repair does not alter gameplay,
optimizer behavior, checkpoint payloads, or referee semantics.

Any retry requires a new v2 contract, new run IDs, new output paths, two new
empty SpecialistDB files, a new readiness record, and final preflights from one
new clean published commit.
