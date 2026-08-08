# Sanmill Exact-Resume Parity Smoke v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The continuous reference and first segmented process are ready for one staged,
bounded launch each after this record is published and their final matching
preflights pass. The second segmented process is conditionally ready: it may
run once only after segment 0001 completes, its checkpoint and SpecialistDB
pass integrity checks, and the frozen exact-resume command returns
`ready_for_smoke` from the same clean published source commit.

This is not a retained training authorization. Any failure, quarantine,
non-periodic update, non-finite value, or semantic parity difference stops the
comparison and consumes the affected one-run authority.

## Gate summary

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Repository | `dev == origin/dev == 134095562641f960ad9e8d3e0d86eadad91ccec5`; tracked worktree clean | One clean published source | Pass |
| Frozen contract | 9,373 bytes; SHA-256 `78720014c9cc7747d746a51150a7d42c0632f7c9394f6200f2d4960066a974fd` | Published staged comparison with fail-closed bounds | Pass |
| Start lineage | Both routes are fresh random seed 42; segmented route alone may exact-resume its immediately preceding checkpoint | No historical checkpoint or cross-lineage import | Pass |
| Continuous output | Absent | New isolated path | Pass |
| Segment outputs | Both absent | New isolated paths | Pass |
| Continuous SpecialistDB | Empty 0/0/0 training rows; unbound; `quick_check=ok`; `sector-corrected-v1`; content SHA-256 `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d` | Empty trusted isolated DB | Pass |
| Segmented SpecialistDB | Same empty content identity, schema, counts, and integrity; separate path | Empty trusted isolated DB reused only within segmented route | Pass |
| Malom | 512 components; manifest identity `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` | Corrected trusted tablebase | Pass |
| HumanDB | `quick_check=ok`; identity `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`; historical Malom labels masked | Frequencies/outcomes only | Pass |
| Rules | `nmm-training-core@2`; semantic digest `sha256:52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a` | Frozen trainer rules | Pass |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary SHA-256 `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`; strict-referee digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` | Exact clean pinned strict runtime | Pass |
| Sanmill determinism | Two fresh-process probes produced the same first turn and observation SHA-256 `0a6c478d75cded748fa397e65831f0cfc0e3c3040248b4741a5704f4f35d03bd` | Fixed-seed fixed-node equality | Pass |
| Components | A2C only; recovery, branches, Sentinel, ValueNet, GapNet, S1A, S1B, imitation mixing, and opening forcing disabled | Explicit smoke boundary | Pass |
| Update timing | `update_every=8`, the minimum valid batch; one primary game exceeds that threshold in the fixed schedule | Periodic update before each segment boundary | Pass |
| Work ceiling | Four primary games, 480 logical plies, two 1,000-node Sanmill-opponent games, two frozen-target games | Bounded parity smoke only | Pass |
| Initial preflights | Both fresh commands returned `ready_for_smoke`; `errors=[]`; `unresolved_decisions=[]` | Passing read-only gates | Pass |
| Exact-resume preflight | Deferred until segment 0001 creates the required checkpoint | Must pass on the same source commit before segment 0002 | Conditional |

## Preflight identities

The uninterrupted command produced:

- config SHA-256
  `36d0849f0c552bbb667f5e719f1963f99b5dcb5e985079e3b645cb7200f34e6d`;
- resume-config SHA-256
  `b85c67ab6f56460553351fa502fe9def995d572e35d00cc4b8ae091c6546122d`;
- experiment digest
  `sha256:53c7dfec4261a8d149e574522542e355dc5318581e1dd734b06cfb13057e44b1`.

The first segmented command produced:

- config SHA-256
  `fd824b85b63830b11b9510bb2225a3cde5e897b5146f8a1306d2036036038f8f`;
- resume-config SHA-256
  `cfe87f495b142739a99a1f3415a99a3125fda5520142ad940a52033b9560fbda`;
- experiment digest
  `sha256:4e9ebd94a98092e1d26549ac06a54509a4acc5d0ec7e49e7b272b3f4cf0a1658`.

The routes intentionally have different database paths and run identities.
Those values do not influence gameplay, and the parity verifier removes only
their documented mutable/path identities before comparing complete semantic
state. Segment 0002 must reproduce segment 0001's resume-config and experiment
identities exactly.

## Verification

```text
67 focused update, segment-stop, checkpoint, preflight, manifest,
Sanmill-referee, and resume-parity tests passed
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Focused Ruff checks passed
git diff --check passed
```

Running Ruff over the entire historical trainer script still reports its
pre-existing import-placement and unused-code baseline. No trainer source was
changed for this plan, and the focused modules and tests selected above pass
Ruff. This record does not claim a repository-wide lint baseline.

## Reviewed command and staged execution

The exact three training commands and semantic comparison command are frozen
in
[the experiment contract](../experiments/sanmill-refereed-exact-resume-parity-smoke-v1.md).
They were reviewed with the following mandatory sequence:

1. publish this readiness record;
2. rerun the continuous and segment-0001 exact-command preflights from that
   published commit;
3. launch the continuous command once, then segment 0001 once;
4. verify their lifecycle, updates, checkpoint envelopes, and databases;
5. without changing Git, run the segment-0002 exact-command preflight;
6. launch segment 0002 once only if that preflight passes; and
7. run the frozen semantic parity command and persist the result.

No source commit, argument, input database, seed, output path, or checkpoint
may be changed between steps 2 and 7. A mismatch is `fatal_stop`, not authority
to retry with a convenient adjustment.
