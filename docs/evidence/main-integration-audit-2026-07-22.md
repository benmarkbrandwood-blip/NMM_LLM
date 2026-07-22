# Maintainer Main Integration Audit — 22 July 2026

## Scope and Claim Boundary

This record audits the maintainer's `main` history through `67af016` and the
separately downloaded `Mills` bundle before either is used by `dev`. It records
repository integration and read-only provenance checks only. It does not authorize a
smoke, retraining run, resume, evaluation, model promotion, database
replacement, push, or publication.

The initially observed tip `b9a13ce` was merged on the temporary integration
branch by commit `8717f1c`. Seventeen conflicts came from `main`'s one-parent
import of an older `dev` snapshot. The newer `dev` side was retained for all seventeen because the
other side removed later fixed-node candidate, quiescence, managed-segment,
exact-resume, paired-evaluation, and provenance safeguards. The merge still
imported the maintainer's non-conflicting plans, openings, settings,
checkpoints, plotting changes, and Generalist v2a source.

During final verification, `main` advanced once more to `67af016`. Merge
commit `4593034` imports its v2a `best_win_rate_at_diff` persistence while
retaining the dev quarantine. That field prevents one class of legacy
best-checkpoint regression, but it does not add optimizer, RNG, target-model,
recovery, rehearsal, database-identity, or exact-resume state.

## Staged Database Bundle

The following files remain in the sibling `Mills` staging directory. Neither
has replaced the active repository-local database.

| Candidate | SHA-256 | Read-only audit |
| --- | --- | --- |
| `human_db.sqlite` | `F0B20D33AEFCBAB9AEDC8537F12FA2E53F7865B0387E2175AFD0EA32D1B90E42` | 745,385,984 bytes; supplied sidecar matches; SQLite `quick_check=ok` |
| `specialist_db.sqlite` | `DF269D692E43815B88373F54B5AB1287022BC6736ECC8A5B95C7FB8A97FCD629` | 290,820,096 bytes; SQLite `quick_check=ok` |

The HumanDB contains:

- metadata `malom_label_version=sector-corrected-v1`;
- build timestamp `2026-07-21T12:38:20` and 95,221 recorded games;
- 2,167,498 positions, all with a Malom WDL value;
- 2,533,886 move rows, of which 2,472,054 have successor Malom WDL;
- position labels: 1,103,921 draw, 474,542 loss, and 589,035 win;
- successor labels: 1,341,882 draw, 506,007 loss, and 624,165 win.

Fifteen deterministic position samples and fifteen deterministic successor
move samples, five from each W/D/L class, were reconstructed from their
canonical state keys and queried through the current corrected Malom adapter.
All 30 matched both stored W/D/L and stored DTW. This is positive validation of
the sampled rows, not a proof of every row or an activation decision.

The SpecialistDB contains:

- metadata `malom_label_version=sector-corrected-v1`;
- 2,112,951 empirical positions;
- 60,117 winning lines and 30 preferred plays;
- zero persisted Malom labels.

It is a valid candidate for an experiment that deliberately retains the
maintainer's empirical self-play history. It is not equivalent to the empty
SpecialistDB used by the corrected fresh baseline and must not replace that
database without a new experiment contract.

The repository-local corrected SpecialistDB is also no longer empty: the
completed managed baseline populated it. Its current SHA-256 is
`1203FC73CD7D0A06E2DD1FFACED5B031DFF8BD704E22B34BA02182FF3865614D`;
it has 132,182 positions, 41,904 current-version Malom labels, 916 winning
lines, no preferred plays, and lineage root
`managed-v4-baseline-v1-segment-0001`. It is preserved completed-run state,
not an input to a second fresh experiment. The staged candidate, completed-run
database, and a newly empty database are three distinct lineages.

## Imported Checkpoints

The merge updated seven tracked legacy checkpoint files. All tensors are
finite and differ from the previous `dev` copies. Their payloads contain model
weights/configuration plus stage, game count, best win rate, difficulty,
source-path text, learning rate, and temperature. They do not contain the
optimizer, RNG state, target model, rolling histories, database identities,
run manifest, or the v2 exact-resume envelope.

| Path below `learned_ai/checkpoints/scaffolded` | SHA-256 | Embedded state |
| --- | --- | --- |
| `s_open_v2/best.pt` | `D020E1442676E16CDCED6C91DAC958817C3A22A283CC293D6E19930A87703701` | game 14,350; difficulty 3; best rate 0.225 |
| `s_mid_v2/best.pt` | `A587AB995224A1D43C99FD2F42E4BFF9C060AC6DA55EDCDDB43A39FC07EF26D2` | game 20,750; difficulty 7; best rate 0.1388889 |
| `s_mid_v2/best3.pt` | `DEA51BE4BFF825AC3C08D1F282168F539B698F850D7B2DBB7968BD9E1C44D43D` | game 11,750; difficulty 3; best rate 0.25 |
| `s_end_v2/best.pt` | `5DE51A1AFD5794374D4394CCE2950957A23F02504B5C5952A062D91414B94BE8` | game 17,600; difficulty 3; best rate 0.15 |
| `s_end_v2/best1.pt` | `D1327BF7E98BAABD582FD1C4166CEF0280B9F178121606C2F835C58A5548FA0F` | game 4,100; difficulty 1; best rate 0.30 |
| `s_gen_v2/best.pt` | `494CEC3F78D3B8F8F05D61D30A7C620D796CC386D5E355CA3FDAA5E3D16A792F` | game 19,250; difficulty 9; best rate 0.25 |
| `s_gen_v2/best6.pt` | `0E024D4402160BEFA4A7DDEDB56735FCA8CC9D924FC069A550F1761E643CA93D` | game 12,750; difficulty 6; best rate 0.30 |

These remain maintainer-`main`, weights-only historical artifacts. Their upload
date does not prove that their training began after the Malom decoder and
database rebuild. They are not `dev` resume inputs, exact-resume checkpoints,
formal baselines, or evidence of corrected retraining.

## Other Main Artifacts

- `data/openings/learned_openings.json` contains 169 entries and has SHA-256
  `E348CBD442BB221588BC96DD7EF0500AB8CA31AA1306B84CFEC422D7D4EF1C8E`.
  It is separate from the pinned Sanmill Oracle review corpus and does not
  change that corpus's identity or expert-review status.
- The downloaded `Mills/train_s_gen_v2a.py` has SHA-256
  `EDB0E4D35981900414A1DC6E870E36F44F23D77EE2693C7D56C4C2ACDC566F79`.
  The `b9a13ce` version first imported by Git had SHA-256
  `FF5760F5D08A9D48B4A5E5E72CF51F74AD6E45EE11B861B388BA231299BE9BB5`.
  The final upstream `67af016` version has SHA-256
  `4DC873E86F5C7B94D6BFC412CDBCAC404D97E4199AAAB97A35E9F77DB4AE8F9B`.
  Therefore the repository history, not the earlier downloaded copy, is the
  relevant review source.
- The v2a fork is 1,011 lines behind the current `dev` trainer and lacks the
  current managed preflight and exact-resume contract. It also silently
  degrades missing required components and uses a fixed SpecialistDB path.
  Commit `76f3ff3` preserves the source but quarantines its runtime entry point.
- The imported SpecialistDB clearing utility originally modified a default
  database in place. Commit `f7c5b19` replaced that behavior with an explicit,
  hash-bound, non-destructive copy migration and focused tests.

## Retraining-Plan Review

The maintainer's `docs/retrain_v2_plan.md` is a useful proposal, not a frozen
run contract. Local implementation work can resolve its flag mismatches,
fail-open loaders, database activation, fixed-work benchmark, and
evidence-ledger requirements. Follow-up comparison with current code and the
v5 design resolves the three apparent model ambiguities without requiring the
maintainer to restate them:

1. Sentinel is the documented DB-free runtime advisor. Malom may generate
   supervision, but every oracle-derived input slot remains masked in training
   and inference. The proposed oracle-exposed Stage 5 is rejected.
2. The existing ValueNet remains an outcome/value estimator. The imported
   next-move ranking proposal is a HumanPolicy research path and cannot reuse
   the ValueNet name, checkpoint, calibration, or promotion path.
3. GapNet retains its implemented current-position target: best composite
   quality minus frequency-weighted observed-human quality. A future temporal
   blunder-hazard model would be a separate experiment.

The aggregate HumanDB has no per-game membership in its `positions` and
`moves` tables, so it cannot by itself produce the plan's required game-level
train/validation split. A compliant implementation must split source games
first, replay them, and join their canonical state keys to the versioned DB, or
rebuild a dataset that retains game identity.

## Post-Merge Verification

The integrated code tip `0d01f31` was verified before this evidence-only
update:

| Scope | Result |
| --- | --- |
| New migration/quarantine plus Oracle, paired, bundle, and lifecycle tests | 24 passed |
| Required Malom, DB-teacher, and label-provenance suite | 102 passed, 498 subtests passed |
| Generalist launch, managed control, preflight, manifest, checkpoint, path, resume, segment, identity, and temperature tests | 86 passed |
| GameAI, search-enhancement, and heuristic-parity tests | 65 passed |
| Native Rust unit suite | 26 passed |

The one initial Generalist failure was a stale ready-case fixture that omitted
the now-required absolute first-segment stop. Preflight correctly returned
`fatal_stop`; commit `0d01f31` supplies `segment_stop_game=250` to the fixture
without weakening the gate, after which all 86 tests passed.

All seven imported checkpoint tensor sets are finite. The merged opening data
loads as 169 learned entries alongside 11 canonical book entries. JSON and
Python syntax probes pass, and the v2a entry point fails before runtime setup as
intended. The Rust suite emits existing unused-code/import warnings but no test
failure. No full-suite-clean claim is made because the repository handover
still records unrelated collection/interface failures outside these focused
groups.

After `main` advanced to `67af016`, the v2a quarantine test and CLI parse probe
were repeated against merge `4593034`; both passed and the launch guard remains
in place. No broader suite was repeated because that upstream delta affects
only quarantined source and the focused boundary test passed.

## Remaining External Evidence

The seven updated checkpoints do not contain enough state to reconstruct their
exact code and database lineage. Keep them as maintainer-`main`, weights-only
history without asking the maintainer to reconstruct details now. Additional
lineage becomes necessary only if a future experiment proposes to adopt one or
describe it as corrected.

The remaining active maintainer input is completion of the 107-position Oracle
review, including the still-open source intent for start 101. That review gates
only the separate Stage-0 corpus freeze. It does not block local safety fixes,
the Git integration, or the contract dispositions above.
