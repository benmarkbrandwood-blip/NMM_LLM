# Retained-v2 held-out corpus exposure audit — 9 August 2026

## Result

Status: `source_only_no_candidate_loaded_no_games_played`

The frozen 64-record twelve-logical-ply corpus was compared, under the same
D4 canonical position keys used by the trainers, with the exact HumanDB and
final SpecialistDB bound to the retained candidate. No model was loaded and no
game or candidate move was evaluated.

| Measure | Result |
| --- | ---: |
| Frozen executable records | 64 |
| HumanDB D4 matches | 30 |
| Final SpecialistDB D4 matches | 1 |
| Zero-match strict-independence subset | 34 |
| Strict subset Book records | 13 |
| Strict subset Perfect DB records | 21 |

The only final SpecialistDB match is `source-core-034`, a HumanDB member with
four empirical samples. All 21 HumanDB members and nine Book members have a
HumanDB position record. None of the 21 Perfect DB members has a HumanDB or
final SpecialistDB D4 match.

The full 64-record schedule remains valuable because it was source-frozen on
1 August, before the retained training run, and was not used by the 29-state
policy-health gate. It is an operational, route-inclusive benchmark, not a
claim that every start is absent from every trainer-visible data source. The
34-record zero-match subset is therefore frozen as a sensitivity analysis of
the same completed ledger, not as a replacement corpus or an extra run.

## Bound identities

| Input | Identity |
| --- | --- |
| Executable corpus file SHA-256 | `3bcf9db2d003d10769b88767763eb7dfb950eecbff578b7c7ff7d1c208e19771` |
| Executable corpus identity | `417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9` |
| Ordered records identity | `e8a1828cb1d7e0e86c686d934e87934c6c12e6a8cf7610974ed8035937ab8cff` |
| HumanDB identity | `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31` |
| SpecialistDB SHA-256 | `ea2df42d6df837588e1a2d87e37bd025c2b612f87695aa9ae16da064aebf62a8` |
| Exposure audit identity | `df5b04128e4cf21f5325e0601596f0fe74b8f54fb6708c6dfd2c6b79fffdc21e` |
| Strict-subset identity | `a01be0c72b395f2a624c2f5ae7538d9d08eaccde0a392dba566bebe2221806f8` |
| Exposure audit file SHA-256 | `6ca9d040e55ed2fdabf1b6bf079c2f2164615fd15e818c99888390dee4de1678` |

The machine-readable record is
[`sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.json`](sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.json).
It lists the ordered membership and support counts of the strict subset.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\audit_heldout_evaluation_corpus.py `
  --corpus docs\experiments\sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json `
  --human-db data\human_db.sqlite `
  --specialist-db data\specialist_db.sanmill_corrected_retained_v2.sqlite `
  --output <new-output.json> `
  --corpus-file-sha256 3bcf9db2d003d10769b88767763eb7dfb950eecbff578b7c7ff7d1c208e19771 `
  --corpus-identity 417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9 `
  --records-identity e8a1828cb1d7e0e86c686d934e87934c6c12e6a8cf7610974ed8035937ab8cff `
  --human-db-identity 8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31 `
  --specialist-db-identity ea2df42d6df837588e1a2d87e37bd025c2b612f87695aa9ae16da064aebf62a8
```

The command opens both databases read-only, requires the masked HumanDB Malom
policy and `sector-corrected-v1` SpecialistDB metadata, and refuses an existing
output. Focused tests cover the frozen corpus identity, drift rejection, and
separation of operational and strict-independence membership.

## Claim boundary

This audit proves only exact D4-key exposure under the two bound databases. It
does not prove semantic novelty, absence from every raw game, or playing
strength. It must not be used to remove a start after candidate results are
known. Both the operational 64-record schedule and its 34-record sensitivity
subset are fixed before any candidate game is run.
