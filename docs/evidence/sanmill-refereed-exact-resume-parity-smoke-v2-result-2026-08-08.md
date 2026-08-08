# Sanmill Exact-Resume Parity Smoke v2 Result — 8 August 2026

## Decision

`passed_exact_resume_parity`

At clean, published commit
`b3d049b51aa6303bc851c9cbcf32b8565b259ab1`, the v2 smoke completed an
uninterrupted two-game route and the same two games split across fresh and
exact-resume processes. All three final preflights passed with no errors or
unresolved decisions. All three lifecycle ledgers ended `completed`.

The repository's existing semantic verifier returned:

```json
{"checkpoint_fields":["model_state","optimizer_state","scheduler_state","scaler_state","rng_state","trainer_state","data_state"],"database_tables":["positions","preferred_plays","winning_lines","meta"],"log_records":2,"status":"passed"}
```

This closes the Sanmill update-capable exact-resume gate. It proves
continuation equivalence for the tested two-game, `batch_games=1`, fixed-seed,
fixed-node domain. It is not strength evidence, not a node-curriculum
selection, and not authority to reuse a smoke checkpoint as long-run lineage.

## Runtime result

Each route scheduled the same Sanmill-opponent first game and frozen-target
second game. Sanmill refereed every complete logical turn. The uninterrupted
route and the combined segmented route produced the same two normalized
training records and the same finite periodic updates:

| Game | Batch steps | Policy loss | Value loss | Entropy | LR | Reason |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 21 | 0.7583574057 | 0.4042743444 | 1.9286332130 | 0.0001 | `periodic` |
| 2 | 26 | 0.0800663903 | 1.1776292324 | 2.7922716141 | 0.00005 | `periodic` |

No route performed a final-flush update. The final continuous and resumed
states both record two completed games and two updates.

Segment 0002's exact-resume preflight bound:

- source checkpoint
  `sanmill-refereed-exact-resume-parity-v2-segment-0001:checkpoint:00000002`;
- source run and database lineage root
  `sanmill-refereed-exact-resume-parity-v2-segment-0001`;
- resume-config SHA-256
  `c1e9ac93d821decb2e31ac0947edbf1f87f6cca61390bd6ec9e89c39980945d7`;
- experiment digest
  `sha256:a8e5f41bf40f8510f62fe0c9ee638de76dee4890ca6f4cff4aac879ce0f1aee4`;
- source SpecialistDB content SHA-256
  `9c48956a89dcc667920098b9b3504e876a5a76b16dcf4dbd9b3f8ec874ea9240`;
  and
- the frozen MIF, rules, Sanmill, model, and feature identities.

The preflight returned `ready_for_smoke`, `errors=[]`, and
`unresolved_decisions=[]` before segment 0002 was launched once.

## Exact comparison scope

The verifier required exact equality for:

- model parameters and buffers;
- Adam optimiser state;
- scheduler and scaler state;
- Python, NumPy, PyTorch, CUDA, game, and component RNG state;
- normalized counters, histories, curriculum/recovery state, frozen target,
  pending experience, and last losses;
- normalized data cursor, caches, buckets, and consumed snapshots;
- the two combined per-game training-log records; and
- positions, preferred plays, winning lines, and metadata semantic rows in
  both SpecialistDB files.

Path, run, source-checkpoint, checkpoint-chain, and database-lineage IDs are
intentionally different and are the only normalized evidence fields. Raw
checkpoint bytes are therefore not expected to match. Neither implementation
nor verifier imports MIF reference gameplay code.

## Integrity checks

All three checkpoint envelopes pass payload verification:

| Checkpoint | Checkpoint ID | Payload SHA-256 |
| --- | --- | --- |
| Continuous final | `sanmill-refereed-exact-resume-parity-v2-continuous:checkpoint:00000003` | `5a39e54702a674e616f804b41445d4af1d08830d43cc8f0965115acc4ed8743a` |
| Segment 0001 final | `sanmill-refereed-exact-resume-parity-v2-segment-0001:checkpoint:00000002` | `a1c69f88a9cc0afa2b96d3eba70bdb110b0860904b1f02a509e52824588f2f0c` |
| Segment 0002 final | `sanmill-refereed-exact-resume-parity-v2-segment-0002:checkpoint:00000002` | `b70d02352a19e94919acfdf9226e3c77c66ab1b3c90639d6fb658f407051fc8a` |

Both final databases pass `quick_check`, retain
`malom_label_version=sector-corrected-v1`, and contain 94 positions, 20
trusted Malom labels, one winning line, and no preferred plays. Their raw
hashes differ because their lineage-root metadata is intentionally different;
the semantic table comparison passed after excluding only that documented
field.

## Persisted ignored artefacts

| Artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| Continuous manifest | 7,127 | `472cc90ea6d9fe0dba98c41494641ac513e3e22b22830e6954e1b40fbc340480` |
| Continuous events | 1,486 | `554c7536b11feda3aa3aa2aae032a7017baa9d6a19ee3f4ab3e8d86d812f3768` |
| Continuous train log | 2,651 | `160d7219e2545c3a935e3d467c1889bb15ec92634bd5e947fd1a476cdc34a0df` |
| Continuous update log | 334 | `e4a4307caa0c86bae3ce1ebd816e3ba77962b29a6c1b0dd341cd581340d70bc7` |
| Continuous latest checkpoint | 2,119,095 | `998b53bd100f414ec7a2e9c98686f73a979742d56a635ab31fa4e3fe13cac2d2` |
| Segment 0001 manifest | 7,136 | `31b76550d7eab809ed44a5aa99ff65c3797b96e192337fee5be6bffa19ce7ef2` |
| Segment 0001 events | 1,492 | `d47962b4961fc5cad0e50988d8be192fb511bf13321a489f37a094f1cca2c435` |
| Segment 0001 train log | 1,309 | `a331ed970c5164f1ff61140be099471ef662ecf93af96dd29039c1f95d684c58` |
| Segment 0001 update log | 167 | `ac000e3cebcd19725bda8cfdad1e4b6124f3bc269dcd64224f19e124b40b417e` |
| Segment 0001 latest checkpoint | 2,119,101 | `16f64752f0cbf76a4aeeb46af64a260b7b91f51847e57d45d149016fd1d89dd8` |
| Segment 0002 manifest | 7,765 | `3090dd443df325e8fa43b5b3f7cd9bc84ab00640d99fad33dba04780bbb20b16` |
| Segment 0002 events | 1,492 | `27c557851a7daab98c53bb8d3711e9b1b36ae3a315d70a9fca128282fdb743c0` |
| Segment 0002 train log | 1,404 | `3a05edd7c7812a08aaf819b7567839a17f5494994d409fa47770ea825329ee1b` |
| Segment 0002 update log | 167 | `b2fb692cab7600071b7406dd61992df15aafe9f951785ce145d2d712923fb58d` |
| Segment 0002 latest checkpoint | 2,119,252 | `5403eb6d3210018c8d2da35698b78b1d6668bb7056b19c85e3bb95c206374c30` |
| Continuous SpecialistDB | 61,440 | `7db059da6ce7c49dc10938d0ab73e9ec22c6130280cacdb216a321085d41c210` |
| Segmented SpecialistDB | 61,440 | `f5d3ed87e8752befaad096d5dd958f70c4ae39cb8883aac3c44372821a7a281c` |

The three one-run authorities and the semantic-comparison authority are
consumed. The next gate is to freeze a managed, fresh Sanmill-refereed
long-run plan from this evidence. That plan must select a documented fixed
node schedule and advancement policy rather than inheriting the local GameAI
55% gate.

