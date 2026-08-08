# Sanmill Corrected Exact-Resume Parity v3 Result — 8 August 2026

## Decision

`passed_exact_resume_parity_on_corrected_source`

At clean published commit
`97b3347c13f790404927645f77b272abe7db3ed2`, an uninterrupted two-game
route and the same two games split across fresh and exact-resume processes
completed successfully. All three staged read-only preflights returned
`ready_for_smoke`, `errors=[]`, and `unresolved_decisions=[]`.

The repository semantic verifier returned:

```json
{"checkpoint_fields":["model_state","optimizer_state","scheduler_state","scaler_state","rng_state","trainer_state","data_state"],"database_tables":["positions","preferred_plays","winning_lines","meta"],"log_records":2,"status":"passed"}
```

This closes the process-boundary continuation gate for the corrected value
bootstrap and frozen-opponent feature route. It is exact only for the tested
seed-42, `batch_games=1`, 1,000-node, two-game domain. It is not strength
evidence and none of the v3 artefacts may enter retained training lineage.

## Update equality

Each route produced the same two finite periodic updates. Neither relied on a
final-flush-only update.

| Game | Batch steps | Policy loss | Value loss | Entropy | LR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21 | 0.7583574057 | 0.4042743444 | 1.9286332130 | 0.0001 |
| 2 | 26 | -0.2973993719 | 0.3758915961 | 2.9488308430 | 0.00005 |

The second process reported exact restoration at game 1, batch 1, and update
1 before producing the identical second record. The verifier compared the
new pending-step `bootstrap_perspective` as part of normalized trainer state;
it did not omit or special-case that field.

## Checkpoint integrity

All checkpoint envelopes passed payload verification:

| Checkpoint | Checkpoint ID | Payload SHA-256 | File SHA-256 |
| --- | --- | --- | --- |
| Continuous | `sanmill-corrected-exact-resume-parity-v3-continuous:checkpoint:00000003` | `705aabc3e9ab11ad280bda423abd490aaf313e200dcb478159da42efe7ebba93` | `48069b74347518de8b9dbc9533b481d23d7c0c51aeff17c80e3671666da18efb` |
| Segment 0001 | `sanmill-corrected-exact-resume-parity-v3-segment-0001:checkpoint:00000002` | `2234ae7402caae8f4069b4a37cb2fa15345214cc3325904032a26754d771a0e2` | `8fc4996472c9a00d012fbb3855b1fa8f23356ee95ab3f59b60e44c25e11dd263` |
| Segment 0002 | `sanmill-corrected-exact-resume-parity-v3-segment-0002:checkpoint:00000002` | `2974f79e721a5ce28d8c770f2614804ce5ecb030936b827898986d3eef7de194` | `01aeec55ec1f70ea590fe7720bc0832ed975507c4cfe1a8e6e198acfdacabfa8` |

Both databases pass the trusted `sector-corrected-v1` contract and contain
the same 94 positions, 20 Malom labels, no winning lines, and no preferred
plays. Their raw hashes differ only because their required lineage-root
metadata differs; semantic table comparison passed:

- continuous SHA-256
  `97526831cd81d8f51b5215e43a7b4d21018d9522de51fe20a20c37b6637f586f`;
- segmented SHA-256
  `d1a22d296446cfcedf822843125382031f23d375e5bde6844a66f088263bac31`.

## Persisted local artefacts

| Artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| Continuous manifest | 7,190 | `c2e35e2d9e88138c3a329436fa89ef4b94f923c08be6c03dd14f0f6b92cf1a7b` |
| Continuous events | 1,489 | `aade65247ad73d0d0ad9c7fcff9e469807b549c42d7c6e768e9c5a8d940e8746` |
| Continuous train log | 2,653 | `de38ee71af21c29d5af913b21d1fdcfc9289299915f04799f73dea3a2858319c` |
| Continuous update log | 335 | `49a8d8c8324b809805ff10bd8ef8a5debbfddf5434e8a337e6ff0c185ae1f7ff` |
| Segment 0001 manifest | 7,199 | `214cee2055ee77804b5d46528e60c14a2b2e5815fca5b8e7861c70f34860a16d` |
| Segment 0001 events | 1,495 | `ef78a01d3b7b5799ab3735cb12244fab0c3538782432249ce9af0eeea3761220` |
| Segment 0001 train log | 1,309 | `a331ed970c5164f1ff61140be099471ef662ecf93af96dd29039c1f95d684c58` |
| Segment 0001 update log | 167 | `ac000e3cebcd19725bda8cfdad1e4b6124f3bc269dcd64224f19e124b40b417e` |
| Segment 0002 manifest | 7,833 | `05b80559b1986cdee3ed2ad102196ff9e949762d4d5dfd7b94ba17f621abf379` |
| Segment 0002 events | 1,495 | `b927c92505dcf7ea676c14413e98c30140bae4c9358ae8d3aa5e04382b21a4e2` |
| Segment 0002 train log | 1,407 | `e7c0ba7deea65b00ecba5633b056d7bf81f1fe42955350096128794b67060337` |
| Segment 0002 update log | 168 | `dd8693d83b6ba7e07695314c16c9d394245cbec6941b387594598ee14aeaad86` |

## Consequence

All three launch authorities and the comparison authority are consumed. The
remaining engineering gate before a retained run is automatic fixed-state
policy-health quarantine after every managed segment. It must invoke the
committed read-only audit, bind each report to the exact checkpoint/database
identity, and stop before constructing the next segment if the frozen
critical-state limits fail.
