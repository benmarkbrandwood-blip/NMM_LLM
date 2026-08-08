# Sanmill Node-Throughput Calibration v1 Result — 8 August 2026

## Decision

`passed_engine_throughput_evidence`

The separately authorised engine-only calibration completed from clean,
published `dev` commit
`723494b4c347080e605bb0812655a2b5c0c3b2ce`. It produced all 720 scheduled
search samples, retained nine repetitions in every one of 80 cells, and
published one atomic local result. The active Windows power scheme did not
change during the run.

This result measures the pinned Sanmill process on eight fixed complete-history
roots. It does not load a model, trainer, optimiser, checkpoint, HumanDB,
SpecialistDB, Malom database, Perfect DB, or opening book. It is not strength
evidence, an end-to-end games-per-hour measurement, a node-ladder decision, an
advancement rule, or authority for an integrated probe, training smoke, or long
run.

## Frozen execution identity

| Field | Value |
| --- | --- |
| Run ID | `sanmill-node-throughput-calibration-v1-20260808-001` |
| NMM_LLM commit | `723494b4c347080e605bb0812655a2b5c0c3b2ce`; clean; `dev == origin/dev` |
| Plan identity | `2dff4e6d37f36af90d9e90943dad8f4bcccbec802615f7eedc1103af32a51290` |
| Plan raw SHA-256 | `2fc47ac72737c30e507eaee47554a70ee7c745ba247d871b57f506a6494e74ca` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee profile | `mif-stable-moving-v1` |
| Referee semantic digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| Rules identity | `3e62cb93a1e0afe4534ce4824d233344816050b547bb8761dd7fe985d8ad399f` |
| Report identity | `56549fe1344d483a90ecc2c94a60887b2b9c81cb4d3a00fb5f4c61d195ec43ea` |
| Raw result SHA-256 | `0d398c24a21c7a4537a70ed09d189d3e208ee41f495454e3e14fe41767dbeb2b` |
| Raw result size | 1,143,676 bytes |

The raw report remains ignored at
`out/diagnostics/sanmill-node-throughput-calibration-v1-20260808-001.json`.
It includes host-local invocation metadata and is therefore not copied into
the portable repository. Its content identity and file identity above bind it
to this record. Independent recomputation after the run removed the
`report_identity` member, canonicalised the remaining report, and reproduced
the claimed report identity exactly.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_sanmill_nodes.py `
  --launch calibration `
  --plan docs\experiments\sanmill-node-throughput-calibration-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-node-throughput-calibration-v1-20260808-001 `
  --output out\diagnostics\sanmill-node-throughput-calibration-v1-20260808-001.json
```

The preflight immediately before launch returned
`ready_for_authorized_calibration`. It verified the published source, pinned
runtime, and all eight fixture identities without performing a timed search.
The owner then supplied the separate one-run launch authority. That authority
is consumed.

## Completed work

The run started at `2026-08-08T03:12:13.303184Z`, completed at
`2026-08-08T03:12:51.212713Z`, and recorded 37.846903 seconds of bounded wall
time.

| Field | Result |
| --- | ---: |
| Fixed roots | 8 |
| Node ceilings | 1,000; 5,000; 25,000; 100,000; 500,000 |
| Modes | cold process; warm sequence |
| Repetitions per mode/root/ceiling cell | 9 |
| Complete cells | 80 / 80 |
| Timed searches | 720 / 720 |
| Sanmill process launches | 405 / 405 maximum |
| Requested node ceilings | 90,864,000 |
| Actual nodes | 79,510,680 |
| Aggregate ceiling utilisation | 87.505% |
| Sum of reported search time | 8.339422 seconds |
| Compound primary-plus-removal samples | 180 |

No cell changed its semantic logical-turn result across its nine repetitions.
All 40 matching root/ceiling cells also returned the same semantic result in
cold and warm modes. Every search left its authoritative root unchanged.

## Fixed-work search measurements

Each row aggregates 72 raw samples: eight roots times nine repetitions. Search
time excludes process startup and history replay. Percentiles use the frozen
nearest-rank method.

| Mode | Ceiling | Median | P90 | Median actual nodes | Median utilisation | Median throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 1,000 | 0.260 ms | 0.390 ms | 1,000 | 100% | 3.620 Mnodes/s |
| cold | 5,000 | 0.740 ms | 1.010 ms | 5,000 | 100% | 6.020 Mnodes/s |
| cold | 25,000 | 2.910 ms | 3.410 ms | 25,000 | 100% | 8.090 Mnodes/s |
| cold | 100,000 | 10.310 ms | 11.750 ms | 100,000 | 100% | 9.070 Mnodes/s |
| cold | 500,000 | 52.860 ms | 60.730 ms | 500,000 | 100% | 9.230 Mnodes/s |
| warm | 1,000 | 0.210 ms | 0.300 ms | 1,000 | 100% | 4.040 Mnodes/s |
| warm | 5,000 | 0.630 ms | 0.800 ms | 5,000 | 100% | 7.300 Mnodes/s |
| warm | 25,000 | 2.440 ms | 2.850 ms | 25,000 | 100% | 9.700 Mnodes/s |
| warm | 100,000 | 9.910 ms | 11.340 ms | 100,000 | 100% | 9.480 Mnodes/s |
| warm | 500,000 | 52.850 ms | 60.770 ms | 500,000 | 100% | 9.220 Mnodes/s |

The median actual-node column is 100% because seven of eight roots use the
entire ceiling. The empty-board root is intentionally phase-depth limited by
`DrawOnHumanExperience`: it uses exactly 52 nodes at every requested ceiling.
Its utilisation therefore falls from 5.20% at 1,000 nodes to 0.0104% at
500,000 nodes. This one root accounts for effectively all of the difference
between requested and actual aggregate nodes.

At the 500,000-node ceiling, the warm persistent-process medians were:

| Root | Median | P90 | Actual nodes | Completed depth | Search calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| placement empty | 0.120 ms | 0.150 ms | 52 | 1 | 1 |
| placement mid | 40.840 ms | 43.400 ms | 500,000 | 11 | 12 |
| placement last | 52.640 ms | 56.290 ms | 500,000 | 16 | 17 |
| movement initial | 53.880 ms | 55.210 ms | 500,000 | 17 | 18 |
| movement mid | 53.930 ms | 56.620 ms | 500,000 | 15 | 16 |
| movement reduced | 41.210 ms | 43.150 ms | 500,000 | 13 | 14 |
| flying Black | 60.770 ms | 65.800 ms | 500,000 | 12 | 13 |
| compound-capable | 60.340 ms | 65.080 ms | 500,000 | 11 | 12 |

Both `movement-reduced` and `compound-capable` selected a complete
Mill-plus-removal turn at every ceiling in both modes. The aggregate budget
and node-accounting checks passed for all 180 such samples.

## Process and replay overhead

Starting and configuring a fresh process had median cost 59.260 ms in cold
mode and 59.690 ms in warm-sequence mode, with nearest-rank P90 values 64.450
ms and 64.750 ms. Hash reset had median cost 0.050 ms. Per-root complete
history replay medians ranged from 0.540 to 0.780 ms in cold mode and from
0.340 to 0.550 ms for non-empty warm roots.

Process startup therefore dominates the 1,000- to 25,000-node search itself.
The current training architecture's persistent process is not merely an
optimisation at those levels; it is required for representative throughput.
At 500,000 nodes, cold and warm search medians converge, so the retained
benefit is mainly startup amortisation rather than a different search result
or a large transposition-table speedup.

## Interpretation boundary

The evidence supports these engineering conclusions:

1. non-empty placement, movement, flying, and compound roots consume their
   fixed node ceiling predictably on this host;
2. empty-board work is controlled by the retained phase-depth policy rather
   than by high node ceilings;
3. a persistent Sanmill process must be measured in the actual training route;
4. 500,000-node non-empty searches have per-turn warm medians of about 41 to
   61 ms and per-root P90 values no higher than 65.8 ms in this corpus; and
5. the fixed-position engine survey is sufficiently stable to design a
   bounded no-update integrated-route probe.

It does not support extrapolating complete games per hour. It excludes learner
policy inference, lookahead feature work, Sanmill authoritative application of
both sides' turns, Python conversion, Malom queries, HumanDB reads,
SpecialistDB writes, logging, optimiser work, checkpoint publication, and
game-length variation. Strength and advancement remain completely unmeasured.

## Remaining gate

The current long-run verdict remains `needs_decision`. A proposed node ladder
must be treated as a resource-controlled curriculum hypothesis, not a strength
scale. Before it can enter a retained plan, one separately authorised
no-update integrated-route probe must measure complete Sanmill-refereed games
at the proposed levels while preserving the exact model, data, rule, and
process boundaries and proving that no optimiser, checkpoint, or mutable
training-database work occurs.
