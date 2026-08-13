# Mature target-refresh replication v1 attempt 002 result

Date: 13 August 2026

Verdict: `no_replicated_material_effect`

The one-shot replication sequence completed all six managed arms and the
predeclared 288-game no-update direct comparison. It did not authorize or
start held-out evaluation, promotion, publication, retained training, or long
training.

## Bound identities

| Artefact | Identity | File SHA-256 |
| --- | --- | --- |
| Plan | `85ad0b99093bc7e81ac6057b92abd8a38cafdc03893b6b46c760dffd3fa5acca` | contract `0210bb93b363f2f48fefb37b360ce9798e77835da2ee3be1ba133b6bca3357b8` |
| Readiness | `7088c1f5b030ca5e39397400e4c252a033dedc6e3ec5e06a2fd7c0e8e1028af0` | `6a7c8a8f27f160c0d7a4b5dfd1d674fc3f187c5a3384ce78e69b5c29283833d8` |
| Parent authorization | `b7384d76b9b906720263b0a8c3d566b8525f57e2118918d85626931afff5d571` | `563dfbbfe2db1d81d470f631ab902a375e8a8d93c1d79c78ee2da6457d43cf6b` |
| Launch | `c71c2ccbe715dcd64aa35197c01acad8bffccb9d26262c17285d42eab50a5508` | `0a169ddfb5c615f6ecba6db1bdcf89442038b50270f0d944d035a1236b128a0c` |
| Completion | `94648404b20a4aba78b2eecc7e53096336f3c0c55c8bdf22ad8fe355a5831d37` | `ba04914ff5e35767f90ab2d7c91308b23683467d1ce2a8725505a8037882c4ff` |
| Development ledger | 288 rows | `e6036cae4c310dd00d53095ba38aafa3e7518864181bfdafed6a314dc8d67cd0` |
| Result | `8559fa7b5720f3b3a2b8b41743ada612a89aef8d37ad191fa2737d4673185142` | `0197ca419b4a5681097027270616e06b7c0b89e737777f60624c8ddbc77a268e` |

The source for training and analysis was clean, synchronized `dev` commit
`8179e8e78398c8f25a97e5058d2b6c9deea6caf4`. There is no sequence failure
record.

## Resource accounting

All six arms stopped at exactly 8,192 post-mature-fork transitions, or 49,152
in total. They consumed 2,324 new training games and 768 fixed 64-transition
A2C updates. Managed training used 0.3861 active hours; the complete sequence,
including read-only analysis, used 0.426902 hours. Requested Sanmill work was
16,450,000 nodes, below the 172,800,000 ceiling.

| Seed | Refresh games | Stale games | Updates per arm | Policy-health result |
| --- | ---: | ---: | ---: | --- |
| 64 | 528 | 328 | 128 | both passed |
| 65 | 410 | 396 | 128 | both passed |
| 66 | 529 | 285 | 128 | both passed |

Every arm retained direct value preservation `1.0`, candidate preservation
`1.0`, and a candidate preserving-logit margin above the frozen `-0.10`
minimum. Same-seed temperature and learning-rate exposure was byte-identical.

The development reporter then loaded the candidates on CPU and completed 288
colour-swapped games. It performed zero optimizer updates, training games,
checkpoint writes, or database writes. The observed Malom tablebase and
HumanDB files were unchanged across the read-only measurement.

## Preregistered decision

For the disjoint replication cohort, `refresh-mature minus stale-control` was
`+0.1215278`, above the `1/12` aggregate boundary. Seed effects were:

- seed 64: `+0.3229167`;
- seed 65: `+0.0833333`; and
- seed 66: `-0.0416667`.

Seeds 64 and 65 met the per-seed refresh-support boundary, the replication
truncation rate was `0.118056`, and the cohort classification was therefore
`material_mature_refresh_direct_effect`.

The independent seeds 67--69 cohort had previously produced a mean effect of
`-0.0763889`. Under the frozen pooled six-seed rule, the combined effect was
only `+0.0225694`, with refresh-supporting seeds 64 and 65 and one opposite
seed, 67. It failed both the pooled `1/12` effect threshold and the minimum of
three supporting seeds. Pooled truncation was safe at `0.107639`.

The final preregistered classification is consequently
`no_replicated_material_effect`, with
`selected_successor_condition=null` and
`automatic_long_run_selection=false`.

## Interpretation boundary

### Observed facts

The two independent cohorts point in opposite directions. The replication
cohort has a material direct effect favouring one mature refresh, while the
pooled six-seed estimate is small and seed-heterogeneous. Policy distributions
also diverged persistently for replication seeds 65 and 66, but this mechanism
evidence cannot replace the pooled direct gate.

### Hypothesis

One frozen-target refresh can materially change a trajectory, but its sign is
seed-dependent at the tested mature boundary. The evidence does not establish
a generally beneficial refresh or a generally beneficial permanently stale
target.

### Supporting evidence

The arms shared same-seed mature forks, exact transition counts, fixed learning
rate, transition-indexed temperature, fixed 1,000-node Sanmill work, and common
random streams for the direct comparison. This isolates the one-time target
intervention within each seed.

### Counterevidence

The cohort directions disagree, seed 66 weakly opposes the replication mean,
and refresh colour was strongly asymmetric in the development games. Training
W/D/L also used each arm's endogenous opponent and is not a transferable
strength comparison.

### Next validation experiment

Do not lower the thresholds, select a cadence from one cohort, or restore the
old every-50-game refresh as though it had been validated. A next retained run
must be a separately frozen research plan. Its target schedule is a new plan
choice, not an automatic consequence of this result, and it requires its own
technical readiness and launch gate.
