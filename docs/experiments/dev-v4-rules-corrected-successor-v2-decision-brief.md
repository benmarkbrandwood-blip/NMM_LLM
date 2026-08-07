# Rules-Corrected Generalist Successor v2 — Decision Brief

## Status

`needs_decision`

This document recommends the next retained Generalist training objective and
resource envelope. It is not an immutable managed plan, an authorization
record, a launch command, or playing-strength evidence.

## Recommended Objective

Train one fresh rules-corrected v4-style Generalist successor under the final
MIF Suite 1.0 and `nmm-training-core@2` identities. The run should establish a
retained, exactly resumable learning trajectory that can later be evaluated
under a separately frozen protocol.

The objective does not include model promotion, publication, a claim that the
candidate is stronger than the completed baseline, or adoption of Sanmill as
the formal referee.

Recommended experiment ID:

```text
dev-v4-rules-corrected-successor-v2
```

The v2 suffix separates the retained long-run lineage from the completed
two-game `dev-v4-rules-corrected-successor-v1` smoke.

## Recommended Resource Envelope

| Product boundary | Recommendation |
| --- | ---: |
| Maximum games | 5,000 |
| Maximum active trainer time | 12 hours |
| Process segment | 250 games |
| Logical-ply truncation | 120 |
| Launch authority | separate decision after exact plan publication |

The 5,000-game budget gives the closest infrastructure and learning-trajectory
comparison with the completed corrected-v4 baseline. The 120-ply ceiling is
an experiment truncation, not a rules draw. It is long enough for the earliest
100-movement-ply no-progress window after ordinary placement, while keeping a
bounded successor comparable with the completed smoke. Captures can reset the
rule window, so some games may still end by experiment truncation; termination
reasons must remain separate in reporting.

## Wall-Time Evidence

The completed 5,000-game, 60-ply managed baseline records 20 completed
segments and 1.906941 hours of summed child-process elapsed time. Its median
250-game segment took 252.431 seconds and its slowest took 731.646 seconds.
The larger calendar interval included operator and host pauses and is not the
managed active-time measure.

Among the 1,825 historical per-game rows still present across that run's
segment logs, mean game length is 38.099 plies and 363 rows, or 19.8904%, stop
at the old 60-ply ceiling. The partial ledger does not represent every one of
the 5,000 counted games and cannot predict a 120-ply run precisely. It does
show that doubling the ceiling affects a material minority rather than proving
that every game doubles in cost.

The successor smoke completed two counted games in about 21.74 seconds of
trainer lifecycle time, but one was a confirmation rollout and the sample did
not exercise later curriculum levels. It remains a functional measurement,
not a throughput estimate.

A 12-hour active-time cap is approximately 6.29 times the completed baseline's
summed process time. It is therefore a conservative resource boundary for the
same 5,000-game objective while accommodating longer rollouts, rules checks,
later curriculum cost, exact-resume verification, and ordinary performance
variance. It is a stop boundary, not a duration promise. Exceeding it would
require a new product decision and a successor plan; the Agent must not expand
the limit automatically.

## Frozen Technical Recommendation

If the product boundary is approved, the Agent should generate a new managed
plan with these technical choices:

- fresh random initialization, with no `--resume` or automatic resume on the
  first segment;
- A2C, with PPO disabled;
- Sentinel, ValueNet, GapNet, S1A warm-start, RL imitation mixing, S1B
  refresher, and trainer-side opening forcing explicitly disabled;
- HumanDB frequencies and outcomes available, with its unversioned Malom
  columns masked;
- a separate new empty SpecialistDB carrying
  `malom_label_version=sector-corrected-v1` and no prior lineage;
- a new isolated managed control/output root, never either smoke output or the
  completed baseline root;
- one Windows process, one CUDA device, and `batch_games=1`;
- fixed seed 42;
- 50% frozen-target and 50% heuristic opponents, with target refresh every 50
  games and the existing adaptive difficulty ladder;
- 500,000 single-threaded native search nodes for each heuristic move and no
  wall-clock search limit;
- complete depth-5 rollout, no branch rollouts, and both primary and branch
  ceilings fixed to 120;
- temperature 0.90, linearly reaching 0.20 at 80% of total game progress;
- `latest.pt` and diagnostic publication every 50 games; and
- 250-game process boundaries followed only by explicit verified
  `exact-resume` from the immediately preceding `latest.pt`.

Managed-plan preparation now also carries `--no-opening-forcing` explicitly.
The omission found after the full-suite audit is fixed and covered by the
managed command-contract regression test; the fix must be remotely published
before a retained plan is created.

## Monitoring and Stop Policy

Every 50 games, record at least the termination-reason mix, difficulty and
games-at-level, realised heuristic depth and node use, temperature, entropy,
chosen probability, reward components, finite policy/value losses, and
checkpoint publication. At each 250-game boundary, verify:

- the append-only controller and child event chains;
- checkpoint payload, optimiser, RNG, trainer-state, and parent identity;
- exact game and update counters;
- SpecialistDB identity, label version, lineage, integrity, and growth;
- MIF, ruleset, Malom, HumanDB, path-config, Git, component, and experiment
  identities; and
- CUDA, CPU, RAM, disk, and segment elapsed-time evidence.

Do not stop because an intermediate win rate looks poor or favourable. Stop
and quarantine for non-finite values, repeated infrastructure failure, wrong
identity or label version, checkpoint or ledger corruption, unexpected
component activation, native fixed-work failure, CUDA failure, or an exhausted
game or wall-time envelope.

## Evidence and Remaining Gates

- The two-game successor smoke passed and performed one finite real Adam
  update from fresh weights.
- The current complete suite executed all 1,235 collected tests: 1,227 passed;
  the eight failures are the single known historical Sanmill installation
  identity condition repeated by eight local bridge tests.
- The managed-plan opening-forcing disable fix has 38 focused management and
  preflight tests passing. Applicable Ruff checks pass; four script-level E402
  findings are pre-existing import-bootstrap structure and are not introduced
  by the fix.
- MIF Suite 1.0 and `nmm-training-core@2` identities are final and already
  persisted by the trainer.
- Sanmill's local origin-counted referee fix is not a training prerequisite.
  Its publication and a new pinned bridge remain a separate evaluation gate.

Before an immutable plan can be prepared, the product owner must approve the
objective and the 5,000-game / 12-active-hour envelope, and the local managed-
plan fix must be pushed so the exact code is remotely recoverable. Plan
publication still will not authorize training. After the generated plan and
its SHA-256 are reviewed, launch requires a second explicit product decision
with no timeout-based default.
