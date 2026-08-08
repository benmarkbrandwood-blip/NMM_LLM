# Sanmill Node Ladder v1 Decision Brief

## Status and requested decision

Status: `needs_decision`

Recommendation: accept the five measured ceilings

```text
1,000 -> 5,000 -> 25,000 -> 100,000 -> 500,000 nodes
```

as `candidate_resource_ladder_v1` for a bounded no-update integrated-route
probe. This is not a request to put the ladder into training, select a
progression rule, launch the probe, or start a long run.

The recommendation is derived from the completed
[engine-only calibration](../evidence/sanmill-node-throughput-calibration-v1-result-2026-08-08.md),
whose raw report has SHA-256
`0d398c24a21c7a4537a70ed09d189d3e208ee41f495454e3e14fe41767dbeb2b`
and canonical report identity
`56549fe1344d483a90ecc2c94a60887b2b9c81cb4d3a00fb5f4c61d195ec43ea`.
No candidate model, trainer, optimiser, or training database participated in
that measurement.

## Why these five ceilings

The values form an approximately logarithmic sequence and are the complete
set of ceilings measured under the pinned production search contract. Keeping
the measured values avoids extrapolation above 500,000 nodes or interpolation
between untested levels.

The following completed-depth statistics use the seven non-empty fixed roots
in warm-process mode. Each root/ceiling result was identical across all nine
repetitions. Search latency is the published cross-root result over all eight
roots, including the intentionally depth-limited empty board.

| Ceiling | Non-empty completed-depth median | Range | Warm search median | Warm P90 |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 4 | 3-5 | 0.21 ms | 0.30 ms |
| 5,000 | 7 | 5-7 | 0.63 ms | 0.80 ms |
| 25,000 | 9 | 7-10 | 2.44 ms | 2.85 ms |
| 100,000 | 11 | 9-13 | 9.91 ms | 11.34 ms |
| 500,000 | 13 | 11-17 | 52.85 ms | 60.77 ms |

The fixed `moving-mid` root selected `d5-c5` at 1,000 nodes, `a4-a7` from
5,000 through 100,000 nodes, and `c4-c5` at 500,000 nodes. The other seven
roots retained their selected complete turn across the ladder. This limited
observation shows that additional work can change a decision, but it neither
ranks the moves nor establishes a strength difference.

The empty-board root used 52 nodes and completed depth 1 at every ceiling
because the retained `DrawOnHumanExperience` policy limits that phase. The
ladder is therefore a requested search-work ceiling, not a guarantee that
every turn consumes the ceiling or reaches a particular depth.

## Rejected interpretations

The five values must not be described as Sanmill skill levels or as five
opponents whose relative strength has already been measured. The calibration
contains no game outcomes. Completed depth is position-dependent, and the
opening-depth policy deliberately overrides the requested ceiling on the
empty board.

A wall-clock ladder is not proposed. Fixed time would make work depend on host
load and hardware and would weaken exact replay. A ceiling above 500,000 nodes
is also not proposed because it was not calibrated. Removing 5,000 or 25,000
nodes merely because both are cheap would discard measured depth separation
before the full route has been observed.

## Training and advancement boundary

The current Sanmill training preflight intentionally accepts only one fixed
work level: it requires `diff_max=1`, a one-member node ladder, and
`curriculum_advance_policy=disabled`. The five-level proposal cannot be
launched through the current trainer without a separate reviewed change.

The existing `legacy-score` gate is not suitable for adopting this ladder. It
mixes results from a changing learner with a training opponent, treats the
gate as evidence of superiority, and was not designed around the current
rules-based draw and truncation records. Nothing in the throughput result
calibrates its score target or probability threshold.

For the first retained Sanmill lineage, the safer hypothesis is a
deterministic resource curriculum whose exposure schedule is fixed before the
run and whose level changes do not claim that the learner has beaten the
previous level. Whether that schedule should use hard stages or a persisted
blend of adjacent ceilings remains deliberately undecided. It must be chosen
only after the integrated probe reports actual route cost and after the owner
selects the training-duration envelope.

## Next gate

If the owner accepts the five ceilings as the probe matrix, the next technical
step is to specify, implement, and preflight a separate no-update
integrated-route probe.
That probe must report each ceiling separately, preserve the production
Sanmill-refereed rollout path, and prove that model weights and all input
databases remain unchanged.

Acceptance of this brief does not authorize that probe. Its implementation,
readiness result, and one-run launch authority remain separate gates, with no
timeout or automatic default.
