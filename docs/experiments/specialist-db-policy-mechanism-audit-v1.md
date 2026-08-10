# SpecialistDB Policy-Mechanism Audit v1

## Status and authority

Status: `executed_coverage_negative`

Audit ID: `specialist-db-policy-mechanism-audit-v1`

This contract authorizes one read-only, no-update mechanism audit. It does not
authorize training, candidate-versus-baseline games, held-out evaluation,
promotion, publication, database rewriting or checkpoint generation.

## Question

The completed retained-v3 run ended with a high draw fraction, low scheduled
entropy and a cumulative SpecialistDB that pooled empirical observations from
both opponent sources, all five Sanmill node stages and every earlier model
state. The curve alone cannot determine whether empirical SpecialistDB
features materially affect the final policy.

This audit asks the narrower functional question:

> Holding the final checkpoint, frozen target, position set, HumanDB, Malom,
> encoder and lookahead route fixed, how much does the final policy change when
> empirical SpecialistDB evidence is projected differently at inference?

It does not estimate the causal effect of having trained with the database and
does not measure playing strength.

## Immutable inputs

| Input | Identity |
| --- | --- |
| Candidate checkpoint | `managed-sanmill-preserving-retained-v3-seed58/segments/segment-0020/latest.pt` |
| Checkpoint SHA-256 | `28e8af274f4fc9dd7e00ce4f7be884c855354218c796888f1c1ab81a4cdc9fa7` |
| Checkpoint ID | `managed-sanmill-preserving-retained-v3-seed58-segment-0020:checkpoint:00000006` |
| Experiment ID | `dev-v4-sanmill-preserving-retained-v3-seed58` |
| Final game count | 5,000 |
| SpecialistDB | `data/specialist_db.sanmill_preserving_retained_v3.seed58.sqlite` |
| SpecialistDB SHA-256 | `82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe` |
| SpecialistDB label version | `sector-corrected-v1` |
| Development corpus | `docs/experiments/dev-v4-phase-covered-corpus-v1.json` |
| Corpus SHA-256 | `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e` |
| Corrected Malom manifest | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| HumanDB identity | `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31` |

The audit must reconstruct the retained-v3 training encoder: no Sentinel,
ValueNet or GapNet; the final frozen target; 12-ply lookahead width;
`sim_ply_depth=5`; HumanDB frequencies and outcomes with historical Malom
columns masked; strict corrected Malom; and SpecialistDB minimum empirical
support of three samples.

The 64-position corpus contains 22 placement, 21 movement and 21 flying
positions, including 29 Malom-critical positions. It is inspected development
data and is intentionally used only for a mechanism diagnosis. It must not be
renamed or reused as independent held-out promotion evidence.

## Fixed projection modes

For every legal complete logical turn, query the same post-move database row
once and derive four deterministic projections:

1. `full`: reproduce the trainer's current compatibility projection. Use the
   empirical W/D/L distribution at three or more samples; otherwise use a
   trusted Malom W/D/L prior when present; otherwise report no hit.
2. `empirical_disabled`: ignore empirical counts and expose only the trusted
   Malom prior. This is the Malom-only projection.
3. `malom_disabled`: ignore the theoretical label and expose only an empirical
   distribution with at least three samples.
4. `all_disabled`: expose no SpecialistDB result.

No mode may change the SQLite file, model weights, target weights, RNG state,
corpus, lookahead advisor or fallback Malom decoder. The encoder's existing
strict fail-closed behaviour remains active.

## Required record

The report must include:

- exact repository, checkpoint, database, corpus, HumanDB and Malom identities;
- SHA-256 of checkpoint and SpecialistDB before and after execution;
- absence of SQLite WAL, SHM and rollback-journal sidecars before and after;
- every position and legal action in stable order;
- per-action post-move theoretical label, empirical W/D/L counts, sample count,
  empirical distribution, availability in each projection and whether the
  empirical modal label disagrees with the theoretical label;
- for every projection, feature-matrix SHA-256, logits, temperature-1 and final
  scheduled-temperature probabilities, argmax action, Malom action quality and
  SpecialistDB hit count;
- aggregate and per-phase action changes, probability distances, database
  coverage and Malom-preservation changes; and
- an explicit material-sensitivity decision under the frozen rule below.

Non-finite features, logits or probabilities, changed legal-action order,
identity drift, a database sidecar, a file-hash change, unavailable required
data or an output path that already exists is a fatal audit failure.

## Preregistered material-sensitivity rule

The primary contrast is `full` versus `empirical_disabled`. Empirical
SpecialistDB evidence is `material` if any one of the following is true across
the fixed 64 positions:

1. at least three policy argmax actions change;
2. at least one of the 29 critical positions changes between a Malom
   value-preserving and a value-downgrading argmax in either direction; or
3. the mean total-variation distance between final-temperature action
   distributions is at least 0.05.

These thresholds are mechanism triggers, not significance tests and not
strength gates. If none is reached, the result is `not_material_on_fixed_corpus`;
that does not prove the database harmless or irrelevant elsewhere.

## Decision routing

If the result is `material`, freeze but do not launch a three-seed,
single-factor calibration comparing the existing full cumulative projection
with empirical reads disabled. All other trainer, data, opponent, seed,
resource and update settings must be paired. Each arm must use its own fresh
isolated SpecialistDB so no arm can contaminate another.

If the result is `not_material_on_fixed_corpus`, do not launch that calibration
solely because of this audit. Return to the competing temperature, opponent,
frozen-target and tactical-conversion hypotheses.

Either result remains development evidence. A future promotion claim requires
a separately frozen, candidate-blind held-out evaluation.

## Result

The immutable report and its interpretation are recorded in:

- `docs/evidence/specialist-db-policy-mechanism-audit-v1-2026-08-10.json`
- `docs/evidence/specialist-db-policy-mechanism-audit-v1-result-2026-08-10.md`

The report's preregistered machine decision is
`not_material_on_fixed_corpus`, but none of the 1,583 legal successors in the
fixed corpus had a usable SpecialistDB projection. The result is therefore a
coverage-negative, uninformative mechanism result. It does not close the
SpecialistDB hypothesis or authorize the conditional calibration.
