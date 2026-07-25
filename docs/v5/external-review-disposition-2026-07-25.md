# External v5 Review Disposition — 25 July 2026

Status: completed design-review disposition.

## Review Boundary

The reviewer read the former monolithic v5 plan without the repository code,
handover, local asset contracts, or later experiment evidence. The review is
therefore valuable as an independent design challenge, but its implementation
claims require repository verification.

The repository checks used for this disposition were:

- current `dev` and refreshed `origin/main` commit graphs;
- [`windows-training-2026-07-20.md`](../handoff/windows-training-2026-07-20.md);
- [`local-training-layout.md`](../local-training-layout.md);
- current Malom and compact `GameEngine` code;
- the corrected-v4 experiment and Stage-0 result;
- the pinned Sanmill strict-bridge v2 result.

The previous monolith is retained at
[`v5-specialist-plan-legacy-2026-07-25.md`](../v5-specialist-plan-legacy-2026-07-25.md).
The current governing entry point is
[`v5-specialist-plan.md`](../v5-specialist-plan.md).

## Factual Reconciliation

### Current decoder

The archived document's statement that the current decoder still ignores the
sector value is stale. Current `ai/malom_db.py` computes
`absolute_key1 = raw_key1 + sector_value` and maps only the virtual extrema to
W/L. The correction and focused provenance tests are accepted. The reviewer's
warning remains useful historically and for requiring a complete comparator,
but it is not a current blocker named “fix the coarse decoder.”

### Current rules engine

The concern about incomplete repetition/no-progress history remains valid.
The compact in-repository engine recognises a fixed oscillation pattern rather
than general state multiplicity and increments a post-placement counter without
the intended complete capture-reset semantics. It cannot be the formal referee
until E0 closes. The pinned Sanmill bridge is currently the independently
validated rule/history integration reference, not a candidate evaluator.

### v4 lineage

The review correctly identifies that pre-correction labels can affect
trajectories, data, recovery, and checkpoint selection. It overgeneralises by
placing all v4 artifacts in one contaminated class. The managed corrected-v4
baseline was fresh-initialised, isolated, and did not load legacy learned
components. It remains eligible as a control or candidate source, but its
completed Stage-0 result is not promotion evidence.

## Disposition Key

- **Accept**: the design problem is valid and the governing plan changes.
- **Accept with qualification**: the core issue is valid, but the review's
  scope or proposed remedy conflicts with current evidence or needs a narrower
  condition.
- **Already partly covered; strengthen**: the old plan noticed the issue but
  buried or incompletely resolved it.

## Findings 1–11: Route and Product Semantics

| # | Disposition | Evidence-based change |
| ---: | --- | --- |
| 1 | **Accept** | Replaced “play like a mature human” as the core optimisation target with three separate objectives: claimed safety/correctness, target-population match result, then optional blinded style. Human-like wording now requires independent style evidence. |
| 2 | **Accept** | D-only specialisation now uses natural-match superiority plus loss and W-conversion non-inferiority against the same-stack compact reference. W-conversion superiority is activated only by an explicit W-specialisation objective. |
| 3 | **Accept** | Oracle non-regression and finite-horizon compact play are explicitly different games. Product modes are now `ordinary_always_move`, `verified_compact`, and `oracle_optional`, each with distinct claims. The architecture comparison precedes the default decision. |
| 4 | **Accept** | Added explicit `oracle_policy`, `verified_runtime_policy`, and optional `ordinary_policy`. A single verified policy may use only `A_allow ∩ A_runtime`; an empty intersection is unavailable. Runtime-only horizon actions require their own supervision. |
| 5 | **Accept** | Positional pack W no longer impersonates full-history liveness. F0-P must establish the minimal sufficient history state and proof feasibility. A pack W action needs an appropriate liveness/conversion certificate for a forced-win claim. |
| 6 | **Accept** | Runtime proof availability, proof cost, pack closure, future availability, and support closure are deployment-aware secondary objectives among oracle-equivalent actions. They are no longer only post hoc exposure diagnostics. |
| 7 | **Accept** | D teaching is conversion-aware. `P(certW)` remains the first tier, while conversion under the frozen W/runtime path, proof rank, and runtime feasibility break statistically indistinguishable first-tier candidates. |
| 8 | **Already partly covered; strengthen** | The old plan had P0-D0 and power screens, but after substantial structure and without a direct global-effect upper bound. F0-H now runs first and can stop all complex human-target work. |
| 9 | **Accept** | F0-P now precedes W-policy dependence and measures minimal history state, graph/SCC size, cache reuse, proof artifacts, memory, and cold/warm completion on natural/cyclic/near-threshold W. |
| 10 | **Accept** | Strict fail-closed is no longer presumed to be the ordinary product default. Always-move, verified, local/full-DB, server, and hybrid forms are compared. Ordinary mode must use an explicit deterministic fallback and make no safety claim. |
| 11 | **Accept with qualification** | Every possibly contaminated legacy checkpoint/optimiser/replay/DB remains shadow-only for release lineage. A clean from-scratch corrected-v4 run is not contaminated merely because it uses v4 architecture and remains a valid control/candidate source. |

## Findings 12–28: Human Modelling and Training

| # | Disposition | Evidence-based change |
| ---: | --- | --- |
| 12 | **Accept** | Human simulation fixes one latent style or ensemble member for the entire rollout. A memoryless average-person rollout is an explicit ablation only. |
| 13 | **Accept with qualification** | Existing HumanDB may support coarse choice modelling, but not assumed high-dimensional state-level guarantees. The first HumanPolicy is broad-Elo, one-step/coarse-structure, short-horizon, and abstaining. F0-H decides actual support. |
| 14 | **Accept** | Every weighting scheme declares whether it targets an average unique player, natural game, traffic-weighted decision, or prospective trial. Player/game normalisation is no longer called neutral. |
| 15 | **Accept** | Oracle/Strategy may enforce proven D4 equivariance. HumanPolicy retains raw UI orientation and colour; D4/colour augmentation is an ablation because human behaviour may not be symmetric. |
| 16 | **Accept** | Replaced ambiguous `safe_mass` with `positional_safe_mass`, `full_rule_allow_mass`, and `runtime_authorisable_mass(X_rt)`. L quantities require explicit horizon terminology. |
| 17 | **Accept** | A trap requires a confidence lower bound above a nonzero probability floor, independent-player/support floors, and OOD/perturbation stability. Softmax nonzero mass is not evidence. |
| 18 | **Accept** | F0-H/T0 directly compare ultra-strong reference, coarse WDL/tie policies, one-step human ranking, and short-horizon teacher value. Full `T_full` is not built unless residual headroom is measurable. |
| 19 | **Accept with qualification** | Verified theory-preserving mode intentionally prohibits D→L. A separately named ordinary/practical-risk mode may optimise measured human results without a safety claim. The plan no longer assumes a large gain is compatible with strict non-regression. |
| 20 | **Accept** | L ordering first reaches a minimum survival threshold, then prioritises draw/comeback rescue, and uses extra delay only secondarily. Purposeless extension is not success. |
| 21 | **Accept** | AB depth/Elo mixtures move to experiment configuration and require sequence-level calibration. Fixed intuitive ratios are not governing facts. |
| 22 | **Accept** | Removed governing sample-percentage tables from the active plan. It retains natural floors, safety coverage, stress/source caps, and requires bounded experiments to select exact ratios. |
| 23 | **Already partly covered; strengthen** | The old plan made most heads optional but still described a large system. T1 now permits only one HumanPolicy, direct teacher, and one policy-only StrategyPolicy. Gap/Self/DAgger/PPO open only for a diagnosed bottleneck. |
| 24 | **Accept** | Confidence bounds are no longer converted by default into policy probabilities. Start with top groups and supported pairwise/listwise relations; calibrate any mass mapping on held-out teacher utility before adoption. |
| 25 | **Accept** | A deterministic algebraic contradiction fails the job. Finite-sample cyclic preference merges its strongly connected component into unknown/tie or causes state abstention; it is not automatically a fatal safety fault. |
| 26 | **Accept** | Allows preregistered stratification, self-normalisation, truncation with bias bounds, control variates, or doubly robust methods. Reports include ESS, variance, max share, and stabilisation bias. |
| 27 | **Accept with qualification** | Same-teacher DAgger cannot correct teacher misspecification. Any proposed round needs independent search/opponent/shadow states and a measured ROI. The old fixed “two rounds” is removed as standing entitlement. |
| 28 | **Accept** | Added explicit target ranges, NaN/Inf gates, per-head gradient norms, weight/mask diagnostics, mixed-precision rules, extreme-input tests, optimiser/RNG/data-cursor persistence, and exact-resume output equality. |

## Findings 29–34: Statistics and Human Evaluation

| # | Disposition | Evidence-based change |
| ---: | --- | --- |
| 29 | **Accept** | Separates release margin `m`, planning alternative `d_plan`, and MDE. If `LCB > m`, power uses `d_plan - m` with `d_plan > m`; a test against zero cannot justify a positive-margin gate. |
| 30 | **Accept** | Default natural-match primary result is match score, with independent loss-rate non-inferiority. Win rate remains key secondary unless preregistered as primary with coherent power. |
| 31 | **Accept** | W-start states are weighted by natural certW entry under current and compact-reference controls, stratified by phase/proof/history/pack route, and deduplicated to avoid puzzle learning. |
| 32 | **Already partly covered; strengthen** | The old plan acknowledged the audit was same-source but still used “independent proxy” language. It is now `independent_model_same_source_proxy`; only new/different-source humans provide external confirmation. |
| 33 | **Accept** | Split human evaluation into latency-matched strategy efficacy and real-latency product effect. Each supports only its declared estimand. |
| 34 | **Accept** | Release gates are compressed into implementation correctness, risk/availability, match result, one primary product effect, and target-device resource/claims. The larger metric panel is diagnostic. |

## Findings 35–38: Runtime and Architecture

| # | Disposition | Evidence-based change |
| ---: | --- | --- |
| 35 | **Accept** | Exact-pack use is per candidate: current exact state plus exact successor yields `A_pack_exact`; unresolved candidates use the prover; the final pool is the per-action union. L still needs survival proof. |
| 36 | **Accept** | Early stop cannot use policy raw mass alone. It also uses independent bounds/prior, top unresolved status, unresolved utility bounds, and a frozen random audit of low-mass actions. |
| 37 | **Accept** | Deleted authority of 50 MB/512 MB planning defaults. Budgets follow a frozen deployment form and representative target device. |
| 38 | **Accept** | F0-A compares local compact, pack, optional full DB, server oracle, hybrid, and ordinary always-move total cost. This does not preselect full DB; it removes the unsupported assumption that it can never be a release option. |

## Findings 39–46: Structure and Governance

| # | Disposition | Evidence-based change |
| ---: | --- | --- |
| 39 | **Accept** | Replaced the monolithic governing file with a short entry point plus six owning specifications. The old file is retained only as a dated historical snapshot. |
| 40 | **Accept** | Active dependencies use F0 feasibility, E0/E1 engineering, T0/T1 training, and R0 release. The archived overlapping Section/Stage/P numbering is not active. |
| 41 | **Accept** | Read-only F0 feasibility runs before comprehensive governance. It cannot train, label, modify databases, or create release evidence. |
| 42 | **Accept** | Heterogeneous review remains mandatory for critical rules/oracle/gold/label/objective/authorisation/release work. Standard reversible work may use self-review plus independent CI; analysis-only uses a compact audit card. |
| 43 | **Accept** | Added a preapproved automatic-repair boundary limited to failing-test-driven, same-module, non-semantic, bounded changes with mandatory reruns. |
| 44 | **Accept** | Replaced overloaded `fatal_stop` language with `input_reject`, `job_fail`, `experiment_block`, `release_block`, `runtime_unavailable`, and `fatal_safety_fault`. |
| 45 | **Accept** | The machine contract now contains only enforceable hashes, DAG, permissions, objective/mode, authority/fallback, error policy, budgets, thresholds/support, resume/isolation, commands, and artifacts. Formulae and research prose remain in their owning docs/code. |
| 46 | **Accept** | Added a pre-processing/release legal and privacy gate for source use, identity linkage, retention, consent, model/commercial distribution, and Malom/pack redistribution. Unknown rights do not pass by being entered in a manifest. |

## Resulting Immediate Order

No training is started by this review. The next v5 work is:

1. F0-R: publish a concise implementation/evidence reconciliation, using the
   already corrected decoder rather than repeating the stale defect.
2. F0-P: determine the minimal history representation and benchmark W/proof
   feasibility.
3. F0-H: compute specialisation headroom and the maximum plausible whole-game
   effect from existing human support.
4. F0-A: compare ordinary, verified compact, pack, optional full DB, server,
   and hybrid product architectures on the intended target device/form.
5. Only after those reports, freeze E0/T0 implementation plans. T1 remains
   closed until both headroom and reference closure justify it.

The corrected-v4 lane, current maintainer experiments, and Sanmill bridge work
may continue as independently labelled evidence sources. None is silently
promoted into v5 lineage by this document change.
