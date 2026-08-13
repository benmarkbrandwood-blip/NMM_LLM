# Windows Training Handover — 20 July 2026 (updated 13 August 2026)

## Executive Summary

## Transfer checkpoint — 13 August 2026

This section is the current handoff checkpoint and takes precedence over older
historical wording below. The repository root is this directory. At handoff
publication the tracked worktree was clean and `dev` was synchronized with
`origin/dev`; a receiving operator must verify the exact current values with
`git rev-parse HEAD` and `git rev-parse origin/dev` rather than relying on a
copied SHA in prose. A real fetch during the attempt-002 readiness correction
advanced `origin/main` to
`40da3ddfced972c418541665ec739b3752edcd1f`. The four commits since
`bc2b87a2` harden the separate Human Move Preference Net and GapNet-v3
training/gating plans and tests. They do not touch the retained Generalist
trainer, managed-run route, Sanmill referee/opponent path or inputs enabled by
this experiment. No cherry-pick is required for the no-refresh baseline; the
new main tip is recorded as reviewed source context only.

The latest authorized mature target-refresh attempt is consumed and must not be
restarted. Its six training arms all completed and passed policy health: 2,529
new games, 49,152 post-fork transitions and 768 A2C updates. The result
publisher then failed closed before any development measurement game because
it incorrectly required canonical-minified JSON for a frozen pretty-printed
reference corpus. The failure identity is `d4e13fba`; the complete failure
record is in
[the attempt-002 failure evidence](../evidence/target-refresh-mature-fork-diagnostic-attempt-002-failure-2026-08-12.md).
There is no result, ledger or completion record for that attempt.

The first zero-training analysis recovery was then explicitly authorized and
launched once. It failed closed before its first development game because the
publisher incorrectly required one-shot mature-fork provenance fields to
remain in ordinary post-fork checkpoint implementation metadata. It wrote no
ledger, result or completion record. The authorization is consumed. Preserve
the exact identities and diagnosis in
[the recovery-v1 failure record](../evidence/target-refresh-mature-fork-analysis-recovery-v1-failure-2026-08-13.md).

Corrective commit `763f20c` compares stable implementation identity while
continuing to validate treatment, transition, configuration, source-checkpoint
and trainer-state lineage. It also makes recovery preflight read and validate
all 12 candidate checkpoints before another authorization can be requested.
A read-only post-fix audit passed all 12 with identity `d3c7e0dd`. This is a
publisher/preflight correction only; it changes no gameplay or training state
and does not revive the consumed authorization.

The isolated successor is frozen in
[recovery v2](../experiments/sanmill-target-refresh-mature-fork-analysis-recovery-v2.md)
under plan identity `32158846`. It keeps the same scientific and resource
contract and uses a new output namespace. At that checkpoint it remained
unlaunched. Commit
`360c878` published the exact recovery-v2 contract and implementation to
`origin/dev`. A final clean preflight at that published commit passed all 12
candidate checkpoints, recorded candidate-audit identity `d3c7e0dd`, and
returned `ready_for_product_authorization` under readiness identity
`fcd38c2f`. The readiness file SHA-256 is `6edca20a`. No authorization,
launch, development game, result, ledger or completion record existed for v2.
The following paragraphs record the subsequent execution and supersede that
pre-handover state.

The historical readiness was preserved byte-for-byte in ignored quarantine,
with unchanged SHA-256 `6edca20a`. After this handover commit was ordinarily
published, clean synchronized `dev` at `77e5d7c` regenerated readiness identity
`13e25cd5`. The product owner authorized exactly that 288-game/3.5-hour
zero-training recovery and requested no repeated prompts for technical steps
inside the authorized contract. The run completed once in 0.047169 active
hours, with zero training, optimizer, database, or checkpoint writes. Its
288-row ledger SHA-256 is `f0d35417`, result identity is `5e7bb7bf`, and the
predeclared classification is `no_material_direct_effect`.

The paired `refresh-mature minus stale-control` mean effect was `-0.076389`,
just below the frozen `-0.083333` material threshold. Seed effects were
`-0.229167`, `+0.010417`, and `-0.010417`; only seed 67 supported stale
control, while two supporting seeds were required. Only seed 67 also retained
a material policy-distribution trigger at both 4,096 and 8,192 transitions.
Neither target condition is selected. Preserve the
[recovery-v2 result evidence](../evidence/target-refresh-mature-fork-analysis-recovery-v2-result-2026-08-13.md).

The smallest independent replication successor is frozen in the
[mature-fork replication attempt-002 contract](../experiments/sanmill-target-refresh-mature-fork-replication-v1-attempt-002.md).
Commit `d380958` generalized the mature-fork tooling and added the
preregistered cross-cohort gate. The plan reuses the untested mature-boundary
`no-refresh` checkpoints for disjoint seeds 64, 65 and 66, each at exactly
8,192 post-game-50 consumed transitions. Seed 64 uses a byte-identical closed
database snapshot because its historical zero-byte WAL and 32,768-byte SHM
sidecar must remain untouched. Seeds 65 and 66 already have closed source
databases. Every input is `sector-corrected-v1` and bound by file identity.

The first replication contract, plan identity `8071e4a0`, was validly
prepared at clean synchronized commit `1e88081`. Readiness identity was
`a4d9fd63` and readiness file SHA-256 was `6995cd6a`; all six policy-health
gates were present, with zero authorization files and zero training segment
directories. Publishing the later handoff-only commit `c8f44f4` advanced
`dev`, so the parent and managed-trainer exact-source gates correctly made
that preparation stale. It remains preserved and must never be authorized or
launched.

Attempt 002, plan identity `85ad0b99`, changes only lineage-owned output and
plan identities. Its scientific design, sources, resources, measurements,
claim boundary, stops and prohibited operations are identical. The contract
and this handoff update must be published together; only then may the six
authorization-free plans be prepared at that final commit. No later tracked
commit may be created before the parent decision and any execution finish.

The frozen aggregate ceiling remains 3,600 training games, 49,152 consumed
transitions, four active hours, 172,800,000 requested Sanmill node ceilings
and 288 no-update development games. There is no retry, recovery, resume,
extension, held-out evaluation, promotion, publication, retained run or
long-training fallback. Verification before the source-only rebind passed 133
expanded focused tests and the required Malom/DB/provenance set with 103 tests
plus 498 subtests.

Current verdict at tracked handoff publication: attempt 002 is designed and
unlaunched. Its ignored readiness and the final parent preflight become the
authoritative machine-local state after preparation. It grants no authority
and is not `ready_for_long_run`; the result gate explicitly sets
`automatic_long_run_selection=false`.

That prelaunch state is now historical. At clean synchronized source
`8179e8e`, attempt 002 was prepared under readiness identity `7088c1f5` and
authorized once under parent identity `b7384d76`. The product owner approved
the exact parent grant and directed the Agent not to ask separately for seeds,
arms, or the included analysis. All six arms completed and passed policy
health, consuming 2,324 new training games, exactly 49,152 post-fork
transitions, 768 A2C updates, and 0.3861 managed active hours. The sequence
then completed all 288 no-update development games without checkpoint or
database writes. Completion identity is `94648404`; result identity is
`8559fa7b`; result file SHA-256 is `0197ca41`. No failure record exists.

The disjoint seeds 64--66 cohort favoured `refresh-mature` by `+0.121528`,
with seed effects `+0.322917`, `+0.083333`, and `-0.041667`. The independent
seeds 67--69 cohort had favoured stale control by `-0.076389`. Their pooled
six-seed effect was only `+0.022569`, below `1/12`, and only seeds 64 and 65
met the supporting-seed boundary rather than the required three. The frozen
classification is `no_replicated_material_effect`, with no selected successor
condition and `automatic_long_run_selection=false`. Preserve the
[replication result evidence](../evidence/target-refresh-mature-fork-replication-v1-attempt-002-result-2026-08-13.md).

The next retained research plan was frozen in
[Sanmill no-refresh retained long v4](../experiments/sanmill-no-refresh-retained-long-v4.md).
It does not reinterpret the pooled null as a selected cadence. It instead
tests a new permanent no-refresh hypothesis from fresh random state at unused
seed 70, with `target-refresh-every=5001`. All retained-v3 reward, component,
opponent, fixed-node curriculum, max-ply, segmentation, monitoring, 5,000-game
and 12-hour choices remain fixed. The source-only document is unlaunched and
grants no training authority. Preparation attempt 001 at source `f1a8974a`
never received authorization and produced zero games. It is now preserved as
`invalidated_unlaunched_never_authorize`; see the
[attempt-001 disposition](../evidence/sanmill-no-refresh-retained-v4-preparation-attempt-001-2026-08-13.md).
Attempt 002 kept seed 70 with new experiment, plan, control and database
identities. At clean synchronized source `12ecd934`, plan identity `2a59a93f`
and readiness identity `a6cd2cd1` passed independent review. The product owner
authorized that exact 5,000-game/12-hour plan once. Segment 0001 passed its
launch preflight and then failed closed before an accepted checkpoint or
segment because the legacy A2C update path cleared an aliased batch before
calculating its behaviour-temperature evidence. The controller stopped on
exit code 1 and did not retry. Preserve the consumed authorization, failed
control directory and mutated 188-position SpecialistDB; see the
[attempt-002 failure evidence](../evidence/sanmill-no-refresh-retained-v4-attempt-002-failure-2026-08-13.md).

Attempt 002 has no no-refresh result. A successor requires a tested source fix
and new experiment, plan, control, database, readiness and authorization
identities. It may retain seed 70 as a fresh seed because no checkpoint was
written, but it is not a resume or recovery of attempt 002.

Commit `cde6a5e` fixes the failure without changing optimizer or gameplay
semantics: the legacy non-exact A2C route now takes a shallow snapshot of its
pending step list before clearing the lineage-owned queue. The focused update
tests pass 30/30 and include the exact queue-clear regression. Attempt 003 was
then prepared from clean synchronized source `662fe160` under plan identity
`1702726f` and readiness identity `77cc65ad`. The product owner authorized the
exact 5,000-game, 12-active-hour plan once.

Attempt 003 completed all 5,000 games and 20 policy-gated segments in 1.9478
active hours without retry or recovery. Target age advanced exactly from 1 to
5,000, so the no-refresh treatment executed. The authorization is consumed.
The final checkpoint file SHA-256 is `295b268e`; its verified payload is
`ed7932bc`. The final SpecialistDB main-file SHA-256 is `3d69d1ac`, with
242,006 positions and 4,185 winning lines.

The mixed 69.32% logged score is not a strength result. The frozen initial
target arm was `2,721 W / 259 D / 0 L`, while the Sanmill arm was
`23 W / 1,185 D / 812 L`. At 500,000 nodes, 605 of 1,051 Sanmill games hit the
120-ply cap. The trainer manifest records `rolling_win=40`, so the dashboard's
legacy `win_rate_200` field represented 40 mixed games; the final 67.5% is 27
wins in 40, not a 200-game KPI. Preserve the
[attempt-003 result evidence](../evidence/sanmill-no-refresh-retained-v4-attempt-003-result-2026-08-13.md).

The v4 comparison to retained v3 is source-confounded as well as
seed-confounded: v3 ran at `3f400135`, before later trainer, manager and
preflight hardening and before the current explicit read/LR mode interfaces.
The new run remains useful as a current-source research baseline, but no
difference from v3 may be attributed solely to target refresh.

### Remaining work for the next operator

1. Preserve recovery-v1 as consumed fail-closed evidence. Do not delete,
   overwrite, repair in place, or reuse its output namespace.
2. Preserve the completed recovery-v2 readiness, authorization, launch,
   288-row ledger, result, completion and logs under their recorded identities.
   Do not rerun, extend, repair in place, or reuse its consumed authorization.
3. Preserve the `no_material_direct_effect` result and null selection. Do not
   choose stale control from seed 67 alone or lower the frozen threshold after
   observing the result.
4. Preserve replication attempt 002 under readiness identity `7088c1f5`,
   authorization identity `b7384d76`, launch identity `c71c2ccb`, completion
   identity `94648404`, and result identity `8559fa7b`. Its grant is consumed;
   do not rerun, resume, extend, overwrite, or relabel it.
5. Preserve the null pooled cadence selection. A next retained run may be
   designed as a new research baseline, but its target schedule must be stated
   as a new plan choice rather than attributed to this result.
6. Preserve no-refresh retained-v4 preparation attempt 001 byte-for-byte and
   never authorize it.
7. Preserve no-refresh retained-v4 attempt 002, its consumed authorization,
   failed event chains, logs and mutated database. Do not retry, resume, repair
   in place or reuse any of them.
8. Preserve completed attempt 003 under plan identity `1702726f`, readiness
   identity `77cc65ad`, source `662fe160`, all 20 accepted segment directories,
   final checkpoint, database, controller ledger and consumed authorization.
   Do not rerun, resume, extend, overwrite, promote or publish it.
9. Keep the next v3/v4 work at `needs_decision`. First freeze whether the
   objective is passivity/mechanism diagnosis or playing-strength relation,
   then add a paired power/precision analysis, prospective ply-120 state
   capture, a strict safety cap, disjoint corpus/exposure audit and a tested
   no-update evaluator. A 16-game pilot is too small to estimate the observed
   2--4% decisive-game rate, and process metrics cannot replace W/D/L for a
   strength claim. Preserve the
   [evaluation decision brief](../experiments/sanmill-retained-v3-v4-evaluation-decision-brief.md).
   No evaluation game is authorized yet.

No candidate-vs-baseline held-out match, model promotion, publication, or long
training is currently authorized. Historical ignored artifacts under `out/`
are evidence inputs and must not be deleted, overwritten, or relabelled.

### Latest state: early target refresh is harmful; later cadence unresolved

The seed-58 `managed-sanmill-preserving-retained-v3-seed58` run is complete
and frozen as evidence. It reached 5,000 games in 20 accepted segments and
1.732867 active hours. Its final checkpoint SHA-256 is
`28e8af274f4fc9dd7e00ce4f7be884c855354218c796888f1c1ab81a4cdc9fa7`,
and its final SpecialistDB main-file SHA-256 is
`82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe`.
The final 200 games were 0 wins, 199 draws and 1 loss. This is a retained
research baseline and diagnostic observation, not promotion, strength, or
trap-learning evidence. See the
[completion evidence](../evidence/sanmill-preserving-retained-v3-result-2026-08-10.md).

A first no-update SpecialistDB audit found no usable coverage on the fixed
phase corpus and is retained as negative coverage evidence only. A second
candidate-blind audit replayed the frozen 12-ply source histories, selected all
100 states with usable empirical support under a preregistered rule, and only
then loaded the completed checkpoint. It found three argmax changes when
empirical reads were suppressed, mean scheduled-temperature total variation
about 0.0174, and no Malom preserving-to-downgrading crossing. The result is a
small material mechanism effect, not evidence that empirical reads caused the
late draw mass.

Commits `98ef28d` and `9e9b8da` add and propagate explicit `full` versus
`theoretical-only` SpecialistDB training reads. Both modes preserve the same
writable database route. Per-rollout diagnostics count queries, available
theoretical and empirical evidence, returned projections and suppressed
empirical reads; counters are isolated per worker thread. The selected mode is
bound in the run manifest and managed trainer command.

The three-seed SpecialistDB read-mode calibration subsequently completed once
under readiness identity
`ee68e2d90d069bd65643d0e02ecb4c408fb522a35d6e19bd6246b1cb1b640b6f`.
All six arms reached exactly 250 games, for 1,500 games and 405 optimizer
updates in total. The immutable result identity is
`90da60538e782c85b5871e35eec4895e44fe76003309b3ad13c417c8868f86de`;
its file SHA-256 is
`e8a8f3aac6076697b9a31c7532880976f4222801d002f988d5f65bf78c8344e9`.
No read mode was selected. The run is mechanism evidence only and does not
authorize held-out evaluation or retained training.

The follow-up two-seed target-refresh/LR diagnostic subsequently completed all
eight arms. It found a repeatable post-game-50 contrast against each arm's own
training target, no learning-rate factor effect, and no Sanmill-facing win.
Because target refresh changed both the trained opponent and the measurement
denominator, it selected no training setting. The successor common-anchor
design therefore separates the training target from a fixed game-50
measurement anchor and matches 16 post-boundary optimiser steps.

Attempt 001 stopped fail closed at its first game-50 measurement anchor because
the checkpoint envelope lacked the dedicated evidence roles. Correction
`e02aca4` and a fresh successor were published. Attempt 002 was then prepared
and authorised once under readiness identity `bcbb625d`. Its two seed-64 arms
completed 122 and 92 training games respectively, each reached 34 optimiser
updates, produced four common-anchor measurement batches and passed policy
health. Their first 50 canonical game rows and anchor model tensors are
identical. The frozen result analyser nevertheless rejected the valid early
optimizer-bounded endings because a reused fixed-game helper required 150
games. The sequence stopped before either seed-65 arm started, and all four
authorisations are consumed. Analysis-only correction `873e126` accepts the
validated actual completion count without changing training or gameplay.
Neither attempt produced target-refresh evidence. See the
[attempt-001 failure record](../evidence/target-refresh-common-anchor-diagnostic-attempt-001-failure-2026-08-10.md)
and
[attempt-002 failure record](../evidence/target-refresh-common-anchor-diagnostic-attempt-002-failure-2026-08-10.md).

Attempt 003 was subsequently published, prepared and authorized once under
plan identity `8cc192f5` and readiness identity `43e9eb75`. All four arms
completed: seed-64 refresh/no-refresh used 122/92 games and 34 optimizer
updates; seed-65 refresh/no-refresh used 110/100 games and 32 optimizer
updates. The run produced 424 training games and 256 no-update measurement
games. All policy-health gates passed, but both fixed-anchor strata remained
at the outcome floor and the result selected no target-refresh setting.

A no-training full-action follow-up now compares the four post-anchor
checkpoints on the same 64 placement/movement/flying positions. Its raw local
report SHA-256 is `02f84765`, and analysis identity is `a885711a`. At the final
checkpoint, temperature-0.2 mean total variation was only `4.60e-5` for seed
64 and `7.09e-5` for seed 65; mean Jensen-Shannon distances were `1.97e-9`
and `6.45e-9`. All 1,583 legal actions had exact Malom quality, and both
conditions selected a preserving top-1 action in every state, but their
preserving probability mass was only about `0.00009` to `0.00019` above an
equal-action uniform reference. The policies are therefore classified
`near_identical` and almost uniform at this horizon, despite material
parameter-space separation. See the
[full-action evidence](../evidence/target-refresh-common-anchor-policy-distribution-2026-08-10.md).

The three-seed equal-transition target-refresh diagnostic was executed once at
clean published training source `33a98696994cddf8be0b1ab516a879f52483ef02`
under plan identity
`b14d69db9a33b005c0a19fbb97e7f5b9a16364f1f74390ae85ff3e9d4edabb97`.
Seeds 64, 65 and 66 each completed a 50-game shared prefix followed by paired
`refresh-once` and `no-refresh` arms. Every arm reached exactly 8,192 consumed
post-fork transitions as 128 batches of 64, with fixed learning rate, finite
updates, trusted Malom coverage and isolated sidecar-free SpecialistDB files.
The treatment reset target age while the control retained it, so the mechanism
separation executed as designed. This is still mechanism evidence only.

The immutable equal-transition result has now been generated from clean,
published analysis commit
`0f8e9eb04e9fe046f72fbe47ed0551eeeafc22d4`. Its SHA-256 is
`b518849fa4ca3339bf1b3e4842cf5c10f20002088e9b4f4074da3891cb2d2ca3`,
its result identity is
`8c6be27feb96d0e50662e299b594140c96b14ec57cf447ecf572fe07757a95dd`,
and the predeclared classification is `inconclusive_late_onset`. Seeds 64 and
65 first crossed a material distribution threshold only at 8,192 transitions;
no seed had a persistent trigger from 4,096 through 8,192, and the Malom-mass
direction reversed between seeds. The result selects neither target-refresh
condition.

The result also exposes game-count schedule coupling. Exact transition
exposure produced different game counts; by the final boundary, refresh seeds
64 and 66 had cooled further and reached the 5,000-node Sanmill level while
their controls remained at 1,000 nodes. Seed 65 provides counterevidence to a
schedule-only explanation, so this is a plausible mediator rather than a
complete diagnosis. Every arm remained winless against its Sanmill training
stratum. Preserve the full
[result evidence](../evidence/target-refresh-equal-transition-diagnostic-result-2026-08-11.md).
No held-out evaluation, promotion, model publication, successor training, or
long run is authorized by the completed diagnostic.

The schedule-isolation v2 successor was frozen under plan identity
`0580389b3d696df9859ac9e7aea6c4b478bf6e791b7e27bf780d2a6e02db5b0b`.
It retains three fresh shared prefixes and six paired arms, fixes Sanmill at
1,000 nodes, indexes temperature by consumed learner transitions, and adds a
candidate-blind 288-game common-anchor development measurement. Its one-shot
parent runner binds the product-facing readiness identity, exact aggregate
resources, fixed CPU analysis route, child authorizations and no-overwrite
records. The previous published identities `1a86e158` and `9c806830` are
superseded only because the source-lineage review advanced to `origin/main`
commit `028ef8e`; no scientific or resource decision changed. The
[main review](../evidence/origin-main-training-review-2026-08-11.md) keeps the
GapNet v3 ledger and HMPN split chain at fatal stop, records the valid NaN fix
as irrelevant to the disabled-GapNet experiment, rejects the unsafe legacy
pickle fallback and transient UI checkpoint selection, and records the
independent UTF-8 portability repair. Fresh ignored databases, plans and
readiness evidence were created only after this source was published.
Readiness required the reviewed main tip to remain an ancestor and reported
later fast-forward descendants separately; unrelated remote movement did not
change the frozen scientific contract.

After technical readiness `e5ed9054` was presented, the product owner replied
“同意” to the exact one-shot parent grant: at most 3,600 contract games, 3,450
actual training games, 49,152 post-fork transitions, six active training
hours, and the fixed 288-game no-update development grid, with no retry,
recovery, resume, extension, held-out work, promotion, publication, or
long-training fallback. No authorization file or launch marker was written
before the final fetch advanced `origin/main`, so it was initially unconsumed.
After a source-lineage-only rebind proved that the scientific design,
resources, claim boundary, stop rules and prohibited operations had not
changed, the owner authorised and the Agent executed the sequence once. The
six arms completed, but the publisher failed closed on valid uniformly CRLF
Windows JSONL before any development game. That training launch and all child
authorisations are consumed.
The canonical delegated product scope, computed from the entire contract after
excluding only `plan_identity` and `lineage`, is
`a92e87bebe87e1a287be37c95c0974cafde662703ee05436a2c30b7d9584211a`
for both the approved `1a86e158` contract and the current source-only refresh.
A focused contract test freezes this digest; any scientific, resource,
measurement, stop-rule, or prohibited-operation change invalidates the grant.

The separately authorised analysis recovery then completed the missing 288
CPU no-update development games once under readiness identity `034ed820`.
It performed zero training games, optimizer updates, database writes and
checkpoint writes. Raw result identity `a4381489` classifies the common-anchor
outcome effect as `no_material_paired_outcome_effect`; the full-action policy
comparison is `inconclusive_late_onset`. The aggregate score contrast is zero
at 4,096 transitions and `+0.04167` for no refresh at 8,192, below the frozen
`0.08333` effect gate. Seeds 67 and 68 show material policy separation only at
the final boundary, while seed 69 remains below every material threshold. The
common Sanmill measurement is severely floor-limited, particularly with the
candidate as White. No target-refresh condition is selected. Preserve the
[result evidence](../evidence/target-refresh-schedule-isolation-diagnostic-v2-result-2026-08-11.md).
The recovery authorization is consumed and gives no held-out, promotion,
publication or long-training authority.

The subsequently authorized target-refresh direct cross-play attempt 001
failed closed before its first game. Source commit `b25fe33`, plan identity
`c7a03214`, readiness identity `6da0b4ac` and authorization identity
`5485baf7` were valid, but the runner requested `policy_seed_w` while the
closed schedule correctly provided `policy_seed_white`. The canonical ledger
is empty; no policy action, logical ply, training game, optimizer update,
database write or checkpoint write occurred. No result or completion record
exists, all read-only identities remained unchanged, and no process remained.
The authorization is consumed and does not permit a retry. Preserve the
[failure evidence](../evidence/target-refresh-direct-crossplay-attempt-001-failure-2026-08-12.md).
A successor must fix and test the closed seed-field mapping, use isolated
attempt-002 output paths, publish a new immutable plan and readiness identity,
and receive new explicit product authorization.

The corrected attempt-003 direct cross-play subsequently completed once under
plan identity `2f1665e5`, readiness identity `9fd354a7`, and authorization
identity `3175570e`. It consumed 288 CPU no-update games as 144 colour-swapped
pairs across seeds 67, 68 and 69 and placement, movement and flying starts.
There were no training games, optimizer updates, database writes or checkpoint
writes. The no-refresh condition scored `178 W / 12 D / 98 L`; its paired
mean score effect over refresh-once was `+0.2777778`. All three seeds and all
three phases supported the same direction, while nine max-ply truncations gave
a safe `0.03125` truncation rate. The frozen classifier returned
`material_no_refresh_direct_effect`.

This is evidence that refreshing the frozen target at the original game-50
boundary was harmful under the tested schedule. It is not evidence that a
target should remain stale forever, and it does not select a retained setting.
The next discriminating experiment must fork each mature no-refresh checkpoint
at 8,192 post-fork transitions, refresh once only in the treatment arm, and
hold subsequent transition exposure, temperature sequence and 1,000-node
Sanmill work equal. Preserve the
[attempt-003 result evidence](../evidence/target-refresh-direct-crossplay-attempt-003-result-2026-08-12.md).
Its evidence identity is `9a5df62d`. No held-out evaluation, long training,
promotion or publication is authorized by this result.

The first source-only preparation for the mature-fork successor was rejected
before authorization because all six generated plans had
`policy_health=null`, contradicting the contract's fail-closed policy-health
gate. Readiness identity `32df3a5b` and all associated ignored preparation
artefacts were moved intact to
`out/quarantine/target-refresh-mature-fork-diagnostic-v1-missing-policy-health-2026-08-12/`.
No authorization, segment, training game, optimizer update, database write or
checkpoint write occurred. Commits `d1c8b4c`, `2cace43` and `c6e0689` enforce
the gate, add the bounded sequence/result pipeline, and isolate successor
paths. Commit `40b85e6` freezes
[attempt 002](../experiments/sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.md)
under plan identity `442c1701`. It is a design contract only: preparation,
readiness review and one aggregate product authorization remain required
before launch.

The corrected mature-fork attempt 002 was subsequently prepared and launched
once under readiness identity `d2860ae0` and authorization identity
`181a8e88`. All six arms completed and passed policy health, consuming 2,529
new training games, exactly 49,152 optimizer transitions, 768 A2C updates and
about 0.403 active hours. The parent publisher then failed closed before the
first of 288 CPU development games because it imposed canonical-minified JSON
framing on a frozen, exact-hash, pretty-printed policy corpus. The replay
corpus and replay audit share that intentional presentation format. No result,
ledger or completion record exists, so no mature-refresh condition has been
selected. Preserve the
[attempt-002 failure evidence](../evidence/target-refresh-mature-fork-diagnostic-attempt-002-failure-2026-08-12.md).
The original launch and all child authorizations are consumed. Do not retry or
resume training. A corrected publisher may be used only through a separately
frozen, zero-training analysis-recovery plan with a new readiness identity and
explicit one-shot launch authority. That
[analysis-recovery plan](../experiments/sanmill-target-refresh-mature-fork-analysis-recovery-v1.md)
is frozen under identity `70fb522b`. A full preflight passed at published
`dev` commit `b3854a7`, establishing that the recovery is technically viable.
Readiness binds the exact current published HEAD, so the corresponding
machine-local identity `45662e3a` became historical when this status was later
committed. Before authorization, regenerate the ignored readiness file against
the latest clean published `dev`; do not copy that self-invalidating identity
back into tracked status prose. The recovery permits no training and has not
been authorized or launched.

On 11 August 2026 the product owner explicitly delegated just-in-time launch
authorization for the remaining bounded plans in this already frozen
equal-transition sequence. The Agent recorded each leaf authorization as
`product-owner-delegated-agent`, retained the original aggregate resource and
claim boundaries, and completed launch orders 3 through 9 without repeated
product prompts. The durable standing-delegation policy now belongs in
`AGENTS.md`, the managed-operations document and the training-readiness skill.
It removes per-arm rubber-stamping but does not authorize a new objective,
resource expansion, held-out work, long training, promotion or publication.

On 11 August 2026 the product owner separately granted standing Git authority
for Codex to publish commits it created and verified from local `dev` to
`origin/dev` by ordinary fast-forward without another prompt. Before each such
push, fetch and prove that the active branch is `dev` and that `origin/dev` is
an ancestor of local `dev`. The grant excludes force-push, merge, rebase,
amend, and every other history rewrite. It is independent of training,
evaluation, promotion, and publication authority and may be revoked at any
time.

The completed SpecialistDB main file itself remains byte-identical. An early
non-immutable audit check created a zero-byte `-wal` and a 32,768-byte `-shm`
beside that historical database. They were not deleted. All accepted policy
audits instead used an ignored, sidecar-free byte-identical snapshot named
`specialist_db.sanmill_preserving_retained_v3.seed58.audit_snapshot.sqlite`.
Do not treat the sidecars as new training data or delete them without an
explicit, recoverable cleanup decision.

The repository is usable on the Windows 11 host and the downloaded databases
and existing model artifacts are in their intended locations. The focused
Malom/provenance and current trainer-contract suites are green. The 7 August
complete run collected and executed 1,235 tests: 1,227 passed and eight
machine-local Sanmill bridge tests failed closed because the moving checkout
changed protected paths relative to their historical strict-v2 pin. The
historical binary bytes also remain unavailable. This is not a clean all-pass
claim; see the
[current complete-test baseline](../evidence/current-complete-test-baseline-2026-08-07.md).
The authorized corrected-v4 managed plan
`managed-v4-baseline-v1` completed 5,000 games in 20 verified segments on
21 July 2026 (UTC). Its completion is lineage and infrastructure evidence, not
playing-strength or promotion evidence. It does not authorize reuse of that
lineage.

The separately authorized rules-corrected successor smoke
`successor-rules-v2-smoke-001` completed on 7 August from clean, published
`dev` commit `5cb44b1`. It used fresh weights, the isolated empty corrected DB,
the final MIF and rules identities, and explicit disable controls. Two counted
games produced one finite 14-step Adam update and a verified version-2
`latest.pt`; the lifecycle chain and post-run database audit pass. This is
infrastructure evidence only. The smoke authorization is consumed and no long
run is authorized. See the
[successor smoke result](../evidence/successor-training-smoke-result-2026-08-07.md).

Later work on 8 August closed the first fresh Sanmill-refereed lineage's
terminal projection, strict-referee, exact-resume, node-throughput,
integrated-route and deterministic resource-schedule gates. The isolated
five-game schedule smoke exercised all five fixed-node levels and completed
five finite A2C updates. The product owner then authorized the bounded
[`managed-sanmill-v4-fresh-v1` long-run contract](../experiments/sanmill-refereed-managed-long-v1.md):
at most 5,000 games or 12 active hours, 120 logical plies per game, and
250-game exact-resume process segments. That run later completed all 5,000
games and 20 segments without an infrastructure failure, but it is now
learning-invalid and must not be resumed or promoted. A fixed-state audit
found that its final policy selected a value-downgrading Malom move on 27 of
29 critical positions and had a mean preserving-minus-downgrading logit
margin of about -0.730.

The root-cause reproduction found two independent training defects. Commit
`4b0420755428d73581108f6e93cd95407b1b72dc` negates value bootstrap when a
successor belongs to the opponent-to-move perspective. Commit
`0fbc9510400c88a493b6e2efdcf7c9e92ae8b150` makes frozen-target opponents use
the same lookahead and SpecialistDB feature route as the learner. A separate
fresh 500-game smoke then completed 110 finite A2C updates and passed the
prespecified policy-direction limits: 28 of 29 critical argmax choices
preserved value and the mean logit margin was +0.0044. Its checkpoint and
database remain smoke evidence only and cannot seed retained training.

Current-source continuous-versus-segmented exact-resume parity passed and is
recorded at `fa2656f8683f8464b14923d84e3c77a8500fd239`. Commit
`c070739a9c94938528e76083cf3ef69f997c7a5a` then added an optional,
hash-bound policy-health audit before each managed segment can become an
exact-resume parent. Its focused and mandatory regression evidence plus a
real-checkpoint controller exercise are recorded at
`9409b4bcdc2b4c559fb75495965dde4f46dde87f`. The replacement
[`managed-sanmill-corrected-retained-v2` contract](../experiments/sanmill-corrected-retained-long-v2.md)
is frozen at `bbe2d32cc2c36153f3c359698aa3c74548eb8fbd`. It requires fresh
seed-42 weights, a new empty corrected SpecialistDB, the same measured
5,000-game fixed-node curriculum, and policy-health quarantine after every
250 games. The authorized plan later completed all 5,000 games and all 20
policy-gated process segments on 9 August. Its final controller state is
`completed`, all boundary audits passed, and the final checkpoint envelope
and SpecialistDB identity verify.

The run required two audited recoveries. Segment 13 stopped after game 3,200
because NMM_LLM rejected a valid Sanmill search that made 30 search calls but
expanded zero new nodes from its persistent transposition table. Published
commit `7049416` corrected that interpretation and `6279139` added
evidence-bound failed-segment recovery. A later outer command timeout
interrupted segment 18 before its first checkpoint; commit `4973e32` made
that host-only case resume from the last accepted game-4,250 boundary. The
segment-13 database was a trusted same-lineage live database whose identity
had advanced past the checkpoint, so the final lineage is auditable but not a
bit-for-bit counterfactual replay of a failure-free run.

The accepted log lineage contains exactly 5,000 unique consecutive game rows,
including the game-3,001 through game-3,200 checkpoint prefix retained in the
quarantined first segment-13 directory. The final policy gate preserved value
on 28 of 29 candidate argmax choices, while direct lookahead preserved all 29;
the mean candidate margin was +2.398080. These are anti-collapse and
infrastructure results, not strength evidence. See the
[completion evidence](../evidence/sanmill-corrected-retained-v2-result-2026-08-09.md)
for exact hashes, recovery events, W/D/L accounting and the next evaluation
gate.

On 9 August the first post-training held-out protocol was frozen at plan
commit `106d015b23debee7d5c8d691195ff958da66f1fc`, with plan identity
`212076e9423b671b83783efef411db3b4a56c8c67ae36a463d381d6939d4d982`.
It uses 64 preregistered twelve-ply starts as 128 colour-swapped games against
the exact 500,000-node Sanmill curriculum endpoint. A source-only audit found
that the complete operational corpus is not wholly data-disjoint: 30 starts
match HumanDB under D4 and one matches the final SpecialistDB. The same frozen
ledger therefore reports a separate 34-start zero-database-match sensitivity
subset. The product owner granted one bounded execution after a tested runner
and final read-only preflight pass. Published `dev` commit
`23dd90008b1d260a054e0c3cb471b8aad71e99a6` passed every final gate, and the
one allowed execution is now complete. The grant is consumed and must not be
used for another run. See the
[evaluation plan](../experiments/sanmill-corrected-retained-v2-heldout-eval-v1.md),
[authorization](../experiments/sanmill-corrected-retained-v2-heldout-eval-v1-authorization.md)
and [exposure audit](../evidence/sanmill-corrected-retained-v2-heldout-exposure-2026-08-09.md).

The dedicated evaluator was published at implementation commit
`e32d9d46a361d2ed6877b669cdf653eba78e3f3c`. It provides the guarded CLI,
canonical hash-chained game ledger, strict result validation, active-time
ceiling, exact missing-suffix resume and recomputable paired/subgroup report.
Pre-publish evidence commit
`43cb0189930d7071403e825c603e2b91feff7b18` records 64/64 fresh-process
prefix replays, a real non-corpus candidate/Sanmill interoperability canary,
79 current focused tests, and 103 Malom/provenance tests with 498 subtests.
Every read-only gate passed except the deliberate publication gate because
local `dev` was still ahead of `origin/dev` at that earlier audit. The seven
preparation commits were subsequently pushed by ordinary fast-forward before
the final preflight; see the historical
[runner readiness evidence](../evidence/sanmill-corrected-retained-v2-heldout-runner-readiness-2026-08-09.md).

The evaluation completed all 64 colour-swapped pairs. The retained candidate
scored 3 wins, 102 draws and 23 losses, or 42.1875%. Its mean paired score
difference was -0.15625 with interval
`[-0.23146381558966117, -0.08103618441033884]`, so the frozen decision is
`candidate_behind`. HumanDB was inconclusive at 48.81%; Book was behind at
40.91%; Perfect DB was behind at 36.90%; and the strict 34-pair zero-match
sensitivity subset was behind at 37.50%. This is fixed-corpus relation
evidence, not a general strength or Elo claim.

After four completed games, a concurrent read of `progress.json` overlapped a
Windows atomic replacement and stopped the process with `WinError 5`. The
original failure bytes are preserved under SHA-256
`63615c1460fe0fc6c567c234bf1b2e368355b6c5fc2f758aa5ae870d38eab6af`.
An exact-resume preflight bound the same commit, specification, host and
four-record ledger, then ran only the missing 124-game suffix. The final
128-record ledger SHA-256 is
`100863efa58381fc736096440bf8ff4a178cd34215ac7b43e3d6f6767fae7892`,
and independent recomputation exactly matches result identity
`8848ad32e588daf2fcd0686be65b337e7fc621faaebdb58bd1dbefc73bcdff81`.
Result evidence commit `8d247e9` records the complete claim boundary. Local
commit `6a88deb` separately adds bounded retries for transient Windows
progress-file readers; it does not alter gameplay or the completed report.
See the
[held-out result evidence](../evidence/sanmill-corrected-retained-v2-heldout-result-2026-08-09.md).

On 8 August the product owner selected a separate fresh
`dev-v4-sanmill-refereed-fresh-v1` lineage after confirming that Sanmill must
participate in training, not merely in later evaluation. The new route starts
from random weights, keeps the completed local-GameAI run immutable, uses
Sanmill as the authoritative complete-history referee for every primary
rollout, and replaces local `GameAI` with fixed-node `go logical` search on
the non-frozen opponent stratum. Its exact ignored runtime is pinned to
Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`, tree
`17b9b0fd51ee8dac54c0454a6935978a47d19e0c`, binary SHA-256
`5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`,
and strict-referee semantic digest
`sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.
The initial contract disables retries, branches, observation-based recovery,
and the uncalibrated local-GameAI advancement gate. Process-level tests cover
cross-process search determinism, full logical turns with compulsory removal,
illegal-action rejection, and an eight-ply learner/opponent rollout whose
entire history is checked by Sanmill. This is implementation evidence only.
The separately authorized `smoke-001` then failed closed during its first
game: Sanmill correctly emitted a game-over FEN with raw action `?`, but the
NMM_LLM mirror called a projector limited to placing and moving states. No
counted game, optimiser update, checkpoint, or training log was produced.
Commit `4e734e4a3105b1a590fbb11ab13c3197cb6a9fce` repairs that terminal-only
projection boundary and adds the exact 43-logical-ply regression. The failed
output and its subsequently diagnostic-written SpecialistDB are quarantined;
the first one-run authorization is consumed. The separately authorized
`smoke-002` then ran from clean published commit `894360d`, after its final
preflight again returned `ready_for_smoke`. It completed both scheduled games,
one finite 47-step A2C update, a verified version-2 `latest.pt`, and a valid
completed lifecycle chain. Its isolated corrected database now contains 94
positions and one winning line and is retained as completed smoke evidence,
not a fresh input. The second one-run authorization is consumed. No retry or
long run was authorized by that smoke. At that point the node ladder,
representative throughput envelope, and later advancement rule remained
unfrozen.
See the
[fresh Sanmill experiment contract](../experiments/dev-v4-sanmill-refereed-fresh-v1.md).
The terminal repair verification reports 182 trainer, launch, checkpoint,
resume, bridge, and referee tests passed with six documented historical
moving-checkout tests deselected, plus 103 Malom and label-provenance tests
with 498 parameterized subtests passed. Ruff and `git diff --check` also pass
for the changed scope. The
[failure record](../evidence/sanmill-refereed-fresh-v1-smoke-001-failure-2026-08-08.md)
owns the raw identities and diagnostic-side-effect boundary. A new full
`tests/` run collected 1,246 tests and reached approximately 16% with no
failure before the 15-minute command limit, so this work does not replace or
upgrade the separately recorded 7 August complete-suite baseline.
The
[smoke-002 result](../evidence/sanmill-refereed-fresh-v1-smoke-002-result-2026-08-08.md)
owns the successful launch identities and explicitly excludes its two games
from node-ladder, strength, and retained-run throughput claims.
The separately authorised
[Sanmill node-throughput calibration v1](../experiments/sanmill-node-throughput-calibration-v1.md)
completed on 8 August. It produced all 720 timed searches over eight
complete-history roots, five fixed-node ceilings, nine repetitions, and
separate cold-process and warm-sequence modes. All 80 cells were semantically
stable, and all 40 matching cold/warm cells selected the same complete logical
turn. The run used 79,510,680 actual nodes under 90,864,000 requested ceilings
and completed in 37.846903 seconds. At 500,000 nodes, the warm cross-root
median was 52.85 ms and P90 was 60.77 ms; the empty-board root remained
phase-depth limited at 52 nodes. Fresh process startup had median cost about
59.7 ms, so the persistent training process must be represented in the next
measurement. The
[result record](../evidence/sanmill-node-throughput-calibration-v1-result-2026-08-08.md)
binds the ignored raw report and its claim boundary. It loads no model,
trainer, checkpoint, or database and does not select a node ladder.

The resulting
[node-ladder decision brief](../experiments/sanmill-node-ladder-v1-decision-brief.md)
now recommends the five measured ceilings
`1,000 -> 5,000 -> 25,000 -> 100,000 -> 500,000` as a probe-only resource
matrix. Across the seven non-empty roots their warm completed-depth medians
were 4, 7, 9, 11, and 13. This is not strength evidence. The current trainer
still fails closed unless the Sanmill lineage uses one fixed work level and
disabled advancement, so the five-level proposal cannot be launched as a
training curriculum. Its acceptance and any later progression schedule remain
owner decisions.

The implemented but unlaunched
[no-update integrated-route probe](../experiments/sanmill-no-update-integrated-route-probe-v1.md)
is designed to measure the production rollout path without learning. Its bounded
matrix contains 30 Sanmill-opponent games and six Sanmill-refereed
frozen-target controls, explicitly separating normal depth-5 from deliberately
oversampled depth-12 routes and both learner colors. It requires a fresh
random model, read-only HumanDB and empty corrected SpecialistDB snapshots,
corrected Malom reads, no optimiser or checkpoint, identical before/after
model and data identities, and at most 227,160,000 requested search nodes.
Its content-addressed plan, production-route controls, strict preflight,
runner, atomic publisher, and focused tests are implemented and published in
commits `913abc7` and `70fcd3c`. The prescribed command passed from clean
`dev == origin/dev == 70fcd3c` and returned
`ready_for_authorized_probe`, while explicitly returning
`launch_authorized=false`. The
[readiness record](../evidence/sanmill-no-update-integrated-route-probe-v1-readiness-2026-08-08.md)
binds the source, plan, runtime, model, data, rule, host, and no-search route
identities. The documentation-only evidence commit must be published and the
preflight repeated from that final clean tip before a launch may be requested.
That commit was published as `98dcf23`, and the repeated preflight passed. The
owner then authorized the frozen command once. The attempt failed closed with
`SanmillBridgeError: Sanmill and NMM board mirrors diverged` while committing
a searched opponent turn. It published no completed report, left no Sanmill
process, and did not change either read-only database or create sidecars. A
post-failure preflight again passed. The current publisher retains no partial
schedule ledger, and the exception lacks the two board projections, history,
actions, and schedule identity, so neither the failing game index nor the
underlying state-field difference may be inferred. The
[failure record](../evidence/sanmill-no-update-integrated-route-probe-v1-failure-2026-08-08.md)
owns the exact boundary. The one-run probe authorization is consumed; no retry
or training run is authorized. Follow-up commits `6fdd662` and `5efae25` add
structured, host-path-free mirror context and an atomic
no-overwrite failure quarantine containing the completed-sample prefix and
failed schedule entry. The relevant 67-test group and the mandatory 103
Malom/provenance tests with 498 parameterized subtests pass. These changes do
not diagnose the root cause or retroactively select a failing game. The
failure and diagnostic-hardening chain is published through `b1a56b6`. A
separate content-addressed minimal diagnostic now selects only exact parent
schedule index zero, with one game, at most 120 logical plies, 60 searches,
and 60,000 requested node ceilings. Index zero is the smallest exact prefix,
not an assertion about where the historical run failed. The diagnostic is
implemented at local commit `bdd6ed1`, with plan identity
`5554489e3278dca88cc4f816e97ced1bdf17e7a89b0e4c02991c808d7087e4b0`.
Its local read-only preflight passed the model, data, runtime, plan, output,
and two-ply zero-search route gates but correctly reports the source as
unpublished. The readiness verdict is `needs_decision`. It remains unlaunched;
publication, a final preflight from `dev == origin/dev`, and a new one-run
authority are still required.

At that historical checkpoint the one-run calibration authority was consumed,
and no integrated route probe, additional smoke, or long run was authorised.
Probe verification then comprised 62 focused tests plus the mandatory 103
Malom/provenance tests and 498 parameterized subtests; those results belong to
the probe readiness record, not to the older calibration test totals recorded
here. The later authorization is the bounded contract linked near the start
of this handover.

The scoped Stage-0 evaluation `dev-v4-formal-paired-eval-v1` completed on
23 July 2026 with protocol decision **`accepted`**. Expert review had rejected
the 64-position corpus and synthetic
one-endpoint-per-named-line alternative and established that
`policy-argmax-v1` zeroes a lookahead feature block used during training. The
draw-lifecycle and partial-ledger restart defects found in the paired runner
are repaired; deterministic start reuse is now rejected; and new specifications
bind the clean Git commit, runtime, device, route, components, and feature
contract. The owner reviewed all 107 generated candidates, requested removal
of original review position 101, and accepted the remaining 106. The resulting
run was a 106-start, placement-only Stage-0 training-signal diagnostic against
scratch initialization. Its replacement corpus and PNG package were
regenerated and audited. A clean read-only audit reverified the corpus,
bundles, isolated
targets, runtime identity, and in-memory specification; 28 focused readiness
tests pass. On 23 July 2026, the product owner explicitly authorized the exact
CPU freeze and 212-game run. Independent recomputation verified 193 candidate
wins, 8 draws, 11 losses, pair-score-difference mean `0.8584905660377359`,
and interval `[0.7972174156720373, 0.9197637164034345]`. This is not a formal
strength or promotion result, and it does not authorize training, a rerun,
promotion, or publication.

Post-Stage-0 preparation is now complete up to new product choices. A strict
training-route bundle preserves the final policy, its six-game-old frozen
target, HumanDB continuation, final SpecialistDB counterfactual features,
corrected Malom early termination, and the historical rollout evaluator. A
real read-only load verified all four artifact identities. A separate
64-position draft covers placement/movement/side-to-move flying as 22/21/21,
is White/Black balanced 32/32, has no exact HumanDB or final SpecialistDB
matches, and includes 64 individual PNGs plus six inspected contact sheets.
The source is seeded legal TGF rules replay, not expert play. The corpus remains
unfrozen and unapproved; no post-Stage-0 candidate game has been run. The Mill
expert has now completed a quick first pass over all 64 panels, supplied a move
for each, identified several unlikely or poor states, judged the spread useful
overall, and proposed additional tactical Mill-choice positions. A product
freeze decision remains; the review is not a blanket acceptance or rejection.

Sanmill bridge v2 also passed on 25 July. It uses the pinned versioned strict
error, logical-turn, and `statejson` interfaces instead of the historical
assertion build. Two fresh processes reproduced the same 57-turn rule-terminal
game, and rule, action, history, aggregate-budget, and local performance probes
passed. This was infrastructure evidence only: no candidate was loaded and
formal candidate-versus-baseline evaluation remains stopped.

The maintainer's `main` history through `67af016` and the 21/22 July staged
upload were integrated and audited without activating their databases or
checkpoints. Later `main` history through fetched tip `bc46b51e` was reviewed
commit by commit on 7 August but was not merged or cherry-picked. The active
decision record is
[`main-integration-audit-2026-08-07.md`](../evidence/main-integration-audit-2026-08-07.md).
The rebuilt HumanDB has current label metadata and matched 30 deterministic
Malom probes; the rebuilt SpecialistDB has current metadata and zero Malom
labels but retains 2.1 million empirical positions. Seven updated checkpoints
remain weights-only maintainer-`main` artifacts with unknown corrected-data
lineage. The older v2a trainer fork is preserved but quarantined on `dev`, and
the imported in-place SpecialistDB clearing tool has been made non-destructive.

### 9 August reward-shaping successor status

Observation facts: the retained-v2 held-out candidate is behind its frozen
Sanmill endpoint. A first-transition audit found 19 candidate WDL downgrades;
16 were complete Mill-forming turns. The full-oracle audit found at least one
exact-WDL-preserving alternative in every state, while the historical trainer
awarded `+0.25` to each of the 16 contradictory Mill formations. A no-update
production-component probe reproduced a legacy reward total of `+4.0` and a
corrected `malom-preserving-only` total of `0.0` on those states, without
loading a candidate or changing an action, board, database, checkpoint or
weight.

Hypothesis: unconditional Mill reward is one causal contributor to the later
preference for visually decisive but value-downgrading Mill turns. Removing
only contradictory Mill reward should lower the exact-Malom downgrade rate
without crossing the existing fixed-state policy-health limits.

Supporting evidence: local commits `704b3a1`, `d90aced`, `3292107` and
`db68937` implement, expose, probe and record the one-factor reward change.
Commit `40370d0` requires an explicit managed-plan seed. The initial experiment
contract is `badcee6`; `bdf42e9` removes four whitespace-only EOF defects; and
`9e5df00` binds the current contract to CUDA and adds fail-closed six-arm
preparation. Commit `b38afe2` separates the 5,000-game trainer schedule horizon
from a hard 500-game controller completion bound, preventing one arm's
authorization from reaching a second segment. Commit `e46359d` adds exact
phase-specific support denominators, and `e200cb1` adds a fail-closed six-arm
result analyzer with complete-window curves and paired-seed decisions. Its
manager subprocess boundary is fixed by `c7c9be8`, which reserves stdout for
one JSON document and sends incidental diagnostics to stderr. The first
published preparation attempt at source `ca179f6` stopped at that boundary;
its partial control directory and byte-identical empty SpecialistDB are kept
under ignored quarantine
`out/quarantine/mill-bonus-ablation-preparation-failed-ca179f6-20260809T062958Z`.

A second source-only attempt at published `649abba` also failed closed before
authorization or training. Its preflight inherited the machine-local historic
output directory instead of the proposed first segment directory, and the
SQLite read-only probe created WAL/SHM sidecars beside the copied empty
SpecialistDB. The five partial files are retained under
`out/quarantine/mill-bonus-ablation-preparation-failed-649abba-20260809T064645Z`;
the quarantined main database still has template SHA-256
`5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`.
All formal control and arm-database targets are absent again.

A third preparation attempt at published `02bb1a3` also stopped before
authorization or training. The first arm's trainer preflight correctly returned
`needs_decision`, with zero technical errors and the absent product
authorization as its only unresolved decision. The preparer incorrectly
required `ready_for_long_run`, so it stopped after creating only the first
unlaunched plan and its byte-identical empty database. Those files are retained
under ignored quarantine
`out/quarantine/mill-bonus-ablation-preparation-failed-02bb1a3-20260809T073257Z`.
Local commit `947a3fe` now accepts exit code 2 only for this exact technical
preflight state, binds the clean Git source, resume configuration, first-segment
identity and isolated output path, and fails closed on any error or additional
decision. Result validation preserves the same authorization-gated meaning.

Local commits `f305ad0`, `1a47d52`, `7aa4e80` and `3bafbf3` respectively make
SpecialistDB preflight immutable and sidecar-closed, reserve trainer preflight
stdout for JSON, bind preflight to the actual first managed segment, and
preserve both subprocess output streams on failure. The current source-only
plan identity is
`5202ab904c6761f645fc2efdbc2d6788979999814a25049883f8a652beecf9e3`.
The post-fix local gate passes `239` focused Generalist, reward, observability,
trainer, manager, checkpoint, exact-resume, policy-health and Sanmill-referee
tests plus `41` route-contract tests. The separate mandatory Malom, DB-teacher
and label-provenance gate passes `103` tests plus `498` parameterized subtests.
The authorization-gated preparation and result paths additionally pass `56`
focused readiness, result, contract and Generalist-preflight tests.
Ruff passes the changed modules; the trainer retains its separately known
legacy lint baseline, and the changed trainer lines pass with those pre-existing
codes excluded. `git diff --check` also passes.

An exploratory run of the older `test_sanmill_uci.py` gate produced four
fail-closed local-installation failures because that suite still pins an older
Sanmill bridge source scope while the checkout is intentionally at the newer
training-referee commit. Its other tests passed. This is not a failure of the
current `sanmill_training_referee` route, but it must not be reported as a full
repository green baseline. No managed arm, authorization, segment, checkpoint
or game exists.

Counter-evidence and limits: the no-update result proves reward-component
behavior, not a learned causal effect. The inspected 29-state diagnostic is
not supervised validation or held-out strength evidence. The failed retained
candidate had only one training seed, and other policy, feature or
optimization effects may also contribute. The current 64-start evaluation was
used to find this mechanism and cannot later be relabelled as an independent
strength gate for the selected successor.

The six-arm experiment above was subsequently frozen with fresh seeds 45, 46
and 47, explicitly authorized once, and completed at clean published source
`fb9b7036e4a08e92331491ed67d3bfdcdfc7bf2f`. Its immutable result identity is
`1118b9ace643f2cdaa14c88bf48676d2460c589d184299cf23d564e0311a915d`;
the result file SHA-256 is
`c26d4ebe890fdc06e88be639b63979b10b964fd1322d03e8ee80c14e2ba49020`.
The preregistered verdict is `inconclusive`: two of three tail comparisons
favoured the downgrade penalty, but the median reduction was only 0.6142
percentage points against the frozen two-point gate. Repetition draws rose in
all three treatment pairs, and the Sanmill-facing reduction was small. The
one-run authorization is consumed. Do not extend, rerun, promote, or lower the
gate after seeing this result. The full observation, hypothesis, support,
counterevidence, per-class metrics and decision boundary are in
[`sanmill-malom-downgrade-penalty-ablation-result-2026-08-09.md`](../evidence/sanmill-malom-downgrade-penalty-ablation-result-2026-08-09.md).

The current successor uses a direct policy auxiliary rather than increasing
the scalar reward penalty. It labels every complete legal turn by exact Malom
WDL and minimizes the negative log of the total policy probability assigned to
all WDL-preserving actions. It does not choose a preferred action among tied
preserving turns, and all-safe states contribute no preference gradient.
Commits `52fc97f`, `5997d00` and `fcdce2a` implement the loss, exact action-label
contract and trainer integration. Commits `9da12fb`, `b2ccecf`, `5c043cf` and
`a36aff4` add and record the no-update gradient probe and bind the coefficient
through managed plans.

Observed probe facts: each of seeds 48, 49 and 50 labels 1,583 actions on the
same 64 inspected development states, with 1,168 preserving and 415
downgrading actions. Twenty-nine states are informative and 35 are all-safe.
All gradients are finite and their descent direction analytically increases
preserving mass. The direct float32 step at learning rate `1e-4` nevertheless
rounds below observable probability resolution. This supports correct wiring
and direction only; it is counterevidence against claiming an effective Adam
coefficient, learned improvement, generalization or strength. The tracked
probe manifest SHA-256 is
`1d7784dfabf8aa59d70adc310d0279b03a08863e69e2a5a009339d9f13394092`;
the ignored raw report SHA-256 is
`ad1e6e3ee7596a872d3129e623d377e083c439f6bcbee23705a10bf8ced1b003`.

The next bounded experiment is the unlaunched
[`sanmill-malom-policy-auxiliary-calibration-smoke-v1`](../experiments/sanmill-malom-policy-auxiliary-calibration-smoke-v1.md).
Its plan identity is
`bdee5fc858b065203d61edbd199e4e77be32262c3fb75a72172e4f7489542aba`.
It compares coefficients `0.00`, `0.03`, `0.10` and `0.30` at fresh seed 51,
one sequential 100-game segment per arm, 1,000-node Sanmill search, and one
isolated empty SpecialistDB per arm. The complete ceiling is 400 games, two
active hours and 11,520,000 requested Sanmill nodes. Its decision uses
scratch-normalized within-arm checkpoint change followed by a
difference-in-differences comparison with control, because each arm's
SpecialistDB evolves independently. It does not select by training W/D/L.

Commits `81f511e`, `553be79`, `2969bf7`, `31e9764` and `c062ed2` freeze the
contract, fail-closed preparation, numeric engineering gates,
SpecialistDB-confound correction and immutable result analyzer. These commits
are local and unpublished at this handover update. The technical gate therefore
correctly stops because `dev != origin/dev`; no arm database, managed plan,
authorization, segment, checkpoint or training process has been created. After
ordinary publication, run the preparer once to create and audit the four
ignored plans and database copies. A passing report may reach only
`ready_for_product_authorization` with `launch_authorized=false`. Starting the
four-arm sequence still requires a separate explicit product authorization;
there is no automatic extension, continuation, promotion or publication.

Read
[`docs/local-training-layout.md`](../local-training-layout.md) for the relative
storage map and machine-local lookup keys, and
[`docs/v5-specialist-plan.md`](../v5-specialist-plan.md) for the modular v5
design and its owning subdocuments.
Machine-specific absolute values are intentionally kept only in the ignored
`data/training_paths.local.json`. Path names shown in committed documents are
relative to the repository root; Markdown link targets are relative to their
containing files so that they render correctly.
The dated monolithic v5 snapshot is historical. The current v5 entry point
records that the coarse sector-corrected decoder is already repaired while the
complete comparator, rule-history, proof, and release questions remain open.

## Repository and Workspace Boundary

- Repository: the Git repository containing this document
- Branch: `dev`
- Remote: `origin`, using
  `git@github.com:benmarkbrandwood-blip/NMM_LLM.git`
- Intended execution host: Windows 11, without a WSL requirement
- Parent directory: data container only; it must not become a Git repository

The current Codex task is already open at the repository root, as confirmed by
`git rev-parse --show-toplevel`. Future tasks should use the same workspace
boundary and begin by reading the repository's [`AGENTS.md`](../../AGENTS.md)
and this file. Consult
[`docs/local-training-layout.md`](../local-training-layout.md) when the
storage relation or machine-local configuration key is needed.

## Git Synchronisation Completed

The earlier rewritten-but-patch-equivalent divergence has been resolved. Before
the update, `5880316` was patch-equivalent to remote `9e46334`, `5a17738` was
patch-equivalent to remote `643a5e7`, and local `06598c9` was the additional
PyO3/Python 3.13 compatibility change.

On 20 July 2026, the owner explicitly authorised local `dev` to replace the
remote branch with `--force-with-lease`. The lease was pinned to remote tip
`643a5e766768239bac030d32afc8915f5f90a570`, and the update completed
successfully. Immediately before the documentation commit containing this
handover, both `dev` and `origin/dev` pointed to:

```text
06598c9dabeabdd613070d3bbc8634bc2f2b3977
```

`git rev-list --left-right --count dev...origin/dev` returned `0 0`. The
handover commit `8751da4` was subsequently pushed and is now the recorded
`origin/dev` tip. Local `dev` then added the independently tested auto-resume
and temperature commits `5eadb4e` and `006715b`, the component-disable commit
`24be10b`, the experiment-definition and smoke-evidence commits `80f4a1f` and
`53d86d1`, and the follow-up maintenance commits through `9c7dceb`. Later local
infrastructure commits through `59a4cf9` add exact-resume hardening, bounded
segments, checkpoint migration and validation, self-describing evaluation
bundles, paired promotion evidence, and the first author-asset refresh. Inspect
the live graph rather than relying on that intermediate snapshot. Later local
commits through `4893fb6` add fail-closed pure-RL controls,
deterministic fixed-node heuristic work with actual-node evidence, and
product-authorized managed training supervision. Inspect the live local and
remote graph before making synchronization claims. The completed
force-with-lease approval is not standing permission for a future push or
history rewrite; obtain fresh authorisation when such an operation becomes
necessary.

At the 21 July formal-evaluation review, local `dev` was at
`bc92d3346c8da55b6cdf1d56b20b7cab10317c75`, one commit ahead of
`origin/dev`, with modified and untracked experiment documents and draft
artifacts. That is not a clean reproducible evaluation freeze point. Recheck
the live graph and working tree before relying on this snapshot.

## 22 July Main Integration and Upload Audit

The maintainer's active `main` tip was initially `b9a13ce` and advanced during
verification to `67af016`. Its history was not compared to `dev` by a blind
tip diff: commit-graph inspection showed that `9d09851` was a one-parent
import close to older `dev` commit `0ad5991`, followed by the
maintainer's plans, assets, and v2a work. Merge commit `8717f1c` records the
integration. All seventeen snapshot conflicts retained the newer `dev` side;
the non-conflicting maintainer artifacts were preserved for audit.

Final merge commit `4593034` imports `67af016`'s v2a per-difficulty
best-rate persistence while retaining the quarantine. It fixes one legacy
best-save threshold across restarts but does not provide the complete `dev`
exact-resume state and therefore does not authorize running v2a.

Two independent safety commits follow that merge:

- `f7c5b19` makes SpecialistDB label clearing an explicit, source-hash-bound
  copy migration and adds three regression tests;
- `76f3ff3` quarantines the older main-lineage v2a runtime entry point and
  removes its unsafe smoke/resume examples while retaining the source for
  reviewed feature porting.

The rebuilt database candidates are intact under the ignored archive
`data/backups/maintainer_upload_20260721`. The former sibling `Mills`
directory was renamed `maintainer_inbox` on 25 July and left empty for future
incoming deliveries. The HumanDB sidecar hash matches, both post-move SQLite
quick checks pass, and 30 sampled HumanDB labels match the current corrected
Malom adapter for W/D/L and DTW. This supports the archived candidate but does
not replace the active HumanDB or change the completed baseline. The archived
SpecialistDB's retained empirical history also makes it a different
experiment input from the fresh baseline DB.

The imported retraining plan remains a proposal. Current code and the v5 design
resolve its apparent model-contract ambiguity locally: Sentinel stays DB-free
with oracle slots masked, the proposed next-move ranker is a separate
HumanPolicy path rather than ValueNet v2, and GapNet retains its implemented
current-position quality-gap target. The imported checkpoints still lack
corrected-data lineage, but no maintainer reconstruction is needed unless a
future experiment proposes to adopt them.
See
[`docs/evidence/main-integration-audit-2026-07-22.md`](../evidence/main-integration-audit-2026-07-22.md)
for exact hashes, counts, conflict policy, and question boundaries. The prior
draft message to the maintainer can wait; a shorter evidence-based question set
should be sent only after this integration audit is complete.

## Environment State

The current local environment was checked as follows:

| Component | State |
| --- | --- |
| Python virtual environment | `.venv`, Python 3.13.1 |
| PyTorch | Importable |
| Native `nmm_core` extension | Importable |
| ChromaDB | 1.5.9, importable |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported memory |
| NVIDIA driver | 610.74 |

`python -m pip check` reports no broken installed requirements. Modules such as
`sentence_transformers`, `faiss`, and `sklearn` are not installed, but they are
not declared by the repository's two requirements files and did not cause the
now-resolved test-collection failures. Do not call them missing project
dependencies without first defining a feature that requires them.

Commit `06598c9` records successful `cargo check --locked`, editable
installation of the CPython 3.13 extension, and fifteen native parity tests.
The extension was rebuilt after the fixed-node API change; an end-to-end probe
used exactly 25,000 requested nodes twice and selected the same move both
times. The full Rust unit suite reported `24 passed`.
The focused Python verification was re-run during this handover:

```text
102 passed, 498 subtests passed
```

The command was:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py -q
```

`scripts/train_s_gen_v2.py --help` also completes successfully. Follow-up
maintenance on `dev` recalibrated two stale GameAI tactical fixtures against a
legal terminal-mill position and replaced tests that depended on an untracked
`data/games` corpus with deterministic JSONL fixtures. The current Sentinel and
TrajectoryDB loader tests therefore execute rather than skip when that local
directory is absent.

The four historical collection errors were stale tests rather than missing
active production interfaces. Commits `08507e0`, `c12b935`, `cc07a81`, and
`af17232` align, respectively, the legal-move, Sentinel feature-builder,
Sentinel label, and Sentinel model tests with the current contracts. A fresh
collection at `af17232` reports `925 tests collected`; the updated files and
their adjacent contract chains report `67 passed`. The mandatory
Malom/provenance rerun again reports `102 passed, 498 subtests passed`.

The first complete run made possible by those collection fixes reports:

```text
47 failed, 878 passed, 498 subtests passed in 3023.67s (0:50:23)
```

That result remains the first credible failing baseline. The failures were then
resolved against their owning contracts rather than skipped, weakened, or made
dependent on network downloads:

- `9e3bda7` gives Stage 3/6 a deterministic local embedding, explicit Chroma
  client lifetime, and a declared Chroma version with public close support;
- `3a57564` derives scaffolded lookahead width from the loaded model width and
  rejects incompatible explicit advisors before tensor execution;
- `39cf56e` preserves a training sample when ValueNet receives a singleton or
  very small dataset;
- `becfe17` and `c56f03a` assert the B-40 and SE-10 feature contributions
  directly instead of assuming no later tactical term also applies;
- `cee5e45` preserves the objective forced-dead-block label on the default V2
  path while continuing to suppress V1 score explanations;
- `901909a` replaces the B-78 pseudo-quiet fixture with an actually legal quiet
  full-game-DB move;
- `f13e9e9` gives the exact B-66 move regressions deterministic node budgets;
- `799a944` checks the B-22 defensive postcondition over all opponent replies;
- `c4c3454` isolates the documented contested-mill late-game contribution; and
- `08e8c33` compares WDL distributions by turn bit, the colour-symmetry
  invariant appropriate for side-to-move table values.

After the Chroma, scaffolded-width, and ValueNet changes, the intermediate
complete run had only the eight independently diagnosed tactical/endgame
failures: `8 failed, 919 passed, 498 subtests passed`. The affected non-solver
files then reported `97 passed`, and the complete 3v3 builder file reported
`13 passed` in 22 minutes 44 seconds.

A final clean-working-tree run at code HEAD `08e8c33` is the current complete
Python baseline:

```text
927 passed, 498 subtests passed in 3098.55s (0:51:38)
```

There are no remaining collection or runtime failures in that suite. This does
not change the separate experiment freeze, provenance, or training-authorization
gates.

This author-bundle review reran the current trainer contract, preflight,
checkpoint-envelope, exact-resume, launch, temperature, data-contract, and
paired-evaluation tests at code HEAD `59a4cf9`: `113 passed`. The mandatory
Malom/provenance group again reported `102 passed, 498 subtests passed`.
The earlier `tests/test_scaffolded_policy.py` result of `22 passed, 3 failed`
identified the real feature-width mismatch fixed by `3a57564`; it is historical,
not a current interface failure. The absence of a regression for PPO old/new
log-probability temperature consistency remains a separate coverage observation,
not a failing test.

## Data and Model State

Files from the Google Drive deliveries are no longer held in an ambiguous
sibling staging directory. Reviewed assets are in their repository-local or
external destinations, while the inactive 21 July database candidates are in
`data/backups/maintainer_upload_20260721`. The former `Mills` directory is now
the empty `maintainer_inbox` for future deliveries. Its role and the relative
destination map are recorded in the
[`docs/local-training-layout.md`](../local-training-layout.md) path list.

Available assets include:

- the 738,091,008-byte HumanDB and 95,389 human-game `.jsonl` files;
- fourteen endgame WDL tables and `fullgame.bin`;
- the complete external Malom directory, with 512 files totalling
  83,582,223,577 bytes;
- Sentinel `best.pt`;
- historical opening, midgame, endgame, and generalist checkpoints;
- value-net and gap-net artefacts.

The assets are present, but they are not all equally trustworthy:

- HumanDB human frequencies, outcomes, and counts remain useful.
- HumanDB's unversioned historical Malom columns are masked by current readers.
- `data/specialist_db.sector_corrected.sqlite` is trusted completed-run state.
  It began empty, but the 5,000-game managed baseline populated it; do not
  describe or reuse it as an empty input for another fresh experiment.
- Both legacy SpecialistDB deliveries are isolated in the ignored backup
  directory and must remain read-only.
- Historical checkpoints and nets pre-date the corrected decoder/provenance
  migration. Retain them as exploratory baselines; do not claim that they were
  trained from corrected labels.
- The original maintainer describes the endgame tables and `fullgame.bin` as
  outputs of their backwards solver. That is a provenance statement, not an
  independent correctness check. A follow-up read-only inventory and sampling
  audit found missing table coverage plus concentrated unknown entries in four
  loaded tables; see
  [`docs/endgame-training-feasibility.md`](../endgame-training-feasibility.md).
  That diagnostic is not a full differential proof. Record hashes and complete
  the reviewed validation before using those files as authoritative labels or
  acceptance evidence.
- `gap_net_path` is deliberately blank in the local path configuration even
  though the files exist. Do not enable it until its label provenance is
  reviewed.

## Completed Correctness Work

The following commits on `dev` form the relevant correction chain:

| Commit | Result |
| --- | --- |
| `44a0fd3` | Corrects sector-adjusted Malom value decoding |
| `98ff63a` | Makes Mill formation plus capture an atomic Malom move query |
| `803eee8` | Resolves rules-terminal states before tablebase lookup |
| `216a77f` | Compares moves with complete oracle values rather than incomplete child fields |
| `8da033e` | Rejects impossible positive move-quality deltas |
| `7cf7725` | Ignores recursively imported game data and SQLite training data |
| `5880316` | Versions persisted Malom labels and gates every direct consumer |
| `5a17738` | Covers suffixed SpecialistDB SQLite files in `.gitignore` |
| `06598c9` | Updates PyO3 to build `nmm_core` under Python 3.13 |

The decoder and capture semantics were also checked against real Malom files:
961 sampled positions matched the corrected reference projection. This
external comparison supports the result, but the project tests and this
repository's rule semantics remain the primary acceptance evidence.

## Persisted-label Behaviour

Current code uses `malom_label_version=sector-corrected-v1` as the trust gate.
It has the following intended behaviour:

- a new or unlabelled SpecialistDB may adopt the current version;
- a labelled but unversioned SpecialistDB is treated as legacy;
- empirical game statistics may still be read from legacy data, but legacy
  Malom priors are ignored;
- new Malom labels cannot be appended to a legacy labelled database;
- HumanDB readers preserve human statistics whilst masking legacy WDL/DTW;
- HumanDB builders refuse to mix corrected labels into a legacy labelled DB;
- direct gap-dataset and trajectory-label consumers require current metadata.

The active HumanDB has 1,560,069 labelled position rows and 1,691,422 labelled
move rows but no label-version key, so its Malom fields are intentionally
untrusted. The active corrected SpecialistDB began empty. After the completed
managed run, a 22 July read-only audit found SHA-256
`1203FC73CD7D0A06E2DD1FFACED5B031DFF8BD704E22B34BA02182FF3865614D`,
SQLite `quick_check=ok`, 132,182 positions, 41,904 current-version Malom
labels, 916 winning lines, no preferred plays, and lineage root
`managed-v4-baseline-v1-segment-0001`.

The 20 July author update added 406 valid human-game JSONL files. Their content
matches `human_games_94559.zip`, and the import manifest grew from 94,134 to
94,540 entries. Four added records have an empty `moves` list and were retained
unchanged from the source package. `data/human_db.sqlite` was not rebuilt, so
its 94,429-game inventory still represents the earlier corpus.
The source ZIP is archived outside Git at
`../human_database/human_games_94559.zip`; its SHA-256 is
`45523234085518031A09725A2DBCAB395E55026787E420A04C37EBA10A0E4D07`.
Do not run the current builder's `--update` mode blindly: all 94,983 existing
`processed_files.file_path` values use the author's `/home/...` absolute path,
so Windows paths would be treated as new files and their statistics would be
added again. Migrate those keys or perform a controlled rebuild before adding
the 406 games to HumanDB.

The accompanying 268,521,472-byte SpecialistDB passed `integrity_check` and
contains 1,954,437 positions with 339,904 labels, but it has no `meta` table and
therefore no trusted label version. It is quarantined as
`data/backups/drive_import_20260720/specialist_db.sqlite.legacy-author-update-20260720`
with SHA-256
`5C6A4EA1ACFB90BF05248580A07DAE7CF4645C09E5A4A69E2EC89EA9EE41811B`.
The active corrected database was not replaced by that author update. The
recorded pre-run SHA-256
`CB4153A14752357587890EB5F8B655AB04AF8242E43BE1C80D4847A11D101A94`
was subsequently superseded by legitimate managed-run writes; its current
identity and counts are recorded above.

The downloaded `build_endgame_db.py` and `build_fullgame_db.py` are byte-for-byte
identical to the repository copies. The downloaded `build_human_db_sha.py` is
an older version that lacks the repository's Malom label-provenance guard, so
it was not copied over `tools/build_human_db_sha.py`.

## Source-note Evidence Boundary

The machine-local `Notes.md` and its screenshots are historical operator
observations, not a specification, test result, or source of authoritative
labels. Path and asset claims in that note were checked independently before
being recorded here. Preferences such as "the generalist is the way to go",
reported difficulty levels, proposed specialist grading changes, expected
Sentinel improvement, and possible trap training remain hypotheses until a
reproducible experiment supports them.

The screenshots also pre-date the corrected Malom decoder, so their Malom
arrows cannot be used as oracle evidence. They do preserve useful diagnostic
leads:

- in one recorded position the policy/Overseer assigned `100%` to `f2` while
  the displayed Sentinel score was `54%`; displayed alternatives included
  `d3` at `92%` and `d1` at `82%`;
- two other `100%` selections coincided with the highest displayed Sentinel
  score, and another position showed a distributed policy, so the screenshots
  do not establish universal policy collapse or universal disagreement;
- the aggregate dashboard shows large policy/value-loss spikes. Its green
  vertical markers are difficulty advances generated by
  `tools/plot_specialist_training.py`, not recovery events.

The note's report that the midgame specialist and generalist reached level 7
and approached level 8 is therefore historical context only. The suggestion
that opening and endgame specialists need different grading is an experiment
proposal, not a diagnosed cause. Before acting on either claim, replay recorded
FENs with a pinned checkpoint and log policy entropy, top-one mass, Sentinel
rank, legal-move coverage, and corrected oracle values; evaluate strength only
with frozen, colour-swapped matches and intervals.

### Newly supplied author-`main` Generalist evidence

The owner confirms that the newly supplied Generalist checkpoints, JSONL logs,
plot, and browser screenshot all came from the maintainer's continuing `main`
training. They are not `dev` artefacts even though a legacy checkpoint embeds a
host directory containing the word `dev`. Exact hashes and the read-only audit
are recorded in
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md).

The delivered `best (copy).pt` is a finite, legacy weights-only `s_gen_v2`
checkpoint at game 17,400 and difficulty 9. That supports the maintainer's
correction from “10/20” to “9/20”, but its exact source commit and full launch
contract remain unknown. It has no optimiser, RNG, data identity, or complete
trainer state and must never initialise or resume the fresh `dev` experiment.

The accompanying log supports a narrower version of the maintainer's policy
observation. Across its first and last 500 rows, `policy_top1_rate` rises from
about 0.42 to 0.84 while entropy falls from about 1.55 to 0.34. However,
`heuristic_top1_rate` also rises, from about 0.30 to 0.51. These fields measure
whether the sampled move equals each argmax; they do not measure strength or
isolate positions where policy and heuristic disagree. The 10,547-row file
also contains duplicate game numbers, six counter regressions, and a mid-log
opponent-schedule change, so it is an appended operational history rather than
one frozen experiment.

The 1,190-row update log raises a separate stop condition for PPO reuse. Its
policy loss has median about `9.88e7`, reaches about `1.71e29`, and ends around
`7.80e21`, while value loss remains ordinary and all values remain finite. The
inspected trainer family records old PPO log probabilities from
temperature-scaled logits but recomputes new log probabilities without that
temperature. The missing exact `main` commit prevents attributing every spike
to that mismatch, but PPO remains quarantined for the first `dev` baseline
until a deterministic ratio test and reviewed fix exist.

The latest browser screenshot proves only that the Generalist checkbox was
selected during one manual game. It does not freeze the actual feature inputs,
opponent, position, colours, or work budget. The author log has only
`phase_bucket=main`, so the reported strong opening and weak endgame profile
still requires a phase-stratified replay before it can guide architecture.

The author-update SpecialistDB does contain 27 promoted preferred plays, which
supports the narrow “favourite plays” statement. It still lacks a `meta` table
and a trusted Malom label version, so it remains quarantined and read-only. The
maintainer also explicitly said the internal endgame files had not been
checked, consistent with keeping them disabled as authoritative inputs.

## Generalist Trainer Corrections

### Auto-resume follows the configured output directory

The machine-specific configuration sends new output to:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected
```

Commit `5eadb4e` changes `_choose_resume_path()` so `--auto-resume-best` reads
`best.pt` from the resolved `args.out_dir`; it no longer falls back to the
historical fixed directory. Regression tests cover explicit-resume precedence,
the configured output path, and isolation from the old directory. The fresh
baseline still intentionally omits both `--resume` and `--auto-resume-best`.

### The CLI temperature schedule controls the loop

Commit `006715b` passes `--temp-start` into the schedule for both fresh and
resumed game counts. Temperature reaches the fixed `0.20` endpoint after 80 per
cent of `--max-games`. Recovery no longer resets temperature: it still restores
the selected weights and applies the existing draw-penalty grace, but
exploration stays on the global schedule. Focused tests cover a custom start,
ordinary decay, endpoint clamping, and the unchanged default schedule.

Commit `fe0b1f1` additionally makes `--temp-start` reject zero, negative, and
non-finite values during argument parsing, before training resources are
opened. Focused tests cover valid decimal and exponential forms plus zero,
negative, `NaN`, infinities, and non-numeric input.

### Final checkpoint reporting matches repository state

Commit `bf9472c` always reports the final `latest.pt` path and reports
`best.pt` only when that file actually exists. The best snapshot is optional:
it is created only at a logging checkpoint after at least 10 heuristic games
when the current win rate strictly improves on the prior best at that
difficulty. Regression tests cover both reporting outcomes and all sides of
that gate.

## First Dev Experiment Decision

The owner selected `dev-v4-malom-corrected-fresh-v1`: a fresh-initialised,
Malom-corrected v4-style Generalist baseline. It does not load the author's
continuing `main` checkpoint, does not use automatic resume, starts with an
empty `sector-corrected-v1` SpecialistDB, and explicitly disables the legacy
Sentinel, ValueNet, and GapNet. The trainer exposes `--no-sentinel`,
`--no-value-net`, and `--no-gap-net` so this choice overrides machine-local
configured paths rather than depending on missing files.

The complete definition, preflight evidence, claim boundary, isolated smoke
command, and result are in
[`docs/experiments/dev-v4-malom-corrected-baseline.md`](../experiments/dev-v4-malom-corrected-baseline.md).

The smoke ran from clean commit
`80f4a1fe525d98706b1b0913083f2c2067f8bf66`, completed one 33-ply game on CUDA,
and exited successfully. It started from scratch, disabled all three legacy
learned inputs, loaded Malom and HumanDB, wrote a trusted disposable
SpecialistDB, and left the active empty baseline DB unchanged. This is
integration evidence only, not strength evidence.

The generated `latest.pt` is readable, but the final console message named
`best.pt` even though no such file was produced by the one-game run. This does
not invalidate the historical smoke. Commit `bf9472c` fixes the message; a
one-game run is now explicitly reported as having no best checkpoint.

The historical smoke's `latest.pt` is a pre-envelope weights-continuation
snapshot; `best.pt` remains optional model-selection evidence. Subsequent
infrastructure now emits a version-2 checkpoint envelope and has proved bounded
exact-resume parity for model, optimiser, scheduler/scaler, counters, rolling
histories, curriculum, target state, component RNGs, data cursor, log state,
and SpecialistDB identity. Initial launch still uses explicit `fresh` mode.
Unscoped automatic resume remains forbidden. Within one separately authorized
immutable managed plan, the supervisor may start a new isolated segment only
from the verified `latest.pt` of the immediately preceding completed segment,
using explicit `exact-resume`. Legacy checkpoints, including every
author-`main` file, remain weights-only and cannot satisfy that gate.

## Managed Run and Stage-0 Evaluation Completion

The separately authorized managed plan `managed-v4-baseline-v1` later
completed `completed_games=5000` and `completed_segments=20`. Its frozen
training commit is `9ee3543195255456b2b3832f8371a8f64d25a6af`, and its plan
SHA-256 is
`3f696e60c508a972dc42c79f630e90ad20e870001190321a13f0c3a12a4251c1`.
The final candidate source is
`managed_v4_baseline_v1/segments/segment-0020/latest.pt`. The candidate and
architecture-matched scratch-init evaluation bundles have both passed CPU
verification.

The paired-runner prerequisites identified by expert review are repaired.
Repetition and 50-move transitions now stop on `engine.finished` and retain the
engine's draw reason. In-progress evidence is fsynced to `<output>.partial`;
same-spec ordered hash-valid prefixes resume only missing games, malformed
prefixes fail closed, and complete evidence is recomputed before atomic final
publication. The specification also rejects duplicate starts and any pair
count above the unique corpus size. New freeze records bind a clean Git commit,
selected CPU/CUDA device, platform, PyTorch, float32, the policy route,
disabled components, and zeroed lookahead features; execution fails closed on
runtime drift. Legacy unbound specifications remain readable and recomputable
but cannot create new game evidence. `python -m pytest
tests/test_paired_evaluation.py -q` reports `15 passed`.

The first formal paired-evaluation proposal was narrowed because:

- pure argmax plus modulo start selection makes repeated starts exact copies,
  invalidating the old 64-start / 256-pair nominal sample size;
- 49 of 107 named lines have 2–42 legal endpoints because removal choices are
  omitted, one line fails replay, and one successful endpoint is terminal;
- 110 raw Sanmill Oracle keys contain 108 stable placement keys that project
  to 107 unique playable NMM positions; the other two are pending removals and
  are retained only as successor provenance;
- the proposed `policy-argmax-v1` route zeroes the 72-feature lookahead block
  supplied during training.

The completed Stage-0 diagnostic used 106 owner-accepted unique stable
Oracle-projected starts, one colour-swapped pair per start, for 212 games
against the verified scratch-init control. Sanmill documents the Oracle as
independently engine-derived, but 28 of 106 selected positions overlap
named-line trajectories and all positions are early placement. It is neither
demonstrated held out nor training-disjoint. Stage 0 therefore tests only
whether a training signal is
visible under a placement-only feature ablation; it is not a strength or
promotion gate.

The generated freeze-compatible list has canonical `start_positions_sha256`
`04bc5782ab79ebeba34d0ff91bcd40fe05e823d539b16ba234b5eedcd123bb9d`.
The review artifact's pre-freeze status was
`owner_review_complete_not_frozen`; the selected list is now frozen by spec
identity `26f80c14d70320aa025c85319791c625e821babb2e542095aeb4711d4c11d48b`.
It links 106 individual PNGs and nine contact sheets. Automated replay found 438 legal
source recommendations and one illegal `c3` recommendation. The associated
source candidate is the owner-excluded original position 101 and remains only
as provenance. Codex inspected every regenerated contact sheet, the
post-exclusion boundary images, and the excluded panel. The combined
corpus/evaluation focused suite reports `28 passed`; owner review and read-only
readiness verification are complete.

The controlling records are:

- [evaluation contract](../experiments/dev-v4-formal-paired-eval-v1.md)
- [expert decision record](../experiments/dev-v4-formal-paired-eval-v1-decision-brief.md)
- [rejected corpus and generated replacement review](../experiments/dev-v4-formal-paired-eval-v1-corpus-review.md)
- [Stage-0 readiness evidence](../evidence/dev-v4-stage0-readiness-2026-07-22.md)
- [Stage-0 result evidence](../evidence/dev-v4-stage0-result-2026-07-23.md)

The authorized run is complete and its one-run authorization is consumed. Its
`accepted` decision means only that a training signal is visible against
random initialization under the placement-only zero-lookahead ablation. Any
route-aligned or phase-covered v2 must be separately preregistered and
authorized; observations may not be pooled as one prespecified sample.

The next-evaluation preparation records are:

- [training-aligned product decision brief](../experiments/dev-v4-training-aligned-evaluation-v1-decision-brief.md)
- [phase-corpus review record](../experiments/dev-v4-phase-covered-corpus-v1-review.md)
- [phase-corpus artifact](../experiments/dev-v4-phase-covered-corpus-v1.json)
- [complete Sanmill book-path contract](../experiments/sanmill-book-path-corpus-v1.md)
- [complete Sanmill book-path artifact](../experiments/sanmill-book-path-corpus-v1.json)
- [Sanmill prefix-diversity audit](../evidence/sanmill-prefix-diversity-audit-2026-07-25.md)
- [twelve-ply layered-prefix contract](../experiments/sanmill-layered-opening-prefix-v2.md)
- [twelve-ply corpus decision brief](../experiments/sanmill-layered-opening-prefix-v2-decision-brief.md)

## Strict Sanmill Bridge Validation

The product owner deferred the current in-repository `GameAI` as the formal
baseline because its compact lifecycle does not preserve authoritative
repetition and no-capture history. The assertion-build bridge v1 passed on
23 July and remains immutable historical evidence. Sanmill then supplied
explicit `StrictFailurePolicy`, `go logical nodes N`, and `statejson`
interfaces. The current bridge v2 is pinned to Sanmill commit
`db65eb3e73189d934d615d0f47519d395193c646`, tree
`b8fa6c0119c2dec4443efc59deab8b7d835e0c88`, and ordinary Windows release
binary SHA-256
`cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`.
It did not load the candidate or run candidate-versus-baseline games.

The v2 bridge uses one thread, MTD(f), IDS, shuffling off, seed 42, fixed node
ceilings, and no wall-clock limit. `StrictFailurePolicy=true` makes rejected
histories and search failures versioned hard errors. The logical-turn path
never enters Perfect DB, patch/trap, depth-4, or random recovery. HumanDB,
Perfect DB, patches, traps, lazy AI, and `FocusOnBlockingPaths` remain inactive
in bridge search. Normal smoke turns send no positive explicit depth, so
Sanmill's non-developer `DrawOnHumanExperience` phase-depth policy remains
active.

`statejson` now supplies the authoritative FEN, complete action and logical-ply
counts, no-capture and repetition counters, legal actions, terminal reason,
standard-rule identity, and history SHA-256. `go logical nodes N` returns an
ordinary action or a Mill-forming action plus its required removal under one
aggregate ceiling. The command does not mutate engine state: NMM_LLM replays
the returned tokens and requires the resulting FEN, counts, history identity,
and outcome to match.

The first full v2 invocation stopped before writing evidence because the
adapter treated a closed terminal snapshot that retained `action=remove` as an
ongoing pending-removal contradiction. A deterministic turn-57 reproduction
proved that the completed removal had already changed the phase to
`game_over`, cleared pending removal and legal actions, and produced the
authoritative `loseFewerThanThree` result. Commit `70de75b` permits only that
terminal combination; the ongoing-state check remains strict. The focused
suite then reported 41 passed, readiness was repeated, and the recorded smoke
passed.

Two fresh processes produced the same 57 complete logical turns, 65 UCI action
tokens, eight removals, and final White win after timing was excluded. Their
semantic identity is
`ae51a16b726e7227f499f054310fed5fbd4b158d8f1b998a4d8cb65d1f7c27bc`.
Black-box probes passed for the 100-ply no-capture draw, threefold repetition,
fewer-than-three loss, compound Mill/removal, capture reset, and
`DrawOnHumanExperience` opening depth.

At a 500,000-node ceiling, the representative movement sample used 500,000
nodes in about 59.7 ms and flying used 500,000 in about 36.2 ms. Placement
completed depth 3 after 1,080 nodes and about 0.15 ms. The explicit depth-8
compound Mill probe used 11,776 nodes in about 13.8 ms. These are single-host
observations, not a frozen formal workload or latency guarantee.

The v2 evidence identity is
`b8e31cb621e95ecdf5708145c3c4c3ba43b0fbae863bd93460db1beba96cd188`.
It is bound to NMM_LLM source commit
`70de75bb8247ec6795b69045ac53558161e6c045`, the exact pinned Sanmill source
and binary, rule identity, strict contract, and corrected opening-book asset.
The complete repository suite was not rerun for this bridge-only update; the
41-test focused result does not replace the prior full-suite baseline.

On 7 August, Sanmill local commit
`a6623f88959f7453594df274fbe1f128af7ff55e` added an opt-in
`mif-stable-moving-v1` strict-referee profile. NMM_LLM-side source inspection
and a fresh black-box process confirmed origin-counted occurrences at logical
plies 0/4/8 and fail-closed rejection after the action-8 terminal draw. The
default live profile is unchanged. The implementation was not yet pushed when
reviewed, so formal referee adoption still requires remote publication, a
clean pinned release build, and a repeated bridge audit. See the
[strict-referee parity record](../evidence/sanmill-strict-referee-parity-2026-08-07.md).

Current and historical bridge records are:

- [v2 human-readable result](../evidence/sanmill-strict-uci-bridge-smoke-v2-2026-07-25.md)
- [v2 machine-readable result](../evidence/sanmill-strict-uci-bridge-smoke-v2-2026-07-25.json)
- [v2 contract](../experiments/sanmill-strict-uci-bridge-smoke-v2.md)
- [historical v1 result](../evidence/sanmill-strict-uci-bridge-smoke-2026-07-23.md)
- [historical v1 contract](../experiments/sanmill-strict-uci-bridge-smoke-v1.md)

The opening-book data defect is closed. Sanmill commit
`69d379a1a4e23395a45706df60f63282da20e85f` removed the occupied-`c3`
recommendation and added authoritative whole-asset legality tests. Commit
`6f080c5a6d15919bf0a45fa5528c45d4487a2b8f` removed a duplicate `c5`
recommendation that otherwise altered rank-biased selection weight. The final
asset SHA-256 is
`cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5`;
all 109 entries and 437 unique recommendations replay legally, with zero
duplicates. These corrections and the later strict/data-query/logical-turn
interfaces are now present on Sanmill `master`.

The missing-provider-interface and NMM_LLM paired-prefix implementation
blockers are closed. Commits `a4e166e` and `d6ea9f5` provide a strict JSONL
client and a deterministic, source-policy-explicit sampler. They verify every
source identity, FEN, history SHA-256, action count, logical count, and stable
primary-plus-removal boundary. One prefix is recorded for both games in a
colour-swapped pair. The focused Sanmill bridge/query/prefix suites report
`60 passed`; the complete repository suite at `d6ea9f5` reports
`1022 passed` and `498 subtests passed` in 3306.78 seconds.

The provisional infrastructure design still proposes 75% corrected-book
prefixes and 25% StrictSteps Perfect DB tied-best prefixes, but the code has no
default mixture. Prefixes cover exactly eight logical player moves in total:
four by each side, or four full rounds, not eight rounds. A Mill-forming move
plus its required removal is one logical move even though it uses two UCI
tokens. Both games replay the same frozen-seed prefix before strict MTD(f)
resumes. No optional database is enabled inside later MTD(f) search.

The corrected book is sparse. A fixed `pair-12` diagnostic generated the same
eight-ply book prefix in two fresh processes, but `pair-0` reached
`book_miss` before its sixth logical move. No fallback was used. Before a
75/25 smoke, use either an eligible book pair-ID set, a frozen complete-path
corpus, or an explicit per-ply source schedule. Do not interpret a book miss
as permission to switch sources at runtime.

The complete-path option is now implemented as inventory-only infrastructure.
Contract commit `8edb148` and implementation commit `024d1f8` produced a
host-path-free artifact from two fresh Sanmill processes. Both enumerations
contained the same 192 complete eight-logical-ply histories, 508 pruned
`book_miss` leaves, and zero fallback or pre-depth terminal leaves. The
artifact corpus identity is
`3bc9bc05a66a1a53255444266388838489020667272fc2ffa7445e7cf44be985`;
its file SHA-256 is
`490537d892e4dc64b0b46331754bab448a3b3d99dad620131cb692916e540ceb`.
The 192 histories end in only 84 distinct FENs. Therefore the inventory does
not silently define uniform path, final-position, or source-rank weighting,
nor does it freeze the proposed book/Perfect DB proportions.

The later source-only diversity audit projects those 84 FENs through the
book's own `ring16` symmetry and finds only seven endpoint orbits. In contrast,
two fresh processes generated the same first 64 fixed-seed StrictSteps
prefixes, comprising 64 unique exact FENs and 64 unique `ring16` orbits with
zero book-orbit overlap. Audit identity
`a7bc734ad3f85d2ae3ab75c901467da7b1835932fefa9aadd6067e1f4a982990`
records this pre-result evidence. The 75/25 proposal is therefore not
recommended as a diversity-first policy. The non-frozen technical
recommendation for 64 opening prefixes is one representative from each of the
seven book orbits plus 57 orbit-unique StrictSteps prefixes. Book-style
exposure would be a different named objective.

The configured Perfect DB returned 24 StrictSteps-tied initial candidates and
reported complete standard-sector coverage. The active HumanDB currently
fails closed with `database_not_immutable` because its SQLite `-shm` sidecar
is non-empty; no sidecar was changed. Whether HumanDB becomes evidence only or
a third source remains unfrozen.

The preceding paragraphs are immutable eight-ply v1 history. They do not
define the current opening-prefix design. On 25 July the product owner and
Mill expert selected a separate twelve-logical-ply v2 contract with Book,
HumanDB, and Perfect DB as independent strata. The eight-ply 7/57 proposal
remains historical evidence and must not be relabelled as a twelve-ply corpus.

The v2 source-only audits are now complete:

- the Oracle Book query graph supplies zero pure twelve-ply routes, while 84
  of 107 named variations supply 112 capture-resolved records and 110 unique
  histories/orbits;
- an online SQLite backup preserved the active HumanDB sidecars and bound a
  92,939-game PlayOK sample containing 83,002 exact histories, of which 5,174
  have at least two-game support; and
- two fresh Sanmill processes generated byte-identical evidence for 128
  StrictSteps routes, all unique by exact history, final FEN, and ring16 orbit,
  with zero overlap against Book or HumanDB.

The maintainer Openings delivery is also archived by identity. Its two Book
files duplicate tracked assets; its 15 learned additions remain an independent
candidate pool, not a formal stratum.

The later 35-row `Book Opening Plays.docx` delivery is separately archived and
audited as `maintainer_expert_curated_play`, a Book subtype rather than a
fourth reporting stratum. Its 36 explicit records all replay legally in two
fresh pinned Sanmill processes and reduce to 34 histories, 33 final FENs, 32
final ring16 orbits, and 14 eight-ply parent orbits. The expert-confirmed
row-19 correction has since been replayed through a separate reviewed-source
lineage, producing 35 histories, 34 final FENs, and 33 final ring16 orbits
without changing the 14 eight-ply parent orbits. One parent supplies 16
children. Six exact histories occur in 29 distinct frozen PlayOK games; no
exact history matches the Sanmill named lines, although nine unique final
FENs and orbits overlap. Row 11's final `c5` is screenshot-derived and remains
explicitly identified as visual evidence, but the original-resolution move
panel is unambiguous and it is not an expert-confirmation blocker.

The product owner accepted the near-balanced
`22 Book / 21 HumanDB / 21 Perfect DB` counts on 1 August; the
[composition decision](../experiments/sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.md)
freezes the ratio while leaving membership pending. The expert review
supplement provides substantial family and
child semantics, establishes that P03 needs extended-family splitting, and
identifies P13-A/P13-B as a symmetry pair. It does not provide a standalone
P14 classification, complete priority tiers, a final P03 partition, or every
primary-child choice. A 31 July direct-message follow-up now closes the P14
name as `Interrupted Knight`, labels the P03 children, distinguishes the outer
and inner Parallel Mill Rush variants, and selects P03 child 001 as the primary
Black response. The frozen audit independently proves that children 001/006
are different-history same-endpoint transpositions and that child 012 exactly
duplicates 006. On 1 August the expert clarified the selection rule: retain
every unique placement pattern, do not spend another place on a different
route to the same placement, and let the project arrange representatives. The
[semantic disposition](../evidence/maintainer-book-opening-plays-semantic-review-2026-07-26.md)
and
[short follow-up](../experiments/sanmill-layered-expert-book-review-follow-up.md)
separate that expert statement from project D4 normalization. The historical
[shortlist](../experiments/sanmill-layered-expert-book-shortlist-proposal-2026-07-31.md)
records the correction sheet he reviewed. The subsequent
[coverage decision](../experiments/sanmill-layered-expert-book-coverage-decision-2026-08-01.md)
freezes 33 D4-unique representatives as the complete Expert Book catalogue and
diagnostic membership while preserving all 36 raw records as provenance. Its
execution contract is not frozen. Status is
`executable_64_prefix_corpus_frozen_evaluation_not_authorized` for the
balanced 64-prefix core, and
the subsequent
[Book core decision](../experiments/sanmill-layered-opening-prefix-v2-book-core-2026-08-01.md)
freezes 15 expert-curated and seven Sanmill named-line members. The 22 records
cover all 14 expert parent orbits and all seven Sanmill declared families and
are unique by exact history, final FEN, and `ring16` orbit. The subsequent
[HumanDB core decision](../experiments/sanmill-layered-opening-prefix-v2-human-core-2026-08-01.md)
freezes 21 genuine PlayOK histories. Frequency-ordered selection reaches
ledger rank 31 after ten `ring16` skips; selected support ranges from 16 to 27
distinct games. The subsequent
[Perfect DB core decision](../experiments/sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.md)
freezes audit routes 000 through 020. All 64 source members are now unique by
exact history, final FEN, and `ring16`. The
[source core decision](../experiments/sanmill-layered-opening-prefix-v2-source-core-2026-08-01.md)
freezes their combined identity. The deterministic
[review package](../experiments/sanmill-layered-opening-prefix-v2-source-core-review-2026-08-01.md)
contains all 64 individual panels and six visually inspected contact sheets;
its manifest identity is
`db37224db6e400a32df9275e5e0665647541c4aa589b327b4317235e2eb27fba`.
The later
[HumanDB execution overlay](../experiments/sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.md)
freezes all 21 missing per-step records. The review images remain visual
membership material rather than execution authority.

The two untracked interchange-format drafts that temporarily dirtied the
referenced Sanmill checkout were removed by its workspace owner. The pinned
installation audit and 60 Sanmill UCI/data-query/prefix regressions then passed
again. No file was imported from the other workspace. The active HumanDB
sidecar remains a separate unresolved local-data condition.

The reference Sanmill checkout has since advanced through protected CLI and
rules paths and its release binary no longer matches the historical strict-v2
identity. The old binary bytes are unavailable. A clean isolated checkout at
the same pinned commit and tree was therefore built and frozen under the
separate
[prefix12 replay runtime decision](../experiments/sanmill-prefix12-human-replay-runtime-2026-08-01.md).
It is resolved through `sanmill_prefix12_checkout` and has release-binary
SHA-256
`6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53`.
This new identity does not replace the historical smoke evidence and is
authorised only for strict source-history replay.

That runtime replay is now complete. Two fresh processes produced exactly
equal ordered transcripts for all 21 HumanDB histories: 273 requests and 273
responses per process, covering 252 logical plies and 13 compound
Mill-and-removal turns. The transcript identity is
`e61bef7940fb1dd9a6fffb67b98640825d72a0ebcfb105627fdaa871173c13fd`;
the frozen HumanDB execution identity is
`1cf88ab8b3afb7c62112a0f2866eed9052587bbf2ef44dc57efa64c2749021d6`.
All 64 source-member execution records now exist. The subsequent
[executable-corpus decision](../experiments/sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.md)
assembles them under identity
`417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.
It preserves the 43-record historical and 21-record exact-HEAD Sanmill binary
identities separately. Evaluation and training remain unauthorised.
The five focused corpus checks pass. The wider layered-prefix run reports 81
passes after explicitly deselecting the known historical Perfect DB local
regeneration check; running that check against the advanced moving checkout
correctly fails closed on protected-path drift.

## Live Malom and Legacy-model Boundary

The old note says `specialist_router.py` was a temporary containment against a
broken Malom decoder. In current code, the specialist and generalist router
score paths still call the feature encoder with `db=None`, while separate Web
and `GameAI` paths can attach and query the now-corrected Malom implementation.
The blanket historical instruction to keep Malom out of all inference has
therefore been superseded, but the active path remains important evidence.

Any smoke or release check must record which route made the decision and test
that route with corrected atomic-capture, terminal-state, perspective, and
full-value semantics. The existing Sentinel, value-net, gap-net, specialist,
and generalist checkpoints all pre-date the correction. They may be used only
as explicitly labelled legacy inputs or ablations; loading one does not make it
a corrected model. Whether Sentinel training improves after corrected labels
is still untested.

The newly supplied browser evidence exposes a second route mismatch. The
trainer constructs its Generalist lookahead with the configured Malom database
as `endgame_db`, but `load_generalist()` does not pass an endgame database and
`GeneralistAgent.score_moves()` still calls the encoder with `db=None`. Although
the browser calls `set_db()` after loading Malom, that score path never consumes
the stored reference. Conversely, the browser constructs the Generalist with
globally loaded Sentinel, ValueNet, GapNet, HumanDB, and SpecialistDB objects;
unchecked UI boxes are not an auditable component-disable contract for those
features.

The candidate side can now be reconstructed as the separately named
`s-gen-v2-training-aligned-v1` route. The route bundle contains both final and
frozen-target weights and binds component flags, HumanDB, final SpecialistDB,
Malom, and fixed depth. Required resources are opened read-only and failures
propagate. The route deliberately retains the trainer's historical
empty-square comparison bug; silently correcting it would define another
experiment. A new paired protocol and competent baseline are not yet frozen,
so this preparation is not execution authority. Every future evaluation must
still emit checkpoint, route, component, data, Malom, and fixed-work
identities.

## Mixed-opponent Handover Copy

The uncommitted mixed-opponent edit from the previous maintainer was preserved
outside the repository as `train_s_gen_v2_handoff_unfinished.py`. Its exact
repository-relative location is recorded under the reference-only `notes`
entry in the
[`docs/local-training-layout.md`](../local-training-layout.md) path list.

The tracked `scripts/train_s_gen_v2.py` was restored afterwards. Do not replace
the tracked script with the archived copy. The current tracked schedule already
supports a configurable frozen self-play ratio and gives 15 per cent of
heuristic games a randomly lower difficulty. It does not implement the full
requested schedule of fixed higher/lower proportions, deliberate blunders, or
value/gap/Sentinel opponent blends.

The archived comments propose a 10/20/10/10/50 per-game schedule and describe
the blended branch as 10 per cent ValueNet, 30 per cent GapNet, and 20 per cent
Sentinel. The code does not establish those claimed inner blend weights:

- it supplies a ValueNet without changing the default zero
  `value_net_blend`;
- it attaches Sentinel in the default advisory mode rather than a 20 per cent
  move-selection override;
- it leaves GapNet on the existing phase-specific defaults rather than a
  uniform 30 per cent blend;
- its blunder branch uses a 25 per cent per-move probability inside selected
  games; that exact event distribution must be documented and tested rather
  than inferred from the prose request.

Those comments express intent, not completed behaviour. The draft also lets
most experimental opponent types affect level-advancement history, which would
confound grading unless each stratum is logged and advancement is defined
against a stable opponent.

That experimental schedule is not required to establish the first corrected
baseline. If revisited, audit each opponent type, sampling probability,
determinism, diagnostics, and failure fallback, then implement and test it as a
new change rather than recovering the interrupted edit wholesale.

## Monitoring and Resource Notes

`scripts/train_s_gen_v2.py` uses a `ThreadPoolExecutor` when `--batch-games` is
greater than one. Game simulation remains substantially CPU-bound, and the
original operator observed that excessive parallelism slowed iteration. Treat
that as a benchmark lead rather than a fixed worker recommendation: record the
worker count, games/hour, CPU and RAM use, GPU utilisation, search settings,
and output/database contention before selecting long-run concurrency. Keep the
first integration smoke at `--batch-games 1`.

The existing monitor can be started from the repository root with:

```powershell
.\.venv\Scripts\python.exe tools\plot_specialist_training.py
```

It refreshes every 20 minutes by default and visualises existing logs; it is a
health monitor, not strength or correctness evidence. Before a long run,
record the log path, refresh interval, checkpoint cadence, stop criteria, and
who or what will inspect stalled games, non-finite losses, recovery loops, and
database growth.

The richer managed-run dashboard is now version controlled at
`tools/serve_managed_training_monitor.py`. Point it at one exact control
directory; for completed no-refresh attempt 003 the loopback command is:

```powershell
.\.venv\Scripts\python.exe tools\serve_managed_training_monitor.py `
  --control-dir learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-no-refresh-retained-v4-seed70-attempt-003 `
  --host 127.0.0.1 --port 8765
```

It reads the real rolling window from every segment manifest, separates the
frozen and Sanmill source windows, and reports rules draws apart from max-ply
truncations. Its bilingual help labels mixed-source win rate as a training
diagnostic rather than strength evidence. The server is read-only with respect
to trainer evidence. Whole-device GPU telemetry is appended beneath the
selected run's ignored `local-monitor` directory only while a managed segment
is actively running; samples outside controller-led training windows are never
presented as training utilisation, and the data remains whole-device rather
than process-exclusive.

Dashboard categorical order and control-value plots must not inherit incidental
log ordering or smoothing. Aggregate outcomes are rendered in fixed
win/draw/loss order. Learning rate is rendered from the raw executed value as a
step function with an explicit change-point note, not as a 50-game moving
average. For no-refresh attempt 003, the evidence is `1e-4` through game 50 and
`5e-5` from game 51 onward; the former diagonal `0.93 -> 0.50` display was a
visualisation artefact caused by smoothing that one discrete change. Canvas
y-axis tick labels share one fixed right edge, so a negative sign extends left
without shifting the numeric value relative to positive ticks. Observation
scope and frozen-plan boundary semantics are stated once in the page-level
evidence key; individual charts do not repeat a badge that obscures the plot or
incorrectly describes plan markers as observations.

No-refresh attempt 003 has no valid training-window GPU telemetry: it completed
at `2026-08-13T06:49:14Z`, while the first local monitor sample was collected at
`2026-08-13T07:31:34Z`. The dashboard therefore reports the training GPU metric
as unavailable and excludes those later live whole-device samples. Do not
reinterpret missing historical telemetry as zero utilisation or backfill it
from post-run observation.

## Deferred and Conditional Work from the Original Notes

- Direct "learn traps" training is not implemented. The v5 plan defines fixed
  trap scenarios for stress testing and diagnosis, which is not evidence that
  a trap curriculum is necessary or effective.
- The v5 teacher/HumanPolicy signal, human-evaluation power, rule/oracle
  semantics, and implementation complexity require the independent reviews
  specified by that plan before their optional branches are opened. They are
  not prerequisites for the minimal corrected v4-style baseline.
- Puzzle repair, Windows/Linux installers, hosting, a book link, and additional
  languages are product backlog ideas. They are outside this training handover
  and carry no implementation commitment.
- Starting a separate Sanmill-trained AI is not an accepted next action. The
  pinned Sanmill checkout is a reference and possible differential-test input
  under the boundary recorded in the local-layout document.

## MIF 1.0 Independent Interoperability Adapter

The NMM_LLM adapter is locked to MIF commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978` and implements all seven
`MIF-INTEROP/1` operations: capabilities, MFEN/MPK canonicalization, finite
rules execution, checkpoint-verifying replay, full-state transform, complete
logical-turn projection, and the `legal-actions-v1` harness projection. It is
intentionally independent of the MIF Python reference runner; the latter is
used only as a separate black-box comparison process.

The adapter's honest runtime claim is narrower than every possible
`mif-finite-rules-v3` variant. It advertises the two frozen candidate corpus
rulesets, accepts patches inside their implemented semantic subset, and fails
closed outside that subset rather than approximating unsupported capture
mechanisms, semantic-state extensions, mill effects, or stalemate policies
with the legacy NMM_LLM board engine. The following candidate records are
historical inputs to the later immutable Suite release; none is a `full` or
conversion claim.

Candidate-3 gameplay is implemented by NMM_LLM commits `748dae2`, `feb4646`,
and `121b663`. Candidate-4 changes only the locked source and corpus identities
in NMM_LLM commit `bbbde2ee4bf1ba0e45e259baa595a29cb85895b9`; it does not alter
the independent state machine. The existing implementation matched all 58
candidate-4 cases before the pin update, including the three new
asymmetric-reserve origin cases. At the clean pin commit, 55 focused tests and
Ruff pass. The generator accepts the clean MIF checkout and all seven frozen
hashes, while MIFCAP publishes the 17-case smoke and 58-case deterministic
corpus identities without claiming a Suite. The exact pin, hashes, scope, and
host-local command-array generator are in
[`docs/interop/mif-1.0-independent-adapter.md`](../interop/mif-1.0-independent-adapter.md).

The raw
[candidate-4 report](../evidence/mif-interop-candidate-4-nmm-reference-report-2026-08-06.json)
has SHA-256
`89dfcd97c914764aa95bcb5e6b6ecdb23686591037dbf8c5493fe8b3dfbc142f`
and records 58/58 equality between the published MIF reference and NMM_LLM at
clean commit `bbbde2ee4bf1ba0e45e259baa595a29cb85895b9`.

Sanmill subsequently published candidate-4 commit
`e6d639d41f079b15ca697268d0c2c21dad5c2bc3` and the tracked three-party
report `interop/evidence/mif-interop-candidate-4-three-project-report-2026-08-06.json`.
That report has SHA-256
`895c04cd69fc00e50bdcd349b150293e52fcc4150c63321d8c9771015f70aaaf`
and records 58/58 with cases digest
`sha256:d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82`
and config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
An independent rerun using MIF `7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`,
Sanmill `e6d639d41f079b15ca697268d0c2c21dad5c2bc3`, and NMM_LLM
`11bebd14e0d538a41a4b43aebfe57ee74c2a2601` reproduced the same report
hash.

Sanmill closed the M3 evidence-chain gap at evidence commit
`9431b95f151502f415f096c7d96ca944e5d578de`. Its companion manifest binds the
three published commits, all seven candidate-4 inputs, and the 58/58 report
identity above. M3 is closed for those exact candidate identities.

Candidate-4 M4 uses MIF launch commit
`40718e80d36ec9c060fc17997568d637a74e6d9f` over the unchanged wire commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`. The fixed launch SHA-256 is
`560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e`;
the reference-baseline SHA-256 is
`29d198dbcf8221fa0235af6a72db9d6a82646b45fc653c584071821a9a4bb61b`.
The prescribed pre-fix run at NMM_LLM
`e2ab05d29885af9a16a9aa5d5f62b1517cf6d91b` reproduced 10/10 seeded runs
and 3/5 mutations at config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.

Tested NMM_LLM implementation commit
`6c1538082fc551203d827782d137a5799c810535` aligns only the two failing
diagnostic shapes; it does not change gameplay or replay semantics. The
[two-party raw report](../evidence/mif-interop-candidate-4-m4-reference-nmm-report-2026-08-07.json)
has SHA-256
`2bc434699902a1c468b604797d4456ee0c968817b057ec4dc8254a623a1ba64c`,
records 10/10 seeded runs and 5/5 mutations, and has config digest
`sha256:c6eb5edc21773c017e7a2d5d9050b38cb08450658a286e64a395f1edc6b7074e`.
The adjacent
[companion manifest](../evidence/mif-interop-candidate-4-m4-reference-nmm-evidence-manifest-2026-08-07.json)
binds the exact MIF and NMM_LLM commits plus Sanmill implementation
`ae9a1d8a16261478631a3a7583cbf35c7b6e0df5`, evidence
`9431b95f151502f415f096c7d96ca944e5d578de`, and its two-party report hash.
The final three-party preflight records 10/10 and 5/5 with config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
This closes NMM_LLM's Candidate-4 M4 differential evidence only. No result
claims MIF Suite 1.0 conformance. Earlier candidate reports remain historical
evidence for their recorded identities.

An additional 1,138-test repository run was attempted without skips, but the
15-minute command limit stopped it at roughly 15% with no failure reported.
It is not a full-suite pass and must not be presented as one.

The later Suite-finalization gate supersedes the pre-Suite capability status
without changing the wire implementation. MIF commit
`3ee7e57c7d4c7208be91f62914f344a587fb0f70` fixes Suite raw SHA-256
`088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4` and
JCS SHA-256
`81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`.
NMM_LLM implementation commit
`a7e7dbd5461cc2d8d8c0a09317d6091598202214` publishes the exact Suite pin
and marks only the six required classes and two Suite rulesets as tested. It
continues to declare no `full` class and `conversion=none`.

The final Reference/NMM_LLM evidence records 58/58 deterministic cases, 10/10
seeded differential runs, 5/5 mutation families, and zero unexplained
differences. The capability, deterministic-report, and differential-report raw
SHA-256 values are respectively
`cd661b1156bf7269f976e050446d01797c9959482f1e1843e21ae3ea7f70dcce`,
`3463f438531fd52847df44fa4186dcba13ed22c7c570a0cc216d9a7eaa797665`,
and `4c86725bfcd1759433374938c8d8eb2a1dacfa6ea3723592eff759162fce8da6`.
The
[Suite evidence manifest](../evidence/mif-suite-1.0-nmm-adapter-evidence-2026-08-07.json)
and [interop record](../interop/mif-1.0-independent-adapter.md) own the complete
identity and scope details. A current three-adapter preflight also passed
58/58, 10/10, and 5/5.

The 66 focused MIF tests and static checks pass. A four-shard run exercised all
1,179 repository tests: 1,170 passed initially, the sole Windows Chroma cleanup
failure passed alone, and eight machine-local Sanmill tests remained
fail-closed because the historical strict-v2 binary bytes are unavailable.
This is not a clean full-repository pass and must not be represented as one.
The Suite claim is only `exact-for-tested-domain`; it is not `full`
conformance and makes no conversion claim.

A later clean single-process run at published `dev` commit `f06d457` collected
and executed all 1,235 current tests without skips. It reported `1,227 passed`,
`8 failed`, and `498 subtests passed` in 2,876.61 seconds. A focused rerun
proved that all eight failures stop at the same protected-source-path check
before any Sanmill query, search, replay, or gameplay assertion. No NMM_LLM
trainer, MIF, Malom, checkpoint, exact-resume, or managed-run test failed. The
full evidence and claim boundary are in the
[7 August complete-test baseline](../evidence/current-complete-test-baseline-2026-08-07.md).

MIF Suite 1.0 is now immutably published as tag `mif-suite-1.0` at release
commit `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`. Suite JCS SHA-256 remains
`81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`;
final evidence SHA-256 is
`2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1`;
release-manifest SHA-256 is
`dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499`.
Training manifests now bind those identities, ruleset
`nmm-training-core@2`, semantic digest
`52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`,
and an independent experiment digest. The MIF publication gate is closed;
the completed corrected retained run binds these identities in its final
checkpoint.

## 10 August Normalized Auxiliary Preparation

The three-seed no-update batch capture is complete. Its tracked interpretation
was committed and ordinarily pushed as
`bcceab547cc7de9177a24964bd021816d656bd7c`. The ignored raw result has
identity `b0dfd3415c55196c59e71cf67e45b00ab5844e9f62fbc9f3bdc31b09a694bd86`
and SHA-256
`2e310ecccf869f16b314093c6f50395e91019839b603d805ad3e0cab9d651fee`.
Across 60 games and 19 captured update batches it observed 1,473 labelled
steps and 453 informative steps. A target policy-head ratio of 0.25 implied
effective coefficients from about 0.0481 to 0.2154, with median about 0.1044.
This is a no-update scale observation, not an optimizer, strength or
coefficient-selection result.

Local commit `702c669a624f3ead7099126c6707e6513ed821c3` implements the
policy-head-normalized mode. It computes detached ordinary policy-head and raw
auxiliary gradient norms per informative batch, targets ratio 0.25, caps the
coefficient at 0.25, fails closed on an invalid auxiliary denominator, applies
zero auxiliary weight when the ordinary gradient is below the floor, and logs
the applied ratio and gradient cosine. Fixed coefficient zero remains the
control and compatibility default.

Commits `702c669a624f3ead7099126c6707e6513ed821c3`,
`bfd59106100d28a1cb046728bfc87b5be6708120` and
`30246f6c610185a7bc48841ce31231f74b48979b` were ordinarily pushed. The
first published preparation at `30246f6` produced six plans, six
`needs_decision` preflights and six byte-identical empty database copies. Its
preliminary readiness identity was
`f684e9624ed878290aef6702c3aab5223a46ccd843f6be295ea49292f04671f6`;
the report SHA-256 was
`c632ed633dc550c5db295cdff865fa39399f3da232372aa9e50f9eaca0891502`.
It created no authorization, segment, checkpoint or result.

Review then found that the only existing policy-auxiliary result publisher was
for four fixed coefficients. It could not interpret normalized-mode update
diagnostics or apply the frozen three-seed paired decision. Those preliminary
plans are therefore superseded and must not be authorized. The publisher
correctly rejects their readiness as binding another contract.

Local commit `a6de71a9ca052f5eccbb4f836067976eee483a89` adds the dedicated
fail-closed result analyzer and immutable publisher before any training. The
[revised three-seed paired calibration](../experiments/sanmill-malom-policy-auxiliary-normalized-calibration-v1.md)
has plan identity
`1b6f8d05047c4de9d6603d9ae1f26714cb1a23b3b96749e76136387a5f0b53ab`.
It binds the analyzer SHA-256
`afbefb7f9bedb0fafda8edf1e313f88f591ed47c995578b76842793f38290aaf`
and publisher SHA-256
`276ca8c34567d507eed225135cc1ec3db4986972c576c174c7935fdecd33f6fe`.
The analyzer validates effective coefficients, ordinary and auxiliary gradient
norms, applied ratios, cap status, cosine, exact labels, phase support, raw and
complete-window curves, all artifact identities and the paired decision. The
control records selected-action Malom diagnostics but does not enumerate the
all-action auxiliary labels.

Local commit `2a263f0d3afd02a9bb5e5fd5b1137424a5b16d2a` makes the
source-only audit report superseded preparation targets instead of implying
that publication alone is sufficient. At that commit, source readiness
identity `960061fd5e134f9240cf281b089de7016ab33e65cbf45ade48db2f9699c95828`
reports 13 existing preliminary targets: six plan directories, six empty
database files and the old readiness report. It authorizes nothing and lists
both publication and quarantine as unresolved gates.

Those 13 targets were subsequently moved without deletion into the ignored
recoverable quarantine
`out/quarantine/malom-policy-aux-normalized-preparation-superseded-30246f6-20260809T192421Z`.
The quarantine records every plan, preflight, controller-event and database
hash, the superseded contract identity, and the original readiness SHA-256.
No authorization, segment, checkpoint, log or counted game existed. A new
source-only audit at local commit `985c523e4c647b788ab0744f4713f45a19d79022`
reports all preparation targets absent under readiness identity
`7dd72688613efd6d2b248fea6a125eadf3a6689db1c02f52ed228e4592fa5161`.
Its only unresolved gate is ordinary publication of the implementation.

Seeds 55, 56 and 57 each compare auxiliary-off control with normalized target
0.25. Each arm is bounded to 100 games and one third active hour, so the whole
sequence is capped at 600 games and two active hours. Only the first,
previously observed 1,000-node Sanmill level is reached. The fixed rule uses
paired scratch-normalized policy changes, at least two positive seed pairs,
median preserving-mass gain of at least 0.001, and explicit entropy,
repetition, identity, label, checkpoint and resource gates. Passing can only
justify designing a later effectiveness experiment.

The final local gate passes 174 normalized-auxiliary, result, trainer,
managed-run, contract and readiness tests. The mandatory Malom, DB-teacher and
label-provenance gate separately passes 103 tests plus 498 parameterized
subtests. Ruff passes every changed Python module and `git diff --check`
passes. This is a focused preparation result, not a new clean full-repository
test claim.

Observed facts support testing normalization as a scale-control mechanism.
They do not establish that its gradient direction cooperates with A2C or that
100 games can show a repeatable effect. Gradient cosine, raw and complete
training curves, all three seeds, the frozen hyperparameters and data versions,
scratch/control baselines, paired ablation changes and phase/opponent/colour/
termination metrics must be considered together. The 29-state diagnostic is
development evidence, not held-out validation.

The authorized six-arm run subsequently completed exactly once under readiness
identity `a5fb75eda17b4609902294f424300cb45f964440852ecfe4a008f1ea70733637`.
All 600 games, 176 optimizer updates, six checkpoints, six policy-health gates,
and six isolated corrected databases passed their infrastructure and numerical
checks. The immutable local result has identity
`669124f2803609fe87fabc15c38a798711e78541ed1a39614cf44837a51a58ac`,
file SHA-256
`0d59bc587d66006255020e5ab3b7faab2f8b9c693a1c139686a475d0e93828bb`,
and verdict `inconclusive_stop_and_redesign`.

The normalized ratio was exactly 0.25 on all 88 treatment updates and every
seed moved the fixed development diagnostic in the intended direction. The
median paired preserving-mass gain was only `0.0000298`, below the frozen
`0.001` gate, while seed 55 exceeded the raw repetition-rate safety limit by
one percentage point. No target is selected and no retained or long run is
authorized. Preserve the detailed
[result evidence](../evidence/sanmill-malom-policy-auxiliary-normalized-calibration-result-2026-08-10.md).

The next safe step is a no-game, disposable target-response audit on the three
persisted treatment final-flush batches. It should compare normalized target
ratios 0.25, 0.50, and 1.00 from the same pre-update model and Adam state,
reproduce the real 0.25 update, report post-Adam response and phase support,
and mutate no persisted artefact. It is diagnosis only and cannot select a
training setting or launch another run.

That audit subsequently completed once. Its raw identity is
`819d84d2ed7bb943260aa0627c22db0c0b94944ea2c058ee3fba3116a49f2fa4`,
its decision identity is
`6f6359df371be56e5b5f25c2a31287363e4d31fe58f87422f4cd46767e6249fc`,
and its verdict is `stop_gradient_ratio_escalation`. Target 0.25 replayed the
persisted production update for every seed, all bounded-response checks
passed, and the median preserving-mass response increased with target. Seed
56 nevertheless moved monotonically in the opposite direction, and all
responses remained near float32 resolution. No normalized target is selected.
Keep the policy auxiliary off in the next retained baseline; a future
KL-constrained or safe-action mechanism is a separate experiment, not a
long-run prerequisite. Preserve the detailed
[audit evidence](../evidence/sanmill-malom-policy-auxiliary-normalized-target-response-2026-08-10.md).

The successor decision is now frozen in
[`sanmill-preserving-retained-long-v3.md`](../experiments/sanmill-preserving-retained-long-v3.md).
It uses fresh, previously unobserved seed 58, `malom-preserving-only`, and no
generic downgrade penalty or policy auxiliary. All other retained-v2
curriculum, referee, optimizer, component-disable, segmentation, resource and
policy-health choices remain fixed. This is a new research baseline, not a
claim that the inconclusive short ablation proved strength improvement.
Preparation still requires a clean published tip, a fresh isolated database,
an immutable managed plan, complete focused verification, and a final
readiness identity. Training remains unlaunched and needs explicit product
authorization against that identity.

The first ignored preparation at published commit `d708e10` stopped before
authorization or training because final preflight found that the experiment
document had retained the seed-42 Sanmill operational identity. The
installation record includes `SearchShuffleSeed`: seed 42 produces
`705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea`,
while the frozen seed 58 produces
`5d436ac3eff3d7a7f186a4a7fb1c656739bafc93baeb5bb4e5b1dbf905dbaf04`.
All seed-independent Sanmill source, tree, binary and referee identities were
unchanged. The preliminary plan and fresh empty database were moved without
deletion to
`out/quarantine/sanmill-preserving-retained-v3-preparation-superseded-d708e10-20260809T232402Z`.
That plan is `fatal_stop` and must never be authorized. Publish the identity
correction, then recreate both exact targets and repeat final preflight; do not
reuse the quarantined preparation.

## Recommended Next Actions

The corrected retained 5,000-game run is complete. Preserve its plan,
authorization, controller ledger, split accepted logs, quarantined recovery
inputs, final checkpoint, policy-health reports and SpecialistDB under the
identities in the completion evidence. Do not exact-resume it, run another
training job, promote it, or publish a strength claim merely because the
controller completed.

The post-completion focused verification passed, but the repository still has
the separately documented moving-checkout Sanmill failures; no new clean
full-suite claim was made. The held-out protocol has also completed, and its
one-run authorization is consumed. Preserve the historical work and advance
the current successor in this order:

1. Preserve the corrected retained plan, authorization, ledgers, every
   accepted segment checkpoint, both recovery bundles, the final candidate
   checkpoint and the final SpecialistDB under their recorded identities.
2. Keep both rebuilt database candidates archived and inactive, and keep every
   imported checkpoint out of the `dev` resume lineage. Ask for additional
   checkpoint lineage only if a future experiment proposes to adopt one; use
   the locally resolved Sentinel, ValueNet/HumanPolicy, and GapNet boundaries
   recorded in the retraining plan.
3. Preserve the completed owner review: original review position 101 is
   excluded, the other 106 are accepted, and the withdrawn concern about 83 is
   not a corpus defect.
4. Preserve the completed Stage-0 spec and final ledger together under their
   recorded hashes; the one-run authorization is consumed.
5. Preserve the `accepted` result as ablation-only training-signal evidence.
   Do not rerun it or treat acceptance as promotion evidence.
6. Preserve both bridge generations, treating the v2 logical-turn result as
   current and v1 as historical. Keep `GameAI` deferred as formal referee. The
   60-turn ceiling was smoke-only, and no candidate-versus-baseline authority
   was consumed.
7. Keep the Sanmill book, data-query, strict-error, logical-turn, and state
   commits pinned by identity even though they are now on Sanmill `master`.
   Do not silently float to later CLI or rule changes.
8. Preserve the eight-ply implementation and 7/57 evidence as v1 history.
   For current work, use the twelve-ply v2 contract and completed Sanmill Book,
   expert Book, HumanDB, and Perfect DB audits. The HumanDB immutable snapshot
   was created without deleting active sidecars. Preserve the accepted
   `22 Book / 21 HumanDB / 21 Perfect DB` balanced-core split. Its 64 source
   members, review manifest, source-member execution records, and combined
   executable corpus are now frozen. Preserve executable-corpus identity
   `417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.
   The expert gate is closed: preserve the frozen
   [33-pattern coverage catalogue](../experiments/sanmill-layered-expert-book-coverage-decision-2026-08-01.md)
   and keep the separately frozen balanced-core subset without changing that
   catalogue. Row 11's screenshot-derived `c5` is
   source-verified and
   retains a visual provenance marker. There is no runtime source fallback.
9. Record the Mill expert's completed first-pass review of all 64 panels. He
   supplied a plausible move for each, marked several unlikely or poor states,
   described the overall spread as useful, and suggested adding positions
   where closing a Mill competes with blocking or enabling a chain Mill. This
   refers to the separate phase-coverage draft, not the already frozen
   22/21/21 twelve-ply opening corpus. It is domain feedback, not an automatic
   phase-corpus freeze. Any later tactical stratum must remain separately
   identified.
10. Preserve the frozen retained-v2 held-out plan, exposure audit, consumed
    authorization, runner, host-interruption archive, 128-record ledger and
    final result evidence. Do not rerun it, pool it with Stage 0 or training
    diagnostics, or promote the candidate. Preserve the later read-only
    transition diagnosis as development evidence; do not re-query or relabel
    the held-out corpus as a new strength gate.
11. Preserve the completed six-arm downgrade-penalty experiment and its
    `inconclusive` verdict. Its authorization is consumed, and reward-only
    escalation is closed.
12. Preserve the completed four-arm Malom policy-auxiliary calibration and its
    `inconclusive_recalibration_required` verdict. Its authorization is
    consumed. No raw coefficient was selected, and coefficient `0.30` must not
    be extended after crossing the frozen loss-scale limit.
13. Preserve the subsequent no-update gradient-interaction evidence. It found
    that the applied auxiliary-to-policy gradient ratio varied from about
    `0.69` to `26.7` across the two recoverable production batches. This is a
    scale-diagnosis result, not authority to choose a normalization target.
14. Preserve the completed three-seed, 60-game no-update batch-capture raw
    result and tracked interpretation under their recorded identities. Its
    one-run authorization is consumed. Do not rerun it or treat its implied
    coefficient distribution as a selected training setting.
15. Preserve the completed normalized calibration, its six consumed
    authorizations, all plans and ledgers, result identity `669124f2`, and the
    recoverable superseded-preparation quarantine. Its verdict is
    `inconclusive_stop_and_redesign`; target 0.25 is not selected. Before any
    further learning run, perform only the separately frozen no-game target
    response audit described above. No retry, extension, resume, promotion,
    model publication, or long training is currently authorized.
16. Preserve the completed normalized target-response audit, raw identity
    `819d84d2`, decision identity `6f6359df`, and verdict
    `stop_gradient_ratio_escalation`. Its one permitted execution is consumed.
    Do not raise the target, rerun a batch, lower the monotonicity threshold,
    or adopt any normalized target in retained training. Keep the auxiliary
    off in the next baseline. Any KL-constrained teacher or safe-action
    sampler requires a separate contract and is not a prerequisite for that
    baseline.
17. Preserve the completed seed-58 Sanmill-preserving retained-v3 baseline,
    its 5,000-game lineage, checkpoint, database, completion evidence and
    consumed authorization. Do not resume or promote it. Preserve both
    SpecialistDB mechanism audits, distinguishing the first zero-coverage
    result from the second coverage-positive material result. Preserve the
    completed three-seed read-mode calibration, result identity `90da6053`,
    all six consumed arm grants and its null selection. Its game-50 target
    refresh and learning-rate change were coupled, so do not use that result
    to assign causality to either mechanism.
18. Preserve published target-refresh/LR preparation commits `0d9f7b8`,
    `00cddd2` and `d450d50`, the clean-source audit, the eight isolated plans,
    and readiness identity `893c38fa`. Their later product authorization was
    limited to the frozen 800-game, two-hour envelope and prohibited retries,
    extensions, held-out evaluation, promotion, publication, and long
    training.
19. Preserve the completed two-seed 2x2 target-refresh/LR diagnostic and its
    [tracked result evidence](../evidence/target-refresh-lr-factorial-diagnostic-result-2026-08-10.md)
    at commit `d2879b2`. All eight arms completed 100 games and all
    policy-health gates passed. The raw result identity is
    `fb6ca7f5`, with file SHA-256 `6ea82762`. The preregistered signed contrast
    is `no refresh - refresh`: it was `+0.7857` for seed 64 and `+0.7656` for
    seed 65 against the endogenous frozen opponent. Fixed-minus-adaptive LR
    and interaction contrasts were zero, while the learner won no
    post-boundary game against the 1,000-node Sanmill stratum. This detects a
    refresh-boundary mechanism signal but does not show that refresh harms
    transferable learning and does not select no-refresh or either LR mode.
    The eight grants are consumed. Before any held-out or long run, prepare a
    separately frozen successor design that keeps a common measurement anchor
    distinct from the training target and controls optimizer-step exposure.
20. Preserve the authorization-free successor design in
    [the common-anchor contract](../experiments/sanmill-target-refresh-common-anchor-diagnostic-v1.md),
    plan identity `8e398233`. It contains four fresh arms: seeds 64 and 65,
    each with target refresh at game 50 or no refresh. Same-seed arms must be
    byte-identical through game 50 and freeze the same development-only model
    anchor there. Each arm then performs exactly 16 A2C optimizer steps and
    records balanced no-update measurements at update deltas 4, 8, 12 and 16
    against that common anchor and a separately reported 1,000-node Sanmill
    opponent. The measurement route cannot read the growing SpecialistDB or
    write training evidence. Commits `6b98a5d` and `d65efea` implement the
    runtime, managed-plan and fail-closed analysis support. The contract is
    unlaunched and unauthorized; exact source publication, fresh databases,
    managed plans and preflights must still be completed before any product
    authorization could be considered.
21. The product owner subsequently authorised that exact four-arm sequence at
    readiness identity `d6ed98be`. Attempt 001 stopped fail closed in the
    first `seed64-refresh` arm at game 50, after 18 optimiser updates, because
    the trainer requested checkpoint role `development_measurement_anchor`
    while the version-2 envelope had not registered either new development
    measurement role. No anchor, candidate checkpoint, accepted segment,
    policy-health result or experiment result exists, and the remaining three
    arms did not start. Treat all four authorisation files as consumed by the
    aborted sequence. The first database contains failed-attempt writes; the
    other three remain empty but are still attempt-001 evidence and must not be
    reused. Local correction commit `e02aca4` extends only the checkpoint role
    vocabulary and adds round-trip tests for both evidence roles. Preserve the
    [failure record](../evidence/target-refresh-common-anchor-diagnostic-attempt-001-failure-2026-08-10.md).
    A retry requires ordinary publication, new plan/control/database identities,
    fresh preflights and a new explicit product authorisation; no automatic
    retry, held-out evaluation or long training is authorised.
22. Preserve attempt 002 under readiness identity `bcbb625d`. Both seed-64
    arms completed and passed policy health at their exact 34-update bound,
    using 122 and 92 training games and 64 no-update measurement games each.
    Their first 50 canonical game rows and anchor model tensors are identical,
    with anchor state SHA-256 `94aed99f`. The frozen analyser then failed
    because the reused policy-health helper required the 150-game safety
    ceiling rather than each arm's validated optimizer-bounded completion
    count. Seed 65 did not start; no result file or causal decision exists.
    All four grants and databases are consumed attempt-002 evidence. Commit
    `873e126` corrects only analysis validation and passes the focused runtime
    and test gates. Preserve the
    [attempt-002 failure record](../evidence/target-refresh-common-anchor-diagnostic-attempt-002-failure-2026-08-10.md).
    Any successor requires publication, wholly new identities, fresh database
    copies and a new explicit product authorisation. Do not resume, retry,
    run held-out evaluation, promote, publish or start long training from this
    incomplete sequence.
23. Preserve the authorization-free
    [attempt-003 contract](../experiments/sanmill-target-refresh-common-anchor-diagnostic-v1-attempt-003.md),
    plan identity `8cc192f5`. Its hypothesis, seeds, four-arm order, target
    refresh treatment, fixed game-50 anchor, optimizer exposure, no-update
    measurement schedule, policy-health gate and resource limits are unchanged.
    Only the lineage-owned paths, attempt metadata, failure evidence and fixed
    analyser identity differ. No attempt-003 database, plan, preflight,
    authorization or segment exists yet. Publish the exact contract source
    normally, re-establish `dev == origin/dev`, then generate only four pristine
    database copies, immutable plans and preflights. Do not create an
    authorization or start training without a later explicit product grant.
24. Preserve the completed
    [equal-transition diagnostic](../experiments/sanmill-target-refresh-equal-transition-diagnostic-v1.md),
    its three shared prefixes, six arm lineages, consumed authorizations,
    ignored raw result, and
    [tracked result evidence](../evidence/target-refresh-equal-transition-diagnostic-result-2026-08-11.md).
    Every arm consumed exactly 8,192 post-fork transitions. Result identity
    `8c6be27f` has classification `inconclusive_late_onset`: seeds 64 and 65
    first crossed a material threshold only at the final boundary, with no
    persistent trigger and opposite Malom-mass directions. Do not select a
    target policy, retry, extend, run held-out evaluation, promote, publish or
    start long training. The next design must isolate the game-count-indexed
    temperature and Sanmill resource schedules before another target-refresh
    setting can be considered.
25. Preserve the completed
    [schedule-isolation v2 contract](../experiments/sanmill-target-refresh-schedule-isolation-diagnostic-v2.md),
    plan identity `0580389b`, and its reviewed `origin/main` lineage through
    `028ef8e`. All three shared prefixes and six arms completed once: 2,832
    distinct training games, 49,152 post-fork transitions and 0.4824 active
    hours. The first publisher failed closed on uniformly CRLF-framed Windows
    JSONL before any of the 288 development games. The separately authorised
    [analysis recovery](../experiments/sanmill-target-refresh-schedule-isolation-analysis-recovery-v1.md)
    then completed all 288 CPU no-update games once under readiness identity
    `034ed820`, with zero optimizer, database or checkpoint writes. Result
    identity `a4381489` classifies the paired outcome effect as
    `no_material_paired_outcome_effect` and policy separation as
    `inconclusive_late_onset`; no condition is selected. Preserve the
    [result evidence](../evidence/target-refresh-schedule-isolation-diagnostic-v2-result-2026-08-11.md).
    All sequence and recovery grants are consumed. Do not retry, extend, run
    held-out evaluation, promote, publish, or start long training from this
    result. A direct paired no-update successor needs a new immutable contract,
    readiness record, and one bounded product authorisation.
26. Preserve target-refresh direct cross-play attempt 001 under source
    `b25fe33`, readiness `6da0b4ac` and the
    [failure record](../evidence/target-refresh-direct-crossplay-attempt-001-failure-2026-08-12.md).
    It failed before ordinal zero because the runner used an abbreviated seed
    field that does not exist in the closed schedule. Its ledger has zero rows,
    all data and checkpoint observations remained unchanged, and its one-run
    authorization is consumed. Do not overwrite or resume its output. A fixed
    attempt 002 requires new output, plan and readiness identities plus a new
    explicit product authorization.
27. Preserve completed target-refresh direct cross-play attempt 003 under
    plan identity `2f1665e5`, readiness identity `9fd354a7`, authorization
    identity `3175570e`, launch identity `7f696aef`, and completion identity
    `fedd31a4`. The independently republished evidence identity is
    `9a5df62d`. Its three-seed and three-phase result supports a material
    no-refresh direct effect at the original game-50 boundary, but does not
    select permanent no-refresh. The next design must test one refresh from a
    mature common fork with equal post-fork optimizer exposure and fixed
    schedules. Do not reinterpret this development corpus as held-out
    strength or use it to authorize promotion, publication or long training.
28. Preserve mature-fork preparation attempt 001 under invalid readiness
    identity `32df3a5b` in its quarantine directory. It had no authorization
    and executed no training. Do not repair, authorize, overwrite or relabel
    it. Use only the isolated
    [attempt-002 contract](../experiments/sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.md),
    plan identity `442c1701`, frozen at commit `40b85e6`. Its corrected plans
    must carry the exact policy-health gate and pass final preflight before a
    single aggregate launch decision is requested. This invalid preparation
    remains unlaunched and has no launch authority.
29. Preserve mature-fork attempt 002 under readiness identity `d2860ae0`,
    authorization identity `181a8e88`, launch identity `34dd77f5`, and failure
    identity `d4e13fba`. All six arms completed once and passed policy health,
    for 2,529 new training games, 49,152 transitions and 768 updates. The
    result publisher rejected the exact-hash, pretty-printed policy corpus
    before loading candidates for analysis or starting any of the 288 CPU
    games. The launch and child grants are consumed. Preserve the
    [failure record](../evidence/target-refresh-mature-fork-diagnostic-attempt-002-failure-2026-08-12.md).
    Do not retry or resume the training sequence. The publisher's
    reference-input and Windows JSONL policies were corrected with focused
    regression tests. Recovery v1, plan identity `70fb522b`, was authorized and
    launched once but failed before its first development game because its
    publisher rejected valid post-fork implementation metadata. It wrote no
    optimizer, database or checkpoint state and produced no ledger, result or
    completion record. Its authorization is consumed; preserve the
    [recovery-v1 failure record](../evidence/target-refresh-mature-fork-analysis-recovery-v1-failure-2026-08-13.md).
30. Preserve the historical prelaunch state of the
    [recovery-v2 contract](../experiments/sanmill-target-refresh-mature-fork-analysis-recovery-v2.md),
    plan identity `32158846`, published with its implementation at commit
    `360c878`. The post-fix read-only audit and final published-source preflight
    both validated all 12 candidate checkpoints under candidate-audit identity
    `d3c7e0dd`. The pre-handover readiness identity `fcd38c2f` and file SHA-256
    `6edca20a` are historical evidence only because the earlier handover commit
    changed the analysis HEAD. At that checkpoint no v2 authorization or launch
    existed. The historical readiness was subsequently preserved outside the
    active namespace and the preflight was rerun from an absent namespace; item
    31 records the resulting one-shot execution.
31. Recovery v2 subsequently completed exactly once under readiness identity
    `13e25cd5`, authorization identity `c02f2ffd`, launch identity `988c6bd1`,
    and completion identity `38363fd3`. Its 288-row ledger SHA-256 is
    `f0d35417`; result identity `5e7bb7bf` classifies the direct effect as
    `no_material_direct_effect`. The aggregate refresh-minus-stale effect was
    `-0.076389`, but only seed 67 crossed the stale-support threshold and only
    seed 67 had persistent material policy separation. No condition is
    selected. Preserve the
    [result evidence](../evidence/target-refresh-mature-fork-analysis-recovery-v2-result-2026-08-13.md),
    do not rerun or extend this consumed recovery, and do not infer held-out
    strength, promotion, publication, or long-training authority from it.
32. Mature-fork replication attempt 002 subsequently completed exactly once
    under readiness identity `7088c1f5`, authorization identity `b7384d76`,
    launch identity `c71c2ccb`, and completion identity `94648404`. Its six
    arms used 2,324 new games, 49,152 post-fork transitions and 768 updates;
    every policy-health gate passed. The 288-row no-update ledger SHA-256 is
    `e6036cae`; result identity `8559fa7b` classifies the pooled six-seed
    evidence as `no_replicated_material_effect`. The replication cohort
    favoured refresh, the prior cohort favoured stale control, and the pooled
    `+0.022569` effect did not meet the `1/12` or three-supporting-seed gates.
    No cadence is selected. Preserve the
    [replication evidence](../evidence/target-refresh-mature-fork-replication-v1-attempt-002-result-2026-08-13.md)
    and do not rerun, resume, extend, promote, publish, or infer long-training
    authority from the consumed sequence.
33. The retained research design is recorded in
    [Sanmill no-refresh retained long v4](../experiments/sanmill-no-refresh-retained-long-v4.md).
    It used fresh seed 70, a fresh empty SpecialistDB, the retained-v3
    configuration, and `target-refresh-every=5001` to test permanent
    no-refresh without claiming that the pooled null selected it. Preparation
    attempt 001 at `f1a8974a` is unlaunched, invalidated and permanently
    non-authorizable because its readiness provenance was incomplete. Preserve
    it under the identities in the
    [attempt-001 disposition](../evidence/sanmill-no-refresh-retained-v4-preparation-attempt-001-2026-08-13.md).
    Attempt 002 failed before an accepted checkpoint and consumed its grant.
    Attempt 003 then completed 5,000 games and 20 segments at source `662fe160`
    under plan identity `1702726f` and readiness identity `77cc65ad`. Preserve
    the [result evidence](../evidence/sanmill-no-refresh-retained-v4-attempt-003-result-2026-08-13.md).
    Its source- and seed-confounded comparison with retained v3 remains
    descriptive, and its grant covers no held-out work. The next evaluator
    must be prospectively instrumented because the training rows do not retain
    cap states or strict rule history.
34. The next source-only branch is the
    [retained-v3/v4 passivity diagnostic](../experiments/sanmill-retained-v3-v4-passivity-diagnostic-v1.md).
    It reuses the 64-start corpus explicitly as non-held-out development data,
    pairs the two named final routes within each start/colour unit, and makes
    strict-referee survival beyond total logical ply 120 its primary process
    metric. The prospective ledger also preserves the full ply-120 referee
    state, history-free Malom context, conditional move-value coverage, rules
    terminations and incomplete safety caps. Its two exact route bundles have
    been exported and CPU verified. The evaluator and help-bearing web report
    were introduced at `82c6650`. A preflight-report/launch-gate hardening fix
    at `361d99a` removes raw corpus bodies from readiness and makes skipped
    tests or prefix replay non-runnable. A subsequent full preflight failed
    closed because the v3 original database has preserved sidecars. The plan
    now binds its existing byte-identical audit snapshot plus a new
    byte-identical, sidecar-free v4 snapshot in the ignored diagnostic root.
    The amended plan identity is `035c68f8`; no product authorization or
    diagnostic game exists yet. Do not reuse the attempt-003 training grant.

The previously executed isolated smoke command was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --paths-config data\training_paths.local.json `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2_v4_malom_corrected_fresh_v1 `
  --specialist-db data\specialist_db.smoke.v4_malom_corrected_fresh_v1.sqlite `
  --no-sentinel `
  --no-value-net `
  --no-gap-net `
  --temp-start 0.90 `
  --seed 42 `
  --max-games 1 `
  --batch-games 1 `
  --max-ply 40 `
  --sim-ply-depth 2 `
  --minimal-rollouts `
  --no-s1a-warmstart
```

The command intentionally omitted `--resume`, `--auto-resume-best`, and `--ppo`.
It exited successfully in approximately 24.4 seconds. Its output and database
remain ignored and separate from the intended long-run paths. It is historical
evidence, not a current launch command: the hardened CLI now requires
`--launch` and `--run-id`, and the reviewed command must state its start mode
explicitly. It also predates an explicit imitation-mix disable control. The
experiment document records its verified contents and the checkpoint
observation.

The original handover's 50,000-game PPO command should not be launched
unchanged. PPO and the more complex opponent mixture are optional experiments
under the v5 plan, not prerequisites for a corrected baseline.

## Recorded and Remaining Owner Decisions

The following choices are recorded for the first `dev` experiment:

- start from random model weights, not a historical checkpoint;
- use the corrected v4-style Generalist path, not claim the staged v5 baseline;
- exclude legacy Sentinel, ValueNet, and GapNet from the first run.

The following choices are recorded for the next formal-evaluation design:

- defer the current `GameAI` and validate pinned Sanmill as rule/history owner;
- use one thread, shuffling off, a fixed seed, and fixed node ceilings rather
  than a wall-clock search limit;
- retain Sanmill's normal `DrawOnHumanExperience` opening-depth behavior;
- retain the corrected Sanmill opening book as one future opening source, but
  keep book play off until a deterministic interface and paired policy pass;
- keep HumanDB, patches, and traps off; keep the perfect database out of MTD(f)
  search while allowing a separately audited tied-best prefix sampler;
- treat the 75% book / 25% StrictSteps perfect-database mix and eight logical
  player moves as a historical provisional smoke design, not a frozen formal
  decision; the later eight-ply 7/57 diversity proposal is also historical;
- use twelve logical plies and separately report Book, genuine HumanDB, and
  StrictSteps Perfect DB strata for the current design; the 22/21/21
  composition, 64 source-member identities, HumanDB strict replays, and
  executable corpus are frozen; the retained-v2 protocol has completed and
  its one-run authorization is consumed;
- preserve all 33 D4-unique expert-curated Book placement patterns in a
  separately reported diagnostic catalogue while retaining all 36 source
  records as provenance; its execution protocol is not yet frozen;
- use 60 complete turns only as a bridge/performance smoke ceiling; it is not a
  rules draw or a formal match-length decision; and
- do not run any additional candidate-versus-baseline games until a later
  immutable contract and explicit launch authorization exist.

The product owner delegated routine technical choices for the authorized
managed baseline to the Agent. The resulting immutable plan used A2C, no
imitation warm-start or mixing, 50/50 frozen/heuristic opponents, 500,000
native nodes per heuristic move, full depth-5 rollout, temperature `0.90` to
`0.20`, 5,000 games, seed 42, single-game batching, and 250-game exact-resume
segments. That plan and its authorization are complete historical contracts;
they are not authority for another run. Its `max_ply=60` cap was truncation,
not a rules draw, and must not be copied into a successor plan without a new
decision. The active ruleset has a 100-movement-logical-ply no-progress draw,
so a full-game cap must exceed the placement phase plus that window if the
rule is to be observable.

The product owner should be asked only once for the objective, aggregate game
or wall-time envelope, initial direct or standing launch authority, later
resource expansion, and publication or promotion. A valid standing delegation
removes repeated per-seed, per-arm and per-segment prompts inside its exact
scope. Technical failures remain Agent diagnosis. The local endgame/fullgame
files also remain exploratory unless separately validated and promoted.

The managed plan and its Stage-0 evaluation are complete, and that older
evaluation's authorization is consumed. The retained-v2 held-out grant, the
six-arm downgrade-penalty grant, the four-arm policy-auxiliary calibration
grant, the no-update batch-capture grant, the retained-v3 grant, all six
SpecialistDB read-calibration grants, all eight target-refresh/LR grants and
all attempt-003 grants are also consumed. The equal-transition sequence has
completed all three prefixes and all six paired arms once. Its immutable
result is `inconclusive_late_onset`, and its delegated leaf authorizations are
consumed. The schedule-isolation successor completed once under plan identity
`0580389b`; its analysis recovery also completed once, and every associated
grant is consumed. Its outcome classifier is
`no_material_paired_outcome_effect`, its policy classifier is
`inconclusive_late_onset`, and no condition is selected. No held-out match,
additional calibration, mature-fork training retry, long
training, model promotion/publication, protocol change, resource expansion, or
history rewrite is authorized by that sequence. Ordinary
verified fast-forward pushes of Codex-created `dev` commits are separately
authorized under the standing Git grant recorded in the executive summary;
all other Git history operations remain outside it.

## Reference Material

- [`docs/endgame-training-feasibility.md`](../endgame-training-feasibility.md):
  read-only analysis of the corrected 9/20 phase observation, supplied
  author-`main` bundle, Generalist runtime route, provisional local WDL
  coverage evidence, and remaining questions for the original maintainer.
- [`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md):
  hashes and reproducible diagnostic findings for the newly supplied
  author-`main` checkpoints, logs, screenshots, and related database claims.
- [`docs/evidence/main-integration-audit-2026-07-22.md`](../evidence/main-integration-audit-2026-07-22.md):
  commit-graph-aware `main` integration, staged rebuilt-database validation,
  updated checkpoint identities, v2a boundary, and remaining maintainer
  confirmations.
- [`docs/retrain_v2_plan.md`](../retrain_v2_plan.md): maintainer proposal for
  Sentinel, ValueNet, and GapNet v2 work; useful design input but not a frozen
  or authorized run contract.
- [`docs/v5-specialist-plan.md`](../v5-specialist-plan.md): modular v5 entry
  point, evidence boundary, feasibility-first route, and links to the owning
  oracle, training, human-data, runtime, release, and governance
  specifications.
- [`docs/managed-training-operations.md`](../managed-training-operations.md):
  durable Agent/product authority boundary, managed contracts, commands,
  status model, and stop policy.
- [`docs/malom-fix.md`](../malom-fix.md): decoder investigation and correction
  background.
- [`docs/specialist-db-fix.md`](../specialist-db-fix.md): legacy SpecialistDB
  contamination background.
- Machine-local Sanmill checkout: independent TGF rules, search, and Perfect
  DB reference, with an existing NMM_LLM coordinate/HumanDB codec. See the
  Sanmill entry in the
  [`docs/local-training-layout.md`](../local-training-layout.md) path index;
  use only at a recorded commit and within the documented integration boundary.
- Machine-local `Notes.md` and screenshots: historical maintainer observations,
  not authoritative facts or acceptance evidence. See the reference-only
  `notes` entry in the
  [`docs/local-training-layout.md`](../local-training-layout.md) path list and
  apply the evidence boundary above.
- Machine-local `train_s_gen_v2_handoff_unfinished.py`: preserved, unfinished
  mixed-opponent draft; see the same local-layout entry and treat it as
  reference-only.
