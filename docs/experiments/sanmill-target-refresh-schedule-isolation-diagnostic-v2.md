# Sanmill target-refresh schedule-isolation diagnostic v2

Status: `designed_unlaunched_needs_publication`

Machine-readable contract:
[`sanmill-target-refresh-schedule-isolation-diagnostic-v2.json`](sanmill-target-refresh-schedule-isolation-diagnostic-v2.json)

Plan identity:
`0580389b3d696df9859ac9e7aea6c4b478bf6e791b7e27bf780d2a6e02db5b0b`

This is a development mechanism experiment. It does not authorize execution,
does not choose a retained model, and is not held-out strength evidence.

The earlier unlaunched identity
`9c8068308d9f74623555371e789ba12a0138a3eb895f0aa379184e8380a39b05`
is preserved in Git history but is superseded. The scientific design and
resource envelope did not change. The current identity only refreshes the
source lineage after reviewing `origin/main` through `028ef8e`, binding the
[review evidence](../evidence/origin-main-training-review-2026-08-11.md), and
recording the independently tested UTF-8 portability repair. No `main`
training or gameplay commit was imported.

The reviewed main tip is an ancestry boundary. Readiness fails if that commit
is no longer an ancestor of `origin/main`, while ordinary fast-forward
descendants are reported by exact current tip and unreviewed-commit count
rather than invalidating this frozen experiment by themselves.

## Observed facts

The completed equal-transition v1 diagnostic matched 8,192 consumed learner
transitions per arm. It first showed material policy separation at the final
boundary in two seeds, but it did not establish a persistent direction. It
also left two post-treatment schedules coupled to game count:

- paired arms reached the same transition boundary after different numbers of
  complete games;
- their training temperatures therefore differed; and
- two refresh arms crossed into a 5,000-node Sanmill stage while their paired
  controls remained at 1,000 nodes.

No v1 arm won or reached a rules draw against Sanmill. That outcome stratum was
floor-limited and could not compare the candidate policies. Wins against the
frozen opponent were also not a common-baseline measurement because the
refresh and no-refresh treatments deliberately changed that opponent.

These facts support a more isolated successor, not either target-refresh
choice. They do not prove that temperature or node work caused the late
separation.

## Hypothesis

A one-time copy of the game-50 candidate into the frozen target causes a
persistent action-distribution difference and a same-direction paired
development-outcome effect when both conditions receive:

- the same first 8,192 optimizer-consumed post-fork transitions;
- the same temperature at every learner-transition ordinal; and
- the same 1,000-node Sanmill training-opponent ceiling.

The contrast is `no-refresh minus refresh-once`. A positive outcome effect
favours no refresh; a negative effect favours refresh once.

## Controls

Seeds 67, 68 and 69 each create one fresh 50-game shared prefix. The two arms
for a seed resume from descriptor-rebound envelopes over the same fork payload
and from independent byte-identical SpecialistDB clones. The only treatment is
the game-50 target decision. No later target refresh is permitted.

The prefix retains the established global-game temperature schedule. At the
fork, both arms start from that exact game-50 temperature. Every subsequent
learner transition receives the temperature for its FIFO post-fork generated
ordinal. Because the pending queue is consumed in order, the first 8,192
consumed transitions have the same temperature sequence in both arms even if
their complete-game counts or pending-queue lengths differ. Pending overflow
beyond the boundary is not trained and is excluded from the comparison.

Sanmill remains at level 1 and 1,000 nodes for the full paired horizon. This
experiment does not test whether a stronger Sanmill opponent improves learning.
It also does not test Sentinel, ValueNet, GapNet, imitation, auxiliary Malom
loss, recovery, opening forcing, or another warm-start. Ben's observations
about auxiliary signals, frozen-opponent share, and opponent strength remain
separate hypotheses for later experiments.

## Candidate-blind development outcomes

Policy distributions are still compared on the frozen 64-state
placement/movement/flying feature corpus at 1,024, 2,048, 4,096 and 8,192
consumed transitions.

At 4,096 and 8,192 transitions, each candidate also plays from the already
frozen 12-position replay corpus. The corpus contains four placement, four
movement, and four flying starts, with source W/D/L coverage. It was selected
and strictly replay-audited before any v2 candidate exists. Each candidate
plays both colours against its own seed's common game-50 anchor, using paired
random seeds and sampling temperature 0.2.

The complete grid is 288 no-update games:

`3 seeds × 2 boundaries × 12 starts × 2 colours × 2 conditions`.

These starts are deliberately a development measurement. They must never be
renamed held-out, used as a formal promotion suite, or interpreted as general
playing strength.

## Frozen decision rule

The policy classifier keeps the predecessor's preregistered JS, total
variation, Malom-preserving mass, phase, seed, and persistence thresholds.
Top-1 disagreement remains interpretive rather than a standalone gate.

An outcome-supported target effect additionally requires all of the following:

- the policy classifier returns `materially_diverged`;
- the aggregate paired score difference at 8,192 transitions has magnitude at
  least `1/12`;
- at least two seeds have the same direction at both 4,096 and 8,192, with
  magnitude at least `1/24` per seed;
- no phase has an opposite final effect larger than `0.25`;
- the selected direction does not raise max-ply truncation rate by more than
  `0.10`; and
- no phase has an opposite signed Malom-mass effect larger than `0.05`.

Passing these gates may select a target-refresh condition for the next
long-run design. It does not launch that run or promote a checkpoint. Failure
or disagreement stops the sequence without automatic retry or extension.

## Evidence and counterevidence to retain

The final report must keep training curves, exact update counts, actual
per-batch temperature minima/means/maxima, Sanmill node observations, all
three seeds, phase and colour strata, termination reasons, policy distances,
Malom mass, and paired W/D/L together. It must explicitly separate:

- observed facts;
- causal hypotheses;
- supporting evidence;
- counterevidence; and
- the next discriminating experiment.

There is no supervised train/validation split in this online RL diagnostic.
Training losses and development outcomes must not be described as validation
curves.

## Resource and authority boundary

The frozen safety ceiling is unchanged: at most 3,600 contract games, 3,450
actual training games, 49,152 post-fork consumed transitions, and six active
hours across the three prefixes and six arms. The 288 no-update development
games are accounted separately and cannot write optimizer or training data.

The contract contains zero launch, segment, promotion, and publication
authority. Preparation may create only fresh database copies, authorization-
free managed plans, preflights, and readiness evidence. A later bounded launch
requires the experiment's separate execution gate.

On 11 August 2026 the product owner replied “同意” to the exact one-shot parent
grant presented for this experiment: at most 3,600 contract games, 3,450
actual training games, 49,152 post-fork consumed transitions, six active
training hours, and the fixed 288-game no-update development grid, with no
retry, recovery, resume, extension, held-out work, promotion, publication, or
long-training fallback. The grant was not consumed because a final fetch
advanced `origin/main` before the authorization record was written. Under the
repository standing-delegation rule, it remains valid for this same experiment
family after a source-lineage-only identity refresh, provided the scientific
design, resource envelope, claim boundary, stop rules, and prohibited actions
remain byte-equivalent. The regenerated parent authorization must still bind
the current technical readiness just in time. The canonical delegated product
scope (the contract excluding only `plan_identity` and `lineage`) is frozen as
`a92e87bebe87e1a287be37c95c0974cafde662703ee05436a2c30b7d9584211a`;
the previous approved contract and this lineage refresh have that same digest.

## One-shot parent execution

The operational entry point is
`scripts/run_target_refresh_schedule_isolation_sequence.py`. Its preflight
revalidates the exact published source, contract and readiness identities,
fresh prefix plans and databases, absent leaf authorizations, absent segment
outputs, and absent deferred arm and result targets.

An explicit parent product decision is recorded separately as
`nmm.target-refresh-schedule-isolation-sequence-authorization.v2`. The grant
binds the exact product-facing sequence readiness identity as well as the
underlying managed readiness, plan identity, source commit, aggregate game,
transition, measurement and wall-time limits, exact launch order, claim
boundary, permitted operations and prohibited actions. The development
publisher is fixed to CPU. As defined above, the six active hours apply to the
three prefixes and six training arms; the no-update analysis remains bounded
separately by its exact 288-game grid. The grant is consumed when the one
parent launch attempt starts.

Within that single attempt, the runner may record each exact child plan as
`product-owner-delegated-agent` immediately before use. It executes one prefix,
prepares only that seed's real fork-derived arms, executes those two arms, and
then proceeds to the next seed. The result publisher runs only after all nine
managed plans complete. The first exception stops the sequence and writes a
failure record; there is no retry, recovery, resume, extension, held-out run,
promotion, publication or long-training fallback.

The parent reconciles all three 50-game prefixes, all six absolute arm game
counts, actual post-fork game execution, and exactly 49,152 consumed learner
transitions before the 288-game no-update publisher may run. Authorization,
launch, failure and result records use exclusive creation and cannot overwrite
an existing one-shot record.
