# Retained-v3/v4 safe-progress audit v1

Status: `frozen_zero_game_audit`

This is a read-only reanalysis of the completed 256-game passivity diagnostic.
It plays no game, loads no policy, updates no model, and writes no training
database or checkpoint. The machine-readable
[plan](sanmill-retained-v3-v4-safe-progress-audit-v1.json) is frozen at identity
`3338ba5979db20d89d81bf4408d2fa1eeef098eefb6d854ef56d707ad268fb73`
and binds implementation commit
`d0b600cb3425f37edfa9fda750e6c542ac40b167`.

## Observed facts / 观察事实

The completed source diagnostic is immutable for this audit:

- source plan identity:
  `035c68f80b94dddb8d139d56c38c86c4fde29fa13de5e19db1f4e1fe484c318e`;
- source spec identity:
  `d47c94e29b0acd1c32f2515b90d82716a59f6c5e767b17c2f30ff829e87e5a9b`;
- source result identity:
  `d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0`;
- source completion identity:
  `fe1f243c26f0a9f987e3e24997e23ab296e5c8c6c7a967b5cf57e6f77e89ac64`;
- ledger SHA-256:
  `c064f29d77cedd42a9ef405ec44dbbda045b47be31092e952568cecb5d49b562`;
- 128 matched start/colour units and 256 completed rules-terminal games; and
- corrected Malom manifest identity:
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`.

The source result found 52/128 retained-v3 games and 62/128 retained-v4 games
still ongoing after total logical ply 120. The paired difference was +7.8125
percentage points, with the preregistered engineering interval entirely above
zero. Eventual W/D/L remained overwhelmingly drawn, and all candidate moves
except one retained-v3 move preserved queryable coarse Malom W/D/L.

A ledger-only descriptive scan, before querying alternatives, found:

| Candidate | Candidate turns | Chosen captures | Turns after ply 120 | Captures after ply 120 | Local-FEN revisits after ply 120 |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained-v3 | 5,663 | 331 | 845 | 12 | 27 |
| no-refresh-v4 | 6,095 | 309 | 1,060 | 10 | 14 |

Those raw counts cannot tell whether a capture or a novel move was available
without lowering theoretical W/D/L. They motivate this audit but are not its
answer. The local-FEN scan begins at the frozen prefix endpoint because the
source ledger does not retain all prefix-internal local FENs.

## Hypotheses / 假设

1. The retained-v4 route may encounter fewer W/D/L-preserving capture
   opportunities, select them less often when available, or both.
2. If neither exposure nor conditional selection differs materially, the
   higher retained-v4 120-ply survival should not be attributed to missed safe
   captures on this corpus.
3. Repetition-like behavior is a separate mechanism. A selected local board
   revisit can be labelled avoidable only when another W/D/L-preserving legal
   action reaches a local FEN not yet seen in the audited suffix.

## Counterevidence and limits / 反证与边界

- Malom is history-free. It cannot adjudicate threefold repetition or the
  strict no-capture rule.
- Coarse W/D/L preservation does not make an action optimal under Malom's full
  draw ordering or distance information, and it is not a playing-strength
  label.
- The two routes still differ in seed, training source, target age and
  SpecialistDB lineage. This audit cannot identify a refresh-cadence effect.
- The development corpus and completed games have already been inspected. The
  paired interval describes only fixed-corpus variation, not a population
  confidence interval or held-out evidence.
- The primary missed-opportunity share combines opportunity exposure and
  selection. The report must therefore show exposure and conditional
  selection separately before any mechanism interpretation.

## Frozen method / 冻结方法

Replay every recorded suffix from its stored prefix-end local FEN and verify
every selected complete move and resulting FEN. On each candidate turn:

1. enumerate every complete legal atomic action with the local rules engine;
2. query the exact parent coarse Malom W/D/L and every legal successor from
   the original mover's perspective;
3. mark a `safe_capture_opportunity` only if at least one complete legal
   capture preserves the parent coarse W/D/L;
4. mark it missed when the recorded move is not one of those preserving
   captures; and
5. separately record selected captures, query coverage, local board revisits,
   and the existence of a preserving unseen successor.

The primary per-game value is missed-safe-capture turns divided by all
candidate turns. For each matched start/colour unit, compute retained-v4 minus
retained-v3, then report the mean and two-sided normal engineering interval
with `z=1.96`.

- lower bound above zero: `v4_higher_missed_safe_capture_share`;
- upper bound below zero: `v3_higher_missed_safe_capture_share`;
- otherwise: `inconclusive`;
- interval half-width above 0.02: `inconclusive_precision`; and
- fewer than 30 safe-capture opportunity turns for either candidate:
  `insufficient_safe_capture_opportunities`.

The web report must show candidate-turn denominators, query coverage,
opportunity exposure, conditional selection, post-ply-120 counts, and help text
that distinguishes local-FEN revisits from strict repetition.

## Work and claim boundary / 工作与声明边界

The workload is exactly zero new games, zero model updates, zero database
writes and zero checkpoint writes. The only generated artefact is an ignored,
identity-bound analysis report beside the completed diagnostic. This read-only
audit does not consume or extend the completed 256-game authorization.

Completion may guide whether a future training proposal should test a
safe-progress tie-break inside the set of W/D/L-preserving actions. It cannot
select such a setting automatically, support a held-out or playing-strength
claim, attribute a difference to refresh cadence, promote a model, or publish
or release a model.
