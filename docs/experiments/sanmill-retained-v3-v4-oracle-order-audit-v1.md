# Retained-v3/v4 complete Malom order audit v1

Status: `frozen_zero_game_audit`

This is a second read-only reanalysis of the completed 256-game passivity
diagnostic. It plays no game, loads no policy, updates no model, and writes no
training database or checkpoint. The machine-readable
[plan](sanmill-retained-v3-v4-oracle-order-audit-v1.json) is frozen at identity
`95e1d5e6640765e14852b9dfc3f2793bf72ee583bc95fc0a3bd1512acb36d23d`
and binds implementation commit
`226074a64f60810218bb3f23d8f33a7d36735f18`.

## Observed facts / 观察事实

The source diagnostic remains bound to result identity
`d250f03d72b535c0249bdf0ada7d5a75d91f7fcc44e8926c4f6dfba35d2e63d0`
and ledger SHA-256
`c064f29d77cedd42a9ef405ec44dbbda045b47be31092e952568cecb5d49b562`.
It found retained-v4 more often rules-ongoing at total logical ply 120, while
eventual W/D/L remained overwhelmingly drawn.

The completed first zero-game audit is also immutable for this plan:

- audit plan identity:
  `3338ba5979db20d89d81bf4408d2fa1eeef098eefb6d854ef56d707ad268fb73`;
- result identity:
  `b60eaf6392d55e520b5a2a493ce7dd8961c05e811a7fd3cbb5375735fe312fea`;
- report SHA-256:
  `d68b8279cc65a429a900388396efde50da00cd668fcf5249b94177cb940a12b1`.

That audit found full coarse-WDL query coverage. Retained-v3 selected a
W/D/L-preserving capture on 330 of 331 safe-capture opportunity turns;
no-refresh-v4 selected one on all 309 of 309. After total logical ply 120,
both selected every available preserving capture. The primary paired missed-
capture difference was inconclusive. Retained-v4 also had fewer local-FEN
revisits after ply 120. The observed prolongation therefore is not explained
by either repeatedly declining an immediately available safe capture or by a
higher local-board revisit rate in this audited suffix.

Coarse W/D/L is deliberately lossy. Malom retains an ultra-strong complete
ordering whose sector-corrected `OracleMoveValue.ordering_key()` has already
been validated against reference behavior, including its sign-dependent
`key2` direction and draw ordering. The corrected manifest identity remains
`f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`.

## Hypotheses / 假设

1. The retained-v4 route may be less aligned than retained-v3 with the
   complete Malom ordering even when both preserve the same coarse W/D/L.
2. If retained-v4 has higher complete-order regret, that positional ordering
   gap is a candidate mechanism for further controlled study; it still does
   not prove that the gap caused longer strict-rules games.
3. If no directional ordering gap exists, the remaining mechanism is more
   likely to involve history-aware liveness, opponent interaction, or policy
   path selection that neither immediate captures nor positional ordering
   alone captures.

## Counterevidence and limits / 反证与边界

- Complete Malom ordering is positional and history-free. It does not include
  threefold repetition or the strict no-capture clock.
- `key2` is not a globally monotonic distance. The audit uses the validated
  two-key comparator only and makes no distance-to-terminal claim.
- A higher ultra-strong ordering grade is the perfect positional baseline,
  not a measured probability of defeating this Sanmill configuration or a
  human.
- The candidates differ in seed, source, frozen-target age and SpecialistDB
  lineage. Any difference describes these named routes only and cannot be
  attributed to refresh cadence.
- The corpus is reused development evidence. The engineering interval is a
  fixed-corpus variation summary, not population inference or held-out
  strength evidence.

## Frozen method / 冻结方法

Replay every recorded suffix and verify every move and resulting local FEN.
For each candidate turn:

1. query the exact parent W/D/L and every complete legal atomic action;
2. retain only actions whose mover-perspective coarse W/D/L equals the exact
   parent projection;
3. require every action in that preserving set to carry a complete
   `OracleMoveValue` in one common parent sector and viewpoint;
4. compare the distinct complete ordering grades with the validated ordering
   key; and
5. record the selected distinct-grade rank.

The normalized ordinal regret is zero for the best grade. Otherwise it is the
zero-based chosen distinct-grade rank divided by the maximum rank, so it lies
in `[0,1]`. Forced or oracle-tied preserving turns contribute zero. This is an
ordinal statistic: the numeric distance between `key1` or `key2` values is not
treated as cardinal regret.

The primary per-game value is mean normalized ordinal regret over turns where
the recorded choice preserves coarse W/D/L and the full preserving set is
orderable. For every matched start/colour unit, compute no-refresh-v4 minus
retained-v3 and report the mean with a two-sided normal engineering interval
using `z=1.96`.

- lower bound above zero: `v4_higher_full_order_regret`;
- upper bound below zero: `v3_higher_full_order_regret`;
- otherwise: `inconclusive`;
- interval half-width above 0.03: `inconclusive_precision`;
- orderable coverage below 0.99 or fewer than 128 supported matched units:
  `insufficient_full_order_coverage`; and
- fewer than 500 distinct-grade choice-opportunity turns for either candidate:
  `insufficient_ordering_opportunities`.

Secondary metrics separate opportunity exposure from selection, report best-
grade selection conditional on an opportunity, retain all denominators and
coverage, and repeat the summaries after total logical ply 120, by candidate
colour, and by source stratum.

## Work and claim boundary / 工作与声明边界

The workload is exactly zero new games, zero model updates, zero database
writes and zero checkpoint writes. The generated report is ignored and
identity-bound beside the completed diagnostic. This audit does not consume or
extend the completed 256-game authorization.

Completion can decide only whether complete positional ordering alignment is
a useful next mechanism to test. It cannot prove passivity causation, identify
a refresh effect, establish playing strength, select a training setting
automatically, promote a model, or publish or release a model.
