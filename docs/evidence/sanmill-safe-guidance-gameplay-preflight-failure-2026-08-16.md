# Safe-Guidance Gameplay Preflight Failure — 16 August 2026

## Disposition

The preflight failed closed before any gameplay.  No measurement-started
marker, game ledger, complete game, or Sanmill search exists.  The once-only
gameplay authorization remains unconsumed, but the frozen rule prohibiting an
automatic retry means that neither a corrected preflight nor the execution
may start without a new explicit product-owner retry decision.

The failed gate was the six-state frozen-guide canary.  This was a harness
false negative, not evidence that the persisted human estimator changed.  A
persisted `A_pos` row wraps its move under `action["move"]`; the canary passed
the outer envelope to a helper that accepts a direct move.  It therefore
derived the empty key `("", "", "")`.  On stored state
`000a54fb708b...`, the nested move and the persisted expected move both have
the key `("", "b4", "")`.  This diagnosis used only existing JSON and no
additional Malom or Sanmill query.

The comparison code has been corrected and a focused regression assertion
has been added, but the corrected canary has deliberately not been executed.
That preserves the no-automatic-retry boundary.

## Frozen identities and resources

- Plan: `1d368c336db5f49493a2abf3c9e7d507c013d9fed3d14cd928ee988575969cc6`
- Start pool: `385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9`
- Start membership: `cb84ed8180b103d7c25d56a5051fb2476047788505ed0cb9f437c39c9048fb15`
- Authorization: `806e7b674c96ca3f5dd98067a09b6c76bda3db2cca12c75d92ba3cc5f7b495e2`
- Failed implementation commit: `2edfbfe02abbd059d54cf2d7068ce04fedca3859`

Pool construction used 10,638 read-only Malom queries.  The failed canary
completed its six structural fixtures and used 1,000 more, for 11,638 total.
It used zero engine searches, zero games, zero model loads, zero fits, zero
training updates, and zero database writes.  All official holdouts,
research-confirmation, and the remaining 108 source-pool records remained
unopened.

The machine-readable failure record is
`docs/evidence/sanmill-safe-guidance-gameplay-preflight-failure-2026-08-16.json`.
No gameplay result, score difference, conversion rate, or product claim can
be made from this stopped attempt.
