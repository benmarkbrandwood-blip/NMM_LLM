# Retained-v2 held-out evaluation v1 authorization

## Status

Status: `unconsumed_pending_runner_and_final_preflight`

The product owner authorized one execution of the frozen
[`sanmill-corrected-retained-v2-heldout-eval-v1`](sanmill-corrected-retained-v2-heldout-eval-v1.md)
protocol on 9 August 2026. The grant is intentionally recorded before the
runner exists so that implementation cannot change the product decision. It
does not permit a game to start until every implementation and final
read-only preflight gate in the plan passes.

The machine-readable authorization is
[`sanmill-corrected-retained-v2-heldout-eval-v1-authorization.json`](sanmill-corrected-retained-v2-heldout-eval-v1-authorization.json).
It binds plan commit
`106d015b23debee7d5c8d691195ff958da66f1fc`, plan identity
`212076e9423b671b83783efef411db3b4a56c8c67ae36a463d381d6939d4d982`
and plan-file SHA-256
`06f168d1687557a9146455fae0a8174c7714b7dd864cfd5a1e2c383c26009b21`.
Its own canonical authorization identity is
`6426ffd109d28145a4148855d70a181d50fd4277068fe01b501934d212378fb1`.

## Authorized scope

The grant covers exactly one 64-pair, 128-game evaluation, with a maximum of
six active evaluator hours, under the candidate, corpus, Sanmill baseline,
rules, statistics and output contract already frozen in the plan. After every
gate passes, the evaluator may launch without another product-choice prompt.

A host interruption may resume only the unplayed suffix of the same validated,
hash-chained ledger. It may not replay a completed game or create a second
sample. The grant is consumed when the first corpus game is opened for play or
the candidate is asked for its first corpus move. Read-only readiness checks,
synthetic protocol probes and non-corpus canaries do not consume it.

## Not authorized

This grant does not permit another training run, a candidate or baseline
substitution, changed starts or thresholds, a larger resource envelope,
result-based early stopping, an extra evaluation, model promotion or model
publication. An invalid or inconclusive result remains final for this grant;
changing the protocol or rerunning requires a new product-owner decision.

No evaluation was started while this authorization was written.
