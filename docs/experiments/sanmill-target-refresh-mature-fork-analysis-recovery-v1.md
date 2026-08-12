# Mature target-refresh analysis recovery v1

The machine-readable contract is
[`sanmill-target-refresh-mature-fork-analysis-recovery-v1.json`](sanmill-target-refresh-mature-fork-analysis-recovery-v1.json).
Its plan identity is recorded in that file after canonicalization.
It is
`70fb522b863ceb583b393697a11894540ce3ab5c5764b6aa8e892ebb7cc451e6`.

## Purpose

Mature target-refresh attempt 002 completed all six bounded training arms and
then failed closed before analysis. The first publisher error treated the
frozen pretty-printed phase corpus as though it were a newly generated
canonical authority file. A read-only audit then found that the six Windows
training logs consistently use CRLF framing, while the original publisher
accepted only LF. Both defects are confined to evidence ingestion and have
focused regression tests. No gameplay or training logic is changed.

This plan does not retry, resume, extend, or otherwise mutate the completed
training. It permits only the missing development analysis over the existing
checkpoints. The owning failure evidence is
[`target-refresh-mature-fork-diagnostic-attempt-002-failure-2026-08-12.md`](../evidence/target-refresh-mature-fork-diagnostic-attempt-002-failure-2026-08-12.md).

## Frozen scope

- Candidate inputs: the six completed seed 67/68/69 refresh-mature and
  stale-control branches only.
- Policy analysis: the same fixed 64-position placement/movement/flying corpus
  at 4,096 and 8,192 post-mature-fork transitions.
- Outcome analysis: exactly 288 CPU no-update games using the parent contract's
  twelve audited starts, four replicates, colour swaps, common random streams,
  same-seed mature common-fork anchors, 0.20 sampling temperature, strict
  Sanmill referee, and 120-logical-ply development ceiling.
- Additional training games and optimizer updates: zero.
- Database and checkpoint writes: zero.
- Maximum additional active time: 3.5 hours. Together with the measured
  0.403-hour training sequence, this stays within the original four-hour
  family ceiling.

The analysis may write only its isolated ledger, result, stdout/stderr, and
one-shot control records. It is development mechanism evidence, not held-out
strength, model promotion/publication, retained-setting selection, or long-run
authority.

## Evidence gates

Preflight revalidates all of the following before an authorization can be
recorded:

1. Exact parent contract, readiness, authorization, launch, and fail-closed
   failure identities.
2. The six completed child plan, authorization and controller-event files.
3. Every initial branch, 4,096/8,192 transition checkpoint, final checkpoint,
   training/update log, policy-health report, and closed SpecialistDB.
4. Exact paired transition, learning-rate, temperature and fixed 1,000-node
   Sanmill schedules, plus the existing policy-health acceptance checks.
5. Frozen HumanDB, Malom, MIF, rules and strict Sanmill installation identities.
6. A clean, published `dev` descendant of the training commit whose changed
   paths are entirely within the explicit analysis-only allowlist.
7. Exact publisher and one-shot controller byte identities and empty isolated
   output targets.

The tracked reference corpora retain their existing pretty-printed bytes and
are accepted only after their exact raw SHA-256 and semantic validators pass.
Generated authority and result evidence retains its canonical identity rules.
Uniform LF and uniform CRLF JSONL are accepted; mixed or unterminated framing,
duplicate keys, non-finite values, and non-object rows remain fatal.

## One-shot workflow

1. Publish this plan and the exact implementation by ordinary fast-forward.
2. Run `--preflight`; it writes only ignored readiness evidence.
3. Bind one explicit product decision to the resulting readiness identity.
4. Run `--launch once` with a unique run ID. The exclusive launch marker
   consumes that authorization before model loading or the first game.
5. Any anomaly writes a fail-closed recovery failure. No automatic retry,
   recovery, held-out run, promotion, publication, or long training follows.

Current tracked state: `frozen_unlaunched_needs_authorization`. A complete
preflight has passed, but readiness binds the exact current published HEAD.
The authoritative launch identity therefore belongs only in the ignored
machine-local `readiness.json` generated after the last tracked change; copying
it into this document would move HEAD and invalidate it. Neither this plan nor
preflight authorizes the 288 development games.
