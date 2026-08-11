# Schedule-isolation v2 attempt-001 failure evidence

## Verdict

The bounded training work completed, but the parent sequence failed closed
before any development game ran. The failure is an evidence-reader defect, not
a demonstrated trainer, checkpoint, database, referee, or numerical failure.
The original launch is consumed and must not be retried.

## Observed facts

- Parent plan identity:
  `0580389b3d696df9859ac9e7aea6c4b478bf6e791b7e27bf780d2a6e02db5b0b`.
- Training source commit:
  `49defb8a79d07a19c035ca6c1f23f266ae5ed2b2`.
- Run ID: `schedule-isolation-v2-2026-08-11-attempt-001`.
- Parent launch identity:
  `4346a846d86d5609e7557b3b06e54cb0aa148a3770825d306d09551bd44804e8`.
- Failure identity:
  `a4aa91bb25a23040c93aeed1166b8290b653b3d4181b682e8222afb3018c9ec5`.
- All three 50-game prefixes and all six post-fork arms completed once.
- The six arm checkpoints reached exactly 8,192 consumed transitions each,
  for 49,152 post-fork transitions in total.
- The six arm absolute game counters were 560, 439, 562, 327, 576 and 518.
  After subtracting the shared 50-game prefix from each resumed arm, the
  experiment executed 2,832 distinct training games, not 3,132.
- Managed active training time was 0.4824 hours, below the 6-hour ceiling.
- Every arm retained fixed 1,000-node Sanmill work and its paired condition
  used the same 128 batches of 64 transitions and byte-identical temperature
  exposure.
- No development outcome ledger or result exists. Therefore zero of the
  planned 288 no-update development games ran.
- The publisher rejected line 1 of the first Windows `update_log.jsonl` before
  loading candidate models or starting an outcome game.
- The rejected file contains 128 LF bytes, all 128 preceded by CR; it is a
  uniformly CRLF-framed, final-newline-terminated Windows JSONL file.

## Hypothesis

The sole observed failure was caused by the publisher accepting only LF JSONL
while the Windows trainer correctly emitted uniform CRLF. Accepting uniform LF
or uniform CRLF, while still rejecting mixed, embedded, or unterminated line
framing, is a semantics-preserving evidence-reader repair.

## Supporting evidence

- Commit `318b23d05729423a89b2f867ee3441700df2424a` added the deterministic
  CRLF reproduction and focused framing tests.
- Commit `1c975edc74a3b2e2f0b7730d8ab3bded9664fb7b` bound result publication to a
  clean, published descendant of the frozen training source and records the
  frozen and executed publisher identities separately.
- A read-only audit of all six real logs succeeds after the framing repair and
  confirms the exact transitions, temperatures, node budgets, and per-seed
  Sanmill game counts.
- No training or gameplay implementation changed after the frozen training
  source; the reporter mechanically rejects any non-allowlisted intervening
  tracked path.

## Counterevidence and remaining gap

- There are no 288-game outcomes yet, so this evidence cannot decide whether a
  one-time target refresh helps, harms, or has no material effect.
- Checkpoint validity for policy comparison and replay outcomes has not yet
  been exercised by the repaired publisher.
- The consumed parent authorization explicitly prohibited automatic recovery.
  A separate, bounded analysis-only authorization is therefore required even
  though no training will be repeated.

## Next validation experiment

Use the immutable
[`analysis-recovery v1 plan`](../experiments/sanmill-target-refresh-schedule-isolation-analysis-recovery-v1.md)
to run exactly one CPU analysis attempt over the already completed checkpoints:
288 fixed development games, zero training games, zero optimizer updates, and
zero database or checkpoint writes. Any anomaly stops the attempt; it cannot
retry, extend, run held-out evaluation, promote or publish a model, or start
long training.
