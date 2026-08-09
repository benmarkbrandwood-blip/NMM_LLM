# Retained-v2 held-out runner pre-publish readiness evidence

Date: 2026-08-09

Status: `implementation_verified_awaiting_publish_and_final_preflight`

This evidence concerns only the frozen retained-v2 held-out evaluation. It is
not an evaluation result, a model-promotion decision, a publication claim, or
authorization for new training.

## Outcome

Implementation commit `e32d9d46a361d2ed6877b669cdf653eba78e3f3c`
provides the dedicated evaluator, guarded CLI, immutable game ledger, exact
suffix-resume checks, active-time ceiling, recomputable statistics, and
focused tests.

The complete read-only preflight passed every non-publication gate. The only
failed gate was:

```text
repository: held-out evaluation requires dev == origin/dev
```

This is intentional. At the audit point, local `dev` was five commits ahead
of `origin/dev`. No corpus game was opened, no corpus move was requested from
the candidate, and the one-run grant remains unconsumed. The evaluation is
therefore not yet ready to launch.

## Verified facts

- The frozen plan, authorization, 64-start corpus, 34-start strict
  independence subset, candidate checkpoint, route bundle, HumanDB,
  SpecialistDB, Malom, MIF release, trainer rules, and strict Sanmill runtime
  all matched their recorded identities.
- The candidate produced the same legal CPU/float32 argmax move twice on a
  synthetic non-corpus board. It was not asked for a corpus move.
- A real two-turn non-corpus interoperability canary completed candidate
  `d6`, then Sanmill `f4`, with no fallback or scoring.
- All 64 twelve-logical-ply starts replayed exactly in 64 fresh strict
  Sanmill processes. The replay observation identity is
  `48c63007f0f6b617ae099419ad23b9f340e895f3df3ccc2c7d049102c88c7648`.
- The fixed Sanmill search ceiling is 500,000 nodes per logical turn. The
  opening canaries completed at depth 1 with 50--52 observed nodes; this is
  valid ceiling behavior with the frozen human-experience opening setting,
  not evidence that every turn consumes 500,000 nodes.
- No competing trainer or evaluator process was found.
- Evaluation result targets were absent and ignored by Git. The existing
  candidate route bundle was verified separately and was not treated as a
  result target.

## Verification

The final pre-publish run reported:

```text
held-out focused tests: 78 passed
mandatory Malom/provenance tests: 103 passed, 498 subtests passed
Ruff: passed
```

An additional dependency batch covering the training referee, node
calibration, route bundle, checkpoint envelope, and run contract reported
`59 passed`. After recording the machine-readable evidence, its identity and
unconsumed-state regression increased the current focused total to
`79 passed`.

The wider legacy `test_sanmill_uci.py` run reported 41 passed and 4 failed.
Those four machine-local tests require an immutable `db65eb3e` binary with
SHA-256 prefix `cac2ec6f`; the current moving Sanmill checkout has changed the
pinned source scope and that exact binary is absent from the configured
runtime locations.
They exercise a different historical smoke contract, not the exact
`a6623f8` training runtime used here. This counterevidence prevents any claim
that the full suite is clean. It does not weaken or replace the held-out
runtime gates, all of which passed against the exact isolated training
checkout.

The current `origin/main` tip was also reviewed at
`bc46b51e69724e12a8e5f17e3ff696b9f88456d9`. Its recent commits concern the
separate v2c, GapNet, puzzle, explorer, and autosave lineages. None should be
cherry-picked into this frozen retained-v2 evaluator before its one allowed
run.

## Remaining gate and exact next action

1. Publish the existing `dev` chain by ordinary push; do not rewrite it.
2. Rerun the complete read-only preflight:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_heldout_evaluation.py preflight
   ```

3. Proceed only if it exits zero and reports
   `ready_for_heldout_evaluation` with every gate passing.
4. The already frozen one-run authorization then permits exactly one launch:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_heldout_evaluation.py run --launch
   ```

An interrupted process may use `preflight-resume` followed by `resume
--launch` only for the missing suffix of the same specification. A completed
or failed run cannot be restarted, and the 1,536-post-prefix-ply safety ceiling
invalidates the run rather than manufacturing a draw or loss.

The machine-readable companion is
[`sanmill-corrected-retained-v2-heldout-runner-readiness-2026-08-09.json`](sanmill-corrected-retained-v2-heldout-runner-readiness-2026-08-09.json).
