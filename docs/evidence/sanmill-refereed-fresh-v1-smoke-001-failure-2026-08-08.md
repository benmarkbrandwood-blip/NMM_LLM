# Fresh Sanmill-Refereed Smoke 001 Failure — 8 August 2026

## Result boundary

Status: `failed_quarantined`

The product owner separately authorised publication of the implementation and
one exact two-game smoke. `dev` and `origin/dev` both pointed to clean commit
`aeac29ca9c4a130bc7378874948b007814f6bfe0` before launch. The final read-only
preflight returned `ready_for_smoke`, with no errors or unresolved decisions.

Run `sanmill-refereed-fresh-v1-smoke-001` then failed closed during its first
primary game. It completed no counted game, no optimiser update, no
checkpoint, and no training or update log. The one-run authorisation is
consumed by this failed launch. This record is failure and integration
evidence, not a completed smoke, throughput benchmark, or training result.

## Frozen launch identity

The run manifest records the complete command and resolved configuration. Its
important identities are:

| Field | Value |
| --- | --- |
| Git commit | `aeac29ca9c4a130bc7378874948b007814f6bfe0` |
| Experiment | `dev-v4-sanmill-refereed-fresh-v1` |
| Experiment digest | `sha256:0841df83481f178e7b221ab3908e072f72430ca0342cf4ebd9e1e94c734cf99a` |
| Config SHA-256 | `69ad66b23129410b98e8b3af663bb3c020e4a2433050d0a48a21b372c2d1b06e` |
| Run ID | `sanmill-refereed-fresh-v1-smoke-001` |
| Workload | two games, one 1,000-node Sanmill level, `max_ply=120`, batch one |
| Schedule | one Sanmill-search game and one frozen-target game at seed 42 |
| Curriculum | disabled |

The ignored raw manifest is 6,935 bytes with SHA-256
`90ba4a339455430807718e9d72fadaffcd44aa50f14fd40409b2f71a057dc67e`.
The three-event ledger is 1,486 bytes with SHA-256
`518f3d8596c0f859a1e22d7ff77c1104375ceb90678ed6db4eb3fa5643dfbfb8`.
It records `preflight_passed`, `training_started`, and `training_failed` with
exception type `PhaseCorpusError`.

## Failure and root cause

The first game reached a legal Sanmill rule terminal. Sanmill's structured
state identified a game-over snapshot, but its raw TGF FEN used phase `o` and
action `?`. That is an expected Sanmill representation: the pinned source maps
`GameOver` to `?`.

NMM_LLM incorrectly passed that terminal FEN to the phase-corpus projector,
which intentionally accepted only stable placing and moving actions `p` and
`s`. The adapter therefore raised:

```text
unsupported stable TGF action: '?'
```

This was an NMM_LLM terminal-mirror omission, not a Sanmill rules defect. The
fail-closed boundary worked: the trainer did not substitute its local referee,
ignore the mismatch, or continue to the second game.

Commit `4e734e4a3105b1a590fbb11ab13c3197cb6a9fce` fixes only that projection
boundary. A caller must explicitly identify a structured terminal state. The
adapter then validates the game-over phase and raw action, normalises a local
copy solely for board projection, and continues to use Sanmill's structured
state as the terminal and outcome authority. Non-terminal projection remains
strict.

## Quarantined database and diagnostic side effect

Immediately after the failed launch, the dedicated SpecialistDB passed
`quick_check`, retained `sector-corrected-v1`, was bound to the failed run ID,
and still contained zero positions, winning lines, preferred plays, or Malom
labels.

Two later no-optimiser diagnostic replays exercised the repaired routes. The
rollout helper nevertheless invokes its evidence-persistence hook, so these
diagnostics wrote 94 positions and one winning line into the already failed
database. No checkpoint or formal game log was produced. This side effect was
detected by a post-diagnostic read-only audit and is retained here rather than
rolled back or hidden.

The quarantined database is now 61,440 bytes with SHA-256
`d9e41bdb47b1ec8d3c10f3ace51b318105ecfabab88aa27c8205bf398262a02d`.
Its `training_lineage_root_run_id` remains
`sanmill-refereed-fresh-v1-smoke-001`. It must never be used as a fresh input,
resume input, or retry database.

## Verification after the repair

- The focused red regression reproduced rejection of an `o ?` terminal FEN
  before the fix.
- Six Sanmill training-referee tests pass after the fix, including replay of
  the exact 43-logical-ply, 48-action-token history that exposed the defect.
- The wider trainer, launch, checkpoint, resume, bridge, and referee selection
  reports `182 passed, 6 deselected`. The deselections are the documented
  historical tests tied to a moving reference checkout, not tests weakened for
  this fix.
- The mandatory Malom and label-provenance group reports `103 passed, 498
  subtests passed`.
- Ruff and `git diff --check` pass for the changed scope.

The repaired deterministic Sanmill-search route ended after 43 logical plies
with `loseNoLegalMoves`; it took about 8.9 seconds and recorded 22 search calls
and 19,423 aggregate nodes. The repaired frozen-target route ended after 52
logical plies with `loseFewerThanThree`; it took about 20.7 seconds. These are
diagnostic route observations. They exclude process publication, optimiser
update, checkpointing, and an intact official run lifecycle, so they are not a
retained throughput benchmark and cannot freeze a long-run node ladder.

## Retry boundary

Any retry must use all of the following:

- a new run ID ending in `smoke-002`;
- a new absent output directory;
- a newly created empty `sector-corrected-v1` SpecialistDB with no lineage;
- the published repair commit and a clean tracked worktree;
- a repeated read-only `ready_for_smoke` preflight; and
- a new explicit one-run authorisation.

The failed output directory and database remain quarantine evidence. No retry,
long run, node curriculum, or advancement rule is authorised by this record.
