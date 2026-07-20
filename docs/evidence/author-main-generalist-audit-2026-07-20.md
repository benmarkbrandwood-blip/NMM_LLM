# Author Main Generalist Diagnostic Audit - 20 July 2026

## Purpose and Evidence Boundary

This document records a read-only audit of the files and observations supplied
by the original maintainer on 20 July 2026. The owner confirms that every model,
log, plot, screenshot, and database in this delivery was produced while the
maintainer continued training from `main`.

The exact `main` commit, launch manifest, and complete checkpoint state were not
supplied. These files are therefore historical diagnostic inputs only. They
must not be used to resume, initialise, validate, or describe the fresh `dev`
experiment. In particular, a `/home/.../dev/...` directory embedded in legacy
checkpoint metadata is a host path, not evidence of Git branch lineage.

## Supplied Bundle

All paths below are relative to the repository root. The files are retained in
the sibling reference-only `../notes` directory rather than imported into Git.

| Asset | SHA-256 | Read-only observation |
| --- | --- | --- |
| `../notes/best (copy).pt` | `335462EC3A503E316EAAEF63A7669F1A725FC488A2C27E29B39EFD0021B804D6` | Legacy `s_gen_v2` weights, game `17400`, difficulty `9`, temperature `0.5956225` |
| `../notes/best6.pt` | `0E024D4402160BEFA4A7DDEDB56735FCA8CC9D924FC069A550F1761E643CA93D` | Legacy `s_gen_v2` weights, game `12750`, difficulty `6`, temperature `0.6769975` |
| `../notes/train_log.jsonl` | `FE332D39E9CA92552EF79A493C175208DAD4168ADA9E484EDEF1A630C58B250B` | 10,547 valid JSON rows; appended history rather than one frozen run |
| `../notes/update_log.jsonl` | `41D68B45FFB31F7B6207AC7FD58EF906F5FAE626C56A971933473F4CC25FE03B` | 1,190 valid JSON rows with finite but severely inflated PPO policy losses |

The checkpoint tensors contain no non-finite values. Both files are legacy
weights-only payloads with model dimensions and a small set of scalar counters;
they do not contain the optimiser, RNG, run contract, data identity, or full
trainer state required for exact resume. Neither hash matches the historical
Generalist checkpoints already tracked in the repository.

The maintainer first described the observed model as level 10 and then
corrected that statement to level 9. The metadata in `best (copy).pt` supports
the corrected level-9 identification, but does not identify the exact source
commit or prove playing strength.

## What the Plot and Logs Show

The supplied plot was inspected at original resolution. It shows a Generalist
snapshot labelled `n=9944`, a near-100% displayed Malom winning-move rate, an
approximately 20% unknown rate, and a growing separation between policy-top-1
and heuristic-top-1 selection rates. The separate browser screenshot shows the
Generalist checkbox selected during a manual game and a search-progress display.
Neither image supplies a position corpus, colours, opponent identity, move
record, fixed work budget, or confidence interval, so neither is formal
strength evidence.

The JSONL files support a narrower statement about policy concentration:

| Metric | First 500 train rows | Last 500 train rows |
| --- | ---: | ---: |
| `policy_top1_rate` | `0.4231` | `0.8353` |
| `heuristic_top1_rate` | `0.2985` | `0.5090` |
| policy entropy | `1.5485` | `0.3413` |
| top-one probability | `0.4353` | `0.8738` |
| Malom winning-move rate | `0.9828` | `0.9927` |
| Malom unknown rate | `0.1926` | `0.1923` |
| rolling win rate | `0.0163` | `0.1439` |
| temperature | `0.8842` | `0.5872` |

In this trainer, `policy_top1_rate` means that the sampled move equalled the
policy argmax; `heuristic_top1_rate` means that the same chosen move equalled
the heuristic argmax. The trend therefore supports “the policy became much
more concentrated on its own argmax.” It does not by itself prove that policy
choices improved, that heuristic preferences were displaced on disagreement
positions, or that the model became stronger. Heuristic agreement also rose,
and the entropy collapse makes overconfidence a live alternative hypothesis.

The training log is not one immutable experiment:

- its recorded game numbers span `2` through `18649`, but only 9,632 game
  numbers are unique;
- 768 game numbers occur more than once, accounting for 915 extra rows;
- game numbers regress six times in file order, consistent with appended
  restarts or rollbacks;
- 10,464 rows name a historical checkpoint source and only 83 name scratch;
- the easy, hard, blunder, and blend variants first appear around games
  11,452 through 11,483, so the opponent schedule changed during the file;
- after game 15,000, the rows are approximately 49.55% standard heuristic,
  19.96% easy, 9.31% hard, 9.75% blunder, 8.75% blend, and 2.68% frozen.

Every row uses `phase_bucket=main`. The log cannot separate opening, midgame,
and endgame behaviour and therefore cannot confirm the reported endgame deficit.

## Numerical and Configuration Risks

The update log contains no `NaN` or infinity, but its policy-loss scale is not
healthy evidence of stable PPO training:

- median policy loss: `98,759,122.5`;
- maximum policy loss: approximately `1.71e29`;
- last recorded policy loss: approximately `7.80e21`;
- absolute policy loss first exceeds `1`, `1e3`, `1e6`, `1e12`, and `1e18`
  around games 2,758, 4,252, 5,785, 7,124, and 10,250 respectively;
- value loss remains finite and comparatively ordinary, with median about
  `0.94` and maximum about `1.22`.

The machine-local note records a `main` launch with `--ppo --temp-start 1.1`,
whereas the log begins near `0.9`. That observation is consistent with the old
trainer bug that ignored `--temp-start`; the bug and recovery reheating have
been removed on `dev`.

There is also a mechanism-level PPO risk in the inspected trainer family.
Rollout collection stores the chosen action's log probability after dividing
policy logits by the sampling temperature. `scaffolded_ppo_update` recomputes
the new log probability from unscaled logits before exponentiating the ratio.
At temperatures other than one, the old/new distributions are inconsistent
even before a model update. The exact author `main` source commit is missing,
so this audit does not claim that the mismatch is the sole cause of the logged
losses. It is nevertheless sufficient to quarantine PPO for the first `dev`
baseline until a deterministic ratio-at-collection regression test fails for
this reason and a reviewed fix passes.

The author also described levels 9 and 10 as roughly ten-second heuristic
opponents and level 20 as roughly twenty seconds. The current trainer's
automatic opponent budget is about 0.80 seconds at level 9, 1.04 seconds at
level 10, and 14 seconds at level 20. The browser uses a different search path,
and its progress display can include the coordinator search that happens before
the Generalist override. Without logging the effective budget and completed
search work, the manual timing statement cannot define the training curriculum.

## Runtime Route and Endgame Boundary

The author considers direct Generalist access to endgame data plausible but is
unsure whether it is desirable. Current code has a concrete train/inference
seam that must be resolved before formal evaluation:

- training constructs `LookaheadAdvisor(endgame_db=db)` with the configured
  Malom database;
- `load_generalist()` constructs the browser Generalist advisor without an
  `endgame_db` argument;
- `GeneralistAgent.score_moves()` explicitly encodes with `db=None`; its
  `set_db()` method stores a database reference that this score path does not
  consume;
- the browser builds the Generalist with globally loaded Sentinel, ValueNet,
  GapNet, HumanDB, and SpecialistDB objects before the UI checkboxes are used.

Consequently, an unchecked UI box is not proof that a feature was absent from
the Generalist feature encoder, and the Generalist's training-time Malom
lookahead is not currently mirrored by its browser inference route. The final
Generalist override also replaces the already-computed coordinator move when
scoring succeeds. A formal evaluation must therefore name the actual route and
emit a component/data manifest rather than infer configuration from the UI.

## Other Supplied Data

The updated legacy SpecialistDB passes SQLite integrity checking and contains
1,954,437 positions, 339,904 unversioned Malom labels, 54,456 winning lines,
and 27 preferred plays. All 27 preferred-play rows are marked promoted. This
supports the maintainer's narrow statement that the database contains favourite
plays, but it does not repair the missing `sector-corrected-v1` metadata. The
database remains quarantined, read-only, and outside the first `dev` experiment.

The maintainer explicitly said that the internal endgame databases had not been
checked. The repository audit independently found missing coverage and heavily
unknown loaded tables. Keep those files disabled as authoritative labels until
the documented validation or rebuild is complete. HumanDB also remains on its
older 94,429-game inventory; its 406 newly delivered game files require a
controlled rebuild or processed-path migration rather than a blind update.

## Consequences for `dev`

The author bundle changes diagnostic priorities, not the fresh `dev` lineage:

1. Keep the first baseline on A2C and omit `--ppo`.
2. Add and test an explicit `--no-imitation-mix` control before claiming a
   pure fresh-RL run; `--no-s1a-warmstart` currently disables only the initial
   warm-start. The current machine happens not to contain the default NPZ, but
   absence is not equivalent to an explicit experiment decision.
3. Make the proposed fixed-work heuristic opponent contract expressible and
   observable. Current automatic negative `--time-budget` values select a
   wall-clock budget rather than disabling wall-clock search.
4. Persist the accepted experiment choices and all effective component/data
   identities in the launch contract, including the imitation-mix state.
5. Run a newly authorised disposable smoke only after those controls exist;
   make it long enough to exercise at least one RL update, not merely process
   initialisation.
6. Before formal strength evaluation, align or deliberately distinguish the
   training and inference routes, freeze a compatible baseline bundle and
   phase-stratified colour-swapped corpus, and record fixed work per move.
7. Treat a legality/Malom replay of the 27 preferred plays and a phase-stratified
   replay of the author checkpoint as useful later diagnostics, not baseline
   prerequisites.
