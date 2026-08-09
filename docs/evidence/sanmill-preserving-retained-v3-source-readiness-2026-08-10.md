# Sanmill-Preserving Retained v3 Source Readiness — 10 August 2026

## Verdict

`needs_decision`

The source, frozen experiment contract, focused tests and local input
identities are technically ready for publication. The only current decision
is ordinary publication of the local `dev` commits. The final fresh database,
immutable managed plan and first-segment preflight must be created only after
`HEAD == origin/dev`; this record does not authorize training.

## Gate summary

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Repository | sole root, branch `dev`, tracked worktree clean | repository root and clean `dev` | pass |
| Tested source | `a26c0383d92a4e020608ffe74c3de5e17f901da4` | frozen retained-v3 contract included | pass |
| Publication | tested source and this evidence are not yet on `origin/dev` | `HEAD == origin/dev` before plan creation | needs decision |
| Maintainer branch | `origin/main` remains `bc46b51e69724e12a8e5f17e3ff696b9f88456d9` | no unreviewed change selected | pass |
| Output isolation | retained-v3 control directory absent | fresh target absent | pass |
| Database isolation | retained-v3 SpecialistDB absent | fresh target absent | pass |
| Empty DB template | 45,056 bytes, expected SHA-256, `quick_check=ok`, current label version and zero rows | exact closed template | pass |
| Sanmill runtime | commit, tree and release binary match the frozen identities | exact isolated training runtime | pass |
| HumanDB | 738,091,008 bytes; main-file SHA-256 unchanged | empirical frequencies and outcomes only | pass |
| Malom manifest | file SHA-256 equals its frozen identity | `sector-corrected-v1` manifest | pass |
| Focused training chain | 265 passed | manager, preflight, checkpoint, resume, referee, reward and health paths pass | pass |
| Mandatory label chain | 103 passed and 498 subtests passed | Malom, DB teacher and provenance pass | pass |
| Static checks | changed and launch-critical modules pass Ruff; `git diff --check` passes | no new static failure | pass |

The pytest cache warning was limited to the host-denied repository cache
directory. Both commands used independent repository-local ignored
`--basetemp` directories and exited zero.

## Observed facts

- The frozen experiment is
  [`sanmill-preserving-retained-long-v3.md`](../experiments/sanmill-preserving-retained-long-v3.md).
  It selects fresh seed 58, `malom-preserving-only`, no generic downgrade
  penalty and a fixed policy-auxiliary coefficient of zero. It otherwise
  retains the v2 optimizer, fixed-node curriculum, opponent mix, segmentation,
  strict referee and policy-health gate.
- Neither the retained-v3 control directory nor its writable SpecialistDB
  exists. No historical checkpoint or database has been selected for reuse.
- The closed SpecialistDB template has SHA-256
  `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`.
  Read-only immutable SQLite inspection returned `quick_check=ok`,
  `malom_label_version=sector-corrected-v1`, zero positions, zero winning
  lines and zero preferred plays. WAL, SHM and rollback-journal sidecars are
  absent.
- The active HumanDB main-file SHA-256 remains
  `d8e22da38273f7c26eb76803ae91fc3fae711f508383ffbe3096c2946912b440`.
  Its run-manifest identity remains
  `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`.
- The corrected Malom manifest SHA-256 remains
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`.
- The isolated Sanmill checkout is detached at
  `a6623f88959f7453594df274fbe1f128af7ff55e`, with tree
  `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`. Its 5,641,216-byte release
  binary has SHA-256
  `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`.
- A fresh fetch found no commit after the previously reviewed
  `origin/main` tip. There is therefore no new maintainer patch to merge or
  cherry-pick before this baseline.

## Hypotheses

1. Replacing unconditional Mill reward with the value-preserving-only rule
   removes a known contradictory signal without introducing the unsupported
   strength claims of the rejected penalty and auxiliary variants.
2. Holding every other retained-v2 choice fixed is the lowest-risk way to
   obtain a clean successor research baseline.

## Supporting evidence

- The no-update production-component probe removed all `+4.0` of immediate
  bonus from the sixteen audited value-downgrading Mill turns while retaining
  reward for value-preserving Mill formation.
- The three-seed learning ablation remained numerically and operationally
  safe and favoured the corrected mode in two of three pairs.
- The direct downgrade penalty and both policy-auxiliary designs failed their
  preregistered effect gates. The final target-response audit explicitly
  returned `stop_gradient_ratio_escalation`.
- The current source passes the exact manager, trainer, checkpoint,
  exact-resume, Sanmill referee, reward-mode and fixed-state health paths used
  by the proposed run.

## Counterevidence and limits

- The preserving-only learning ablation was inconclusive: its median effect
  was 0.396 percentage points against a frozen five-point gate. This run is a
  semantic correction baseline, not evidence that strength will improve.
- The successor uses a new seed, so it is not a strict one-factor causal
  comparison with retained-v2 seed 42.
- The prior 64-start held-out corpus and 29-state policy-health corpus have
  been used during diagnosis and selection. Neither is independent validation
  for v3 promotion.
- No new complete repository suite was run. The existing complete-suite
  record still has eight known fail-closed tests tied to unavailable historical
  Sanmill bridge bytes. The trainer also retains its documented legacy Ruff
  findings; all modules changed for the current preparation and the
  launch-critical support modules pass Ruff.
- Training W/D/L, policy-health metrics and curves are diagnostics, not a
  held-out strength or publication result.

## Next validation experiment

After an explicitly authorized ordinary push:

1. require `HEAD == origin/dev` and a clean tracked worktree;
2. copy the exact closed template once to the frozen retained-v3 database
   path, then recheck hash, schema, counts and sidecar absence;
3. generate the ignored managed plan with the command in the experiment
   contract and bind its plan SHA-256;
4. run the exact first-segment read-only preflight and require
   `ready_for_long_run`, `errors=[]`, `unresolved_decisions=[]`, matching
   resume configuration and no competing owner; and
5. request product authorization against that exact readiness identity before
   launching any game.

No retry, extension, evaluation, promotion, publication or long training is
authorized by this source-readiness record.
