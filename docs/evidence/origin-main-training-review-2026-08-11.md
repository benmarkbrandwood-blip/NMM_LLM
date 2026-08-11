# Origin/main training review — 11 August 2026

## Scope and verdict

This review covers the linear `origin/main` range
`728ddad9f1d7f95fae75a351669d39cd117b1259..0cfb651424d089908988f48129fe3ab3de5b010e`.
It was performed before refreshing the unlaunched schedule-isolation v2
contract. No merge, cherry-pick, or history rewrite was used.

The range contains seven commits:

| Commit | Subject | Disposition on `dev` |
| --- | --- | --- |
| `6d61d401ba55e3b98e3df4b2e24a2965fa51eb32` | GapNet v3 session ledger | Do not adopt or consume; production blockers remain |
| `0e0224bc540ee8e29a1bb336f80ba1cf53dac18b` | Reject invalid GapNet target values | Correct, but irrelevant while GapNet is explicitly disabled |
| `2a152af50724ffb9d9ae9d50235e30fbe49328a2` | Reconcile the GapNet v3 plan | Useful override notice, but not a production-readiness result |
| `4853296432f6093042c6cdf6f86ceaa9d3c1f5f3` | Read the phase guide as UTF-8 | Correct portability fix; independently implemented and tested on `dev` |
| `ec567b29ebccbf4fc9182beb08e23280255cb099` | Add an HMPN session-ledger extraction mode | Do not adopt or run; the recorded ledger is not the source of the applied split |
| `9efe0ba91d34266aff0684d0ddcc2e57ec336a74` | Add an HMPN trainer ledger guard | Useful overwrite warning, but it inherits the unbound dataset identity |
| `0cfb651424d089908988f48129fe3ab3de5b010e` | Mark GapNet Batch 3a/3b progress | Do not treat the completion claims as readiness evidence |

The verdict for the current schedule-isolation experiment is **no relevant
training implementation to import**. GapNet and the human move policy teacher
remain disabled by contract. The UTF-8 change is an independent general
portability repair and does not alter the experiment's gameplay, optimizer,
data, or schedule semantics.

The verdict for the `main` GapNet v3 production data chain remains **fatal
stop**: do not generate or consume a canonical production ledger or HMPN v3
dataset, and do not start the proposed teacher or GapNet v3 retained runs from
this implementation.

## Observed facts

### Session-ledger implementation

Direct review and isolated reproductions found these blockers:

- the governing checklist originally froze a GapNet-specific split
  seed/version, but the builder calls the older `game_level_split()` contract;
- `--limit-files` can overwrite the canonical output without an incomplete
  marker, so a smoke ledger can be mistaken for a complete production ledger;
- malformed JSONL records and an empty production directory succeed instead
  of failing closed with file and line evidence;
- source files are hashed and then reopened for parsing, allowing recorded
  identity bytes and consumed bytes to diverge during a long scan;
- output replacement is unconditional and non-atomic, with no default
  no-clobber policy or complete source/script provenance;
- the direct-root-only corpus boundary is not frozen even though nested
  held-out files exist; and
- the file-stem fallback counter increments before deduplication and can
  exceed the number of fallback-derived sessions.

### HMPN extraction path

Commit `ec567b2` accepts both `--session-ledger` and `--session-index`, but the
two inputs are not cryptographically or semantically connected:

- the ledger is parsed only to print and copy its identity fields;
- the actual `state_key -> split mask` mapping comes entirely from the
  independently produced session index;
- `_load_state_key_masks()` receives no ledger, and verifies only that the
  session index names the current HMPN dataset `metadata.npz` SHA-256; and
- neither the session index nor its builder records the ledger SHA-256, ledger
  files-manifest SHA-256, ledger completeness, or the ledger session/split
  mapping.

The existing `tools/build_session_index.py` independently rescans direct-root
`*.jsonl` files, recomputes `game_level_split(session_id)`, permits
`--limit-files`, silently skips malformed JSON and invalid move states, and
uses a file-stem fallback. It does not accept a ledger input. Consequently an
old, partial, or differently derived session index can be paired with an
unrelated ledger while the extractor records both hashes in apparently
authoritative provenance.

The strict-single-tier mask rule itself is conservative: it keeps only masks
`0b001`, `0b010`, and `0b100`, and drops mixed or uncovered state keys. That
property does not prove that the masks came from the recorded ledger.

The extractor also creates its output directory before completing input
validation and publishes its arrays and metadata directly rather than through
an atomic, no-clobber transaction. A failed production extraction can
therefore leave an output path that is not a complete immutable dataset.

### HMPN trainer guard

Commit `9efe0ba` prevents one exact session-ledger dataset configuration from
overwriting the historical v2 teacher filename and surfaces copied ledger
fields in model provenance. It does not validate that those fields are
complete or that the session index was derived from that ledger. The guard
also deliberately permits arbitrary custom output paths, including existing
ones, so the checklist statement that it “enforces” the proposed v3 filename
is stronger than the code.

This guard is useful defence in depth after a trustworthy dataset exists. It
does not make the current dataset chain trustworthy.

### Focused tests

An isolated archive of `origin/main` at `0cfb651` passed the nine focused
GapNet/ledger/HMPN files, **62/62**, using the repository environment and a
snapshot-local pytest base directory. These tests exercise deterministic
ledger output, strict mask disposition, the metadata/index SHA check, trainer
path guards, and the numeric target loader.

They do not test a mismatched ledger/index pair, a partial index presented as
complete, ledger-derived split equivalence, atomic publication, or fail-closed
production ingestion. Their green result supports the implemented helpers but
does not close the production blockers.

### Target-value loader

Commit `0e0224b` correctly rejects NaN and infinity in primary and uniform
targets, permits only the documented NaN support sentinel in empirical
targets, and rejects empirical infinity. This closes the reviewed numeric
loader defect. It neither repairs the data identity chain nor affects the
current Generalist experiment, where GapNet is disabled.

### GapNet plan reconciliation and progress document

Commit `2a152af` adds an authoritative correction table and marks the old
architecture and Stage D/E gate rows as superseded. Commit `0cfb651` then marks
Batch 3a as landed and Batch 3b tooling as done. The latter document claims
that ledger consumers verify the files-manifest hash and that both extractors
consume the ledger as their source of truth. The reviewed HMPN extractor does
not establish either claim. Progress labels are planning metadata, not
acceptance evidence.

### Phase-strategy encoding

Commit `4853296` changes an optional Markdown read from the Windows locale
default to explicit UTF-8. The same latent call existed on `dev`, although the
optional guide itself is not tracked there. Dev commit `2e89365` implements
the one-line repair independently and adds an import-time regression that
forces the optional guide path, verifies the explicit encoding, and parses
non-ASCII text. The focused regression passes. This is not a cherry-pick and
does not import `main` gameplay or training code.

## Hypotheses

1. A production ledger or HMPN dataset built by the reviewed chain can bind a
   different or incomplete corpus from the one its metadata appears to
   describe.
2. A teacher trained from that dataset could report session-isolated
   train/validation/test provenance even when its masks were generated by an
   unrelated or partial index.
3. The seven reviewed commits have no causal bearing on the target-refresh
   schedule-isolation contrast because that experiment disables GapNet,
   imitation, Sentinel, ValueNet, and the HMPN teacher path.

## Supporting evidence

- The ledger and session-index builders have separate scans, inputs,
  completeness behavior, and provenance, while the extractor performs no
  cross-identity check.
- The session-index builder has no ledger parameter; the extractor's mask
  loader has no ledger parameter; the new synthetic tests construct no ledger
  when validating masks.
- Isolated reproductions showed canonical partial-output replacement,
  malformed-input acceptance, empty-input acceptance, and the fallback-count
  mismatch in the ledger builder.
- The schedule-isolation machine contract explicitly disables GapNet,
  Sentinel, ValueNet, imitation, and warm start.
- No reviewed `main` commit implements the exact-transition,
  transition-indexed-temperature, fixed-node, common-anchor outcome design.

## Counterevidence and limits

- The ledger has useful deterministic ordering and file identity fields. The
  extractor's strict mask rule and metadata/index SHA check are also useful.
  The verdict concerns the missing end-to-end identity chain, not those local
  mechanisms.
- The 62 passing focused tests show the intended helper paths work. They do
  not test the production failure modes above.
- No HMPN v3 production dataset or candidate model was generated during this
  review, so this document does not claim measured leakage or poor model
  quality. It establishes that the proposed provenance cannot yet rule those
  failures out.
- The UTF-8 regression proves the read contract, not behavior of an untracked
  phase guide or any LLM-assisted playing-strength effect.

## Next validation work

For GapNet/HMPN v3, make the frozen ledger the actual split authority. Either
derive the state-key masks directly from ledger session assignments or bind a
session index to the exact ledger SHA-256, files-manifest SHA-256, complete
source boundary, split function/version, and session mapping, then verify all
of them in the extractor. Fail closed on malformed, empty, partial, or drifting
input; distinguish smoke artifacts from production artifacts; and publish
atomically with no-clobber semantics. Only after those gates pass should the
project report per-split state/event/band/phase counts or request a teacher
run.

This work is owned by `main` and is not a prerequisite for the disabled-GapNet
Generalist diagnostic.

For `dev`, update only the schedule-isolation lineage to this reviewed main
tip, keep `cherry_picks_selected` empty, bind this evidence by SHA-256, and
regenerate all ignored readiness and database artifacts from the next clean,
published `dev` commit. No candidate, training result, or strength claim was
used in this source review.
