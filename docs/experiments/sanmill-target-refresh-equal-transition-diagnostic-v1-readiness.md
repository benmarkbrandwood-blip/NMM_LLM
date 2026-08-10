# Equal-transition target-refresh diagnostic readiness

Status: `implementation_complete_unlaunched_needs_publication`.

The machine-readable source of truth is
[`sanmill-target-refresh-equal-transition-diagnostic-v1.json`](sanmill-target-refresh-equal-transition-diagnostic-v1.json).
Neither document authorizes training.

## Staged preparation

The sequence has three shared prefixes and six treatment arms. It cannot
truthfully create all nine preflights at once. Each arm must exact-resume from
an actual game-50 fork checkpoint and from a byte-identical clone of the
closed SpecialistDB produced by that same prefix. Those two artefacts do not
exist until the prefix has run.

The first preparation stage may therefore create only three fresh prefix
plans, three isolated template-derived SpecialistDB files, and three
read-only preflight reports. The implemented later arm-preparation stage
audits one completed prefix at a time, clones its closed database separately
for both arms, publishes two descriptor-rebound envelopes without changing
the source payload bytes, and generates two new plans and preflights. It also
sets `allow_safe_exact_resume=false`: the initial external fork resume is
permitted, but no continuation, retry, or second segment is authorized.
Placeholder checkpoints or databases are forbidden.

## Frozen sequence

For each of seeds 64, 65 and 66, the prefix ends at game 50 immediately before
the scheduled target refresh. The paired arms then start from the same fork:

- `refresh-once` copies the candidate into the frozen target and resets its
  age once;
- `no-refresh` preserves the target and age;
- both suppress every later automatic target refresh;
- both consume ordered 64-transition batches and stop at exactly 8,192
  post-fork consumed transitions;
- both preserve incomplete pending transitions without a final flush.

Read-only comparisons use the existing 64-position phase corpus at 1,024,
2,048, 4,096 and 8,192 transitions. No outcome games, held-out evaluation,
promotion, publication or long-run launch are part of this diagnostic.

## Resource boundary

The later complete sequence is capped at 3,450 actually executed games and
six active wall hours. These are safety ceilings, not successful-completion
criteria. Any incomplete scientific boundary, identity drift, non-finite
state, database mutation, Sanmill error or checkpoint failure stops the whole
sequence without automatic retry or extension.
