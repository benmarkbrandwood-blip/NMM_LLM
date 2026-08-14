# F0-H0 v1 split-constraint correction -- 15 August 2026

Status: `superseded_by_corrected_split_design`

This record supersedes only the split-stop conclusion made by the F0-H0 v1
screen.  It does not delete, rewrite, or reinterpret the frozen v1 plan,
membership, result manifest, or narrative evidence.  Their bytes and
identities remain historical evidence of the procedure that was actually
run.

## Exact defect

The v1 rule combined both of the following requirements:

1. every game is retained whole in exactly one partition; and
2. every game involving a player is assigned to that player's one partition.

Let `p(v)` be the partition assigned to player vertex `v`.  For every game
edge `(u, v)`, retaining the whole game while satisfying both players'
assignments requires `p(u) = p(v)`.  Equality propagates along every path.
Consequently, all vertices and all games in each connected component must
collapse into one partition.

The v1 prerequisite therefore tested whether the player graph admitted a
non-empty **zero-cut** four-way split.  It did not measure whether useful
player holdouts remain after a controlled number of cross-boundary games are
discarded.  A connected graph necessarily fails that rule regardless of its
degree distribution, community structure, calendar structure, or the number
of games that would survive a nonzero cut.  That algebraic consequence is not
a corpus-specific feasibility finding.

## Frozen v1 artefacts preserved

The corrected measurement independently rechecked these unchanged files:

| Role | Frozen identity | File SHA-256 |
| --- | --- | --- |
| v1 plan | `95a802625867906ab453ed7a52bbba1e0202b08473b10f897ba81c87fb59d530` | `3ec32220b220f019b3a60c8a2e1519eae9933933a28b5ccf16072b06b78e2136` |
| v1 membership | `e41da5fbf1a2ba60441273664c6834dafcb54bcb79d541f3349a948f7cac5dd4` | `1ef901a6776bab15b96fa4ab25273223ae2028568f619fb35f6ecd96094b26c4` |
| v1 result | `714627f8be20bc45a267c97752171644040fc1273a24f82a570a7cb83512fe82` | `84226cb96e1e7775a896220b3b9cee84b48f3f0562fb68c81ad6bdf28473692e` |

The original narrative remains at
[F0-H0 v1 evidence](f0-h0-human-feasibility-screen-2026-08-14.md).  It is no
longer the governing split-feasibility conclusion.  Its recorded zero-cut
execution is not being relabelled as a corrected experiment.

## Replacement scope

The replacement is limited to split-feasibility measurement.  The corrected
plan freezes three separate candidate designs:

- a player cut that discards cross-boundary games;
- calendar holdouts with time-only and time-plus-unseen-player counts; and
- player-owned decisions with explicit trajectory and `ring16` leakage.

The corrected result is documented in
[the split-feasibility evidence](f0-h0-corrected-split-feasibility-2026-08-15.md)
and its
[machine manifest](f0-h0-corrected-split-feasibility-manifest-2026-08-15.json).
It selects no design and makes no F0-H0 feasibility, continuation, or stop
decision.  The four scientific F0-H0 dimensions remain unrun.

F0-D0 corpus and manifest identities are unchanged.  This correction grants
no authority for Malom queries, E0, F0-H1, T0-H-pilot, games, search, model
loading, training, database work, source-pool access, promotion, deployment,
publication, or release.
