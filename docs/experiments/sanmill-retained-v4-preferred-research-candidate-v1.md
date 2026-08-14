# Retained-v4 preferred research candidate decision

Date: 2026-08-14

Status: `preferred_research_candidate_selected_research_only`

Authority: `product-owner-direct`

The product owner accepted the recommendation to nominate the exact
`retained-v4-no-refresh` route as the preferred research candidate after the
completed high-precision held-out comparison. This is a research disposition,
not model promotion, deployment, publication or release.

## Selected candidate identity

| Field | Frozen value |
| --- | --- |
| Candidate ID | `retained-v4-no-refresh` |
| Route | `s-gen-v2-training-aligned-v1` |
| Route-bundle identity | `817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f` |
| Bundle file-list identity | `f701206d2686a958082991034009d3499acc135ed0b044c4b0d1bd1a6010be77` |
| Checkpoint ID | `managed-sanmill-no-refresh-retained-v4-seed70-attempt-003-segment-0020:checkpoint:00000006` |
| Checkpoint file SHA-256 | `295b268e697255908f9c7517f4697ca251a10ec0f13d922cbcbab2260fb5105d` |
| Checkpoint payload SHA-256 | `ed7932bc7c11b1aa41274ea0de7bd08902812b1188ca4739b6d0d8dc15e46727` |
| SpecialistDB SHA-256 | `3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed` |
| SpecialistDB label version | `sector-corrected-v1` |

The decision is bound to the complete route, including its read-only
SpecialistDB and feature contract. It does not select a bare weights file
under another inference route.

## Evidence basis

The decision uses plan
`6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714`,
result
`8d7a4a0aefdd9b0716cccfa3a8d9ace44493c870cc9c9eed885bc7fd35c74730`
and completion
`8949a8fdc0c38772b40e348d7e645ec70aed3cd76663d426320154b0e708ac7c`.
Across 253 independent frozen starts, retained-v4 exceeded retained-v3 by
`+1.6798pp`, with a 95% start-clustered engineering interval of
`[+0.6195pp, +2.7402pp]` and a `1.0604pp` half-width. All 1,012 games reached
strict rules terminals and none hit the safety cap.

The result supports choosing between these two named research routes. It does
not identify target refresh as the cause, estimate population Elo, or prove
that retained-v4 is ready to replace a deployed model. Its absolute score rate
against the pinned 500,000-node Sanmill endpoint was `42.7866%`.

## Operational meaning

This decision permits future source-only planning records to:

- name the exact retained-v4 route above as the preferred research candidate;
- keep retained-v3 as a frozen comparator rather than an equally preferred
  successor; and
- show the research-only disposition in the local evaluation dashboard.

This decision does **not** authorize:

- copying, renaming or replacing a canonical `best.pt`, active model or model
  registry entry;
- using the checkpoint to resume, fine-tune or initialize another training
  lineage;
- another evaluation game, use of the unconsumed source-pool suffix, or a new
  opponent match;
- promotion, deployment, publication, release or external distribution; or
- refresh-causal, equivalence, universal-strength or Elo claims.

Any future plan that uses the candidate must bind the exact route and input
identities above. A different database, feature route, checkpoint, precision,
component set or runtime is a different candidate.

## Next gate

No additional v3/v4 route-selection match is recommended: that relative
question is answered. Before any promotion proposal, the product owner must
define the intended deployment slot and an absolute acceptance contract. That
contract should freeze a genuinely unconsumed, deployment-representative
corpus; colour pairing; strict rules; primary score or non-inferiority floor;
latency and resource ceilings; compatibility and failure gates; claim scope;
and a bounded game/time envelope.

The remaining 108 records in the already frozen source pool are still
unevaluated, but they belong to the same source family and are not automatically
converted into a promotion gate after observing the first-prefix result.

Readiness verdict for the nomination itself: complete.

Readiness verdict for any promotion, deployment, new evaluation or training:
`needs_decision`.
