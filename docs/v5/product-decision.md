# v5 Product Decision

Status: governing initial product scope; deployment and resource constraints
remain open.

## Product Role

The first v5 product is an additional top-strength/research opponent. It does
not replace the current ten difficulty levels, personality presets, adaptive
difficulty, tournament opponents, or LLM coaching surfaces.

Those existing modes may later reuse a validated v5 component only under their
own product contract. A deliberately weakened, instructional, stylistic, or
adaptive opponent is not required to maximise the same utility as the
top-strength opponent and must not inherit a verified claim merely because it
shares code or weights.

The initial v5 objective is therefore:

1. preserve the exact legality, evidence, and safety boundary claimed by the
   selected top-strength product mode;
2. improve terminal match score against the frozen target population while
   satisfying loss-rate and applicable runtime non-inferiority constraints;
3. measure style, naturalness, explanation quality, and repeat-play experience
   without letting them overrule the first two items.

This is not a decision that one neural model must control every existing
opponent mode.

## Product Personas

| Persona | Initial v5 status | Primary product objective |
| --- | --- | --- |
| Top-strength/research opponent | In scope | Match result, correctness, runtime reliability, and precisely bounded claims |
| Verified-theory opponent | Research lane only until recursive viability and target-device availability close | The exact whole-game property and support domain proved by its invariant |
| Coach/teaching opponent | Existing product retained; v5 integration deferred | Instruction quality, calibrated challenge, and explanation fidelity |
| Personality/relaxed opponent | Existing product retained; v5 integration deferred | Style, variety, calibrated difficulty, and repeat-play experience |
| Adaptive-difficulty opponent | Existing product retained; v5 integration deferred | Player-level challenge calibration and retention, including deliberate weakening where declared |
| Tournament roster | Existing product retained; v5 integration deferred | Frozen opponent identity, diversity, and separately calibrated ratings |

## Current Host Boundary

The current Web/FastAPI application on the Windows development machine is the
first integration and measurement host. That fact does not freeze a release
device, operating-system floor, local-versus-server architecture, or hardware
budget.

Before F0-A1 can select a release architecture, the product owner must freeze:

- whether the top-strength opponent must work fully offline;
- supported product host and minimum/representative devices;
- maximum installed/download size and startup memory;
- acceptable visible move latency and any background compute;
- whether a separately versioned server Oracle is permitted;
- engineering-time, CPU/GPU-hour, hosting, and maintenance ceilings; and
- redistribution constraints for tablebase-derived artifacts.

Until those values are recorded, F0-A0 may inventory and measure alternatives
but F0-A1 cannot declare a final architecture or make a numerical budget
governing.

## Relationship to Existing Features

- Existing difficulty, personality, adaptive, tournament, and coaching
  behaviour must remain unchanged by the first v5 candidate.
- A v5 route is exposed under a separately named top/research mode.
- Ordinary, positional-exact, bounded-survival, theory-preserving, and Oracle
  service forms are separate contracts, not hidden fallbacks for one another.
- A later decision to replace an existing opponent or share a v5 component
  requires a new persona-specific objective, route test, and claims review.

## Decision Boundary

This document freezes the role of v5 but not the final deployment architecture.
Architecture selection remains blocked until the open host, offline, resource,
cost, and redistribution constraints above are signed. No training,
evaluation, or release is authorised by this product decision.
