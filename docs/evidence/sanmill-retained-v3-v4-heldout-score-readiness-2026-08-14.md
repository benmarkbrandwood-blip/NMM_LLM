# Retained-v3/v4 high-precision held-out score readiness

Date: 2026-08-14

Verdict: `needs_decision`

The 253-start / 1,012-game / target-1.5pp held-out score plan is technically
ready for an exact plan-bound product authorization. That authorization is
absent. No authorization file, runtime specification, launch record, game
ledger, progress report, result, completion or failure record exists, and no
candidate game has been played under this plan.

## Frozen identities

| Item | Identity |
| --- | --- |
| Plan | `6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714` |
| Plan file SHA-256 | `7e523265bc8d0b6ae5d861919033a86af2bbdb9238d54991f57d57d8161a0f24` |
| Implementation commit | `5eb142383f710c17377deedc8b1cfcc5287daa02` |
| Plan commit | `592950062515f697b52c8e8d355aa091da4cc839` |
| Held-out source pool | `2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7` |
| Pool records | `4e5f9ecf7508a995b74af6a36bcf966c89d9141940770ebb21c3629446830a31` |
| Frozen 253-record prefix | `99951a691c106a86aa5e4affc16ced2b63866e2dd589379527d068b022003c7b` |
| Successor input snapshot | `1ceb11ce5cec1ff44a9c1f03d69a961f191bdd951d7fe31ef078bb40bf2874c3` |
| Stable source readiness | `f233c991aa66a8699fac8952fd0c758a5fabb09de7a0e66ba3043635934b2b08` |

The complete no-authorization readiness identity observed in both preflights
was `08ad581a0b753c63ea5bde115a994dde6d25b64148f08ede482c6b40236912e0`.
The source-readiness identity deliberately excludes later named status-only
documentation, but fails closed on plan or runtime-code changes.

## Product decision bound by the plan

- 253 independent starts, with 99 placement, 98 movement and 56 flying;
- two candidate colours and two candidates per start, totalling 1,012 games;
- one start is the independent unit after averaging its two colour-specific
  v4-minus-v3 score differences;
- paired game score is the primary endpoint, with win/draw/loss scored
  1/0.5/0;
- target 95% engineering half-width is at most 1.5 percentage points; and
- four active evaluator hours is a safety ceiling, not a target duration.

This fixed-width design estimates a named-route score difference. It does not
predeclare an equivalence margin. A zero-crossing interval is inconclusive,
not evidence that the candidates are equal. Process metrics remain secondary
and cannot replace score after outcomes are observed.

## Repeated technical preflight

The command

```powershell
.\.venv\Scripts\python.exe scripts\run_retained_heldout_score.py preflight
```

was run twice after the implementation and immutable plan commits were
published to clean `dev`. Both runs returned CLI exit code 2 only because the
separate product-authorization gate was absent. They produced identical source
and full readiness identities and the same observations:

- repository and plan binding passed;
- the output namespace contained only the frozen read-only input directory;
- successor-owned route bundles and sidecar-free SpecialistDB copies matched;
- the 253-record prefix and adjacent 1,012-game schedule matched the plan;
- both exact candidate routes and checkpoints passed CPU verification;
- the pinned strict Sanmill runtime and deterministic node canary passed;
- all 253 complete histories replayed strictly as nonterminal without loading
  a candidate or selecting a move;
- no competing trainer or evaluator owned the namespace;
- focused evaluator, runner, ledger, input, route and web tests passed; and
- mandatory Malom, DB-teacher and label-provenance tests plus Ruff passed.

The HumanDB warning that unversioned historical Malom columns are masked is
expected. Corrected Malom data remains bound through the separate
`sector-corrected-v1` manifest. Candidate moves requested during corpus
verification: zero. Candidate games played: zero.

## Verification counts

The exact focused preflight test selection completed with `80 passed`. The
mandatory provenance selection completed with `103 passed, 498 subtests
passed`. An earlier combined implementation verification completed with `188
passed, 498 subtests passed`; Ruff and `git diff --check` were clean before the
immutable plan was frozen.

## Authorization boundary

A later launch decision must name both:

- plan identity
  `6620821e879f53058d15990cd0e8c884ae62fec213b3d96200e8894c20e19714`;
  and
- source-readiness identity
  `f233c991aa66a8699fac8952fd0c758a5fabb09de7a0e66ba3043635934b2b08`.

Only a grant within the plan's 1,012-game, four-active-hour and node ceilings
may launch. A host interruption may use same-spec missing-suffix exact resume
only if the authorization says so. Automatic retry, semantic-failure recovery,
expansion, training or update, equivalence/Elo/population/refresh-causal
claims, promotion, publication and release remain prohibited.
