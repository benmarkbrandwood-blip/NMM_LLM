# GapNet v3 — design plan (revision 1)

*Compiled 2026-07-30.  Read-only planning artefact.  This document* *does not modify code, activate any database, retrain any model,* *change gameplay, or connect the proposed signal to gen 2b or gen 3.* *Every downstream action is gated by the promotion criteria at the* *end of the plan.*

The plan supersedes `Downloads/gap\_net\_v3\_pre\_plan.md` as authoritative; the pre-plan is treated as non-authoritative research notes.  Where the pre-plan and the current code disagreed, the code wins and the discrepancy is recorded in §3.  
  
I also found several internal conflicts in the current GapNet plan:  
  
- G\_v is defined separately by Elo band, but the proposed 79-feature GapNet has no band input. The same position can therefore have three different targets for one identical input, forcing the network to average them. We need either band conditioning, separate per-band models, or one explicitly frozen population mixture.  
- Component D is defined only when all legal moves remain in the same W/D/L class, while the global fail-closed rule rejects a position if any component is unavailable. That would remove the mixed-tier W→D, W→L and D→L positions that are the most important examples.  
- Component D also subtracts distance values across successor positions, which conflicts with the plan’s own prohibition on cross-sector DTW/DTM subtraction. I think Component D should be deferred entirely.  
- Equal OracleMoveValue alternatives must receive equal zero rank regret. A deterministic notation tie-break may order serialization, but it must not create artificial regret.  
- A game-level split has to occur before human move counts are aggregated. The present v3 database stores aggregated state/move/Elo counts and no session or player identity, so assigning aggregated states to games afterwards cannot prevent leakage and cannot support the proposed player-held-out test.  
- Objective Oracle regret is symmetry-invariant, but human move probability may contain genuine orientation, colour or UI biases. D4 invariance should therefore be an ablation, not a mandatory promotion gate for G\_v.  
- The plan says the current GapNet leaf default is ai\_side, but the code at 5c9ca45 actually defaults to opp\_side. Because human\_correction adds a positive value from the side-to-move perspective and negamax then changes sign, the current route may avoid rather than exploit opponent blunder zones. A hand-computed two-ply sign regression is needed before reusing this path.  
- Terminal child values should remain exact, but expectimax should still average those exact child values under P\_h. One terminal continuation should not automatically make the whole node equal pure minimax.  
  
I therefore agree with the thin-slice recommendation, with one adjustment: I would keep W→D, W→L and D→L as three separate outputs rather than collapsing every downgrade into one binary component.


## 1. Purpose and non-goals

**Purpose.**  Specify how to build a signal that predicts, for each position and each opposing-player profile, **how much objective quality** **that player is expected to concede through their next move**, and document the gates that must close before this signal is (a) trusted by any search algorithm at play time or (b) used as an input feature, auxiliary target, or bounded reward term in gen 3 training.

**Non-goals.**  This plan does **not**:

- Retrain ValueNet, HumanPrefNet, or Sentinel.

- Rebuild or modify the current active `data/human\_db.sqlite`.

- Activate `data/human\_db\_candidate.sqlite` over the active DB.

- Overwrite or replace `data/gap\_net.npz` (the artefact `web/app.py` loads today; verified `web/app.py:256-262`).

- Wire any v3 signal into `ai/game\_ai.py`, `ai/heuristics.py`, `scripts/train\_s\_gen\_v2b.py`, or a future gen 3 trainer.

- Choose a topology-aware board encoder; §M2 of the archived discussion plan explicitly defers this to a fresh branch.


## 2. Five separate concepts — locked definitions

The pre-plan blurred several concepts under the single word "regret". The plan requires them to remain formally distinct.  Every downstream artefact must state which of these five it produces.

### 2.1 Human move policy — `P\_h`

> `P\_h(move | state, band-or-profile) ∈ \[0, 1\]`, summing to 1 over every legal move at `state`.

Produced today by **HumanMovePolicyNet** (`ai/human\_move\_policy\_advisor.py`, 82-input MLP, softmax over every legal move, count-weighted CE loss per `docs/human\_move\_policy\_net\_plan.md`).  Also derivable directly from `data/human\_db\_candidate.sqlite:moves\_elo\_bins` where support is sufficient (see §7.1).

- Elo band today is `lower / middle / upper` per `learned\_ai/data/elo\_binning.py`.

- A `PlayerProfile`-conditioned variant (`ai/player\_profile.py`) is proposed as a later ablation, per the Toronto blunder-prediction finding that position-complexity dominates player-skill features — but v3 ships with band-only conditioning to keep the input shape and provenance chain honest.

### 2.2 Objective move regret — `R\_v`

> `R\_v(state, move) ∈ ObjectiveQuality`, derived only from `malom\_label\_version = "sector-corrected-v1"` Malom values via the complete `OracleValue` / `OracleMoveValue` ordering, evaluated from the **original mover's perspective**, with rules-terminal successors resolved.

`R\_v` is a **direct offline computation**, not a learned model.  It is undefined when either the parent or the successor is not covered by Malom.  It is versioned.  §5 specifies it.

Key discipline (reviewer §11 in `human\_move\_policy\_net\_plan.md`; §G1 in `docs/archive/discussion\_plan.md`):

- **Never** compute `R\_v` from raw `key2` subtraction.  Malom's `key2` is context-dependent (`ai/malom\_db.py:598-604` — the ordering flips at `key1 = 0` and reverses on the `key1 \> 0` branch).

- **Never** compute `R\_v` from `dtw` subtraction across positions with different sectors.

- **Never** silently substitute `R\_v = 0` for a missing lookup.  Return an explicit `unavailable`.

### 2.3 Expected human regret — `G\_v`

> `G\_v(state, band-or-profile) = Σ\_\{m ∈ legal(state)\} P\_h(m | state, band) · R\_v(state, m)`

`G\_v` is the composite target the GapNet v3 candidate model learns to produce.  It is defined **only where every `R\_v(state, m)` for every** **legal `m` at `state` is available** (see §5.5 fail-closed rule).  It inherits the unit and ordering of `R\_v` — no new scalar mapping is introduced by the composition.

**`G\_v` is not a probability.**  It is a scalar in the objective quality space (unit chosen in §5.3) or an ordered tuple of separately tracked components.

### 2.4 Gameplay exploitation

How a search algorithm at play time may consume `P\_h`, `R\_v`, or `G\_v` without corrupting negamax signs, terminal-value dominance, or worst-case safety.  §9 lists the four candidate consumption modes (pure minimax baseline, human-policy expectimax at opponent nodes, bounded mixture, shadow mode).  This is **separate from the signal** **definition** — the same `G\_v` can be consumed by any of the four modes and the plan does not require choosing one at signal-training time.

### 2.5 Training consumption

Whether and how gen 3 (or a later cycle) uses `P\_h` / `R\_v` / `G\_v` during training.  Four consumption modes, in the recommended order of increasing risk (§10):

- Logged diagnostic (write to `train\_log.jsonl`, do not touch loss).

- Auxiliary prediction head (learner predicts `G\_v`, does not use it).

- Input feature to policy / value network (learner conditions on it).

- Bounded reward-shaping term (learner is rewarded for reducing it).

Direct reward shaping is **not authorised** by this plan; it becomes a separately-authorised experiment after logged-feature and auxiliary-head experiments have shown that the signal moves the right way and does not induce reward hacking (§10.4).


## 3. Repository state — verified claims vs pre-plan

Every row below was verified against the current commit (`HEAD = 5c9ca45` at planning time, post-rename).  "V" = verified as claimed by the pre-plan.  "DISCREPANCY" = correction to the pre-plan.

### 3.1 Verified

| Pre-plan claim | Verification |
| - | - |
| HumanPrefNet has top-1 = 48.35 %, Spearman r = 0.191 | `data/eval\_human\_pref\_net\_result.json`; §6d of `retrain\_v2\_plan.md` |
| HumanMovePolicyNet code + tests landed (23 tests green) | Commits `397d828`, `16571a6`, `780d3a2` (rename `5c9ca45`); Phase 3 marked COMPLETE in `human\_move\_policy\_net\_plan.md:255-278` |
| HumanMovePolicyNet has NOT been trained at scale yet | `data/human\_move\_policy\_dataset/` exists (metadata + memmap); `data/human\_move\_policy\_net.npz` does NOT exist on disk |
| `data/human\_db\_candidate.sqlite` (v3) exists, not activated | `docs/DATABASES.md` §5; validator report `data/human\_db\_candidate.sqlite.validation.json` reports `ok=True` at candidate SHA `df71395f…` |
| Current GapNet is loaded live by `web/app.py` | `web/app.py:256-262` loads `data/gap\_net.npz` and passes to every `GameAI` construction |
| Current GapNet is applied only at leaf nodes (depth==0) | `ai/game\_ai.py:1953-1997` — `\_negamax` depth-zero path applies `human\_correction()` conditionally on `gap\_net\_leaf\_mode`; verified default is `"ai\_side"` (`ai/game\_ai.py:492`) |
| `human\_correction()` is additive, capped by phase | `ai/heuristics.py:3609-3652`; `\_GAP\_SCALE = 3000`; phase-cap map at line 3632 uses `gap\_blend\_place/move/fly` (12/20/5 %); returns `e\_v2 + bonus` |
| Malom `OracleMoveValue` ordering is context-dependent | `ai/malom\_db.py:580-604`: `compare\_oracle\_move\_values` uses key1 primary, key2 secondary with flip rules at key1==0 and reverse on key1\>0 |
| Malom `move\_value()` handles child probes with sector correction | `ai/malom\_db.py:966-981`; `undo\_negate\_oracle\_value` at 581-644 |
| §G1 target-semantics blocker on current GapNet | `docs/archive/discussion\_plan.md:272-285` and `scripts/build\_gap\_dataset.py:11-31` — dataset uses `SENTINEL\_WEIGHT × sentinel\_q + (1-SENTINEL\_WEIGHT) × heuristic\_q\_norm` composite, NOT `malom\_top\_q − malom\_q\_of\_hp\_top` |


### 3.2 Discrepancies

**DISCREPANCY-1.**  Pre-plan §2 row "Current GapNet models opportunity gap = Malom best-move quality minus weighted quality of human moves actually played".  The current `scripts/build\_gap\_dataset.py` (`SENTINEL\_WEIGHT = 0.6`, lines 71, 203) computes gap from a **sentinel + heuristic composite**, not from Malom values at all. The `y\_hp` auxiliary label (line 407, `\# per-plan: |malom\_optimal\_q - malom\_q\_of\_hp\_top|`) is stored **separately** and only mixed in at training time when `tools/train\_gap\_net.py --hp-blend \> 0` is passed (default 0.0, line 42-43).  The dataset target and the `y\_hp` label are two different things.

- **Owning document to fix:** `docs/archive/discussion\_plan.md` §G1 already flags this — the pre-plan's summary of that concern is imprecise.  This plan is now the authoritative statement.

**DISCREPANCY-2.**  Pre-plan §2 row "GapNet ... wired into `ai/game\_ai.py`, `ai/heuristics.py` (`gap\_blend\_place`, `gap\_blend\_move` caps)" implies HumanPrefNet participates in the leaf correction.  It does not — `human\_correction()`'s signature (`ai/heuristics.py:3609-3617`) takes `gap\_net` only.  HumanPrefNet is used **elsewhere** in `ai/game\_ai.py` at lines 1643-1666 and 2646, for the `humanlike\_blend` re-ranking of the top candidates *after* the negamax has finished — a different code path.

- **Owning document to fix:** none needed; this plan documents the actual wiring so v3 will not misroute.

**DISCREPANCY-3.**  Pre-plan §3 item 6 asks "is GapNet v2's existing `opportunity gap` signal now redundant with expected-regret?".  There is no `gap\_net\_v2.npz` on disk; the currently-live model is at `data/gap\_net.npz` and was trained on a legacy pipeline.  A "v2" dataset builder exists (`build\_gap\_dataset.py:6-8` mentions v2) but its output has never been promoted per §G1.  So the redundancy question is malformed: v3 is not competing with a shipped v2, it is competing with **an unshipped-and-blocked v2 concept** and the currently-live v1-composite artefact.

- **Owning document to fix:** none — v3 will not overwrite either.

**DISCREPANCY-4.**  Pre-plan §3 item 7 states current GapNet correction is "applied only at the AI's own leaf nodes ... exclusive to AI side".  Verified partially: the default `gap\_net\_leaf\_mode` IS `"ai\_side"` (`ai/game\_ai.py:492`), but the attribute is public and supports `"opp\_side"`, `"both"`, and `"off"` (verified at `ai/game\_ai.py:1974-1983`).  Symmetric application is already possible today; nothing needs to be added to `\_negamax` to try it. The pre-plan's implication that "symmetric application would be a new code path" is wrong.

- **Owning document to fix:** `AI\_INTERNALS.md` (not in scope of this edit — flag for future doc sweep).

**DISCREPANCY-5.**  Pre-plan §2 row "Malom regret — `docs/human\_move\_policy\_net\_plan.md` and the Phase-1 audit both say: raw `key2` subtraction is not a valid regret target".  Verified against `docs/human\_move\_policy\_net\_plan.md:418-441` and `docs/human\_moves\_audit\_phase1.md:343-355`.  Correct.

**DISCREPANCY-6.**  Pre-plan §5 "framing" item 4 says "HumanMovePolicyNet Phase 4 eval must show acceptable calibration ... before GapNet v3 consumes it as ground truth for where humans will falter".  The eval script that was landed (`tools/eval\_human\_move\_policy\_net.py`) reports event NLL, top-1/3/5, ECE, per-band + per-phase + per-transition strata.  It does **NOT** report Brier score, corrected multiclass calibration, uniform / empirical-frequency baselines side-by-side, per-legal-move-count stratum, an independent test partition, OOD / coverage / abstention diagnostics, or immutable hashes (verified: `grep OOD | abstain | coverage tools/eval\_human\_move\_policy\_net.py` returns no matches). This plan's §7.1 lists what has to be added to that script before HumanMovePolicyNet becomes a v3 dependency.

- **Owning document to fix:** `docs/human\_move\_policy\_net\_plan.md` Phase 4 section is optimistic about eval completeness.  A follow-up revision to that doc will note the missing metrics under a new "Phase 4b — additional diagnostics" heading.


## 4. Blockers that must close before v3 proceeds

The plan does not authorise any of the following work until every blocker in this section has landed evidence.

### 4.1 HumanMovePolicyNet Phase 4 gates

Required before `P\_h` is trusted as a v3 dependency (§7):

1. ✅ **DONE (2026-07-30/31)** — Train to convergence on `data/human\_move\_policy\_dataset/` (v1, 22 epochs, val NLL 1.5953, ~11.6 h).  First-run model saved as `data/human\_move\_policy\_net.npz`. A v2 re-extraction + re-train is underway (see item 3 below); the v2 candidate will be `data/human\_move\_policy\_net\_v2\_candidate.npz`.

2. ✅ **DONE (2026-07-31)** — `tools/eval\_human\_move\_policy\_net.py` rewritten (Phase 4b).  Now reports:

   - **Brier score** per band, phase, transition, lmc.

   - **Temperature scaling** (pass-1 logit collection, `scipy` `minimize\_scalar` on val NLL, pass-2 full eval with T\*).

   - **Uniform** baseline in every stratum (NLL, Brier, top-k, ECE).

   - **Empirical-frequency** baseline at `≥ min\_support` positions, side-by-side with the model.

   - **Per-legal-move-count** strata: `lmc\_2-5`, `lmc\_6-10`, `lmc\_11-20`, `lmc\_21+`.

   - **OOD** row: positions not reached by any train-split sample.

   - **Abstention** row: positions where any successor encoding fails.

   - **Game-val-only** diagnostic stratum (requires `--session-index`).

   - **Test set** (`--run-test-set`, single-shot, v2 datasets only). Files: `tools/eval\_human\_move\_policy\_net.py`.

3. ✅ **DONE (2026-07-31)** — Split rework:

   - `learned\_ai/data/human\_db\_split.py` v2 adds `three\_way\_split()` (5 % test / 15 % val / 80 % train) and `game\_level\_split()`. `in\_val\_bucket` unchanged (all other consumers unaffected).

   - `tools/extract\_human\_move\_policy\_dataset.py` v2 emits `sample\_split` int8 array (0=train, 1=val, 2=test) and backward-compat `sample\_is\_val`.

   - `tools/build\_session\_index.py` (new) scans 97 138 JSONL game files → `game\_split\_mask` + `player\_split\_mask` uint8 bitmask arrays.

   - `tools/train\_human\_move\_policy\_net.py` updated: detects v2 dataset, uses `sample\_split` for train/val/test; new default output `data/human\_move\_policy\_net\_v2\_candidate.npz`. **Pending (user will run):** re-extraction (~400 s), re-train (~11 h), session-index build (~variable), full Phase 4b eval → produce `data/gap\_v3\_prerequisite\_eval.json`.

4. ✅ **DONE (2026-07-31)** — Provenance chain already inherited via `.npz`-embedded `provenance\_json` (verified: candidate DB SHA-256, dataset git commit, feature version, elo band config all present in `data/human\_move\_policy\_dataset/provenance.json` and model .npz). The split-manifest version is now "v2" (bumped from "v1").

Owning document: this plan; `docs/human\_move\_policy\_net\_plan.md` §Phase 4b documents the extended eval requirements.  Run `data/gap\_v3\_prerequisite\_eval.json` to close gate §16 "Stage B eval".

### 4.2 Candidate database

Before the v3 dataset extraction reads from `data/human\_db\_candidate.sqlite`:

1. **Do not activate** the candidate DB over the active DB.  A validated candidate may be **read directly** for experimentation without activation (per plan preamble).  The v3 extractor will read from `data/human\_db\_candidate.sqlite` and record its SHA-256 into the extracted dataset provenance.

2. **Malom-version enforcement gap.**  The current builder writes `malom\_label\_version` into `meta` only after `\_annotate\_malom(...)` completes without error (`tools/\_human\_db\_build.py:642`), but the validator (`tools/validate\_human\_db\_candidate.py`) treats a missing `malom\_label\_version` as absent — it does not fail closed on presence of `malom\_wdl` rows without a version stamp.  Follow-up fix (out of scope for v3 code, in scope for a v3 blocker): validator must fail when `positions.malom\_wdl IS NOT NULL` and `meta.malom\_label\_version IS NULL`.  Until that lands, the v3 dataset must abort if the candidate DB has any Malom-labelled row with a missing / non-`sector-corrected-v1` version.

3. **Update double-count risk.**  `--update` uses `file\_path` as `processed\_files` PK (`tools/\_human\_db\_build.py:98`, verified) and only checks per-file SHA-256.  A host-path collision would double-count.  Reviewer §12 already asked for logical-source-identity — deferred.  V3 requires the DB it reads to have been built with **`--rebuild`** (not `--update`) on the same host as the extraction, and records the `processed\_files.sha256` list SHA into the dataset provenance.

### 4.3 Existing GapNet — §G1 discipline

The currently-live `data/gap\_net.npz` will remain live throughout v3 development.  §G1 remains open; v3 does not close it (v3 is a different target, not a rebuild of the existing composite).

The v3 candidate .npz filename is **`data/gap\_net\_v3\_candidate.npz`**. The un-suffixed `data/gap\_net.npz` is not touched. `data/gap\_net\_v2.npz` (if it appears) is not touched.  A `.pre-v3` backup rule mirrors the DB pattern.

### 4.4 `malom.query\_regret` does not exist

The pre-plan cites `malom.query\_regret(board, move)` as if it were an existing API.  It is not (verified: no such method in `ai/malom\_db.py`).  The v3 plan requires this API to be added **offline first**, tested against a golden corpus (§6), and versioned, before any dataset extraction proceeds.  §5.4 specifies the contract.


## 5. Objective move regret — `R\_v` specification

### 5.1 Contract

```
malom\_db.query\_regret(  
    parent\_board:  BoardState,     \# the position from which the move is played  
    move:          dict,           \# legal move dict as returned by get\_all\_legal\_moves  
) -\> RegretResult
```

Where `RegretResult` is a **dataclass** carrying:

- `available: bool` — True iff both `parent\_board` and `parent\_board.apply\_move(move)` returned a non-None `OracleValue` OR the child is a rules-terminal (§5.4 handling).

- `omv: OracleMoveValue | None` — the complete oracle move value from the parent's perspective, produced via `malom.move\_value(parent\_val, child\_val)` or `malom.terminal\_move\_value(parent\_val, terminal\_outcome)`.

- `wdl\_transition: str | None` — one of the 6 audit categories from `tools/audit\_human\_moves.py::\_classify\_transition`.

- `best\_omv: OracleMoveValue` — the maximum `OracleMoveValue` over all legal moves at `parent\_board`, computed via `compare\_oracle\_move\_values`.  `None` iff any legal successor's OMV is unavailable (fail-closed rule §5.5).

- `components: dict\[str, float | None\]` — the versioned scalarizations of §5.3, EACH computed independently.  Any component that cannot be computed reports `None` (never `0.0`).

- `regret\_version: str` — a hard-coded module-level constant such as `"regret\_v1"`; loaded at import time; recorded in every downstream artefact.

- `malom\_label\_version: str` — passed through from the DB; asserted equal to `sector-corrected-v1` (fail-closed).

The function **must never** return a valid result when either input probe fails and the child is not a rules-terminal.  It must never return a `float` or `int` as a top-level scalar — only the ordered tuple.  A caller can extract a component after acknowledging its version.

### 5.2 Never do

- `key2` subtraction across two `OracleValue`s (`ai/malom\_db.py:598-604` ordering flips on `key1` sign — subtraction is not meaningful).

- `dtw` subtraction across parent and child.

- `sector\_value` arithmetic across different sectors.

- Silent zero substitution.

- Silent uniform-distribution fallback.

- Silent neutral-default of any kind.

Any code path that would compute one of the above must instead return `available = False` with an `unavailable\_reason` string.

### 5.3 Candidate scalarizations — kept as separate components

**Rationale (reviewer request):** do not collapse regret into a single arbitrary scalar until the dataset has been trained and evaluated with each component as a candidate target.  Retain as separate outputs in the first dataset so the training pipeline can select between them or train multi-head.

Version each component with a suffix (e.g. `regret\_v1.rank\_loss`).

**Component A — probability of outcome-class downgrade** **(`class\_downgrade\_prob`).**  Zero or one.  The move's `wdl\_transition` under the mover's POV; `1.0` iff the transition is in the set `\{win\_to\_draw, win\_to\_loss, draw\_to\_loss\}`; `0.0` iff in `\{win\_preserved, draw\_preserved, all\_losing\}`; `None` iff `label\_inconsistency` or `unlabelled`.  Simplest possible target; used as a baseline.

**Component B — expected frozen W/D/L utility loss** **(`wdl\_utility\_loss`).**  Given `U(W)=+1, U(D)=0, U(L)=-1`, this is `U(best\_move\_after\_mover\_pov) − U(this\_move\_after\_mover\_pov)` where the "mover POV" outcome is the flipped `malom\_wdl\_after`.  Range `\[0, 2\]`.  Frozen — the utility mapping is a versioned constant; do not tune it during training.

**Component C — normalised ordinal rank loss** **(`ordinal\_rank\_loss`).**  Sort the legal moves by `compare\_oracle\_move\_values` (ties broken deterministically by canonical notation).  `rank(this\_move) − rank(best\_move)` divided by `(n\_legal\_moves − 1)` if `n\_legal\_moves ≥ 2`, else `0.0`.  Range `\[0, 1\]`.  Directly uses the OracleMoveValue ordering as the reviewer required.

**Component D — within-class distance component** **(`within\_class\_distance`).**  **Only** where its ordering and scale are formally specified.  Candidate spec: for two moves with the same mover-POV WDL, the distance is `sign(mover\_pov\_wdl) × (dtw\_this − dtw\_best) / max\_dtw\_in\_position` where `max\_dtw\_in\_position` is the largest `abs(dtw)` observed across the legal moves.  Only defined when all legal moves share the same mover-POV WDL — otherwise `None`.

Component D is the highest-risk of the four (any within-class metric depends on Malom internals) and is retained separately so the trainer can be evaluated with and without it.

### 5.4 Rules-terminal handling

If `parent\_board.apply\_move(move)` yields a rules-terminal successor (no legal moves, or captured piece count breaches rules), the caller must use `malom.terminal\_move\_value(parent\_val, terminal\_outcome)` (`ai/malom\_db.py:983`) instead of probing the child.  `R\_v` for a terminal child inherits the terminal outcome via the parent's oracle value.  §6 requires the golden corpus to cover:

- placement-terminal (piece reduces to 2)

- movement-terminal (no legal moves)

- capture-terminal (own piece falls to 2 after opponent capture)

with the correct `terminal\_outcome` supplied.

### 5.5 Fail-closed rule

A single unavailable component in a single position makes `G\_v` unavailable for that position.  The extractor must record it under `positions\_abstained\_reason` and **not** include the position in the training set.  There is no zero-imputation, no uniform-imputation, no neutral-default.  Coverage of the resulting dataset is reported as an artefact (§15) so the reader can see what fraction of the candidate DB is trainable.


## 6. Golden corpus for oracle-regret validation

Before `malom.query\_regret` is trusted for dataset extraction, it must be proved correct on a hand-verified corpus.  Location: `tests/fixtures/malom\_regret\_golden.json` (new file, populated by Stage A).

### 6.1 Categories the corpus must cover

Each row records: `state\_key`, `move\_notation`, `expected\_wdl\_transition`, `expected\_regret\_v1\_class\_downgrade\_prob`, `expected\_regret\_v1\_wdl\_utility\_loss`, `expected\_regret\_v1\_ordinal\_rank` (as fraction of tie-broken position), `expected\_regret\_v1\_within\_class\_distance` (nullable), plus a human comment.

- **Placement phase — early**: opening moves at plies 1-5, both a known winning selection and a known drawing alternative.

- **Placement phase — with capture**: a mill-closing move where a capture is required, both a good capture target and a bad one.

- **Movement phase — no capture**: mid-game positions with a mix of winning and drawing moves.

- **Movement phase — with capture**: forced-capture chain positions.

- **Fly phase — 3v3 endgame**: at least two positions, one drawn and one winning-for-mover.

- **Terminal successors**: three terminal-outcome cases (each of the three rules-terminals above); asserts the terminal-move-value path.

- **Every W/D/L transition category** as classified by `\_classify\_transition`: win\_preserved, win\_to\_draw, win\_to\_loss, draw\_preserved, draw\_to\_loss, all\_losing.  A position for each, with the expected regret values.

- **Equal-value alternatives**: at least one position where two moves have identical `OracleMoveValue`; assert `rank\_loss = 0` for the second-picked move under the tie-break rule.

- **Symmetry transforms**: for two positions, hand-write the D4-mirror variant and assert `R\_v` is invariant modulo notation-transform (Stage A must include a test that runs each golden row through all 8 D4 symmetries and asserts unchanged output).

- **Missing Malom coverage**: a position deliberately picked from a sector not present on the developer's machine (verified by `malom.query\_value(board)` returning None); assert `RegretResult.available = False` and the correct `unavailable\_reason`.

### 6.2 Fixture-test discipline

- Golden rows carry no computed values — only human-verified ground truth.

- The test file loads each row, runs `malom.query\_regret`, asserts on every recorded field.

- A test that fails because Malom is unavailable on the CI runner skips (per repo convention with `\_MALOM\_AVAILABLE`) rather than emitting a false pass.

- The corpus is versioned in git; every subsequent Malom-fix commit must run this suite.


## 7. Human move policy — `P\_h` source and ablation

### 7.1 What v3 depends on

**Primary:** the `HumanMovePolicyAdvisor.probs(board, legal\_moves, elo\_band)` API (`ai/human\_move\_policy\_advisor.py:126-171` at rename head).  Verified: pure numpy, softmax over every legal move, uniform-fallback on degenerate scores.

**Secondary:** for positions with `≥ min\_support` observed events in `data/human\_db\_candidate.sqlite:moves\_elo\_bins`, v3 will additionally compute the **direct empirical `P\_h`** as `total\_band(m) / Σ\_m' total\_band(m')`.  The extractor stores both. Downstream can then choose:

- Model-only training target: use `HumanMovePolicyAdvisor.probs`.

- Empirical-baseline `G\_v`: use `moves\_elo\_bins`-derived `P\_h`.

- Hybrid: use empirical where support permits, model elsewhere, reporting fraction of each in the training set.

### 7.2 Ablation: HumanMovePolicyNet vs HumanPrefNet

The reviewer requires an ablation that shows HumanPrefNet does **not** add independent information over HumanMovePolicyNet.  Rationale: the pre-plan and other planning docs assume HumanPrefNet is useful for "where humans will play well" while HumanMovePolicyNet models "where they will play at all".  If the ablation shows HumanPrefNet is either redundant or noise on top of HumanMovePolicyNet's output, HumanPrefNet is dropped from v3.

Ablation protocol:

- Compute `G\_v` three ways on a hand-picked stratified sample of positions (~500 per band):

  - `G\_v(band)` from HumanMovePolicyNet alone.

  - `G\_v(band)` from HumanMovePolicyNet mixed with HumanPrefNet's softmax at a temperature range (`\{0.5, 1.0, 2.0\}`), with the mix weight swept.

  - `G\_v(band)` from HumanPrefNet alone (as a baseline).

- Compare to a ground-truth `G\_v\_empirical(band)` computed from `moves\_elo\_bins` on the subset with `≥ 25` plays.

- Report **per-band MAE** and **per-band Spearman r** between each variant and the empirical ground truth.

Success criterion for keeping HumanPrefNet in v3: HumanPrefNet mix must reduce MAE by ≥ 5 % relative to HumanMovePolicyNet alone in at least two of the three bands.  Otherwise the pipeline commits to HumanMovePolicyNet only.

### 7.3 What v3 does NOT depend on

- Player-profile embeddings.  Deferred until the Elo-band-only variant is evaluated end-to-end.

- Position-complexity features (Toronto blunder-prediction finding). Deferred to a later ablation.

- Any HumanPrefNet retrain.  §H2 / §H4 in `discussion\_plan.md` are out of scope.


## 8. Expected human regret — `G\_v` dataset

### 8.1 Formula

```
G\_v(state, band, component\_c) = Σ\_\{m ∈ legal(state)\} P\_h(m | state, band) · R\_v(state, m).components\[c\]
```

Where `c` ranges over the components of §5.3.  The training target per `(state, band)` sample is a **length-4 vector** (four components), not a single scalar.

**Undefined when:**

- Any `R\_v(state, m)` component `c` returns `None` (fail-closed §5.5).

- `state`'s canonical `state\_key` is present in a held-out split.

- `state` has `n\_legal\_moves \< 2` (no gap possible).

### 8.2 Multiple outputs, not one scalar

The candidate GapNet v3 model has **four output heads** (one per component in §5.3), each trained with mean-squared error against its own target.  Downstream consumers can:

- Use one head at a time (recommended for shadow mode; §9.4).

- Weight the heads at inference (with the weights stored as part of the model's `.npz` metadata).

This preserves the reviewer's requirement that the plan not collapse regret into "one arbitrary scalar" during training.

### 8.3 Dataset construction rules

**Extraction pipeline (Stage D):**

1. Enumerate every `(state\_key, band)` with `≥ 1` legal move at the parent and full Malom coverage over every legal successor.

2. For each such `(state\_key, band)`:

   - Reconstruct `parent\_board` via `board\_from\_state\_key`.

   - Enumerate `legal\_moves`.

   - Compute `P\_h(m | band)` per §7.1.

   - Compute `R\_v(state, m).components` per §5.  If any component is unavailable for any legal `m`, abstain from this `(state\_key, band)` — write a row to `abstained.jsonl` with the reason and continue.

   - Compute the four `G\_v` components.

   - Record: `(state\_key, band, mover\_color, n\_legal\_moves, phase, G\_v\_class\_downgrade, G\_v\_wdl\_utility, G\_v\_ordinal\_rank, G\_v\_within\_class\_distance, P\_h\_source ∈ \{"model", "empirical", "hybrid"\})`.

3. Save as a numpy memmap keyed by sample index; save metadata to `data/gap\_net\_v3\_dataset/metadata.npz`.

4. Provenance: `candidate\_db\_sha256`, `malom\_label\_version`, `regret\_version`, `human\_move\_policy\_net\_sha256`, `human\_pref\_net\_sha256` (if used per §7.2 ablation), `feature\_version`, `git\_commit`, `built\_at`.  Every one required or extraction aborts.

### 8.4 Splits and provenance

- **Train / validation / test** partitions grouped by `session\_id` (game-level split), NOT by `state\_key`.  A position that appears in multiple games is assigned to the split of the game with the smallest deterministic hash.

- **Player-level held-out slice** — a diagnostic split.  Reserve the top-10 most prolific movers to a held-out slice reported separately.

- **Untouched final test set** — 5 % of `session\_id`s reserved before any model selection.  Never loaded except for the final report.

- Provenance record inherits from the extracted dataset (§8.3) plus a `split\_version` constant and the immutable seed for the game-hash.


## 9. Gameplay exploitation — approaches to compare

Do not simply extend the current additive leaf bonus.  Compare the four modes below in a `shadow` configuration (§9.4) before any of them is authorised for real play.

### 9.1 Baseline — pure minimax

Search runs unchanged; `G\_v` is computed at every ply but not consumed.  Establishes the "no signal" reference line.  The metric reported is expected score at difficulty 5 across 40 games / colour against the humanlike-blend opponent, per Step 6e of `retrain\_v2\_plan.md`.

### 9.2 Human-policy expectimax at opponent nodes

At opponent nodes (`board.turn != self.color`), replace the negamax minimum with

```
V\_opp(board) = Σ\_\{m ∈ legal(board)\} P\_h(m | band) · V\_child(board.apply\_move(m))
```

Requires an explicit expectimax path in `\_negamax` — verified absent in current code (Agent report §8).  Design constraints:

- **Sign discipline:** the expectimax value at an opponent node is the expected score of the resulting position **from the mover's POV** at that node, negated once (matching negamax convention).  A test must confirm that at a two-ply horizon with a known `P\_h` and a known `V\_child`, the returned root score matches the hand-computed expectimax value to full precision.

- **Terminal outcomes dominate.**  When `V\_child` is a terminal win or loss (Malom-terminal or rules-terminal), the expectimax value must never be smoothed by `P\_h` — the terminal score is returned exactly.  A test locks this: a position where `P\_h` puts 0.1 on a loss and 0.9 on a draw at ply 1 must return exactly the same value as pure minimax at ply 2 if the ply-1 loss is Malom-terminal.

### 9.3 Bounded mixture (worst-case ⊕ human-policy)

```
V\_opp(board) = (1-α) · min\_\{m\} V\_child(...) + α · E\_\{m ~ P\_h\} V\_child(...)
```

`α ∈ \[0, 1\]`, capped at a small value (`α\_max = 0.30`).  Never approaches 1: worst-case safety must dominate at high confidence. The cap is a **hard constant** in code, not a search parameter. Tests must confirm the mixture is always dominated by the worst-case value on positions where the worst legal move is a Malom loss.

### 9.4 Shadow mode

Search runs pure minimax (§9.1) as the authoritative selector.  In parallel, the same position is scored under each of the other three modes and the resulting score differences are logged to `data/logs/gap\_v3\_shadow\_YYYYMMDD.jsonl`:

```
\{  
  "state\_key": ...,  
  "chosen\_move\_by\_minimax": ...,  
  "would\_have\_chosen\_by\_expectimax": ...,  
  "would\_have\_chosen\_by\_mixture\_alpha\_0.3": ...,  
  "shadow\_score\_delta\_expectimax": ...,  
  "shadow\_score\_delta\_mixture": ...  
\}
```

**Shadow mode changes no moves.**  It is the first-step deployment mode and must land before any live consumption is authorised.  The regression tests in §14 assert that shadow mode does not affect the game outcome even in seeded games.

### 9.5 Perspective conversions — locked

Every consumption path must record and assert:

- `parent\_board.turn` — the mover at the parent position.

- `move` — from the mover's move set.

- `successor.turn` — the opponent (mover flipped).

- `board\_to\_features(successor, parent\_board.turn)` — successor features from the ORIGINAL MOVER'S POV (matches the training contract in `docs/human\_move\_policy\_net\_plan.md:302-311`).

- `P\_h(m | state, band)` — probability that the current-node mover plays `m`.

- `R\_v(state, m).omv` — the OracleMoveValue is in the current-node mover's perspective.  Negated once when compared against a sibling-node score (negamax convention).

- Malom `outcome` fields — child returns opponent's-POV; mover-POV is `\_FLIP\[after\]`.

A dedicated test file `tests/test\_gap\_v3\_perspective\_conversions.py` locks these six conversions with concrete assertions.

### 9.6 Two-ply regression tests

Required for §9.2 and §9.3.  Construction:

- Position where the AI (root) has two legal moves: `M\_safe` leads to a drawn position, `M\_trap` leads to a position where the opponent has a plausible-but-losing continuation.

- Under pure minimax at depth 2, both root moves score identically (opponent plays optimally so the trap doesn't fire).

- Under expectimax with a known `P\_h` at the opponent node that places significant mass on the losing continuation, `M\_trap` scores higher at the root than `M\_safe`.

- The test constructs three cases: expected `M\_trap` preferred, expected `M\_safe` preferred, expected tie.

If any of the three fails, the code path is disabled and the test records the failure signature.

### 9.7 Terminal-outcome dominance

Explicit assertions in the code path (via `assert` statements guarded by a `SEARCH\_INVARIANT\_CHECKS` env var so they can be turned off in production runs):

- `V\_opp(board) == V\_child(board)` when `board` has a single legal successor.

- `V\_opp(board)` at a Malom-terminal opponent node equals the terminal outcome exactly (no smoothing).

- The shadow-mode logged deltas never exceed a hard cap (positions where the delta exceeds the cap are flagged as candidates for bugs).


## 10. Training consumption pathway for gen 3

Progression from low-risk to high-risk.  Each step is a separate experiment, separately authorised.

### 10.1 Logged feature (default)

`G\_v(state, band)` (all four components) is computed per ply and written to `train\_log.jsonl` under a new column `gap\_v3.\<component\>`.  The training loop **does not consume it**. The offline plotter (`tools/plot\_specialist\_training.py`) can render its distribution over training.

Purpose: verify the signal is stable, correlated with observed outcomes, and not degenerate.  Success signal: after N games, plotted `G\_v` correlates positively with subsequent-ply Malom regret at significance `p \< 0.001`.

### 10.2 Auxiliary prediction head

The gen 3 policy net grows an additional head that predicts `G\_v` (any single component chosen by an offline ablation).  The auxiliary loss is added with a small hard-coded weight (`aux\_loss\_weight = 0.05`) and does **not** back-prop into the value or policy heads.

Purpose: build a compact learned representation of expected human regret without letting it modify the policy directly.

Success signal: the auxiliary head's held-out MSE matches the offline oracle `G\_v` computation within a documented tolerance; the policy / value heads remain unchanged on frozen-model bench.

### 10.3 Input feature to policy / value network

The gen 3 policy net conditions on `G\_v` as an input feature.  The signal enters through concatenation with the board features (dim +4), not through a gate.  Trained end-to-end.

Purpose: allow the policy to steer toward positions where opponents have high expected regret, while retaining terminal-outcome dominance through the existing minimax structure.

Success signal: gen 3 with `G\_v` input outperforms gen 3 without `G\_v` input on the humanlike-blend-opponent bench at Step 6e.

### 10.4 Bounded reward shaping (separately authorised)

Not part of this plan.  Requires:

- Successful landings of §10.1, §10.2, §10.3 in that order.

- Explicit tests for reward hacking (positions where the policy learns to force `G\_v` up in ways that reduce actual win rate).

- A hard bound: `reward\_shaping\_term\_magnitude ≤ 0.1 · terminal\_reward\_magnitude` at every training step.

- Terminal-outcome dominance test: on positions where the terminal outcome is known, the shaping term is exactly zero.

Owning document to author when this experiment is authorised: a separate `gap\_net\_v3\_reward\_shaping\_plan.md`.


## 11. Architecture — v3 candidate model

### 11.1 The 79-feature MLP as a compatibility baseline

The current shared 79-feature encoding (`ai/value\_net.board\_to\_features`) is the baseline input; the four regret-component heads sit on top.  The baseline model is `79 → 128 → 64 → 32 → 4` (input dim = 79 for the state — no band one-hot in the baseline; band is applied at target computation time via §7).

**This shape is not declared permanent.**  It is retained for comparability with HumanMovePolicyNet, ValueNet, and the current GapNet.  Ablations against alternative encoders are permitted in subsequent revisions.

### 11.2 Symmetry augmentation + invariance tests

- Data augmentation: for each `(state\_key, band)` sample, generate its 7 non-identity D4 transforms during training with probability 1/8.  Every transformed sample carries the same targets (verified invariant to symmetry by §6 golden corpus).

- Invariance test: after training, held-out positions and their symmetry transforms must produce identical head outputs to within a tolerance of `1e-3`.  A test file locks this.

### 11.3 Topology / graph-aware encoding — deferred

`docs/archive/discussion\_plan.md` §M2 explicitly defers topology- aware board representations to a fresh branch.  V3 respects that deferral; a topology encoder ablation is a candidate for a v4-style follow-up.


## 12. Naming, filenames, activation discipline

- Candidate model: `data/gap\_net\_v3\_candidate.npz`.

- Candidate dataset: `data/gap\_net\_v3\_dataset/` (memmap + metadata).

- Log outputs: `data/logs/gap\_v3\_shadow\_YYYYMMDD.jsonl`.

- Golden fixture: `tests/fixtures/malom\_regret\_golden.json`.

- Regret-version constant: `learned\_ai/data/regret\_version.py` with `REGRET\_VERSION = "regret\_v1"`.

**Never** overwritten by any v3 code:

- `data/gap\_net.npz` (loaded live by `web/app.py:256-262`).

- `data/gap\_net\_v2.npz` (if present — reserved for a hypothetical §G1 fix, not for v3).

- `data/human\_db.sqlite` (active DB).

- `data/human\_pref\_net.npz`, `data/human\_move\_policy\_net.npz`.

Promotion of `gap\_net\_v3\_candidate.npz` to `gap\_net.npz` is a **separate later decision** (see §16).


## 13. Implementation stages — atomic

Each stage is a single reviewable commit or a small commit series that lands a discrete deliverable.  Stages are gated: no stage may proceed while its predecessor's evidence is missing.

**Stage A — `malom.query\_regret` API + golden corpus.**  Adds the API in §5.1 to `ai/malom\_db.py` behind a versioned constant.  Populates the golden fixture in `tests/fixtures/malom\_regret\_golden.json`. Ships `tests/test\_malom\_regret\_v1.py` locking every category in §6.1 including symmetry invariance and the missing-Malom abstention path.

**Stage B — HumanMovePolicyNet Phase 4b diagnostics.**  Extends `tools/eval\_human\_move\_policy\_net.py` with the metrics in §4.1 (Brier, corrected calibration, uniform / empirical baselines, per- legal-move-count, OOD, abstention).  Reworks `learned\_ai/data/human\_db\_split.py` to add game-level and player-level splits.  Trains the current dataset to convergence, saves `data/human\_move\_policy\_net\_candidate.npz`, runs the full eval, produces `data/gap\_v3\_prerequisite\_eval.json`.

**Stage C — Direct-lookup `G\_v` computation script.**  Offline computation on the candidate DB's high-support positions. `tools/compute\_g\_v\_direct.py` reads `moves\_elo\_bins` and the `query\_regret` API, produces `data/gap\_v3\_direct\_gv.parquet` with per- component `G\_v` values.  This is the empirical baseline that all subsequent model training must beat.

**Stage D — Extract v3 dataset.**  `tools/extract\_gap\_v3\_dataset.py` executes the extraction pipeline in §8.3.  Emits `metadata.npz` + memmap + provenance JSON at `data/gap\_net\_v3\_dataset/`.  Emits `abstained.jsonl` with a coverage summary.  A regression test on a tiny synthetic slice locks the four-component structure and the fail-closed rule.

**Stage E — Train v3 candidate model.**  `tools/train\_gap\_net\_v3.py` implements the four-head MLP in §11.  Saves `data/gap\_net\_v3\_candidate.npz` with full provenance.  Trains against the game-split train partition; monitors validation loss on the game-split val partition; never touches the game-split test partition during selection.

**Stage F — Held-out evaluation.**  `tools/eval\_gap\_net\_v3.py` executes the full evaluation on the untouched test partition, including the player-level diagnostic slice.  Reports every metric in §15.  The test partition is not run more than once per candidate model — a second run would leak.

**Stage G — Shadow-mode integration.**  Adds a shadow-mode flag to `ai/game\_ai.py` gated behind an env var; the flag routes the game through the shadow-mode logger (§9.4).  A dedicated regression test plays a seeded deterministic game with and without shadow mode enabled; the game outcome and move sequence must be identical.

**Stage H — Two-ply regression tests.**  `tests/test\_gap\_v3\_two\_ply.py` locks the three tests in §9.6, plus the perspective-conversion tests in §9.5.

**Stage I — Logged-feature training experiment.**  Adds the log column in §10.1 to `scripts/train\_s\_gen\_v2b.py`'s `train\_log.jsonl` under a flag `--log-gap-v3 \<path-to-candidate\>`.  The gen 3 trainer, when it exists, inherits this hook.

Later stages (§10.2 auxiliary head, §10.3 input feature) are authored as separate plans after Stage I lands its first analysis.


## 14. Regression test suite

Every test file below is required before its stage promotes.

- `tests/test\_malom\_regret\_v1.py` — the §6 golden corpus, symmetry invariance, missing-Malom abstention (Stage A).

- `tests/test\_gap\_v3\_perspective\_conversions.py` — the six conversions in §9.5 (Stage G, imported earlier for CI).

- `tests/test\_gap\_v3\_two\_ply.py` — the three two-ply scenarios in §9.6 (Stage H).

- `tests/test\_gap\_v3\_dataset\_faillclosed.py` — assert the extractor abstains on positions with any missing `R\_v` component (Stage D).

- `tests/test\_gap\_v3\_provenance.py` — assert the .npz carries every provenance field (Stage E).

- `tests/test\_gap\_v3\_symmetry\_invariance.py` — assert model output is identical modulo D4 transforms (Stage E).

- `tests/test\_gap\_v3\_shadow\_mode.py` — assert shadow mode does not change move sequences on a seeded game (Stage G).

- `tests/test\_gap\_v3\_no\_zero\_default.py` — assert missing Malom never becomes 0 anywhere in the pipeline (imported at every stage; each stage adds its own assertions).

- `tests/test\_gap\_v3\_terminal\_dominance.py` — assert Malom-terminal and rules-terminal successors are never smoothed by `P\_h` (Stage G / H).

- `tests/test\_gap\_v3\_ph\_source\_documented.py` — assert every dataset row records `P\_h\_source ∈ \{model, empirical, hybrid\}` (Stage D).


## 15. Evidence artefacts

Every stage produces a discrete artefact.  Every artefact is committed alongside its owning stage's commit.

| Stage | Artefact | Content |
| - | - | - |
| A | `data/malom\_regret\_v1\_report.json` | Golden-corpus test outcomes; hashes of the fixture and the API |
| B | `data/gap\_v3\_prerequisite\_eval.json` | Full Phase 4b eval of HumanMovePolicyNet: NLL, Brier, ECE, per-band, per-phase, per-transition, per-legal-move-count, OOD, abstention, uniform/empirical baselines, temperature-scaling report |
| B (side) | `data/human\_move\_policy\_net\_candidate.npz` | Trained candidate model (do NOT rename until v3 gates open) |
| C | `data/gap\_v3\_direct\_gv.parquet` | Empirical `G\_v` from high-support positions; the training-time baseline the model must beat |
| D | `data/gap\_net\_v3\_dataset/` | Memmap + metadata + provenance + abstained.jsonl |
| E | `data/gap\_net\_v3\_candidate.npz` | Trained candidate model with full provenance |
| F | `data/gap\_v3\_test\_report.json` | Untouched-test-set metrics per component per band per phase per stratum, with a plain-language "did the model beat baselines" summary |
| G | `data/logs/gap\_v3\_shadow\_YYYYMMDD.jsonl` | Shadow-mode logs from a seeded game batch |
| H | (no dataset artefact; tests are the artefact) | — |
| I | `data/gap\_v3\_logged\_feature\_summary.json` | Distribution of the logged `G\_v` component over a training run; correlation with subsequent-ply Malom regret |



## 16. Promotion gates

`data/gap\_net\_v3\_candidate.npz` may be renamed to `data/gap\_net\_v3.npz` (the promoted-but-not-live name) **only** if every one of the following holds against the Stage F artefact.  The promoted `gap\_net\_v3.npz` may be renamed to `data/gap\_net.npz` (the live name, replacing the current file) **only** after a further round of live-adjacent testing that is not authorised by this plan.

| Gate | Threshold |
| - | - |
| Stage A tests | All pass; golden corpus has ≥ 3 rows per category in §6.1 |
| Stage B eval (HumanMovePolicyNet Phase 4b) | Event-weighted NLL ≤ uniform-baseline − 20 % relative in every band; ECE ≤ 0.05 in every band after temperature scaling; OOD rate \< 10 % |
| Stage C direct `G\_v` | Sanity: `G\_v\_wdl\_utility\_loss` is monotonic decreasing in Elo band (upper \< middle \< lower) |
| Stage D extraction | Coverage ≥ 60 % of `moves\_elo\_bins`-eligible `(state\_key, band)` samples; `abstained.jsonl` reasons summable |
| Stage E training | Per-component held-out MSE improves over: (a) uniform-`P\_h` baseline by ≥ 30 % relative, (b) empirical-`P\_h` baseline (where support permits) by ≥ 10 % relative |
| Stage F test-set | Same thresholds on the untouched test partition |
| Stage F symmetry | D4-invariance test tolerance ≤ 1e-3 across held-out positions |
| Stage G shadow mode | Zero move-sequence divergences on a 100-game seeded batch |
| Stage H two-ply | All three cases pass |
| §9.5 perspective | All six conversions pass |
| Terminal dominance | Zero smoothing-of-terminal violations in Stage G logs |


If any gate misses by less than 1 pp / 1 % relative, the case is re-run once with a different seed.  Repeated near-misses are treated as fails.


## 17. Rollback

**During development.**  Any Stage may be reverted independently; the git history keeps the artefacts.  Rolling back Stage E is a `git revert` of that commit; no data is lost.

**After live promotion (not authorised by this plan).**  If a future `data/gap\_net.npz` promotion later needs to be undone:

```
cp data/gap\_net.npz data/gap\_net.v3.YYYYMMDD.bak  
mv data/gap\_net.v1.pre-v3.YYYYMMDD.bak data/gap\_net.npz
```

The current `data/gap\_net.npz` must be backed up to `data/gap\_net.v1.pre-v3.YYYYMMDD.bak` **before** any promotion.


## 18. Unresolved decisions

Each row records what evidence would resolve it.  These are not authorised for silent selection during implementation; they are either pinned to Stage-B evidence or escalated to a separate ask.

**D-1.  Which `R\_v` component becomes the primary training target?** Evidence needed: Stage E per-component MSE deltas, cross-referenced with Stage F test-set improvements.  If two components are close (within 5 % relative), retain both as separate heads at inference.

**D-2.  Does HumanPrefNet stay in the pipeline?** Evidence: §7.2 ablation.  Default action if ablation is inconclusive: drop HumanPrefNet from v3, keep it available for the humanlike-blend game inference path (which does use it — verified `ai/game\_ai.py:2646`).

**D-3.  Does the model condition on Elo band explicitly (input** **feature) or use `G\_v(band)` externally (target only)?** Default: **target only** — the model learns per-position `G\_v` as if band were a training-time query.  A band-conditioned model becomes a Stage-F ablation.

**D-4.  Is Component D (within-class distance) shipped?** Evidence: Stage F per-component test-set MSE and its correlation with observed outcomes.  If Component D adds noise, drop it.

**D-5.  Coverage-vs-quality tradeoff.** Higher `min\_support` for direct `P\_h` gives cleaner ground truth but smaller training set.  Sweep `min\_support ∈ \{5, 10, 25\}` at Stage C and record the coverage-vs-noise curve.  Choose the operating point that maximises Stage E improvement.

**D-6.  Player-profile conditioning (§7.3).** Requires per-account history in the JSONL (not currently stored in HumanDB).  Deferred until a schema change is authorised.

**D-7.  Do we ship a compatibility-baseline model?** The 79-feature MLP is the baseline.  If a topology encoder ablation later beats it, the baseline stays in-repo as the fallback.


## 19. Non-goals — recap

- Not retraining ValueNet, HumanPrefNet, Sentinel.

- Not activating the candidate DB over `data/human\_db.sqlite`.

- Not overwriting `data/gap\_net.npz` at any stage.

- Not wiring v3 into `ai/game\_ai.py`, `ai/heuristics.py`, or `scripts/train\_s\_gen\_v2b.py` until every Stage F, G, H gate passes.

- Not choosing a topology-aware encoder (§M2 deferred).

- Not implementing reward-shaping consumption (§10.4 requires a separate authorised plan).

- Not implementing player-profile conditioning (§D-6 deferred).

- Not silently subtracting `key2` or `dtw` anywhere.

- Not returning 0 / uniform / neutral defaults on missing Malom lookups — every code path in every stage must **fail closed**.


*End of plan.  Every subsequent action described here is contingent* *on the gates in §16.  No code, data, or gameplay change is* *authorised by writing this document.*

