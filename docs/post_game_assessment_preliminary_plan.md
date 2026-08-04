# Post-Game Assessment — Preliminary Plan

> **Status: Future direction — not yet prioritised.**
> Prerequisites in priority order: GapNet v3, BlunderNet definition, AI improvements, opening
> player. Only then does this become an active implementation target.
>
> This document is a preliminary design, not an authoritative implementation plan. It will
> need a review pass against the live codebase before any code is written.

---

## Motivation

The hypothesis to validate is: **comparing the same position under a short node budget and a
long node budget reveals moves that are falsely opportunistic — profitable-looking in a shallow
search but poor once the reply sequence is extended.** NMM_LLM is an unusually good candidate
for game review because it combines an exact-play oracle (Malom), real search (negamax with
evaluate_v2), human behaviour data (HumanDB / HumanMovePolicyNet), and an existing LLM/debrief
infrastructure that already produces prose from structured facts.

The strongest case for calling a move a horizon blunder is when all four conditions hold:

1. The move ranks highly (top-1 or top-2) under the short search.
2. Its rank or score drops materially under a longer search.
3. That result is stable across adjacent node budgets.
4. Malom independently confirms the move caused an objective WDL downgrade.

Malom establishes *whether* the move was poor. The short/long principal variations *explain*
why the shallow search was misled. "Material" and "adjacent" are left as open parameters to
calibrate during Stage 1 validation — do not fix numbers before seeing data.

---

## Stage 1 — Thin Slice (Initial Implementation)

**Goal:** Validate the horizon-effect hypothesis using only existing infrastructure and the Malom
DB. No new trained models are required. Stage 1 does **not** depend on GapNet v3, BlunderNet,
AI improvements, or the opening player — those are all Stage 2 concerns.

The six steps below follow the order recommended in preliminary review. Each step must be
working and verified before the next step adds anything.

### Step 1 — Extend the Existing Debriefer

Build on `ai/debriefer.py`, `tools/debrief.py`, and `MillsLLM.debrief_game()` /
`MillsLLM.debrief_position()`. Do not create a parallel `GameAssessor` class beside the
existing debriefer — that would split the codebase for no benefit. All Stage 1 additions are
extensions to `GameDebriefer` and its data classes.

### Step 2 — Deterministic Root Scorer

The existing `score_move()` call inside `GameDebriefer.analyse()` has three problems that make
it unsuitable for horizon comparison:

- It has a 3-second wall-clock deadline and can assign the current worst score to any move
  left unsearched when the deadline expires.
- It does not record completed depth, node count, or principal variation.
- Its result is not reproducible across runs.

Add a parallel, deterministic root scorer (does **not** replace `score_move()` in live play —
that stays unchanged). The new scorer must satisfy all of the following constraints:

- Identical engine and evaluation settings across both node budgets being compared.
- Single-threaded.
- Fixed node budgets (not time limits).
- No random move variance.
- No opening or trajectory DB bonuses.
- No Sentinel, ValueNet, GapNet, or HumanPref intervention during search.
- Records per root move at **both** budgets: score, rank, completed depth, node count, and
  principal variation.

Note on the leaf evaluator: the search leaf calls Rust `evaluate_v2()`, not the Python
`evaluate()` in `ai/heuristics.py`. The Python heuristic and its named terms (mobility,
mill-count, etc.) can be used to *explain* motifs in LLM commentary, but they must not be
presented as what produced the search result.

### Step 3 — Malom Adjudication of Horizon Disagreements

After scoring, check each horizon-disagreement ply against Malom:

- Preferred oracle: whole-game Malom DB (`ai/malom_db.py`), if mounted. Covers placement,
  movement, and fly phase.
- Fallback: retrograde WDL tables (`data/endgame/*.wdl`), fly phase only. Already available
  at inference time without an external mount.

Obtain `malom_wdl_before` and `malom_wdl_after` (both from the mover's perspective).

Objective downgrade classes — the only valid positive cases:
- `win_to_draw`
- `win_to_loss`
- `draw_to_loss`

Exclude from adjudication: positions where the pre-move state was already a loss (an
all-losing position is not a blunder opportunity), inconsistent labels, and any position where
Malom abstains. Fail closed per position — never silently substitute zero or a neutral default.
Record `abstained_reason` and `oracle_source` (`malom_full` / `retrograde_wdl` / `none`) in
every annotation.

Classification:
- `confirmed_blunder`: horizon disagreement present **and** Malom confirms a downgrade class.
- `horizon_disagree`: disagreement present but Malom abstains or is unavailable.
- Neither flag is set if the short and long searches agree.

A deeper search can also be unstable or wrong, so `horizon_disagree` without Malom confirmation
must not be presented to the user as a blunder.

### Step 4 — Validation on a Small Game Sample

Before adding any further signals, validate the thin slice on 5–10 games (mix of human-vs-human
and human-vs-AI), with the user reviewing the flagged moves directly. Gate Step 5 and
beyond on this review:

- Are the `confirmed_blunder` annotations meaningful in practice?
- Does the LLM commentary remain factually accurate and not hallucinate move quality?
- Is the fraction of `horizon_disagree`-only flags (Malom unavailable or abstained) acceptably
  low, or does it indicate a data coverage problem to fix first?

### Step 5 — Human Policy and Population Context

Add population comparison **only after** Step 4 validation confirms objective move quality is
trustworthy. Population signals answer a different question — "was this move human-like?" —
not "was it objectively good?" They must not be used in quality adjudication.

Sources (preference rule: if `n > 50` events at this position/Elo-band in
`data/human_db_candidate.sqlite:moves_elo_bins`, use the direct empirical distribution;
otherwise fall back to `HumanMovePolicyAdvisor.probs()`):

- `policy_prob`: probability a typical player of the same band would have played this move.
- `policy_prob_source`: `"empirical"` (n > 50) or `"learned"` (n ≤ 50).
- `policy_support_n`: the n used to decide which source applies.
- `population_move`: the top-1 move under the selected source, with its frequency.

A move is `unconventional` if `policy_prob` falls below a configurable threshold. Unconventional
moves are *described* (the population choice and its frequency), not penalised unless Malom or
the horizon comparison has already identified them as poor.

### Step 6 — LLM Synthesis

The LLM's role is unchanged from the existing debriefer: prose synthesis over already-decided
facts. The LLM does not adjudicate quality.

Extensions to the existing `MillsLLM.debrief_game()` prompt for Stage 1:

- Add a `HORIZON EFFECTS` section for `confirmed_blunder` plies: short-search PV, long-search
  PV, Malom downgrade class, oracle source.
- Add a `POPULATION COMPARISON` section for unconventional plies (Step 5, post-validation).
- Hard rule in the prompt: Malom outputs are W/D/L (discrete). Never present them as continuous
  figures like 0.51 or 0.23 — those would be model probabilities or empirical win rates, and
  must be labelled as such if used.
- Distinguish `oracle_source` (full Malom vs retrograde table) in every claim about move
  quality.

Extend the existing debrief LLM call budget minimally — do not add a new call per flagged move
until Stage 2 validation confirms that inline per-move commentary remains non-hallucinatory.

---

## Stage 2 — Full Implementation

**Status: Depends on Stage 1 validation, GapNet v3 training (Stages A–F of
`docs/gap_net_v3_plan.md`), BlunderNet purpose freeze (gate below), and
HumanMovePolicyNet checkpoint.**

### Gate: BlunderNet Purpose Freeze

Stage 2 does not begin until the BlunderNet's purpose has been chosen and frozen in a separate
design decision. The choice is not obvious:

- **Malom runtime surrogate**: predicts objective WDL downgrade when Malom is unavailable at
  inference time. This purpose is already largely served by Sentinel + GapNet v3; a dedicated
  BlunderNet becomes clearly useful only if it adds player, Elo, history, or behavioural
  features that those models do not capture.
- **Human-blunder predictor**: predicts the probability that a human player makes a
  downgrading move. HumanMovePolicyNet + Malom already computes this quantity directly, so
  a separate model needs to bring genuinely new input features to justify the training cost.

#### Corrected BlunderNet Label Definition

If BlunderNet is built, the label definition from the pre-plan's §2.8 is incorrect and must not
be used. `malom_wdl_after (opponent POV) == "W"` is insufficient because it includes moves from
positions that were already lost — those are not blunder opportunities.

Correct positive class:
- `malom_wdl_before` is W or D (mover's perspective, pre-move)
- **AND** `malom_wdl_after` creates a downgrade: `win_to_draw`, `win_to_loss`, or
  `draw_to_loss`

This requires both pre-move **and** post-move Malom values. Exclude: all-losing pre-move
positions, inconsistent labels, Malom abstentions.

HumanPrefNet is a pairwise ranking model — it is not the exact complement of BlunderNet.
The "mirror" framing from the pre-plan is imprecise and should not be used in code comments
or documentation.

### 2.1 GapNet v3 Integration

Integrate once `docs/gap_net_v3_plan.md` Stages A–F are complete. In the assessor:

- `G_v(state, band)` — position-level expected human regret — is **context** for the LLM
  ("this was a position where players at your level often concede quality"), not a quality
  adjudicator.
- The played move's own `R_v` components (`class_downgrade_prob`, `wdl_utility_loss`,
  `ordinal_rank_loss`) are per-ply explanatory signals alongside the Malom adjudication.

### 2.2 Signal Hierarchy (replaces `consensus_score`)

The pre-plan's fixed weighted sum is not appropriate. Malom, GapNet v3, Sentinel, and
BlunderNet are not independent votes — several trace back to the same Malom labels. Replace
the `consensus_score` field and the weighted formula with an explicit, layered hierarchy:

1. **Authority (adjudication):** Malom — whole-game DB or retrograde table — determines
   objective move quality when a result is available.
2. **Explanation:** Short/long search PVs, Sentinel, GapNet v3 `R_v` components explain
   *why* the shallow search was misled and *what motif* the position involves.
3. **Uncertain fallback:** Sentinel + GapNet v3 serve as uncertain proxies when Malom
   abstains. Label them as uncertain, not as adjudicators.
4. **Population context (separate question):** HumanMovePolicyNet / empirical DB answer
   "was this move human-like?" — not "was it objectively good?" Never allow a high human
   frequency to override a Malom-confirmed downgrade.

The legacy `data/gap_net.npz` (trained on a `SENTINEL_WEIGHT × sentinel_q + heuristic_q_norm`
composite, not on Malom values) must not be used in user-facing commentary. It may be reported
as an internal diagnostic if needed, but must be explicitly labelled as heuristic/sentinel-
derived if surfaced at all.

### 2.3 Turning Point Detection and Counterfactual

- `t* = argmax_t |wdl_curve[t] - wdl_curve[t-1]|` over the Malom WDL curve where available,
  falling back to the heuristic curve.
- Optional counterfactual (gated behind a `deep_analysis` flag): re-search from the turning
  point FEN at a fixed node budget, probe Malom for the best available alternative, record
  `counterfactual_gain = best_available_wdl - played_move_wdl`. Confirm against Malom before
  calling an alternative "better".

### 2.4 Full AnnotatedMove Record

Extend the `CriticalMoment` dataclass (or replace it with a per-ply `AnnotatedMove`) to carry:

- Short-search and long-search records (score, rank, depth, nodes, PV each).
- `horizon_disagree`, `confirmed_blunder`.
- Malom fields: `malom_wdl_before`, `malom_wdl_after`, `oracle_source`.
- Sentinel: `sentinel_score`.
- GapNet v3 (when available): `r_v_class_downgrade_prob`, `r_v_wdl_utility_loss`,
  `r_v_ordinal_rank_loss`. (`r_v_within_class_distance` deferred pending `gap_net_v3_plan.md`
  decision D-4.)
- BlunderNet (when available and purpose frozen): `blunder_prob`.
- Population: `policy_prob`, `policy_prob_source`, `policy_support_n`, `population_move`.
- HumanDB aggregate: `human_win_rate_after`, `avg_plies_to_end`.
- Derived: `is_unconventional`, `is_strong`, `turning_point_score`.
- Provenance: which oracle and which `P_h` source were used per ply.

### 2.5 Full LLM Synthesis

Structured prompt sections for the full `debrief_game()` call:

- **GAME FACTS**: result, termination type, piece counts, total plies.
- **TURNING POINT**: ply, Malom downgrade class (W/D/L — not decimal), oracle source,
  counterfactual best alternative if `deep_analysis` was run.
- **HORIZON EFFECTS**: confirmed blunder plies with short/long PV contrast.
- **POPULATION COMPARISON**: unconventional plies, source (empirical/learned), support n.
- **STRONG MOVES**: high-quality plies confirmed by Malom.

Hard constraint carried forward from Stage 1: Malom outputs are discrete W/D/L. Any
continuous figure in the prompt must be labelled as a model probability or empirical win rate.

Validate LLM commentary on a broader sample before shipping any frontend that presents it to
users. Confirm: does `llama3.1:8b` respect signal attribution, or does it invent quality claims
not in the fact block?

### 2.6 Frontend UI

After the annotation pipeline and LLM synthesis are validated:

- "Review Game" button in the post-game overlay.
- Annotation markers on the eval graph: confirmed blunder (red), horizon disagreement
  (orange, Malom absent), strong (green), unconventional (yellow).
- Click a marker → inline comment + best alternative highlighted on board if available.
- Narrative summary in the MillsAI chat panel.

---

## Dependencies and Build Order

| Component | Stage | Requires | New training? |
|-----------|-------|----------|---------------|
| Deterministic root scorer | 1 | Refactor of existing search | No |
| Horizon disagreement detection | 1 | Root scorer | No |
| Malom adjudication | 1 | Root scorer + Malom DB mount | No |
| Debriefer extension + LLM synthesis (thin slice) | 1 | Above three | No |
| Stage 1 game validation | 1 | Stage 1 complete + user review | No |
| Human policy comparison | 1 (Step 5) | HumanMovePolicyNet checkpoint | No |
| BlunderNet purpose freeze | Gate | Stage 1 validation | No |
| BlunderNet | 2 | Purpose freeze + corrected labels | Yes |
| GapNet v3 integration | 2 | `gap_net_v3_plan.md` Stages A–F | Yes (separate plan) |
| Signal hierarchy + full AnnotatedMove | 2 | BlunderNet + GapNet v3 + human policy | No |
| Turning point counterfactual | 2 | Malom + root scorer | No |
| Full LLM synthesis | 2 | Full annotation pipeline | No |
| Frontend UI | 2 | Full synthesis validated | No |

Stage 1 depends only on the existing debriefer and a Malom DB mount. It is **not** blocked
by GapNet v3, BlunderNet, AI improvements, or the opening player.

---

## Open Questions

1. **Horizon-disagreement thresholds.** What counts as "material" drop and "adjacent" budget
   for stability? Leave open until Stage 1 validation data exists.

2. **BlunderNet class imbalance.** The downgrade-positive class is a small minority of events.
   Weight the loss against a held-out set — do not assume a fixed weight.

3. **`within_class_distance` (GapNet v3 Component D).** Explicitly deferred pending
   `gap_net_v3_plan.md` decision D-4. Do not surface in user-facing commentary until that
   decision closes.

4. **`n > 50` threshold.** Fixed at 50 for the assessor's `P_h` source selection. If
   `gap_net_v3_plan.md` Stage C sweep suggests a better operating point for training, review
   independently — the two parameters serve different purposes and should not be silently
   inherited from each other.

5. **Player-history personalisation.** Recurring-pattern commentary via `bad_moves.json`
   and ChromaDB — defer to a later revision after Stage 2 validation.

6. **Legacy GapNet exposure.** Confirm as a product decision before Stage 2 ships any LLM
   text: the legacy `data/gap_net.npz` composite must not appear in user-facing commentary
   once GapNet v3 exists alongside it.
