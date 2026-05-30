# Rust Acceleration Layer — Integration Plan

This document is the canonical reference for the `feat/rust-core` Rust
acceleration layer (`native/nmm_core`, exposed to Python as `nmm_core`). It
records the EXACT board mapping, D4 symmetry, mill table, adjacency table,
heuristic terms, and DB-key formats copied from the Python source so the Rust
port reproduces them byte-for-byte.

> **Non-negotiable:** Rust never invents new indexing, transforms, mill/adjacency
> tables, or key formats. Everything below is copied verbatim from the Python
> sources cited in each section.

---

## 1. Canonical square index map

Source of truth: `game/board.py::POSITIONS` (duplicated in
`game/zobrist.py::_SQUARES` and `ai/board_symmetry.py::_POSITIONS`).

Bit index `i` in the Rust `Board.white`/`Board.black` `u32` corresponds to
`POSITIONS[i]`. The 24-char FEN board string (`BoardState.to_fen_string`) is also
in this exact order.

| idx | name | ring   | centred coords (x,y) | connections |
|----:|------|--------|----------------------|------------:|
|  0  | a7   | outer  | (-3,  3)             | 2 (corner)  |
|  1  | d7   | outer  | ( 0,  3)             | 3 (cross)   |
|  2  | g7   | outer  | ( 3,  3)             | 2 (corner)  |
|  3  | g4   | outer  | ( 3,  0)             | 3 (cross)   |
|  4  | g1   | outer  | ( 3, -3)             | 2 (corner)  |
|  5  | d1   | outer  | ( 0, -3)             | 3 (cross)   |
|  6  | a1   | outer  | (-3, -3)             | 2 (corner)  |
|  7  | a4   | outer  | (-3,  0)             | 3 (cross)   |
|  8  | b6   | middle | (-2,  2)             | 2 (corner)  |
|  9  | d6   | middle | ( 0,  2)             | 4 (cardinal)|
| 10  | f6   | middle | ( 2,  2)             | 2 (corner)  |
| 11  | f4   | middle | ( 2,  0)             | 4 (cardinal)|
| 12  | f2   | middle | ( 2, -2)             | 2 (corner)  |
| 13  | d2   | middle | ( 0, -2)             | 4 (cardinal)|
| 14  | b2   | middle | (-2, -2)             | 2 (corner)  |
| 15  | b4   | middle | (-2,  0)             | 4 (cardinal)|
| 16  | c5   | inner  | (-1,  1)             | 2 (corner)  |
| 17  | d5   | inner  | ( 0,  1)             | 3 (cross)   |
| 18  | e5   | inner  | ( 1,  1)             | 2 (corner)  |
| 19  | e4   | inner  | ( 1,  0)             | 3 (cross)   |
| 20  | e3   | inner  | ( 1, -1)             | 2 (corner)  |
| 21  | d3   | inner  | ( 0, -1)             | 3 (cross)   |
| 22  | c3   | inner  | (-1, -1)             | 2 (corner)  |
| 23  | c4   | inner  | (-1,  0)             | 3 (cross)   |

Color encoding (matches `_PIECE_BITS` in `ai/fullgame_db.py`): `. = 0b00`,
`W = 0b01`, `B = 0b10`.

---

## 2. D4 symmetry

Source: `ai/board_symmetry.py`. The D4 group has 8 elements built from matrices
applied to centred coordinates `(x,y) -> (ax+by, cx+dy)`:

| idx | name           | matrix (a,b,c,d) | inverse |
|----:|----------------|------------------|--------:|
|  0  | identity       | ( 1, 0, 0, 1)    | 0       |
|  1  | rot 90° CCW    | ( 0,-1, 1, 0)    | 3       |
|  2  | rot 180°       | (-1, 0, 0,-1)    | 2       |
|  3  | rot 270° CCW   | ( 0, 1,-1, 0)    | 1       |
|  4  | flip x-axis    | (-1, 0, 0, 1)    | 4       |
|  5  | flip y-axis    | ( 1, 0, 0,-1)    | 5       |
|  6  | main diagonal  | ( 0, 1, 1, 0)    | 6       |
|  7  | anti-diagonal  | ( 0,-1,-1, 0)    | 7       |

`SYM_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]`.

### 2a. Pre-computed board permutations

`_BOARD_PERM[sym_idx]` maps `POSITIONS index -> new POSITIONS index`. These are
computed at import time in Python; the values below are the EXACT arrays Rust
must hard-code (`D4_TRANSFORMS` in `symmetry.rs`).

**Apply semantics (critical):** `_apply_board_sym` sets
`result[perm[old_idx]] = board_24[old_idx]`. So `perm[i]` is the destination
index of the piece currently at index `i`.

```
0 (identity):     [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
1 (rot90 CCW):    [ 6, 7, 0, 1, 2, 3, 4, 5,14,15, 8, 9,10,11,12,13,22,23,16,17,18,19,20,21]
2 (rot180):       [ 4, 5, 6, 7, 0, 1, 2, 3,12,13,14,15, 8, 9,10,11,20,21,22,23,16,17,18,19]
3 (rot270 CCW):   [ 2, 3, 4, 5, 6, 7, 0, 1,10,11,12,13,14,15, 8, 9,18,19,20,21,22,23,16,17]
4 (flip x-axis):  [ 2, 1, 0, 7, 6, 5, 4, 3,10, 9, 8,15,14,13,12,11,18,17,16,23,22,21,20,19]
5 (flip y-axis):  [ 6, 5, 4, 3, 2, 1, 0, 7,14,13,12,11,10, 9, 8,15,22,21,20,19,18,17,16,23]
6 (main diag):    [ 4, 3, 2, 1, 0, 7, 6, 5,12,11,10, 9, 8,15,14,13,20,19,18,17,16,23,22,21]
7 (anti-diag):    [ 0, 7, 6, 5, 4, 3, 2, 1, 8,15,14,13,12,11,10, 9,16,23,22,21,20,19,18,17]
```

### 2b. Canonicalisation

`canonical_board_str(board_24)` (in `ai/board_symmetry.py`) returns the
**lexicographically smallest** of the 8 transformed 24-char strings, plus the
lowest sym_idx achieving it. Rust replicates by applying each transform to the
24-char string and taking the lex-min (string comparison of `.`/`W`/`B` chars,
i.e. ASCII byte order `'.'=46 < 'B'=66 < 'W'=87`).

Bitboard canonicalisation (`canonical_key(white, black)`) is provided for the
search/TT but the **DB key** path always goes through the 24-char string form
to guarantee byte-identical keys.

### 2c. Notation transforms

`transform_notation(notation, sym_idx)` transforms move strings (`"d2"`,
`"a1-b1"`, `"a1-b1xa4"`, `"d2xa4"`) position-by-position using the same D4
perm (per-position via `transform_pos`). Used for opening/trajectory keys.

---

## 3. Mill table

Source: `game/board.py::MILLS` (16 triples). As index triples:

```
( 0, 1, 2)  a7 d7 g7        ( 8, 9,10)  b6 d6 f6      (16,17,18)  c5 d5 e5
( 2, 3, 4)  g7 g4 g1        (10,11,12)  f6 f4 f2      (18,19,20)  e5 e4 e3
( 4, 5, 6)  g1 d1 a1        (12,13,14)  f2 d2 b2      (20,21,22)  e3 d3 c3
( 6, 7, 0)  a1 a4 a7        (14,15, 8)  b2 b4 b6      (22,23,16)  c3 c4 c5
( 1, 9,17)  d7 d6 d5        ( 3,11,19)  g4 f4 e4
( 5,13,21)  d1 d2 d3        ( 7,15,23)  a4 b4 c4
```

---

## 4. Adjacency table

Source: `game/board.py::ADJACENCY`. As index → neighbour indices:

```
 0 a7: [ 1,  7]            8 b6: [ 9, 15]          16 c5: [17, 23]
 1 d7: [ 0,  2,  9]        9 d6: [ 8, 10,  1, 17]  17 d5: [16, 18,  9]
 2 g7: [ 1,  3]           10 f6: [ 9, 11]          18 e5: [17, 19]
 3 g4: [ 2,  4, 11]       11 f4: [10, 12,  3, 19]  19 e4: [18, 20, 11]
 4 g1: [ 3,  5]           12 f2: [11, 13]          20 e3: [19, 21]
 5 d1: [ 4,  6, 13]       13 d2: [12, 14,  5, 21]  21 d3: [20, 22, 13]
 6 a1: [ 5,  7]           14 b2: [13, 15]          22 c3: [21, 23]
 7 a4: [ 6,  0, 15]       15 b4: [14,  8,  7, 23]  23 c4: [22, 16, 15]
```

---

## 5. Phase detection

Source: `game/rules.py::get_game_phase(board, color)`:

- `place` — `pieces_placed[color] < 9`
- `fly`   — `pieces_placed[color] == 9 AND pieces_on_board[color] <= 3`
- `move`  — otherwise (placed all 9, ≥ 4 on board)

`can_fly`: `pieces_placed == 9 and pieces_on_board <= 3`.
`is_blocked`: phase == move AND zero legal moves.

### Terminal detection (`game/rules.py::is_terminal`)

A player loses when:
1. They placed all 9 and have `< 3` on board, OR
2. It is their turn, they are in move phase, and they have no legal moves.

Returns `(terminal, winner)`.

---

## 6. Move generation

Source: `game/board.py` + `game/rules.py::get_all_legal_moves`.

- **Placement** (phase==place): every empty square is a target.
- **Movement** (phase==move): for each own piece, each adjacent empty square.
- **Flying** (phase==fly): for each own piece, every empty square.
- A partial move that forms a mill is expanded into one complete move per legal
  capture (`legal_captures`): opponent pieces NOT in a mill; if every opponent
  piece is in a mill, all opponent pieces are capturable.
- Otherwise the complete move has `capture = None`.

`does_form_mill`: applying the placement/movement (ignoring capture) puts the
moved piece into a fully-own mill containing the `to` square.

---

## 7. Heuristic scoring

Source: `ai/heuristics.py::evaluate`. The Rust port reproduces the **integer
base formula** (the dominant signal) using the phase weight tuples and the
integer feature helpers. UI float scaling (`HeuristicWeights.*_scale`) defaults
to 100 (no-op) and is applied identically when non-default.

### Phase weight tuples `_WEIGHTS[phase] = (mill, block, piece_diff, two_cfg, dbl_mill, win_cfg)`

```
place: (30, 12, 12, 5,  0,    0)
move:  (30, 48, 12, 5, 50,    0)
fly:   (32,350,  2, 0, 90, 1190)
```

### Auxiliary phase weights

```
_MOB_WEIGHTS         place 3   move 8   fly 20
_THREAT_WEIGHTS      place 15  move 18  fly 80
_CYCLE_WEIGHTS       place 8   move 22  fly 80
_FORK_WEIGHTS        place 6   move 14  fly 55
_HERD_WEIGHTS        place 6   move 18  fly 0
_NEAR_BLOCKED_WEIGHTS place 0  move 30  fly 0
_WRAP_WEIGHTS        place 0   move 40  fly 60
_FLY_ASYM_WEIGHTS    place 0   move 80  fly 0
_DOMINATION_WEIGHTS  place 0   move 150 fly 80
position_value weight: 4 * (our_pos - opp_pos)
```

### Base formula (integer)

```
base = mill_w*(our_mills-opp_mills) + block_w*blocked + w[2]*piece_diff
     + w[3]*(our_two-opp_two) + w[4]*(our_dbl-opp_dbl) + w[5]*win_cfg
     + mob_w*(our_mob-opp_mob) + THREAT*(our_thr-opp_thr) + 4*(our_pos-opp_pos)
     + CYCLE*(our_cycle-opp_cycle) + FORK*(our_fork-opp_fork)
     + HERD*(our_herd-opp_herd) + NEAR_BLOCKED*(our_squeeze-opp_squeeze)
     + WRAP*(our_wrap-opp_wrap) + FLY_ASYM*fly_asym + DOM*(our_dom-opp_dom)
```

(`blocked = _blocked_count(opp)`, `our_squeeze = _squeeze_count(opp)`,
`opp_squeeze = _squeeze_count(color)` — note the perspective swaps.)

Terminal: returns `+INF` if winner==color else `-INF` (`INF = 10_000_000`).

### Integer feature helpers (ported & parity-tested)

`_closed_mills`, `_blocked_count`, `_two_configs`, `_double_mills`,
`_win_config`, `_mobility` (fly capped at `_FLY_MOBILITY_CAP=5`), `_mill_threats`
(phase-aware reachability), `_position_value` (cardinal=5, cross=3, corner=2),
`_mill_cycle_ready`, `_fork_threats`, `_encirclement`, `_squeeze_count`,
`_mill_wrapping_pressure`, `_fly_asymmetry`, `_open_mill_domination`.

> The Python `evaluate` adds many further phase-conditional float-scaled terms
> (assembly gradients, convergence, sealed-2cfg, etc.). The Rust **evaluate** is
> a self-contained engine evaluation built on the base formula above; it is used
> by the Rust coarse-grained search only. The Python heuristic remains the
> default evaluator. See Risk Register §10.

Node classes (from `ai/heuristics.py`):
- `_CARDINAL_NODES` = {b4, d2, d6, f4} (idx 9, 11, 13, 15)
- `_CROSS_NODES_3`  = {d7, g4, d1, a4, d5, e4, d3, c4} (idx 1,3,5,7,17,19,21,23)
- corners = the rest.

---

## 8. Tactical motifs

Source: `ai/game_ai.py` move ordering + mandatory-block logic, and
`ai/heuristics.py` motif helpers. Ported/exposed for parity:
- `_immediate_mill_threats` — closing squares of opponent 2-configs reachable in
  one move (move phase: friendly-adjacency check; fly: any).
- mill-forming detection for move ordering.
- fork threats, mill-cycle readiness, encirclement, squeeze (see §7).
- double-mill convergence, sealed/free 2-configs (Python-only; not in Rust eval).

---

## 9. DB key generation

| DB | Source | Key form | Symmetry |
|----|--------|----------|----------|
| **FullGame** | `ai/fullgame_db.py::_encode_canonical` | 9 bytes: 6-byte LE packed 2-bit/square over **canonical** board24, then `turn`(0/1), `placed_w`, `placed_b` | `canonical_board_str` (lex-min D4) |
| **Endgame** (trajectory-style) | `ai/endgame_db.py` | string `"<canonical board24>|<turn>"` | `canonical_board_str` / `board_query_canonicals` |
| **EndgameSolved (WDL)** | `ai/endgame_solved_db.py::encode_position_id` | combinatorial position id (not symmetry-normalised; piece-count indexed file) | none (separate per-(nW,nB) file) |
| **Trajectory** | `ai/trajectory_db.py` | pipe-joined `canonical_sequence` of move notations | `canonical_sequence` (lex-min D4 over move strings) |
| **Opening** | `ai/opening_recognizer.py` | canonical move-notation sequence + FEN signatures | `canonical_sequence` / per-position `transform_notation` |

### FullGame key encoding (exact)

```python
_PIECE_BITS = {".":0, "W":1, "B":2}
val = sum(_PIECE_BITS[ch] << (i*2) for i,ch in enumerate(board24))   # board24 = canonical
key = val.to_bytes(6, "little") + bytes((0 if turn=="W" else 1, placed_w & 0xFF, placed_b & 0xFF))
```

Rust `py_db_key(white, black, turn, placed_w, placed_b)` rebuilds the 24-char
string from the bitboards, canonicalises (lex-min D4), then emits the identical
9 bytes. `py_opening_key(notations, depth)` returns the pipe-joined canonical
sequence (and sym_idx) exactly as `canonical_sequence`.

---

## 10. Search parameters

Source: `ai/game_ai.py`.
- Fixed-depth table (difficulty 1–4): `{1:2, 2:3, 3:4, 4:5}`.
- Time limits (s): `{1:0.3, 2:0.8, 3:2.5, 4:6.0}`; difficulty 5+ iterative deepen.
- Negamax + alpha-beta, killer moves (2/depth), history heuristic (Σdepth²),
  transposition table (`ai/transposition_table.py`, 2^18 slots, depth-preferred,
  EXACT/LOWER/UPPER flags), Zobrist hash (`game/zobrist.py`, fixed seed
  `0x9E3779B97F4A7C15`).
- The Rust search (`search.rs`) is **coarse-grained**: Python calls
  `py_get_best_move(white, black, white_placed, black_placed, stm, max_depth,
  time_limit_ms)` once and Rust runs the whole alpha-beta + iterative-deepening
  internally with its own Zobrist + TT. Python never calls per-node.

---

## 11. Risk register

| Risk | Mitigation |
|------|-----------|
| **Square indexing drift** | Single source = `POSITIONS`; Rust hard-codes the same 24 names/indices; parity test round-trips FEN. |
| **D4 transform composition / rotation vs reflection order** | Copy the 8 pre-computed `_BOARD_PERM` arrays verbatim; unit-test every transform against Python output for all suite boards. |
| **Canonicalisation tie-break** | Python takes lowest sym_idx on ties; Rust iterates 0..8 and keeps first strict-min — identical. Parity-tested. |
| **DB key endianness** | FullGame uses `to_bytes(6,"little")`; Rust emits little-endian 6 bytes + 3 trailing bytes; byte-compared in `test_symmetry_parity.py`. |
| **Lex-min string ordering** | ASCII byte order of `.`/`B`/`W`; Rust compares `&[u8]` / `&str` (same ordering). |
| **Move ordering divergence** | Rust search is self-contained (own engine), so its ordering need not match Python's; it is NOT required to return the same move as Python — only legal, sane play. Parity is enforced on primitives (movegen, mills, keys), not on chosen move. |
| **Flying trigger** | Exactly `placed==9 and on_board<=3`; copied verbatim; parity-tested across fly positions. |
| **Capture all-in-mills rule** | `legal_captures`: prefer non-mill; fall back to all if every opp piece in a mill. Copied verbatim; parity-tested. |
| **Float heuristic terms** | Rust evaluate intentionally implements only the integer base formula; the full Python float heuristic stays the default. Documented; not claimed as byte-identical. |
| **Heuristic UI scaling (floats)** | Rust mirrors Python `int(w * scale / 100)` truncation when scale != 100; default 100 is a no-op. |
| **chromadb optional dependency** | Parity tests import `game.*`/`ai.board_symmetry`/`ai.fullgame_db` directly (no chromadb). |
| **Rust module missing at runtime** | `ai/native_core.py` try/except import with Python fallback; game launches either way. |
| **Zobrist seed reproducibility** | Rust search uses its own internal Zobrist (search-local); not shared with Python TT, so no cross-process key requirement. |

---

## 12. PyO3 surface (`lib.rs`)

```
py_canonical_key(white:u32, black:u32) -> (u32, u32)
py_apply_transform(bits:u32, idx:usize) -> u32
py_legal_moves(white,black,wp,bp,stm) -> Vec<(Option<u8>,u8,Option<u8>)>
py_forms_mill(white,black,sq,color) -> bool
py_count_mills(white,black,color) -> u32
py_detect_phase(wp,bp,won,bon,stm) -> u8        # 0 place,1 move,2 fly
py_evaluate(white,black,wp,bp,stm) -> i64
py_get_best_move(white,black,wp,bp,stm,max_depth,time_ms) -> (Option<u8>,u8,Option<u8>)
py_db_key(white,black,turn,placed_w,placed_b) -> Vec<u8>     # 9-byte fullgame key
py_canonical_board_str(board24:&str) -> (String, usize)
py_opening_key(notations:Vec<String>, depth:usize) -> (String, usize)
py_transform_notation(notation:&str, idx:usize) -> Option<String>
```
