//! Board helpers: adjacency table, phase detection, terminal detection,
//! make_move, and FEN-string parsing. Copied verbatim from `game/board.py`
//! and `game/rules.py`. See `docs/RUST_INTEGRATION_PLAN.md` §4–6.

use crate::movegen::has_any_move;
use crate::types::{Board, Color, Move, Phase, N_SQUARES};

/// POSITIONS order, used for FEN parse/round-trip.
pub const POSITIONS: [&str; 24] = [
    "a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4", "b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4",
    "c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4",
];

/// Adjacency as bit masks, indexed by POSITIONS index. Copied from
/// `game/board.py::ADJACENCY`.
pub const ADJACENCY: [u32; 24] = build_adjacency();

const fn m(indices: &[u8]) -> u32 {
    let mut out = 0u32;
    let mut i = 0;
    while i < indices.len() {
        out |= 1u32 << indices[i];
        i += 1;
    }
    out
}

const fn build_adjacency() -> [u32; 24] {
    [
        m(&[1, 7]),         // 0 a7
        m(&[0, 2, 9]),      // 1 d7
        m(&[1, 3]),         // 2 g7
        m(&[2, 4, 11]),     // 3 g4
        m(&[3, 5]),         // 4 g1
        m(&[4, 6, 13]),     // 5 d1
        m(&[5, 7]),         // 6 a1
        m(&[6, 0, 15]),     // 7 a4
        m(&[9, 15]),        // 8 b6
        m(&[8, 10, 1, 17]), // 9 d6
        m(&[9, 11]),        // 10 f6
        m(&[10, 12, 3, 19]),// 11 f4
        m(&[11, 13]),       // 12 f2
        m(&[12, 14, 5, 21]),// 13 d2
        m(&[13, 15]),       // 14 b2
        m(&[14, 8, 7, 23]), // 15 b4
        m(&[17, 23]),       // 16 c5
        m(&[16, 18, 9]),    // 17 d5
        m(&[17, 19]),       // 18 e5
        m(&[18, 20, 11]),   // 19 e4
        m(&[19, 21]),       // 20 e3
        m(&[20, 22, 13]),   // 21 d3
        m(&[21, 23]),       // 22 c3
        m(&[22, 16, 15]),   // 23 c4
    ]
}

/// Per-color phase. Matches `game/rules.py::get_game_phase`.
pub fn get_phase(board: &Board, color: Color) -> Phase {
    if board.placed(color) < 9 {
        Phase::Place
    } else if board.count(color) <= 3 {
        Phase::Fly
    } else {
        Phase::Move
    }
}

#[inline]
pub fn can_fly(board: &Board, color: Color) -> bool {
    board.placed(color) == 9 && board.count(color) <= 3
}

/// True when color is in move phase and has no legal moves.
pub fn is_blocked(board: &Board, color: Color) -> bool {
    if get_phase(board, color) != Phase::Move {
        return false;
    }
    !has_any_move(board, color)
}

/// Terminal detection. Matches `game/rules.py::is_terminal`.
/// Returns `Some(winner)` if terminal, else `None`.
pub fn terminal_winner(board: &Board) -> Option<Color> {
    for &color in &[Color::White, Color::Black] {
        if board.placed(color) == 9 && board.count(color) < 3 {
            return Some(color.opponent());
        }
    }
    let current = board.side_to_move;
    if get_phase(board, current) == Phase::Move && is_blocked(board, current) {
        return Some(current.opponent());
    }
    None
}

#[inline]
pub fn is_terminal(board: &Board) -> bool {
    terminal_winner(board).is_some()
}

/// Apply a complete move, returning a new Board (copy-make). Mirrors
/// `BoardState.apply_move`. Does not validate legality.
pub fn make_move(board: &Board, mv: &Move) -> Board {
    let color = board.side_to_move;
    let mut white = board.white;
    let mut black = board.black;
    let mut wp = board.white_placed;
    let mut bp = board.black_placed;

    let to_mask = 1u32 << mv.to;
    match color {
        Color::White => {
            if let Some(f) = mv.from {
                white &= !(1u32 << f);
            } else {
                wp += 1;
            }
            white |= to_mask;
        }
        Color::Black => {
            if let Some(f) = mv.from {
                black &= !(1u32 << f);
            } else {
                bp += 1;
            }
            black |= to_mask;
        }
    }

    if let Some(cap) = mv.capture {
        let cm = 1u32 << cap;
        match color {
            Color::White => black &= !cm,
            Color::Black => white &= !cm,
        }
    }

    Board {
        white,
        black,
        white_placed: wp,
        black_placed: bp,
        side_to_move: color.opponent(),
    }
}

/// Parse the 24-char board portion of `BoardState.to_fen_string` into bitboards.
/// `board24` chars are `.`/`W`/`B` in POSITIONS order.
pub fn parse_board24(board24: &str, turn: Color, placed_w: u8, placed_b: u8) -> Board {
    let bytes = board24.as_bytes();
    let mut white = 0u32;
    let mut black = 0u32;
    for i in 0..N_SQUARES {
        match bytes.get(i) {
            Some(b'W') => white |= 1 << i,
            Some(b'B') => black |= 1 << i,
            _ => {}
        }
    }
    Board {
        white,
        black,
        white_placed: placed_w,
        black_placed: placed_b,
        side_to_move: turn,
    }
}

/// Serialize bitboards to the 24-char `.`/`W`/`B` POSITIONS-order string.
pub fn board24_string(white: u32, black: u32) -> String {
    let mut s = String::with_capacity(N_SQUARES);
    for i in 0..N_SQUARES {
        let mask = 1u32 << i;
        if white & mask != 0 {
            s.push('W');
        } else if black & mask != 0 {
            s.push('B');
        } else {
            s.push('.');
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_symmetric() {
        for a in 0..24usize {
            for b in 0..24usize {
                let a_adj_b = ADJACENCY[a] & (1 << b) != 0;
                let b_adj_a = ADJACENCY[b] & (1 << a) != 0;
                assert_eq!(a_adj_b, b_adj_a, "adjacency asymmetric {a} {b}");
            }
        }
    }

    #[test]
    fn parse_roundtrip() {
        let s = "W.B.W.B.W.B.W.B.W.B.W.B.";
        let bd = parse_board24(s, Color::White, 9, 9);
        assert_eq!(board24_string(bd.white, bd.black), s);
    }

    #[test]
    fn phase_logic() {
        let bd = Board {
            white: 0,
            black: 0,
            white_placed: 3,
            black_placed: 0,
            side_to_move: Color::White,
        };
        assert_eq!(get_phase(&bd, Color::White), Phase::Place);
    }
}
