//! Mill table and mill detection. Copied verbatim from `game/board.py::MILLS`.
//! See `docs/RUST_INTEGRATION_PLAN.md` §3.

use crate::types::{Board, Color};

/// 16 mill triples as POSITIONS indices.
pub const MILLS: [[u8; 3]; 16] = [
    // Outer ring sides
    [0, 1, 2],   // a7 d7 g7
    [2, 3, 4],   // g7 g4 g1
    [4, 5, 6],   // g1 d1 a1
    [6, 7, 0],   // a1 a4 a7
    // Middle ring sides
    [8, 9, 10],  // b6 d6 f6
    [10, 11, 12], // f6 f4 f2
    [12, 13, 14], // f2 d2 b2
    [14, 15, 8], // b2 b4 b6
    // Inner ring sides
    [16, 17, 18], // c5 d5 e5
    [18, 19, 20], // e5 e4 e3
    [20, 21, 22], // e3 d3 c3
    [22, 23, 16], // c3 c4 c5
    // Cross-ring connecting lines
    [1, 9, 17],  // d7 d6 d5
    [3, 11, 19], // g4 f4 e4
    [5, 13, 21], // d1 d2 d3
    [7, 15, 23], // a4 b4 c4
];

/// Precomputed bit masks for each mill (lazily-free: const fn would be nicer but
/// keep it simple with a runtime-built table accessed via `mill_mask`).
#[inline]
pub fn mill_mask(i: usize) -> u32 {
    let m = MILLS[i];
    (1u32 << m[0]) | (1u32 << m[1]) | (1u32 << m[2])
}

/// True if `color` would have a completed mill involving `square`.
pub fn forms_mill(board: &Board, square: u8, color: Color) -> bool {
    let bits = board.bits(color);
    for i in 0..16 {
        let mm = mill_mask(i);
        if mm & (1 << square) != 0 && (bits & mm) == mm {
            return true;
        }
    }
    false
}

/// Count fully-owned mills for `color`.
pub fn count_mills(board: &Board, color: Color) -> u32 {
    let bits = board.bits(color);
    let mut n = 0;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (bits & mm) == mm {
            n += 1;
        }
    }
    n
}

/// True if every piece of `color` is part of some mill.
pub fn all_in_mills(board: &Board, color: Color) -> bool {
    let bits = board.bits(color);
    if bits == 0 {
        return false;
    }
    let mut covered = 0u32;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (bits & mm) == mm {
            covered |= mm;
        }
    }
    (bits & !covered) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Color;

    fn board_from_idx(white: &[u8], black: &[u8]) -> Board {
        let mut w = 0u32;
        let mut b = 0u32;
        for &i in white {
            w |= 1 << i;
        }
        for &i in black {
            b |= 1 << i;
        }
        Board {
            white: w,
            black: b,
            white_placed: 9,
            black_placed: 9,
            side_to_move: Color::White,
        }
    }

    #[test]
    fn test_simple_mill() {
        let bd = board_from_idx(&[0, 1, 2], &[]);
        assert!(forms_mill(&bd, 0, Color::White));
        assert!(forms_mill(&bd, 1, Color::White));
        assert_eq!(count_mills(&bd, Color::White), 1);
    }

    #[test]
    fn test_cross_mill() {
        // d7 d6 d5 = idx 1 9 17
        let bd = board_from_idx(&[1, 9, 17], &[]);
        assert_eq!(count_mills(&bd, Color::White), 1);
        assert!(forms_mill(&bd, 9, Color::White));
    }

    #[test]
    fn test_no_mill() {
        let bd = board_from_idx(&[0, 1], &[2]);
        assert!(!forms_mill(&bd, 0, Color::White));
        assert_eq!(count_mills(&bd, Color::White), 0);
    }
}
