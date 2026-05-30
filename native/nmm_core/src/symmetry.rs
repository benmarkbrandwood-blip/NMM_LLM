//! D4 symmetry transforms + canonicalisation. The 8 permutation arrays are
//! copied verbatim from Python `ai/board_symmetry.py::_BOARD_PERM` (extracted at
//! build-design time). See `docs/RUST_INTEGRATION_PLAN.md` §2.
//!
//! Apply semantics (matching Python `_apply_board_sym`):
//!   result[perm[old_idx]] = board[old_idx]
//! i.e. `perm[i]` is the DESTINATION index of the piece at index `i`.

use crate::types::N_SQUARES;

/// 8 D4 transforms as POSITIONS-index permutations.
/// Order: 0 identity, 1 rot90CCW, 2 rot180, 3 rot270CCW,
///        4 flip-x, 5 flip-y, 6 main-diag, 7 anti-diag.
pub const D4_TRANSFORMS: [[u8; 24]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    [6, 7, 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13, 22, 23, 16, 17, 18, 19, 20, 21],
    [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19],
    [2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 18, 19, 20, 21, 22, 23, 16, 17],
    [2, 1, 0, 7, 6, 5, 4, 3, 10, 9, 8, 15, 14, 13, 12, 11, 18, 17, 16, 23, 22, 21, 20, 19],
    [6, 5, 4, 3, 2, 1, 0, 7, 14, 13, 12, 11, 10, 9, 8, 15, 22, 21, 20, 19, 18, 17, 16, 23],
    [4, 3, 2, 1, 0, 7, 6, 5, 12, 11, 10, 9, 8, 15, 14, 13, 20, 19, 18, 17, 16, 23, 22, 21],
    [0, 7, 6, 5, 4, 3, 2, 1, 8, 15, 14, 13, 12, 11, 10, 9, 16, 23, 22, 21, 20, 19, 18, 17],
];

/// Inverse transform index for each D4 element.
pub const SYM_INVERSE: [usize; 8] = [0, 3, 2, 1, 4, 5, 6, 7];

/// Apply transform `idx` to a 24-bit board mask. Bit at `i` moves to `perm[i]`.
pub fn apply_transform(bits: u32, idx: usize) -> u32 {
    if idx == 0 {
        return bits;
    }
    let perm = &D4_TRANSFORMS[idx];
    let mut out = 0u32;
    for i in 0..N_SQUARES {
        if bits & (1 << i) != 0 {
            out |= 1u32 << perm[i];
        }
    }
    out
}

/// Apply transform `idx` to a 24-byte board string (`.`/`W`/`B`).
/// `result[perm[i]] = src[i]`.
pub fn apply_transform_str(src: &[u8], idx: usize) -> [u8; 24] {
    let perm = &D4_TRANSFORMS[idx];
    let mut out = [b'.'; 24];
    for i in 0..N_SQUARES {
        out[perm[i] as usize] = src[i];
    }
    out
}

/// Canonical (lex-min over D4) 24-char board string. Returns (canonical, sym_idx)
/// where sym_idx is the lowest index achieving the minimum (matching
/// `canonical_board_str`).
pub fn canonical_board_str(board24: &str) -> (String, usize) {
    let src = board24.as_bytes();
    let mut best: [u8; 24] = {
        let mut a = [b'.'; 24];
        a[..N_SQUARES].copy_from_slice(&src[..N_SQUARES]);
        a
    };
    let mut best_idx = 0usize;
    for idx in 1..8 {
        let t = apply_transform_str(src, idx);
        if t < best {
            best = t;
            best_idx = idx;
        }
    }
    (String::from_utf8(best.to_vec()).unwrap(), best_idx)
}

/// Canonical (white, black) bitboard pair: lex-min over the 8 transforms,
/// comparing (white, black) tuples. Used for search/TT, not DB keys.
pub fn canonical_key(white: u32, black: u32) -> (u32, u32) {
    let mut best = (white, black);
    for idx in 1..8 {
        let cand = (apply_transform(white, idx), apply_transform(black, idx));
        if cand < best {
            best = cand;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each transform must be a bijection of 0..24.
    #[test]
    fn transforms_are_permutations() {
        for (idx, perm) in D4_TRANSFORMS.iter().enumerate() {
            let mut seen = [false; 24];
            for &p in perm.iter() {
                assert!(!seen[p as usize], "transform {idx} not a bijection");
                seen[p as usize] = true;
            }
            assert!(seen.iter().all(|&x| x), "transform {idx} missing indices");
        }
    }

    /// Identity must be a no-op.
    #[test]
    fn identity_noop() {
        let bits = 0b1010_1100_0011u32;
        assert_eq!(apply_transform(bits, 0), bits);
    }

    /// rot90 applied 4 times == identity.
    #[test]
    fn rot90_order_four() {
        let bits = 0b0000_0000_1011_0101u32;
        let mut b = bits;
        for _ in 0..4 {
            b = apply_transform(b, 1);
        }
        assert_eq!(b, bits);
    }

    /// Each transform composed with its inverse == identity.
    #[test]
    fn inverse_roundtrip() {
        let bits = 0b1101_0010_1001_0110u32;
        for idx in 0..8 {
            let t = apply_transform(bits, idx);
            let back = apply_transform(t, SYM_INVERSE[idx]);
            assert_eq!(back, bits, "inverse failed for transform {idx}");
        }
    }

    /// Reflections are self-inverse (4,5,6,7) and rot180 (2).
    #[test]
    fn self_inverse_elements() {
        let bits = 0b1010_0101_1100_0011u32;
        for &idx in &[2usize, 4, 5, 6, 7] {
            let t = apply_transform(bits, idx);
            let back = apply_transform(t, idx);
            assert_eq!(back, bits, "element {idx} not self-inverse");
        }
    }

    #[test]
    fn canonical_is_min() {
        let board = "W.......................";
        let (canon, _idx) = canonical_board_str(board);
        // The single W can be rotated to the lexicographically smallest spot.
        // canonical must be <= the original under byte comparison.
        assert!(canon.as_bytes() <= board.as_bytes());
    }
}
