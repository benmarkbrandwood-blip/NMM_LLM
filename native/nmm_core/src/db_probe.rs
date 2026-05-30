//! DB key generation (Stage A: Rust computes the key, Python does the I/O).
//! Must be byte-identical to Python. See `docs/RUST_INTEGRATION_PLAN.md` §9.

use crate::board::board24_string;
use crate::symmetry::canonical_board_str;

/// `_PIECE_BITS`: '.'=0, 'W'=1, 'B'=2.
#[inline]
fn piece_bits(ch: u8) -> u64 {
    match ch {
        b'W' => 1,
        b'B' => 2,
        _ => 0,
    }
}

/// FullGame DB 9-byte key. Mirrors `ai/fullgame_db.py::_encode_canonical`
/// applied to the canonical board string:
///   6-byte LE packed 2-bit/square + turn(0/1) + placed_w + placed_b
pub fn fullgame_key(white: u32, black: u32, turn: u8, placed_w: u8, placed_b: u8) -> Vec<u8> {
    let board24 = board24_string(white, black);
    let (canon, _sym) = canonical_board_str(&board24);
    let bytes = canon.as_bytes();
    let mut val: u64 = 0;
    for (i, &ch) in bytes.iter().enumerate() {
        val |= piece_bits(ch) << (i * 2);
    }
    // 6 little-endian bytes of val, then turn, placed_w, placed_b.
    let le = val.to_le_bytes();
    let mut out = Vec::with_capacity(9);
    out.extend_from_slice(&le[..6]);
    out.push(if turn == 0 { 0 } else { 1 });
    out.push(placed_w);
    out.push(placed_b);
    out
}

/// Endgame DB key string: "<canonical board24>|<turn>" (turn as 'W'/'B').
pub fn endgame_key(white: u32, black: u32, turn: u8) -> String {
    let board24 = board24_string(white, black);
    let (canon, _sym) = canonical_board_str(&board24);
    let t = if turn == 0 { 'W' } else { 'B' };
    format!("{canon}|{t}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_nine_bytes() {
        let k = fullgame_key(0, 0, 0, 0, 0);
        assert_eq!(k.len(), 9);
        assert_eq!(k, vec![0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn turn_and_counts_encoded() {
        let k = fullgame_key(0, 0, 1, 5, 7);
        assert_eq!(k[6], 1);
        assert_eq!(k[7], 5);
        assert_eq!(k[8], 7);
    }
}
