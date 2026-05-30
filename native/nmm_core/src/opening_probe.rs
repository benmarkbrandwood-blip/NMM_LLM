//! Opening / trajectory key generation: move-notation symmetry transforms and
//! canonical sequences. Mirrors `ai/board_symmetry.py::transform_notation`,
//! `transform_pos`, and `canonical_sequence`.
//! See `docs/RUST_INTEGRATION_PLAN.md` §2c, §9.

use crate::board::POSITIONS;
use crate::symmetry::D4_TRANSFORMS;

/// Position name -> index (0..24), or None.
fn pos_index(name: &str) -> Option<usize> {
    POSITIONS.iter().position(|&p| p == name)
}

/// Transform a single position label by symmetry `sym_idx`.
/// Mirrors `transform_pos`: identity returns the same; otherwise map via perm.
pub fn transform_pos(pos: &str, sym_idx: usize) -> Option<String> {
    if sym_idx == 0 {
        return Some(pos.to_string());
    }
    let i = pos_index(pos)?;
    let dest = D4_TRANSFORMS[sym_idx][i] as usize;
    Some(POSITIONS[dest].to_string())
}

/// Transform a move notation string by symmetry `sym_idx`.
/// Handles: "d2", "a1-b1", "a1-b1xa4", "d2xa4". Returns None if any position is
/// unmapped. Mirrors `transform_notation`.
pub fn transform_notation(notation: &str, sym_idx: usize) -> Option<String> {
    if sym_idx == 0 {
        return Some(notation.to_string());
    }

    let mut cap_suffix = String::new();
    let base: &str;
    if let Some(xi) = notation.find('x') {
        base = &notation[..xi];
        let t_cap = transform_pos(&notation[xi + 1..], sym_idx)?;
        cap_suffix = format!("x{t_cap}");
    } else {
        base = notation;
    }

    if let Some(dash) = base.find('-') {
        let from_pos = &base[..dash];
        let to_pos = &base[dash + 1..];
        let t_from = transform_pos(from_pos, sym_idx)?;
        let t_to = transform_pos(to_pos, sym_idx)?;
        return Some(format!("{t_from}-{t_to}{cap_suffix}"));
    }

    let t_pos = transform_pos(base, sym_idx)?;
    Some(format!("{t_pos}{cap_suffix}"))
}

/// Transform a whole sequence; None if any notation is unmapped.
fn transform_sequence(notations: &[String], sym_idx: usize) -> Option<Vec<String>> {
    if sym_idx == 0 {
        return Some(notations.to_vec());
    }
    let mut out = Vec::with_capacity(notations.len());
    for n in notations {
        out.push(transform_notation(n, sym_idx)?);
    }
    Some(out)
}

/// Canonical (lex-min pipe-joined) sequence + sym_idx. Mirrors
/// `canonical_sequence`.
pub fn canonical_sequence(notations: &[String]) -> (Vec<String>, usize) {
    let mut best_joined = notations.join("|");
    let mut best_seq = notations.to_vec();
    let mut best_idx = 0usize;
    for sym_idx in 1..8 {
        if let Some(t) = transform_sequence(notations, sym_idx) {
            let j = t.join("|");
            if j < best_joined {
                best_joined = j;
                best_seq = t;
                best_idx = sym_idx;
            }
        }
    }
    (best_seq, best_idx)
}

/// Opening key for a prefix of `depth` notations: pipe-joined canonical form +
/// sym_idx.
pub fn opening_key(notations: &[String], depth: usize) -> (String, usize) {
    let d = depth.min(notations.len());
    let prefix = &notations[..d];
    let (seq, idx) = canonical_sequence(prefix);
    (seq.join("|"), idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_notation() {
        assert_eq!(transform_notation("a1-b1xa4", 0).unwrap(), "a1-b1xa4");
    }

    #[test]
    fn placement_transform_roundtrip() {
        // rot180 of a7(idx0) is g1(idx4) per perm[0]=4.
        assert_eq!(transform_pos("a7", 2).unwrap(), "g1");
    }

    #[test]
    fn canonical_sequence_picks_min() {
        let seq = vec!["g1".to_string()];
        let (canon, _idx) = canonical_sequence(&seq);
        assert!(canon.join("|") <= "g1".to_string());
    }
}
