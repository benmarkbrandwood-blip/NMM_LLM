//! Zobrist hashing + transposition table for the Rust-internal search.
//!
//! This Zobrist is search-local (not shared with Python's `game/zobrist.py`),
//! so cross-process key compatibility is NOT required. The structure mirrors
//! `ai/transposition_table.py`: depth-preferred replacement, EXACT/LOWER/UPPER
//! flags. See `docs/RUST_INTEGRATION_PLAN.md` §10.

use crate::types::{Board, Color, N_SQUARES};

pub const EXACT: u8 = 0;
pub const LOWER_BOUND: u8 = 1;
pub const UPPER_BOUND: u8 = 2;

const TT_BITS: usize = 20; // 2^20 slots
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = (TT_SIZE as u64) - 1;

/// SplitMix64 for deterministic key generation (fixed seed).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

pub struct Zobrist {
    piece: [[u64; N_SQUARES]; 2],
    placed_done: [u64; 2],
    side: u64,
}

impl Zobrist {
    pub fn new() -> Self {
        let mut rng = SplitMix64(0x9E3779B97F4A7C15);
        let mut piece = [[0u64; N_SQUARES]; 2];
        for c in 0..2 {
            for s in 0..N_SQUARES {
                piece[c][s] = rng.next();
            }
        }
        let placed_done = [rng.next(), rng.next()];
        let side = rng.next();
        Zobrist {
            piece,
            placed_done,
            side,
        }
    }

    pub fn hash(&self, board: &Board) -> u64 {
        let mut h = 0u64;
        let mut w = board.white;
        while w != 0 {
            let i = w.trailing_zeros() as usize;
            h ^= self.piece[0][i];
            w &= w - 1;
        }
        let mut b = board.black;
        while b != 0 {
            let i = b.trailing_zeros() as usize;
            h ^= self.piece[1][i];
            b &= b - 1;
        }
        if board.white_placed >= 9 {
            h ^= self.placed_done[0];
        }
        if board.black_placed >= 9 {
            h ^= self.placed_done[1];
        }
        if board.side_to_move == Color::Black {
            h ^= self.side;
        }
        h
    }
}

#[derive(Clone, Copy)]
pub struct TtEntry {
    pub key: u64,
    pub depth: u8,
    pub score: i64,
    pub flag: u8,
    pub best_idx: u16, // index into the move list, or u16::MAX if none
}

pub struct TranspositionTable {
    table: Vec<Option<TtEntry>>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        TranspositionTable {
            table: vec![None; TT_SIZE],
        }
    }

    pub fn clear(&mut self) {
        for e in self.table.iter_mut() {
            *e = None;
        }
    }

    pub fn lookup(&self, key: u64) -> Option<TtEntry> {
        let e = self.table[(key & TT_MASK) as usize];
        match e {
            Some(entry) if entry.key == key => Some(entry),
            _ => None,
        }
    }

    pub fn store(&mut self, entry: TtEntry) {
        let idx = (entry.key & TT_MASK) as usize;
        match self.table[idx] {
            Some(existing) if existing.depth > entry.depth => {}
            _ => self.table[idx] = Some(entry),
        }
    }
}
