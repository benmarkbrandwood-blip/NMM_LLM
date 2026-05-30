//! Core types: Color, Phase, Move, Board.
//!
//! Bit index `i` in `Board.white` / `Board.black` corresponds to
//! `POSITIONS[i]` in the Python `game/board.py` ordering. See
//! `docs/RUST_INTEGRATION_PLAN.md` §1.

pub const N_SQUARES: usize = 24;
pub const FULL_MASK: u32 = (1 << N_SQUARES) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn opponent(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub fn from_u8(v: u8) -> Color {
        if v == 0 {
            Color::White
        } else {
            Color::Black
        }
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            Color::White => 0,
            Color::Black => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Place,
    Move,
    Fly,
}

impl Phase {
    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            Phase::Place => 0,
            Phase::Move => 1,
            Phase::Fly => 2,
        }
    }
}

/// A fully-specified move. `from = None` for placement; `capture = None` when no
/// mill is formed. Indices are POSITIONS indices (0..24).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Option<u8>,
    pub to: u8,
    pub capture: Option<u8>,
}

/// Compact bitboard board state. `white_placed`/`black_placed` are cumulative
/// placement counts (0..9). `side_to_move` is whose turn it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    pub white: u32,
    pub black: u32,
    pub white_placed: u8,
    pub black_placed: u8,
    pub side_to_move: Color,
}

impl Board {
    #[inline]
    pub fn occupied(&self) -> u32 {
        self.white | self.black
    }

    #[inline]
    pub fn empty(&self) -> u32 {
        !self.occupied() & FULL_MASK
    }

    #[inline]
    pub fn bits(&self, color: Color) -> u32 {
        match color {
            Color::White => self.white,
            Color::Black => self.black,
        }
    }

    #[inline]
    pub fn count(&self, color: Color) -> u32 {
        self.bits(color).count_ones()
    }

    #[inline]
    pub fn placed(&self, color: Color) -> u8 {
        match color {
            Color::White => self.white_placed,
            Color::Black => self.black_placed,
        }
    }

    /// Color at a square: 0=empty, 1=white, 2=black (matches `_PIECE_BITS`).
    #[inline]
    pub fn piece_bits_at(&self, sq: usize) -> u8 {
        let m = 1u32 << sq;
        if self.white & m != 0 {
            1
        } else if self.black & m != 0 {
            2
        } else {
            0
        }
    }
}

/// Iterate set bits (square indices) of a 24-bit mask.
#[inline]
pub fn iter_bits(mut mask: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let i = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            Some(i)
        }
    })
}
