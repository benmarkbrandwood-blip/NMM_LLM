//! Phase detection re-exports. The logic lives in `board.rs` (`get_phase`,
//! `can_fly`, `terminal_winner`) to avoid duplication. This module provides the
//! integer encoding used by the Python-facing API: 0=place, 1=move, 2=fly.

use crate::board::get_phase;
use crate::types::{Board, Color};

pub fn detect_phase_u8(board: &Board, color: Color) -> u8 {
    get_phase(board, color).as_u8()
}
