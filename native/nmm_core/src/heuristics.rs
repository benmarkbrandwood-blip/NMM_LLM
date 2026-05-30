//! Integer feature helpers + base evaluate. Ports the integer terms of
//! `ai/heuristics.py::evaluate`. See `docs/RUST_INTEGRATION_PLAN.md` §7.
//!
//! The full Python heuristic adds many float-scaled phase-conditional terms;
//! this Rust evaluate implements the integer BASE formula used by the
//! self-contained Rust search. Python remains the default evaluator.

use crate::board::{get_phase, terminal_winner, ADJACENCY};
use crate::mills::mill_mask;
use crate::types::{Board, Color, Phase, N_SQUARES};

pub const INF: i64 = 10_000_000;
const FLY_MOBILITY_CAP: i64 = 5;

// Cardinal nodes (4-conn): b4,d2,d6,f4 = idx 15,13,9,11
const CARDINAL_MASK: u32 = (1 << 9) | (1 << 11) | (1 << 13) | (1 << 15);
// Cross nodes (3-conn): d7,g4,d1,a4,d5,e4,d3,c4 = 1,3,5,7,17,19,21,23
const CROSS3_MASK: u32 =
    (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 17) | (1 << 19) | (1 << 21) | (1 << 23);

// Phase weight tuples (mill, block, piece_diff, two_cfg, dbl_mill, win_cfg).
fn weights(phase: Phase) -> [i64; 6] {
    match phase {
        Phase::Place => [30, 12, 12, 5, 0, 0],
        Phase::Move => [30, 48, 12, 5, 50, 0],
        Phase::Fly => [32, 350, 2, 0, 90, 1190],
    }
}

fn mob_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 3,
        Phase::Move => 8,
        Phase::Fly => 20,
    }
}
fn threat_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 15,
        Phase::Move => 18,
        Phase::Fly => 80,
    }
}
fn cycle_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 8,
        Phase::Move => 22,
        Phase::Fly => 80,
    }
}
fn fork_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 6,
        Phase::Move => 14,
        Phase::Fly => 55,
    }
}
fn herd_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 6,
        Phase::Move => 18,
        Phase::Fly => 0,
    }
}
fn near_blocked_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 30,
        _ => 0,
    }
}
fn wrap_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 40,
        Phase::Fly => 60,
        _ => 0,
    }
}
fn fly_asym_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 80,
        _ => 0,
    }
}
fn domination_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 150,
        Phase::Fly => 80,
        _ => 0,
    }
}

pub fn closed_mills(board: &Board, color: Color) -> i64 {
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

pub fn blocked_count(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if own & (1 << sq) != 0 && (ADJACENCY[sq] & empty) == 0 {
            count += 1;
        }
    }
    count
}

pub fn two_configs(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            count += 1;
        }
    }
    count
}

pub fn double_mills(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if own & (1 << sq) == 0 {
            continue;
        }
        let mut n = 0;
        let sq_mask = 1u32 << sq;
        for i in 0..16 {
            let mm = mill_mask(i);
            if mm & sq_mask != 0 && (own & mm) == mm {
                n += 1;
            }
        }
        if n >= 2 {
            count += 1;
        }
    }
    count
}

pub fn win_config(board: &Board, opp: Color) -> i64 {
    if board.placed(opp) == 9 && board.count(opp) <= 3 {
        1
    } else {
        0
    }
}

pub fn mobility(board: &Board, color: Color) -> i64 {
    let phase = get_phase(board, color);
    if phase == Phase::Fly {
        let empty = board.empty().count_ones() as i64;
        return FLY_MOBILITY_CAP.min(empty);
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if own & (1 << sq) != 0 {
            count += (ADJACENCY[sq] & empty).count_ones() as i64;
        }
    }
    count
}

pub fn mill_threats(board: &Board, color: Color) -> i64 {
    let phase = get_phase(board, color);
    let can_place = board.placed(color) < 9;
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            let empty_sq = (empty & mm).trailing_zeros() as usize;
            let reachable = match phase {
                Phase::Place => can_place,
                Phase::Fly => true,
                Phase::Move => {
                    // own piece adjacent to closing sq, excluding pieces inside this mill
                    let adj_own = ADJACENCY[empty_sq] & own & !mm;
                    adj_own != 0
                }
            };
            if reachable {
                count += 1;
            }
        }
    }
    count
}

pub fn position_value(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let mut total = 0;
    total += (own & CARDINAL_MASK).count_ones() as i64 * 5;
    total += (own & CROSS3_MASK).count_ones() as i64 * 3;
    let corners = own & !CARDINAL_MASK & !CROSS3_MASK;
    total += corners.count_ones() as i64 * 2;
    total
}

pub fn mill_cycle_ready(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (own & mm) != mm {
            continue;
        }
        // any piece in the mill with a free adjacent square
        let mut ready = false;
        for sq in 0..N_SQUARES {
            if mm & (1 << sq) != 0 && (ADJACENCY[sq] & empty) != 0 {
                ready = true;
                break;
            }
        }
        if ready {
            count += 1;
        }
    }
    count
}

pub fn fork_threats(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    // open mills = two-configs
    let mut open_masks: Vec<u32> = Vec::new();
    for i in 0..16 {
        let mm = mill_mask(i);
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            open_masks.push(mm);
        }
    }
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if own & (1 << sq) == 0 {
            continue;
        }
        let sq_mask = 1u32 << sq;
        let n = open_masks.iter().filter(|&&mm| mm & sq_mask != 0).count();
        if n >= 2 {
            count += 1;
        }
    }
    count
}

pub fn encirclement(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let opp = board.bits(color.opponent());
    let own = board.bits(color);
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if opp & (1 << sq) != 0 {
            count += (ADJACENCY[sq] & own).count_ones() as i64;
        }
    }
    count
}

pub fn squeeze_count(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for sq in 0..N_SQUARES {
        if own & (1 << sq) != 0 && (ADJACENCY[sq] & empty).count_ones() == 1 {
            count += 1;
        }
    }
    count
}

pub fn mill_wrapping_pressure(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    if get_phase(board, opp) == Phase::Fly {
        return 0;
    }
    let opp_bits = board.bits(opp);
    let own = board.bits(color);
    let mut total = 0;
    for i in 0..16 {
        let mm = mill_mask(i);
        if (opp_bits & mm) != mm {
            continue;
        }
        // covered: own pieces adjacent to any mill piece, not in the mill itself
        let mut covered = 0u32;
        for sq in 0..N_SQUARES {
            if mm & (1 << sq) != 0 {
                covered |= ADJACENCY[sq] & own & !mm;
            }
        }
        total += covered.count_ones() as i64;
    }
    total
}

pub fn fly_asymmetry(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    let color_fly = board.placed(color) >= 9 && board.count(color) == 3;
    let opp_fly = board.placed(opp) >= 9 && board.count(opp) == 3;
    if color_fly && !opp_fly {
        return 1;
    }
    if opp_fly && !color_fly && board.count(color) <= 5 {
        return -1;
    }
    0
}

pub fn open_mill_domination(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    let own_pieces = board.count(color) as i64;
    let opp_pieces = board.count(opp) as i64;
    if own_pieces < 6 || opp_pieces > 5 {
        return 0;
    }
    (two_configs(board, color) - (opp_pieces - 1)).max(0)
}

/// Integer base evaluate from `color`'s perspective. Mirrors the base formula in
/// `ai/heuristics.py::evaluate` (terminal + base term sum). Float-scaled
/// extras are intentionally omitted (see module doc).
pub fn evaluate_base(board: &Board, color: Color) -> i64 {
    if let Some(winner) = terminal_winner(board) {
        return if winner == color { INF } else { -INF };
    }
    let opp = color.opponent();
    let phase = get_phase(board, color);
    let w = weights(phase);

    let our_mills = closed_mills(board, color);
    let opp_mills = closed_mills(board, opp);
    let blocked = blocked_count(board, opp);
    let piece_diff = board.count(color) as i64 - board.count(opp) as i64;
    let our_two = two_configs(board, color);
    let opp_two = two_configs(board, opp);
    let our_dbl = double_mills(board, color);
    let opp_dbl = double_mills(board, opp);
    let win_cfg = win_config(board, opp);
    let our_mob = mobility(board, color);
    let opp_mob = mobility(board, opp);
    let our_thr = mill_threats(board, color);
    let opp_thr = mill_threats(board, opp);
    let our_pos = position_value(board, color);
    let opp_pos = position_value(board, opp);
    let our_cycle = mill_cycle_ready(board, color);
    let opp_cycle = mill_cycle_ready(board, opp);
    let our_fork = fork_threats(board, color);
    let opp_fork = fork_threats(board, opp);
    let our_herd = encirclement(board, color);
    let opp_herd = encirclement(board, opp);
    let our_squeeze = squeeze_count(board, opp); // opponent near-blocked (good)
    let opp_squeeze = squeeze_count(board, color); // own near-blocked (bad)
    let our_wrap = mill_wrapping_pressure(board, color);
    let opp_wrap = mill_wrapping_pressure(board, opp);
    let fly_asym = fly_asymmetry(board, color);
    let our_dom = open_mill_domination(board, color);
    let opp_dom = open_mill_domination(board, opp);

    w[0] * (our_mills - opp_mills)
        + w[1] * blocked
        + w[2] * piece_diff
        + w[3] * (our_two - opp_two)
        + w[4] * (our_dbl - opp_dbl)
        + w[5] * win_cfg
        + mob_weight(phase) * (our_mob - opp_mob)
        + threat_weight(phase) * (our_thr - opp_thr)
        + 4 * (our_pos - opp_pos)
        + cycle_weight(phase) * (our_cycle - opp_cycle)
        + fork_weight(phase) * (our_fork - opp_fork)
        + herd_weight(phase) * (our_herd - opp_herd)
        + near_blocked_weight(phase) * (our_squeeze - opp_squeeze)
        + wrap_weight(phase) * (our_wrap - opp_wrap)
        + fly_asym_weight(phase) * fly_asym
        + domination_weight(phase) * (our_dom - opp_dom)
}
