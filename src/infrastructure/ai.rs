use crate::domain::models::{BoardState, Player};
use crate::domain::services::PlayerStrategy;
use crate::infrastructure::persistence::{BitBoard, BitBoardState, WinningMasks};
use crate::infrastructure::symmetries::SymmetryHandler;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[derive(Default, Debug)]
pub struct SearchStats {
    pub nodes_searched: AtomicUsize,
    pub tt_hits: AtomicUsize,
    pub tt_exact_hits: AtomicUsize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Flag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
    None = 3,
}

impl Flag {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Flag::Exact,
            1 => Flag::LowerBound,
            2 => Flag::UpperBound,
            _ => Flag::None,
        }
    }
}

// Packed atomic entry
// Word 1: Key (Full u64 hash)
// Word 2: Data packed as:
//   - Score: i16 (bits 0-15) - rebased to u16
//   - Depth: u8  (bits 16-23)
//   - Flag:  u8  (bits 24-25)
//   - BestMove: u16 (bits 26-41) (0xFFFF means None)
#[derive(Default)]
struct AtomicTranspositionEntry {
    key: AtomicU64,
    data: AtomicU64,
}

struct LockFreeTT {
    entries: Vec<AtomicTranspositionEntry>,
    mask: usize,
}

impl LockFreeTT {
    fn new(size_mb: usize) -> Self {
        // Each entry is 16 bytes.
        let num_entries = (size_mb * 1024 * 1024) / 16;
        let size = num_entries.next_power_of_two();

        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(AtomicTranspositionEntry::default());
        }

        Self {
            entries,
            mask: size - 1,
        }
    }

    fn get(&self, hash: u64) -> Option<(i32, u8, Flag, Option<usize>)> {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let entry_key = entry.key.load(Ordering::Relaxed);
        if entry_key != hash {
            return None;
        }

        let data = entry.data.load(Ordering::Relaxed);

        // Unpack
        let score_u16 = (data & 0xFFFF) as u16;
        let score = (score_u16 as i32) - 10000; // Offset back

        let depth = ((data >> 16) & 0xFF) as u8;
        let flag_u8 = ((data >> 24) & 0x3) as u8;
        let best_move_u16 = ((data >> 26) & 0xFFFF) as u16;

        let best_move = if best_move_u16 == 0xFFFF {
            None
        } else {
            Some(best_move_u16 as usize)
        };

        Some((score, depth, Flag::from_u8(flag_u8), best_move))
    }

    fn store(&self, hash: u64, score: i32, depth: u8, flag: Flag, best_move: Option<usize>) {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        // Pack
        // Score: -10000 to 10000. Add 10000 to make it u16 compatible (0 to 20000)
        let score_rebased = (score + 10000).clamp(0, 65535) as u64;
        let depth_bits = (depth as u64) << 16;
        let flag_bits = (flag as u64) << 24;
        let move_bits = match best_move {
            Some(m) => (m as u64) << 26,
            None => 0xFFFF << 26,
        };

        let new_data = score_rebased | depth_bits | flag_bits | move_bits;

        // Simple replacement policy: Always replace if depth is greater or equal
        // Or if the slot is empty (key mismatch implicitly handled by overwrite)

        // For strict correctness in a race, we might want to check the existing depth,
        // but for a game engine concurrent access, "racy" overwrite is often acceptable and faster.
        // We will do a relaxed load to check depth to avoid thrashing valuable deep nodes with shallow ones.

        let existing_data = entry.data.load(Ordering::Relaxed);
        let existing_depth = ((existing_data >> 16) & 0xFF) as u8;

        // If new entry is deeper or equal, OR if the existing entry is from a different position (collision/eviction)
        // Actually, if it's a DIFFERENT position, we usually prefer to keep the one with higher depth,
        // or just overwrite. Simple strategy: Depth-preferred.

        let existing_key = entry.key.load(Ordering::Relaxed);

        // Overwrite if:
        // 1. Slot is empty (key == 0, though hash can be 0, but unlikely to impact much)
        // 2. Different key (Collision) -> usually overwrite or bucket logic. We'll overwrite.
        // 3. Same key, new depth >= existing depth.

        if existing_key != hash || depth >= existing_depth {
            entry.key.store(hash, Ordering::Relaxed);
            entry.data.store(new_data, Ordering::Relaxed);
        }
    }
}

pub struct MinimaxBot {
    transposition_table: LockFreeTT,
    zobrist_keys: Vec<[u64; 2]>,
    symmetries: Option<SymmetryHandler>,
    max_depth: usize,
    strategic_values: Vec<usize>,
    pub stats: SearchStats,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        MinimaxBot {
            transposition_table: LockFreeTT::new(256), // 256MB default
            zobrist_keys: Vec::new(),
            symmetries: None,
            max_depth,
            strategic_values: Vec::new(),
            stats: SearchStats::default(),
        }
    }

    fn ensure_initialized(&mut self, board: &BitBoardState) {
        if self.zobrist_keys.len() < board.total_cells() {
            let mut rng_state: u64 = 0xDEADBEEF + board.total_cells() as u64;
            let mut next_rand = || -> u64 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                rng_state
            };
            self.zobrist_keys
                .resize_with(board.total_cells(), || [next_rand(), next_rand()]);
        }

        if self.symmetries.is_none() {
            self.symmetries = Some(SymmetryHandler::new(board.dimension, board.side));
        }

        if self.strategic_values.len() < board.total_cells() {
            self.strategic_values = (0..board.total_cells())
                .map(|i| self.get_strategic_value(board, i))
                .collect();
        }
    }

    fn get_strategic_value(&self, board: &BitBoardState, index: usize) -> usize {
        match &*board.winning_masks {
            WinningMasks::Small { map, .. } => map.get(index).map_or(0, |v| v.len()),
            WinningMasks::Medium { map, .. } => map.get(index).map_or(0, |v| v.len()),
            WinningMasks::Large { map, .. } => map.get(index).map_or(0, |v| v.len()),
        }
    }

    fn initialize_rolling_hashes(&self, board: &BitBoardState) -> Vec<u64> {
        let handler = self.symmetries.as_ref().unwrap();
        let mut hashes = vec![0; handler.maps.len()];

        for (sym_idx, map) in handler.maps.iter().enumerate() {
            let mut h = 0;
            for cell_idx in 0..board.total_cells() {
                if let Some(p) = board.get_cell(cell_idx) {
                    let mapped_cell = map[cell_idx];
                    let p_idx = match p {
                        Player::X => 0,
                        Player::O => 1,
                    };
                    h ^= self.zobrist_keys[mapped_cell][p_idx];
                }
            }
            hashes[sym_idx] = h;
        }
        hashes
    }

    fn update_hashes(&self, hashes: &mut [u64], cell_idx: usize, player: Player) {
        let handler = self.symmetries.as_ref().unwrap();
        let p_idx = match player {
            Player::X => 0,
            Player::O => 1,
        };

        for (sym_idx, map) in handler.maps.iter().enumerate() {
            let mapped_cell = map[cell_idx];
            hashes[sym_idx] ^= self.zobrist_keys[mapped_cell][p_idx];
        }
    }

    fn get_canonical_hash_fast(&self, hashes: &[u64]) -> u64 {
        *hashes.iter().min().unwrap_or(&0)
    }

    fn get_sorted_moves(&self, board: &BitBoardState) -> Vec<usize> {
        let mut moves: Vec<usize> = (0..board.total_cells())
            .filter(|&idx| board.get_cell(idx).is_none())
            .collect();

        moves.sort_by(|&a, &b| self.strategic_values[b].cmp(&self.strategic_values[a]));
        moves
    }

    fn minimax(
        &self,
        board: &mut BitBoardState,
        depth: usize,
        current_player: Player,
        mut alpha: i32,
        mut beta: i32,
        rolling_hashes: &mut Vec<u64>,
    ) -> i32 {
        let alpha_orig = alpha;

        // OPTIMIZATION: Only use full symmetry canonicalization at shallow depths.
        // At deeper depths, just use the first hash (identity).
        let canonical_hash = if depth < 4 {
            self.get_canonical_hash_fast(rolling_hashes)
        } else {
            rolling_hashes[0]
        };

        let remaining_depth = if self.max_depth > depth {
            (self.max_depth - depth) as u8
        } else {
            0
        };

        if let Some((score, entry_depth, flag, _)) = self.transposition_table.get(canonical_hash) {
            if entry_depth >= remaining_depth {
                self.stats.tt_hits.fetch_add(1, Ordering::Relaxed);

                match flag {
                    Flag::Exact => {
                        self.stats.tt_exact_hits.fetch_add(1, Ordering::Relaxed);
                        return score;
                    }
                    Flag::LowerBound => alpha = alpha.max(score),
                    Flag::UpperBound => beta = beta.min(score),
                    Flag::None => {}
                }

                if alpha >= beta {
                    return score;
                }
            }
        }

        let mut tt_move = None;
        if let Some((_, _, _, best_move)) = self.transposition_table.get(canonical_hash) {
            tt_move = best_move;
        }

        if let Some(winner) = board.check_win() {
            return match winner {
                Player::X => 1000 - depth as i32,
                Player::O => -1000 + depth as i32,
            };
        }

        if board.is_full() && board.check_win().is_none() {
            return 0;
        }

        if depth >= self.max_depth {
            return self.evaluate(board);
        }

        self.stats.nodes_searched.fetch_add(1, Ordering::Relaxed);

        let opponent = current_player.opponent();
        let mut moves = self.get_sorted_moves(board);

        if let Some(tm) = tt_move {
            if let Some(pos) = moves.iter().position(|&m| m == tm) {
                moves.swap(0, pos);
            }
        }

        let mut best_val = match current_player {
            Player::X => i32::MIN,
            Player::O => i32::MAX,
        };
        let mut best_move = None;

        for idx in moves {
            board.set_cell(idx, current_player).unwrap();

            self.update_hashes(rolling_hashes, idx, current_player);

            let win = board.p1.check_win_at(&board.winning_masks, idx)
                || board.p2.check_win_at(&board.winning_masks, idx);

            let val = if win {
                match current_player {
                    Player::X => 1000 - depth as i32,
                    Player::O => -1000 + depth as i32,
                }
            } else {
                self.minimax(board, depth + 1, opponent, alpha, beta, rolling_hashes)
            };

            board.clear_cell(idx);
            self.update_hashes(rolling_hashes, idx, current_player);

            match current_player {
                Player::X => {
                    if val > best_val {
                        best_val = val;
                        best_move = Some(idx);
                    }
                    alpha = alpha.max(val);
                }
                Player::O => {
                    if val < best_val {
                        best_val = val;
                        best_move = Some(idx);
                    }
                    beta = beta.min(val);
                }
            }

            if beta <= alpha {
                break;
            }
        }

        if (current_player == Player::X && best_val == i32::MIN)
            || (current_player == Player::O && best_val == i32::MAX)
        {
            best_val = 0;
        }

        let flag = if best_val <= alpha_orig {
            Flag::UpperBound
        } else if best_val >= beta {
            Flag::LowerBound
        } else {
            Flag::Exact
        };

        self.transposition_table
            .store(canonical_hash, best_val, remaining_depth, flag, best_move);

        best_val
    }

    fn evaluate(&self, board: &BitBoardState) -> i32 {
        let mut score = 0;
        match &*board.winning_masks {
            WinningMasks::Small { masks, .. } => {
                let p1 = match board.p1 {
                    BitBoard::Small(b) => b,
                    _ => 0,
                };
                let p2 = match board.p2 {
                    BitBoard::Small(b) => b,
                    _ => 0,
                };
                for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 {
                            score += 10;
                        } else if x == 1 {
                            score += 1;
                        }
                    } else if x == 0 {
                        if o == 2 {
                            score -= 10;
                        } else if o == 1 {
                            score -= 1;
                        }
                    }
                }
            }
            WinningMasks::Medium { masks, .. } => {
                let p1 = match board.p1 {
                    BitBoard::Medium(b) => b,
                    _ => 0,
                };
                let p2 = match board.p2 {
                    BitBoard::Medium(b) => b,
                    _ => 0,
                };
                for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 {
                            score += 10;
                        } else if x == 1 {
                            score += 1;
                        }
                    } else if x == 0 {
                        if o == 2 {
                            score -= 10;
                        } else if o == 1 {
                            score -= 1;
                        }
                    }
                }
            }
            _ => {}
        }
        score
    }
}

impl PlayerStrategy<BitBoardState> for MinimaxBot {
    fn get_best_move(&mut self, board: &BitBoardState, player: Player) -> Option<usize> {
        self.ensure_initialized(board);

        let mut available_moves = self.get_sorted_moves(board);
        if available_moves.is_empty() {
            return None;
        }

        if available_moves.len() == 1 {
            return Some(available_moves[0]);
        }

        let original_max_depth = self.max_depth;

        let mut current_best_move_entry: Option<(usize, i32)> = None;

        for d in 1..=original_max_depth {
            self.max_depth = d;

            if let Some((best_mv, _)) = current_best_move_entry {
                if let Some(pos) = available_moves.iter().position(|&m| m == best_mv) {
                    available_moves.swap(0, pos);
                }
            }

            // To properly parallelize with the new lock-free structure (which uses Interior Mutability via Atomics),
            // we can share the reference to the table.

            // Note: self.transposition_table is now Sync (because Atomics are Sync).
            // So we can reference it directly.

            let best_move_entry = available_moves
                .par_iter()
                .map(|&mv| {
                    let mut work_board = board.clone();
                    let mut rolling_hashes = self.initialize_rolling_hashes(&work_board);

                    work_board.set_cell(mv, player).unwrap();
                    self.update_hashes(&mut rolling_hashes, mv, player);

                    let score = self.minimax(
                        &mut work_board,
                        0,
                        player.opponent(),
                        i32::MIN + 1,
                        i32::MAX - 1,
                        &mut rolling_hashes,
                    );

                    (mv, score)
                })
                .max_by(|a, b| match player {
                    Player::X => a.1.cmp(&b.1),
                    Player::O => b.1.cmp(&a.1),
                });

            current_best_move_entry = best_move_entry;

            println!(
                "ID Depth: {}, Nodes: {}, TT Hits: {}, Exact Hits: {}",
                d,
                self.stats.nodes_searched.load(Ordering::Relaxed),
                self.stats.tt_hits.load(Ordering::Relaxed),
                self.stats.tt_exact_hits.load(Ordering::Relaxed)
            );
        }

        self.max_depth = original_max_depth;

        current_best_move_entry.map(|(mv, _)| mv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::Player;
    use crate::domain::services::PlayerStrategy;
    use crate::infrastructure::persistence::BitBoardState;

    #[test]
    fn test_minimax_multiprocess_smoke_test() {
        let board = BitBoardState::new(2);
        let mut bot = MinimaxBot::new(9);
        let best_move = bot.get_best_move(&board, Player::X);
        assert!(best_move.is_some());
    }

    #[test]
    fn test_minimax_blocks_win() {
        let mut board = BitBoardState::new(2);
        board.set_cell(0, Player::X).unwrap();
        board.set_cell(3, Player::O).unwrap();
        board.set_cell(4, Player::O).unwrap();

        let mut bot = MinimaxBot::new(5);
        let best_move = bot.get_best_move(&board, Player::X);

        assert_eq!(best_move, Some(5));
    }
}
