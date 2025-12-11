use crate::domain::models::{BoardState, Player};
use crate::domain::services::PlayerStrategy;
use crate::infrastructure::persistence::{BitBoard, BitBoardState, WinningMasks}; // WinningMasks needed for strategic value? Yes.
use crate::infrastructure::symmetries::SymmetryHandler;
use rayon::prelude::*;
use std::sync::Arc;

pub mod transposition;
use transposition::{Flag, LockFreeTT};

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MinimaxBot {
    transposition_table: Arc<LockFreeTT>,
    zobrist_keys: Vec<[u64; 2]>,
    symmetries: Option<SymmetryHandler>,
    max_depth: usize,
    strategic_values: Vec<usize>,
    killer_moves: Vec<[AtomicUsize; 2]>,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        let mut killer_moves = Vec::with_capacity(max_depth + 1);
        for _ in 0..=max_depth {
            killer_moves.push([AtomicUsize::new(usize::MAX), AtomicUsize::new(usize::MAX)]);
        }

        MinimaxBot {
            transposition_table: Arc::new(LockFreeTT::new(64)),
            zobrist_keys: Vec::new(),
            symmetries: None,
            max_depth,
            strategic_values: Vec::new(),
            killer_moves,
        }
    }

    fn store_killer(&self, depth: usize, move_idx: usize) {
        if depth >= self.killer_moves.len() {
            return;
        }
        let k0 = self.killer_moves[depth][0].load(Ordering::Relaxed);
        if k0 != move_idx {
            // Shift 0 to 1
            self.killer_moves[depth][1].store(k0, Ordering::Relaxed);
            // Store new at 0
            self.killer_moves[depth][0].store(move_idx, Ordering::Relaxed);
        }
    }

    fn ensure_initialized(&mut self, board: &BitBoardState) {
        if self.zobrist_keys.len() < board.total_cells() {
            let mut rng_state: u64 = 0xDEADBEEF + board.total_cells() as u64;
            let mut next_rand = || -> u64 {
                // simple LCG
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
            WinningMasks::Small { map_offsets, .. } => map_offsets
                .get(index)
                .map_or(0, |&(_, count)| count as usize),
            WinningMasks::Medium { map_offsets, .. } => map_offsets
                .get(index)
                .map_or(0, |&(_, count)| count as usize),
            WinningMasks::Large { map_offsets, .. } => map_offsets
                .get(index)
                .map_or(0, |&(_, count)| count as usize),
        }
    }

    fn calculate_zobrist_hash(&self, board: &BitBoardState) -> u64 {
        let mut h = 0;
        for cell_idx in 0..board.total_cells() {
            if let Some(p) = board.get_cell(cell_idx) {
                let p_idx = match p {
                    Player::X => 0,
                    Player::O => 1,
                };
                h ^= self.zobrist_keys[cell_idx][p_idx];
            }
        }
        h
    }

    fn get_sorted_moves_into(
        &self,
        board: &BitBoardState,
        buffer: &mut [usize],
        best_move_hint: Option<usize>,
        depth: usize,
        current_player: Player,
    ) -> usize {
        let mut count = 0;
        for idx in 0..board.total_cells() {
            if board.get_cell(idx).is_none() {
                if count < buffer.len() {
                    buffer[count] = idx;
                    count += 1;
                }
            }
        }

        let valid_moves = &mut buffer[0..count];

        let k0 = if depth < self.killer_moves.len() {
            self.killer_moves[depth][0].load(Ordering::Relaxed)
        } else {
            usize::MAX
        };
        let k1 = if depth < self.killer_moves.len() {
            self.killer_moves[depth][1].load(Ordering::Relaxed)
        } else {
            usize::MAX
        };
        let killer0 = if k0 == usize::MAX { None } else { Some(k0) };
        let killer1 = if k1 == usize::MAX { None } else { Some(k1) };

        let me = if current_player == Player::X {
            &board.p1
        } else {
            &board.p2
        };
        let op = if current_player == Player::X {
            &board.p2
        } else {
            &board.p1
        };

        // Sort in place using heuristics
        valid_moves.sort_unstable_by(|&a, &b| {
            if Some(a) == best_move_hint {
                return std::cmp::Ordering::Less;
            }
            if Some(b) == best_move_hint {
                return std::cmp::Ordering::Greater;
            }

            // Winning Move Check
            let a_wins = me.check_win_with_added_piece(&board.winning_masks, a);
            let b_wins = me.check_win_with_added_piece(&board.winning_masks, b);
            if a_wins != b_wins {
                return if a_wins {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }

            // Forced Move Check
            let a_blocks = op.check_win_with_added_piece(&board.winning_masks, a);
            let b_blocks = op.check_win_with_added_piece(&board.winning_masks, b);
            if a_blocks != b_blocks {
                return if a_blocks {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }

            let is_k_a = Some(a) == killer0 || Some(a) == killer1;
            let is_k_b = Some(b) == killer0 || Some(b) == killer1;
            if is_k_a != is_k_b {
                return if is_k_a {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }

            self.strategic_values[b].cmp(&self.strategic_values[a])
        });

        count
    }

    #[inline]
    fn get_line_score(x: u32, o: u32) -> i32 {
        if o == 0 {
            if x == 2 {
                return 10;
            }
            if x == 1 {
                return 1;
            }
        } else if x == 0 {
            if o == 2 {
                return -10;
            }
            if o == 1 {
                return -1;
            }
        }
        0
    }

    fn calculate_score_delta(&self, board: &BitBoardState, index: usize, player: Player) -> i32 {
        let mut delta = 0;
        match &*board.winning_masks {
            WinningMasks::Small {
                cell_mask_lookup, ..
            } => {
                let p1 = match board.p1 {
                    BitBoard::Small(b) => b,
                    _ => 0,
                };
                let p2 = match board.p2 {
                    BitBoard::Small(b) => b,
                    _ => 0,
                };
                if let Some(masks) = cell_mask_lookup.get(index) {
                    for &m in masks {
                        let x = (p1 & m).count_ones();
                        let o = (p2 & m).count_ones();
                        let old_score = Self::get_line_score(x, o);

                        let (nx, no) = match player {
                            Player::X => (x + 1, o),
                            Player::O => (x, o + 1),
                        };
                        let new_score = Self::get_line_score(nx, no);
                        delta += new_score - old_score;
                    }
                }
            }
            WinningMasks::Medium {
                cell_mask_lookup, ..
            } => {
                let p1 = match board.p1 {
                    BitBoard::Medium(b) => b,
                    _ => 0,
                };
                let p2 = match board.p2 {
                    BitBoard::Medium(b) => b,
                    _ => 0,
                };
                if let Some(masks) = cell_mask_lookup.get(index) {
                    for &m in masks {
                        let x = (p1 & m).count_ones();
                        let o = (p2 & m).count_ones();
                        let old_score = Self::get_line_score(x, o);

                        let (nx, no) = match player {
                            Player::X => (x + 1, o),
                            Player::O => (x, o + 1),
                        };
                        let new_score = Self::get_line_score(nx, no);
                        delta += new_score - old_score;
                    }
                }
            }
            WinningMasks::Large {
                masks,
                map_flat,
                map_offsets,
            } => {
                if index < map_offsets.len() {
                    let (start, count) = map_offsets[index];
                    let range = start as usize..(start + count) as usize;
                    for &i in &map_flat[range] {
                        let mask_chunks = &masks[i];
                        let mut x = 0;
                        let mut o = 0;
                        match (&board.p1, &board.p2) {
                            (BitBoard::Large(v1), BitBoard::Large(v2)) => {
                                for (k, m) in mask_chunks.iter().enumerate() {
                                    if k < v1.len() {
                                        x += (v1[k] & m).count_ones();
                                    }
                                    if k < v2.len() {
                                        o += (v2[k] & m).count_ones();
                                    }
                                }
                            }
                            _ => {}
                        }
                        let old_score = Self::get_line_score(x, o);
                        let (nx, no) = match player {
                            Player::X => (x + 1, o),
                            Player::O => (x, o + 1),
                        };
                        let new_score = Self::get_line_score(nx, no);
                        delta += new_score - old_score;
                    }
                }
            }
        }
        delta
    }

    fn minimax(
        &self,
        board: &mut BitBoardState,
        depth: usize,
        current_player: Player,
        mut alpha: i32,
        mut beta: i32,
        current_hash: u64,
        current_score: i32,
    ) -> i32 {
        let alpha_orig = alpha;

        let remaining_depth = if self.max_depth > depth {
            (self.max_depth - depth) as u8
        } else {
            0
        };

        let mut best_move_hint = None;

        if let Some((score, entry_depth, flag, best_move)) =
            self.transposition_table.get(current_hash)
        {
            if entry_depth >= remaining_depth {
                match flag {
                    Flag::Exact => return score,
                    Flag::LowerBound => alpha = alpha.max(score),
                    Flag::UpperBound => beta = beta.min(score),
                }
                if alpha >= beta {
                    return score;
                }
            }
            best_move_hint = best_move.map(|idx| idx as usize);
        }

        // REMOVED redundant board.check_win() here. Caller guarantees validity or handles win state in loop.

        if board.is_full() {
            return 0; // Draw
        }

        if depth >= self.max_depth {
            return current_score;
        }

        let opponent = current_player.opponent();

        // Use a stack buffer for moves.
        let mut moves_buf = [0usize; 256];
        let count = self.get_sorted_moves_into(
            board,
            &mut moves_buf,
            best_move_hint,
            depth,
            current_player,
        );
        let moves = &moves_buf[0..count];

        let mut best_val = match current_player {
            Player::X => i32::MIN,
            Player::O => i32::MAX,
        };

        // Track best move for TT
        let mut best_move_idx = best_move_hint;

        let p_idx = match current_player {
            Player::X => 0,
            Player::O => 1,
        };

        for &idx in moves {
            // Incremental score
            let score_delta = self.calculate_score_delta(board, idx, current_player);
            let next_score = current_score + score_delta;

            board.set_cell(idx, current_player).unwrap();

            let new_hash = current_hash ^ self.zobrist_keys[idx][p_idx];

            let win = board.p1.check_win_at(&board.winning_masks, idx)
                || board.p2.check_win_at(&board.winning_masks, idx);

            let val = if win {
                match current_player {
                    Player::X => 1000 - depth as i32,
                    Player::O => -1000 + depth as i32,
                }
            } else {
                self.minimax(
                    board,
                    depth + 1,
                    opponent,
                    alpha,
                    beta,
                    new_hash,
                    next_score,
                )
            };

            board.clear_cell(idx);

            match current_player {
                Player::X => {
                    if val > best_val {
                        best_val = val;
                        best_move_idx = Some(idx);
                    }
                    alpha = alpha.max(val);
                }
                Player::O => {
                    if val < best_val {
                        best_val = val;
                        best_move_idx = Some(idx);
                    }
                    beta = beta.min(val);
                }
            }

            if beta <= alpha {
                self.store_killer(depth, idx);
                break;
            }
        }

        // If no moves were made (should be covered by is_full), best_val remains init.
        // Logic for bounds adjustment
        if (current_player == Player::X && best_val == i32::MIN)
            || (current_player == Player::O && best_val == i32::MAX)
        {
            // This happens if no moves are valid, which implies full board, but we checked is_full.
            // Or if we pruned everything? But we loop at least once if not full.
            // If moves is empty, loop doesn't run.
            // If is_full check didn't catch it for some reason?
            // Let's assume best_val is updated or 0 if empty?
            // Actually, if loop doesn't run, best_val is MIN/MAX.
            // is_full() checks count.
            best_val = 0;
        }

        let flag = if best_val <= alpha_orig {
            Flag::UpperBound
        } else if best_val >= beta {
            Flag::LowerBound
        } else {
            Flag::Exact
        };

        let best_move_u16 = best_move_idx.map(|i| i as u16);

        self.transposition_table.store(
            current_hash,
            best_val,
            remaining_depth,
            flag,
            best_move_u16,
        );

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

        let mut moves_buf = [0usize; 256];
        let count = self.get_sorted_moves_into(board, &mut moves_buf, None, 0, player);
        let available_moves = &moves_buf[0..count];

        if available_moves.is_empty() {
            return None;
        }

        // --- FIRST MOVE SEQUENTIAL SEARCH ---
        let first_move = available_moves[0];
        let mut work_board = board.clone();

        let initial_hash = self.calculate_zobrist_hash(&work_board);
        let initial_score = self.evaluate(board);

        let p_idx = match player {
            Player::X => 0,
            Player::O => 1,
        };

        let first_delta = self.calculate_score_delta(board, first_move, player);
        let first_next_score = initial_score + first_delta;

        work_board.set_cell(first_move, player).unwrap();

        let next_hash = initial_hash ^ self.zobrist_keys[first_move][p_idx];

        let win = work_board
            .p1
            .check_win_at(&work_board.winning_masks, first_move)
            || work_board
                .p2
                .check_win_at(&work_board.winning_masks, first_move);

        let first_score = if win {
            match player {
                Player::X => 1000,
                Player::O => -1000,
            }
        } else {
            self.minimax(
                &mut work_board,
                0, // depth 0 for minimax call (actually depth 1 of game)
                player.opponent(),
                i32::MIN + 1,
                i32::MAX - 1,
                next_hash,
                first_next_score,
            )
        };

        let (alpha, beta) = match player {
            Player::X => (first_score, i32::MAX - 1),
            Player::O => (i32::MIN + 1, first_score),
        };

        if available_moves.len() == 1 {
            return Some(first_move);
        }

        // --- REMAINING MOVES PARALLEL SEARCH ---
        // Need to collect moves to a Vec for par_iter, or use array slice?
        // available_moves is a slice. par_iter works on slice.

        let best_move_entry = available_moves[1..]
            .par_iter()
            .map(|&mv| {
                let mut work_board = board.clone();

                // Recompute hash for each thread/task? Or pass efficient delta?
                // Just compute from scratch or clone?
                // Since we clone board, hash is same as initial.
                // We can reuse `initial_hash`.
                let delta = self.calculate_score_delta(board, mv, player);
                let next_score = initial_score + delta;

                work_board.set_cell(mv, player).unwrap();
                let next_hash = initial_hash ^ self.zobrist_keys[mv][p_idx];

                let win = work_board.p1.check_win_at(&work_board.winning_masks, mv)
                    || work_board.p2.check_win_at(&work_board.winning_masks, mv);

                let score = if win {
                    match player {
                        Player::X => 1000,
                        Player::O => -1000,
                    }
                } else {
                    self.minimax(
                        &mut work_board,
                        0,
                        player.opponent(),
                        alpha,
                        beta,
                        next_hash,
                        next_score,
                    )
                };

                (mv, score)
            })
            .max_by(|a, b| match player {
                Player::X => a.1.cmp(&b.1),
                Player::O => b.1.cmp(&a.1),
            });

        if let Some((best_parallel_move, best_parallel_score)) = best_move_entry {
            match player {
                Player::X => {
                    if best_parallel_score > first_score {
                        return Some(best_parallel_move);
                    }
                }
                Player::O => {
                    if best_parallel_score < first_score {
                        return Some(best_parallel_move);
                    }
                }
            }
        }

        Some(first_move)
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
        let board = BitBoardState::new(2); // 3x3
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

        // Board:
        // X . .  (0, 1, 2)
        // O O .  (3, 4, 5)
        // . . .  (6, 7, 8)

        let mut bot = MinimaxBot::new(5);
        let best_move = bot.get_best_move(&board, Player::X);

        assert_eq!(best_move, Some(5));
    }
}
