use crate::domain::models::{Player, BoardState};
use crate::domain::services::PlayerStrategy;
use crate::infrastructure::persistence::{BitBoardState, BitBoard, WinningMasks};
use crate::infrastructure::symmetries::SymmetryHandler;
use dashmap::DashMap;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Flag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
struct TranspositionEntry {
    score: i32,
    depth: u8,
    flag: Flag,
}

pub struct MinimaxBot {
    transposition_table: DashMap<u64, TranspositionEntry>,
    zobrist_keys: Vec<[u64; 2]>,
    symmetries: Option<SymmetryHandler>,
    max_depth: usize,
    strategic_values: Vec<usize>,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        MinimaxBot {
            transposition_table: DashMap::new(),
            zobrist_keys: Vec::new(),
            symmetries: None,
            max_depth,
            strategic_values: Vec::new(),
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

        let canonical_hash = self.get_canonical_hash_fast(rolling_hashes);
        let remaining_depth = if self.max_depth > depth {
            (self.max_depth - depth) as u8
        } else {
            0
        };

        if let Some(entry) = self.transposition_table.get(&canonical_hash) {
            if entry.depth >= remaining_depth {
                match entry.flag {
                    Flag::Exact => return entry.score,
                    Flag::LowerBound => alpha = alpha.max(entry.score),
                    Flag::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        // board.check_draw() is expensive? No, check status via board logic?
        // BitBoardState implements check_draw via is_full and check_win
        // Wait, minimax uses `check_draw` but logic here separates Win check from Full check if possible for speed.
        // Actually the original AI checked draw via is_full only if no win?
        // checking is_full is fast.
        
        // Optimisation: check win at last move? The original code did that deep in loop.
        // At top of minimax, we just check if game over?
        // Original code: 
        // if board.check_draw() return 0
        // check_draw() uses `combined.is_full() && check_win().is_none()`. 
        // This calculates check_win().
        
        // We can trust BoardState::check_win which delegates to BitBoard.
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

        let opponent = current_player.opponent(); 
        let moves = self.get_sorted_moves(board);

        let mut best_val = match current_player {
            Player::X => i32::MIN,
            Player::O => i32::MAX,
        };

        for idx in moves {
            board.set_cell(idx, current_player).unwrap(); // Use convenience helper or set_cell? 
            // BitBoardState doesn't have 'make_move', it has 'set_cell' from BoardState trait.
            // Oh right, I need to use trait methods or impl methods.
            
            self.update_hashes(rolling_hashes, idx, current_player);

            // Win check optimization
            // BitBoardState methods needed: check_win_at?
            // Existing BoardState trait check_win is global.
            // But BitBoard implementation has 'check_win_at'. 
            // We can access it because we depend on BitBoardState struct directly!
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

            board.clear_cell(idx); // Trait method
            self.update_hashes(rolling_hashes, idx, current_player);

            match current_player {
                Player::X => {
                    best_val = best_val.max(val);
                    alpha = alpha.max(val);
                }
                Player::O => {
                    best_val = best_val.min(val);
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
            // Stalemate? Or no moves left? 'is_full' checked earlier.
            best_val = 0;
        }

        let flag = if best_val <= alpha_orig {
            Flag::UpperBound
        } else if best_val >= beta {
            Flag::LowerBound
        } else {
            Flag::Exact
        };

        let entry = TranspositionEntry {
            score: best_val,
            depth: remaining_depth,
            flag,
        };
        self.transposition_table.insert(canonical_hash, entry);

        best_val
    }

    fn evaluate(&self, board: &BitBoardState) -> i32 {
         let mut score = 0;
         match &*board.winning_masks {
            WinningMasks::Small { masks, .. } => {
                let p1 = match board.p1 { BitBoard::Small(b) => b, _ => 0 };
                let p2 = match board.p2 { BitBoard::Small(b) => b, _ => 0 };
                for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 { score += 10; } else if x == 1 { score += 1; }
                    } else if x == 0 {
                        if o == 2 { score -= 10; } else if o == 1 { score -= 1; }
                    }
                }
            }
            WinningMasks::Medium { masks, .. } => {
                 let p1 = match board.p1 { BitBoard::Medium(b) => b, _ => 0 };
                 let p2 = match board.p2 { BitBoard::Medium(b) => b, _ => 0 };
                 for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 { score += 10; } else if x == 1 { score += 1; }
                    } else if x == 0 {
                        if o == 2 { score -= 10; } else if o == 1 { score -= 1; }
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

        let available_moves = self.get_sorted_moves(board);
        if available_moves.is_empty() {
            return None;
        }

        // --- FIRST MOVE SEQUENTIAL SEARCH ---
        // Search the first move (best according to heuristics) sequentially
        // to establish a good alpha/beta bound.
        let first_move = available_moves[0];
        let mut work_board = board.clone();
        let mut rolling_hashes = self.initialize_rolling_hashes(&work_board);
        
        work_board.set_cell(first_move, player).unwrap();
        self.update_hashes(&mut rolling_hashes, first_move, player);

        let first_score = self.minimax(
            &mut work_board,
            0,
            player.opponent(),
            i32::MIN + 1,
            i32::MAX - 1,
            &mut rolling_hashes,
        );

        let (alpha, beta) = match player {
            Player::X => (first_score, i32::MAX - 1),
            Player::O => (i32::MIN + 1, first_score),
        };

        if available_moves.len() == 1 {
            return Some(first_move);
        }

        // --- REMAINING MOVES PARALLEL SEARCH ---
        // Search the rest in parallel with stricter bounds.
        // Note: We cannot easily update alpha/beta across threads dynamically without
        // complex synchronization (atomics), but the first move usually provides
        // the most significant cut.
        
        let best_move_entry = available_moves[1..].par_iter().map(|&mv| {
            let mut work_board = board.clone();
            let mut rolling_hashes = self.initialize_rolling_hashes(&work_board);

            work_board.set_cell(mv, player).unwrap();
            self.update_hashes(&mut rolling_hashes, mv, player);

            let score = self.minimax(
                &mut work_board,
                0,
                player.opponent(),
                alpha,
                beta,
                &mut rolling_hashes,
            );
            
            (mv, score)
        }).max_by(|a, b| {
            match player {
                Player::X => a.1.cmp(&b.1),
                Player::O => b.1.cmp(&a.1),
            }
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
    use crate::infrastructure::persistence::BitBoardState;
    use crate::domain::models::Player;
    use crate::domain::services::PlayerStrategy;

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
