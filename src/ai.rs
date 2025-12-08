use crate::{HyperBoard, Player};
use std::collections::HashMap;

pub struct MinimaxBot {
    transposition_table: HashMap<u64, i32>,
    zobrist_keys: Vec<[u64; 2]>, // [cell_idx][player_idx]
    max_depth: usize,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        MinimaxBot {
            transposition_table: HashMap::new(),
            zobrist_keys: Vec::new(),
            max_depth,
        }
    }
    
    fn ensure_zobrist_initialized(&mut self, size: usize) {
        if self.zobrist_keys.len() < size {
             let mut rng_state: u64 = 0xDEADBEEF + size as u64; 
             // Simple LCG
             let mut next_rand = || -> u64 {
                 rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                 rng_state
             };
             
             self.zobrist_keys.resize_with(size, || [next_rand(), next_rand()]);
        }
    }

    pub fn get_best_move(&mut self, board: &HyperBoard, player: Player) -> Option<usize> {
        self.ensure_zobrist_initialized(board.total_cells());
        
        let mut best_score = match player {
            Player::X => i32::MIN,
            Player::O => i32::MAX,
        };
        let mut best_move = None;
        let alpha = i32::MIN;
        let beta = i32::MAX;
        
        let mut work_board = board.clone();
        
        // Initial hash
        let current_hash = self.compute_hash(&work_board);

        let mut available_moves = Vec::new();
        // Updated loop
        for idx in 0..work_board.total_cells() {
            if work_board.get_cell(idx).is_none() {
                available_moves.push(idx);
            }
        }

        let opponent = match player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        for &mv in &available_moves {
            // Make move
            work_board.make_move(mv, player).unwrap(); // Should be safe
            let move_hash = current_hash ^ self.get_zobrist_key(mv, player);

            let score = self.minimax(&mut work_board, 0, opponent, alpha, beta, move_hash);
            
            // Unmake move - needs clear_cell
            work_board.clear_cell(mv);
            
            match player {
                Player::X => {
                    if score > best_score {
                        best_score = score;
                        best_move = Some(mv);
                    }
                },
                Player::O => {
                    if score < best_score {
                        best_score = score;
                        best_move = Some(mv);
                    }
                }
            }
        }
        
        best_move.or(available_moves.first().copied())
    }
    
    fn compute_hash(&self, board: &HyperBoard) -> u64 {
        let mut hash = 0;
        for i in 0..board.total_cells() {
            if let Some(p) = board.get_cell(i) {
                hash ^= self.get_zobrist_key(i, p);
            }
        }
        hash
    }
    
    fn get_zobrist_key(&self, index: usize, player: Player) -> u64 {
        let p_idx = match player { Player::X => 0, Player::O => 1 };
        self.zobrist_keys[index][p_idx]
    }

    #[allow(clippy::too_many_arguments)]
    fn minimax(
        &mut self,
        board: &mut HyperBoard,
        depth: usize,
        current_player: Player, // The player whose turn it is
        mut alpha: i32,
        mut beta: i32,
        current_hash: u64,
    ) -> i32 {
        if let Some(winner) = board.check_win() {
            return match winner {
                Player::X => 1000 - depth as i32,
                Player::O => -1000 + depth as i32,
            };
        }

        if depth >= self.max_depth {
            return self.evaluate(board);
        }
        
        // is_full check?
        // Can check if moves available or expensive is_full in board
        // Let's rely on move generation: if no moves, draw (return 0)
        
        // But need to know if it's a draw vs just max depth.
        // check_draw is O(N).
        if board.check_draw() {
             return 0;
        }

        if let Some(&score) = self.transposition_table.get(&current_hash) {
            return score;
        }

        let opponent = match current_player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        let result;
        match current_player {
            Player::X => { // Maximizing
                let mut max_eval = i32::MIN;
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                        board.make_move(idx, Player::X).unwrap();
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::X);
                        
                        let eval = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                        board.clear_cell(idx); // Undo
                        
                        max_eval = max_eval.max(eval);
                        alpha = alpha.max(eval);
                        if beta <= alpha {
                            break;
                        }
                    }
                }
                // If no moves made and not win, it's a draw, handled above?
                // If max_eval is still MIN (no moves), then it's a draw -> 0
                if max_eval == i32::MIN { result = 0; } else { result = max_eval; }
            },
            Player::O => { // Minimizing
                let mut min_eval = i32::MAX;
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                         board.make_move(idx, Player::O).unwrap();
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::O);
                        
                        let eval = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                         board.clear_cell(idx); // Undo
                        
                        min_eval = min_eval.min(eval);
                        beta = beta.min(eval);
                        if beta <= alpha {
                            break;
                        }
                    }
                }
                if min_eval == i32::MAX { result = 0; } else { result = min_eval; }
            }
        }
        
        self.transposition_table.insert(current_hash, result);
        result
    }

    fn evaluate(&self, board: &HyperBoard) -> i32 {
        let mut score = 0;
        // This is tricky: we removed winning_lines from public API.
        // We have winning_masks.
        // Evaluating partial lines with bitmasks is harder than with vectors of indices unless we keep indices.
        // The current implementation of evaluate iterates winning_lines.
        
        // Option A: Re-expose winning_lines_indices?
        // Option B: Implement evaluate using bitwise ops (count set bits in masks).
        // (board & mask).popcount() -> how many X or O in that line.
        
        // Option B is much faster!
        
        match &board.winning_masks {
            crate::bitboard::WinningMasks::Small(masks) => {
                let p1_board = match board.p1 { crate::bitboard::BitBoard::Small(b) => b, _ => 0 };
                let p2_board = match board.p2 { crate::bitboard::BitBoard::Small(b) => b, _ => 0 };
                
                for &mask in masks {
                    let x_count = (p1_board & mask).count_ones();
                    let o_count = (p2_board & mask).count_ones();
                    
                    if o_count == 0 {
                        if x_count == 2 { score += 10; }
                        else if x_count == 1 { score += 1; }
                    } else if x_count == 0 {
                        if o_count == 2 { score -= 10; }
                        else if o_count == 1 { score -= 1; }
                    }
                }
            },
            crate::bitboard::WinningMasks::Medium(masks) => {
                let p1_board = match board.p1 { crate::bitboard::BitBoard::Medium(b) => b, _ => 0 };
                let p2_board = match board.p2 { crate::bitboard::BitBoard::Medium(b) => b, _ => 0 };
                
                for &mask in masks {
                    let x_count = (p1_board & mask).count_ones();
                    let o_count = (p2_board & mask).count_ones();
                    
                    if o_count == 0 {
                        if x_count == 2 { score += 10; }
                        else if x_count == 1 { score += 1; }
                    } else if x_count == 0 {
                        if o_count == 2 { score -= 10; }
                        else if o_count == 1 { score -= 1; }
                    }
                }
            },
            _ => {
                 // Fallback or ignore for Large
                 // Without easy iteration over lines, we skip heuristiceval for N>=5 or implement generic.
                 // For now, return 0 or maybe basic material count?
            }
        }
        
        score
    }

}
