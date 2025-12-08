use crate::{HyperBoard, Player};
use std::collections::HashMap;

// 1. Define the types of values we can store
#[derive(Clone, Copy, PartialEq)]
enum Flag {
    Exact,
    LowerBound, // Alpha (At most this score) - Note: Naming conventions vary, but usually Alpha failure = UpperBound of value
    UpperBound, // Beta (At least this score)
}

// 2. The entry to store in the Hash Map
#[derive(Clone, Copy)]
struct TranspositionEntry {
    score: i32,
    depth: u8, // Using u8 to save space, standard depth won't exceed 255
    flag: Flag,
    // Optional: best_move: Option<usize> (Good for move ordering later)
}

pub struct MinimaxBot {
    transposition_table: HashMap<u64, TranspositionEntry>,
    zobrist_keys: Vec<[u64; 2]>,
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
             let mut next_rand = || -> u64 {
                 rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                 rng_state
             };
             self.zobrist_keys.resize_with(size, || [next_rand(), next_rand()]);
        }
    }

    pub fn get_best_move(&mut self, board: &HyperBoard, player: Player) -> Option<usize> {
        self.ensure_zobrist_initialized(board.total_cells());
        // Clear table between moves if you want to save memory, 
        // but keeping it makes the AI stronger over time (though memory heavier).
        // self.transposition_table.clear(); 

        let mut best_score = i32::MIN;
        let mut best_move = None;
        let mut alpha = i32::MIN + 1; // +1 buffer for safety
        let beta = i32::MAX - 1;
        
        let mut work_board = board.clone();
        let current_hash = self.compute_hash(&work_board);

        // Basic move ordering: check available cells
        let mut available_moves = Vec::new();
        for idx in 0..work_board.total_cells() {
            if work_board.get_cell(idx).is_none() {
                available_moves.push(idx);
            }
        }
        
        // Optimization: Shuffle available_moves or heuristic sort here prevents worst-case complexity

        let opponent = match player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        for &mv in &available_moves {
            work_board.make_move(mv, player).unwrap();
            let move_hash = current_hash ^ self.get_zobrist_key(mv, player);

            // We are calling minimax for the opponent, so negate the result
            // Note: Your original code had specific Player::X/O logic. 
            // Standard minimax usually employs Negamax to simplify this, 
            // but I will adapt your specific X/O maximizing/minimizing logic.
            
            // However, your original root loop logic was slightly distinct from the recursive step.
            // Let's align it. 
            let score = self.minimax(&mut work_board, 0, opponent, alpha, beta, move_hash);
            
            work_board.clear_cell(mv);
            
            // X wants to Maximize, O wants to Minimize.
            // But usually, get_best_move wants to find the best move for *Player*.
            // So we always want the move that yields the 'best' score for 'Player'.
            
            let is_better = match player {
                Player::X => score > best_score,
                Player::O => score < best_score || best_score == i32::MIN, // Fix initialization for O
            };

            // Fix O's initial best_score logic in the loop
            if best_move.is_none() {
                best_score = score;
                best_move = Some(mv);
            } else if match player { Player::X => score > best_score, Player::O => score < best_score } {
                best_score = score;
                best_move = Some(mv);
                // Update Alpha/Beta at the root
                if player == Player::X {
                    alpha = alpha.max(score);
                } else {
                    // This variable is actually beta for the root search if we were recursing
                    // but locally here we just track best.
                }
            } else if score == best_score {
                 // Keep your tie-breaker logic if desired, it is valid.
                 // Omitting for brevity in this fix, but insert `does_move_block` here.
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
        current_player: Player,
        mut alpha: i32,
        mut beta: i32,
        current_hash: u64,
    ) -> i32 {
        let alpha_orig = alpha; // Save original alpha for TT flag determination

        // 1. Transposition Table Lookup
        // We only use the cached value if the cached depth is >= what we plan to search
        let remaining_depth = if self.max_depth > depth { (self.max_depth - depth) as u8 } else { 0 };

        if let Some(entry) = self.transposition_table.get(&current_hash) {
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

        // 2. Base Cases
        if let Some(winner) = board.check_win() {
            // Adjust score by depth to prefer faster wins / slower losses
            return match winner {
                Player::X => 1000 - depth as i32,
                Player::O => -1000 + depth as i32,
            };
        }
        
        if board.check_draw() {
             return 0;
        }

        if depth >= self.max_depth {
            return self.evaluate(board);
        }

        let opponent = match current_player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        let mut best_val; 

        // 3. Recursion
        match current_player {
            Player::X => { // Maximizing
                best_val = i32::MIN;
                // Move generation... in a real engine, you'd sort these using the TT's best_move if available
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                        board.make_move(idx, Player::X).unwrap();
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::X);
                        
                        let val = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                        
                        board.clear_cell(idx); 
                        
                        best_val = best_val.max(val);
                        alpha = alpha.max(val);
                        if beta <= alpha {
                            break; // Beta Cutoff
                        }
                    }
                }
                // Handle case where no moves exist but check_draw/check_win failed (shouldn't happen with correct logic)
                if best_val == i32::MIN { best_val = 0; } 
            },
            Player::O => { // Minimizing
                best_val = i32::MAX;
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                        board.make_move(idx, Player::O).unwrap();
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::O);
                        
                        let val = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                        
                        board.clear_cell(idx); 
                        
                        best_val = best_val.min(val);
                        beta = beta.min(val);
                        if beta <= alpha {
                            break; // Alpha Cutoff
                        }
                    }
                }
                if best_val == i32::MAX { best_val = 0; }
            }
        }
        
        // 4. Store in Transposition Table
        let flag = if best_val <= alpha_orig {
            Flag::UpperBound // We failed low, so the true value is at most alpha
        } else if best_val >= beta {
            Flag::LowerBound // We failed high, so the true value is at least beta
        } else {
            Flag::Exact // We found a value between alpha and beta
        };

        let entry = TranspositionEntry {
            score: best_val,
            depth: remaining_depth,
            flag,
        };
        
        self.transposition_table.insert(current_hash, entry);

        best_val
    }

    fn evaluate(&self, board: &HyperBoard) -> i32 {
        let mut score = 0;
        
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
            _ => {}
        }
        
        score
    }
    
    // Helper for manual blocking check - kept from your original code if needed
    fn does_move_block(&self, board: &HyperBoard, mv: usize, player: Player) -> bool {
        let opponent = match player { Player::X => Player::O, Player::O => Player::X };
        let mut test_board = board.clone();
        if test_board.make_move(mv, opponent).is_ok() {
            if test_board.check_win() == Some(opponent) {
                return true;
            }
        }
        false
    }
}