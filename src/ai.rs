use crate::{HyperBoard, Player};
use std::collections::HashMap;

pub struct MinimaxBot {
    transposition_table: HashMap<u64, i32>,
    zobrist_keys: Vec<[u64; 2]>, // [cell_idx][player_idx]
    max_depth: usize,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        // Initialize Zobrist keys with a simple LCG to avoid dependencies

        
        // We might need keys for up to N=6 or whatever max is supported.
        // But the bot is instantiated per game? No, per move technically in main loop it's part of Game struct.
        // Game creates bot with fixed max depth, but board size depends on dimension.
        // The bot doesn't know the board size until it sees it.
        // Or we can resize dynamically.
        // For simplicity, let's just make it grow lazily or initialize large enough.
        // But `HyperBoard` size scales exponentially.
        // Better: Initialize it when first called or just passing dimension to new?
        // Passed `dimension` to `HyperBoard`, but here we don't know it.
        // Let's assume we initialize it large (e.g. 1000) or check in `get_best_move`.
        // Actually, easiest is to lazy init or refactor `new` to take dimension/size.
        // But Game creates it with just depth.
        // I will make `zobrist_keys` empty initially and populate on first use if needed, 
        // or just accept re-init cost if I change signature.
        // Changing `new` signature is cleanest since `Game` knows dimension.
        
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
        self.ensure_zobrist_initialized(board.cells.len());
        // Persistent memoization: Do NOT clear the table.
        // self.transposition_table.clear();
        
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
        for (idx, cell) in work_board.cells.iter().enumerate() {
            if cell.is_none() {
                available_moves.push(idx);
            }
        }

        let opponent = match player {
            Player::X => Player::O,
            Player::O => Player::X,
        };

        for &mv in &available_moves {
            // Make move
            work_board.cells[mv] = Some(player);
            let move_hash = current_hash ^ self.get_zobrist_key(mv, player);

            // Pass 'opponent' as current_player (it's their turn next)
            // We NO LONGER pass root_player, scoring is absolute (X+, O-)
            let score = self.minimax(&mut work_board, 0, opponent, alpha, beta, move_hash);
            
            // Unmake move
            work_board.cells[mv] = None;
            
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
        for (i, cell) in board.cells.iter().enumerate() {
            if let Some(p) = cell {
                hash ^= self.get_zobrist_key(i, *p);
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
        
        let is_full = board.cells.iter().all(|c| c.is_some());
        if is_full {
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
                let len = board.cells.len();
                for idx in 0..len {
                    if board.cells[idx].is_none() {
                        board.cells[idx] = Some(Player::X);
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::X);
                        
                        let eval = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                        board.cells[idx] = None; // Undo
                        
                        max_eval = max_eval.max(eval);
                        alpha = alpha.max(eval);
                        if beta <= alpha {
                            break;
                        }
                    }
                }
                result = max_eval;
            },
            Player::O => { // Minimizing
                let mut min_eval = i32::MAX;
                let len = board.cells.len();
                for idx in 0..len {
                    if board.cells[idx].is_none() {
                        board.cells[idx] = Some(Player::O);
                        let new_hash = current_hash ^ self.get_zobrist_key(idx, Player::O);
                        
                        let eval = self.minimax(board, depth + 1, opponent, alpha, beta, new_hash);
                        board.cells[idx] = None; // Undo
                        
                        min_eval = min_eval.min(eval);
                        beta = beta.min(eval);
                        if beta <= alpha {
                            break;
                        }
                    }
                }
                result = min_eval;
            }
        }
        
        self.transposition_table.insert(current_hash, result);
        result
    }

    fn evaluate(&self, board: &HyperBoard) -> i32 {
        let mut score = 0;
        // Evaluation is always relative to Player X (Positive for X, Negative for O)
        
        for line in &board.winning_lines {
            let mut x_count = 0;
            let mut o_count = 0;
            
            for &idx in line {
                match board.cells[idx] {
                    Some(Player::X) => x_count += 1,
                    Some(Player::O) => o_count += 1,
                    _ => {}
                }
            }

            if o_count == 0 {
                if x_count == 2 { score += 10; }
                else if x_count == 1 { score += 1; }
            } else if x_count == 0 {
                if o_count == 2 { score -= 10; }
                else if o_count == 1 { score -= 1; }
            }
        }
        
        score
    }

}
