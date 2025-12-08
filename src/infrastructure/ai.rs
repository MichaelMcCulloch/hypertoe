use crate::domain::models::{Player, BoardState};
use crate::domain::services::PlayerStrategy;
use crate::infrastructure::persistence::{BitBoardState, BitBoard, WinningMasks};
use crate::infrastructure::symmetries::SymmetryHandler;
use rustc_hash::FxHashMap;

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
    transposition_table: FxHashMap<u64, TranspositionEntry>,
    zobrist_keys: Vec<[u64; 2]>,
    symmetries: Option<SymmetryHandler>,
    max_depth: usize,
    strategic_values: Vec<usize>,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        MinimaxBot {
            transposition_table: FxHashMap::default(),
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
        match &board.winning_masks {
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
        &mut self,
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
        if board.check_win().is_some() {
             // If we're here, the *previous* player made a winning line.
             // But we usually check 'win_at' immediately after move.
             // If existing logic assumes we check here:
             // return specific score?
             // Actually, original code checked 'check_win_at' inside the loop.
             // checks `check_draw` at start.
             
             // If someone won, returning score is tricky based on depth?
             // Let's rely on loop check.
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
         match &board.winning_masks {
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

        let mut best_score = match player {
            Player::X => i32::MIN,
            Player::O => i32::MAX,
        };
        let mut best_move = None;
        let mut alpha = i32::MIN + 1;
        let beta = i32::MAX - 1;

        let mut work_board = board.clone();
        let mut rolling_hashes = self.initialize_rolling_hashes(&work_board);
        let available_moves = self.get_sorted_moves(&work_board);

        let opponent = player.opponent();

        for &mv in &available_moves {
            work_board.set_cell(mv, player).unwrap();
            self.update_hashes(&mut rolling_hashes, mv, player);

            let score = self.minimax(
                &mut work_board,
                0,
                opponent,
                alpha,
                beta,
                &mut rolling_hashes,
            );

            work_board.clear_cell(mv);
            self.update_hashes(&mut rolling_hashes, mv, player);

            let is_better = match player {
                Player::X => score > best_score,
                Player::O => score < best_score,
            };

            if is_better {
                best_score = score;
                best_move = Some(mv);
                if player == Player::X {
                    alpha = alpha.max(score);
                }
            }
        }

        best_move.or(available_moves.first().copied())
    }
}
