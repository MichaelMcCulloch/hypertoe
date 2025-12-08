// src/ai.rs
use crate::{HyperBoard, Player};
use crate::symmetries::SymmetryHandler;
use std::collections::HashMap;

// --- Transposition Table Types ---

#[derive(Clone, Copy, PartialEq, Debug)]
enum Flag {
    Exact,      // The score is exact
    LowerBound, // The score is >= val (Beta cutoff previously)
    UpperBound, // The score is <= val (Alpha cutoff previously)
}

#[derive(Clone, Copy)]
struct TranspositionEntry {
    score: i32,
    depth: u8, // Store how deep we searched to get this result
    flag: Flag,
}

// --- Minimax Bot ---

pub struct MinimaxBot {
    transposition_table: HashMap<u64, TranspositionEntry>,
    zobrist_keys: Vec<[u64; 2]>, // [cell_idx][player_idx]
    symmetries: Option<SymmetryHandler>,
    max_depth: usize,
}

impl MinimaxBot {
    pub fn new(max_depth: usize) -> Self {
        MinimaxBot {
            transposition_table: HashMap::new(),
            zobrist_keys: Vec::new(),
            symmetries: None,
            max_depth,
        }
    }
    
    // Initialize Zobrist keys and Symmetry maps lazily
    fn ensure_initialized(&mut self, board: &HyperBoard) {
        if self.zobrist_keys.len() < board.total_cells() {
             let mut rng_state: u64 = 0xDEADBEEF + board.total_cells() as u64; 
             let mut next_rand = || -> u64 {
                 rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                 rng_state
             };
             
             self.zobrist_keys.resize_with(board.total_cells(), || [next_rand(), next_rand()]);
        }
        
        if self.symmetries.is_none() {
            self.symmetries = Some(SymmetryHandler::new(board.dimension, board.side));
        }
    }

    pub fn get_best_move(&mut self, board: &HyperBoard, player: Player) -> Option<usize> {
        self.ensure_initialized(board);
        
        let mut best_score = i32::MIN;
        let mut best_move = None;
        let mut alpha = i32::MIN + 1;
        let beta = i32::MAX - 1;
        
        let mut work_board = board.clone();

        // Get all legal moves
        let mut available_moves = Vec::new();
        for idx in 0..work_board.total_cells() {
            if work_board.get_cell(idx).is_none() {
                available_moves.push(idx);
            }
        }
        
        // Optimization: Sort moves? (omitted for brevity, but helps pruning)

        let opponent = match player { Player::X => Player::O, Player::O => Player::X };

        // Root Search
        for &mv in &available_moves {
            work_board.make_move(mv, player).unwrap();
            
            // Note: We don't pass a rolling hash here because the board changed 
            // and we need to recalculate canonical hash inside anyway.
            let score = self.minimax(&mut work_board, 0, opponent, alpha, beta);
            
            work_board.clear_cell(mv);
            
            // Logic for the root player (We want to MAXIMIZE our score)
            // Note: If player is O, the minimax returns a score from O's perspective?
            // Actually, in the minimax below, I used Negamax-style or explicit logic?
            // Let's look at the implementation below. It returns absolute score.
            // X wants +ve, O wants -ve.
            
            let is_better = match player {
                Player::X => score > best_score,
                Player::O => {
                     // For O, "Best" is the lowest score. 
                     // But we initialize best_score to i32::MIN which is confusing.
                     // Let's reset best_score based on player.
                     if best_score == i32::MIN && best_move.is_none() { true }
                     else { score < best_score }
                }
            };

            // Handling initialization for O
            if best_move.is_none() {
                best_score = score;
                best_move = Some(mv);
            } else if is_better {
                best_score = score;
                best_move = Some(mv);
                
                // Tighten bounds
                if player == Player::X {
                    alpha = alpha.max(score);
                } else {
                    // For O, we aren't tightening 'beta' for the loop, 
                    // because alpha/beta in Minimax are usually "My Guaranteed Best" vs "Opponent's Best Counter".
                    // But effectively, if we found a move that gives -10, we know we can get at least -10.
                }
            }
        }
        
        best_move.or(available_moves.first().copied())
    }

    fn get_canonical_hash(&self, board: &HyperBoard) -> u64 {
        let handler = self.symmetries.as_ref().unwrap();
        let mut min_hash = u64::MAX;

        // Iterate all symmetries to find the "Canonical" (smallest hash) view of the board
        for map in &handler.maps {
            let mut current_hash: u64 = 0;
            for real_idx in 0..board.total_cells() {
                if let Some(player) = board.get_cell(real_idx) {
                    let sym_idx = map[real_idx];
                    let p_idx = match player { Player::X => 0, Player::O => 1 };
                    current_hash ^= self.zobrist_keys[sym_idx][p_idx];
                }
            }
            if current_hash < min_hash {
                min_hash = current_hash;
            }
        }
        min_hash
    }

    fn minimax(
        &mut self,
        board: &mut HyperBoard,
        depth: usize,
        current_player: Player, 
        mut alpha: i32,
        mut beta: i32,
    ) -> i32 {
        let alpha_orig = alpha;
        
        // 1. Calculate Hash & TT Lookup
        // We calculate the hash of the board's CANONICAL form (lowest hash among all symmetries).
        // This automatically handles all 384 rotations/reflections in 4D.
        let canonical_hash = self.get_canonical_hash(board);
        
        // We only trust the cache if it searched at least as deep as we plan to go now.
        let remaining_depth = if self.max_depth > depth { (self.max_depth - depth) as u8 } else { 0 };

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

        // 2. Terminal Checks
        if let Some(winner) = board.check_win() {
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

        let opponent = match current_player { Player::X => Player::O, Player::O => Player::X };
        let mut best_val;

        // 3. Recursive Search
        match current_player {
            Player::X => { // Maximizing
                best_val = i32::MIN;
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                        board.make_move(idx, Player::X).unwrap();
                        let val = self.minimax(board, depth + 1, opponent, alpha, beta);
                        board.clear_cell(idx);
                        
                        best_val = best_val.max(val);
                        alpha = alpha.max(val);
                        if beta <= alpha { break; } // Beta Cutoff
                    }
                }
                if best_val == i32::MIN { best_val = 0; } // Should only happen on draw, handled above
            },
            Player::O => { // Minimizing
                best_val = i32::MAX;
                for idx in 0..board.total_cells() {
                    if board.get_cell(idx).is_none() {
                        board.make_move(idx, Player::O).unwrap();
                        let val = self.minimax(board, depth + 1, opponent, alpha, beta);
                        board.clear_cell(idx);
                        
                        best_val = best_val.min(val);
                        beta = beta.min(val);
                        if beta <= alpha { break; } // Alpha Cutoff
                    }
                }
                if best_val == i32::MAX { best_val = 0; }
            }
        }
        
        // 4. Store in TT
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

    fn evaluate(&self, board: &HyperBoard) -> i32 {
        let mut score = 0;
        
        // Optimized evaluation using bitmasks from board
        match &board.winning_masks {
            crate::bitboard::WinningMasks::Small(masks) => {
                let p1 = match board.p1 { crate::bitboard::BitBoard::Small(b) => b, _ => 0 };
                let p2 = match board.p2 { crate::bitboard::BitBoard::Small(b) => b, _ => 0 };
                for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 { score += 10; } else if x == 1 { score += 1; }
                    } else if x == 0 {
                        if o == 2 { score -= 10; } else if o == 1 { score -= 1; }
                    }
                }
            },
            crate::bitboard::WinningMasks::Medium(masks) => {
                let p1 = match board.p1 { crate::bitboard::BitBoard::Medium(b) => b, _ => 0 };
                let p2 = match board.p2 { crate::bitboard::BitBoard::Medium(b) => b, _ => 0 };
                for &mask in masks {
                    let x = (p1 & mask).count_ones();
                    let o = (p2 & mask).count_ones();
                    if o == 0 {
                        if x == 2 { score += 10; } else if x == 1 { score += 1; }
                    } else if x == 0 {
                        if o == 2 { score -= 10; } else if o == 1 { score -= 1; }
                    }
                }
            },
            _ => {}
        }
        score
    }
}