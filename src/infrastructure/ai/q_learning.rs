use crate::domain::models::{BoardState, Coordinate, Player};
use crate::domain::services::PlayerStrategy;
use crate::infrastructure::persistence::BitBoardState;
use crate::infrastructure::symmetries::SymmetryHandler;
use dashmap::DashMap;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct QKey {
    pub dimension: usize,
    pub p1_small: u32,
    pub p1_medium: u128,
    pub p1_large: Vec<u64>,
    pub p2_small: u32,
    pub p2_medium: u128,
    pub p2_large: Vec<u64>,
}

impl From<&BitBoardState> for QKey {
    fn from(state: &BitBoardState) -> Self {
        use crate::infrastructure::persistence::BitBoard;

        let (p1_small, p1_medium, p1_large) = match &state.p1 {
            BitBoard::Small(v) => (*v, 0, vec![]),
            BitBoard::Medium(v) => (0, *v, vec![]),
            BitBoard::Large(v) => (0, 0, v.clone()),
        };

        let (p2_small, p2_medium, p2_large) = match &state.p2 {
            BitBoard::Small(v) => (*v, 0, vec![]),
            BitBoard::Medium(v) => (0, *v, vec![]),
            BitBoard::Large(v) => (0, 0, v.clone()),
        };

        QKey {
            dimension: state.dimension,
            p1_small,
            p1_medium,
            p1_large,
            p2_small,
            p2_medium,
            p2_large,
        }
    }
}

pub type QTable = DashMap<QKey, HashMap<usize, f64>>;

#[derive(Clone)]
pub struct QLearner {
    q_table: Arc<QTable>,
    symmetry_handler: Arc<SymmetryHandler>,
    epsilon: f64,
    learning_rate: f64,
    discount_factor: f64,
}

impl QLearner {
    pub fn new(epsilon: f64, dimension: usize) -> Self {
        // Assume side=3 for standard hyper tictactoe, or we could pass it.
        // BitBoardState uses typical side=3.
        let symmetry_handler = Arc::new(SymmetryHandler::new(dimension, 3));

        Self {
            q_table: Arc::new(DashMap::new()),
            symmetry_handler,
            epsilon,
            learning_rate: 0.1,
            discount_factor: 0.9,
        }
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let q_table_map: HashMap<QKey, HashMap<usize, f64>> = bincode::deserialize_from(reader)?;
        let dashmap = DashMap::new();
        for (k, v) in &q_table_map {
            dashmap.insert(k.clone(), v.clone());
        }
        // We need dimension to recreate symmetry handler.
        // For now, we'll infer it from the first key or pass it?
        // The `load` signature needs updating or we hardcode/guess.
        // Actually, we can peak the first key from the loaded map.
        let dimension = q_table_map.keys().next().map(|k| k.dimension).unwrap_or(3);
        let symmetry_handler = Arc::new(SymmetryHandler::new(dimension, 3));

        Ok(Self {
            q_table: Arc::new(dashmap),
            symmetry_handler,
            epsilon: 0.0, // Default to inference mode
            learning_rate: 0.1,
            discount_factor: 0.9,
        })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        // DashMap doesn't support direct serialization, convert to HashMap first
        let map: HashMap<_, _> = self
            .q_table
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();
        bincode::serialize_into(writer, &map)?;
        Ok(())
    }

    pub fn train(&self, num_games: u64, dimension: usize) -> (f64, usize) {
        let batch_size = 1000;
        let num_batches = num_games / batch_size;

        // We need an atomic float for parallel max tracking. Since Rust doesn't have AtomicF64 natively in std,
        // Given the low contention (once per game end), a simple reduced result might be better.

        let max_delta = (0..num_batches)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut batch_max_delta = 0.0;

                for _ in 0..batch_size {
                    let mut board = BitBoardState::new(dimension);
                    let mut current_player = Player::X;

                    while board.check_win().is_none() && !board.is_full() {
                        let available_moves = self.get_available_moves(&board);

                        if available_moves.is_empty() {
                            break;
                        }

                        // For current state S, get canonical S' and map M.
                        let (key, map_idx) = self.get_canonical(&board);
                        let map = &self.symmetry_handler.maps[map_idx];

                        // Available moves also need to be mapped to canonical space to choose action
                        // But wait, the Q-table stores `(canonical_state, canonical_action)`.
                        // So we need to choose `canonical_action`.
                        let canonical_moves: Vec<usize> =
                            available_moves.iter().map(|&mv| map[mv]).collect();

                        // Epsilon-greedy in canonical space
                        let canonical_action = if rng.gen_range(0.0..1.0) < self.epsilon {
                            *canonical_moves.choose(&mut rng).unwrap()
                        } else {
                            self.get_best_action(&key, &canonical_moves)
                        };

                        // Map chosen canonical action BACK to real action to play on board
                        // We need inverse map. Or search map.
                        // map[real] = canonical => real such that map[real] == canonical_action
                        // Since maps are bijective permutations, this is unique.
                        let action = map.iter().position(|&val| val == canonical_action).unwrap();

                        let mut next_board = board.clone();
                        if next_board
                            .set_cell(Coordinate(action), current_player)
                            .is_ok()
                        {
                            let reward = self.calculate_reward(&next_board, current_player);

                            // For next state, we just need its value
                            let (next_key, _) = self.get_canonical(&next_board); // Don't care about map/action for value

                            let max_next_q = if reward != 0.0 || next_board.is_full() {
                                0.0
                            } else {
                                // To get max value of next state, we look at all available canonical moves from that state
                                // But wait, `get_available_moves` returns real moves.
                                // `next_key` corresponds to `next_board` transformed.
                                // The available moves in `next_key` are the transformed available moves of `next_board`.
                                // So we can just query the Q-table for that key. The actions in the Q-table ARE the canonical ones.
                                // We don't even need to map the next moves explicitly if `get_best_value` iterates the stored map?
                                // No, `get_best_value` takes `moves`.
                                // So we need: `next_board` available moves -> transformed -> `canonical_next_moves`.

                                let next_moves_real = self.get_available_moves(&next_board);
                                // We need the map used for `next_key` to transform these moves.
                                // Re-call get_canonical is slightly inefficient but correct.
                                let (_, next_map_idx) = self.get_canonical(&next_board);
                                let next_map = &self.symmetry_handler.maps[next_map_idx];
                                let next_canonical_moves: Vec<usize> =
                                    next_moves_real.iter().map(|&mv| next_map[mv]).collect();

                                let best_opponent_val =
                                    self.get_best_value(&next_key, &next_canonical_moves);
                                -best_opponent_val
                            };

                            let current_q = *self
                                .q_table
                                .entry(key.clone())
                                .or_default()
                                .entry(canonical_action)
                                .or_insert(0.0);
                            let diff = reward + self.discount_factor * max_next_q - current_q;
                            let new_q = current_q + self.learning_rate * diff;

                            let delta = diff.abs() * self.learning_rate;
                            if delta > batch_max_delta {
                                batch_max_delta = delta;
                            }

                            self.q_table
                                .entry(key)
                                .or_default()
                                .insert(canonical_action, new_q);

                            board = next_board;
                            current_player = current_player.opponent();
                        } else {
                            break;
                        }
                    }
                }
                batch_max_delta
            })
            .reduce(|| 0.0, f64::max);

        (max_delta, self.q_table.len())
    }

    fn get_available_moves(&self, board: &BitBoardState) -> Vec<usize> {
        let mut moves = Vec::new();
        for i in 0..board.total_cells() {
            if board.get_cell(Coordinate(i)).is_none() {
                moves.push(i);
            }
        }
        moves
    }

    fn get_best_action(&self, key: &QKey, moves: &[usize]) -> usize {
        let mut best_action = moves[0];
        let mut max_val = -f64::INFINITY;

        let binding = self.q_table.entry(key.clone()).or_default();

        for &mv in moves {
            let val = *binding.get(&mv).unwrap_or(&0.0);
            if val > max_val {
                max_val = val;
                best_action = mv;
            }
        }
        best_action
    }

    fn get_best_value(&self, key: &QKey, moves: &[usize]) -> f64 {
        if moves.is_empty() {
            return 0.0;
        }
        let mut max_val = -f64::INFINITY;
        if let Some(entry) = self.q_table.get(key) {
            for &mv in moves {
                let val = *entry.get(&mv).unwrap_or(&0.0);
                if val > max_val {
                    max_val = val;
                }
            }
            if max_val == -f64::INFINITY {
                0.0
            } else {
                max_val
            }
        } else {
            0.0
        }
    }

    fn calculate_reward(&self, board: &BitBoardState, player: Player) -> f64 {
        if let Some(winner) = board.check_win() {
            if winner == player { 1.0 } else { -1.0 }
        } else if board.is_full() {
            0.5
        } else {
            0.0
        }
    }

    // Returns (Canonical Key, Index of Map used)
    fn get_canonical(&self, board: &BitBoardState) -> (QKey, usize) {
        let mut min_key = QKey::from(board); // Default identity
        let mut min_map_idx = 0;

        for (i, map) in self.symmetry_handler.maps.iter().enumerate() {
            if i == 0 {
                continue;
            } // Identity already computed

            // Transform
            let transformed = self.apply_symmetry(board, map);
            let key = QKey::from(&transformed);

            if key < min_key {
                min_key = key;
                min_map_idx = i;
            }
        }

        (min_key, min_map_idx)
    }

    fn apply_symmetry(&self, board: &BitBoardState, map: &[usize]) -> BitBoardState {
        // Reconstruct a new board with bits moved
        let mut new_board = BitBoardState::new(board.dimension);

        // P1
        for i in 0..board.total_cells {
            use crate::infrastructure::persistence::BitBoard;
            let is_p1 = match &board.p1 {
                BitBoard::Small(v) => (v & (1 << i)) != 0,
                BitBoard::Medium(v) => (v & (1 << i)) != 0,
                BitBoard::Large(v) => {
                    let idx = i / 64;
                    if idx < v.len() {
                        (v[idx] & (1 << (i % 64))) != 0
                    } else {
                        false
                    }
                }
            };
            if is_p1 {
                let dest = map[i];
                let _ = new_board.set_cell(Coordinate(dest), Player::X);
            }

            let is_p2 = match &board.p2 {
                BitBoard::Small(v) => (v & (1 << i)) != 0,
                BitBoard::Medium(v) => (v & (1 << i)) != 0,
                BitBoard::Large(v) => {
                    let idx = i / 64;
                    if idx < v.len() {
                        (v[idx] & (1 << (i % 64))) != 0
                    } else {
                        false
                    }
                }
            };
            if is_p2 {
                let dest = map[i];
                let _ = new_board.set_cell(Coordinate(dest), Player::O);
            }
        }
        new_board
    }
}

impl PlayerStrategy<BitBoardState> for QLearner {
    fn get_best_move(&mut self, board: &BitBoardState, _player: Player) -> Option<Coordinate> {
        // Changed return type
        let moves = self.get_available_moves(board);
        if moves.is_empty() {
            return None;
        }
        if moves.is_empty() {
            return None;
        }

        // Inference with symmetry
        let (key, map_idx) = self.get_canonical(board);
        let map = &self.symmetry_handler.maps[map_idx];

        // Map moves to canonical
        let canonical_moves: Vec<usize> = moves.iter().map(|&mv| map[mv]).collect();

        // Get best canonical action
        let canonical_action = self.get_best_action(&key, &canonical_moves);

        // Map back
        let action = map.iter().position(|&val| val == canonical_action).unwrap();

        Some(Coordinate(action))
    }
}
