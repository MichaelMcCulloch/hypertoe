use crate::bitboard::{BitBoard, WinningMasks};
pub mod ai;
pub mod bitboard;
mod display;
pub mod game;
pub mod symmetries;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] // Added Hash
pub enum Player {
    X,
    O,
}

use std::fmt;

#[derive(Clone)]
pub struct HyperBoard {
    pub dimension: usize,
    pub p1: BitBoard,
    pub p2: BitBoard,
    pub winning_masks: WinningMasks,
    pub side: usize,
    total_cells: usize,
}

impl fmt::Display for HyperBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", display::render_board(self))
    }
}

impl HyperBoard {
    pub fn new(dimension: usize) -> Self {
        let side: usize = 3;
        let total_cells = side.pow(dimension as u32);

        let p1 = BitBoard::new(dimension, side);
        let p2 = BitBoard::new(dimension, side);

        let winning_masks = Self::generate_winning_masks(dimension, side);

        HyperBoard {
            dimension,
            p1,
            p2,
            winning_masks,
            side,
            total_cells,
        }
    }

    pub fn total_cells(&self) -> usize {
        self.total_cells
    }

    pub fn get_cell(&self, index: usize) -> Option<Player> {
        if self.p1.get_bit(index) {
            Some(Player::X)
        } else if self.p2.get_bit(index) {
            Some(Player::O)
        } else {
            None
        }
    }

    pub fn clear_cell(&mut self, index: usize) {
        self.p1.clear_bit(index);
        self.p2.clear_bit(index);
    }
    pub fn get_strategic_value(&self, index: usize) -> usize {
        match &self.winning_masks {
            WinningMasks::Small { map, .. } => map.get(index).map_or(0, |v| v.len()),
            WinningMasks::Medium { map, .. } => map.get(index).map_or(0, |v| v.len()),
            WinningMasks::Large { map, .. } => map.get(index).map_or(0, |v| v.len()),
        }
    }
    fn generate_winning_masks(dimension: usize, side: usize) -> WinningMasks {
        let lines_indices = Self::generate_winning_lines_indices(dimension, side);
        let total_cells = side.pow(dimension as u32);

        // Helper to build the map: for every cell, which line indices involve it?
        let mut map: Vec<Vec<usize>> = vec![vec![]; total_cells];

        for (line_idx, line) in lines_indices.iter().enumerate() {
            for &cell_idx in line {
                map[cell_idx].push(line_idx);
            }
        }

        if total_cells <= 32 {
            let mut masks = Vec::new();
            for line in lines_indices {
                let mut mask: u32 = 0;
                for idx in line {
                    mask |= 1 << idx;
                }
                masks.push(mask);
            }
            WinningMasks::Small { masks, map }
        } else if total_cells <= 128 {
            let mut masks = Vec::new();
            for line in lines_indices {
                let mut mask: u128 = 0;
                for idx in line {
                    mask |= 1 << idx;
                }
                masks.push(mask);
            }
            WinningMasks::Medium { masks, map }
        } else {
            let mut masks = Vec::new();
            let num_u64s = (total_cells + 63) / 64;
            for line in lines_indices {
                let mut mask_chunks = vec![0u64; num_u64s];
                for idx in line {
                    let vec_idx = idx / 64;
                    mask_chunks[vec_idx] |= 1 << (idx % 64);
                }
                masks.push(mask_chunks);
            }
            WinningMasks::Large { masks, map }
        }
    }

    fn generate_winning_lines_indices(dimension: usize, side: usize) -> Vec<Vec<usize>> {
        let mut lines = Vec::new();
        let all_directions = Self::get_canonical_directions(dimension);

        for dir in all_directions {
            let valid_starts = Self::get_valid_starts(dimension, side, &dir);
            for start in valid_starts {
                let mut line = Vec::new();
                let mut current = start.clone();
                let mut valid = true;

                for _ in 0..side {
                    if let Some(idx) = Self::coords_to_index(&current, side) {
                        line.push(idx);

                        // Move to next
                        for (i, d) in dir.iter().enumerate() {
                            let next_val = current[i] as isize + d;
                            current[i] = next_val as usize; // Assumes valid start makes this safe-ish
                        }
                    } else {
                        valid = false;
                        break;
                    }
                }

                if valid && line.len() == side {
                    lines.push(line);
                }
            }
        }
        lines
    }

    fn get_canonical_directions(dimension: usize) -> Vec<Vec<isize>> {
        let mut dirs = Vec::new();
        let num_dirs = 3_usize.pow(dimension as u32);

        for i in 0..num_dirs {
            let mut dir = Vec::new();
            let mut temp = i;
            let mut has_nonzero = false;
            let mut first_nonzero_is_positive = false;
            for _ in 0..dimension {
                let digit = temp % 3;
                temp /= 3;
                let val = match digit {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    _ => unreachable!(),
                };
                dir.push(val);
            }

            // Re-check canonical property on the generated vector
            for &val in &dir {
                if val != 0 {
                    has_nonzero = true;
                    if val > 0 {
                        first_nonzero_is_positive = true;
                    }
                    break;
                }
            }

            if has_nonzero && first_nonzero_is_positive {
                dirs.push(dir);
            }
        }
        dirs
    }

    fn get_valid_starts(dimension: usize, side: usize, dir: &[isize]) -> Vec<Vec<usize>> {
        let num_cells = side.pow(dimension as u32);
        let mut starts = Vec::new();

        for i in 0..num_cells {
            let coords = Self::index_to_coords(i, dimension, side);
            // Check if end point is valid
            let end_coords = coords.clone();
            let mut possible = true;
            for (c_idx, &d) in dir.iter().enumerate() {
                let start_val = end_coords[c_idx] as isize;
                let end_val = start_val + d * (side as isize - 1);

                if end_val < 0 || end_val >= side as isize {
                    possible = false;
                    break;
                }
            }
            if possible {
                starts.push(coords);
            }
        }
        starts
    }

    fn index_to_coords(index: usize, dimension: usize, side: usize) -> Vec<usize> {
        let mut coords = vec![0; dimension];
        let mut temp = index;
        for i in 0..dimension {
            coords[i] = temp % side;
            temp /= side;
        }
        coords
    }

    fn coords_to_index(coords: &[usize], side: usize) -> Option<usize> {
        let mut index = 0;
        let mut multiplier = 1;
        for &c in coords {
            if c >= side {
                return None;
            }
            index += c * multiplier;
            multiplier *= side;
        }
        Some(index)
    }

    pub fn make_move(&mut self, index: usize, player: Player) -> Result<(), String> {
        if index >= self.total_cells {
            return Err("Index out of bounds".to_string());
        }
        if self.p1.get_bit(index) || self.p2.get_bit(index) {
            return Err("Cell already occupied".to_string());
        }

        match player {
            Player::X => self.p1.set_bit(index),
            Player::O => self.p2.set_bit(index),
        }
        Ok(())
    }

    // New optimized version
    pub fn check_win_at(&self, index: usize) -> Option<Player> {
        // Only check lines passing through 'index'
        if self.p1.check_win_at(&self.winning_masks, index) {
            return Some(Player::X);
        }
        if self.p2.check_win_at(&self.winning_masks, index) {
            return Some(Player::O);
        }
        None
    }

    // Existing global check
    pub fn check_win(&self) -> Option<Player> {
        if self.p1.check_win(&self.winning_masks) {
            return Some(Player::X);
        }
        if self.p2.check_win(&self.winning_masks) {
            return Some(Player::O);
        }
        None
    }

    pub fn check_draw(&self) -> bool {
        // Check if full.
        // Counting bits would require popcount.
        // Iterating is safer for now or adding popcount to BitBoard.
        // Or simply: total_cells calls to get_bit? Slow.
        // Add `is_full` to BitBoard?
        // Fallback:
        // Or we can track move count in Game?
        // Let's implement an imperfect check: verify all cells are occupied.
        // Efficiency: (p1 | p2).popcount() == total_cells?
        // For now, simple loop:
        (0..self.total_cells).all(|i| self.p1.get_bit(i) || self.p2.get_bit(i))
            && self.check_win().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_lines_count() {
        let board = HyperBoard::new(2);
        // (5^2 - 3^2)/2 = 8
        match board.winning_masks {
            WinningMasks::Small(masks) => assert_eq!(masks.len(), 8),
            _ => panic!("Wrong mask type"),
        }
    }

    #[test]
    fn test_3d_lines_count() {
        let board = HyperBoard::new(3);
        // 49
        match board.winning_masks {
            WinningMasks::Small(masks) => assert_eq!(masks.len(), 49),
            _ => panic!("Wrong mask type"),
        }
    }

    #[test]
    fn test_4d_lines_count() {
        let board = HyperBoard::new(4);
        // 272
        match board.winning_masks {
            WinningMasks::Medium(masks) => assert_eq!(masks.len(), 272),
            _ => panic!("Wrong mask type"),
        }
    }

    #[test]
    fn test_win_detection() {
        let mut board = HyperBoard::new(2);
        board.make_move(0, Player::X).unwrap();
        board.make_move(1, Player::X).unwrap();
        board.make_move(2, Player::X).unwrap();
        assert_eq!(board.check_win(), Some(Player::X));
    }

    #[test]
    fn test_diagonal_win_20_11_02() {
        let mut board = HyperBoard::new(2);
        board.make_move(2, Player::X).unwrap();
        board.make_move(4, Player::X).unwrap();
        board.make_move(6, Player::X).unwrap();
        assert_eq!(board.check_win(), Some(Player::X));
    }

    #[test]
    fn test_reproduce_check_win_through_center() {
        let mut board = HyperBoard::new(3);
        // 4, 13, 22
        board.make_move(4, Player::X).unwrap();
        board.make_move(13, Player::X).unwrap();
        board.make_move(22, Player::X).unwrap();

        assert_eq!(
            board.check_win(),
            Some(Player::X),
            "Failed to detect Z-axis win through center"
        );
    }

    #[test]
    fn test_reproduce_ai_blocking() {
        let mut board = HyperBoard::new(3);
        // Setup state where O missed the block
        // X has 4, 13
        // O has 0, 5 (just distractor moves)

        board.make_move(3, Player::X).unwrap();
        board.make_move(4, Player::X).unwrap();
        board.make_move(13, Player::X).unwrap();

        board.make_move(0, Player::O).unwrap();
        board.make_move(5, Player::O).unwrap();

        // It is O's turn
        let mut bot = crate::ai::MinimaxBot::new(3); // Depth 3 should be enough
        let best_move = bot.get_best_move(&board, Player::O);

        assert_eq!(
            best_move,
            Some(22),
            "AI failed to block the winning move at 22"
        );
    }
}
