
pub mod ai;
pub mod game;
mod display;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] // Added Hash
pub enum Player {
    X,
    O,
}

use std::fmt;

#[derive(Clone)]
pub struct HyperBoard {
    pub dimension: usize,
    pub cells: Vec<Option<Player>>,
    pub winning_lines: Vec<Vec<usize>>,
    pub side: usize, // Currently hardcoded to 3
}

impl fmt::Display for HyperBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", display::render_board(self))
    }
}


impl HyperBoard {
    pub fn new(dimension: usize) -> Self {
        let side: usize = 3;
        let num_cells = side.pow(dimension as u32);
        let cells = vec![None; num_cells];
        let winning_lines = Self::generate_winning_lines(dimension, side);

        HyperBoard {
            dimension,
            cells,
            winning_lines,
            side,
        }
    }

    fn generate_winning_lines(dimension: usize, side: usize) -> Vec<Vec<usize>> {
        let mut lines = Vec::new();
        
        // A line is defined by a start position and a direction vector.
        // The direction vector has components in {-1, 0, 1}.
        // Not all zeros.
        // We can iterate through all cells as potential starting points.
        // And iterate through all 3^N - 1 directions.
        // But more efficiently:
        // A line exists if we can take `side - 1` steps in a specific direction 
        // and stay within bounds.
        
        // Let's model coordinates as Vec<usize>.
        // Direction as Vec<isize>.
        
        // To avoid duplicates (e.g. forward vs backward), we can enforce a
        // "canonical" direction. e.g. the first non-zero component must be positive.
        
        let all_directions = Self::get_canonical_directions(dimension);
        
        for dir in all_directions {
            // For a given direction, find all valid start positions.
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
                            if next_val < 0 || next_val >= side as isize {
                                // This check handles the boundary, though get_valid_starts 
                                // should theoretically prevent starting where we can't finish.
                                // But inside the loop we need to update `current`.
                            }
                            current[i] = next_val as usize;
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
                
                // We generate in reverse order effectively, but order doesn't matter for the set.
                // However to check canonical:
                // Let's treat the vector as d_0, d_1, ... 
                // We want the first non-zero element to be +1.
                // Wait, simply iterating 0..3^N includes all.
                // We want to avoid 0,0,...0
                // And we want to avoid v and -v being counted twice.
                
                dir.push(val);
            }
            
            // Re-check canonical property on the generated vector
            // Find first non-zero
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
        // Iterate all cells. A cell is a valid start if:
        // cell + (side-1)*dir is within bounds.
        // Since we only maintain canonical directions (first non-zero is +),
        // we mainly worry about upper bounds for + components and lower bounds for - components.
        
        // Actually, easiest is just Iterate all coords, check if end is in bounds.
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
            if c >= side { return None; }
            index += c * multiplier;
            multiplier *= side;
        }
        Some(index)
    }

    pub fn make_move(&mut self, index: usize, player: Player) -> Result<(), String> {
        if index >= self.cells.len() {
            return Err("Index out of bounds".to_string());
        }
        if self.cells[index].is_some() {
            return Err("Cell already occupied".to_string());
        }
        self.cells[index] = Some(player);
        Ok(())
    }

    pub fn check_win(&self) -> Option<Player> {
        for line in &self.winning_lines {
            let first_idx = line[0];
            if let Some(first_p) = self.cells[first_idx] {
                let mut all_match = true;
                for &idx in &line[1..] {
                    if self.cells[idx] != Some(first_p) {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    return Some(first_p);
                }
            }
        }
        None
    }

    pub fn check_draw(&self) -> bool {
        self.cells.iter().all(|c| c.is_some()) && self.check_win().is_none()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_lines_count() {
        let board = HyperBoard::new(2);
        // (5^2 - 3^2)/2 = (25 - 9)/2 = 8
        assert_eq!(board.winning_lines.len(), 8);
    }
    
    #[test]
    fn test_3d_lines_count() {
        let board = HyperBoard::new(3);
        // (5^3 - 3^3)/2 = (125 - 27)/2 = 98/2 = 49
        assert_eq!(board.winning_lines.len(), 49);
    }
    
    #[test]
    fn test_4d_lines_count() {
        let board = HyperBoard::new(4);
        // (5^4 - 3^4)/2 = (625 - 81)/2 = 544/2 = 272
        assert_eq!(board.winning_lines.len(), 272);
    }


    #[test]
    fn test_win_detection() {
        let mut board = HyperBoard::new(2);
        // X X X
        // . . .
        // . . .
        board.make_move(0, Player::X).unwrap();
        board.make_move(1, Player::X).unwrap();
        board.make_move(2, Player::X).unwrap();
        assert_eq!(board.check_win(), Some(Player::X));
    }

    #[test]
    fn test_diagonal_win_20_11_02() {
        let mut board = HyperBoard::new(2);
        // . . X  (2)
        // . X .  (4)
        // X . .  (6)
        board.make_move(2, Player::X).unwrap();
        board.make_move(4, Player::X).unwrap();
        board.make_move(6, Player::X).unwrap();
        
        // Debug print lines if needed
        // println!("Lines: {:?}", board.winning_lines);
        
        assert_eq!(board.check_win(), Some(Player::X));
    }
}
