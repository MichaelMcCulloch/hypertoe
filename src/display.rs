use crate::{HyperBoard, Player};
use std::fmt;

struct Canvas {
    width: usize,
    height: usize,
    buffer: Vec<char>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Canvas {
            width,
            height,
            buffer: vec![' '; width * height],
        }
    }

    fn put(&mut self, x: usize, y: usize, c: char) {
        if x < self.width && y < self.height {
            self.buffer[y * self.width + x] = c;
        }
    }


}

impl fmt::Display for Canvas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                write!(f, "{}", self.buffer[y * self.width + x])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn render_board(board: &HyperBoard) -> String {
    // Determine size
    let (w, h) = calculate_size(board.dimension);
    let mut canvas = Canvas::new(w, h);
    
    // Recursive draw
    draw_recursive(board, board.dimension, &mut canvas, 0, 0, 0);
    
    canvas.to_string()
}

fn calculate_size(dim: usize) -> (usize, usize) {
    if dim == 0 { return (1, 1); } // Single cell if we ever drilled down that far, but base is dim 1 or 2
    if dim == 1 { return (3, 1); } // 1D line: X X X
    
    if dim == 2 {
        // 2D:
        // X X X
        // X X X
        // X X X
        // Size: 3 normal + 2 spaces? Or just packed?
        // Let's use 1 char spacing for readability.
        // 3 chars + 2 spaces = 5 width.
        // 3 lines = 3 height.
        return (5, 3);
    }
    
    let (child_w, child_h) = calculate_size(dim - 1);
    
    // Alternating axis strategy
    // Dim 3: Horizontal stack of 3 Dim 2
    // Dim 4: Vertical stack of 3 Dim 3
    
    // If dim is odd (3, 5): Horizontal
    // If dim is even (4, 6): Vertical
    
    if dim % 2 != 0 {
        // Horizontal
        // width = 3 * child_w + 2 * padding
        // height = child_h
        let gap = 2; // Increased gap for higher dim separation
        (child_w * 3 + gap * 2, child_h)
    } else {
        // Vertical
        // width = child_w
        // height = 3 * child_h + 2 * padding
        let gap = 1; // Vertical gap
        (child_w, child_h * 3 + gap * 2)
    }
}

fn draw_recursive(
    board: &HyperBoard, 
    current_dim: usize, 
    canvas: &mut Canvas, 
    x: usize, 
    y: usize, 
    base_index: usize
) {
    let side = 3;
    
    if current_dim == 2 {
        // Render 3x3 grid
        // index mapping:
        // (0,0) -> base_index
        // (1,0) -> base_index + 1
        // (2,0) -> base_index + 2
        // (0,1) -> base_index + 3 ?? 
        // No, standard mapping: x + y*side + z*side^2 ...
        // In this recursion, we are peeling off higher dimensions.
        // The remaining indices are [0..9] relative to base_index.
        
        for dy in 0..3 {
            for dx in 0..3 {
                let cell_idx = base_index + dx + dy * side;
                let char_to_draw = match board.cells[cell_idx] {
                    Some(Player::X) => 'X',
                    Some(Player::O) => 'O',
                    None => '.',
                };
                // Grid spacing:
                // 0 -> 0
                // 1 -> 2
                // 2 -> 4
                canvas.put(x + dx * 2, y + dy, char_to_draw);
            }
        }
        return;
    }
    
    // Recursive step
    let (child_w, child_h) = calculate_size(current_dim - 1);
    let stride = side.pow((current_dim - 1) as u32);
    
    if current_dim % 2 != 0 {
        // Horizontal arrangement
        // 0 -> left, 1 -> middle, 2 -> right
        let gap = 2;
        for i in 0..3 {
            let next_x = x + i * (child_w + gap);
            let next_y = y;
            let next_base = base_index + i * stride;
            draw_recursive(board, current_dim - 1, canvas, next_x, next_y, next_base);
            
            // Draw visual separator if not last
            if i < 2 {
                let sep_x = next_x + child_w + gap/2 - 1;
                // Draw a vertical bar in the gap?
                for k in 0..child_h {
                    canvas.put(sep_x, next_y + k, '|');
                }
            }
        }
    } else {
        // Vertical arrangement
        // 0 -> top, 1 -> middle, 2 -> bottom
        let gap = 1;
        for i in 0..3 {
            let next_x = x;
            let next_y = y + i * (child_h + gap);
            let next_base = base_index + i * stride;
            draw_recursive(board, current_dim - 1, canvas, next_x, next_y, next_base);
            
             // Draw visual separator if not last
            if i < 2 {
                let sep_y = next_y + child_h;
                // Draw a horizontal DASH
                for k in 0..child_w {
                   canvas.put(next_x + k, sep_y, '-'); 
                }
            }
        }
    }
}
