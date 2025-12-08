use hypertictactoe::HyperBoard;

fn main() {
    println!("N-Dimensional Tic-Tac-Toe");
    
    for dim in 2..=6 {
        let mut board = HyperBoard::new(dim);
        println!("Dimension: {}, Side: {}, Winning Lines: {}", 
            board.dimension, board.side, board.winning_lines.len());
            
        // Make some dummy moves to show something
        if dim == 2 {
            let _ = board.make_move(0, hypertictactoe::Player::X);
            let _ = board.make_move(4, hypertictactoe::Player::O);
            let _ = board.make_move(8, hypertictactoe::Player::X);
        }
        
        println!("{}", board);
        println!("----------------------------------------");
    }
}
