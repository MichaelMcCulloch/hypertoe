use hypertictactoe::domain::models::{Player, BoardState};
use hypertictactoe::domain::services::PlayerStrategy;
use hypertictactoe::infrastructure::ai::MinimaxBot;
use hypertictactoe::infrastructure::persistence::BitBoardState;

fn create_board(dimension: usize, moves: &[(usize, Player)]) -> BitBoardState {
    let mut board = BitBoardState::new(dimension);
    for &(idx, player) in moves {
        board.set_cell(idx, player).expect("Failed to set cell in test setup");
    }
    board
}

#[test]
fn test_2d_win_in_1() {
    // 3x3 Board
    // X X .
    // O O .
    // . . .
    // X to move, should play 2 to win.
    
    let moves = vec![
        (0, Player::X), (3, Player::O),
        (1, Player::X), (4, Player::O),
    ];
    let board = create_board(2, &moves);
    let mut bot = MinimaxBot::new(9); // Full depth search feasible for 2D
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(2), "Minimax failed to find immediate win in 2D");
}

#[test]
fn test_2d_block_in_1() {
    // 3x3 Board
    // X X .
    // O . .
    // . . .
    // O to move. X threatens win at 2. O must block at 2.
    
    let moves = vec![
        (0, Player::X), (3, Player::O),
        (1, Player::X),
    ];
    let board = create_board(2, &moves);
    let mut bot = MinimaxBot::new(9);
    
    let best_move = bot.get_best_move(&board, Player::O);
    assert_eq!(best_move, Some(2), "Minimax failed to block immediate loss in 2D");
}

#[test]
fn test_2d_win_in_2_fork() {
    // 3x3 Board - Creating a fork
    // X . .
    // . O .
    // . . X
    // O to move. (This is center opening for O usually leads to draw, let's try a corner setup)
    
    // Better fork scenario:
    // X has (0), (8). O has (4).
    // X plays (2). Now X has threats:
    // 1. Column 2 (2, 5, 8) -> requires O block at 5
    // 2. Row 0 (0, 1, 2) -> requires O block at 1
    // Fork!
    
    // Setup:
    // X . .
    // . O .
    // . . X
    // Turn: X
    let moves = vec![
        (0, Player::X), (4, Player::O), (8, Player::X)
    ];
    // If X plays 2:
    // X . X
    // . O .
    // . . X
    // Threatens 0-1-2 and 2-5-8.
    // Or 6:
    // X . .
    // . O .
    // X . X
    // Threatens 6-7-8 and 0-3-6.
    
    // Let's verify X chooses a fork move (2 or 6).
    let board = create_board(2, &moves);
    let mut bot = MinimaxBot::new(9);
    
    let best_move = bot.get_best_move(&board, Player::X);
    // 2 and 6 are symmetric optimal moves here.
    assert!([2, 6].contains(&best_move.unwrap()), "Minimax failed to find fork move in 2D. Got {:?}", best_move);
}

#[test]
fn test_3d_win_in_1() {
    // 3x3x3 Board (index range 0..27)
    // Win line along Z axis at (0,0): 0, 9, 18.
    // Setup: X at 0, 9. O elsewhere. X to play.
    
    let moves = vec![
        (0, Player::X), (1, Player::O), // Layer 0
        (9, Player::X), (2, Player::O), // Layer 1
    ];
    let board = create_board(3, &moves);
    
    // 3D search space is big, but depth 3 should handle "Win in 1" (depth 1 check effectively)
    let mut bot = MinimaxBot::new(3); 
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(18), "Minimax failed to find immediate win in 3D");
}

#[test]
fn test_3d_block_in_1() {
    // 3x3x3 Board
    // X threatening win on Z axis at (0,0): 0, 9, 18.
    // O to move.
    
    let moves = vec![
        (0, Player::X), (1, Player::O),
        (9, Player::X),
    ];
    let board = create_board(3, &moves);
    let mut bot = MinimaxBot::new(3);
    
    let best_move = bot.get_best_move(&board, Player::O);
    assert_eq!(best_move, Some(18), "Minimax failed to block immediate loss in 3D");
}

#[test]
fn test_4d_win_in_1() {
    // 3x3x3x3 Board
    // Indexing can be tricky, but linear indices work same way.
    // Stride for 4th dim is 27.
    // Win line along W axis at (0,0,0): 0, 27, 54.
    
    let moves = vec![
        (0, Player::X), (1, Player::O),
        (27, Player::X), (2, Player::O),
    ];
    let board = create_board(4, &moves);
    // Shallow search for 4D
    let mut bot = MinimaxBot::new(2); 
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(54), "Minimax failed to find immediate win in 4D");
}

#[test]
fn test_4d_block_in_1() {
    // 3x3x3x3 Board
    // X threatening win 0, 27, 54.
    // O to play.
    
    let moves = vec![
        (0, Player::X), (1, Player::O),
        (27, Player::X),
    ];
    let board = create_board(4, &moves);
    let mut bot = MinimaxBot::new(2);
    
    let best_move = bot.get_best_move(&board, Player::O);
    assert_eq!(best_move, Some(54), "Minimax failed to block immediate loss in 4D");
}
