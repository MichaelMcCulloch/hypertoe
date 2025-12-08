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
    
    
    
    
    
    
    let moves = vec![
        (0, Player::X), (3, Player::O),
        (1, Player::X), (4, Player::O),
    ];
    let board = create_board(2, &moves);
    let mut bot = MinimaxBot::new(9); 
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(2), "Minimax failed to find immediate win in 2D");
}

#[test]
fn test_2d_block_in_1() {
    
    
    
    
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    let moves = vec![
        (0, Player::X), (4, Player::O), (8, Player::X)
    ];
    
    
    
    
    
    
    
    
    
    
    
    
    let board = create_board(2, &moves);
    let mut bot = MinimaxBot::new(9);
    
    let best_move = bot.get_best_move(&board, Player::X);
    
    assert!([2, 6].contains(&best_move.unwrap()), "Minimax failed to find fork move in 2D. Got {:?}", best_move);
}

#[test]
fn test_3d_win_in_1() {
    
    
    
    
    let moves = vec![
        (0, Player::X), (1, Player::O), 
        (9, Player::X), (2, Player::O), 
    ];
    let board = create_board(3, &moves);
    
    
    let mut bot = MinimaxBot::new(3); 
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(18), "Minimax failed to find immediate win in 3D");
}

#[test]
fn test_3d_block_in_1() {
    
    
    
    
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
    
    
    
    
    
    let moves = vec![
        (0, Player::X), (1, Player::O),
        (27, Player::X), (2, Player::O),
    ];
    let board = create_board(4, &moves);
    
    let mut bot = MinimaxBot::new(2); 
    
    let best_move = bot.get_best_move(&board, Player::X);
    assert_eq!(best_move, Some(54), "Minimax failed to find immediate win in 4D");
}

#[test]
fn test_4d_block_in_1() {
    
    
    
    
    let moves = vec![
        (0, Player::X), (1, Player::O),
        (27, Player::X),
    ];
    let board = create_board(4, &moves);
    let mut bot = MinimaxBot::new(2);
    
    let best_move = bot.get_best_move(&board, Player::O);
    assert_eq!(best_move, Some(54), "Minimax failed to block immediate loss in 4D");
}
