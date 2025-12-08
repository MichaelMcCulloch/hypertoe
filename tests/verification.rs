use hypertictactoe::domain::models::{BoardState, Player};
use hypertictactoe::domain::services::PlayerStrategy;
use hypertictactoe::infrastructure::ai::MinimaxBot;
use hypertictactoe::infrastructure::persistence::BitBoardState;
use std::sync::atomic::Ordering;

#[test]
fn test_id_and_tt_stats() {
    let board = BitBoardState::new(3);
    let mut bot = MinimaxBot::new(5);

    println!("Starting search...");
    let best_move = bot.get_best_move(&board, Player::X);

    let nodes = bot.stats.nodes_searched.load(Ordering::Relaxed);
    let tt_hits = bot.stats.tt_hits.load(Ordering::Relaxed);
    let tt_exact_hits = bot.stats.tt_exact_hits.load(Ordering::Relaxed);

    println!(
        "Final Stats - Nodes: {}, TT Hits: {}, Exact Hits: {}",
        nodes, tt_hits, tt_exact_hits
    );

    assert!(best_move.is_some());
    assert!(nodes > 0, "Should search some nodes");

    assert!(tt_hits > 0, "Should have transposition table hits");
}
