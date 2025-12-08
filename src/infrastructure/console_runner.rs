use crate::application::game_service::GameService;
use crate::domain::models::{BoardState, GameResult};
use crate::domain::services::Clock;
use std::fmt::Display;

pub struct ConsoleRunner;

impl ConsoleRunner {
    pub fn run<'a, S, C>(mut game_service: GameService<'a, S, C>)
    where
        S: BoardState + Display,
        C: Clock,
    {
        println!("Starting Game...");
        // Initial state
        println!("{}", game_service.board_state());

        loop {
            // Check win/loss first? Or after move?
            // GameService should expose status.

            match game_service.check_status() {
                GameResult::Win(p) => {
                    println!("Player {:?} Wins!", p);
                    break;
                }
                GameResult::Draw => {
                    println!("It's a Draw!");
                    break;
                }
                GameResult::InProgress => {}
            }

            let (player, time) = game_service.current_turn_info();
            println!("Player {:?}'s turn (Time: {:?})", player, time);

            match game_service.play_next_turn() {
                Ok(new_status) => {
                    println!("{}", game_service.board_state());
                    if let GameResult::InProgress = new_status {
                        // Continue
                    } else {
                        // Loop will catch it next iteration or we can break here
                    }
                }
                Err(e) => {
                    println!("Error making move: {}", e);
                    // Decide if we break or retry. For now, retry (loop continues)
                    // But if it's a bot error, we might be stuck.
                    // If it's human error (invalid input), we want retry.
                    // The current GameService implementation of `get_best_move` loops for human, so it shouldn't return error easily?
                    // Actually `play_next_turn` might return error.
                    break;
                }
            }
        }
    }
}
