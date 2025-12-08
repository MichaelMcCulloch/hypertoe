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

        println!("{}", game_service.board_state());

        loop {
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
                    } else {
                    }
                }
                Err(e) => {
                    println!("Error making move: {}", e);

                    break;
                }
            }
        }
    }
}
