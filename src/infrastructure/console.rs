use crate::domain::models::{BoardState, Coordinate, Player};
use crate::domain::services::PlayerStrategy;
use std::io::{self, Write};

pub struct HumanConsolePlayer;

impl HumanConsolePlayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S: BoardState> PlayerStrategy<S> for HumanConsolePlayer {
    fn get_best_move(&mut self, board: &S, _player: Player) -> Option<Coordinate> {
        loop {
            print!("Enter move index (0-{}): ", board.total_cells() - 1);
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            match input.trim().parse::<usize>() {
                Ok(idx) => {
                    if idx < board.total_cells() && board.get_cell(Coordinate(idx)).is_none() {
                        return Some(Coordinate(idx));
                    } else if idx >= board.total_cells() {
                        println!("Index out of bounds");
                    } else {
                        println!("Cell already occupied");
                    }
                }
                Err(_) => println!("Invalid number"),
            }
        }
    }
}
