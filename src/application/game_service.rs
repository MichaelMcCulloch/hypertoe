use crate::domain::models::{Board, BoardState, GameResult, Player};
use crate::domain::services::{Clock, PlayerStrategy};
use std::fmt::Display;

pub struct GameService<'a, S: BoardState, C: Clock> {
    board: Board<S>,
    clock: C,
    player_x: Box<dyn PlayerStrategy<S> + 'a>, // Boxing traits requires lifetime if they capture env?
    player_o: Box<dyn PlayerStrategy<S> + 'a>,
    turn: Player,
}

impl<'a, S: BoardState + Display, C: Clock> GameService<'a, S, C> {
    pub fn new(
        board: Board<S>,
        clock: C,
        player_x: Box<dyn PlayerStrategy<S> + 'a>,
        player_o: Box<dyn PlayerStrategy<S> + 'a>,
    ) -> Self {
        GameService {
            board,
            clock,
            player_x,
            player_o,
            turn: Player::X,
        }
    }

    pub fn start(&mut self) {
        println!("Starting Game...");
        println!("{}", self.board.state());

        loop {
            match self.board.check_status() {
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

            let start_time = self.clock.now();
            println!("Player {:?}'s turn (Time: {:?})", self.turn, start_time);

            let strategy = match self.turn {
                Player::X => &mut self.player_x,
                Player::O => &mut self.player_o,
            };

            // Assuming get_best_move is blocking for now
            if let Some(move_idx) = strategy.get_best_move(self.board.state(), self.turn) {
                match self.board.make_move(move_idx, self.turn) {
                    Ok(_) => {
                        println!("{}", self.board.state()); // Display board
                        self.turn = self.turn.opponent();
                    }
                    Err(e) => println!("Error making move: {}", e),
                }
            } else {
                println!("No moves available? Check logic.");
                break;
            }
        }
    }
}
