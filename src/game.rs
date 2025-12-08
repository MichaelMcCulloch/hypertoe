use crate::{HyperBoard, Player};
use crate::ai::MinimaxBot;
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlayerType {
    Human,
    CPU,
}

pub struct Game {
    board: HyperBoard,
    player_x: PlayerType,
    player_o: PlayerType,
    turn: Player,
    bot: MinimaxBot,
}

impl Game {
    pub fn new(dimension: usize, player_x: PlayerType, player_o: PlayerType) -> Self {
        Game {
            board: HyperBoard::new(dimension),
            player_x,
            player_o,
            turn: Player::X,
            bot: MinimaxBot::new(3), // Depth limit 3 for starters
        }
    }

    pub fn start(&mut self) {
        println!("Starting N-Dimensional Tic-Tac-Toe (Dim: {})", self.board.dimension);
        println!("X: {:?} | O: {:?}", self.player_x, self.player_o);
        println!("{}", self.board);

        loop {
            if let Some(winner) = self.board.check_win() {
                println!("Player {:?} Wins!", winner);
                break;
            }
            if self.board.check_draw() {
                println!("It's a Draw!");
                break;
            }

            println!("Player {:?}'s turn ({:?})", self.turn, self.current_player_type());
            
            let move_idx = match self.current_player_type() {
                PlayerType::Human => self.get_human_move(),
                PlayerType::CPU => {
                    println!("Thinking...");
                    self.bot.get_best_move(&self.board, self.turn).unwrap_or(0)
                }
            };

            match self.board.make_move(move_idx, self.turn) {
                Ok(_) => {
                    println!("{}", self.board);
                    self.switch_turn();
                }
                Err(e) => {
                    println!("Invalid move: {}", e);
                    // Don't switch turn, retry
                }
            }
        }
    }

    fn current_player_type(&self) -> PlayerType {
        match self.turn {
            Player::X => self.player_x,
            Player::O => self.player_o,
        }
    }

    fn switch_turn(&mut self) {
        self.turn = match self.turn {
            Player::X => Player::O,
            Player::O => Player::X,
        };
    }

    fn get_human_move(&self) -> usize {
        loop {
            print!("Enter move index (0-{}): ", self.board.cells.len() - 1);
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            
            // Should properly parse coordinates too maybe? 
            // For now, raw index is easiest for debugging, 
            // but for user UX, maybe x,y,z is better?
            // Let's stick to raw index for MVP as requested.
            match input.trim().parse::<usize>() {
                Ok(idx) => return idx,
                Err(_) => println!("Invalid number"),
            }
        }
    }
}
