use crate::domain::models::{BoardState, Game, GameResult, Player};
use crate::domain::services::{Clock, PlayerStrategy};
use std::fmt::Display;
use std::time::Duration;

pub struct GameService<'a, S: BoardState, C: Clock> {
    game: Game<S>,
    clock: C,
    player_x: Box<dyn PlayerStrategy<S> + 'a>,
    player_o: Box<dyn PlayerStrategy<S> + 'a>,
}

impl<'a, S: BoardState + Display, C: Clock> GameService<'a, S, C> {
    pub fn new(
        game: Game<S>,
        clock: C,
        player_x: Box<dyn PlayerStrategy<S> + 'a>,
        player_o: Box<dyn PlayerStrategy<S> + 'a>,
    ) -> Self {
        GameService {
            game,
            clock,
            player_x,
            player_o,
        }
    }

    pub fn board_state(&self) -> &S {
        self.game.board().state()
    }

    pub fn check_status(&self) -> GameResult {
        self.game.status()
    }

    pub fn current_turn_info(&self) -> (Player, Duration) {
        (self.game.current_player(), self.clock.now())
    }

    pub fn play_next_turn(&mut self) -> Result<GameResult, String> {
        let current_player = self.game.current_player();

        let strategy = match current_player {
            Player::X => &mut self.player_x,
            Player::O => &mut self.player_o,
        };

        if let Some(coord) = strategy.get_best_move(self.game.board().state(), current_player) {
            self.game.play_turn(coord)
        } else {
            Err("Strategy returned no move".to_string())
        }
    }
}
