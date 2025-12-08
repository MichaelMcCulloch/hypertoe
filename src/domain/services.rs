use std::time::Duration;
use crate::domain::models::{BoardState, Player};

pub trait Clock {
    fn now(&self) -> Duration;
}

pub trait PlayerStrategy<S: BoardState> {
    fn get_best_move(&mut self, board: &S, player: Player) -> Option<usize>;
}
