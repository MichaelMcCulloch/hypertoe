use crate::domain::models::{BoardState, Player};
use std::time::Duration;

pub trait Clock {
    fn now(&self) -> Duration;
}

pub trait PlayerStrategy<S: BoardState> {
    fn get_best_move(&mut self, board: &S, player: Player) -> Option<usize>;
}
