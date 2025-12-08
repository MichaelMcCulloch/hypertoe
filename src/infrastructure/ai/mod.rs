pub mod minimax;
pub mod q_learning;
pub mod transposition;

pub use minimax::{MinimaxBot, SearchStats};
pub use q_learning::QLearner;
