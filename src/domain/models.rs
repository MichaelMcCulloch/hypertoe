use std::fmt::Debug;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coordinate(pub usize);

impl From<usize> for Coordinate {
    fn from(val: usize) -> Self {
        Coordinate(val)
    }
}

impl Coordinate {
    pub fn new(val: usize) -> Self {
        Coordinate(val)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    X,
    O,
}

impl Player {
    pub fn opponent(&self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Win(Player),
    Draw,
    InProgress,
}

pub trait BoardState: Debug + Clone {
    fn new(dimension: usize) -> Self
    where
        Self: Sized;
    fn dimension(&self) -> usize;
    fn side(&self) -> usize;
    fn total_cells(&self) -> usize;
    fn get_cell(&self, coord: Coordinate) -> Option<Player>;
    fn set_cell(&mut self, coord: Coordinate, player: Player) -> Result<(), String>;
    fn clear_cell(&mut self, coord: Coordinate);
    fn check_win(&self) -> Option<Player>;
    fn is_full(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct Board<S: BoardState> {
    state: S,
}

impl<S: BoardState> Board<S> {
    pub fn new(dimension: usize) -> Self {
        Self {
            state: S::new(dimension),
        }
    }

    pub fn dimension(&self) -> usize {
        self.state.dimension()
    }

    pub fn make_move(&mut self, coord: Coordinate, player: Player) -> Result<(), String> {
        self.state.set_cell(coord, player)
    }

    pub fn get_cell(&self, coord: Coordinate) -> Option<Player> {
        self.state.get_cell(coord)
    }

    pub fn check_status(&self) -> GameResult {
        if let Some(winner) = self.state.check_win() {
            return GameResult::Win(winner);
        }
        if self.state.is_full() {
            return GameResult::Draw;
        }
        GameResult::InProgress
    }

    pub fn state(&self) -> &S {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub struct Game<S: BoardState> {
    board: Board<S>,
    current_player: Player,
    status: GameResult,
}

impl<S: BoardState> Game<S> {
    pub fn new(dimension: usize) -> Self {
        Self {
            board: Board::new(dimension),
            current_player: Player::X,
            status: GameResult::InProgress,
        }
    }

    pub fn board(&self) -> &Board<S> {
        &self.board
    }

    pub fn current_player(&self) -> Player {
        self.current_player
    }

    pub fn status(&self) -> GameResult {
        self.status
    }

    pub fn play_turn(&mut self, coord: Coordinate) -> Result<GameResult, String> {
        if self.status != GameResult::InProgress {
            return Err("Game is already over".to_string());
        }

        self.board.make_move(coord, self.current_player)?;

        self.status = self.board.check_status();

        if self.status == GameResult::InProgress {
            self.current_player = self.current_player.opponent();
        }

        Ok(self.status)
    }
}
