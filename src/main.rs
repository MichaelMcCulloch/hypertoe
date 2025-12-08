use hypertictactoe::application::game_service::GameService;
use hypertictactoe::domain::models::Game;
use hypertictactoe::domain::services::PlayerStrategy;
use hypertictactoe::infrastructure::ai::MinimaxBot;
use hypertictactoe::infrastructure::console::HumanConsolePlayer;
use hypertictactoe::infrastructure::console_runner::ConsoleRunner;
use hypertictactoe::infrastructure::persistence::BitBoardState;
use hypertictactoe::infrastructure::time::SystemClock;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let num_threads = if available_threads > 2 {
        available_threads - 2
    } else {
        1
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let mut dimension = 3;
    let mut player_x_type = "h";
    let mut player_o_type = "c";
    let mut depth = usize::MAX;

    if args.len() > 1 {
        if let Ok(d) = args[1].parse::<usize>() {
            dimension = d;
        }
    }

    let mut train_mode = false;
    let q_table_path = "qtable.bin".to_string();

    if args.len() > 2 {
        let mode = args[2].as_str();
        if mode == "Q" {
            train_mode = true;
        } else if mode.len() >= 2 {
            player_x_type = &mode[0..1];
            player_o_type = &mode[1..2];
        }
    }

    if train_mode {
        use hypertictactoe::infrastructure::ai::QLearner;
        println!(
            "Starting Q-Learning training for dimension {}...",
            dimension
        );
        let learner = if std::path::Path::new(&q_table_path).exists() {
            println!("Loading existing Q-table...");
            QLearner::load(&q_table_path).unwrap_or_else(|e| {
                println!("Failed to load Q-table: {}. Starting fresh.", e);
                QLearner::new(0.1, dimension)
            })
        } else {
            QLearner::new(0.1, dimension)
        };

        let _num_games = 1_000_000;

        println!("Training on billions of games (simulated loop)...");
        let games_per_epoch = 100_000;
        let mut total_games = 0u64;

        loop {
            let (max_delta, visited_states) = learner.train(games_per_epoch, dimension);
            total_games += games_per_epoch;

            if max_delta < 0.000001 && total_games > 100_000 {
                println!("Converged! Max Delta: {:.6}", max_delta);
                learner.save(&q_table_path).expect("Failed to save Q-table");
                break;
            }

            if total_games % 1_000_000 == 0 {
                println!(
                    " trained {} games... States: {}, Max Delta: {:.6}",
                    total_games, visited_states, max_delta
                );
                learner.save(&q_table_path).expect("Failed to save Q-table");
            }
        }
        return;
    }

    if args.len() > 3 {
        if let Ok(d) = args[3].parse::<usize>() {
            depth = d;
        }
    }

    let load_q_player = || {
        use hypertictactoe::infrastructure::ai::QLearner;
        if std::path::Path::new(&q_table_path).exists() {
            QLearner::load(&q_table_path).expect("Failed to load Q-table")
        } else {
            panic!("No Q-table found. Run with mode 'Q' to train first.");
        }
    };

    let player_x: Box<dyn PlayerStrategy<BitBoardState>> = match player_x_type {
        "h" => Box::new(HumanConsolePlayer::new()),
        "c" => Box::new(MinimaxBot::new(depth)),
        "q" => Box::new(load_q_player()),
        _ => Box::new(HumanConsolePlayer::new()),
    };

    let player_o: Box<dyn PlayerStrategy<BitBoardState>> = match player_o_type {
        "h" => Box::new(HumanConsolePlayer::new()),
        "c" => Box::new(MinimaxBot::new(depth)),
        "q" => Box::new(load_q_player()),
        _ => Box::new(MinimaxBot::new(depth)),
    };

    let clock = SystemClock::new();

    let game = Game::<BitBoardState>::new(dimension);

    let game_service = GameService::new(game, clock, player_x, player_o);
    ConsoleRunner::run(game_service);
}
