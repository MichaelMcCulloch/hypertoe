use hypertictactoe::game::{Game, PlayerType};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Default config
    let mut dimension = 3;
    let mut player_x = PlayerType::Human;
    let mut player_o = PlayerType::CPU;

    // Simple arg parsing
    // Usage: cargo run -- [dim] [mode]
    // mode: hh, hc, ch, cc
    if args.len() > 1 {
        if let Ok(d) = args[1].parse::<usize>() {
            dimension = d;
        }
    }
    if args.len() > 2 {
        match args[2].as_str() {
            "hh" => { player_x = PlayerType::Human; player_o = PlayerType::Human; },
            "hc" => { player_x = PlayerType::Human; player_o = PlayerType::CPU; },
            "ch" => { player_x = PlayerType::CPU; player_o = PlayerType::Human; },
            "cc" => { player_x = PlayerType::CPU; player_o = PlayerType::CPU; },
            _ => println!("Unknown mode, defaulting to Human vs CPU"),
        }
    }

    let mut game = Game::new(dimension, player_x, player_o);
    game.start();
}
