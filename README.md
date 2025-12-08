# HyperTicTacToe

Experimenting with higher-dimensional Tic-Tac-Toe using Rust, optimized bitboards, and Domain-Driven Design.

## Overview

HyperTicTacToe is a high-performance implementation of Tic-Tac-Toe that scales to N-dimensions (2D, 3D, 4D, etc.). It features a robust AI powered by Minimax with Alpha-Beta pruning, Transposition Tables, and Symmetry Reduction.

The codebase adheres to **Domain-Driven Design (DDD)** principles, ensuring a clean separation between the core game logic (Domain), standard game flow (Application), and technical details like I/O and State Persistence (Infrastructure).

## Features

- **N-Dimensional Gameplay**: Play on 2D (3x3), 3D (3x3x3), or even 4D boards.
- **High Performance AI**:
    - **Minimax**: Parallelized with Rayon, using Iterative Deepening.
    - **Optimization**: AVX2 SIMD intrinsics for bitboard operations.
    - **Symmetry Reduction**: Reduces search space by up to 48x in 3D.
    - **Lock-Free Transposition Table**: Shared hash map for caching board evaluations across threads.
- **Q-Learning**: Experimental support for Reinforcement Learning training.

## Architecture

The project is structured into three main layers:

- **Domain** (`src/domain`): The heart of the software. Contains the `Game` Aggregate Root, `Coordinate` Value Object, and traits like `BoardState` and `PlayerStrategy`. Pure Rust, no side effects.
- **Application** (`src/application`): Orchestrates the game flow. `GameService` coordinates turns and rules without knowing about the console or specific AI implementations.
- **Infrastructure** (`src/infrastructure`): Implementation details. Console I/O, `BitBoard` representation (SIMD optimized), and specific AI strategies (`MinimaxBot`, `QLearner`).

## Usage

### Prerequisites
- Rust (latest stable)

### Running the Game

To play a standard 3D game (Human vs Computer):

```bash
cargo run
```

### Command Line Arguments

```bash
cargo run -- [dimension] [mode] [depth]
```

- **dimension**: The dimension of the board (e.g., `2` for normal Tic-Tac-Toe, `3` for 3D). Default is `3`.
- **mode**: A two-character string specifying players for X and O.
    - `h`: Human
    - `c`: Computer (Minimax)
    - `q`: Q-Learner (requires trained model)
    - Example: `hc` (Human vs Computer), `cc` (Computer vs Computer).
- **depth**: Minimax search depth limit.

**Examples:**

Play 2D Tic-Tac-Toe against the bot:
```bash
cargo run -- 2 hc
```

Watch two bots play in 3D:
```bash
cargo run -- 3 cc
```

Train Q-Learning agent (experimental):
```bash
cargo run -- 3 Q
```
