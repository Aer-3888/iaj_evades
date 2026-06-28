# IAJ Evades

A Rust reimplementation of the [Evades](https://evades.io) dodging game, with a Deep Q-Network agent trained from scratch and a web dashboard for live training control and visualization.

The neural network and the DQN training loop are implemented by hand in Rust (no ML framework), parallelized with Rayon. The agent learns to survive in an arena of moving enemies using a raycast-based vision of its surroundings.

## Project Structure

- **rust_evades**: core game engine, with a graphical window mode and a fast headless mode.
- **rust_evades_dqn**: the DQN agent, training, evaluation, and benchmark tooling (CLI).
- **rust_evades_web**: Axum web server plus a React/TypeScript dashboard for real-time training control and visualization over WebSockets.

## Tech Stack

- **Engine and agent**: Rust 2021, hand-written neural network, Rayon for parallel batch training, `minifb` for the graphical window.
- **Dashboard**: Rust (Axum, Tokio) backend, React + TypeScript + Vite frontend, Tailwind CSS, Recharts, live updates over WebSockets.

## Prerequisites

- **Rust** (edition 2021): [install Rust](https://www.rust-lang.org/tools/install).
- **Node.js** v18 or newer (only for the dashboard frontend): [install Node.js](https://nodejs.org/).
- On Linux, the graphical crate `minifb` may need X11 headers (`libx11-dev` on Debian/Ubuntu).

## Quick Start

Clone the repository:

```bash
git clone https://github.com/Aer-3888/iaj_evades.git
cd iaj_evades
```

### Watch the trained agent play

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller model --model best_model.json
```

### Train an agent

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train --output-dir training_runs/my_run
```

### Run the web dashboard

```bash
# Build the frontend once (outputs to rust_evades_web/dist)
cd rust_evades_web/frontend && npm install && npm run build

# Start the server from the web crate
cd .. && cargo run --release
```

Open [http://localhost:8080](http://localhost:8080) once you see `listening on 0.0.0.0:8080`.

A full command reference (training options, evaluation, benchmarks, resuming) is in [COMMANDS.md](COMMANDS.md).

## Features

- **Two agent variants**: a baseline `dqn` and a deeper `dqn2` raycast model, selectable at training time.
- **From-scratch DQN**: replay buffer, target network, and Huber loss implemented in Rust, with multi-threaded batch training and bad-seed focusing to harden the agent against its weakest scenarios.
- **Multiple maps**: open (infinite space), closed (corridor with a goal), and arena.
- **Web dashboard**: start, stop, and resume training, tune game and training parameters live, switch maps, and watch the agent through a raycast visualizer, all updating in real time over WebSockets.
- **Headless mode and benchmarks**: reproducible CLI training, evaluation, and timed benchmark runs with JSON reports.

## Models

`best_model.json` is a pre-trained checkpoint you can run directly. New runs write their checkpoints and `final_model.json` under the `--output-dir` you pass to `train`.
