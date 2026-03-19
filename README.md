# IAJ Evades

A Rust-based implementation of the Evades game engine with Deep Q-Learning (DQN) capabilities and a web-based management dashboard.

## Project Structure

- **rust_evades**: Core game engine and standalone graphical player.
- **rust_evades_dqn**: DQN implementation, training, and evaluation tools.
- **rust_evades_web**: Web-based dashboard for real-time visualization, training control, and configuration.

## Prerequisites

- **Rust**: [Install Rust](https://www.rust-lang.org/tools/install) (Edition 2021).
- **Node.js**: [Install Node.js](https://nodejs.org/) (v18 or newer) for the frontend dashboard.
- **C Dependencies**: `minifb` (the graphical crate) may require X11 development headers on Linux (`libx11-dev` on Debian/Ubuntu).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd iaj_evades
   ```

2. Build the web frontend:
   ```bash
   cd rust_evades_web/frontend
   npm install
   npm run build
   cd ../..
   ```

## Usage

### 1. Web Dashboard (Recommended)
The web dashboard provides a unified interface to control training, adjust game physics, and view the simulation.

```bash
cd rust_evades_web
cargo run --release
```
Once the server is running, open [http://localhost:8080](http://localhost:8080) in your browser.

### 2. Standalone Simulation
Run the simulation locally with a window using either manual or model-based control.

**Manual Control:**
```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller right
```

**Model Control (using a specific JSON model):**
```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller model --model best_model.json
```

### 3. CLI Training
You can also run training sessions directly from the command line.

**Start New Training:**
```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train --output-dir training_runs/my_run
```

**Resume Training:**
```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --resume-model training_runs/my_run/final_model.json \
  --output-dir training_runs/my_run_resumed
```

**Evaluate a Model:**
```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- evaluate --model best_model.json
```

## Features

- **Dual Map Designs**: Switch between **Open Map** (infinite space) and **Closed Map** (corridor with goal) in real-time.
- **Real-time Monitoring**: Visualize agent behavior and training metrics (reward, loss, survival) via WebSockets.
- **Dynamic Configuration**: Change game parameters like speed, radius, and spawn intervals without restarting.
- **Model Management**: Select and load different brain checkpoints (.json) directly from the dashboard.
- **Optimized Training**: Multi-threaded batch training with bad-seed focusing to improve agent robustness.
