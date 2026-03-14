# Evades Commands

This repo now has three Rust projects:

- `rust_evades` - the game itself
- `rust_evades_dqn` - the default DQN trainer
- `rust_evades_neat` - the NEAT trainer kept as an alternative

## Build And Test

Build the game:

```bash
cargo build --release --manifest-path rust_evades/Cargo.toml
```

Build the default DQN trainer:

```bash
cargo build --release --manifest-path rust_evades_dqn/Cargo.toml
```

Build the NEAT trainer:

```bash
cargo build --release --manifest-path rust_evades_neat/Cargo.toml
```

Run game tests:

```bash
cargo test --manifest-path rust_evades/Cargo.toml
```

Run DQN trainer tests:

```bash
cargo test --manifest-path rust_evades_dqn/Cargo.toml
```

Run NEAT trainer tests:

```bash
cargo test --manifest-path rust_evades_neat/Cargo.toml
```

## Play The Game Manually

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller right
```

Notes:

- `--controller right` avoids the model requirement and lets you play manually.
- In the window: `WASD` or arrow keys move, `R` resets, `F3` toggles FPS, `Esc` quits.

## Watch A Trained Model Play

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --model training_runs/dqn_default/final_model.json
```

With a fixed seed:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --seed 2 --model training_runs/dqn_default/final_model.json
```

Notes:

- `B` toggles between model control and manual control.
- The loader supports both DQN and NEAT JSON models.
- By default, model playback uses `--controller model`.

## Run Headless Simulation

Headless with a trained model:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --headless --episodes 100 --model training_runs/dqn_default/final_model.json
```

Headless with simple right-only movement:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --headless --episodes 100 --controller right
```

Notes:

- Headless mode runs uncapped as fast as possible.

## Train A Model - DQN Default

Recommended DQN run:

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train --output-dir training_runs/dqn_default
```

1-hour style DQN run:

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train --output-dir training_runs/dqn_one_hour --episodes 20000 --checkpoint-every 250
```

Default DQN training behavior:

- reward is per-frame survival time
- 24 fixed seeds starting at seed `2`
- 2 random seeds per cycle
- action repeat `4`

## Train A Model - NEAT Alternative

Recommended long run:

```bash
cargo run --release --manifest-path rust_evades_neat/Cargo.toml -- train --output-dir training_runs/long_neat
```

Heavier overnight run:

```bash
cargo run --release --manifest-path rust_evades_neat/Cargo.toml -- train --output-dir training_runs/overnight --population 384 --generations 2500 --checkpoint-every 25
```

Default training seed schedule:

- 24 fixed seeds starting at seed `2`
- plus 2 rotating random seeds

## Evaluate A Saved Model

Evaluate a DQN model:

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- evaluate --model training_runs/dqn_default/final_model.json
```

Evaluate a NEAT model:

```bash
cargo run --release --manifest-path rust_evades_neat/Cargo.toml -- evaluate --model training_runs/long_neat/final_model.json
```

## Files To Change Default Settings

Game defaults:

- `rust_evades/src/config.rs` - world size, corridor size, player speed, enemy count, enemy speeds, wall-ball count/speed/radius, render FPS, colors

Game CLI defaults:

- `rust_evades/src/main.rs` - default CLI behavior like controller mode, headless defaults, episode count, seed flags

Game window behavior:

- `rust_evades/src/render.rs` - controls, HUD text, model/manual toggle behavior, FPS display

Model playback/input encoding:

- `rust_evades/src/model_player.rs` - model loading, raycast observation layout, action decoding for both DQN and NEAT

Default DQN trainer CLI defaults:

- `rust_evades_dqn/src/main.rs` - default output directory, episode count, checkpoint frequency, seed schedule, action repeat

Default DQN trainer behavior:

- `rust_evades_dqn/src/trainer.rs` - replay buffer, epsilon schedule, target network sync, reward shaping, fixed/random seed schedule

DQN network layout:

- `rust_evades_dqn/src/network.rs` - MLP architecture, forward pass, gradient updates

DQN observations:

- `rust_evades_dqn/src/observation.rs` - 36 raycasts, ray deltas, player x-delta input construction

Trainer CLI defaults:

- `rust_evades_neat/src/main.rs` - default output directory, population, generations, checkpoint frequency, seed schedule flags

Trainer long-run defaults:

- `rust_evades_neat/src/trainer.rs` - default population size, generations, trainer seed, checkpoint cadence, fixed training seed list

NEAT mutation/species defaults:

- `rust_evades_neat/src/neat.rs` - mutation rates, add-node/add-connection chances, compatibility threshold, elite count, tournament size

Training observations:

- `rust_evades_neat/src/observation.rs` - 36 raycasts, ray deltas, player x-delta input construction

Fitness / reward shaping:

- `rust_evades_dqn/src/trainer.rs` - per-frame survival reward and DQN optimization defaults
- `rust_evades_neat/src/trainer.rs` - NEAT evaluation reward shaping and speciation tuning

## Output Files

Training saves models like:

- `training_runs/.../best_model.json`
- `training_runs/.../final_model.json`
- `training_runs/.../checkpoint_gen_XXXX.json`
