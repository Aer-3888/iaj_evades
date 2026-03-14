# Evades Commands

This repo now has two Rust projects:

- `rust_evades` - the game itself
- `rust_evades_neat` - the NEAT trainer that learns to play it

## Build And Test

Build the game:

```bash
cargo build --release --manifest-path rust_evades/Cargo.toml
```

Build the trainer:

```bash
cargo build --release --manifest-path rust_evades_neat/Cargo.toml
```

Run game tests:

```bash
cargo test --manifest-path rust_evades/Cargo.toml
```

Run trainer tests:

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
cargo run --release --manifest-path rust_evades/Cargo.toml -- --model training_runs/long_neat/final_model.json
```

With a fixed seed:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --seed 2 --model training_runs/long_neat/final_model.json
```

Notes:

- `B` toggles between model control and manual control.
- By default, model playback uses `--controller model`.

## Run Headless Simulation

Headless with a trained model:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --headless --episodes 100 --model training_runs/long_neat/final_model.json
```

Headless with simple right-only movement:

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --headless --episodes 100 --controller right
```

Notes:

- Headless mode runs uncapped as fast as possible.

## Train A Model

Recommended long run:

```bash
cargo run --release --manifest-path rust_evades_neat/Cargo.toml -- train --output-dir training_runs/long_neat
```

Heavier overnight run:

```bash
cargo run --release --manifest-path rust_evades_neat/Cargo.toml -- train --output-dir training_runs/overnight --population 384 --generations 2500 --checkpoint-every 25
```

Default training seed schedule:

- 40 seeds starting at seed `2`
- that means seeds `2..41`

## Evaluate A Saved Model

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

- `rust_evades/src/neat_player.rs` - model loading, raycast observation layout, action decoding

Trainer CLI defaults:

- `rust_evades_neat/src/main.rs` - default output directory, population, generations, checkpoint frequency, seed schedule flags

Trainer long-run defaults:

- `rust_evades_neat/src/trainer.rs` - default population size, generations, trainer seed, checkpoint cadence, fixed training seed list

NEAT mutation/species defaults:

- `rust_evades_neat/src/neat.rs` - mutation rates, add-node/add-connection chances, compatibility threshold, elite count, tournament size

Training observations:

- `rust_evades_neat/src/observation.rs` - 36 raycasts, ray deltas, player x-delta input construction

Fitness / reward shaping:

- `rust_evades_neat/src/trainer.rs` - rightward reward weighting, progress bonus, goal bonus, time penalty

## Output Files

Training saves models like:

- `training_runs/.../best_model.json`
- `training_runs/.../final_model.json`
- `training_runs/.../checkpoint_gen_XXXX.json`
