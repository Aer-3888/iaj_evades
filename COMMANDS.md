# Commands

## Play Manually

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller right
```

## Train DQN

Default training uses the bad-seed-focused mode and will report both average and worst-seed metrics.

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --output-dir training_runs/dqn_default
```

## Train DQN With Original Seed Handling

This keeps the old behavior: no extra oversampling of weak seeds, and checkpoint selection based on timeouts, average survival, then average return.

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --output-dir training_runs/dqn_original \
  --seed-focus-mode original
```

## Resume DQN Training

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --resume-model training_runs/dqn_default/final_model.json \
  --output-dir training_runs/dqn_resumed
```

## Resume DQN Training With Original Seed Handling

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --resume-model training_runs/dqn_original/final_model.json \
  --output-dir training_runs/dqn_original_resumed \
  --seed-focus-mode original
```

## Evaluate DQN

Evaluation now prints average and worst-seed survival/return metrics.

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- evaluate --model training_runs/dqn_default/final_model.json
```

## Run Web Dashboard (Live Visualizer & Training Control)

The dashboard allows you to change game settings, start/stop DQN training sessions, and view the game in real-time.

```bash
# Build frontend (once)
cd rust_evades_web/frontend && npm install && npm run build

# Run dashboard server from the web directory
cd ../ && cargo run --release
```
Open `http://localhost:8080` in your browser once you see the `listening on 0.0.0.0:8080` log.

## Run game with best model
```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller model --model best_model.json
```
