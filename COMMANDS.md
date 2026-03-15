# Commands

## Play Manually

```bash
cargo run --release --manifest-path rust_evades/Cargo.toml -- --controller right
```

## Train DQN

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train --output-dir training_runs/dqn_default
```

## Resume DQN Training

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- train \
  --resume-model training_runs/dqn_default/final_model.json \
  --output-dir training_runs/dqn_resumed
```

## Evaluate DQN

```bash
cargo run --release --manifest-path rust_evades_dqn/Cargo.toml -- evaluate --model training_runs/dqn_default/final_model.json
```
