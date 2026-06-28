# Progress

Performance log for the DQN2 Arena training path. The goal is to speed up training without changing learning parameters, network type, or game settings, and to choose each next optimization from measured bottlenecks rather than broad refactors.

## Benchmark Setup

- `full-training`: real DQN2 Arena training, default `160` episodes.
- `simulated-survival`: mirrored loop with synthetic `40s` survival episodes, default `4` episodes.
- Section timing and reports are written under `training_runs/benchmarks/.../benchmark_report.json`.
- Earlier accepted optimization: skip batch diagnostics unless `progress_tx.is_some()`, so CLI training and benchmarks do not pay for dashboard-only metrics.

## Baseline

- Full benchmark: `81.818s` total, `63.869s` train batch core, `14.609s` evaluation.
- Simulated benchmark: `252.262s` total, `141.061s` train batch core, `21.132s` evaluation.

## Iteration 1 (accepted)

Changes in `rust_evades_dqn/src/network.rs`:

- Switched parallel batch accumulation to Rayon `fold`/`reduce` with per-worker `BatchGradients`.
- Removed cloned output activations during backprop and collapsed delta propagation to a single vector.
- Added `max_predict()` so TD targets do max-only inference instead of allocating full output vectors, with an equivalence test.

Result:

- Full benchmark: `67.474s` total, `54.798s` train batch core (total `-17.5%`, core `-14.2%`).
- Simulated benchmark: `196.520s` total, `130.102s` train batch core (total `-22.1%`, core `-7.8%`).

## Iteration 2 (accepted)

- In `rust_evades_dqn/src/trainer.rs`, the optimizer runtime selection now benchmarks several chunk-size candidates and keeps the fastest. On the test machine it selects parallel mode with chunk size `15`.

Result versus iteration 1:

- Full benchmark: `65.664s` total, `52.634s` train batch core (total `-2.7%`, core `-3.9%`).
- Simulated benchmark: `191.034s` total, `124.787s` train batch core (total `-2.8%`, core `-4.1%`).

## Attempted and Rejected

- Broad scratch-buffer allocation cleanup in `network.rs`: regressed, reverted.
- Ref-based sampled-transition batches: no consistent win, reverted.

## Next Target

- Add transposed layer weights for the backprop delta pass in `network.rs`. `train_batch_core` is still the dominant cost and that loop uses cache-unfriendly strided reads.
