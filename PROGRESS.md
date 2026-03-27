# Progress

## Current Goals

- Keep the DQN2 Arena benchmark/reporting path reproducible and comparable across iterations.
- Preserve the accepted `train_batch_core` speedups without changing learning parameters, network type, or game settings.
- Choose the next optimization from the current measured bottlenecks instead of broad refactors.

## Optimizations Attempted And Results

- Added reproducible benchmark runs and JSON reports:
  - `full-training`: real DQN2 Arena training, default `160` episodes.
  - `simulated-survival`: mirrored loop with synthetic `40s` survival episodes, default `4` episodes.
- Added section timing and benchmark reports under `training_runs/benchmarks/.../benchmark_report.json`.
- Added and maintained `PROGRESS.md`.
- Accepted earlier optimization:
  - Skip batch diagnostics work unless `progress_tx.is_some()` so CLI training and benchmarks do not pay for dashboard-only metrics.
- Baseline entering the current `train_batch_core` work:
  - Full benchmark: `81.818s` total, `63.869s` train batch core, `14.609s` evaluation.
  - Simulated benchmark: `252.262s` total, `141.061s` train batch core, `88.134s` evaluation runtime choice, `21.132s` evaluation.
- Accepted optimization iteration 1:
  - In [network.rs](/home/mikael/iaj/true_game/iaj_evades-optimization_attempt/rust_evades_dqn/src/network.rs:108) switched parallel batch accumulation to Rayon `fold`/`reduce` with per-worker `BatchGradients`.
  - In [network.rs](/home/mikael/iaj/true_game/iaj_evades-optimization_attempt/rust_evades_dqn/src/network.rs:208) removed cloned output activations during backprop and collapsed delta propagation to a single vector.
  - In [network.rs](/home/mikael/iaj/true_game/iaj_evades-optimization_attempt/rust_evades_dqn/src/network.rs:300) added `max_predict()` so TD targets do max-only inference instead of allocating full output vectors.
  - Added an equivalence test for `max_predict()` in [network.rs](/home/mikael/iaj/true_game/iaj_evades-optimization_attempt/rust_evades_dqn/src/network.rs:373).
  - Result:
    - Full benchmark: `67.474s` total, `54.798s` train batch core.
    - Simulated benchmark: `196.520s` total, `130.102s` train batch core.
  - Impact:
    - Full total: `-17.5%`
    - Full train batch core: `-14.2%`
    - Simulated total: `-22.1%`
    - Simulated train batch core: `-7.8%`
- Subagent review after iteration 1 recommended:
  - tuning reduction bandwidth / chunk size in parallel training,
  - vectorized dense math kernels,
  - reducing runtime chooser overhead.
- Accepted optimization iteration 2:
  - In [trainer.rs](/home/mikael/iaj/true_game/iaj_evades-optimization_attempt/rust_evades_dqn/src/trainer.rs:596) changed optimizer runtime selection to benchmark several chunk-size candidates and keep the fastest one.
  - Current chooser selects parallel mode with `chunk size 15` on this machine.
  - Result versus accepted iteration 1 baseline:
    - Full benchmark: `65.664s` total, `52.634s` train batch core.
    - Simulated benchmark: `191.034s` total, `124.787s` train batch core.
  - Impact:
    - Full total: `-2.7%`
    - Full train batch core: `-3.9%`
    - Simulated total: `-2.8%`
    - Simulated train batch core: `-4.1%`
- Attempted and rejected:
  - broad scratch-buffer allocation cleanup in `network.rs`: regressed, reverted.
  - ref-based sampled-transition batches: no consistent win, reverted.
- Subagent review after iteration 2 recommended:
  - transposed layer weights for contiguous backprop delta reads,
  - flattening `BatchGradients` buffers to remove nested `Vec<Vec<f32>>`,
  - reducing evaluation/optimizer runtime chooser overhead with caching or cheaper probes.

## Next Target

- Next likely optimization target: add transposed weights for the backprop delta pass in `network.rs`, because `train_batch_core` is still dominant and that loop still uses cache-unfriendly strided reads.
