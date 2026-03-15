use serde::{Deserialize, Serialize};

use crate::{
    network::Layer,
    observation::{INPUT_SIZE, RAY_COUNT},
};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EvaluationSummary {
    pub average_survival_time: f32,
    pub average_return: f32,
    pub average_evades: f32,
    pub timeouts: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedModel {
    pub model_type: String,
    pub format_version: u32,
    pub input_size: usize,
    pub output_size: usize,
    pub ray_count: usize,
    pub hidden_sizes: Vec<usize>,
    pub training_seeds: Vec<u64>,
    pub random_seeds_per_cycle: usize,
    pub action_repeat: usize,
    pub episodes_completed: usize,
    #[serde(default)]
    pub total_steps_completed: usize,
    pub best_metrics: EvaluationSummary,
    pub layers: Vec<Layer>,
}

impl SavedModel {
    pub fn new(
        hidden_sizes: Vec<usize>,
        training_seeds: Vec<u64>,
        random_seeds_per_cycle: usize,
        action_repeat: usize,
        episodes_completed: usize,
        total_steps_completed: usize,
        best_metrics: EvaluationSummary,
        layers: Vec<Layer>,
    ) -> Self {
        Self {
            model_type: "dqn".to_string(),
            format_version: 3,
            input_size: INPUT_SIZE,
            output_size: 9,
            ray_count: RAY_COUNT,
            hidden_sizes,
            training_seeds,
            random_seeds_per_cycle,
            action_repeat,
            episodes_completed,
            total_steps_completed,
            best_metrics,
            layers,
        }
    }
}
