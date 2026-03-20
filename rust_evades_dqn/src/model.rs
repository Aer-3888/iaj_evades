use serde::{Deserialize, Serialize};

use crate::{
    network::Layer,
    observation::{INPUT_SIZE, RAY_COUNT, DQN2_INPUT_SIZE, DQN2_RAY_COUNT},
};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EvaluationSummary {
    pub average_survival_time: f32,
    pub average_return: f32,
    pub average_evades: f32,
    pub min_survival_time: f32,
    pub min_return: f32,
    pub timeouts: u32,
}

/// The model type identifier stored in the JSON payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    #[default]
    Dqn,
    Dqn2,
}

impl ModelType {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelType::Dqn => "dqn",
            ModelType::Dqn2 => "dqn2",
        }
    }

    pub fn input_size(self) -> usize {
        match self {
            ModelType::Dqn => INPUT_SIZE,
            ModelType::Dqn2 => DQN2_INPUT_SIZE,
        }
    }

    pub fn ray_count(self) -> usize {
        match self {
            ModelType::Dqn => RAY_COUNT,
            ModelType::Dqn2 => DQN2_RAY_COUNT,
        }
    }
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
        model_kind: ModelType,
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
            model_type: model_kind.as_str().to_string(),
            format_version: 4,
            input_size: model_kind.input_size(),
            output_size: 9,
            ray_count: model_kind.ray_count(),
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
