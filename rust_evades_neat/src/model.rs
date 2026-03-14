use serde::{Deserialize, Serialize};

use crate::{
    neat::{EvaluationSummary, Genome},
    observation::{INPUT_SIZE, RAY_COUNT},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedModel {
    pub format_version: u32,
    pub input_size: usize,
    pub output_size: usize,
    pub ray_count: usize,
    pub training_seeds: Vec<u64>,
    pub generations_completed: usize,
    pub best_metrics: EvaluationSummary,
    pub genome: Genome,
}

impl SavedModel {
    pub fn new(
        training_seeds: Vec<u64>,
        generations_completed: usize,
        best_metrics: EvaluationSummary,
        genome: Genome,
    ) -> Self {
        Self {
            format_version: 1,
            input_size: INPUT_SIZE,
            output_size: 9,
            ray_count: RAY_COUNT,
            training_seeds,
            generations_completed,
            best_metrics,
            genome,
        }
    }
}
