use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::{
    game::{Action, GameState},
    sensing::{ObservationBuilder, INPUT_SIZE},
};

#[derive(Clone, Debug)]
pub struct ModelController {
    network: CompiledDqnNetwork,
    observation: ObservationBuilder,
}

impl ModelController {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = fs::read_to_string(path)
            .with_context(|| format!("failed to read model {}", path.display()))?;
        let network = CompiledDqnNetwork::load(&json)?;

        Ok(Self {
            network,
            observation: ObservationBuilder::default(),
        })
    }

    pub fn reset(&mut self, state: &GameState) {
        self.observation.reset(state);
    }

    pub fn choose_action(&mut self, state: &GameState) -> Action {
        let inputs = self.observation.build(state);
        let outputs = self.network.activate(&inputs);
        let index = outputs
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        Action::ALL[index]
    }
}

#[derive(Clone, Debug, Deserialize)]
struct DqnLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct DqnSavedModel {
    model_type: String,
    input_size: usize,
    output_size: usize,
    layers: Vec<DqnLayer>,
}

#[derive(Clone, Debug)]
struct CompiledDqnNetwork {
    layers: Vec<DqnLayer>,
}

impl CompiledDqnNetwork {
    fn load(json: &str) -> Result<Self> {
        let saved: DqnSavedModel =
            serde_json::from_str(json).context("failed to decode DQN model")?;
        if saved.model_type != "dqn" {
            anyhow::bail!("unsupported model type {}", saved.model_type);
        }
        if saved.input_size != INPUT_SIZE {
            anyhow::bail!(
                "model input size {} does not match expected {}",
                saved.input_size,
                INPUT_SIZE
            );
        }
        if saved.output_size != Action::ALL.len() {
            anyhow::bail!(
                "model output size {} does not match expected {}",
                saved.output_size,
                Action::ALL.len()
            );
        }
        Ok(Self {
            layers: saved.layers,
        })
    }

    fn activate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut activations = inputs.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut next = vec![0.0; layer.output_size];
            for out in 0..layer.output_size {
                let mut sum = layer.biases[out];
                let row = out * layer.input_size;
                for input in 0..layer.input_size {
                    sum += layer.weights[row + input] * activations[input];
                }
                next[out] = if index + 1 == self.layers.len() {
                    sum
                } else {
                    sum.max(0.0)
                };
            }
            activations = next;
        }
        activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::GameConfig, game::GameState};

    #[test]
    fn observation_shape_matches_training() {
        let state = GameState::new(GameConfig::default(), Some(2));
        let mut builder = ObservationBuilder::default();
        let observation = builder.build(&state);
        assert_eq!(observation.len(), INPUT_SIZE);
    }
}
