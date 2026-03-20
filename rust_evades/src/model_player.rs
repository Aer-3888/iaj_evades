use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::{
    game::{Action, GameState},
    sensing::{DualRayObservationBuilder, ObservationBuilder, DQN2_INPUT_SIZE, INPUT_SIZE},
};

/// Which input/observation schema the model expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Dqn,
    Dqn2,
}

/// Holds the correct observation builder for the loaded model kind.
#[derive(Clone, Debug)]
enum ObsState {
    Dqn(ObservationBuilder),
    Dqn2(DualRayObservationBuilder),
}

impl ObsState {
    fn reset(&mut self, state: &GameState) {
        match self {
            ObsState::Dqn(b) => b.reset(state),
            ObsState::Dqn2(b) => b.reset(state),
        }
    }

    fn build_vec(&mut self, state: &GameState) -> Vec<f32> {
        match self {
            ObsState::Dqn(b) => b.build(state).to_vec(),
            ObsState::Dqn2(b) => b.build(state).to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelController {
    network: CompiledDqnNetwork,
    obs: ObsState,
    pub kind: ModelKind,
}

impl ModelController {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = fs::read_to_string(path)
            .with_context(|| format!("failed to read model {}", path.display()))?;
        let network = CompiledDqnNetwork::load(&json)?;
        let kind = network.kind;
        let obs = match kind {
            ModelKind::Dqn => ObsState::Dqn(ObservationBuilder::default()),
            ModelKind::Dqn2 => ObsState::Dqn2(DualRayObservationBuilder::default()),
        };
        Ok(Self { network, obs, kind })
    }

    pub fn reset(&mut self, state: &GameState) {
        self.obs.reset(state);
    }

    pub fn choose_action(&mut self, state: &GameState) -> Action {
        let inputs = self.obs.build_vec(state);
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
    kind: ModelKind,
}

impl CompiledDqnNetwork {
    fn load(json: &str) -> Result<Self> {
        let saved: DqnSavedModel =
            serde_json::from_str(json).context("failed to decode DQN model")?;

        let (kind, expected_input) = match saved.model_type.as_str() {
            "dqn" => (ModelKind::Dqn, INPUT_SIZE),
            "dqn2" => (ModelKind::Dqn2, DQN2_INPUT_SIZE),
            other => anyhow::bail!("unsupported model type {}", other),
        };

        if saved.input_size != expected_input {
            anyhow::bail!(
                "model input size {} does not match expected {} for type {}",
                saved.input_size,
                expected_input,
                saved.model_type
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
            kind,
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

    #[test]
    fn dual_observation_shape_matches_dqn2_input_size() {
        let state = GameState::new(GameConfig::default(), Some(2));
        let mut builder = DualRayObservationBuilder::default();
        let observation = builder.build(&state);
        assert_eq!(observation.len(), DQN2_INPUT_SIZE);
    }
}
