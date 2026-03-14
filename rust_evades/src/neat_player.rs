use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::game::{Action, GameState};

const RAY_COUNT: usize = 36;
const INPUT_SIZE: usize = RAY_COUNT * 2 + 1;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
enum NodeKind {
    Input,
    Bias,
    Hidden,
    Output,
}

#[derive(Clone, Debug, Deserialize)]
struct NodeGene {
    id: u64,
    kind: NodeKind,
    order: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct ConnectionGene {
    input: u64,
    output: u64,
    weight: f32,
    enabled: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct Genome {
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

#[derive(Clone, Debug, Deserialize)]
struct SavedModel {
    input_size: usize,
    output_size: usize,
    genome: Genome,
}

#[derive(Clone, Debug)]
pub struct ModelController {
    network: CompiledNetwork,
    observation: ObservationBuilder,
}

impl ModelController {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = fs::read_to_string(path)
            .with_context(|| format!("failed to read model {}", path.display()))?;
        let saved: SavedModel = serde_json::from_str(&json)
            .with_context(|| format!("failed to parse model {}", path.display()))?;

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
            network: CompiledNetwork::from_genome(&saved.genome, INPUT_SIZE),
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

#[derive(Clone, Debug)]
struct ObservationBuilder {
    previous_rays: [f32; RAY_COUNT],
    previous_x: f32,
    initialized: bool,
}

impl Default for ObservationBuilder {
    fn default() -> Self {
        Self {
            previous_rays: [0.0; RAY_COUNT],
            previous_x: 0.0,
            initialized: false,
        }
    }
}

impl ObservationBuilder {
    fn reset(&mut self, state: &GameState) {
        self.previous_rays = sample_rays(state);
        self.previous_x = state.player.body.pos.x;
        self.initialized = true;
    }

    fn build(&mut self, state: &GameState) -> [f32; INPUT_SIZE] {
        if !self.initialized {
            self.reset(state);
        }

        let rays = sample_rays(state);
        let mut observation = [0.0; INPUT_SIZE];
        observation[..RAY_COUNT].copy_from_slice(&rays);

        for index in 0..RAY_COUNT {
            observation[RAY_COUNT + index] = rays[index] - self.previous_rays[index];
        }

        let max_step = state.config.player_speed * state.config.fixed_timestep;
        observation[INPUT_SIZE - 1] = if max_step > 0.0 {
            ((state.player.body.pos.x - self.previous_x) / max_step).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        self.previous_rays = rays;
        self.previous_x = state.player.body.pos.x;
        observation
    }
}

fn sample_rays(state: &GameState) -> [f32; RAY_COUNT] {
    let mut samples = [0.0; RAY_COUNT];
    let origin_x = state.player.body.pos.x;
    let origin_y = state.player.body.pos.y;
    let max_distance = (state.config.world_width.powi(2) + state.config.corridor_height().powi(2))
        .sqrt()
        .max(1.0);

    for (index, sample) in samples.iter_mut().enumerate() {
        let angle = (index as f32) * 10.0_f32.to_radians();
        let dir_x = angle.cos();
        let dir_y = -angle.sin();

        let wall_distance = raycast_wall_distance(state, origin_x, origin_y, dir_x, dir_y);
        let enemy_distance = state
            .enemies
            .iter()
            .filter_map(|enemy| {
                raycast_circle_distance(
                    origin_x,
                    origin_y,
                    dir_x,
                    dir_y,
                    enemy.body.pos.x,
                    enemy.body.pos.y,
                    enemy.body.radius,
                )
            })
            .fold(f32::INFINITY, f32::min);

        let hit_distance = wall_distance.min(enemy_distance);
        let clearance = (hit_distance - state.player.body.radius).max(0.0);
        *sample = (clearance / max_distance).clamp(0.0, 1.0);
    }

    samples
}

fn raycast_wall_distance(
    state: &GameState,
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
) -> f32 {
    let mut best = f32::INFINITY;
    let epsilon = 1.0e-6;

    if dir_x.abs() > epsilon {
        if dir_x > 0.0 {
            best = best.min((state.config.world_width - origin_x) / dir_x);
        } else {
            best = best.min((0.0 - origin_x) / dir_x);
        }
    }

    if dir_y.abs() > epsilon {
        if dir_y > 0.0 {
            best = best.min((state.config.corridor_bottom - origin_y) / dir_y);
        } else {
            best = best.min((state.config.corridor_top - origin_y) / dir_y);
        }
    }

    best.max(0.0)
}

fn raycast_circle_distance(
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Option<f32> {
    let offset_x = origin_x - center_x;
    let offset_y = origin_y - center_y;
    let projection = offset_x * dir_x + offset_y * dir_y;
    let c = offset_x * offset_x + offset_y * offset_y - radius * radius;

    if c > 0.0 && projection > 0.0 {
        return None;
    }

    let discriminant = projection * projection - c;
    if discriminant < 0.0 {
        return None;
    }

    let mut distance = -projection - discriminant.sqrt();
    if distance < 0.0 {
        distance = 0.0;
    }
    Some(distance)
}

#[derive(Clone, Debug)]
struct CompiledNetwork {
    node_kinds: Vec<NodeKind>,
    input_positions: Vec<usize>,
    output_positions: Vec<usize>,
    incoming: Vec<Vec<(usize, f32)>>,
}

impl CompiledNetwork {
    fn from_genome(genome: &Genome, input_count: usize) -> Self {
        let mut nodes = genome.nodes.clone();
        nodes.sort_by(|left, right| {
            left.order
                .partial_cmp(&right.order)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.id.cmp(&right.id))
        });

        let mut node_indices = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            node_indices.insert(node.id, index);
        }

        let mut incoming = vec![Vec::new(); nodes.len()];
        for connection in genome
            .connections
            .iter()
            .filter(|connection| connection.enabled)
        {
            if let (Some(&src), Some(&dst)) = (
                node_indices.get(&connection.input),
                node_indices.get(&connection.output),
            ) {
                incoming[dst].push((src, connection.weight));
            }
        }

        let input_positions = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| matches!(node.kind, NodeKind::Input).then_some(index))
            .take(input_count)
            .collect::<Vec<_>>();
        let output_positions = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| matches!(node.kind, NodeKind::Output).then_some(index))
            .collect::<Vec<_>>();

        Self {
            node_kinds: nodes.iter().map(|node| node.kind).collect(),
            input_positions,
            output_positions,
            incoming,
        }
    }

    fn activate(&self, inputs: &[f32]) -> Vec<f32> {
        let mut values = vec![0.0; self.node_kinds.len()];
        for (input_index, &position) in self.input_positions.iter().enumerate() {
            values[position] = inputs.get(input_index).copied().unwrap_or_default();
        }

        for (index, kind) in self.node_kinds.iter().enumerate() {
            match kind {
                NodeKind::Input => {}
                NodeKind::Bias => values[index] = 1.0,
                NodeKind::Hidden | NodeKind::Output => {
                    let mut sum = 0.0;
                    for &(source, weight) in &self.incoming[index] {
                        sum += values[source] * weight;
                    }
                    values[index] = sum.tanh();
                }
            }
        }

        self.output_positions
            .iter()
            .map(|&position| values[position])
            .collect()
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
