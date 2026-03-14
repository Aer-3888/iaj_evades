use std::collections::HashMap;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeKind {
    Input,
    Bias,
    Hidden,
    Output,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u64,
    pub kind: NodeKind,
    pub order: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub innovation: u64,
    pub input: u64,
    pub output: u64,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct EvaluationSummary {
    pub fitness: f32,
    pub average_progress: f32,
    pub average_rightward_reward: f32,
    pub wins: u32,
}

#[derive(Clone, Debug)]
pub struct MutationConfig {
    pub weight_perturb_chance: f32,
    pub weight_reset_chance: f32,
    pub add_connection_chance: f32,
    pub add_node_chance: f32,
    pub toggle_connection_chance: f32,
    pub mutate_only_chance: f32,
    pub crossover_chance: f32,
    pub compatibility_threshold: f32,
    pub compatibility_disjoint: f32,
    pub compatibility_excess: f32,
    pub compatibility_weight: f32,
    pub elite_per_species: usize,
    pub tournament_size: usize,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            weight_perturb_chance: 0.85,
            weight_reset_chance: 0.10,
            add_connection_chance: 0.20,
            add_node_chance: 0.08,
            toggle_connection_chance: 0.02,
            mutate_only_chance: 0.25,
            crossover_chance: 0.75,
            compatibility_threshold: 3.0,
            compatibility_disjoint: 1.0,
            compatibility_excess: 1.0,
            compatibility_weight: 0.4,
            elite_per_species: 1,
            tournament_size: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct InnovationTracker {
    input_ids: Vec<u64>,
    bias_id: u64,
    output_ids: Vec<u64>,
    next_node_id: u64,
    next_innovation: u64,
    connection_innovations: HashMap<(u64, u64), u64>,
}

impl InnovationTracker {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        let input_ids = (0..input_count as u64).collect::<Vec<_>>();
        let bias_id = input_count as u64;
        let output_ids = ((input_count as u64 + 1)..(input_count as u64 + 1 + output_count as u64))
            .collect::<Vec<_>>();

        Self {
            input_ids,
            bias_id,
            output_ids,
            next_node_id: input_count as u64 + 1 + output_count as u64,
            next_innovation: 0,
            connection_innovations: HashMap::new(),
        }
    }

    pub fn initial_genome(&mut self, rng: &mut impl Rng) -> Genome {
        let mut nodes = Vec::with_capacity(self.input_ids.len() + 1 + self.output_ids.len());
        for id in &self.input_ids {
            nodes.push(NodeGene {
                id: *id,
                kind: NodeKind::Input,
                order: 0.0,
            });
        }
        nodes.push(NodeGene {
            id: self.bias_id,
            kind: NodeKind::Bias,
            order: 0.0,
        });
        for id in &self.output_ids {
            nodes.push(NodeGene {
                id: *id,
                kind: NodeKind::Output,
                order: 1.0,
            });
        }

        let all_inputs = self
            .input_ids
            .iter()
            .copied()
            .chain(std::iter::once(self.bias_id))
            .collect::<Vec<_>>();
        let output_ids = self.output_ids.clone();
        let mut connections = Vec::with_capacity(all_inputs.len() * output_ids.len());
        for input in all_inputs {
            for output in &output_ids {
                connections.push(ConnectionGene {
                    innovation: self.connection_innovation(input, *output),
                    input,
                    output: *output,
                    weight: rng.gen_range(-1.0..1.0),
                    enabled: true,
                });
            }
        }

        let mut genome = Genome {
            nodes,
            connections,
            fitness: 0.0,
        };
        genome.sort_genes();
        genome
    }

    pub fn alloc_node_id(&mut self) -> u64 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    pub fn connection_innovation(&mut self, input: u64, output: u64) -> u64 {
        if let Some(existing) = self.connection_innovations.get(&(input, output)) {
            *existing
        } else {
            let innovation = self.next_innovation;
            self.next_innovation += 1;
            self.connection_innovations
                .insert((input, output), innovation);
            innovation
        }
    }
}

impl Genome {
    pub fn sort_genes(&mut self) {
        self.nodes.sort_by(|left, right| {
            left.order
                .partial_cmp(&right.order)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.id.cmp(&right.id))
        });
        self.connections
            .sort_by_key(|connection| connection.innovation);
    }

    pub fn compile(&self, input_count: usize) -> CompiledNetwork {
        CompiledNetwork::from_genome(self, input_count)
    }

    pub fn mutate(
        &mut self,
        rng: &mut impl Rng,
        tracker: &mut InnovationTracker,
        config: &MutationConfig,
    ) {
        if rng.gen::<f32>() < config.weight_perturb_chance {
            self.mutate_weights(rng, config.weight_reset_chance);
        }
        if rng.gen::<f32>() < config.add_connection_chance {
            self.add_connection(rng, tracker);
        }
        if rng.gen::<f32>() < config.add_node_chance {
            self.add_node(rng, tracker);
        }
        if rng.gen::<f32>() < config.toggle_connection_chance {
            self.toggle_random_connection(rng);
        }
        self.sort_genes();
    }

    pub fn crossover(fitter: &Genome, other: &Genome, rng: &mut impl Rng) -> Genome {
        let mut other_by_innovation = HashMap::with_capacity(other.connections.len());
        for connection in &other.connections {
            other_by_innovation.insert(connection.innovation, connection);
        }

        let mut chosen_connections = Vec::with_capacity(fitter.connections.len());
        for connection in &fitter.connections {
            let chosen = if let Some(matching) = other_by_innovation.get(&connection.innovation) {
                if rng.gen_bool(0.5) {
                    (*matching).clone()
                } else {
                    connection.clone()
                }
            } else {
                connection.clone()
            };
            chosen_connections.push(chosen);
        }

        let mut used_nodes = HashMap::new();
        for node in fitter.nodes.iter().chain(other.nodes.iter()) {
            used_nodes.entry(node.id).or_insert_with(|| node.clone());
        }

        let mut child = Genome {
            nodes: used_nodes.into_values().collect(),
            connections: chosen_connections,
            fitness: 0.0,
        };
        child.sort_genes();
        child
    }

    pub fn compatibility_distance(&self, other: &Genome, config: &MutationConfig) -> f32 {
        let mut left = 0;
        let mut right = 0;
        let mut excess = 0.0;
        let mut disjoint = 0.0;
        let mut matching = 0.0;
        let mut weight_difference = 0.0;

        while left < self.connections.len() && right < other.connections.len() {
            let left_gene = &self.connections[left];
            let right_gene = &other.connections[right];
            if left_gene.innovation == right_gene.innovation {
                matching += 1.0;
                weight_difference += (left_gene.weight - right_gene.weight).abs();
                left += 1;
                right += 1;
            } else if left_gene.innovation < right_gene.innovation {
                disjoint += 1.0;
                left += 1;
            } else {
                disjoint += 1.0;
                right += 1;
            }
        }

        excess += (self.connections.len() - left) as f32;
        excess += (other.connections.len() - right) as f32;
        let normalizer = self.connections.len().max(other.connections.len()).max(1) as f32;
        let average_weight_difference = if matching > 0.0 {
            weight_difference / matching
        } else {
            0.0
        };

        config.compatibility_excess * excess / normalizer
            + config.compatibility_disjoint * disjoint / normalizer
            + config.compatibility_weight * average_weight_difference
    }

    fn mutate_weights(&mut self, rng: &mut impl Rng, reset_chance: f32) {
        for connection in &mut self.connections {
            if rng.gen::<f32>() < reset_chance {
                connection.weight = rng.gen_range(-1.0..1.0);
            } else {
                connection.weight += rng.gen_range(-0.4..0.4);
            }
            connection.weight = connection.weight.clamp(-3.0, 3.0);
        }
    }

    fn add_connection(&mut self, rng: &mut impl Rng, tracker: &mut InnovationTracker) {
        for _ in 0..64 {
            let from = &self.nodes[rng.gen_range(0..self.nodes.len())];
            let to = &self.nodes[rng.gen_range(0..self.nodes.len())];

            if matches!(from.kind, NodeKind::Output)
                || matches!(to.kind, NodeKind::Input | NodeKind::Bias)
            {
                continue;
            }
            if from.order + 1.0e-6 >= to.order {
                continue;
            }
            if self
                .connections
                .iter()
                .any(|connection| connection.input == from.id && connection.output == to.id)
            {
                continue;
            }

            self.connections.push(ConnectionGene {
                innovation: tracker.connection_innovation(from.id, to.id),
                input: from.id,
                output: to.id,
                weight: rng.gen_range(-1.0..1.0),
                enabled: true,
            });
            return;
        }
    }

    fn add_node(&mut self, rng: &mut impl Rng, tracker: &mut InnovationTracker) {
        let enabled_indices = self
            .connections
            .iter()
            .enumerate()
            .filter_map(|(index, connection)| connection.enabled.then_some(index))
            .collect::<Vec<_>>();
        if enabled_indices.is_empty() {
            return;
        }

        let split_index = enabled_indices[rng.gen_range(0..enabled_indices.len())];
        let split_connection = self.connections[split_index].clone();
        self.connections[split_index].enabled = false;

        let Some(input_node) = self
            .nodes
            .iter()
            .find(|node| node.id == split_connection.input)
        else {
            return;
        };
        let Some(output_node) = self
            .nodes
            .iter()
            .find(|node| node.id == split_connection.output)
        else {
            return;
        };

        let new_order = (input_node.order + output_node.order) * 0.5;
        let new_node_id = tracker.alloc_node_id();
        self.nodes.push(NodeGene {
            id: new_node_id,
            kind: NodeKind::Hidden,
            order: new_order,
        });
        self.connections.push(ConnectionGene {
            innovation: tracker.connection_innovation(split_connection.input, new_node_id),
            input: split_connection.input,
            output: new_node_id,
            weight: 1.0,
            enabled: true,
        });
        self.connections.push(ConnectionGene {
            innovation: tracker.connection_innovation(new_node_id, split_connection.output),
            input: new_node_id,
            output: split_connection.output,
            weight: split_connection.weight,
            enabled: true,
        });
    }

    fn toggle_random_connection(&mut self, rng: &mut impl Rng) {
        if let Some(connection) = self.connections.choose_mut(rng) {
            connection.enabled = !connection.enabled;
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompiledNetwork {
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

    pub fn activate(&self, inputs: &[f32]) -> Vec<f32> {
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

pub fn speciate(population: &[Genome], config: &MutationConfig) -> Vec<Species> {
    let mut species = Vec::<Species>::new();

    for (index, genome) in population.iter().enumerate() {
        if let Some(existing) = species.iter_mut().find(|species| {
            genome.compatibility_distance(&species.representative, config)
                < config.compatibility_threshold
        }) {
            existing.members.push(index);
        } else {
            species.push(Species {
                representative: genome.clone(),
                members: vec![index],
            });
        }
    }

    species
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_network_outputs_expected_count() {
        let mut tracker = InnovationTracker::new(73, 9);
        let mut rng = StdRng::seed_from_u64(7);
        let genome = tracker.initial_genome(&mut rng);
        let network = genome.compile(73);
        let outputs = network.activate(&vec![0.0; 73]);
        assert_eq!(outputs.len(), 9);
    }
}
