use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

#[derive(Clone, Debug)]
struct ForwardCache {
    activations: Vec<Vec<f32>>,
    pre_activations: Vec<Vec<f32>>,
}

impl Network {
    pub fn new(sizes: &[usize], rng: &mut impl Rng) -> Self {
        let mut layers = Vec::with_capacity(sizes.len().saturating_sub(1));
        for window in sizes.windows(2) {
            let input_size = window[0];
            let output_size = window[1];
            let scale = (2.0 / input_size as f32).sqrt();
            let weights = (0..input_size * output_size)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            let biases = vec![0.0; output_size];
            layers.push(Layer {
                input_size,
                output_size,
                weights,
                biases,
            });
        }
        Self { layers }
    }

    pub fn from_layers(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        self.forward(input).activations.pop().unwrap_or_default()
    }

    pub fn train_batch(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
    ) -> f32 {
        let mut weight_grads = self
            .layers
            .iter()
            .map(|layer| vec![0.0; layer.weights.len()])
            .collect::<Vec<_>>();
        let mut bias_grads = self
            .layers
            .iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect::<Vec<_>>();
        let mut total_loss = 0.0;

        for transition in batch {
            let cache = self.forward(&transition.state);
            let q_values = cache.activations.last().cloned().unwrap_or_default();
            let next_q_values = if transition.done {
                vec![0.0; q_values.len()]
            } else {
                target_network.predict(&transition.next_state)
            };
            let next_best = next_q_values
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0);
            let target = transition.reward
                + if transition.done {
                    0.0
                } else {
                    gamma * next_best
                };
            let prediction = q_values[transition.action];
            let diff = prediction - target;
            total_loss += diff * diff;

            let mut deltas = vec![vec![0.0; q_values.len()]];
            deltas[0][transition.action] = 2.0 * diff;

            for layer_index in (0..self.layers.len()).rev() {
                let delta = deltas.pop().unwrap();
                let activation_input = &cache.activations[layer_index];
                for out in 0..self.layers[layer_index].output_size {
                    bias_grads[layer_index][out] += delta[out];
                    let row = out * self.layers[layer_index].input_size;
                    for input in 0..self.layers[layer_index].input_size {
                        weight_grads[layer_index][row + input] +=
                            delta[out] * activation_input[input];
                    }
                }

                if layer_index > 0 {
                    let mut previous_delta = vec![0.0; self.layers[layer_index].input_size];
                    for input in 0..self.layers[layer_index].input_size {
                        let mut sum = 0.0;
                        for out in 0..self.layers[layer_index].output_size {
                            let row = out * self.layers[layer_index].input_size;
                            sum += self.layers[layer_index].weights[row + input] * delta[out];
                        }
                        let z = cache.pre_activations[layer_index - 1][input];
                        previous_delta[input] = if z > 0.0 { sum } else { 0.0 };
                    }
                    deltas.push(previous_delta);
                }
            }
        }

        let batch_scale = 1.0 / batch.len().max(1) as f32;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            for (weight, grad) in layer
                .weights
                .iter_mut()
                .zip(weight_grads[layer_index].iter())
            {
                *weight -= learning_rate * grad * batch_scale;
            }
            for (bias, grad) in layer.biases.iter_mut().zip(bias_grads[layer_index].iter()) {
                *bias -= learning_rate * grad * batch_scale;
            }
        }

        total_loss * batch_scale
    }

    fn forward(&self, input: &[f32]) -> ForwardCache {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        activations.push(input.to_vec());

        for (index, layer) in self.layers.iter().enumerate() {
            let previous = activations.last().cloned().unwrap_or_default();
            let mut z = vec![0.0; layer.output_size];
            let mut a = vec![0.0; layer.output_size];
            for out in 0..layer.output_size {
                let row = out * layer.input_size;
                let mut sum = layer.biases[out];
                for input_index in 0..layer.input_size {
                    sum += layer.weights[row + input_index] * previous[input_index];
                }
                z[out] = sum;
                a[out] = if index + 1 == self.layers.len() {
                    sum
                } else {
                    sum.max(0.0)
                };
            }
            pre_activations.push(z);
            activations.push(a);
        }

        ForwardCache {
            activations,
            pre_activations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::INPUT_SIZE;
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

    #[test]
    fn network_outputs_expected_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let network = Network::new(&[INPUT_SIZE, 32, 9], &mut rng);
        let output = network.predict(&vec![0.0; INPUT_SIZE]);
        assert_eq!(output.len(), 9);
    }
}
