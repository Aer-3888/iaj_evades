use rand::Rng;
use rayon::prelude::*;
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

#[derive(Clone, Debug)]
struct BatchGradients {
    weight_grads: Vec<Vec<f32>>,
    bias_grads: Vec<Vec<f32>>,
    total_loss: f32,
}

impl BatchGradients {
    fn zero_for(layers: &[Layer]) -> Self {
        Self {
            weight_grads: layers
                .iter()
                .map(|layer| vec![0.0; layer.weights.len()])
                .collect(),
            bias_grads: layers
                .iter()
                .map(|layer| vec![0.0; layer.biases.len()])
                .collect(),
            total_loss: 0.0,
        }
    }

    fn merge(&mut self, other: Self) {
        self.total_loss += other.total_loss;
        for (left, right) in self.weight_grads.iter_mut().zip(other.weight_grads) {
            for (left_grad, right_grad) in left.iter_mut().zip(right) {
                *left_grad += right_grad;
            }
        }
        for (left, right) in self.bias_grads.iter_mut().zip(other.bias_grads) {
            for (left_grad, right_grad) in left.iter_mut().zip(right) {
                *left_grad += right_grad;
            }
        }
    }
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
        huber_delta: f32,
        gradient_clip_norm: f32,
    ) -> f32 {
        let mut gradients = self.accumulate_batch(target_network, batch, gamma, huber_delta);
        clip_global_norm(
            &mut gradients.weight_grads,
            &mut gradients.bias_grads,
            gradient_clip_norm,
        );
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }

    pub fn train_batch_parallel(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
        chunk_size: usize,
    ) -> f32 {
        let network = &*self;
        let mut gradients = batch
            .par_chunks(chunk_size.max(1))
            .fold(
                || BatchGradients::zero_for(&network.layers),
                |mut gradients, chunk| {
                    network.accumulate_batch_into(
                        target_network,
                        chunk,
                        gamma,
                        huber_delta,
                        &mut gradients,
                    );
                    gradients
                },
            )
            .reduce(
                || BatchGradients::zero_for(&network.layers),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            );
        clip_global_norm(
            &mut gradients.weight_grads,
            &mut gradients.bias_grads,
            gradient_clip_norm,
        );
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }

    fn apply_gradients(
        &mut self,
        gradients: BatchGradients,
        batch_len: usize,
        learning_rate: f32,
    ) -> f32 {
        let batch_scale = 1.0 / batch_len.max(1) as f32;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            for (weight, grad) in layer
                .weights
                .iter_mut()
                .zip(gradients.weight_grads[layer_index].iter())
            {
                *weight -= learning_rate * grad * batch_scale;
            }
            for (bias, grad) in layer
                .biases
                .iter_mut()
                .zip(gradients.bias_grads[layer_index].iter())
            {
                *bias -= learning_rate * grad * batch_scale;
            }
        }

        gradients.total_loss * batch_scale
    }

    fn accumulate_batch(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        huber_delta: f32,
    ) -> BatchGradients {
        let mut gradients = BatchGradients::zero_for(&self.layers);
        self.accumulate_batch_into(target_network, batch, gamma, huber_delta, &mut gradients);
        gradients
    }

    fn accumulate_batch_into(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        huber_delta: f32,
        gradients: &mut BatchGradients,
    ) {
        for transition in batch {
            gradients.total_loss += self.accumulate_transition(
                target_network,
                transition,
                gamma,
                huber_delta,
                &mut gradients.weight_grads,
                &mut gradients.bias_grads,
            );
        }
    }

    fn accumulate_transition(
        &self,
        target_network: &Network,
        transition: &super::trainer::Transition,
        gamma: f32,
        huber_delta: f32,
        weight_grads: &mut [Vec<f32>],
        bias_grads: &mut [Vec<f32>],
    ) -> f32 {
        let cache = self.forward(&transition.state);
        let q_values = cache.activations.last().map(Vec::as_slice).unwrap_or(&[]);
        let next_best = if transition.done {
            0.0
        } else {
            target_network.max_predict(&transition.next_state)
        };
        let target = transition.reward
            + if transition.done {
                0.0
            } else {
                gamma * next_best
            };
        let prediction = q_values[transition.action];
        let diff = prediction - target;

        let (loss, grad_wrt_prediction) = huber(diff, huber_delta);

        let mut delta = vec![0.0; q_values.len()];
        delta[transition.action] = grad_wrt_prediction;

        for layer_index in (0..self.layers.len()).rev() {
            let activation_input = &cache.activations[layer_index];
            let layer = &self.layers[layer_index];

            for out in 0..layer.output_size {
                bias_grads[layer_index][out] += delta[out];
                let row = out * layer.input_size;
                for input in 0..layer.input_size {
                    weight_grads[layer_index][row + input] += delta[out] * activation_input[input];
                }
            }

            if layer_index > 0 {
                let mut previous_delta = vec![0.0; layer.input_size];
                for input in 0..layer.input_size {
                    let mut sum = 0.0;
                    for out in 0..layer.output_size {
                        let row = out * layer.input_size;
                        sum += layer.weights[row + input] * delta[out];
                    }
                    let z = cache.pre_activations[layer_index - 1][input];
                    previous_delta[input] = if z > 0.0 { sum } else { 0.0 };
                }
                delta = previous_delta;
            }
        }

        loss
    }

    fn forward(&self, input: &[f32]) -> ForwardCache {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        activations.push(input.to_vec());

        for (index, layer) in self.layers.iter().enumerate() {
            let previous = activations.last().map(Vec::as_slice).unwrap_or(&[]);
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

    fn max_predict(&self, input: &[f32]) -> f32 {
        let mut current = input.to_vec();
        let mut next = Vec::new();

        for (index, layer) in self.layers.iter().enumerate() {
            next.clear();
            next.resize(layer.output_size, 0.0);
            for out in 0..layer.output_size {
                let row = out * layer.input_size;
                let mut sum = layer.biases[out];
                for input_index in 0..layer.input_size {
                    sum += layer.weights[row + input_index] * current[input_index];
                }
                next[out] = if index + 1 == self.layers.len() {
                    sum
                } else {
                    sum.max(0.0)
                };
            }
            std::mem::swap(&mut current, &mut next);
        }

        current.into_iter().fold(f32::NEG_INFINITY, f32::max)
    }
}

fn huber(diff: f32, delta: f32) -> (f32, f32) {
    let abs_diff = diff.abs();
    if abs_diff <= delta {
        (0.5 * diff * diff, diff)
    } else {
        (delta * (abs_diff - 0.5 * delta), delta * diff.signum())
    }
}

fn clip_global_norm(
    weight_grads: &mut [Vec<f32>],
    bias_grads: &mut [Vec<f32>],
    max_norm: f32,
) -> f32 {
    let mut sq_norm = 0.0;
    for layer in weight_grads.iter() {
        for &g in layer.iter() {
            sq_norm += g * g;
        }
    }
    for layer in bias_grads.iter() {
        for &g in layer.iter() {
            sq_norm += g * g;
        }
    }

    let norm = sq_norm.sqrt();
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-8);
        for layer in weight_grads.iter_mut() {
            for g in layer.iter_mut() {
                *g *= scale;
            }
        }
        for layer in bias_grads.iter_mut() {
            for g in layer.iter_mut() {
                *g *= scale;
            }
        }
    }
    norm
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

    #[test]
    fn max_predict_matches_predict_output_max() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let network = Network::new(&[INPUT_SIZE, 32, 16, 9], &mut rng);
        let input = (0..INPUT_SIZE)
            .map(|index| ((index as f32 * 0.17).sin() * 0.5) + 0.1)
            .collect::<Vec<_>>();

        let predicted_max = network
            .predict(&input)
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
        let direct_max = network.max_predict(&input);

        assert!((predicted_max - direct_max).abs() < 1.0e-5);
    }

    #[test]
    fn huber_small_error_regime() {
        let delta = 1.0;
        let diff = 0.5;
        let (loss, grad) = super::huber(diff, delta);
        assert_eq!(loss, 0.5 * 0.5 * 0.5);
        assert_eq!(grad, 0.5);
    }

    #[test]
    fn huber_large_error_regime() {
        let delta = 1.0;
        let diff = 2.0;
        let (loss, grad) = super::huber(diff, delta);
        assert_eq!(loss, 1.0 * (2.0 - 0.5 * 1.0));
        assert_eq!(grad, 1.0);

        let diff_neg = -3.0;
        let (loss_neg, grad_neg) = super::huber(diff_neg, delta);
        assert_eq!(loss_neg, 1.0 * (3.0 - 0.5 * 1.0));
        assert_eq!(grad_neg, -1.0);
    }

    #[test]
    fn test_clip_global_norm() {
        let mut weight_grads = vec![vec![3.0, 4.0]]; // norm = 5
        let mut bias_grads = vec![vec![0.0]];
        let pre_clip_norm = super::clip_global_norm(&mut weight_grads, &mut bias_grads, 10.0);
        assert_eq!(pre_clip_norm, 5.0);
        assert_eq!(weight_grads[0][0], 3.0);
        assert_eq!(weight_grads[0][1], 4.0);

        let mut weight_grads2 = vec![vec![6.0, 8.0]]; // norm = 10
        let mut bias_grads2 = vec![vec![0.0]];
        let pre_clip_norm2 = super::clip_global_norm(&mut weight_grads2, &mut bias_grads2, 5.0);
        assert_eq!(pre_clip_norm2, 10.0);
        // scaled by 5/10 = 0.5
        assert_eq!(weight_grads2[0][0], 3.0);
        assert_eq!(weight_grads2[0][1], 4.0);
    }
}
