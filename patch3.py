with open("rust_evades_dqn/src/network.rs", "r") as f:
    content = f.read()

# 1. train_batch signature and body
content = content.replace(
"""    pub fn train_batch(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
    ) -> f32 {
        let gradients = self.accumulate_batch(target_network, batch, gamma);
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }""",
"""    pub fn train_batch(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
    ) -> f32 {
        let mut gradients = self.accumulate_batch(target_network, batch, gamma, huber_delta);
        clip_global_norm(&mut gradients.weight_grads, &mut gradients.bias_grads, gradient_clip_norm);
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }""")

# 2. train_batch_parallel signature and body
content = content.replace(
"""    pub fn train_batch_parallel(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        chunk_size: usize,
    ) -> f32 {
        let network = &*self;
        let gradients = batch
            .par_chunks(chunk_size.max(1))
            .map(|chunk| network.accumulate_batch(target_network, chunk, gamma))
            .reduce(
                || BatchGradients::zero_for(&network.layers),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            );
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }""",
"""    pub fn train_batch_parallel(
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
            .map(|chunk| network.accumulate_batch(target_network, chunk, gamma, huber_delta))
            .reduce(
                || BatchGradients::zero_for(&network.layers),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            );
        clip_global_norm(&mut gradients.weight_grads, &mut gradients.bias_grads, gradient_clip_norm);
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }""")

# 3. accumulate_batch signature and body
content = content.replace(
"""    fn accumulate_batch(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
    ) -> BatchGradients {
        let mut gradients = BatchGradients::zero_for(&self.layers);
        for transition in batch {
            gradients.total_loss += self.accumulate_transition(
                target_network,
                transition,
                gamma,
                &mut gradients.weight_grads,
                &mut gradients.bias_grads,
            );
        }
        gradients
    }""",
"""    fn accumulate_batch(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        huber_delta: f32,
    ) -> BatchGradients {
        let mut gradients = BatchGradients::zero_for(&self.layers);
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
        gradients
    }""")

# 4. accumulate_transition signature and huber calculation
content = content.replace(
"""    fn accumulate_transition(
        &self,
        target_network: &Network,
        transition: &super::trainer::Transition,
        gamma: f32,
        weight_grads: &mut [Vec<f32>],
        bias_grads: &mut [Vec<f32>],
    ) -> f32 {""",
"""    fn accumulate_transition(
        &self,
        target_network: &Network,
        transition: &super::trainer::Transition,
        gamma: f32,
        huber_delta: f32,
        weight_grads: &mut [Vec<f32>],
        bias_grads: &mut [Vec<f32>],
    ) -> f32 {""")

content = content.replace(
"""        let prediction = q_values[transition.action];
        let diff = prediction - target;

        let mut deltas = vec![vec![0.0; q_values.len()]];
        deltas[0][transition.action] = 2.0 * diff;""",
"""        let prediction = q_values[transition.action];
        let diff = prediction - target;

        let (loss, grad_wrt_prediction) = huber(diff, huber_delta);

        let mut deltas = vec![vec![0.0; q_values.len()]];
        deltas[0][transition.action] = grad_wrt_prediction;""")

content = content.replace(
"""                deltas.push(previous_delta);
            }
        }

        diff * diff
    }""",
"""                deltas.push(previous_delta);
            }
        }

        loss
    }""")

# 5. Add huber and clip_global_norm functions before #[cfg(test)]
helpers = """
fn huber(diff: f32, delta: f32) -> (f32, f32) {
    let abs_diff = diff.abs();
    if abs_diff <= delta {
        (0.5 * diff * diff, diff)
    } else {
        (
            delta * (abs_diff - 0.5 * delta),
            delta * diff.signum(),
        )
    }
}

fn clip_global_norm(weight_grads: &mut [Vec<f32>], bias_grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
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

#[cfg(test)]"""
parts = content.split('\n#[cfg(test)]\n')
content = parts[0] + "\n" + helpers + '\n' + parts[1]

# 6. Add tests for huber and clip_global_norm at the end of the module
tests = """    #[test]
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
"""

content = content.replace("    }\n}\n", "    }\n\n" + tests)
if "    }\n}" in content and not content.endswith(tests):
    content = content.replace("    }\n}", "    }\n\n" + tests)

with open("rust_evades_dqn/src/network.rs", "w") as f:
    f.write(content)
