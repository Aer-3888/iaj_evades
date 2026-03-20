with open("rust_evades_dqn/src/network.rs", "r") as f:
    text = f.read()

text = text.replace(
"""    pub fn train_batch(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
    ) -> f32 {""",
"""    pub fn train_batch(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
    ) -> f32 {""")

text = text.replace(
"""        let gradients = self.accumulate_batch(target_network, batch, gamma);
        self.apply_gradients(gradients, batch.len(), learning_rate)""",
"""        let mut gradients = self.accumulate_batch(target_network, batch, gamma, huber_delta);
        clip_global_norm(&mut gradients.weight_grads, &mut gradients.bias_grads, gradient_clip_norm);
        self.apply_gradients(gradients, batch.len(), learning_rate)""")

text = text.replace(
"""    pub fn train_batch_parallel(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        chunk_size: usize,
    ) -> f32 {""",
"""    pub fn train_batch_parallel(
        &mut self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
        chunk_size: usize,
    ) -> f32 {""")

text = text.replace(
"""            .map(|chunk| network.accumulate_batch(target_network, chunk, gamma))""",
"""            .map(|chunk| network.accumulate_batch(target_network, chunk, gamma, huber_delta))""")

text = text.replace(
"""        self.apply_gradients(gradients, batch.len(), learning_rate)
    }

    fn apply_gradients(""",
"""        clip_global_norm(&mut gradients.weight_grads, &mut gradients.bias_grads, gradient_clip_norm);
        self.apply_gradients(gradients, batch.len(), learning_rate)
    }

    fn apply_gradients(""")

text = text.replace(
"""    fn accumulate_batch(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
    ) -> BatchGradients {""",
"""    fn accumulate_batch(
        &self,
        target_network: &Network,
        batch: &[super::trainer::Transition],
        gamma: f32,
        huber_delta: f32,
    ) -> BatchGradients {""")

text = text.replace(
"""            gradients.total_loss += self.accumulate_transition(
                target_network,
                transition,
                gamma,
                &mut gradients.weight_grads,
                &mut gradients.bias_grads,
            );""",
"""            gradients.total_loss += self.accumulate_transition(
                target_network,
                transition,
                gamma,
                huber_delta,
                &mut gradients.weight_grads,
                &mut gradients.bias_grads,
            );""")

text = text.replace(
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

text = text.replace(
"""        let prediction = q_values[transition.action];
        let diff = prediction - target;

        let mut deltas = vec![vec![0.0; q_values.len()]];
        deltas[0][transition.action] = 2.0 * diff;""",
"""        let prediction = q_values[transition.action];
        let diff = prediction - target;

        let (loss, grad_wrt_prediction) = huber(diff, huber_delta);

        let mut deltas = vec![vec![0.0; q_values.len()]];
        deltas[0][transition.action] = grad_wrt_prediction;""")

text = text.replace(
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

functions = """
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
parts = text.split('\n#[cfg(test)]\n')
text = parts[0] + "\n" + functions + '\n' + parts[1]

tests = """
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
}"""
text = text.rsplit("}\n", 1)[0] + tests + "\n"

with open("rust_evades_dqn/src/network.rs", "w") as f:
    f.write(text)
