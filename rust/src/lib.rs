//! High-performance Top-K Sparse Autoencoder (SAE) and Contrastive Activation Steering kernels.
//!
//! Provides zero-dependency Rust kernels for:
//! - Top-K sparse feature encoding (`encode_topk`)
//! - Sparse feature decoding (`decode_sparse`)
//! - Contrastive Activation Addition (CAA) vector extraction & cosine intervention (`apply_steering_vector`)

/// Represents a sparse feature activation `(feature_idx, activation_value)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseFeature {
    pub index: usize,
    pub value: f64,
}

/// Top-K Sparse Autoencoder kernel in Rust.
#[derive(Debug, Clone)]
pub struct TopKSparseAutoencoder {
    pub d_model: usize,
    pub d_sae: usize,
    pub k: usize,
    /// Encoder weights flattened row-major: `[d_sae, d_model]`
    pub w_enc: Vec<f64>,
    /// Encoder bias: `[d_sae]`
    pub b_enc: Vec<f64>,
    /// Decoder weights flattened row-major: `[d_model, d_sae]`
    pub w_dec: Vec<f64>,
    /// Pre-encoder / decoder centering bias: `[d_model]`
    pub b_dec: Vec<f64>,
}

impl TopKSparseAutoencoder {
    /// Encode a dense hidden state `h` (`[d_model]`) into exact Top-K positive sparse features.
    pub fn encode_topk(&self, h: &[f64]) -> Vec<SparseFeature> {
        assert_eq!(h.len(), self.d_model, "Input dimension mismatch");
        let centered: Vec<f64> = h.iter().zip(self.b_dec.iter()).map(|(x, b)| x - b).collect();

        let mut pre_acts: Vec<SparseFeature> = (0..self.d_sae)
            .map(|i| {
                let row_offset = i * self.d_model;
                let dot: f64 = centered
                    .iter()
                    .zip(&self.w_enc[row_offset..row_offset + self.d_model])
                    .map(|(a, b)| a * b)
                    .sum();
                let val = (dot + self.b_enc[i]).max(0.0);
                SparseFeature { index: i, value: val }
            })
            .filter(|f| f.value > 0.0)
            .collect();

        pre_acts.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
        pre_acts.truncate(self.k);
        pre_acts
    }

    /// Reconstruct dense activation `h_hat` (`[d_model]`) from active Top-K sparse features.
    pub fn decode_sparse(&self, active_features: &[SparseFeature]) -> Vec<f64> {
        let mut h_hat = self.b_dec.clone();
        for feat in active_features {
            for d in 0..self.d_model {
                h_hat[d] += self.w_dec[d * self.d_sae + feat.index] * feat.value;
            }
        }
        h_hat
    }
}

/// Apply a Contrastive Activation Addition (CAA) steering vector `v_steer` with multiplier `alpha`
/// to hidden state `h`: `h' = h + alpha * v_steer`.
pub fn apply_steering_vector(h: &[f64], v_steer: &[f64], alpha: f64) -> Vec<f64> {
    assert_eq!(h.len(), v_steer.len(), "Dimension mismatch in steering vector");
    h.iter()
        .zip(v_steer.iter())
        .map(|(x, v)| x + alpha * v)
        .collect()
}

/// Compute normalized Contrastive Activation Addition (CAA) vector from mean difference
/// between verified positive activations and negative/hallucinated activations.
pub fn compute_caa_vector(pos_mean: &[f64], neg_mean: &[f64]) -> Vec<f64> {
    assert_eq!(pos_mean.len(), neg_mean.len(), "Dimension mismatch");
    let diff: Vec<f64> = pos_mean
        .iter()
        .zip(neg_mean.iter())
        .map(|(p, n)| p - n)
        .collect();
    let norm = diff.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    diff.into_iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caa_and_steering() {
        let pos = vec![2.0, 1.0, 0.0];
        let neg = vec![0.0, 1.0, 0.0];
        let v = compute_caa_vector(&pos, &neg);
        assert!((v[0] - 1.0).abs() < 1e-9);
        let steered = apply_steering_vector(&neg, &v, 2.5);
        assert!((steered[0] - 2.5).abs() < 1e-9);
    }
}
