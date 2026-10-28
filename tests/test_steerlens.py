import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from steerlens import (
    ReasoningTelemetry,
    SparseAutoencoder,
    compute_caa_vector,
    compute_subspace_projector,
    fold_steering_into_linear_bias,
    fold_steering_into_lora_delta,
)


class TestSteerLens(unittest.TestCase):
    def setUp(self) -> None:
        # Construct synthetic 8-dim activations where dimensions 0..1 encode verified
        # deductive planning and dimensions 6..7 encode hallucinated guessing.
        self.pos_acts = [
            [2.4, 1.9, 0.2, -0.1, 0.0, 0.1, -1.2, -1.0],
            [2.1, 2.2, 0.1, 0.0, -0.1, 0.2, -1.0, -1.3],
            [2.6, 1.8, -0.1, 0.2, 0.1, 0.0, -1.1, -0.9],
            [2.3, 2.0, 0.0, 0.1, 0.2, -0.1, -1.3, -1.1],
        ]
        self.neg_acts = [
            [-1.5, -1.2, 0.1, 0.0, -0.2, 0.1, 2.2, 2.0],
            [-1.3, -1.4, -0.1, 0.1, 0.0, -0.1, 2.5, 1.9],
            [-1.6, -1.1, 0.2, -0.1, 0.1, 0.0, 2.1, 2.3],
            [-1.4, -1.3, 0.0, 0.2, -0.1, 0.2, 2.3, 2.1],
        ]

    def test_sae_training_and_sparsity(self) -> None:
        sae = SparseAutoencoder(d_model=8, d_sae=24, top_k=3, seed=7)
        all_acts = self.pos_acts + self.neg_acts
        initial_loss = sae.reconstruction_loss(all_acts)
        history = sae.fit(all_acts, epochs=30, lr=0.06)
        final_loss = sae.reconstruction_loss(all_acts)

        self.assertLess(final_loss, initial_loss)
        self.assertLess(history[-1], history[0])

        # Verify exact Top-K sparsity constraint
        for h in all_acts:
            f = sae.encode(h)
            non_zero = [v for v in f if v > 0.0]
            self.assertLessEqual(len(non_zero), 3)
            self.assertGreaterEqual(len(non_zero), 1)

        # Verify diagnostic features distinguish verified vs. hallucinated activations
        diag = sae.top_diagnostic_features(self.pos_acts, self.neg_acts, top_n=3)
        self.assertEqual(len(diag), 3)
        self.assertGreater(diag[0].differential_score, 0.5)
        self.assertGreater(diag[0].selectivity_index, 0.5)

    def test_caa_steering_and_logit_intervention(self) -> None:
        steer_vec = compute_caa_vector(
            self.pos_acts, self.neg_acts, layer_name="model.layers.12.mlp"
        )
        self.assertEqual(steer_vec.d_model, 8)
        self.assertGreater(steer_vec.raw_magnitude, 4.0)
        self.assertGreater(steer_vec.cosine_margin, 4.0)

        # Unembedding direction favoring verified tokens (dims 0 and 1) over hallucinated (6, 7)
        unembed_verified = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]

        for neg_h in self.neg_acts:
            logit_before = sum(x * w for x, w in zip(neg_h, unembed_verified))
            steered_h = steer_vec.apply(neg_h, alpha=4.0)
            logit_after = sum(x * w for x, w in zip(steered_h, unembed_verified))
            self.assertGreater(logit_after, logit_before + 3.0)
            self.assertGreater(logit_after, 0.0)

    def test_fold_steering_into_parameters(self) -> None:
        steer_vec = compute_caa_vector(self.pos_acts, self.neg_acts)
        # Downstream linear projection W (2 x 8) and bias b (2,)
        W = [
            [0.5, 0.5, 0.1, 0.0, 0.0, 0.0, -0.4, -0.4],
            [-0.3, -0.2, 0.0, 0.1, 0.0, 0.0, 0.6, 0.5],
        ]
        b = [0.1, -0.1]
        alpha = 2.5

        folded_b = fold_steering_into_linear_bias(W, b, steer_vec, alpha=alpha)
        sample_h = self.neg_acts[0]
        steered_h = steer_vec.apply(sample_h, alpha=alpha)

        # Verify W @ steered_h + b == W @ sample_h + folded_b to machine precision
        for row_idx in range(2):
            runtime_out = sum(W[row_idx][d] * steered_h[d] for d in range(8)) + b[row_idx]
            folded_out = sum(W[row_idx][d] * sample_h[d] for d in range(8)) + folded_b[row_idx]
            self.assertAlmostEqual(runtime_out, folded_out, places=10)

        # Also test folding into rank-1 LoRA delta
        lora_b, lora_a = fold_steering_into_lora_delta(
            steer_vec, trigger_key_direction=[1.0, 0.0, 0.0, 0.0], alpha=2.0
        )
        self.assertEqual(len(lora_b), 8)
        self.assertEqual(len(lora_a), 4)

    def test_subspace_projector_and_telemetry(self) -> None:
        sae = SparseAutoencoder(d_model=8, d_sae=16, top_k=3, seed=11)
        sae.fit(self.pos_acts + self.neg_acts, epochs=20, lr=0.06)
        steer_vec = compute_caa_vector(self.pos_acts, self.neg_acts)

        pos_diag = sae.top_diagnostic_features(self.pos_acts, self.neg_acts, top_n=2)
        neg_diag = sae.top_diagnostic_features(self.neg_acts, self.pos_acts, top_n=2)

        telemetry = ReasoningTelemetry(
            sae=sae,
            steering_vector=steer_vec,
            verified_feature_ids=[d.feature_id for d in pos_diag],
            hallucination_feature_ids=[d.feature_id for d in neg_diag],
        )
        unembed_verified = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]
        report = telemetry.evaluate_trajectory(self.pos_acts, unembed_verified, alpha=1.5)

        self.assertGreater(report.planning_vs_guessing_score, 0.0)
        self.assertGreater(report.steering_alignment_after, report.steering_alignment_before)
        self.assertGreater(report.verified_logit_shift, 0.0)

        proj = compute_subspace_projector([steer_vec.direction, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        cleaned = proj.erase_bad_subspace(self.pos_acts[0], strength=1.0)
        # Projection along basis vectors should now be ~0
        for u in proj.basis:
            dot_after = sum(c * u_i for c, u_i in zip(cleaned, u))
            self.assertAlmostEqual(dot_after, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
