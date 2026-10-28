"""
Sparse Autoencoder (Top-K & L1 ReLU SAE) for Mechanistic Interpretability.

Decomposes dense polysemantic hidden layer activations h in R^d into high-dimensional
sparse monosemantic features f in R^m (m >> d), and identifies diagnostic features
that distinguish verified reasoning steps from unverified/hallucinated branches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: Sequence[float], eps: float = 1e-12) -> List[float]:
    n = _l2_norm(v)
    if n < eps:
        return [0.0 for _ in v]
    return [x / n for x in v]


@dataclass(frozen=True)
class FeatureDiagnostic:
    """Diagnostic statistics for a single monosemantic SAE feature."""

    feature_id: int
    pos_mean_activation: float
    neg_mean_activation: float
    differential_score: float
    pos_fire_rate: float
    neg_fire_rate: float
    selectivity_index: float


class SparseAutoencoder:
    """
    Top-K / L1 ReLU Sparse Autoencoder (SAE).

    Decomposes hidden states `h in R^d` into sparse feature activations `f in R^m`
    where `m = expansion_factor * d` (or explicit `d_sae`), enforcing exact `k`
    active features per token (`top_k`) and unit-norm decoder columns.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        top_k: int = 8,
        l1_coeff: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if d_model <= 0 or d_sae <= 0:
            raise ValueError("d_model and d_sae must be positive integers.")
        if top_k <= 0 or top_k > d_sae:
            raise ValueError("top_k must satisfy 1 <= top_k <= d_sae.")

        self.d_model = d_model
        self.d_sae = d_sae
        self.top_k = top_k
        self.l1_coeff = l1_coeff

        rng = random.Random(seed)
        # Initialize decoder dictionary atoms (d_sae atoms, each of dimension d_model)
        # with unit L2 norm, and tie encoder initialization to transpose of decoder.
        self.w_dec: List[List[float]] = []
        for _ in range(d_sae):
            raw = [rng.gauss(0.0, 1.0) for _ in range(d_model)]
            self.w_dec.append(_normalize(raw))

        self.w_enc: List[List[float]] = [list(atom) for atom in self.w_dec]
        self.b_enc: List[float] = [0.0 for _ in range(d_sae)]
        self.b_dec: List[float] = [0.0 for _ in range(d_model)]

    def encode(self, h: Sequence[float]) -> List[float]:
        """
        Encode a single hidden state `h` (`R^d`) into a sparse feature vector `f` (`R^m`)
        with at most `self.top_k` non-zero positive activations.
        """
        if len(h) != self.d_model:
            raise ValueError(f"Expected activation of length {self.d_model}, got {len(h)}")

        centered = [h[i] - self.b_dec[i] for i in range(self.d_model)]
        pre_acts: List[Tuple[float, int]] = []
        for feat_idx in range(self.d_sae):
            z = _dot(self.w_enc[feat_idx], centered) + self.b_enc[feat_idx]
            # Soft-threshold with L1 penalty and ReLU
            act = max(0.0, z - self.l1_coeff)
            if act > 0.0:
                pre_acts.append((act, feat_idx))

        # Keep exact Top-K highest positive activations
        pre_acts.sort(key=lambda item: item[0], reverse=True)
        active = pre_acts[: self.top_k]

        f = [0.0] * self.d_sae
        for val, idx in active:
            f[idx] = val
        return f

    def encode_batch(self, activations: Sequence[Sequence[float]]) -> List[List[float]]:
        """Encode a batch of hidden activations."""
        return [self.encode(h) for h in activations]

    def decode(self, f: Sequence[float]) -> List[float]:
        """
        Decode sparse feature vector `f` (`R^m`) back into reconstructed activation `h_hat` (`R^d`).
        """
        if len(f) != self.d_sae:
            raise ValueError(f"Expected feature vector of length {self.d_sae}, got {len(f)}")

        h_hat = list(self.b_dec)
        for feat_idx, val in enumerate(f):
            if val != 0.0:
                atom = self.w_dec[feat_idx]
                for d in range(self.d_model):
                    h_hat[d] += val * atom[d]
        return h_hat

    def reconstruction_loss(self, activations: Sequence[Sequence[float]]) -> float:
        """Compute mean squared reconstruction error (MSE) over a batch of activations."""
        if not activations:
            return 0.0
        total_sq = 0.0
        for h in activations:
            f = self.encode(h)
            h_hat = self.decode(f)
            total_sq += sum((h[d] - h_hat[d]) ** 2 for d in range(self.d_model)) / self.d_model
        return total_sq / len(activations)

    def fit(
        self,
        activations: Sequence[Sequence[float]],
        epochs: int = 25,
        lr: float = 0.08,
    ) -> List[float]:
        """
        Train the Sparse Autoencoder on a dataset of hidden activations using
        projected dictionary gradient descent with unit-norm decoder constraints.

        Returns the per-epoch MSE reconstruction loss history.
        """
        if not activations:
            raise ValueError("activations dataset cannot be empty.")

        n_samples = len(activations)
        # Center b_dec on the empirical dataset mean
        for d in range(self.d_model):
            self.b_dec[d] = sum(h[d] for h in activations) / n_samples

        loss_history: List[float] = []
        for _ in range(epochs):
            epoch_sq_err = 0.0
            for h in activations:
                f = self.encode(h)
                h_hat = self.decode(f)
                residual = [h[d] - h_hat[d] for d in range(self.d_model)]
                epoch_sq_err += sum(r * r for r in residual) / self.d_model

                centered = [h[d] - self.b_dec[d] for d in range(self.d_model)]
                for feat_idx, act in enumerate(f):
                    if act > 0.0:
                        # Update decoder atom and encoder weights along residual
                        for d in range(self.d_model):
                            grad_dec = -residual[d] * act
                            self.w_dec[feat_idx][d] -= lr * grad_dec
                            grad_enc = -residual[d] * centered[d]
                            self.w_enc[feat_idx][d] -= lr * grad_enc
                        self.b_enc[feat_idx] += lr * 0.1 * _dot(residual, self.w_dec[feat_idx])
                        # Renormalize decoder dictionary column to unit L2 norm
                        self.w_dec[feat_idx] = _normalize(self.w_dec[feat_idx])

            loss_history.append(epoch_sq_err / n_samples)
        return loss_history

    def top_diagnostic_features(
        self,
        pos_acts: Sequence[Sequence[float]],
        neg_acts: Sequence[Sequence[float]],
        top_n: int = 5,
    ) -> List[FeatureDiagnostic]:
        """
        Identify which monosemantic sparse features differentially fire during
        verified reasoning (`pos_acts`) vs. hallucinated/dead-end branches (`neg_acts`).

        Returns the top `top_n` features sorted by differential activation score.
        """
        if not pos_acts or not neg_acts:
            raise ValueError("Both pos_acts and neg_acts must be non-empty.")

        pos_encoded = self.encode_batch(pos_acts)
        neg_encoded = self.encode_batch(neg_acts)

        diagnostics: List[FeatureDiagnostic] = []
        n_pos = len(pos_encoded)
        n_neg = len(neg_encoded)

        for feat_id in range(self.d_sae):
            pos_vals = [row[feat_id] for row in pos_encoded]
            neg_vals = [row[feat_id] for row in neg_encoded]

            pos_mean = sum(pos_vals) / n_pos
            neg_mean = sum(neg_vals) / n_neg
            diff = pos_mean - neg_mean

            pos_fires = sum(1 for v in pos_vals if v > 0.0) / n_pos
            neg_fires = sum(1 for v in neg_vals if v > 0.0) / n_neg

            denom = pos_mean + neg_mean + 1e-9
            selectivity = (pos_mean - neg_mean) / denom

            diagnostics.append(
                FeatureDiagnostic(
                    feature_id=feat_id,
                    pos_mean_activation=pos_mean,
                    neg_mean_activation=neg_mean,
                    differential_score=diff,
                    pos_fire_rate=pos_fires,
                    neg_fire_rate=neg_fires,
                    selectivity_index=selectivity,
                )
            )

        diagnostics.sort(key=lambda d: d.differential_score, reverse=True)
        return diagnostics[:top_n]

    def feature_steering_vector(self, feature_id: int, scale: float = 1.0) -> List[float]:
        """
        Extract the causal decoder direction associated with a specific monosemantic
        SAE feature so it can be used directly for targeted activation clamping/steering.
        """
        if not (0 <= feature_id < self.d_sae):
            raise IndexError(f"feature_id {feature_id} out of range [0, {self.d_sae})")
        return [scale * x for x in self.w_dec[feature_id]]
