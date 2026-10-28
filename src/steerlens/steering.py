"""
Contrastive Activation Addition (CAA) and Subspace Steering Vectors.

Given verified winning reasoning branches vs. losing/hallucinated branches from
verifier-guided tree search, computes portable activation steering vectors
`v_steer` that can either:
1. Be injected dynamically at inference time: `h' = h + alpha * v_steer`
2. Be folded directly into model bias or rank-1 LoRA parameter deltas so the
   steering improvement can be shipped as a zero-overhead parameter update!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple


def _l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SteeringVector:
    """
    A portable Contrastive Activation Addition (CAA) steering vector for a model layer.
    """

    layer_name: str
    direction: List[float]
    raw_magnitude: float
    pos_sample_count: int
    neg_sample_count: int
    cosine_margin: float
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def d_model(self) -> int:
        return len(self.direction)

    def apply(self, h: Sequence[float], alpha: float = 1.0) -> List[float]:
        """
        Inject the steering vector into a hidden activation `h`:
        `h' = h + alpha * v_steer`
        """
        if len(h) != len(self.direction):
            raise ValueError(
                f"Activation dim {len(h)} does not match steering vector dim {len(self.direction)}"
            )
        return [x + alpha * v for x, v in zip(h, self.direction)]

    def apply_batch(
        self, activations: Sequence[Sequence[float]], alpha: float = 1.0
    ) -> List[List[float]]:
        """Apply steering intervention across a batch of hidden states."""
        return [self.apply(h, alpha=alpha) for h in activations]

    def project_alignment(self, h: Sequence[float]) -> float:
        """
        Compute the scalar projection (dot product) of `h` onto the unit steering direction.
        Positive values indicate alignment with verified reasoning trajectories.
        """
        return _dot(h, self.direction)


@dataclass
class SubspaceProjector:
    """
    Orthonormal subspace steering projector spanned by top contrastive directions.
    """

    layer_name: str
    basis: List[List[float]]  # k orthonormal vectors in R^d

    def erase_bad_subspace(self, h: Sequence[float], strength: float = 1.0) -> List[float]:
        """
        Project out components along a hallucination subspace:
        `h' = h - strength * sum_i (u_i^T h) u_i`
        """
        out = list(h)
        for u in self.basis:
            coeff = _dot(out, u)
            for d in range(len(out)):
                out[d] -= strength * coeff * u[d]
        return out


def compute_caa_vector(
    pos_acts: Sequence[Sequence[float]],
    neg_acts: Sequence[Sequence[float]],
    layer_name: str = "layers.16.mlp",
    normalize: bool = True,
) -> SteeringVector:
    """
    Compute a Contrastive Activation Addition (CAA) steering vector from paired or
    unpaired verified (`pos_acts`) vs. refuted (`neg_acts`) hidden activations:

        v_raw = E[h_pos] - E[h_neg]
        v_steer = v_raw / ||v_raw||_2
    """
    if not pos_acts or not neg_acts:
        raise ValueError("pos_acts and neg_acts must both be non-empty.")

    d_model = len(pos_acts[0])
    if any(len(x) != d_model for x in pos_acts) or any(len(x) != d_model for x in neg_acts):
        raise ValueError("All activation vectors must share the same dimension d_model.")

    n_pos = len(pos_acts)
    n_neg = len(neg_acts)

    pos_mean = [sum(h[d] for h in pos_acts) / n_pos for d in range(d_model)]
    neg_mean = [sum(h[d] for h in neg_acts) / n_neg for d in range(d_model)]

    diff = [pos_mean[d] - neg_mean[d] for d in range(d_model)]
    raw_mag = _l2_norm(diff)
    if raw_mag < 1e-12:
        unit_dir = [0.0] * d_model
    else:
        unit_dir = [x / raw_mag for x in diff] if normalize else list(diff)

    # Compute mean separation margin along the unit contrastive direction
    unit_ref = [x / max(raw_mag, 1e-12) for x in diff]
    pos_proj = sum(_dot(h, unit_ref) for h in pos_acts) / n_pos
    neg_proj = sum(_dot(h, unit_ref) for h in neg_acts) / n_neg
    margin = pos_proj - neg_proj

    return SteeringVector(
        layer_name=layer_name,
        direction=unit_dir,
        raw_magnitude=raw_mag,
        pos_sample_count=n_pos,
        neg_sample_count=n_neg,
        cosine_margin=margin,
    )


def compute_subspace_projector(
    contrastive_vectors: Sequence[Sequence[float]],
    layer_name: str = "layers.16.mlp",
) -> SubspaceProjector:
    """
    Construct an orthonormal basis via modified Gram-Schmidt over multiple
    contrastive directions so multi-domain steering vectors do not interfere.
    """
    basis: List[List[float]] = []
    for vec in contrastive_vectors:
        v = list(vec)
        for u in basis:
            proj = _dot(v, u)
            v = [v[i] - proj * u[i] for i in range(len(v))]
        norm = _l2_norm(v)
        if norm > 1e-9:
            basis.append([x / norm for x in v])
    return SubspaceProjector(layer_name=layer_name, basis=basis)


def fold_steering_into_linear_bias(
    next_layer_weight: Sequence[Sequence[float]],
    next_layer_bias: Sequence[float],
    steering_vector: SteeringVector,
    alpha: float = 1.0,
) -> List[float]:
    """
    Fold an activation steering intervention `h' = h + alpha * v_steer` into the
    bias parameter of a downstream linear layer `y = W h' + b = W h + (b + alpha * W v_steer)`.

    This converts a runtime activation hook into a zero-overhead static parameter delta!
    """
    d_out = len(next_layer_weight)
    if len(next_layer_bias) != d_out:
        raise ValueError("next_layer_bias length must match number of rows in next_layer_weight.")

    v = steering_vector.direction
    new_bias: List[float] = []
    for row_idx in range(d_out):
        row = next_layer_weight[row_idx]
        if len(row) != len(v):
            raise ValueError("Weight matrix column count must match steering vector dimension.")
        shift = alpha * _dot(row, v)
        new_bias.append(next_layer_bias[row_idx] + shift)
    return new_bias


def fold_steering_into_lora_delta(
    steering_vector: SteeringVector,
    trigger_key_direction: Sequence[float],
    alpha: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Convert a steering vector `v_steer` (`R^{d_out}`) and a normalized trigger key
    direction `k_trigger` (`R^{d_in}`) into a rank-1 LoRA weight update `(lora_B, lora_A)`
    where `Delta W = lora_B @ lora_A^T = (alpha * v_steer) @ k_trigger^T`.

    When input activation `x` aligns with `k_trigger`, `Delta W x` injects `alpha * v_steer`.
    """
    k_norm = _l2_norm(trigger_key_direction)
    if k_norm < 1e-12:
        raise ValueError("trigger_key_direction must have non-zero norm.")
    lora_a = [x / k_norm for x in trigger_key_direction]
    lora_b = [alpha * v for v in steering_vector.direction]
    return lora_b, lora_a
