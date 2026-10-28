"""
steerlens: Mechanistic Interpretability, Sparse Autoencoder (SAE) Probes &
Contrastive Activation Steering Parameters.
"""

from .sae import SparseAutoencoder, FeatureDiagnostic
from .steering import (
    SteeringVector,
    SubspaceProjector,
    compute_caa_vector,
    compute_subspace_projector,
    fold_steering_into_linear_bias,
    fold_steering_into_lora_delta,
)
from .inspector import ReasoningTelemetry, TelemetryReport

__all__ = [
    "SparseAutoencoder",
    "FeatureDiagnostic",
    "SteeringVector",
    "SubspaceProjector",
    "compute_caa_vector",
    "compute_subspace_projector",
    "fold_steering_into_linear_bias",
    "fold_steering_into_lora_delta",
    "ReasoningTelemetry",
    "TelemetryReport",
]
