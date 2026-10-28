"""
Glass-box Reasoning Telemetry Analyzer (`ReasoningTelemetry`).

Monitors sparse feature entropy, planning-vs-guessing score, and layer-wise
steering alignment before and after causal interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

from .sae import SparseAutoencoder
from .steering import SteeringVector


@dataclass(frozen=True)
class TelemetryReport:
    """Glass-box mechanistic telemetry summary for a reasoning trajectory."""

    mean_feature_entropy: float
    planning_vs_guessing_score: float
    steering_alignment_before: float
    steering_alignment_after: float
    verified_logit_shift: float
    active_monosemantic_features: List[int]


class ReasoningTelemetry:
    """
    Analyzes hidden states during tree-search or chain-of-thought deduction using
    a trained `SparseAutoencoder` and `SteeringVector`.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        steering_vector: SteeringVector,
        verified_feature_ids: Sequence[int],
        hallucination_feature_ids: Sequence[int],
    ) -> None:
        self.sae = sae
        self.steering_vector = steering_vector
        self.verified_feature_ids = list(verified_feature_ids)
        self.hallucination_feature_ids = list(hallucination_feature_ids)

    @staticmethod
    def _normalized_shannon_entropy(sparse_vec: Sequence[float]) -> float:
        positive = [v for v in sparse_vec if v > 0.0]
        if len(positive) <= 1:
            return 0.0
        total = sum(positive)
        probs = [v / total for v in positive]
        raw_h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        return raw_h / math.log(len(positive))

    def evaluate_trajectory(
        self,
        hidden_states: Sequence[Sequence[float]],
        unembed_verified_dir: Sequence[float],
        alpha: float = 1.5,
    ) -> TelemetryReport:
        """
        Compute mechanistic interpretability telemetry on a sequence of hidden states
        and measure the causal impact of applying `self.steering_vector` with multiplier `alpha`.
        """
        if not hidden_states:
            raise ValueError("hidden_states cannot be empty.")

        entropies: List[float] = []
        plan_scores: List[float] = []
        align_before: List[float] = []
        align_after: List[float] = []
        logit_shifts: List[float] = []
        active_feature_counts: Dict[int, float] = {}

        for h in hidden_states:
            f = self.sae.encode(h)
            entropies.append(self._normalized_shannon_entropy(f))

            for idx, val in enumerate(f):
                if val > 0.0:
                    active_feature_counts[idx] = active_feature_counts.get(idx, 0.0) + val

            v_mass = sum(f[i] for i in self.verified_feature_ids if 0 <= i < len(f))
            h_mass = sum(f[i] for i in self.hallucination_feature_ids if 0 <= i < len(f))
            plan_score = (v_mass - h_mass) / (v_mass + h_mass + 1e-6)
            plan_scores.append(plan_score)

            steered_h = self.steering_vector.apply(h, alpha=alpha)
            align_before.append(self.steering_vector.project_alignment(h))
            align_after.append(self.steering_vector.project_alignment(steered_h))

            logit_before = sum(x * w for x, w in zip(h, unembed_verified_dir))
            logit_after = sum(x * w for x, w in zip(steered_h, unembed_verified_dir))
            logit_shifts.append(logit_after - logit_before)

        top_features = sorted(
            active_feature_counts.keys(),
            key=lambda k: active_feature_counts[k],
            reverse=True,
        )[:5]

        n = len(hidden_states)
        return TelemetryReport(
            mean_feature_entropy=sum(entropies) / n,
            planning_vs_guessing_score=sum(plan_scores) / n,
            steering_alignment_before=sum(align_before) / n,
            steering_alignment_after=sum(align_after) / n,
            verified_logit_shift=sum(logit_shifts) / n,
            active_monosemantic_features=top_features,
        )
