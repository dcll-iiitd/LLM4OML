# ============================================================================
# File: src/metrics/judge_metrics.py
# ============================================================================
"""Judge reliability metrics: ESA, SC, JRS — unchanged logic from v2, minor cleanup."""

from typing import List, Set
import numpy as np
from ..models.types import ErrorSet, JudgeMetrics
from .error_tracker import ErrorTracker


class JudgeReliabilityCalculator:
    """Computes multi-judge agreement and reliability metrics."""

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Parameters
        ----------
        alpha : weight for ESA (agreement component of JRS)
        beta  : weight for z-score penalty (outlier penalty)
        """
        self.alpha = alpha
        self.beta = beta

    def compute_esa(self, error_set_1: ErrorSet, error_set_2: ErrorSet) -> float:
        """
        Error Set Agreement (Jaccard similarity).
        ESA = |E_i ∩ E_j| / |E_i ∪ E_j|
        Returns 1.0 if both judges found zero errors (unanimous clean).
        """
        e1 = ErrorTracker.get_all_errors(error_set_1)
        e2 = ErrorTracker.get_all_errors(error_set_2)
        union = e1 | e2
        if not union:
            return 1.0
        return len(e1 & e2) / len(union)

    def compute_mean_esa_per_judge(self, error_sets: List[ErrorSet]) -> List[float]:
        """Mean pairwise ESA for each judge against all others."""
        n = len(error_sets)
        mean_esas = []
        for i in range(n):
            esas = [
                self.compute_esa(error_sets[i], error_sets[j])
                for j in range(n)
                if j != i
            ]
            mean_esas.append(float(np.mean(esas)) if esas else 0.0)
        return mean_esas

    def compute_z_scores(self, scores: List[float]) -> List[float]:
        """Standardised z-scores for outlier detection."""
        if len(scores) <= 1:
            return [0.0] * len(scores)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std < 1e-9:
            return [0.0] * len(scores)
        return [(s - mean) / std for s in scores]

    def compute_sc(self, scores: List[float]) -> float:
        """Score Consistency = variance of scores across judges (lower = better)."""
        if len(scores) <= 1:
            return 0.0
        return float(np.var(scores))

    def compute_jrs(
        self,
        error_sets: List[ErrorSet],
        scores: List[float],
    ) -> List[JudgeMetrics]:
        """
        Judge Reliability Score per judge.
        JRS_i = alpha * ESA_i - beta * |z_i|
        """
        mean_esas = self.compute_mean_esa_per_judge(error_sets)
        z_scores = self.compute_z_scores(scores)
        sc = self.compute_sc(scores)

        return [
            {
                "esa": mean_esas[i],
                "sc": sc,
                "z_score": z_scores[i],
                "jrs": self.alpha * mean_esas[i] - self.beta * abs(z_scores[i]),
            }
            for i in range(len(error_sets))
        ]

    @staticmethod
    def detect_outliers(z_scores: List[float], threshold: float = 2.0) -> List[int]:
        """Return indices of judges whose |z_score| exceeds threshold."""
        return [i for i, z in enumerate(z_scores) if abs(z) > threshold]