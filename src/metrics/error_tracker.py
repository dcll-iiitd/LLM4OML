# ============================================================================
# File: src/metrics/error_tracker.py
# ============================================================================
"""Step-level error set tracking and management."""

from typing import Set, Tuple
from ..models.types import ErrorSet


class ErrorTracker:
    """Tracks and compares error sets across iterations."""

    def __init__(self):
        self.history: list[ErrorSet] = []

    def add_iteration(self, error_set: ErrorSet) -> None:
        self.history.append(error_set)

    @staticmethod
    def get_all_errors(error_set: ErrorSet) -> Set[int]:
        """Union of all error step indices."""
        return (
            set(error_set.get("hallucinations", set()))
            | set(error_set.get("missing_steps", set()))
            | set(error_set.get("operator_errors", set()))
            | set(error_set.get("assumption_violations", set()))
        )

    @staticmethod
    def compute_error_delta(
        prev: ErrorSet, curr: ErrorSet
    ) -> Tuple[Set[int], Set[int]]:
        """
        Returns (fixed_errors, new_errors).
        fixed = errors that existed before but are gone now.
        new   = errors that did not exist before but appear now.
        """
        tracker = ErrorTracker()
        prev_all = tracker.get_all_errors(prev)
        curr_all = tracker.get_all_errors(curr)
        return prev_all - curr_all, curr_all - prev_all

    @staticmethod
    def error_set_from_metrics(metrics) -> ErrorSet:
        """Build an ErrorSet from an EvaluationMetrics object."""
        return {
            "hallucinations": set(metrics.hallucination_steps),
            "missing_steps": set(metrics.missing_step_indices),
            "operator_errors": set(metrics.operator_error_steps),
            "assumption_violations": set(metrics.assumption_violation_steps),
        }

    @staticmethod
    def merge_error_sets(sets: list[ErrorSet], mode: str = "union") -> ErrorSet:
        """
        Merge multiple error sets.
        mode='union'    → include a step if ANY judge flagged it
        mode='majority' → include a step if MAJORITY of judges flagged it
        """
        if not sets:
            return {
                "hallucinations": set(),
                "missing_steps": set(),
                "operator_errors": set(),
                "assumption_violations": set(),
            }

        n = len(sets)
        keys = ["hallucinations", "missing_steps", "operator_errors", "assumption_violations"]
        result: ErrorSet = {}

        for key in keys:
            all_steps: Set[int] = set()
            for es in sets:
                all_steps |= set(es.get(key, set()))

            if mode == "majority":
                result[key] = {
                    step
                    for step in all_steps
                    if sum(step in set(es.get(key, set())) for es in sets) > n / 2
                }
            else:  # union
                result[key] = all_steps

        return result