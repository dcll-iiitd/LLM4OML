#!/usr/bin/env python3
# ============================================================================
# src/metrics/oscillation_analyzer.py
# ============================================================================
"""
Oscillation Analysis for H3: Regression Penalty Impact on Correction Stability
===============================================================================

HYPOTHESIS H3:
  Imposing a strict regression penalty factor dynamically compresses 
  oscillatory correction behavior and accelerates path convergence.

WHAT THIS DOES:
  Tracks error state transitions per proof step across iterations and computes:
  
  1. Step-level oscillation: How many times does step S toggle between error/no-error?
  2. Trajectory convergence: How many iterations until stable?
  3. Convergence stability: Does the proof "settle" (no more changes)?
  4. Global oscillation score: Aggregated instability across all steps
  5. Penalty effect: Correlation of w_rp_pen with oscillation metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StepErrorHistory:
    """Track error state of a single step across iterations."""
    step_id: int
    error_states: List[bool]  # [iter1, iter2, iter3, ...] where True = has error
    toggles: int = 0
    last_change_iteration: int = -1
    is_oscillating: bool = False  # True if >1 toggle


@dataclass
class TrajectoryOscillationMetrics:
    """Oscillation metrics for a single proof trajectory."""
    task_id: str
    num_iterations: int
    total_steps: int
    
    # Convergence metrics
    iteration_to_stability: int  # -1 if never converges in window
    converged: bool
    convergence_iteration: int  # first iteration with no further changes
    
    # Oscillation metrics
    steps_with_oscillation: int  # count of steps that toggle >1 time
    total_toggles: int  # total # of step state changes across all steps
    mean_toggles_per_step: float
    max_toggles_single_step: int
    
    # Global stability score
    oscillation_index: float  # [0, 1] where 0 = no oscillation, 1 = maximum
    stability_score: float    # [0, 1] where 1 = perfect convergence
    
    # Regression effect
    w_rp_pen: float = 0.0  # weight from configuration
    
    def __repr__(self) -> str:
        return (
            f"OscillationMetrics(task={self.task_id}, "
            f"iters={self.num_iterations}, "
            f"converged={self.converged} @ iter{self.convergence_iteration}, "
            f"oscillation_index={self.oscillation_index:.3f}, "
            f"stability={self.stability_score:.3f})"
        )


# ============================================================================
# Oscillation Analyzer
# ============================================================================

class OscillationAnalyzer:
    """Detects and quantifies oscillatory behavior in correction trajectories."""

    def __init__(self):
        self.step_histories: Dict[int, StepErrorHistory] = {}

    def analyze_trajectory(
        self,
        error_sequence: List[Set[int]],
        w_rp_pen: float = 0.0,
        task_id: str = "unknown",
    ) -> TrajectoryOscillationMetrics:
        """
        Analyze oscillation in a correction trajectory.
        
        Parameters
        ----------
        error_sequence : list of error sets
            error_sequence[i] = set of error step IDs at iteration i
            Must be ordered from iteration 0 to final iteration
        w_rp_pen : float
            Regression penalty weight used in this trajectory's correction
        task_id : str
            Identifier for this trajectory
        
        Returns
        -------
        TrajectoryOscillationMetrics
        """
        self.step_histories = {}
        
        # Step 1: Build error state history per step
        all_steps = set()
        for error_set in error_sequence:
            all_steps.update(error_set)
        
        # Initialize history for all steps seen
        for step_id in all_steps:
            self.step_histories[step_id] = StepErrorHistory(step_id=step_id, error_states=[])
        
        # Track state at each iteration
        for iter_idx, error_set in enumerate(error_sequence):
            for step_id in all_steps:
                has_error = step_id in error_set
                self.step_histories[step_id].error_states.append(has_error)
        
        # Step 2: Compute toggles per step
        total_toggles = 0
        steps_with_oscillation = 0
        max_toggles = 0
        
        for step_id, history in self.step_histories.items():
            toggles = self._count_toggles(history.error_states)
            history.toggles = toggles
            total_toggles += toggles
            
            if toggles > 1:
                steps_with_oscillation += 1
                history.is_oscillating = True
            
            max_toggles = max(max_toggles, toggles)
            
            # Track last change
            for iter_idx in range(len(history.error_states) - 1, -1, -1):
                if iter_idx > 0:
                    if history.error_states[iter_idx] != history.error_states[iter_idx - 1]:
                        history.last_change_iteration = iter_idx
                        break
        
        # Step 3: Detect convergence
        convergence_iter, converged = self._detect_convergence(error_sequence)
        
        # Step 4: Compute aggregate metrics
        num_iters = len(error_sequence)
        total_steps = len(all_steps)
        mean_toggles = total_toggles / total_steps if total_steps > 0 else 0.0
        
        # Oscillation index: [0, 1] where higher = more oscillation
        # Formula: (mean_toggles_per_step / max_possible) normalized
        # Max possible = number of state changes, which is roughly num_iters - 1
        max_possible_toggles = num_iters - 1 if num_iters > 1 else 1
        oscillation_index = self._compute_oscillation_index(
            total_toggles,
            total_steps,
            max_possible_toggles,
        )
        
        # Stability score: [0, 1] where higher = more stable
        # Perfect = converges at iter 1 (oscillation_index=0)
        stability_score = 1.0 - oscillation_index
        
        return TrajectoryOscillationMetrics(
            task_id=task_id,
            num_iterations=num_iters,
            total_steps=total_steps,
            iteration_to_stability=convergence_iter if not converged else convergence_iter,
            converged=converged,
            convergence_iteration=convergence_iter,
            steps_with_oscillation=steps_with_oscillation,
            total_toggles=total_toggles,
            mean_toggles_per_step=mean_toggles,
            max_toggles_single_step=max_toggles,
            oscillation_index=oscillation_index,
            stability_score=stability_score,
            w_rp_pen=w_rp_pen,
        )

    @staticmethod
    def _count_toggles(states: List[bool]) -> int:
        """Count transitions in a boolean state sequence."""
        if len(states) < 2:
            return 0
        toggles = 0
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                toggles += 1
        return toggles

    @staticmethod
    def _detect_convergence(error_sequence: List[Set[int]]) -> Tuple[int, bool]:
        """
        Detect when trajectory reaches stable state (no more changes).
        
        Returns
        -------
        (iteration, converged) where:
          - iteration: first iteration with no further changes
          - converged: True if reached stability before end
        """
        for i in range(1, len(error_sequence)):
            if error_sequence[i] == error_sequence[i - 1]:
                # Found convergence: error set hasn't changed
                return i, True
        
        # Never converged (kept changing to the end)
        return len(error_sequence) - 1, False

    @staticmethod
    def _compute_oscillation_index(
        total_toggles: int,
        total_steps: int,
        max_toggles: int,
    ) -> float:
        """
        Compute oscillation index in [0, 1].
        
        Formula: oscillation_index = total_toggles / (total_steps * max_toggles)
        Clamped to [0, 1].
        """
        if total_steps == 0 or max_toggles == 0:
            return 0.0
        
        raw_index = total_toggles / (total_steps * max_toggles)
        return min(1.0, raw_index)


# ============================================================================
# H3 Analysis: Penalty Effect on Oscillation
# ============================================================================

class H3AnalysisResult:
    """Results of H3 hypothesis testing."""

    def __init__(self):
        self.metrics_by_penalty: Dict[float, List[TrajectoryOscillationMetrics]] = {}

    def add_metrics(
        self,
        w_rp_pen: float,
        metrics: TrajectoryOscillationMetrics,
    ) -> None:
        """Add oscillation metrics for a given penalty weight."""
        if w_rp_pen not in self.metrics_by_penalty:
            self.metrics_by_penalty[w_rp_pen] = []
        self.metrics_by_penalty[w_rp_pen].append(metrics)

    def summarize(self) -> Dict[float, Dict[str, float]]:
        """
        Compute summary statistics per penalty weight.
        
        Returns
        -------
        dict mapping w_rp_pen → {
            "mean_oscillation_index": float,
            "mean_convergence_iteration": float,
            "mean_stability_score": float,
            "num_trajectories": int,
        }
        """
        summary = {}
        
        for w_rp_pen in sorted(self.metrics_by_penalty.keys()):
            metrics_list = self.metrics_by_penalty[w_rp_pen]
            
            oscillation_indices = [m.oscillation_index for m in metrics_list]
            convergence_iters = [m.convergence_iteration for m in metrics_list]
            stability_scores = [m.stability_score for m in metrics_list]
            
            summary[w_rp_pen] = {
                "mean_oscillation_index": float(np.mean(oscillation_indices)),
                "std_oscillation_index": float(np.std(oscillation_indices)),
                "mean_convergence_iteration": float(np.mean(convergence_iters)),
                "std_convergence_iteration": float(np.std(convergence_iters)),
                "mean_stability_score": float(np.mean(stability_scores)),
                "std_stability_score": float(np.std(stability_scores)),
                "num_trajectories": len(metrics_list),
            }
        
        return summary

    def test_h3_hypothesis(self) -> Dict[str, any]:
        """
        Test H3: Higher regression penalty → lower oscillation.
        
        H3 predicts: As w_rp_pen increases, oscillation_index should decrease.
        
        Returns
        -------
        dict with:
          - h3_verified: bool (True if penalty reduces oscillation)
          - penalty_effect_slope: float (slope of oscillation vs penalty)
          - convergence_effect_slope: float (slope of convergence vs penalty)
          - statistical_significance: float (Spearman correlation p-value)
        """
        penalties = sorted(self.metrics_by_penalty.keys())
        
        if len(penalties) < 3:
            logger.warning("Need at least 3 penalty values to test H3")
            return {
                "h3_verified": False,
                "reason": "insufficient_penalty_values",
            }
        
        mean_oscillations = []
        mean_convergences = []
        
        for w_rp_pen in penalties:
            metrics_list = self.metrics_by_penalty[w_rp_pen]
            mean_oscillations.append(np.mean([m.oscillation_index for m in metrics_list]))
            mean_convergences.append(np.mean([m.convergence_iteration for m in metrics_list]))
        
        # Linear fit: oscillation vs penalty
        penalties_arr = np.array(penalties)
        oscillations_arr = np.array(mean_oscillations)
        
        # Compute Spearman correlation (monotonic relationship)
        from scipy.stats import spearmanr
        try:
            spearman_r, spearman_p = spearmanr(penalties_arr, oscillations_arr)
        except:
            spearman_r, spearman_p = np.nan, np.nan
        
        # Linear regression slope
        if len(penalties) >= 2:
            polyfit = np.polyfit(penalties_arr, oscillations_arr, 1)
            slope_oscillation = float(polyfit[0])
        else:
            slope_oscillation = np.nan
        
        # Convergence improvement slope
        convergences_arr = np.array(mean_convergences)
        if len(penalties) >= 2:
            polyfit_conv = np.polyfit(penalties_arr, convergences_arr, 1)
            slope_convergence = float(polyfit_conv[0])
        else:
            slope_convergence = np.nan
        
        # H3 is verified if:
        # 1. Higher penalty → lower oscillation (negative slope)
        # 2. Statistically significant (p < 0.05)
        h3_verified = (
            slope_oscillation < 0 and
            (spearman_p < 0.05 if not np.isnan(spearman_p) else False)
        )
        
        return {
            "h3_verified": h3_verified,
            "penalty_effect_on_oscillation": {
                "slope": slope_oscillation,
                "interpretation": (
                    "negative = penalty reduces oscillation (supports H3)"
                    if slope_oscillation < 0
                    else "positive = penalty increases oscillation (contradicts H3)"
                ),
            },
            "penalty_effect_on_convergence": {
                "slope": slope_convergence,
                "interpretation": (
                    "negative = penalty accelerates convergence (supports H3)"
                    if slope_convergence < 0
                    else "positive = penalty slows convergence (contradicts H3)"
                ),
            },
            "spearman_correlation": {
                "r": float(spearman_r) if not np.isnan(spearman_r) else None,
                "p_value": float(spearman_p) if not np.isnan(spearman_p) else None,
                "significant": bool(spearman_p < 0.05) if not np.isnan(spearman_p) else False,
            },
            "summary_statistics": self.summarize(),
        }
