# ============================================================================
# File: src/metrics/correction_metrics.py
# ============================================================================
"""
Regression-Aware Correction Score (RACS) computation — v3 fixed implementation.

ROOT CAUSE of the always-0.7 bug in v2:
  With weights (0.5, 0.3, 0.2) and formula  RACS = w_err*ERR - w_rp*RP + w_tfp*TFP
  When a proof goes from N errors → 0 errors:
    ERR = 1.0   (all previous errors fixed)
    RP  = 0.0   (no new errors introduced)
    TFP = 1.0   (all flagged steps addressed)
  ⟹ RACS = 0.5*1.0 - 0.3*0.0 + 0.2*1.0 = 0.70 — always 0.70!

  The ceiling is structural: max(RACS) = w_err + w_tfp = 0.5 + 0.2 = 0.70.

FIXES in v3:
  1. Weights now sum to 1.0 correctly: ERR=0.5, RP_penalty=0.3 kept as penalty
     but TFP raised to 0.5, then normalised so max reachable = 1.0.
  2. New formula (calibrated):
       RACS = (w_err * ERR + w_tfp * TFP) * (1 - w_rp_pen * RP_clamped)
     This multiplicative penalty means:
       - Perfect correction (ERR=1, TFP=1, RP=0) → RACS = 1.0
       - Perfect but with some regression      → RACS < 1.0 (proportional)
       - No fix + regression                   → RACS can approach 0
  3. RP is normalised to [0,1] before applying the penalty weight.
  4. When error_set_previous is empty (first iteration baseline), a synthetic
     "first-iteration RACS" is computed from static quality alone.

NOTE: This metric is also known as CRS (Correction Reasoning Score) in some
contexts, but RACS (Regression-Aware Correction Score) is the preferred name
as it emphasizes the regression penalty mechanism which distinguishes it from
naive correction metrics.
"""

import numpy as np
import logging
from typing import Set, Optional
from ..models.types import ErrorSet
from ..models.schemas import CorrectionMetricsOutput, EvaluationMetrics
from .error_tracker import ErrorTracker

logger = logging.getLogger(__name__)


class CorrectionMetricsCalculator:
    """
    Computes ERR, RP, TFP and overall CRS with calibrated weights.

    Parameters
    ----------
    w_err   : weight for Error Resolution Rate     (default 0.60)
    w_tfp   : weight for Targeted Fix Precision    (default 0.40)
    w_rp_pen: penalty weight for Regression (multiplied into 1-RP term)
              (default 0.20 — aggressive penalty for introducing new errors)
    """

    def __init__(
        self,
        w_err: float = 0.60,
        w_tfp: float = 0.40,
        w_rp_pen: float = 0.20,
    ):
        if abs(w_err + w_tfp - 1.0) > 1e-6:
            raise ValueError(
                f"w_err + w_tfp must equal 1.0, got {w_err} + {w_tfp} = {w_err + w_tfp}"
            )
        self.w_err = w_err
        self.w_tfp = w_tfp
        self.w_rp_pen = w_rp_pen
        self.tracker = ErrorTracker()

    # ------------------------------------------------------------------
    # Component metrics
    # ------------------------------------------------------------------

    def compute_err(self, prev_errors: Set[int], curr_errors: Set[int]) -> float:
        """
        Error Resolution Rate.
        ERR = |E_prev \\ E_curr| / |E_prev|
        Returns 1.0 if there were no previous errors (nothing to fix = perfect).
        """
        if len(prev_errors) == 0:
            return 1.0
        fixed = prev_errors - curr_errors
        return len(fixed) / len(prev_errors)

    def compute_rp_normalised(
        self, prev_errors: Set[int], curr_errors: Set[int]
    ) -> float:
        """
        Regression Penalty, normalised to [0, 1].
        RP_raw = |E_curr \\ E_prev| / max(|E_prev|, 1)
        RP_norm = min(RP_raw, 1.0)   (clamp so the multiplier stays in [0,1])
        """
        new_errors = curr_errors - prev_errors
        if not new_errors:
            return 0.0
        denom = max(len(prev_errors), 1)
        rp_raw = len(new_errors) / denom
        return min(rp_raw, 1.0)

    def compute_tfp(
        self,
        flagged_steps: Set[int],
        curr_errors: Set[int],
    ) -> float:
        """
        Targeted Fix Precision.
        TFP = |F \\ E_curr| / |F|
        Returns 1.0 if nothing was flagged (evaluator found no specific issues).
        """
        if len(flagged_steps) == 0:
            return 1.0
        fixed_flagged = flagged_steps - curr_errors
        return len(fixed_flagged) / len(flagged_steps)

    # ------------------------------------------------------------------
    # Composite CRS
    # ------------------------------------------------------------------

    def compute_crs(
        self,
        prev_error_set: ErrorSet,
        curr_error_set: ErrorSet,
        flagged_steps: Set[int],
        total_steps_current: int = 0,  # kept for API compat, unused now
    ) -> CorrectionMetricsOutput:
        """
        Compute the full Regression-Aware Correction Score (RACS).

        Formula (multiplicative penalty):
            quality = w_err * ERR + w_tfp * TFP
            RACS_raw = quality * (1 - w_rp_pen * RP_norm)
            RACS     = clip(RACS_raw, 0, 1)

        Maximum achievable:
            ERR=1, TFP=1, RP_norm=0 → RACS = 1.0 * 1.0 = 1.0  ✓
        """
        prev_all = self.tracker.get_all_errors(prev_error_set)
        curr_all = self.tracker.get_all_errors(curr_error_set)

        err = self.compute_err(prev_all, curr_all)
        rp_norm = self.compute_rp_normalised(prev_all, curr_all)
        tfp = self.compute_tfp(flagged_steps, curr_all)

        quality = self.w_err * err + self.w_tfp * tfp
        crs_raw = quality * (1.0 - self.w_rp_pen * rp_norm)
        crs = float(np.clip(crs_raw, 0.0, 1.0))

        is_mixed = bool(prev_all - curr_all) and bool(curr_all - prev_all)

        if is_mixed:
            logger.warning(
                "Mixed transition: fixed=%d, introduced=%d, CRS=%.3f (raw=%.3f)",
                len(prev_all - curr_all),
                len(curr_all - prev_all),
                crs,
                crs_raw,
            )

        return CorrectionMetricsOutput(
            error_resolution_rate=err,
            regression_penalty=rp_norm,         # normalised [0,1]
            targeted_fix_precision=tfp,
            correction_reasoning_score=crs,
            crs_raw=crs_raw,
            errors_fixed=len(prev_all - curr_all),
            errors_introduced=len(curr_all - prev_all),
            flagged_fixed=len(flagged_steps - curr_all),
            total_flagged=len(flagged_steps),
            is_mixed_transition=is_mixed,
            prev_error_count=len(prev_all),
            curr_error_count=len(curr_all),
        )

    def compute_first_iteration_crs(
        self, metrics: EvaluationMetrics
    ) -> CorrectionMetricsOutput:
        """
        Synthetic RACS for the very first iteration (no previous error set).
        Derived purely from static quality scores rather than delta-tracking.

        Formula:
            static_quality = (completeness_score / 5.0) * 0.6
                           + (assumption_use_score / 5.0) * 0.4
            error_penalty  = num_errors / (num_errors + 1)   [sigmoid-like]
            RACS_first     = static_quality * (1 - 0.5 * error_penalty)
        """
        static_quality = (
            (metrics.completeness_score / 5.0) * 0.6
            + (metrics.assumption_use_score / 5.0) * 0.4
        )
        n_errors = metrics.total_errors
        error_penalty = n_errors / (n_errors + 1) if n_errors >= 0 else 0.0
        crs_raw = static_quality * (1.0 - 0.5 * error_penalty)
        crs = float(np.clip(crs_raw, 0.0, 1.0))

        # Build a dummy "previous" error set = empty, "current" = all errors found
        empty_set: ErrorSet = {
            "hallucinations": set(),
            "missing_steps": set(),
            "operator_errors": set(),
            "assumption_violations": set(),
        }
        curr_set = ErrorTracker.error_set_from_metrics(metrics)
        curr_all = ErrorTracker.get_all_errors(curr_set)
        flagged = set(metrics.flagged_steps)

        return CorrectionMetricsOutput(
            error_resolution_rate=0.0,      # No prior to resolve from
            regression_penalty=0.0,
            targeted_fix_precision=1.0,     # Nothing was flagged before
            correction_reasoning_score=crs,
            crs_raw=crs_raw,
            errors_fixed=0,
            errors_introduced=len(curr_all),
            flagged_fixed=0,
            total_flagged=len(flagged),
            is_mixed_transition=False,
            prev_error_count=0,
            curr_error_count=len(curr_all),
        )