# ============================================================================
# File: src/config/temperatures.py
# ============================================================================
"""
Per-phase temperature configuration.

WHY DIFFERENT TEMPERATURES PER PHASE:
  - Generation (prover): Higher temperature (0.7) encourages creative,
    exploratory proof strategies rather than always defaulting to the same
    textbook approach.

  - Evaluation (judge): Near-zero temperature (0.05) is critical for
    reproducible, deterministic structured JSON output. High temperature
    here causes inconsistent verdicts and malformed JSON.

  - Correction (prover iter 2+): Moderate temperature (0.4) — lower than
    generation since we are making targeted edits, not free-form writing.
    Too high and the model rewrites unrelated sections; too low and it
    makes minimal/cosmetic changes.

  - Verification (judge iter 2+): Same near-zero as evaluation (0.05)
    because this is also a structured assessment task.
"""

from dataclasses import dataclass


@dataclass
class TemperatureConfig:
    """Temperatures used at each phase of the pipeline."""

    # Prover temperatures
    generation: float = 0.7      # Initial proof generation (creative)
    correction: float = 0.4      # Subsequent correction iterations (targeted)

    # Judge temperatures
    evaluation: float = 0.05     # Initial evaluation (deterministic JSON)
    verification: float = 0.05   # Re-evaluation after correction (deterministic)


# Default configuration used across the project
DEFAULT_TEMPERATURES = TemperatureConfig()


def get_prover_temperature(iteration: int, config: TemperatureConfig = DEFAULT_TEMPERATURES) -> float:
    """Return the appropriate prover temperature for a given iteration."""
    return config.generation if iteration == 0 else config.correction


def get_judge_temperature(iteration: int, config: TemperatureConfig = DEFAULT_TEMPERATURES) -> float:
    """Return the appropriate judge temperature for a given iteration."""
    return config.evaluation if iteration <= 1 else config.verification