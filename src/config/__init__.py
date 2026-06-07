# ============================================================================
# File: src/config/__init__.py
# ============================================================================
from .temperatures import TemperatureConfig, DEFAULT_TEMPERATURES, get_prover_temperature, get_judge_temperature

__all__ = [
    "TemperatureConfig",
    "DEFAULT_TEMPERATURES",
    "get_prover_temperature",
    "get_judge_temperature",
]