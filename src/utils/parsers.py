# ============================================================================
# File: src/utils/parsers.py
# ============================================================================
"""Parsing utilities for LLM outputs."""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse and clean LLM responses."""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text that may contain markdown fences."""
        # Try ```json ... ``` first
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try raw { ... }
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug("JSON decode error: %s", e)
            # Attempt repair: remove trailing commas
            repaired = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    @staticmethod
    def clean_latex(text: str) -> str:
        """Clean LaTeX output from LLM (remove markdown fences, ensure structure)."""
        text = re.sub(r"```latex\s*", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        if r"\documentclass" not in text:
            text = (
                r"\documentclass{article}"
                "\n"
                r"\begin{document}"
                "\n"
                + text
                + "\n"
                + r"\end{document}"
            )
        return text.strip()

    @staticmethod
    def extract_error_steps(text: str) -> Dict[str, list]:
        """
        Extract step-level error indices from feedback text.
        Looks for patterns like: Hallucination Steps: [1, 5, 12]
        """
        patterns = {
            "hallucinations": r"Hallucination Steps?:\s*\[([\d,\s]*)\]",
            "missing_steps": r"Missing Steps?:\s*\[([\d,\s]*)\]",
            "operator_errors": r"Operator Error Steps?:\s*\[([\d,\s]*)\]",
            "assumption_violations": r"Assumption Violation Steps?:\s*\[([\d,\s]*)\]",
            "flagged_steps": r"Flagged Steps?:\s*\[([\d,\s]*)\]",
        }
        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1).strip():
                result[key] = [
                    int(s.strip()) for s in match.group(1).split(",") if s.strip()
                ]
            else:
                result[key] = []
        return result