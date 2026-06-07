# ============================================================================
# File: src/llm/base_provider.py
# ============================================================================
"""Base LLM provider with retry logic and per-call temperature override."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import time
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        timeout: int = 600,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Make the actual API request.
        Implementations MUST honour the `temperature` parameter when provided.
        """
        pass

    def invoke(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Invoke the LLM with retry logic.

        Parameters
        ----------
        messages    : list of {"role": ..., "content": ...} dicts
        temperature : overrides the instance default when provided
        **kwargs    : passed through to _make_request
        """
        effective_temp = temperature if temperature is not None else self.temperature
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    wait = self.retry_delay * attempt
                    logger.info(
                        "[%s] Retry %d/%d (waiting %ds)...",
                        self.__class__.__name__, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)

                return self._make_request(messages, temperature=effective_temp, **kwargs)

            except Exception as exc:
                last_error = exc
                msg = str(exc)

                if "504" in msg or "timeout" in msg.lower():
                    logger.warning("[%s] Timeout on attempt %d", self.__class__.__name__, attempt + 1)
                elif "429" in msg or "rate limit" in msg.lower():
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "[%s] Rate-limit on attempt %d; sleeping %ds",
                        self.__class__.__name__, attempt + 1, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning("[%s] Error on attempt %d: %s", self.__class__.__name__, attempt + 1, msg)

        raise RuntimeError(
            f"All {self.max_retries} retries exhausted. Last error: {last_error}"
        )