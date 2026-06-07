# ============================================================================
# File: src/llm/openrouter_provider.py
# ============================================================================
"""OpenRouter provider."""

from typing import Optional, List, Dict
import requests
from .base_provider import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": kwargs.get("http_referer", "http://localhost:3000"),
            "X-Title": kwargs.get("x_title", "Convergence Proof Agent"),
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]