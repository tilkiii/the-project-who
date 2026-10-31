"""
Anthropic Claude provider for who??.
Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus.
Uses httpx directly to call https://api.anthropic.com/v1/messages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from who.providers.base import BaseLLMProvider
from who.utils.logger import get_logger

logger = get_logger("who.providers.anthropic")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude integration."""

    provider_id: str = "anthropic"
    display_name: str = "Anthropic Claude"
    default_model: str = "claude-3-5-haiku-20241022"
    default_embedding_model: str = ""  # Anthropic has no embedding endpoint
    env_keys: List[str] = ["ANTHROPIC_API_KEY"]

    def _get_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY", "").strip() or None

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        target_model = model or os.getenv("ANTHROPIC_MODEL", self.default_model)
        if target_model.startswith("anthropic/"):
            target_model = target_model[len("anthropic/"):]

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=45.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content_blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            return "".join(text_parts)

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        # Anthropic does not provide an embedding API; fallback to local hash embedding
        from who.providers.offline_provider import hash_embedding
        return hash_embedding(text)
