"""
OpenAI-compatible LLM and Embedding provider for who??.
Handles OpenAI, Groq, DeepSeek, OpenRouter, Mistral, Ollama, and vLLM.
Uses httpx directly without external heavy dependencies.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from src.providers.base import BaseLLMProvider
from src.utils.logger import get_logger

logger = get_logger("who.providers.openai_compatible")


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic provider for any OpenAI-compatible API (OpenAI, Groq, DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        provider_id: str,
        display_name: str,
        env_keys: List[str],
        base_url: str,
        default_model: str,
        default_embedding_model: str = "",
        requires_api_key: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider_id = provider_id
        self.display_name = display_name
        self.env_keys = env_keys
        self._default_base_url = base_url
        self.default_model = default_model
        self.default_embedding_model = default_embedding_model
        self.requires_api_key = requires_api_key
        self.extra_headers = extra_headers or {}

    def _get_api_key(self) -> Optional[str]:
        for k in self.env_keys:
            val = os.getenv(k, "").strip()
            if val:
                return val
        return None

    def _get_base_url(self) -> str:
        # Check custom base url env var (e.g. OPENAI_API_BASE, OLLAMA_API_BASE)
        env_base = os.getenv(f"{self.provider_id.upper()}_API_BASE")
        if env_base:
            return env_base.rstrip("/")
        if self.provider_id == "openai":
            alt = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            if alt:
                return alt.rstrip("/")
        return self._default_base_url.rstrip("/")

    def is_available(self) -> bool:
        if not self.requires_api_key:
            # Only probe local servers if explicitly configured or requested
            has_custom_base = bool(os.getenv(f"{self.provider_id.upper()}_API_BASE"))
            requested = os.getenv("WHO_PROVIDER", "").strip().lower() == self.provider_id
            if not has_custom_base and not requested:
                return False

            try:
                base = self._get_base_url()
                with httpx.Client(timeout=0.8) as client:
                    r = client.get(f"{base}/models" if "localhost" in base else base)
                    return r.status_code < 500
            except Exception:
                return False
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
        if self.requires_api_key and not api_key:
            raise ValueError(f"API key missing for {self.display_name}. Set {self.env_keys[0]}.")

        target_model = model or os.getenv(f"{self.provider_id.upper()}_MODEL", self.default_model)
        # Strip provider prefix if present
        if target_model.startswith(f"{self.provider_id}/"):
            target_model = target_model[len(f"{self.provider_id}/"):]

        base_url = self._get_base_url()
        endpoint = f"{base_url}/chat/completions"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or ""
            return ""

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        api_key = self._get_api_key()
        if self.requires_api_key and not api_key:
            raise ValueError(f"API key missing for {self.display_name}.")

        target_model = model or os.getenv(f"{self.provider_id.upper()}_EMBEDDING_MODEL", self.default_embedding_model)
        if not target_model:
            # Fall back to hash embedding if provider has no embedding model
            from src.providers.offline_provider import hash_embedding
            return hash_embedding(text)

        if target_model.startswith(f"{self.provider_id}/"):
            target_model = target_model[len(f"{self.provider_id}/"):]

        base_url = self._get_base_url()
        endpoint = f"{base_url}/embeddings"

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": target_model,
            "input": text,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
