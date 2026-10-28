"""
Provider Manager and Registry for who??.
Orchestrates multi-provider detection, routing, and status auditing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import BaseLLMProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.offline_provider import OfflineProvider
from src.providers.openai_provider import OpenAICompatibleProvider
from src.utils.logger import get_logger

logger = get_logger("who.providers.manager")


class ProviderManager:
    """Manages all supported LLM and embedding providers."""

    def __init__(self) -> None:
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._offline_fallback = OfflineProvider()
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        # 1. Google Gemini (primary recommendation for the user)
        self.register(GeminiProvider())

        # 2. OpenAI
        self.register(
            OpenAICompatibleProvider(
                provider_id="openai",
                display_name="OpenAI (GPT-4o / GPT-4o-mini)",
                env_keys=["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
                default_embedding_model="text-embedding-3-small",
            )
        )

        # 3. Anthropic Claude
        self.register(AnthropicProvider())

        # 4. Groq (ultra-fast open models)
        self.register(
            OpenAICompatibleProvider(
                provider_id="groq",
                display_name="Groq Cloud (Llama 3.3 / Mixtral)",
                env_keys=["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
                default_model="llama-3.3-70b-versatile",
                default_embedding_model="",
            )
        )

        # 5. DeepSeek
        self.register(
            OpenAICompatibleProvider(
                provider_id="deepseek",
                display_name="DeepSeek (V3 / R1)",
                env_keys=["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1",
                default_model="deepseek-chat",
                default_embedding_model="",
            )
        )

        # 6. OpenRouter (universal proxy)
        self.register(
            OpenAICompatibleProvider(
                provider_id="openrouter",
                display_name="OpenRouter (Multi-model Gateway)",
                env_keys=["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                default_model="google/gemini-2.5-flash",
                default_embedding_model="",
                extra_headers={"HTTP-Referer": "https://github.com/tilkiii/who", "X-Title": "who??"},
            )
        )

        # 7. Mistral AI
        self.register(
            OpenAICompatibleProvider(
                provider_id="mistral",
                display_name="Mistral AI",
                env_keys=["MISTRAL_API_KEY"],
                base_url="https://api.mistral.ai/v1",
                default_model="mistral-small-latest",
                default_embedding_model="mistral-embed",
            )
        )

        # 8. Local Ollama (no key required)
        ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        self.register(
            OpenAICompatibleProvider(
                provider_id="ollama",
                display_name="Ollama (Local Open Source Models)",
                env_keys=[],
                base_url=f"{ollama_base}/v1" if not ollama_base.endswith("/v1") else ollama_base,
                default_model="llama3.2",
                default_embedding_model="nomic-embed-text",
                requires_api_key=False,
            )
        )

        # 9. Local vLLM (no key required)
        vllm_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        self.register(
            OpenAICompatibleProvider(
                provider_id="vllm",
                display_name="vLLM (Local High-Throughput Engine)",
                env_keys=[],
                base_url=vllm_base,
                default_model="mistral-7b",
                default_embedding_model="",
                requires_api_key=False,
            )
        )

    def register(self, provider: BaseLLMProvider) -> None:
        self.providers[provider.provider_id] = provider

    def detect_active_provider(self) -> BaseLLMProvider:
        """
        Detect and return the active provider based on environment keys or explicit override.
        Priority:
          1. Explicit override via WHO_PROVIDER or LLM_PROVIDER env variable
          2. Gemini (if key found)
          3. OpenAI (if key found)
          4. Anthropic (if key found)
          5. Groq (if key found)
          6. DeepSeek (if key found)
          7. OpenRouter (if key found)
          8. Mistral (if key found)
          9. Ollama (if running locally)
          10. vLLM (if running locally)
          11. Offline Fallback
        """
        # Explicit override
        override = (os.getenv("WHO_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
        if override and override in self.providers:
            provider = self.providers[override]
            if provider.is_available():
                return provider
            logger.warning(f"Requested provider '{override}' is not fully configured (missing API key).")

        # Standard priority check
        priority_order = [
            "gemini",
            "openai",
            "anthropic",
            "groq",
            "deepseek",
            "openrouter",
            "mistral",
            "ollama",
            "vllm",
        ]

        for pid in priority_order:
            provider = self.providers.get(pid)
            if provider and provider.is_available():
                return provider

        return self._offline_fallback

    def get_provider(self, provider_id: Optional[str] = None) -> BaseLLMProvider:
        if provider_id and provider_id in self.providers:
            return self.providers[provider_id]
        return self.detect_active_provider()

    def list_all_statuses(self) -> List[Dict[str, Any]]:
        active = self.detect_active_provider()
        results = []
        for pid, p in self.providers.items():
            info = p.get_info()
            info["is_active"] = (p.provider_id == active.provider_id)
            results.append(info)
        # Add offline info
        off_info = self._offline_fallback.get_info()
        off_info["is_active"] = (active.provider_id == "offline")
        results.append(off_info)
        return results


# Global singleton instance
provider_manager = ProviderManager()
