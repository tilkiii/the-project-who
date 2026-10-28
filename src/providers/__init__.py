"""
Provider abstraction package for who??.
Supports Google Gemini, OpenAI, Anthropic, Groq, DeepSeek, OpenRouter, Ollama, and offline fallback.
"""

from src.providers.base import BaseLLMProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.manager import ProviderManager, provider_manager
from src.providers.offline_provider import OfflineProvider
from src.providers.openai_provider import OpenAICompatibleProvider

__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "OfflineProvider",
    "ProviderManager",
    "provider_manager",
]
