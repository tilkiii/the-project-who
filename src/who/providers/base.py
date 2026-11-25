"""
Base class for LLM and Embedding Providers in who??.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """Abstract base provider for LLM completion and embeddings."""

    provider_id: str = "base"
    display_name: str = "Base Provider"
    default_model: str = ""
    default_embedding_model: str = ""
    env_keys: List[str] = []

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready to be used."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response text from prompt and optional system instructions."""
        pass

    @abstractmethod
    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Generate normalized vector embedding for the given text."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Metadata for CLI status display."""
        return {
            "id": self.provider_id,
            "name": self.display_name,
            "available": self.is_available(),
            "default_model": self.default_model,
            "default_embedding": self.default_embedding_model,
            "env_keys": self.env_keys,
        }
