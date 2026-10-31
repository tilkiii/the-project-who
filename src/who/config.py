"""
Configuration management and multi-provider LLM abstraction for who??.

Supported AI Providers:
  • Google Gemini   (GEMINI_API_KEY / GOOGLE_API_KEY)
  • OpenAI          (OPENAI_API_KEY)
  • Anthropic       (ANTHROPIC_API_KEY)
  • Groq            (GROQ_API_KEY)
  • DeepSeek        (DEEPSEEK_API_KEY)
  • OpenRouter      (OPENROUTER_API_KEY)
  • Mistral         (MISTRAL_API_KEY)
  • Ollama          (Local, no key required - OLLAMA_API_BASE)
  • vLLM            (Local, no key required - VLLM_API_BASE)

Configure keys in a `.env` file at the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load .env file BEFORE any provider or module is imported!
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)
else:
    load_dotenv(override=False)

from pydantic import BaseModel, Field

from who.providers.manager import provider_manager
from who.providers.offline_provider import hash_embedding
from who.utils.logger import get_logger
from who.utils.text_cleaner import sanitize_slug

logger = get_logger("who.config")


class TargetDirectories(BaseModel):
    """Encapsulates the standard filesystem layout for a target profile."""

    target_name: str
    target_slug: str
    base_dir: Path
    raw_data_dir: Path
    learned_skills_dir: Path
    vault_dir: Path
    chroma_dir: Path

    def ensure_exists(self) -> TargetDirectories:
        """Create all profile subdirectories if they do not exist."""
        for d in [self.base_dir, self.raw_data_dir, self.learned_skills_dir, self.vault_dir, self.chroma_dir]:
            d.mkdir(parents=True, exist_ok=True)
        return self


class Settings(BaseModel):
    """Global configuration settings for who??."""

    profiles_dir: Path = Field(default_factory=lambda: Path(os.getenv("PROFILES_DIR", "profiles")))

    # Selected provider override (gemini, openai, anthropic, groq, deepseek, ollama, etc.)
    provider_override: Optional[str] = Field(default_factory=lambda: os.getenv("WHO_PROVIDER") or os.getenv("LLM_PROVIDER"))

    # Active provider properties (resolved dynamically)
    @property
    def active_provider(self) -> str:
        return provider_manager.detect_active_provider().provider_id

    @property
    def active_provider_name(self) -> str:
        return provider_manager.detect_active_provider().display_name

    @property
    def default_llm_model(self) -> str:
        env_model = os.getenv("DEFAULT_LLM_MODEL")
        if env_model:
            return env_model
        return provider_manager.detect_active_provider().default_model

    @property
    def default_embedding_model(self) -> str:
        env_emb = os.getenv("DEFAULT_EMBEDDING_MODEL")
        if env_emb:
            return env_emb
        return provider_manager.detect_active_provider().default_embedding_model

    # API Keys
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    deepseek_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))

    # Local Engine URLs
    ollama_api_base: str = Field(default_factory=lambda: os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
    vllm_api_base: str = Field(default_factory=lambda: os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"))

    # Neo4j Settings
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # Epistemic Settings
    min_confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.65")))
    evidence_min_quotes: int = Field(default_factory=lambda: int(os.getenv("EVIDENCE_MIN_QUOTES", "2")))
    strict_epistemic_mode: bool = Field(default_factory=lambda: os.getenv("STRICT_EPISTEMIC_MODE", "true").lower() in ("true", "1", "yes"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))

    def get_target_dirs(self, target_name: str) -> TargetDirectories:
        """Resolve and ensure standard profile directory structure for a target."""
        slug = sanitize_slug(target_name)
        base = self.profiles_dir / slug
        raw_data = base / "raw_data"
        learned_skills = base / "learned_skills"
        vault = base / "vault"
        chroma = vault / "chroma"
        td = TargetDirectories(
            target_name=target_name,
            target_slug=slug,
            base_dir=base,
            raw_data_dir=raw_data,
            learned_skills_dir=learned_skills,
            vault_dir=vault,
            chroma_dir=chroma,
        )
        td.ensure_exists()
        return td


# Singleton configuration settings
settings = Settings()


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    provider_id: Optional[str] = None,
) -> str:
    """
    Unified LLM call supporting Google Gemini, OpenAI, Claude, Groq, DeepSeek, and Ollama.
    Automatically routes to the configured provider or falls back safely.
    """
    provider = provider_manager.get_provider(provider_id)
    try:
        return provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning(
            f"Provider '{provider.display_name}' call failed: {e}. "
            "Falling back to local corpus-grounded synthesis."
        )
        fallback = provider_manager.get_provider("offline")
        return fallback.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_embedding(text: str, model: Optional[str] = None, provider_id: Optional[str] = None) -> List[float]:
    """
    Generate normalized vector embeddings via the active provider (e.g. Gemini text-embedding-004,
    OpenAI text-embedding-3-small) or fallback to hash-based pseudo embedding.
    """
    provider = provider_manager.get_provider(provider_id)
    try:
        return provider.embed(text=text, model=model)
    except Exception as e:
        logger.debug(f"Embedding via '{provider.display_name}' failed ({e}). Using hash embedding.")
        return hash_embedding(text)


def _fallback_llm_generation(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Backward compatibility helper for offline generation."""
    return provider_manager.get_provider("offline").generate(prompt, system_prompt)
