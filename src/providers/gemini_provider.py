"""
Google Gemini provider for who??.
Supports Gemini 3.1 Flash Lite, Gemini 3.8 Flash, and Gemini Embedding.
Uses official google-genai SDK with resilient model-chain fallback.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from src.providers.base import BaseLLMProvider
from src.utils.logger import get_logger

logger = get_logger("who.providers.gemini")

# Fast, verified model priority list for generation and embeddings
CANDIDATE_GENERATION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

CANDIDATE_EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI integration."""

    provider_id: str = "gemini"
    display_name: str = "Google Gemini"
    default_model: str = "gemini-3.5-flash-lite"
    default_embedding_model: str = "gemini-embedding-001"
    env_keys: List[str] = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self) -> None:
        self._client: Any = None
        self._init_sdk()

    def _get_api_key(self) -> Optional[str]:
        key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
        if not key:
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_file = Path(__file__).resolve().parent.parent.parent / ".env"
                if env_file.exists():
                    load_dotenv(dotenv_path=env_file, override=False)
                else:
                    load_dotenv(override=False)
                key = (
                    os.getenv("GEMINI_API_KEY", "").strip()
                    or os.getenv("GOOGLE_API_KEY", "").strip()
                )
            except Exception:
                pass
        return key or None

    def _init_sdk(self) -> None:
        api_key = self._get_api_key()
        if not api_key:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.debug(f"google-genai SDK initialization note: {e}")
            self._client = None

    def is_available(self) -> bool:
        key = self._get_api_key()
        if key and self._client is None:
            self._init_sdk()
        return bool(key)

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
            raise ValueError("GEMINI_API_KEY is not set.")

        if self._client is None:
            self._init_sdk()

        preferred_model = model or os.getenv("GEMINI_MODEL", self.default_model)
        if preferred_model.startswith("gemini/"):
            preferred_model = preferred_model[len("gemini/"):]

        # Build trial models order
        trial_models = [preferred_model]
        for m in CANDIDATE_GENERATION_MODELS:
            if m not in trial_models:
                trial_models.append(m)

        # 1. Try google-genai SDK with trial models
        if self._client is not None:
            from google.genai import types
            config_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            config = types.GenerateContentConfig(**config_kwargs)

            for target_model in trial_models:
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )
                    if response and hasattr(response, "text") and response.text:
                        return response.text
                except Exception as e:
                    logger.debug(f"Gemini SDK generation with model '{target_model}' error: {e}")

        # 2. REST fallback
        for target_model in trial_models:
            try:
                out = self._generate_rest(
                    api_key=api_key,
                    model=target_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if out:
                    return out
            except Exception as e:
                logger.debug(f"Gemini REST fallback with model '{target_model}' error: {e}")

        raise RuntimeError("All Gemini candidate models failed to generate content.")

    def _generate_rest(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_chunks = [p.get("text", "") for p in parts if "text" in p]
                return "".join(text_chunks)
            return ""

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if self._client is None:
            self._init_sdk()

        preferred_model = model or os.getenv("GEMINI_EMBEDDING_MODEL", self.default_embedding_model)
        if preferred_model.startswith("gemini/"):
            preferred_model = preferred_model[len("gemini/"):]

        trial_models = [preferred_model]
        for m in CANDIDATE_EMBEDDING_MODELS:
            if m not in trial_models:
                trial_models.append(m)

        # 1. Try google-genai SDK
        if self._client is not None:
            for target_model in trial_models:
                try:
                    resp = self._client.models.embed_content(
                        model=target_model,
                        contents=text,
                    )
                    if resp and hasattr(resp, "embeddings") and resp.embeddings:
                        values = resp.embeddings[0].values
                        if values:
                            return list(values)
                except Exception as e:
                    logger.debug(f"Gemini embed SDK with model '{target_model}' error: {e}")

        # 2. Universal hash fallback
        from src.providers.offline_provider import hash_embedding
        return hash_embedding(text)
