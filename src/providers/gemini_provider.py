"""
Google Gemini provider for who??.
Supports Gemini 2.5 Flash, 2.5 Pro, 1.5 Flash, and text-embedding-004.
Primary implementation uses google-genai official SDK with resilient HTTP fallback.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from src.providers.base import BaseLLMProvider
from src.utils.logger import get_logger

logger = get_logger("who.providers.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI integration."""

    provider_id: str = "gemini"
    display_name: str = "Google Gemini"
    default_model: str = "gemini-2.5-flash"
    default_embedding_model: str = "text-embedding-004"
    env_keys: List[str] = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self) -> None:
        self._client: Any = None
        self._init_sdk()

    def _get_api_key(self) -> Optional[str]:
        return (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or None
        )

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
            raise ValueError("GEMINI_API_KEY is not set.")

        target_model = model or os.getenv("GEMINI_MODEL", self.default_model)
        # Strip provider prefix if given as 'gemini/gemini-2.5-flash'
        if target_model.startswith("gemini/"):
            target_model = target_model[len("gemini/"):]

        # 1. Try google-genai SDK
        if self._client is not None:
            try:
                from google.genai import types
                config_kwargs: Dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if system_prompt:
                    config_kwargs["system_instruction"] = system_prompt

                config = types.GenerateContentConfig(**config_kwargs)
                response = self._client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"google-genai SDK generation error ({e}), trying REST endpoint fallback...")

        # 2. Resilient REST endpoint fallback
        return self._generate_rest(
            api_key=api_key,
            model=target_model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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

        with httpx.Client(timeout=45.0) as client:
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

        target_model = model or os.getenv("GEMINI_EMBEDDING_MODEL", self.default_embedding_model)
        if target_model.startswith("gemini/"):
            target_model = target_model[len("gemini/"):]

        # 1. Try google-genai SDK
        if self._client is not None:
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
                logger.debug(f"google-genai embed error ({e}), trying REST fallback...")

        # 2. REST fallback
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:embedContent?key={api_key}"
        payload = {
            "model": f"models/{target_model}",
            "content": {"parts": [{"text": text}]},
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding", {}).get("values", [])
            return [float(x) for x in embedding]
