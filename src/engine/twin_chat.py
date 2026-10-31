"""
Interactive digital twin chat engine for who??.
Supports tone toggling between formal public keynote vs informal 1-on-1 dialogue.
Corresponds to Layer 7 (Character Simulation Memory).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import console, get_logger
from src.utils.text_cleaner import tokenize_words, tr_lower
from src.vault.vector_embedder import VectorEmbedder

logger = get_logger("who.engine.twin_chat")


class TwinChat:
    """Simulates real-time interactive dialogue with the digital twin with dynamic tone modes."""

    def __init__(self, target_name: str, target_dirs: Any, mode: str = "autodetect"):
        self.target_name = target_name
        self.dirs = target_dirs
        self.mode = mode.lower()  # 'autodetect', 'formal', 'informal'
        self.history: List[Dict[str, str]] = []
        self.vector_store = VectorEmbedder(target_name, chroma_dir=self.dirs.chroma_dir)

        # Load character config & mindset
        self.character = self._load_character()
        self.mindset_summary = self._load_mindset()

    def respond(self, user_message: str) -> str:
        """Generate twin response conforming to character lore, epistemic rules, and selected tone."""
        # 1. Determine active tone
        active_tone = self._resolve_tone(user_message)

        # 2. Retrieve relevant context quotes
        context_chunks = self.vector_store.search_similar(user_message, top_k=3)
        quotes = [c["text"] for c in context_chunks if c.get("text")]
        quotes_ctx = "\n".join(f"- Quotation: \"{q}\"" for q in quotes[:2])

        # 3. Assemble System Prompt
        style = self.character.get("style", {})
        all_style = ", ".join(style.get("all", ["probabilistic", "evidence-based"]))
        tone_instruction = (
            "TONE: Formal public keynote register. Use precise, analytical vocabulary, explicit reasoning, and structural rigor."
            if active_tone == "formal"
            else "TONE: Direct 1-on-1 conversational register. Use collegial, succinct, candid phrasing while preserving empirical integrity."
        )

        mindset_excerpt = self.mindset_summary[:500].strip()
        system_prompt = (
            f"You are the cognitive digital twin of {self.target_name}.\n"
            f"REASONING & STYLE CONSTRAINTS:\n"
            f"- Adhere to behavioral style: {all_style}.\n"
            f"- {tone_instruction}\n"
            "- Always maintain scientific rigor and probabilistic humility: do not fabricate personal certainty beyond evidence.\n"
            f"- BIO CONTEXT: {' '.join(self.character.get('bio', []))}\n"
            + (f"- COGNITIVE MINDSET SUMMARY:\n{mindset_excerpt}\n" if mindset_excerpt else "")
        )

        # Conversation memory
        conv_memory = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in self.history[-6:])

        user_prompt = (
            f"PRIMARY EVIDENCE MEMORY:\n{quotes_ctx}\n\n"
            f"RECENT DIALOGUE:\n{conv_memory}\n\n"
            f"USER: {user_message}\n\n"
            f"{self.target_name.upper()} RESPONSE:"
        )

        response_text = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.35)

        # Save to interaction memory
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "twin", "content": response_text})

        return response_text

    def _resolve_tone(self, user_message: str) -> str:
        """Resolve tone mode if set to autodetect."""
        if self.mode in ("formal", "informal"):
            return self.mode

        # Autodetect heuristic (Turkish & English)
        tokens = set(tokenize_words(user_message, min_len=2))
        informal_markers = {
            "hey", "hi", "yo", "sup", "gonna", "wanna", "lol", "dude", "bro", "cool",
            "selam", "merhaba", "naber", "nasılsın", "hocam", "dostum", "kanka", "slm",
        }
        if tokens & informal_markers:
            return "informal"
        if len(tokens) < 4 and "?" not in user_message:
            return "informal"
        return "formal"

    def _load_character(self) -> Dict[str, Any]:
        """Load character.json if available."""
        char_file = self.dirs.vault_dir / "character.json"
        if char_file.exists():
            try:
                with open(char_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"name": self.target_name, "bio": [f"Digital twin of {self.target_name}"], "style": {}}

    def _load_mindset(self) -> str:
        """Load mindset summary markdown."""
        mindset_file = self.dirs.vault_dir / "mindset_summary.md"
        if mindset_file.exists():
            try:
                return mindset_file.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""
