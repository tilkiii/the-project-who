"""
Interactive digital twin chat engine.
Simulates real-time dialogue using persona lore, active behavioral guardrails,
and memory retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import console, get_logger
from src.utils.text_cleaner import is_boilerplate_or_junk
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

        # Load character config, mindset & operational boundary rules
        self.character = self._load_character()
        self.mindset_summary = self._load_mindset()
        self.operational_rules = self._load_operational_rules()

    def respond(self, user_message: str) -> str:
        """Generate twin response conforming to character lore, epistemic rules, and selected tone."""
        # 1. Determine active tone
        active_tone = self._resolve_tone(user_message)

        # 2. Retrieve relevant context quotes (filter out boilerplate & web junk)
        quotes_ctx = ""
        user_msg_clean = user_message.strip()
        is_greeting_or_chatter = len(user_msg_clean.split()) <= 4 and any(
            w in user_msg_clean.lower() for w in ["selam", "merhaba", "hey", "lan", "naber", "günaydın", "iyi günler", "siz"]
        )

        if not is_greeting_or_chatter:
            context_chunks = self.vector_store.search_similar(user_message, top_k=4)
            valid_quotes = []
            for c in context_chunks:
                text = (c.get("text") or "").strip()
                if text and len(text) > 35 and not is_boilerplate_or_junk(text):
                    valid_quotes.append(text)
            if valid_quotes:
                quotes_ctx = "\n".join(f"- Alıntı: \"{q[:250]}\"" for q in valid_quotes[:2])

        # 3. Assemble System Prompt
        style = self.character.get("style", {})
        all_style = ", ".join(style.get("all", ["rasyonel", "kanıt odaklı", "ölçülü"]))
        
        rules_text = "\n".join(f"- {r}" for r in self.operational_rules[:5]) if self.operational_rules else "- Rasyonel, ölçülü ve kurumsal bütünlüğü koru."

        tone_instruction = (
            "TON: Resmi ve akademik üslup. Analitik sözcükler, yapılandırılmış mantıksal argümanlar ve kavramsal derinlik kullan."
            if active_tone == "formal"
            else "TON: 1'e 1 doğrudan sohbet üslubu. Samimi veya lakayt ifadelere dahi nezaketli, olgun, sakin ve saygın bir dille mukabele et."
        )

        system_prompt = (
            f"Sen {self.target_name}'in bilişsel dijital ikizisin (cognitive digital twin).\n"
            f"Kullanıcı seninle doğrudan canlı bir sohbet yürütüyor.\n\n"
            f"DİYALOG KURALLARI:\n"
            f"1. KESİNLİKLE bir istihbarat raporu, analiz metni veya üçüncü şahıs formatında ('Corpus kanıt analizi', 'Hedef profil şöyledir') konuşma! Sen üçüncü şahıs bir yapay zeka analisti değilsin; bizzat {self.target_name} olarak birinci tekil şahıs ('Ben') dilinde yanıt ver.\n"
            f"2. Hitabın ve Duruşun: Entelektüel, kibar, rasyonel, öz-denetimi yüksek ve stoik. Karşındaki kişi gündelik, argo, lakayt ('lan', 'siz ekledik ya', 'ne haber') konuşsa bile, vakarını ve nezaketini koruyarak muhatabına doğrudan hitap et (Örn: 'Efendim, haklısınız...', 'Şöyle izah edeyim...').\n"
            f"3. Dil: Yanıtını doğrudan ve akıcı Türkçe ile ver.\n"
            f"4. Davranışsal Karar Sınırları ve Kuralların:\n{rules_text}\n"
            f"5. Biyografi & Entelektüel Kimlik: {' '.join(self.character.get('bio', []))}\n"
        )

        # Conversation memory
        conv_memory = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in self.history[-6:])

        user_prompt_parts = []
        if quotes_ctx:
            user_prompt_parts.append(f"Hafızandaki Birincil Kaynak/Düşünce Alıntıları:\n{quotes_ctx}\n")
        if conv_memory:
            user_prompt_parts.append(f"Önceki Sohbet Geçmişi:\n{conv_memory}\n")
        user_prompt_parts.append(f"Kullanıcı: {user_message}\n\n{self.target_name} Yanıtı:")

        user_prompt = "\n".join(user_prompt_parts)

        response_text = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.4)

        # Clean any accidental offline note leak if offline was used
        if "> [!NOTE]" in response_text and "offline modda" in response_text:
            # Keep only the persona dialogue part
            lines = [l for l in response_text.splitlines() if not l.startswith(">") and "offline modda" not in l and "Corpus Kanıt" not in l]
            response_text = "\n".join(lines).strip()

        # Save to interaction memory
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "twin", "content": response_text})

        return response_text

    def _resolve_tone(self, user_message: str) -> str:
        """Resolve tone mode if set to autodetect."""
        if self.mode in ("formal", "informal"):
            return self.mode

        # Autodetect heuristic
        msg_lower = user_message.lower()
        informal_markers = {"hey", "hi", "yo", "sup", "gonna", "wanna", "lol", "dude", "bro", "cool", "lan", "ya", "yok", "naber", "selam"}
        if any(m in msg_lower.split() for m in informal_markers) or len(user_message.split()) < 6:
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

    def _load_operational_rules(self) -> List[str]:
        """Load derived empirical and behavioral rules from generated skill."""
        skill_file = self.dirs.learned_skills_dir / "generated_mapping_skill.json"
        if skill_file.exists():
            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    rules = data.get("generation_rules", [])
                    return [r for r in rules if r.startswith("[") or "Ground all" not in r]
            except Exception:
                pass
        return []

