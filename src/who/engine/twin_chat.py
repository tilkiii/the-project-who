"""
Interactive digital twin chat engine.
Simulates real-time dialogue using persona lore, active behavioral guardrails,
CAPS situational context rules, Two-Stage Stylometric JSD Referee Gate, and memory retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from who.config import call_llm
from who.engine.evaluator import ProfileEvaluator
from who.learn.feature_extractor import CAPS_CONTEXT_MARKERS
from who.utils.logger import console, get_logger
from who.utils.text_cleaner import is_boilerplate_or_junk
from who.vault.vector_embedder import VectorEmbedder

logger = get_logger("who.engine.twin_chat")


class TwinChat:
    """Simulates real-time interactive dialogue with the digital twin with dynamic tone modes and Stylometric Referee Gate."""

    def __init__(self, target_name: str, target_dirs: Any, mode: str = "autodetect"):
        self.target_name = target_name
        self.dirs = target_dirs
        self.mode = mode.lower()  # 'autodetect', 'formal', 'informal'
        self.history: List[Dict[str, str]] = []
        self.last_turn_diagnostics: Dict[str, Any] = {}
        self.vector_store = VectorEmbedder(target_name, chroma_dir=self.dirs.chroma_dir)
        self.evaluator = ProfileEvaluator(target_name, target_dirs)

        # Load character config, mindset & operational boundary rules
        self.character = self._load_character()
        self.mindset_summary = self._load_mindset()
        self.operational_rules = self._load_operational_rules()

    INFORMAL_STYLE_MARKERS = (
        "küçük harf", "samimi", "gündelik", "casual", "informal",
        "kısa mesaj", "düşük noktalama", "sıfır ünlem", "colloquial",
    )
    FORMAL_QUERY_TERMS = {
        "teorem", "akademik", "metodoloji", "makale", "hipotez",
        "theoretical", "institutional", "equilibrium",
    }

    def _detect_caps_context(self, user_message: str) -> str:
        """Identify active Mischel CAPS situational context from user message."""
        words = set(re.findall(r"\b\w+\b", user_message.lower()))
        scores = {
            ctx: sum(1 for w in words if w in markers)
            for ctx, markers in CAPS_CONTEXT_MARKERS.items()
        }
        if any(scores.values()):
            return max(scores, key=scores.get)
        return "INTERPERSONAL_DIALOGUE"

    def _is_informal_persona(self) -> bool:
        """Detect whether character.json describes a casual/informal messaging voice."""
        style = self.character.get("style", {})
        tokens: List[str] = []
        if isinstance(style, dict):
            for key in ("all", "chat"):
                vals = style.get(key, [])
                if isinstance(vals, list):
                    tokens.extend(str(v).lower() for v in vals)
        adjectives = self.character.get("adjectives", [])
        if isinstance(adjectives, list):
            tokens.extend(str(a).lower() for a in adjectives)
        joined = " ".join(tokens)
        return any(marker in joined for marker in self.INFORMAL_STYLE_MARKERS)

    def _format_dialogue_examples(self, limit: int = 4) -> str:
        """Format character.json messageExamples into readable sample exchanges."""
        examples = self.character.get("messageExamples", [])
        if not isinstance(examples, list):
            return ""
        formatted_blocks: List[str] = []
        for thread in examples[:limit]:
            if not isinstance(thread, list):
                continue
            lines: List[str] = []
            for turn in thread:
                if not isinstance(turn, dict):
                    continue
                content = turn.get("content", {})
                text = content.get("text", "").strip() if isinstance(content, dict) else str(content).strip()
                if not text:
                    continue
                speaker = str(turn.get("user", "")).strip()
                label = "Muhatap" if not speaker or speaker == "{{user1}}" or speaker != self.target_name else self.target_name
                lines.append(f"{label}: {text}")
            if lines:
                formatted_blocks.append("\n".join(lines))
        return "\n---\n".join(formatted_blocks)

    def _format_post_examples(self, limit: int = 5) -> str:
        """Format character.json postExamples as verbatim style samples."""
        posts = self.character.get("postExamples", [])
        if not isinstance(posts, list):
            return ""
        items = [f'- "{str(p).strip()}"' for p in posts[:limit] if str(p).strip()]
        return "\n".join(items)

    def build_system_prompt(
        self,
        user_message: str,
        active_tone: Optional[str] = None,
        active_caps_context: Optional[str] = None,
    ) -> str:
        """Assemble a persona-grounded system prompt dynamically from character.json and operational rules."""
        resolved_tone = active_tone or self._resolve_tone(user_message)
        resolved_caps = active_caps_context or self._detect_caps_context(user_message)

        style = self.character.get("style", {})
        style_all = style.get("all", []) if isinstance(style, dict) else []
        style_chat = style.get("chat", []) if isinstance(style, dict) else []
        adjectives = self.character.get("adjectives", [])

        style_parts: List[str] = []
        if style_all:
            style_parts.append(f"Genel Üslup: {', '.join(str(s) for s in style_all)}")
        if style_chat:
            style_parts.append(f"Sohbet Tarzı: {', '.join(str(s) for s in style_chat)}")
        if adjectives:
            style_parts.append(f"Nitelikler: {', '.join(str(a) for a in adjectives)}")
        style_summary = " | ".join(style_parts) or "doğal, tutarlı ve kendi karakter özelliklerine sadık"

        tone_instruction = (
            "TON: Yapılandırılmış ve analitik sohbet üslubu; ancak hedefin kendi stil profilinden ve örnek diyaloglarından kopma."
            if resolved_tone == "formal"
            else "TON: 1'e 1 doğal ve gerçekçi sohbet üslubu. Hedefin gerçek mesajlaşma ritmini, kısalığını, büyük/küçük harf ve noktalama alışkanlığını koru; gereksiz resmiyet veya yapay kurumsal kalıplar kullanma."
        )

        sections = [
            f"Sen {self.target_name}'in bilişsel dijital ikizisin (cognitive digital twin).",
            "Kullanıcı seninle doğrudan canlı bir sohbet yürütüyor.",
            f"AKTİF BAĞLAMSAL DURUM (CAPS): {resolved_caps} | {tone_instruction}",
            "",
            "DİYALOG KURALLARI:",
            (
                f"1. KESİNLİKLE bir istihbarat raporu, analiz metni veya üçüncü şahıs formatında "
                f"('Corpus kanıt analizi', 'Hedef profil şöyledir') konuşma! "
                f"Bizzat {self.target_name} olarak birinci tekil şahıs ('Ben') dilinde yanıt ver."
            ),
            (
                f"2. Stil ve Ses Sadakati ({style_summary}): "
                "Aşağıdaki gerçek ifade örneklerinin ve diyalogların ritmini, büyük/küçük harf kullanımını, "
                "noktalama tercihlerini, cümle uzunluğunu ve duygusal tonunu birebir yakala. "
                "Hedefin kendi örneklerinde yer almayan yapay resmi hitap kalıpları, aşırı ağdalı nezaket "
                "ifadeleri veya robotik kurumsal cümleler kurma."
            ),
            (
                "3. Dil: Kullanıcı hangi dilde yazıyorsa o dilde (Türkçe hitap edildiyse doğal ve akıcı Türkçe ile) "
                "yanıt ver; profilde gözlenen doğal kod değiştirme veya yazım alışkanlıklarını koru."
            ),
        ]

        bio_items = self.character.get("bio", [])
        lore_items = self.character.get("lore", [])[:5]
        post_block = self._format_post_examples()
        dialogue_block = self._format_dialogue_examples()

        profile_lines: List[str] = []
        if bio_items:
            profile_lines.append(f"- Biyografi: {' '.join(str(b) for b in bio_items)}")
        if lore_items:
            profile_lines.append("- Arka Plan & Kimlik Çapaları:\n" + "\n".join(f"  * {l}" for l in lore_items))
        if post_block:
            profile_lines.append(f"- Gerçek İfade Örnekleri (postExamples):\n{post_block}")
        if dialogue_block:
            profile_lines.append(f"- Örnek Diyaloglar (messageExamples):\n{dialogue_block}")

        if profile_lines:
            sections.append("4. Kimlik ve Gerçek Ses Profili (character.json):\n" + "\n".join(profile_lines))

        if self.operational_rules:
            rules_text = "\n".join(f"- {r}" for r in self.operational_rules[:4])
            sections.append(
                "5. Davranışsal Karar Sınırları (Üslup, hitap ve kelime seçiminde hedefin kendi character.json "
                f"stili ve gerçek ses örnekleri jenerik kurumsal ifadelere göre önceliklidir):\n{rules_text}"
            )

        return "\n".join(sections)

    def respond(self, user_message: str) -> str:
        """
        Generate twin response conforming to character lore, epistemic rules, CAPS context,
        and Two-Stage Stylometric JSD Referee Gate.
        """
        # 1. Determine active tone & CAPS situational context
        active_tone = self._resolve_tone(user_message)
        active_caps_context = self._detect_caps_context(user_message)

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

        # 3. Assemble Dynamic Persona-Grounded System Prompt
        system_prompt = self.build_system_prompt(
            user_message,
            active_tone=active_tone,
            active_caps_context=active_caps_context,
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

        # Stage 1: Candidate Persona Generation
        response_text = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.4)

        # Clean any accidental offline note leak if offline was used
        if "> [!NOTE]" in response_text and "offline modda" in response_text:
            lines = [
                l for l in response_text.splitlines()
                if not l.startswith(">") and "offline modda" not in l and "Corpus Kanıt" not in l
            ]
            response_text = "\n".join(lines).strip()

        # Stage 2: Stylometric Fingerprint & Epistemic Referee Gate
        referee_audit = self.evaluator.verify_stylometric_and_epistemic_fidelity(response_text)
        if referee_audit.get("boundary_violations"):
            cleaned_lines = [
                line for line in response_text.splitlines()
                if "corpus kanıt analizi" not in line.lower() and "hedef profil" not in line.lower()
            ]
            response_text = "\n".join(cleaned_lines).strip() or response_text
            referee_audit = self.evaluator.verify_stylometric_and_epistemic_fidelity(response_text)

        self.last_turn_diagnostics = {
            "active_tone": active_tone,
            "active_caps_context": active_caps_context,
            "stylometric_jsd": referee_audit.get("stylometric_jsd", 0.05),
            "stylometric_fidelity_score": referee_audit.get("stylometric_fidelity_score", 0.85),
            "referee_gate_passed": referee_audit.get("referee_gate_passed", True),
            "evidence_ledger": referee_audit.get("evidence_ledger", []),
        }

        # Save to interaction memory
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "twin", "content": response_text})

        return response_text

    def _resolve_tone(self, user_message: str) -> str:
        """Resolve tone mode if set to autodetect."""
        if self.mode in ("formal", "informal"):
            return self.mode

        msg_lower = user_message.lower()
        msg_tokens = set(re.findall(r"\b\w+\b", msg_lower))
        if self._is_informal_persona() and not (msg_tokens & self.FORMAL_QUERY_TERMS):
            return "informal"

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
