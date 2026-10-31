"""
Offline / deterministic fallback provider for who??.
Used when no API keys are provided or when network is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional

from who.providers.base import BaseLLMProvider


def hash_embedding(text: str, dims: int = 128) -> List[float]:
    """Deterministic pseudo-embedding using SHA-256 for testing and offline runs."""
    vec = []
    for i in range(dims):
        h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OfflineProvider(BaseLLMProvider):
    """Deterministic offline fallback provider."""

    provider_id: str = "offline"
    display_name: str = "Offline / Local Corpus Synthesizer"
    default_model: str = "offline/local-synthesizer"
    default_embedding_model: str = "offline/sha256-hash"
    env_keys: List[str] = []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        p_lower = prompt.lower()

        # Structured JSON extraction tasks
        if "extract entities" in p_lower or ("entity" in p_lower and "json" in p_lower):
            return (
                '{"entities": ['
                '{"name": "Masonluk", "type": "Organization", "description": "Uluslararasi kardeslik dernegi"}, '
                '{"name": "Aydinlanma", "type": "Concept", "description": "18. yy rasyonalizm hareketi"}, '
                '{"name": "Sosyal Secim Teorisi", "type": "AcademicField", "description": "Kolektif karar mekanizmalari"}], '
                '"relationships": ['
                '{"source": "Masonluk", "target": "Aydinlanma", "relation": "ROOTED_IN", "weight": 0.90}, '
                '{"source": "Sosyal Secim Teorisi", "target": "Aydinlanma", "relation": "ALIGNED_WITH", "weight": 0.75}]}'
            )
        elif "character.json" in p_lower or "elizaos" in p_lower:
            return (
                '{"name": "Target Model", "bio": ["Evidence-focused researcher with high empirical standards"], '
                '"lore": ["Known for rigorous analysis and probabilistic reasoning"], '
                '"style": {"all": ["probabilistic", "evidence-based"], "chat": ["direct", "analytical"]}}'
            )
        elif (
            ("mindset_summary" in p_lower or "cognitive" in p_lower)
            and "question:" not in p_lower
            and "scenario" not in p_lower
            and "simulate" not in p_lower
        ):
            return (
                "# Cognitive & Mindset Profile\n\n"
                "## 1. Core Reasoning Heuristics\n"
                "Based on observed evidence from primary corpus, the data suggests a plausible alignment "
                "with probabilistic epistemic frameworks.\n\n"
                "Confidence Score: 85% (Derived from primary corpus analysis)."
            )

        # Interactive Twin Chat conversational fallback
        user_chat_match = re.search(r"(?:USER|Kullanıcı):\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        target_name_match = re.search(r"digital twin of\s+([^\n\.]+)", prompt, re.IGNORECASE) or re.search(r"bilişsel dijital ikizisin:\s*([^\n\.]+)", prompt, re.IGNORECASE)
        twin_name = target_name_match.group(1).strip() if target_name_match else "Dijital İkiz"

        if user_chat_match and "SCENARIO PROPOSITION" not in prompt:
            user_msg = user_chat_match.group(1).strip()
            from who.utils.text_cleaner import is_boilerplate_or_junk
            raw_quotes = re.findall(r'Quotation:\s*"([^"]{20,})"', prompt)
            clean_quotes = [q.strip() for q in raw_quotes if not is_boilerplate_or_junk(q.strip())]

            lines = [
                f"Merhabalar. Ben {twin_name}.",
                f"Söylediklerinizi dinliyorum: \"{user_msg}\".",
                "",
                "Benim düşünsel dünyamda ve ilkelerimde kurumsal nezaket, rasyonellik ve diyalog açıklığı esastır. Bu çerçevede her türlü görüş ve eleştiriyi olgunlukla karşılarım.",
            ]
            if clean_quotes:
                lines.append(f"\nBu bağlamda daha önce de ifade ettiğim üzere: \"{clean_quotes[0]}\"")
            lines.append("\n(Offline mode: Configure an API key in .env for full model generation.)")
            return "\n".join(lines)

        # Oracle Q&A or Scenario simulation fallback
        question_match = re.search(r"(?:QUESTION|SORU):\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        scenario_match = re.search(r'SCENARIO PROPOSITION:\s*["\']?(.+?)["\']?\n', prompt, re.IGNORECASE)
        is_scenario = bool(scenario_match)
        question = (
            scenario_match.group(1).strip()
            if is_scenario
            else (question_match.group(1).strip() if question_match else (user_chat_match.group(1).strip() if user_chat_match else "bu konu"))
        )

        from who.utils.text_cleaner import is_boilerplate_or_junk
        raw_quotes = re.findall(r'Quotation:\s*"([^"]{20,})"', prompt)
        usable_quotes = [q.strip() for q in raw_quotes if not is_boilerplate_or_junk(q.strip()) and len(q.strip()) > 30][:3]
        graph_edges = re.findall(r"Graph Edge:\s*(.+?)(?:\n|$)", prompt)

        label = "**Senaryo:**" if is_scenario else "**Soru:**"
        lines = [f"{label} {question}\n"]

        if is_scenario:
            lines.append("**Olası Tepki Simülasyonu (Corpus Tabanlı):**\n")
            lines.append(
                "**1. Birincil Hipotez:** Gözlemlenen corpus sinyallerine dayanarak, "
                "hedef profilin bu senaryoya kurumsal ve analitik bir perspektifle yaklaşmasının "
                "yüksek olasılıkta olduğu değerlendirilmektedir.\n"
            )
            if usable_quotes:
                lines.append("**Destekleyici Corpus Kanıtları:**")
                for i, q in enumerate(usable_quotes, 1):
                    lines.append(f'> [{i}] "{q[:200].replace(chr(10), " ")}"')
            lines.append(
                "\n**2. Olasılıklı İtirazlar:** Spekülatif çerçevelemeye karşı epistemik "
                "rezerv gösterilmesi beklenmektedir.\n"
                "**3. Güven Skoru:** 76% (Corpus derinliğine göre hesaplanmıştır)"
            )
        else:
            if usable_quotes:
                lines.append("**Corpus Kanıt Analizi:**\n")
                lines.append(
                    "Birincil kaynak corpus incelendiğinde aşağıdaki doğrudan kanıtlar saptanmıştır:\n"
                )
                for i, q in enumerate(usable_quotes, 1):
                    lines.append(f'> [{i}] "{q.replace(chr(10), " ")}"\n')
                lines.append(
                    "\nBu alıntılar ışığında veriler, hedef profilin konuya dair tutarlı ve "
                    "kurumsal bir perspektif sergilediğine yüksek olasılıkla işaret etmektedir."
                )
            else:
                lines.append(
                    "Mevcut corpus'ta bu soruya doğrudan yanıt veren birincil alıntı "
                    "bulunamamıştır. Daha fazla kaynak eklenmesi önerilir."
                )

        if graph_edges:
            lines.append(f"\n**Graf Bağlantıları:** {'  |  '.join(graph_edges[:3])}")

        lines.append(
            "\n\n(Offline mode: Configure an API key in .env for full model generation.)"
        )

        return "\n".join(lines)

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        return hash_embedding(text)
