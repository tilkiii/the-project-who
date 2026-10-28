"""
Knowledge graph builder for who??.
Extracts LightRAG entities, concepts, and relationships backed by primary source quotes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger("who.vault.builder")


class GraphBuilder:
    """Builds LightRAG-inspired entity-concept relationship graphs with quote backing."""

    def __init__(self, target_name: str):
        self.target_name = target_name

    def extract_graph_elements(self, text: str, source_id: str = "corpus") -> Dict[str, Any]:
        """
        Extract entities, concepts, and relationships from text.
        Every relationship is tied to a primary source quote.
        """
        cleaned = clean_text(text)
        if len(cleaned) < 50:
            return {"entities": [], "relationships": []}

        # 1. Attempt LLM-based extraction
        prompt = (
            f"Extract key entities, concepts, and relationships for target '{self.target_name}' from the following text.\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "entities": [{"name": "string", "type": "Person|Organization|Concept|Methodology|Principle", "description": "string"}],\n'
            '  "relationships": [{"source": "string", "target": "string", "relation": "ADVOCATES|CRITIQUES|DEVELOPS|REJECTS|COLLABORATES_WITH|DEPENDS_ON", "weight": 0.0-1.0, "quote": "exact quote"}]\n'
            "}\n\n"
            f"TEXT TO ANALYZE:\n{cleaned[:2500]}"
        )

        try:
            response = call_llm(prompt, temperature=0.1)
            # Find json block
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])
                if entities or relationships:
                    # Enrich with source_id
                    for r in relationships:
                        r["source_doc"] = source_id
                        if "quote" not in r or not r["quote"]:
                            r["quote"] = cleaned[:100] + "..."
                    return {"entities": entities, "relationships": relationships}
        except Exception as e:
            logger.debug(f"LLM graph extraction failed ({e}), using heuristic rule extraction.")

        # 2. Rule-based Heuristic Fallback
        return self._heuristic_graph_extraction(cleaned, source_id)

    def extract_from_corpus(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate graph elements across multiple documents."""
        all_entities: Dict[str, Dict[str, Any]] = {}
        all_relationships: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            source_id = doc.get("source_url") or doc.get("file_name") or f"doc_{idx}"
            content = doc.get("content", "")
            elements = self.extract_graph_elements(content, source_id=source_id)

            for ent in elements.get("entities", []):
                name = ent.get("name", "").strip()
                if name and name not in all_entities:
                    all_entities[name] = ent

            all_relationships.extend(elements.get("relationships", []))

        # Always ensure target is root entity
        if self.target_name not in all_entities:
            all_entities[self.target_name] = {
                "name": self.target_name,
                "type": "Person",
                "description": f"Primary target individual: {self.target_name}",
            }

        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships,
        }

    def _heuristic_graph_extraction(self, text: str, source_id: str) -> Dict[str, Any]:
        """Extract conceptual patterns using linguistic heuristics."""
        entities = []
        relationships = []

        # Find capitalized multi-word phrases or key technical nouns
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)?\b", text)
        filtered = [c for c in set(candidates) if len(c) > 3 and c != self.target_name][:6]

        for cand in filtered:
            entities.append({
                "name": cand,
                "type": "Concept",
                "description": f"Key concept observed in {source_id}",
            })

            # Check for relations
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if cand.lower() in s.lower()]
            quote = sentences[0] if sentences else text[:80] + "..."

            rel_type = "ADVOCATES"
            s_lower = quote.lower()
            if any(w in s_lower for w in ["reject", "critique", "flaw", "against", "oppose"]):
                rel_type = "CRITIQUES"
            elif any(w in s_lower for w in ["build", "develop", "create", "author"]):
                rel_type = "DEVELOPS"
            elif any(w in s_lower for w in ["partner", "collaborate", "team"]):
                rel_type = "COLLABORATES_WITH"

            relationships.append({
                "source": self.target_name,
                "target": cand,
                "relation": rel_type,
                "weight": 0.8,
                "quote": quote,
                "source_doc": source_id,
            })

        return {"entities": entities, "relationships": relationships}
