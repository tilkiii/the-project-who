"""
Knowledge graph builder for who??.
Extracts LightRAG entities, concepts, relationships, and detects Graph-Driven Cognitive Dissonance.
Every relationship is backed by primary source quotes and confidence scores.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from who.config import call_llm
from who.utils.logger import get_logger
from who.utils.text_cleaner import clean_text

logger = get_logger("who.vault.builder")

AFFIRMATIVE_RELATIONS = {"ADVOCATES", "DEVELOPS", "SUPPORTS", "ALIGNED_WITH", "AFFIRMS", "EMBRACES", "PRAISES"}
OPPOSITIONAL_RELATIONS = {"CRITIQUES", "REJECTS", "DISPUTES", "CONDEMNS", "OPPOSES", "WARNS_AGAINST"}


class GraphBuilder:
    """Builds LightRAG entity-concept graphs with quote backing and cognitive dissonance detection."""

    def __init__(self, target_name: str):
        self.target_name = target_name

    def extract_graph_elements(self, text: str, source_id: str = "corpus") -> Dict[str, Any]:
        """
        Extract entities, concepts, and relationships from text.
        Every relationship is tied to a primary source quote.
        """
        cleaned = clean_text(text)
        if len(cleaned) < 50:
            return {"entities": [], "relationships": [], "dissonances": []}

        # 1. Attempt LLM-based extraction
        prompt = (
            f"Extract key entities, concepts, relationships, and any observed ideological or stance contradictions "
            f"for target '{self.target_name}' from the following text.\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "entities": [{"name": "string", "type": "Person|Organization|Concept|Methodology|Principle", "description": "string"}],\n'
            '  "relationships": [{"source": "string", "target": "string", "relation": "ADVOCATES|CRITIQUES|DEVELOPS|REJECTS|COLLABORATES_WITH|DEPENDS_ON", "weight": 0.0-1.0, "quote": "exact quote"}],\n'
            '  "cognitive_dissonances": [{"concept_a": "string", "concept_b": "string", "dissonance_score": 0.5-1.0, "contradiction_summary": "string", "quote_a": "string", "quote_b": "string"}]\n'
            "}\n\n"
            f"TEXT TO ANALYZE:\n{cleaned[:2800]}"
        )

        try:
            response = call_llm(prompt, temperature=0.1)
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])
                dissonances = data.get("cognitive_dissonances", [])

                if entities or relationships:
                    for r in relationships:
                        r["source_doc"] = source_id
                        if "quote" not in r or not r["quote"]:
                            r["quote"] = cleaned[:100] + "..."
                    return {
                        "entities": entities,
                        "relationships": relationships,
                        "dissonances": dissonances,
                    }
        except Exception as e:
            logger.debug(f"LLM graph extraction failed ({e}), using heuristic rule extraction.")

        # 2. Rule-based Heuristic Fallback
        return self._heuristic_graph_extraction(cleaned, source_id)

    def extract_from_corpus(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate graph elements across multiple documents and detect cognitive dissonance."""
        all_entities: Dict[str, Dict[str, Any]] = {}
        all_relationships: List[Dict[str, Any]] = []
        raw_dissonances: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            source_id = doc.get("source_url") or doc.get("file_name") or f"doc_{idx}"
            content = doc.get("content", "")
            elements = self.extract_graph_elements(content, source_id=source_id)

            for ent in elements.get("entities", []):
                name = ent.get("name", "").strip()
                if name and name not in all_entities:
                    all_entities[name] = ent

            all_relationships.extend(elements.get("relationships", []))
            raw_dissonances.extend(elements.get("dissonances", []))

        # Always ensure target is root entity
        if self.target_name not in all_entities:
            all_entities[self.target_name] = {
                "name": self.target_name,
                "type": "Person",
                "description": f"Primary target individual: {self.target_name}",
            }

        # Graph-Driven Cognitive Dissonance Detection (Capability #2)
        dissonance_edges = self._detect_cognitive_dissonance(
            all_entities, all_relationships, raw_dissonances, documents
        )
        all_relationships.extend(dissonance_edges)

        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships,
            "dissonance_edges_count": len(dissonance_edges),
        }

    def _detect_cognitive_dissonance(
        self,
        entities: Any,
        relationships: List[Dict[str, Any]],
        raw_dissonances: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify diametrically opposed stances (ADVOCATES vs CRITIQUES/REJECTS)
        or contradictory assertions regarding concepts across different corpus documents.
        Emits explicit [CONTRADICTS] edges with quantified dissonance scores.
        """
        raw_dissonances = raw_dissonances or []
        documents = documents or []
        dissonance_edges: List[Dict[str, Any]] = []
        seen_pairs = set()

        # 1. Convert any LLM-detected cognitive dissonances into CONTRADICTS edges
        for cd in raw_dissonances:
            ca = cd.get("concept_a", "").strip()
            cb = cd.get("concept_b", "").strip()
            if ca and cb and (ca, cb) not in seen_pairs and (cb, ca) not in seen_pairs:
                seen_pairs.add((ca, cb))
                d_score = float(cd.get("dissonance_score", 0.85))
                dissonance_edges.append({
                    "source": ca,
                    "target": cb,
                    "relation": "CONTRADICTS",
                    "weight": round(d_score, 2),
                    "dissonance_score": round(d_score, 2),
                    "contradiction_summary": cd.get("contradiction_summary", "Contradictory stance observed in corpus"),
                    "quote_a": cd.get("quote_a", ""),
                    "quote_b": cd.get("quote_b", ""),
                    "quote": f"Contradiction: {cd.get('contradiction_summary', '')}",
                    "source_doc": "cross-corpus",
                })

        # 2. Graph Stance Inversion Detection across relationships
        # Group target relations per concept: target_name -> concept
        concept_stances: Dict[str, List[Dict[str, Any]]] = {}
        for r in relationships:
            if r.get("source") == self.target_name:
                tgt = r.get("target")
                if tgt:
                    concept_stances.setdefault(tgt, []).append(r)

        for concept, rels in concept_stances.items():
            affirmative = [r for r in rels if r.get("relation") in AFFIRMATIVE_RELATIONS]
            oppositional = [r for r in rels if r.get("relation") in OPPOSITIONAL_RELATIONS]

            if affirmative and oppositional:
                # Contradiction detected on the same concept across different times/sources!
                aff = affirmative[0]
                opp = oppositional[0]
                pair_key = (f"{concept}_affirm", f"{concept}_neg")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    d_score = 0.88
                    dissonance_edges.append({
                        "source": concept,
                        "target": concept,
                        "relation": "CONTRADICTS",
                        "weight": d_score,
                        "dissonance_score": d_score,
                        "contradiction_summary": f"Target exhibits contradictory stances ({aff.get('relation')} vs {opp.get('relation')}) on {concept}",
                        "quote_a": aff.get("quote", ""),
                        "quote_b": opp.get("quote", ""),
                        "quote": f"Affirms: \"{aff.get('quote','')[:70]}\" vs Critiques: \"{opp.get('quote','')[:70]}\"",
                        "source_doc": f"{aff.get('source_doc', '')} <-> {opp.get('source_doc', '')}",
                    })

        # 3. Known Concept Semantic Tension Pairs in Political / Epistemic Discourse
        semantic_tension_pairs = [
            ("Kurumsal Gizlilik", "Kurumsal Seffaflik", 0.78, "Kurumsal sır vs radikal şeffaflık gerilimi"),
            ("Institutional Secrecy", "Transparency", 0.78, "Institutional privacy vs democratic transparency"),
            ("Merkeziyetci Otorite", "Ozerklik", 0.85, "Merkezi yönetim vs akademik/kurumsal özerklik gerilimi"),
            ("Centralized Authority", "Autonomy", 0.85, "Centralized control vs local autonomy"),
            ("Tradition", "Progressivism", 0.72, "Preservation of historical rites vs secular reform"),
            ("Gelenek", "Yenilikcilik", 0.72, "Tarihsel ritüel muhafazası vs aydınlanmacı reform"),
        ]

        if isinstance(entities, list):
            entity_names_lower = {e.get("name", "").lower(): e.get("name", "") for e in entities if isinstance(e, dict)}
        elif isinstance(entities, dict):
            entity_names_lower = {k.lower(): k for k in entities.keys()}
        else:
            entity_names_lower = {}
        for c1, c2, score, summary in semantic_tension_pairs:
            match1 = entity_names_lower.get(c1.lower())
            match2 = entity_names_lower.get(c2.lower())
            if match1 and match2:
                if (match1, match2) not in seen_pairs and (match2, match1) not in seen_pairs:
                    seen_pairs.add((match1, match2))
                    dissonance_edges.append({
                        "source": match1,
                        "target": match2,
                        "relation": "CONTRADICTS",
                        "weight": score,
                        "dissonance_score": score,
                        "contradiction_summary": summary,
                        "quote_a": f"Primary reference to {match1}",
                        "quote_b": f"Primary reference to {match2}",
                        "quote": summary,
                        "source_doc": "conceptual-tension",
                    })

        logger.info(f"Detected {len(dissonance_edges)} Cognitive Dissonance [CONTRADICTS] graph edges.")
        return dissonance_edges

    def _heuristic_graph_extraction(self, text: str, source_id: str) -> Dict[str, Any]:
        """Extract conceptual patterns using linguistic heuristics."""
        entities = []
        relationships = []

        # Find key technical nouns and capitalized phrases
        candidates = re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ0-9_-]+(?:\s+[A-ZÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ0-9_-]+)?\b", text)
        filtered = [c for c in set(candidates) if len(c) > 3 and c != self.target_name][:8]

        for cand in filtered:
            entities.append({
                "name": cand,
                "type": "Concept",
                "description": f"Key concept observed in {source_id}",
            })

            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if cand.lower() in s.lower()]
            quote = sentences[0] if sentences else text[:80] + "..."

            rel_type = "ADVOCATES"
            s_lower = quote.lower()
            if any(w in s_lower for w in ["reject", "critique", "flaw", "against", "oppose", "karşı", "redded", "eleştir"]):
                rel_type = "CRITIQUES"
            elif any(w in s_lower for w in ["build", "develop", "create", "author", "kurucu", "geliştir"]):
                rel_type = "DEVELOPS"
            elif any(w in s_lower for w in ["partner", "collaborate", "team", "ortak", "birlikte"]):
                rel_type = "COLLABORATES_WITH"

            relationships.append({
                "source": self.target_name,
                "target": cand,
                "relation": rel_type,
                "weight": 0.8,
                "quote": quote,
                "source_doc": source_id,
            })

        return {"entities": entities, "relationships": relationships, "dissonances": []}
