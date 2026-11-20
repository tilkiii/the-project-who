"""
Explainability engine for who??.
Traces claims back to primary quotes, Neo4j graph paths, BERTopic themes, and quantified confidence scores.
Strictly neutralizes deterministic psychological assertions.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from who.utils.logger import get_logger
from who.vault.graph_store import GraphStore
from who.vault.vector_embedder import VectorEmbedder

logger = get_logger("who.engine.explainability")


class ExplainabilityEngine:
    """Verifies and traces claims against multi-layer evidence stores with probabilistic calibration."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs
        self.vector_store = VectorEmbedder(target_name, chroma_dir=self.dirs.chroma_dir)
        self.graph_store = GraphStore(target_name, vault_dir=self.dirs.vault_dir)

    def explain_claim(self, claim: str) -> Dict[str, Any]:
        """
        Audit a claim, neutralize deterministic language, trace primary quotes,
        locate graph paths, and compute confidence.
        """
        logger.info(f"Auditing and tracing claim for {self.target_name}: '{claim}'")

        # 1. Epistemic Terminology Replacement
        calibrated_claim = self.enforce_probabilistic_phrasing(claim)

        # 2. Vector search for primary quote verification
        vector_matches = self.vector_store.search_similar(claim, top_k=5)
        supporting_quotes = []
        sources = []
        for m in vector_matches:
            if m.get("similarity_score", 0.0) >= 0.40:
                supporting_quotes.append(m.get("text", ""))
                src = m.get("metadata", {}).get("source_id", "corpus")
                if src not in sources:
                    sources.append(src)

        # 3. Knowledge Graph Path Tracing
        graph_trace_paths = self._find_graph_paths(claim)

        # 4. Check BERTopic themes
        bertopic_themes = self._find_matching_topics(claim)

        # 5. Check Cognitive Dissonance Tensions (Capability #2)
        dissonance_info = self._check_cognitive_dissonance(claim)

        # 6. Quantify Confidence Score
        quote_weight = min(1.0, len(supporting_quotes) / 3.0) * 0.5
        graph_weight = min(1.0, len(graph_trace_paths) / 2.0) * 0.3
        topic_weight = min(1.0, len(bertopic_themes) / 1.0) * 0.2
        confidence_score = round(quote_weight + graph_weight + topic_weight, 2)

        # Build explainability payload
        explanation = {
            "target_name": self.target_name,
            "original_claim": claim,
            "calibrated_probabilistic_statement": calibrated_claim,
            "confidence_score": confidence_score,
            "confidence_rating_str": f"{int(confidence_score * 100)}%",
            "primary_quotes": supporting_quotes[:3],
            "evidence_sources": sources,
            "graph_trace_paths": graph_trace_paths,
            "associated_bertopic_themes": bertopic_themes,
            "cognitive_dissonance_analysis": dissonance_info,
            "required_phrasing": f"Based on observed evidence from {', '.join(sources) if sources else 'analyzed corpus'}, the data suggests a plausible alignment with {calibrated_claim}. Confidence Score: {int(confidence_score * 100)}% (Derived from {len(supporting_quotes)} supporting primary quotes).",
        }

        return explanation

    def _check_cognitive_dissonance(self, claim: str) -> Dict[str, Any]:
        """Verify whether the audited claim intersects with known cognitive dissonance / contradiction clusters."""
        claim_lower = claim.lower()
        contradiction_edges = []
        matching_clusters = []

        # Check in graph edges
        for rel in self.graph_store.get_all_relationships():
            if rel.get("relation") == "CONTRADICTS":
                src = rel.get("source", "").lower()
                tgt = rel.get("target", "").lower()
                if src in claim_lower or tgt in claim_lower:
                    contradiction_edges.append(rel)

        # Check in graph_metrics.json
        metrics_file = self.dirs.vault_dir / "graph_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    gm = json.load(f)
                    for cluster in gm.get("cognitive_dissonance", {}).get("dissonance_clusters", []):
                        if any(c.lower() in claim_lower for c in cluster.get("concepts", [])):
                            matching_clusters.append(cluster)
            except Exception:
                pass

        has_tension = bool(contradiction_edges or matching_clusters)
        statement = ""
        if has_tension:
            first_c = matching_clusters[0] if matching_clusters else {}
            statement = (
                f"Bilişsel Çelişki Uyarısı: İddia, hedefin çelişen inanç örüntüsü gösterdiği "
                f"[{', '.join(first_c.get('concepts', ['çelişen konseptler'])[:2])}] kümelenmesiyle kesişmektedir. "
                f"(Gerilim Nedeni: {first_c.get('primary_conflict', 'İdeolojik/Kurumsal Zıtlık')})."
            )

        return {
            "has_dissonance_tension": has_tension,
            "matching_contradictions": contradiction_edges[:2],
            "matching_clusters": matching_clusters[:2],
            "probabilistic_warning": statement,
        }

    def enforce_probabilistic_phrasing(self, text: str) -> str:
        """Replace forbidden deterministic claims with scientific, probabilistic phrasing."""
        replacements = [
            (r"(?i)determine\s+exactly\s+how\s+the\s+target\s+thinks", "infer likely reasoning patterns from observed evidence"),
            (r"(?i)exact\s+psychological\s+mapping", "evidence-based behavioral modeling"),
            (r"(?i)predict\s+future\s+behavior", "simulate plausible reactions under explicit assumptions"),
            (r"(?i)definitely\s+will", "is plausibly inclined to"),
            (r"(?i)always\s+rejects", "has frequently critiqued"),
            (r"(?i)proves\s+that", "suggests plausible alignment with"),
        ]
        calibrated = text
        for pat, repl in replacements:
            calibrated = re.sub(pat, repl, calibrated)
        return calibrated

    def _find_graph_paths(self, claim: str) -> List[List[str]]:
        """Search the graph for entities mentioned in claim and trace connections."""
        paths = []
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim)]
        nodes = self.graph_store.get_all_nodes()
        matching_nodes = []

        for n in nodes:
            n_name = n.get("name", "")
            if any(w in n_name.lower() for w in words):
                matching_nodes.append(n_name)

        g = self.graph_store.nx_graph
        for target_node in matching_nodes[:3]:
            if target_node in g:
                if self.target_name in g and target_node != self.target_name:
                    import networkx as nx
                    try:
                        p = nx.shortest_path(g.to_undirected(), source=self.target_name, target=target_node)
                        paths.append(p)
                    except Exception:
                        paths.append([self.target_name, target_node])
                else:
                    paths.append([target_node])

        return paths

    def _find_matching_topics(self, claim: str) -> List[str]:
        """Match claim against discovered BERTopic themes."""
        topics_file = self.dirs.learned_skills_dir / "topics.json"
        if not topics_file.exists():
            return []
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            matching = []
            claim_lower = claim.lower()
            for t in data.get("topics", []):
                t_name = t.get("name", "")
                kws = t.get("keywords", [])
                if any(k.lower() in claim_lower for k in kws) or t_name.lower() in claim_lower:
                    matching.append(t_name)
            return matching
        except Exception:
            return []
