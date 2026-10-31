"""
Oracle Q&A engine for who??.
Answers complex queries backed by Neo4j graph relationships, vector quotes, and probabilistic confidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from who.config import call_llm, settings
from who.utils.logger import get_logger
from who.vault.graph_store import GraphStore
from who.vault.vector_embedder import VectorEmbedder

logger = get_logger("who.engine.oracle")


class Oracle:
    """Evidence-backed analytical Q&A engine."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs
        self.vector_store = VectorEmbedder(target_name, chroma_dir=self.dirs.chroma_dir)
        self.graph_store = GraphStore(target_name, vault_dir=self.dirs.vault_dir)

    def ask(self, query: str) -> Dict[str, Any]:
        """Answer query about the target with strict evidence citations and confidence scoring."""
        logger.info(f"Oracle processing query for {self.target_name}: '{query}'")

        # 1. Retrieve vector chunks
        vector_results = self.vector_store.search_similar(query, top_k=4)
        quotes = [v["text"] for v in vector_results if v.get("text")]
        sources = list(set(v.get("metadata", {}).get("source_id", "corpus") for v in vector_results))

        # 2. Retrieve graph relationships matching query terms
        graph_rels = []
        for rel in self.graph_store.get_all_relationships():
            if any(term in rel.get("target", "").lower() or term in rel.get("source", "").lower() for term in query.lower().split()):
                graph_rels.append(f"{rel.get('source')} -[{rel.get('relation')}]-> {rel.get('target')}")

        # 3. Read mindset summary if exists
        mindset_path = self.dirs.vault_dir / "mindset_summary.md"
        mindset_ctx = mindset_path.read_text(encoding="utf-8") if mindset_path.exists() else ""

        # 4. Formulate Prompt
        context_block = "\n".join(f"- Quotation: \"{q}\"" for q in quotes[:3]) or "- No direct quote available."
        graph_block = "\n".join(f"- Graph Edge: {r}" for r in graph_rels[:4]) or "- No direct graph link found."

        system_prompt = (
            f"You are the Oracle OSINT Evidence Engine for target: {self.target_name}.\n"
            "STRICT SCIENTIFIC EPISTEMOLOGY:\n"
            "- Ground all statements in observed evidence.\n"
            "- Never make absolute or deterministic claims about hidden intentions.\n"
            "- Frame your answer using: 'Based on observed evidence from [Sources]...', 'The data suggests a plausible alignment with...'\n"
            "- Compute and explicitly output an estimated Confidence Score: X% based on quote depth.\n"
        )

        user_prompt = (
            f"QUESTION: {query}\n\n"
            f"CORPUS QUOTES:\n{context_block}\n\n"
            f"KNOWLEDGE GRAPH PATHS:\n{graph_block}\n\n"
            f"MINDSET EXCERPT:\n{mindset_ctx[:600]}\n\n"
            "Synthesize an evidence-backed probabilistic answer with citations."
        )

        raw_answer = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

        # Calculate quantified confidence score
        confidence_score = min(0.95, 0.55 + (len(quotes) * 0.1) + (len(graph_rels) * 0.05))

        return {
            "target_name": self.target_name,
            "query": query,
            "answer": raw_answer,
            "confidence_score": round(confidence_score, 2),
            "primary_quotes": quotes[:3],
            "sources": sources,
            "graph_paths": graph_rels,
        }
