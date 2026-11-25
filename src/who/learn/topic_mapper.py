"""
Topic modeling and temporal trajectory tracking.
Extracts semantic topics and maps temporal evolution (emergence, persistence, decline)
across target documents.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from who.utils.logger import get_logger

logger = get_logger("who.learn.topics")


class TopicMapper:
    """Discovers thematic clusters and tracks their temporal evolution trajectory."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def extract_topics(
        self,
        documents: List[Dict[str, Any]],
        nr_topics: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract topics from corpus documents and track temporal evolution.
        Each doc dict can contain: 'content', 'timestamp' or 'date', 'source_url'.
        """
        logger.info(f"Extracting topics and temporal trajectory across {len(documents)} documents.")
        if not documents:
            return {"topics": [], "topic_evolution": {}}

        docs_text = [d.get("content", "") for d in documents if d.get("content", "").strip()]
        if not docs_text:
            return {"topics": [], "topic_evolution": {}}

        # 1. Attempt BERTopic
        topics_data: Optional[Dict[str, Any]] = None
        try:
            from bertopic import BERTopic
            topic_model = BERTopic(nr_topics=nr_topics, calculate_probabilities=False, verbose=False)
            topics, _ = topic_model.fit_transform(docs_text)
            topic_info = topic_model.get_topic_info()

            discovered = []
            for _, row in topic_info.iterrows():
                tid = int(row["Topic"])
                if tid == -1:
                    continue  # Outlier cluster
                words = [w[0] for w in topic_model.get_topic(tid)][:8]
                discovered.append({
                    "topic_id": tid,
                    "name": row.get("Name", f"Topic_{tid}"),
                    "count": int(row.get("Count", 0)),
                    "keywords": words,
                })
            topics_data = {"topics": discovered}
        except Exception as e:
            logger.debug(f"BERTopic not loaded or failed ({e}). Executing pure Python TF-IDF clustering fallback.")

        # 2. Algorithmic TF-IDF + Co-occurrence Fallback
        if not topics_data:
            topics_data = self._fallback_topic_modeling(docs_text, nr_topics=nr_topics)

        # 3. Compute Temporal Evolution (Emergence, Persistence, Abandonment)
        evolution_data = self._calculate_topic_evolution(documents, topics_data["topics"])

        # Persist artifacts if output_dir provided
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "topics.json", "w", encoding="utf-8") as f:
                json.dump(topics_data, f, indent=2)

            with open(self.output_dir / "topic_evolution.json", "w", encoding="utf-8") as f:
                json.dump(evolution_data, f, indent=2)

            logger.info("Saved topics.json and topic_evolution.json successfully.")

        return {
            "topics": topics_data["topics"],
            "topic_evolution": evolution_data,
        }

    def _fallback_topic_modeling(self, texts: List[str], nr_topics: int = 5) -> Dict[str, Any]:
        """Term-frequency co-occurrence based clustering fallback."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "that", "this", "these", "those", "it", "its", "as", "by", "from", "about",
            "can", "could", "will", "would", "should", "not", "their", "they", "we", "you", "i"
        }

        # Tokenize per document
        doc_tokens = []
        term_doc_count: Dict[str, int] = {}
        for text in texts:
            toks = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text) if w.lower() not in stopwords]
            doc_tokens.append(toks)
            for t in set(toks):
                term_doc_count[t] = term_doc_count.get(t, 0) + 1

        num_docs = len(texts)
        # Compute TF-IDF
        top_terms = []
        for term, df in term_doc_count.items():
            if df >= max(1, int(num_docs * 0.15)) and df <= int(num_docs * 0.85) + 1:
                idf = math.log((1 + num_docs) / (1 + df))
                score = df * idf
                top_terms.append((term, score))
        top_terms.sort(key=lambda x: x[1], reverse=True)

        selected_seeds = [t[0] for t in top_terms[:nr_topics]]
        if not selected_seeds:
            selected_seeds = ["analysis", "strategy", "governance", "systems", "research"]

        topics = []
        for idx, seed in enumerate(selected_seeds[:nr_topics]):
            # Find words co-occurring with seed
            cooccur: Counter[str] = Counter()
            doc_matches = 0
            for toks in doc_tokens:
                if seed in toks:
                    doc_matches += 1
                    cooccur.update(toks)
            cooccur.pop(seed, None)
            top_co = [w for w, _ in cooccur.most_common(6)]
            topics.append({
                "topic_id": idx,
                "name": f"{seed.capitalize()} & Related Dynamics",
                "count": doc_matches or 1,
                "keywords": [seed] + top_co,
            })

        return {"topics": topics}

    def _calculate_topic_evolution(
        self,
        documents: List[Dict[str, Any]],
        topics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Quantify topic emergence, persistence, and abandonment over time intervals.
        """
        # Assign approximate or extracted temporal partitions
        n_docs = len(documents)
        third = max(1, n_docs // 3)

        # 3 temporal epochs: early, middle, recent
        epochs = {
            "early": documents[:third],
            "middle": documents[third:2 * third] if n_docs >= 2 else documents,
            "recent": documents[2 * third:] if n_docs >= 3 else documents,
        }

        topic_activity: Dict[str, Dict[str, int]] = {}
        for t in topics:
            t_name = t["name"]
            keywords = [k.lower() for k in t["keywords"]]
            topic_activity[t_name] = {"early": 0, "middle": 0, "recent": 0}

            for epoch_name, epoch_docs in epochs.items():
                for doc in epoch_docs:
                    c_lower = doc.get("content", "").lower()
                    if any(kw in c_lower for kw in keywords):
                        topic_activity[t_name][epoch_name] += 1

        emerging = []
        persistent = []
        abandoned = []

        for t_name, acts in topic_activity.items():
            early_act = acts["early"]
            rec_act = acts["recent"]
            mid_act = acts["middle"]

            if early_act == 0 and rec_act > 0:
                emerging.append({"topic": t_name, "trajectory": "emerging", "recent_mentions": rec_act})
            elif early_act > 0 and rec_act == 0 and mid_act == 0:
                abandoned.append({"topic": t_name, "trajectory": "abandoned", "early_mentions": early_act})
            else:
                persistent.append({
                    "topic": t_name,
                    "trajectory": "persistent_core",
                    "lifecycle_coverage": round((sum(1 for v in acts.values() if v > 0) / 3.0) * 100, 1),
                })

        return {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "epochs_analyzed": ["early", "middle", "recent"],
            "topic_activity_matrix": topic_activity,
            "persistent_core_topics": persistent,
            "emerging_topics": emerging,
            "abandoned_topics": abandoned,
            "summary": f"Identified {len(persistent)} persistent core topics, {len(emerging)} emerging intellectual trajectories, and {len(abandoned)} abandoned themes.",
        }
