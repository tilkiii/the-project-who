"""
Semantic vector memory and embedding store for who??.
Embeds text chunks into ChromaDB enriched with metadata tags (source_type, visibility, speaker).
Corresponds to Layer 2 (Semantic Embedding Memory).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_embedding, settings
from src.utils.logger import get_logger
from src.utils.text_cleaner import chunk_text

logger = get_logger("who.vault.vector")


class VectorEmbedder:
    """Manages ChromaDB vector collections and semantic similarity search."""

    def __init__(self, target_name: str, chroma_dir: Optional[Path] = None):
        self.target_name = target_name
        self.chroma_dir = chroma_dir
        self.collection = None
        self.local_chunks: List[Dict[str, Any]] = []

        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client or local vector store fallback."""
        if self.chroma_dir:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                safe_col_name = f"who_{self.target_name.lower().replace(' ', '_')}"[:63]
                self.collection = client.get_or_create_collection(
                    name=safe_col_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Initialized ChromaDB collection '{safe_col_name}' at {self.chroma_dir}")
                return
            except Exception as e:
                logger.debug(f"ChromaDB persistent client not available ({e}). Using embedded vector fallback.")

        # Fallback local index file
        self._load_local_vector_index()

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Chunk, embed, and store documents with rich metadata."""
        total_added = 0
        all_chunks: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            source_type = doc.get("source_type", "document")
            speaker = doc.get("speaker", "TARGET_PRIMARY")
            source_id = doc.get("source_url") or doc.get("file_name") or f"doc_{idx}"
            content = doc.get("content", "")

            # Chunk document
            chunks = chunk_text(
                content,
                chunk_size=800,
                chunk_overlap=120,
                metadata={
                    "source_id": str(source_id),
                    "source_type": str(source_type),
                    "speaker": str(speaker),
                    "visibility": "public",
                    "target_name": self.target_name,
                },
            )

            for c_idx, c in enumerate(chunks):
                chunk_id = f"{source_id}_chunk_{c_idx}"
                embedding = get_embedding(c["text"])
                chunk_record = {
                    "id": chunk_id,
                    "text": c["text"],
                    "embedding": embedding,
                    "metadata": {
                        "source_id": str(source_id),
                        "source_type": str(source_type),
                        "speaker": str(speaker),
                        "visibility": "public",
                        "target_name": self.target_name,
                    },
                }
                all_chunks.append(chunk_record)

        # Store in Chroma if active
        if self.collection is not None and all_chunks:
            try:
                self.collection.upsert(
                    ids=[c["id"] for c in all_chunks],
                    documents=[c["text"] for c in all_chunks],
                    embeddings=[c["embedding"] for c in all_chunks],
                    metadatas=[c["metadata"] for c in all_chunks],
                )
                logger.info(f"Upserted {len(all_chunks)} chunks to ChromaDB.")
            except Exception as e:
                logger.error(f"ChromaDB upsert error: {e}")

        # Store in local fallback list and persist
        self.local_chunks.extend(all_chunks)
        self._persist_local_vector_index()
        return len(all_chunks)

    def search_similar(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Find the top-k most relevant text chunks for a query."""
        if not query.strip():
            return []

        query_emb = get_embedding(query)

        # 1. Query ChromaDB if active
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                )
                items = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0] if "distances" in results else [0.2] * len(docs)
                for d, m, dist in zip(docs, metas, distances):
                    score = round(1.0 - (dist if dist is not None else 0.2), 3)
                    items.append({
                        "text": d,
                        "metadata": m,
                        "similarity_score": max(0.0, min(1.0, score)),
                    })
                if items:
                    return items
            except Exception as e:
                logger.debug(f"ChromaDB search query failed ({e}). Running local cosine search.")

        # 2. Local cosine similarity fallback
        scored = []
        for c in self.local_chunks:
            emb = c.get("embedding", [])
            sim = self._cosine_similarity(query_emb, emb)
            scored.append({
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}),
                "similarity_score": round(sim, 3),
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1)) or 1e-9
        norm2 = math.sqrt(sum(b * b for b in v2)) or 1e-9
        return dot / (norm1 * norm2)

    def _persist_local_vector_index(self) -> None:
        """Save local vector index to json in chroma directory."""
        if not self.chroma_dir:
            return
        idx_file = self.chroma_dir / "local_vector_index.json"
        try:
            with open(idx_file, "w", encoding="utf-8") as f:
                json.dump(self.local_chunks, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not persist local vector index: {e}")

    def _load_local_vector_index(self) -> None:
        """Load local vector index if present."""
        if not self.chroma_dir:
            return
        idx_file = self.chroma_dir / "local_vector_index.json"
        if idx_file.exists():
            try:
                with open(idx_file, "r", encoding="utf-8") as f:
                    self.local_chunks = json.load(f)
            except Exception as e:
                logger.debug(f"Could not read local vector index: {e}")
