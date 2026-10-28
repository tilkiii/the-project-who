"""
Graph Data Science (GDS) and graph analytics for who??.
Computes centrality, PageRank, community detection, shortest paths, and influence metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

from src.config import settings
from src.utils.logger import get_logger
from src.vault.graph_store import GraphStore

logger = get_logger("who.vault.analytics")


class GraphAnalytics:
    """Executes Neo4j GDS and NetworkX graph analytics algorithms."""

    def __init__(self, graph_store: GraphStore, vault_dir: Optional[Path] = None):
        self.store = graph_store
        self.vault_dir = vault_dir

    def run_analytics(self) -> Dict[str, Any]:
        """Compute full graph metrics suite (PageRank, Centrality, Communities, Strengths)."""
        logger.info(f"Computing graph metrics and GDS analytics for target '{self.store.target_name}'.")

        g = self.store.nx_graph
        if len(g.nodes) == 0:
            logger.warning("Graph contains no nodes to analyze.")
            return self._empty_metrics()

        # 1. Degree Centrality
        degree_cent = nx.degree_centrality(g)

        # 2. PageRank (Influence Score)
        try:
            pagerank_scores = nx.pagerank(g, alpha=0.85, max_iter=200)
        except Exception:
            pagerank_scores = {n: round(1.0 / len(g.nodes), 4) for n in g.nodes}

        # 3. Betweenness Centrality (Brokerage / Bridge concepts)
        try:
            betweenness = nx.betweenness_centrality(g)
        except Exception:
            betweenness = {n: 0.0 for n in g.nodes}

        # 4. Closeness Centrality
        try:
            closeness = nx.closeness_centrality(g)
        except Exception:
            closeness = {n: 0.0 for n in g.nodes}

        # 5. Community Structure / Concept Clustering
        undirected_g = g.to_undirected()
        communities: List[List[str]] = []
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comm_sets = list(greedy_modularity_communities(undirected_g))
            communities = [list(c) for c in comm_sets]
        except Exception:
            communities = [list(c) for c in nx.connected_components(undirected_g)]

        # 6. Relationship Strengths
        relationship_strengths = []
        for u, v, data in g.edges(data=True):
            relationship_strengths.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATES_TO"),
                "weight": float(data.get("weight", 0.8)),
                "quote": data.get("quote", ""),
            })
        relationship_strengths.sort(key=lambda x: x["weight"], reverse=True)

        # Format node metrics summary
        node_metrics = []
        for node in g.nodes:
            node_metrics.append({
                "node": node,
                "type": g.nodes[node].get("type", "Concept"),
                "pagerank": round(pagerank_scores.get(node, 0.0), 4),
                "degree_centrality": round(degree_cent.get(node, 0.0), 4),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 4),
                "closeness_centrality": round(closeness.get(node, 0.0), 4),
            })
        node_metrics.sort(key=lambda x: x["pagerank"], reverse=True)

        # Community clusters summary
        clusters_summary = []
        for idx, comm in enumerate(communities):
            clusters_summary.append({
                "cluster_id": idx,
                "size": len(comm),
                "members": comm[:10],
            })

        analytics_payload = {
            "target_name": self.store.target_name,
            "total_nodes": len(g.nodes),
            "total_relationships": len(g.edges),
            "top_influential_concepts": node_metrics[:10],
            "community_clusters": clusters_summary,
            "density": round(nx.density(g), 4),
            "is_connected": nx.is_weakly_connected(g) if len(g.nodes) > 1 else True,
        }

        # Persist artifacts if vault_dir is configured
        if self.vault_dir:
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            with open(self.vault_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
                json.dump(analytics_payload, f, indent=2)

            with open(self.vault_dir / "relationship_strengths.json", "w", encoding="utf-8") as f:
                json.dump(relationship_strengths, f, indent=2)

            logger.info("Saved graph_metrics.json and relationship_strengths.json successfully.")

        return analytics_payload

    def find_shortest_path(self, source_concept: str, target_concept: str) -> List[str]:
        """Find the shortest path between two concepts in the knowledge graph."""
        g = self.store.nx_graph
        if source_concept not in g or target_concept not in g:
            return []
        try:
            return nx.shortest_path(g.to_undirected(), source=source_concept, target=target_concept)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "target_name": self.store.target_name,
            "total_nodes": 0,
            "total_relationships": 0,
            "top_influential_concepts": [],
            "community_clusters": [],
            "density": 0.0,
        }
