"""
Graph Data Science (GDS), Complex Network Physics, and Ising Spin-Glass Frustration analytics for who??.
Computes centrality, PageRank, community detection, shortest paths, Spectral Laplacian
Algebraic Connectivity (Fiedler Value), and Ising Hamiltonian Cognitive Dissonance Frustration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from who.config import settings
from who.utils.logger import get_logger
from who.vault.graph_store import GraphStore

logger = get_logger("who.vault.analytics")


class GraphAnalytics:
    """Executes Neo4j GDS, NetworkX graph analytics, and Complex Network Physics (Ising / Spectral) algorithms."""

    def __init__(self, graph_store: GraphStore, vault_dir: Optional[Path] = None):
        self.store = graph_store
        self.vault_dir = vault_dir

    def run_analytics(self) -> Dict[str, Any]:
        """Compute full graph metrics suite (PageRank, Centrality, Communities, Cognitive Dissonance, Ising & Spectral Physics)."""
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

        # 5. Cognitive Dissonance & Ising Spin-Glass Frustration Subgraph Analysis
        dissonance_analysis = self._analyze_cognitive_dissonance(g, betweenness)

        # 6. Spectral Graph Physics (Laplacian Algebraic Connectivity / Fiedler Value)
        undirected_g = g.to_undirected()
        spectral_physics = self._compute_spectral_laplacian_metrics(undirected_g)

        # 7. Community Structure / Concept Clustering (Modularity)
        communities: List[List[str]] = []
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comm_sets = list(greedy_modularity_communities(undirected_g))
            communities = [list(c) for c in comm_sets]
        except Exception:
            communities = [list(c) for c in nx.connected_components(undirected_g)]

        # 8. Relationship Strengths
        relationship_strengths = []
        for u, v, data in g.edges(data=True):
            relationship_strengths.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATES_TO"),
                "weight": float(data.get("weight", 0.8)),
                "dissonance_score": float(data.get("dissonance_score", 0.0)),
                "quote": data.get("quote", ""),
            })
        relationship_strengths.sort(key=lambda x: x["weight"], reverse=True)

        # Format node metrics summary
        node_metrics = []
        for node in g.nodes:
            d_node_score = dissonance_analysis["node_dissonance_scores"].get(node, 0.0)
            node_metrics.append({
                "node": node,
                "type": g.nodes[node].get("type", "Concept"),
                "pagerank": round(pagerank_scores.get(node, 0.0), 4),
                "degree_centrality": round(degree_cent.get(node, 0.0), 4),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 4),
                "closeness_centrality": round(closeness.get(node, 0.0), 4),
                "dissonance_index": round(d_node_score, 3),
                "ising_spin_state": dissonance_analysis["ising_physics"]["spin_configuration"].get(node, 1),
            })
        node_metrics.sort(key=lambda x: x["pagerank"], reverse=True)

        # Community clusters summary
        clusters_summary = []
        for idx, comm in enumerate(communities):
            cluster_dissonance_count = sum(1 for n in comm if n in dissonance_analysis["high_dissonance_nodes"])
            clusters_summary.append({
                "cluster_id": idx,
                "size": len(comm),
                "members": comm[:10],
                "dissonance_friction_nodes": cluster_dissonance_count,
            })

        analytics_payload = {
            "target_name": self.store.target_name,
            "total_nodes": len(g.nodes),
            "total_relationships": len(g.edges),
            "top_influential_concepts": node_metrics[:10],
            "community_clusters": clusters_summary,
            "cognitive_dissonance": {
                "overall_dissonance_index": dissonance_analysis["overall_index"],
                "total_contradiction_edges": dissonance_analysis["total_contradictions"],
                "high_dissonance_concepts": dissonance_analysis["high_dissonance_concepts"][:6],
                "dissonance_clusters": dissonance_analysis["dissonance_clusters"],
                "ising_spin_glass_physics": dissonance_analysis["ising_physics"],
                "probabilistic_statement": dissonance_analysis["probabilistic_statement"],
            },
            "network_physics": {
                "ising_spin_glass": dissonance_analysis["ising_physics"],
                "spectral_laplacian": spectral_physics,
            },
            "density": round(nx.density(g), 4),
            "is_connected": nx.is_weakly_connected(g) if len(g.nodes) > 1 else True,
        }

        if self.vault_dir:
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            with open(self.vault_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
                json.dump(analytics_payload, f, indent=2)

            with open(self.vault_dir / "relationship_strengths.json", "w", encoding="utf-8") as f:
                json.dump(relationship_strengths, f, indent=2)

            logger.info("Saved graph_metrics.json (with Ising Spin-Glass & Spectral Physics) successfully.")

        return analytics_payload

    def _compute_ising_spin_glass_energy(self, g: nx.DiGraph) -> Dict[str, Any]:
        """
        Model the concept graph as a signed Ising Spin-Glass system (Toulouse, 1977; Heider Structural Balance):
        - Hamiltonian: H(s) = - sum_{(i,j)} J_{ij} * s_i * s_j, where s_i in {-1, +1}
        - J_{ij} > 0 for associative/supportive edges (ferromagnetic alignment)
        - J_{ij} < 0 for [CONTRADICTS] edges (antiferromagnetic conflict)
        Uses greedy coordinate descent (zero-temperature Glauber relaxation) to find the ground-state
        spin configuration s* and computes the Structural Frustration Index Phi in [0, 1].
        """
        nodes = list(g.nodes)
        if not nodes:
            return {
                "hamiltonian_ground_energy_H": 0.0,
                "frustrated_bonds_count": 0,
                "structural_frustration_index": 0.0,
                "spin_configuration": {},
            }

        # Build signed coupling dictionary J_{(u, v)}
        couplings: List[Tuple[str, str, float]] = []
        total_abs_coupling = 0.0
        for u, v, data in g.edges(data=True):
            if u == v:
                continue
            is_contra = data.get("relation") == "CONTRADICTS" or float(data.get("dissonance_score", 0.0)) > 0.0
            weight = float(data.get("dissonance_score") or data.get("weight", 0.8))
            j_val = -abs(weight) if is_contra else abs(weight)
            couplings.append((u, v, j_val))
            total_abs_coupling += abs(j_val)

        # Initialize spins s_i = +1
        spins: Dict[str, int] = {n: 1 for n in nodes}

        # Coordinate descent (Glauber zero-T spin relaxation) to minimize H = -sum J_ij s_i s_j
        for _ in range(12):
            flipped_any = False
            for n in nodes:
                local_field = 0.0
                for u, v, j_val in couplings:
                    if u == n:
                        local_field += j_val * spins[v]
                    elif v == n:
                        local_field += j_val * spins[u]
                # Energy contribution of node n is -s_n * local_field
                if local_field < -1e-6 and spins[n] == 1:
                    spins[n] = -1
                    flipped_any = True
                elif local_field > 1e-6 and spins[n] == -1:
                    spins[n] = 1
                    flipped_any = True
            if not flipped_any:
                break

        hamiltonian_h = 0.0
        frustrated_bonds = 0
        frustrated_weight = 0.0
        for u, v, j_val in couplings:
            bond_term = j_val * spins[u] * spins[v]
            hamiltonian_h += -bond_term
            if bond_term < 0:
                frustrated_bonds += 1
                frustrated_weight += abs(j_val)

        frustration_index = round(frustrated_weight / total_abs_coupling, 4) if total_abs_coupling > 0 else 0.0

        return {
            "hamiltonian_ground_energy_H": round(hamiltonian_h, 4),
            "frustrated_bonds_count": frustrated_bonds,
            "total_couplings": len(couplings),
            "structural_frustration_index": frustration_index,
            "spin_configuration": spins,
        }

    def _compute_spectral_laplacian_metrics(self, ug: nx.Graph) -> Dict[str, Any]:
        """
        Compute Spectral Graph Theory metrics on the concept network:
        - Algebraic Connectivity (Fiedler Value lambda_2 of the Graph Laplacian L = D - A)
        - Spectral Radius (lambda_max)
        - Assortativity / Average Clustering Coefficient
        """
        n_nodes = len(ug.nodes)
        if n_nodes < 2:
            return {
                "algebraic_connectivity_lambda2": 0.0,
                "spectral_radius_lambda_max": 0.0,
                "average_clustering_coefficient": 0.0,
                "topology_regime": "SPARSE_SINGLETON",
            }

        try:
            avg_clustering = round(float(nx.average_clustering(ug)), 4)
        except Exception:
            avg_clustering = 0.0

        # Extract largest connected component for Fiedler value
        largest_cc_nodes = max(nx.connected_components(ug), key=len)
        sub_g = ug.subgraph(largest_cc_nodes).copy()

        lambda_2 = 0.0
        lambda_max = 0.0
        if len(sub_g.nodes) >= 2:
            try:
                import numpy as np
                lap_matrix = nx.laplacian_matrix(sub_g).toarray().astype(float)
                eigvals = np.sort(np.real(np.linalg.eigvalsh(lap_matrix)))
                lambda_2 = round(float(max(0.0, eigvals[1])), 4)
                lambda_max = round(float(max(0.0, eigvals[-1])), 4)
            except Exception:
                # Degree-based analytical bound fallback if numpy/scipy not invoked
                degrees = sorted(d for _, d in sub_g.degree())
                lambda_2 = round(float(degrees[0]) * 0.5, 4)
                lambda_max = round(float(degrees[-1]) * 1.5, 4)

        if lambda_2 >= 1.0:
            regime = "COHESIVE_STRONGLY_COUPLED_MANIFOLD"
        elif lambda_2 >= 0.25:
            regime = "MODULAR_MULTI_CLUSTER_TOPOLOGY"
        else:
            regime = "WEAKLY_COUPLED_FRAGMENTED_GRAPH"

        return {
            "algebraic_connectivity_lambda2": lambda_2,
            "spectral_radius_lambda_max": lambda_max,
            "average_clustering_coefficient": avg_clustering,
            "largest_component_size": len(sub_g.nodes),
            "topology_regime": regime,
        }

    def _analyze_cognitive_dissonance(
        self,
        g: nx.DiGraph,
        betweenness: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Analyze [CONTRADICTS] relationships in the graph.
        Identifies contradiction subgraphs, calculates node cognitive dissonance scores,
        computes Ising Spin-Glass frustration energy, and constructs cognitive dissonance clusters.
        """
        contradiction_edges = []
        node_dissonance: Dict[str, float] = {}

        for u, v, data in g.edges(data=True):
            if data.get("relation") == "CONTRADICTS" or "dissonance_score" in data:
                score = float(data.get("dissonance_score") or data.get("weight", 0.8))
                contradiction_edges.append({
                    "source": u,
                    "target": v,
                    "dissonance_score": round(score, 2),
                    "summary": data.get("contradiction_summary", f"Contradiction between {u} and {v}"),
                    "quote_a": data.get("quote_a", ""),
                    "quote_b": data.get("quote_b", ""),
                })
                node_dissonance[u] = node_dissonance.get(u, 0.0) + score
                node_dissonance[v] = node_dissonance.get(v, 0.0) + score

        high_dissonance_nodes = sorted(node_dissonance.keys(), key=lambda n: node_dissonance[n], reverse=True)
        high_dissonance_list = []
        for n in high_dissonance_nodes:
            high_dissonance_list.append({
                "concept": n,
                "dissonance_score": round(node_dissonance[n], 2),
                "betweenness_centrality": round(betweenness.get(n, 0.0), 3),
            })

        d_subgraph = nx.Graph()
        for edge in contradiction_edges:
            d_subgraph.add_edge(edge["source"], edge["target"], **edge)

        dissonance_clusters = []
        for idx, comp in enumerate(nx.connected_components(d_subgraph)):
            comp_nodes = list(comp)
            comp_edges = [e for e in contradiction_edges if e["source"] in comp and e["target"] in comp]
            mean_score = sum(e["dissonance_score"] for e in comp_edges) / (len(comp_edges) or 1)
            primary_edge = comp_edges[0] if comp_edges else {}

            dissonance_clusters.append({
                "cluster_id": idx + 1,
                "concepts": comp_nodes,
                "mean_dissonance_score": round(mean_score, 2),
                "primary_conflict": primary_edge.get("summary", "Ideological tension"),
                "quote_a": primary_edge.get("quote_a", ""),
                "quote_b": primary_edge.get("quote_b", ""),
            })

        ising_physics = self._compute_ising_spin_glass_energy(g)
        total_edges = len(g.edges) or 1
        overall_index = round(min(1.0, (len(contradiction_edges) * 2.0) / total_edges), 2)

        if dissonance_clusters:
            top_cluster = dissonance_clusters[0]
            concepts_str = " ile ".join(top_cluster["concepts"][:2])
            prob_stmt = (
                f"Bilişsel Çelişki Kümesi: [{concepts_str}] arasında "
                f"%{int(top_cluster['mean_dissonance_score'] * 100)} olasılıkla çelişen inanç/tutum örüntüsü saptanmıştır "
                f"(Ising Taban Enerjisi H={ising_physics['hamiltonian_ground_energy_H']}, "
                f"Yapısal Frustrasyon Φ={ising_physics['structural_frustration_index']}; "
                f"{top_cluster.get('primary_conflict', '')})."
            )
        else:
            prob_stmt = (
                "Bilişsel Çelişki Düzeyi: Gözlemlenen graf bağlantılarında belirgin bir tutarsızlık saptanmamış olup "
                "kavramsal bütünlük yüksek oranda tutarlıdır (Çelişki Skoru: %0, Ising Frustrasyon Φ=0.0)."
            )

        return {
            "overall_index": overall_index,
            "total_contradictions": len(contradiction_edges),
            "high_dissonance_concepts": high_dissonance_list,
            "high_dissonance_nodes": set(high_dissonance_nodes),
            "node_dissonance_scores": node_dissonance,
            "dissonance_clusters": dissonance_clusters,
            "ising_physics": ising_physics,
            "probabilistic_statement": prob_stmt,
        }

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
            "cognitive_dissonance": {
                "overall_dissonance_index": 0.0,
                "total_contradiction_edges": 0,
                "high_dissonance_concepts": [],
                "dissonance_clusters": [],
                "ising_spin_glass_physics": {
                    "hamiltonian_ground_energy_H": 0.0,
                    "frustrated_bonds_count": 0,
                    "structural_frustration_index": 0.0,
                    "spin_configuration": {},
                },
                "probabilistic_statement": "Grafik verisi bulunamadı.",
            },
            "network_physics": {
                "ising_spin_glass": {
                    "hamiltonian_ground_energy_H": 0.0,
                    "frustrated_bonds_count": 0,
                    "structural_frustration_index": 0.0,
                },
                "spectral_laplacian": {
                    "algebraic_connectivity_lambda2": 0.0,
                    "spectral_radius_lambda_max": 0.0,
                },
            },
            "density": 0.0,
        }
