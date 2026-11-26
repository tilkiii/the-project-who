"""
Graph Data Science (GDS), Complex Network Physics, and Ising Spin-Glass Frustration analytics for who??.
Computes centrality, PageRank, community detection, shortest paths,
Unsigned Spectral Laplacian Algebraic Connectivity (Fiedler Value lambda_2),
Signed Graph Laplacian Spectral Frustration (Harary lambda_min & Rayleigh Bound),
Z2 Gauge-Fixed Exact / Simulated-Annealing Ising Ground States, and P(|q|) Replica Overlap Histograms.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from who.utils.logger import get_logger
from who.vault.graph_store import GraphStore

logger = get_logger("who.vault.analytics")


class GraphAnalytics:
    """Executes Neo4j GDS, NetworkX graph analytics, and Complex Network Physics (Ising / Signed Spectral) algorithms."""

    def __init__(self, graph_store: GraphStore, vault_dir: Optional[Path] = None):
        self.store = graph_store
        self.vault_dir = vault_dir

    def run_analytics(self) -> Dict[str, Any]:
        """Compute full graph metrics suite (PageRank, Centrality, Communities, Cognitive Dissonance, Ising & Signed Spectral Physics)."""
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

        # 5. Cognitive Dissonance & Z2 Gauge-Fixed Ising Spin-Glass Analysis
        dissonance_analysis = self._analyze_cognitive_dissonance(g, betweenness)

        # 6. Spectral Graph Physics (Unsigned Fiedler lambda_2 + Signed Laplacian Harary Frustration lambda_min + Rayleigh Bound)
        undirected_g = g.to_undirected()
        spectral_physics = self._compute_spectral_laplacian_metrics(
            g,
            undirected_g,
            frustrated_weight_f=dissonance_analysis["ising_physics"].get("frustrated_bond_weight_F", 0.0),
        )

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

            logger.info("Saved graph_metrics.json (with Z2 Gauge-Fixed Ising & Signed Spectral Physics) successfully.")

        return analytics_payload

    def _compute_ising_spin_glass_energy(
        self,
        g: nx.DiGraph,
        n_restarts: int = 12,
        sweeps_per_temp: int = 4,
    ) -> Dict[str, Any]:
        """
        Model the concept graph as a signed Ising Spin-Glass system (Toulouse, 1977; Heider Structural Balance):
        - Hamiltonian: H(s) = - sum_{(i,j)} J_{ij} * s_i * s_j, where s_i in {-1, +1}
        - Enforces Z2 Gauge Fixing (s_0 = +1, or s -> s_0 * s) so global spin-flip twins (s and -s)
          are mapped to the same canonical gauge representative.
        - Runs multi-sweep, multi-restart Simulated Annealing AND exact brute-force enumeration for N <= 12
          to verify SA ground-state recovery accuracy, compute the P(|q|) replica overlap histogram,
          and count distinct gauge-fixed ground states.
        """
        nodes = list(g.nodes)
        if not nodes:
            return {
                "hamiltonian_ground_energy_H": 0.0,
                "frustrated_bonds_count": 0,
                "frustrated_bond_weight_F": 0.0,
                "structural_frustration_index": 0.0,
                "replica_overlap_stability": 1.0,
                "replica_overlap_histogram_P_q": {"0.00-0.25": 0.0, "0.25-0.50": 0.0, "0.50-0.75": 0.0, "0.75-1.00": 1.0},
                "distinct_ground_states_count": 1,
                "sa_exact_ground_state_recovery_rate": 1.0,
                "annealing_restarts": n_restarts,
                "spin_configuration": {},
                "corpus_external_fields_h": {},
                "node_salience_m": {},
                "serialized_couplings_J": [],
            }

        couplings: List[Tuple[str, str, float]] = []
        total_abs_coupling = 0.0
        adj_map: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
        node_pos_weight: Dict[str, float] = {n: 0.0 for n in nodes}
        node_neg_weight: Dict[str, float] = {n: 0.0 for n in nodes}

        neg_stance_tokens = {
            "reject", "oppose", "never", "against", "refuse", "unacceptable", "threat", "risk",
            "reddet", "karşı", "asla", "kabul edilemez", "tehdit", "engel", "veto", "zarar",
        }
        pos_stance_tokens = {
            "support", "advocate", "value", "protect", "essential", "core", "preserve", "retain",
            "savun", "destek", "değer", "koru", "temel", "otonomi", "kalır", "şeffaf", "hak",
        }

        for u, v, data in g.edges(data=True):
            if u == v:
                continue
            rel = str(data.get("relation", "RELATES_TO")).upper()
            is_contra = rel in {"CONTRADICTS", "OPPOSES", "REJECTS"} or float(data.get("dissonance_score", 0.0)) > 0.0
            weight = float(data.get("dissonance_score") or data.get("weight", 0.8))
            j_val = -abs(weight) if is_contra else abs(weight)
            couplings.append((u, v, j_val))
            total_abs_coupling += abs(j_val)
            adj_map[u].append((v, j_val))
            adj_map[v].append((u, j_val))

            quote_lower = str(data.get("quote", "") or data.get("quote_a", "")).lower()
            if is_contra:
                node_neg_weight[v] += abs(weight)
                node_pos_weight[u] += 0.5 * abs(weight)
            else:
                node_pos_weight[u] += abs(weight)
                node_pos_weight[v] += abs(weight)
            if any(tok in quote_lower for tok in neg_stance_tokens):
                node_neg_weight[v] += 0.35
            if any(tok in quote_lower for tok in pos_stance_tokens):
                node_pos_weight[u] += 0.35

        corpus_external_fields_h: Dict[str, float] = {}
        node_salience_m: Dict[str, float] = {}
        max_total_w = max(
            [node_pos_weight[n] + node_neg_weight[n] for n in nodes] + [1.0]
        )
        for n in nodes:
            n_data = g.nodes[n]
            explicit_h = n_data.get("external_field_h")
            if explicit_h is not None:
                h_val = float(explicit_h)
            else:
                pw = node_pos_weight[n]
                nw = node_neg_weight[n]
                n_lower = str(n).lower()
                if any(tok in n_lower for tok in neg_stance_tokens):
                    nw += 0.6
                if any(tok in n_lower for tok in pos_stance_tokens):
                    pw += 0.6
                h_val = math.tanh((pw - nw) / max(1.0, pw + nw)) if (pw + nw) > 0 else 0.20
            corpus_external_fields_h[n] = round(max(-1.0, min(1.0, h_val)), 4)
            salience = (node_pos_weight[n] + node_neg_weight[n] + 0.25) / (max_total_w + 0.25)
            node_salience_m[n] = round(max(0.15, min(1.0, salience)), 4)

        def _eval_energy(s_cfg: Dict[str, int]) -> float:
            return sum(-j_val * s_cfg[u] * s_cfg[v] for u, v, j_val in couplings)

        def _gauge_fix(s_cfg: Dict[str, int]) -> Dict[str, int]:
            """Break Z2 symmetry (s -> -s) by fixing the anchor node spin to +1."""
            anchor_spin = s_cfg[nodes[0]] if s_cfg[nodes[0]] != 0 else 1
            return {n: s_cfg[n] * anchor_spin for n in nodes}

        rng = random.Random(42)
        cooling_temps = [2.0, 1.2, 0.7, 0.35, 0.15, 0.05, 0.0]
        annealed_minima: List[Dict[str, int]] = []
        annealed_energies: List[float] = []
        best_spins: Dict[str, int] = {n: 1 for n in nodes}
        best_energy = _eval_energy(best_spins)

        for r_idx in range(max(1, n_restarts)):
            if r_idx == 0:
                spins = {n: 1 for n in nodes}
            else:
                spins = {n: rng.choice([-1, 1]) for n in nodes}

            for temp in cooling_temps:
                for _ in range(max(1, sweeps_per_temp)):
                    for n in nodes:
                        local_field = sum(j_val * spins[nbr] for nbr, j_val in adj_map[n])
                        delta_e = 2.0 * spins[n] * local_field
                        if delta_e < -1e-9:
                            spins[n] = -spins[n]
                        elif temp > 0.0 and delta_e < 12.0:
                            if rng.random() < math.exp(-delta_e / temp):
                                spins[n] = -spins[n]

            spins_gf = _gauge_fix(spins)
            e_curr = _eval_energy(spins_gf)
            annealed_minima.append(spins_gf)
            annealed_energies.append(e_curr)
            if e_curr < best_energy - 1e-9:
                best_energy = e_curr
                best_spins = dict(spins_gf)

        # For N <= 12, perform exact Z2 gauge-fixed brute-force enumeration (2^(N-1) states)
        # to guarantee global ground state and measure SA exact recovery rate
        exact_ground_count = 0
        if len(nodes) <= 12 and couplings:
            exact_best_e = best_energy
            exact_ground_states: List[Dict[str, int]] = []
            n_free = len(nodes) - 1
            for mask in range(1 << n_free):
                cand = {nodes[0]: 1}
                for bit_idx in range(n_free):
                    cand[nodes[bit_idx + 1]] = 1 if ((mask >> bit_idx) & 1) else -1
                e_val = _eval_energy(cand)
                if e_val < exact_best_e - 1e-7:
                    exact_best_e = e_val
                    exact_ground_states = [cand]
                elif abs(e_val - exact_best_e) <= 1e-7:
                    exact_ground_states.append(cand)
            if exact_ground_states:
                best_energy = exact_best_e
                best_spins = dict(exact_ground_states[0])
                exact_ground_count = len(exact_ground_states)

        sa_hits = sum(1 for e_val in annealed_energies if abs(e_val - best_energy) <= 1e-6)
        sa_recovery_rate = round(sa_hits / max(1, len(annealed_energies)), 4)

        # Count distinct gauge-fixed ground states found across replicas (or exact enumeration)
        distinct_gs_tuples = {
            tuple(s_cfg[n] for n in nodes)
            for s_cfg, e_val in zip(annealed_minima, annealed_energies)
            if abs(e_val - best_energy) <= 1e-6
        }
        distinct_gs_count = max(exact_ground_count, len(distinct_gs_tuples), 1)

        # Compute gauge-fixed replica overlap distribution P(|q|) across annealed minima
        overlaps: List[float] = []
        hist_bins = {"0.00-0.25": 0, "0.25-0.50": 0, "0.50-0.75": 0, "0.75-1.00": 0}
        for i in range(len(annealed_minima)):
            for j in range(i + 1, len(annealed_minima)):
                s_a, s_b = annealed_minima[i], annealed_minima[j]
                q_ab = abs(sum(s_a[n] * s_b[n] for n in nodes)) / len(nodes)
                overlaps.append(q_ab)
                if q_ab < 0.25:
                    hist_bins["0.00-0.25"] += 1
                elif q_ab < 0.50:
                    hist_bins["0.25-0.50"] += 1
                elif q_ab < 0.75:
                    hist_bins["0.50-0.75"] += 1
                else:
                    hist_bins["0.75-1.00"] += 1

        total_pairs = max(1, len(overlaps))
        p_q_hist = {k: round(v / total_pairs, 4) for k, v in hist_bins.items()} if overlaps else {
            "0.00-0.25": 0.0, "0.25-0.50": 0.0, "0.50-0.75": 0.0, "0.75-1.00": 1.0
        }
        replica_stability = round(sum(overlaps) / len(overlaps), 4) if overlaps else 1.0

        frustrated_bonds = 0
        frustrated_weight_f = 0.0
        for u, v, j_val in couplings:
            bond_term = j_val * best_spins[u] * best_spins[v]
            if bond_term < 0:
                frustrated_bonds += 1
                frustrated_weight_f += abs(j_val)

        frustration_index = round(frustrated_weight_f / total_abs_coupling, 4) if total_abs_coupling > 0 else 0.0

        return {
            "hamiltonian_ground_energy_H": round(best_energy, 4),
            "frustrated_bonds_count": frustrated_bonds,
            "frustrated_bond_weight_F": round(frustrated_weight_f, 4),
            "total_couplings": len(couplings),
            "structural_frustration_index": frustration_index,
            "replica_overlap_stability": replica_stability,
            "replica_overlap_histogram_P_q": p_q_hist,
            "distinct_ground_states_count": distinct_gs_count,
            "sa_exact_ground_state_recovery_rate": sa_recovery_rate,
            "annealing_restarts": n_restarts,
            "spin_configuration": best_spins,
            "corpus_external_fields_h": corpus_external_fields_h,
            "node_salience_m": node_salience_m,
            "serialized_couplings_J": [
                {"u": u, "v": v, "J": round(j_val, 4)} for u, v, j_val in couplings[:64]
            ],
        }

    @staticmethod
    def _symmetric_eigenvalues(matrix: List[List[float]]) -> List[float]:
        """
        Compute sorted real eigenvalues of a real symmetric matrix.
        Uses NumPy if available, otherwise executes pure-Python cyclic Jacobi rotations.
        """
        n = len(matrix)
        if n == 0:
            return []
        if n == 1:
            return [float(matrix[0][0])]

        try:
            import numpy as np
            arr = np.array(matrix, dtype=float)
            eigs = np.sort(np.real(np.linalg.eigvalsh(arr)))
            return [float(x) for x in eigs]
        except Exception:
            pass

        a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
        for _ in range(35):
            max_off = 0.0
            for p in range(n):
                for q in range(p + 1, n):
                    apq = abs(a[p][q])
                    if apq > max_off:
                        max_off = apq
                    if apq < 1e-10:
                        continue
                    diff = a[q][q] - a[p][p]
                    if abs(diff) < 1e-12:
                        t = 1.0 if a[p][q] >= 0 else -1.0
                    else:
                        theta = diff / (2.0 * a[p][q])
                        t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(1.0 + theta * theta))
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    temp = a[p][q]
                    a[p][q] = 0.0
                    a[q][p] = 0.0
                    a[p][p] -= t * temp
                    a[q][q] += t * temp
                    for r in range(n):
                        if r != p and r != q:
                            arp = a[r][p]
                            arq = a[r][q]
                            a[r][p] = arp - s * (arq + tau * arp)
                            a[p][r] = a[r][p]
                            a[r][q] = arq + s * (arp - tau * arq)
                            a[q][r] = a[r][q]
            if max_off < 1e-9:
                break
        return sorted(a[i][i] for i in range(n))

    def _compute_spectral_laplacian_metrics(
        self,
        dg: nx.DiGraph,
        ug: nx.Graph,
        frustrated_weight_f: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute both Unsigned and Signed Spectral Graph Theory metrics:
        1. Unsigned Graph Laplacian (L = D - A):
           - Algebraic Connectivity (Fiedler Value lambda_2)
           - Spectral Radius (lambda_max)
        2. Signed Graph Laplacian (L_signed = D_{|A|} - A_signed, Kunegis et al. 2010 / Harary Theorem):
           - Smallest Eigenvalue lambda_min(L_signed) & degree-normalized lambda_min / <d>
           - Rayleigh-Harary Upper Bound Check: lambda_min <= 4 * F / n  <=>  F >= n * lambda_min / 4.
        """
        n_nodes = len(ug.nodes)
        if n_nodes < 2:
            return {
                "algebraic_connectivity_lambda2": 0.0,
                "spectral_radius_lambda_max": 0.0,
                "signed_laplacian_frustration_lambda_min": 0.0,
                "normalized_signed_lambda_min": 0.0,
                "rayleigh_frustration_lower_bound_F": 0.0,
                "rayleigh_bound_satisfied": True,
                "is_structurally_balanced": True,
                "average_clustering_coefficient": 0.0,
                "topology_regime": "SPARSE_SINGLETON",
            }

        try:
            avg_clustering = round(float(nx.average_clustering(ug)), 4)
        except Exception:
            avg_clustering = 0.0

        largest_cc_nodes = list(max(nx.connected_components(ug), key=len))
        sub_g = ug.subgraph(largest_cc_nodes).copy()

        lambda_2 = 0.0
        lambda_max = 0.0
        signed_lambda_min = 0.0
        norm_signed_lambda_min = 0.0
        m = len(sub_g.nodes)

        if m >= 2:
            node_list = list(sub_g.nodes)
            node_idx = {n: i for i, n in enumerate(node_list)}

            a_unsigned = [[0.0] * m for _ in range(m)]
            a_signed = [[0.0] * m for _ in range(m)]

            for u, v, data in dg.edges(data=True):
                if u in node_idx and v in node_idx and u != v:
                    i, j = node_idx[u], node_idx[v]
                    is_contra = data.get("relation") == "CONTRADICTS" or float(data.get("dissonance_score", 0.0)) > 0.0
                    w = abs(float(data.get("dissonance_score") or data.get("weight", 0.8)))
                    a_unsigned[i][j] = w
                    a_unsigned[j][i] = w
                    signed_w = -w if is_contra else w
                    a_signed[i][j] = signed_w
                    a_signed[j][i] = signed_w

            l_unsigned = [[0.0] * m for _ in range(m)]
            l_signed = [[0.0] * m for _ in range(m)]
            total_deg = 0.0
            for i in range(m):
                deg_i = sum(a_unsigned[i])
                total_deg += deg_i
                for j in range(m):
                    if i == j:
                        l_unsigned[i][j] = deg_i
                        l_signed[i][j] = deg_i
                    else:
                        l_unsigned[i][j] = -a_unsigned[i][j]
                        l_signed[i][j] = -a_signed[i][j]

            eig_unsigned = self._symmetric_eigenvalues(l_unsigned)
            eig_signed = self._symmetric_eigenvalues(l_signed)

            lambda_2 = round(float(max(0.0, eig_unsigned[1])), 4) if len(eig_unsigned) >= 2 else 0.0
            lambda_max = round(float(max(0.0, eig_unsigned[-1])), 4) if eig_unsigned else 0.0
            signed_lambda_min = round(float(max(0.0, eig_signed[0])), 6) if eig_signed else 0.0
            mean_deg = (total_deg / m) if m > 0 else 1.0
            norm_signed_lambda_min = round(signed_lambda_min / max(mean_deg, 1e-6), 4)

        is_balanced = signed_lambda_min < 1e-5
        rayleigh_lower_bound_f = round((m * signed_lambda_min) / 4.0, 4)
        rayleigh_ok = (frustrated_weight_f + 1e-4) >= rayleigh_lower_bound_f

        if lambda_2 >= 1.0:
            regime = "COHESIVE_STRONGLY_COUPLED_MANIFOLD"
        elif lambda_2 >= 0.25:
            regime = "MODULAR_MULTI_CLUSTER_TOPOLOGY"
        else:
            regime = "WEAKLY_COUPLED_FRAGMENTED_GRAPH"

        return {
            "algebraic_connectivity_lambda2": lambda_2,
            "spectral_radius_lambda_max": lambda_max,
            "signed_laplacian_frustration_lambda_min": round(signed_lambda_min, 4),
            "normalized_signed_lambda_min": norm_signed_lambda_min,
            "rayleigh_frustration_lower_bound_F": rayleigh_lower_bound_f,
            "rayleigh_bound_satisfied": rayleigh_ok,
            "is_structurally_balanced": is_balanced,
            "average_clustering_coefficient": avg_clustering,
            "largest_component_size": m,
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
        computes Z2 Gauge-Fixed Ising Spin-Glass frustration energy, and constructs cognitive dissonance clusters.
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
            concepts_str = " and ".join(top_cluster["concepts"][:2])
            prob_stmt = (
                f"Cognitive Dissonance Cluster: Conflicting belief/attitude pattern detected between [{concepts_str}] "
                f"at {int(top_cluster['mean_dissonance_score'] * 100)}% likelihood "
                f"(Ising Ground Energy H={ising_physics['hamiltonian_ground_energy_H']}, "
                f"Structural Frustration Φ={ising_physics['structural_frustration_index']}; "
                f"{top_cluster.get('primary_conflict', '')})."
            )
        else:
            prob_stmt = (
                "Cognitive Dissonance Level: No pronounced contradiction detected across observed graph edges; "
                "conceptual coherence is high (Dissonance Score: 0%, Ising Frustration Φ=0.0)."
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

    @staticmethod
    def construct_harary_balanced_counterfactual(
        nodes: List[str],
        couplings: List[Tuple[str, str, float]],
    ) -> Dict[str, Any]:
        """
        Construct the Harary Structurally Balanced Counterfactual (E2_BAL) for a signed graph:
        Preserves exact topology, nodes, and edge magnitudes |J_{ij}|, while flipping the sign of every
        frustrated bond violated in the h=0 ground state s* so signed Laplacian lambda_min == 0.
        Enables isolating true cycle frustration from generic thermal dispersion at temperature T_b.
        """
        if not nodes or not couplings:
            return {
                "balanced_couplings": list(couplings),
                "flipped_bonds_count": 0,
                "original_frustrated_weight_F": 0.0,
                "balanced_signed_lambda_min": 0.0,
                "ground_state_spins": {n: 1 for n in nodes},
            }

        n_nodes = len(nodes)
        adj_map: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
        for u, v, j_val in couplings:
            if u != v and u in adj_map and v in adj_map:
                adj_map[u].append((v, float(j_val)))
                adj_map[v].append((u, float(j_val)))

        def _e0(s_cfg: Dict[str, int]) -> float:
            return sum(-float(j_val) * s_cfg[u] * s_cfg[v] for u, v, j_val in couplings)

        best_spins: Dict[str, int] = {n: 1 for n in nodes}
        best_e = _e0(best_spins)

        if n_nodes <= 12:
            n_free = n_nodes - 1
            for mask in range(1 << n_free):
                cand = {nodes[0]: 1}
                for bit_idx in range(n_free):
                    cand[nodes[bit_idx + 1]] = 1 if ((mask >> bit_idx) & 1) else -1
                e_c = _e0(cand)
                if e_c < best_e - 1e-9:
                    best_e = e_c
                    best_spins = cand
        else:
            rng = random.Random(42)
            for r_idx in range(12):
                spins = {n: 1 for n in nodes} if r_idx == 0 else {n: rng.choice([-1, 1]) for n in nodes}
                for temp in [2.0, 1.0, 0.4, 0.1, 0.0]:
                    for _ in range(4):
                        for n in nodes:
                            loc = sum(jv * spins[nbr] for nbr, jv in adj_map[n])
                            de = 2.0 * spins[n] * loc
                            if de < -1e-9 or (temp > 0.0 and de < 12.0 and rng.random() < math.exp(-de / temp)):
                                spins[n] = -spins[n]
                if _e0(spins) < best_e - 1e-9:
                    best_e = _e0(spins)
                    best_spins = dict(spins)

        balanced_couplings: List[Tuple[str, str, float]] = []
        flipped_count = 0
        orig_f = 0.0
        for u, v, j_val in couplings:
            jv = float(j_val)
            if jv * best_spins.get(u, 1) * best_spins.get(v, 1) < -1e-9:
                orig_f += abs(jv)
                flipped_count += 1
                balanced_couplings.append((u, v, -jv))
            else:
                balanced_couplings.append((u, v, jv))

        node_idx = {n: i for i, n in enumerate(nodes)}
        l_bal = [[0.0] * n_nodes for _ in range(n_nodes)]
        for u, v, jv in balanced_couplings:
            if u in node_idx and v in node_idx and u != v:
                i, j = node_idx[u], node_idx[v]
                l_bal[i][i] += abs(jv)
                l_bal[j][j] += abs(jv)
                l_bal[i][j] -= jv
                l_bal[j][i] -= jv

        bal_eigs = GraphAnalytics._symmetric_eigenvalues(l_bal)
        bal_lam_min = max(0.0, float(bal_eigs[0])) if bal_eigs else 0.0

        return {
            "balanced_couplings": balanced_couplings,
            "flipped_bonds_count": flipped_count,
            "original_frustrated_weight_F": round(orig_f, 4),
            "balanced_signed_lambda_min": round(bal_lam_min, 6),
            "ground_state_spins": best_spins,
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
                    "frustrated_bond_weight_F": 0.0,
                    "structural_frustration_index": 0.0,
                    "replica_overlap_stability": 1.0,
                    "distinct_ground_states_count": 1,
                    "spin_configuration": {},
                },
                "probabilistic_statement": "No graph data available.",
            },
            "network_physics": {
                "ising_spin_glass": {
                    "hamiltonian_ground_energy_H": 0.0,
                    "frustrated_bonds_count": 0,
                    "frustrated_bond_weight_F": 0.0,
                    "structural_frustration_index": 0.0,
                    "replica_overlap_stability": 1.0,
                },
                "spectral_laplacian": {
                    "algebraic_connectivity_lambda2": 0.0,
                    "spectral_radius_lambda_max": 0.0,
                    "signed_laplacian_frustration_lambda_min": 0.0,
                    "normalized_signed_lambda_min": 0.0,
                    "rayleigh_frustration_lower_bound_F": 0.0,
                    "rayleigh_bound_satisfied": True,
                    "is_structurally_balanced": True,
                },
            },
            "density": 0.0,
        }
