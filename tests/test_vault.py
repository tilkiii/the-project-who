"""
Tests for Module 3: who_vault (Graph Builder, Graph Store, Graph Analytics, Vector Embedder, Mindset Profiler,
Ising Spin-Glass Frustration Energy, and Spectral Laplacian Algebraic Connectivity).
"""

from pathlib import Path
import pytest

from who.vault.graph_analytics import GraphAnalytics
from who.vault.graph_builder import GraphBuilder
from who.vault.graph_store import GraphStore
from who.vault.mindset_profiler import MindsetProfiler
from who.vault.vector_embedder import VectorEmbedder


def test_graph_builder():
    gb = GraphBuilder("Alice")
    text = "Alice advocates for Decentralized Protocols because they eliminate single points of failure. She critiques Centralized Monopolies."
    elements = gb.extract_graph_elements(text, source_id="test_doc")

    assert "entities" in elements
    assert "relationships" in elements
    assert len(elements["relationships"]) > 0


def test_graph_store_and_export(tmp_path: Path):
    store = GraphStore("Alice", vault_dir=tmp_path)
    entities = [
        {"name": "Alice", "type": "Person", "description": "Target individual"},
        {"name": "Decentralized Protocols", "type": "Concept", "description": "Resilient systems"},
        {"name": "Cryptographic Proofs", "type": "Methodology", "description": "Formal verification"},
    ]
    relationships = [
        {"source": "Alice", "target": "Decentralized Protocols", "relation": "ADVOCATES", "weight": 0.9, "quote": "Decentralized protocols eliminate failure."},
        {"source": "Decentralized Protocols", "target": "Cryptographic Proofs", "relation": "DEPENDS_ON", "weight": 0.85, "quote": "Protocols depend on proofs."},
    ]
    store.store_graph(entities, relationships)

    nodes = store.get_all_nodes()
    rels = store.get_all_relationships()
    assert len(nodes) == 3
    assert len(rels) == 2

    exported = store.export_graph_json()
    assert (tmp_path / "concept_network.json").exists()
    assert len(exported["nodes"]) == 3
    store.close()


def test_graph_analytics(tmp_path: Path):
    store = GraphStore("Alice", vault_dir=tmp_path)
    entities = [
        {"name": "Alice", "type": "Person"},
        {"name": "NodeA", "type": "Concept"},
        {"name": "NodeB", "type": "Concept"},
    ]
    relationships = [
        {"source": "Alice", "target": "NodeA", "relation": "ADVOCATES", "weight": 0.9},
        {"source": "NodeA", "target": "NodeB", "relation": "CONNECTS_TO", "weight": 0.8},
        {"source": "Alice", "target": "NodeB", "relation": "CITES", "weight": 0.7},
    ]
    store.store_graph(entities, relationships)

    ga = GraphAnalytics(store, vault_dir=tmp_path)
    metrics = ga.run_analytics()

    assert metrics["total_nodes"] == 3
    assert metrics["total_relationships"] == 3
    assert len(metrics["top_influential_concepts"]) == 3
    assert "network_physics" in metrics
    assert "spectral_laplacian" in metrics["network_physics"]
    spec = metrics["network_physics"]["spectral_laplacian"]
    assert spec["algebraic_connectivity_lambda2"] > 0.0
    assert spec["signed_laplacian_frustration_lambda_min"] == pytest.approx(0.0, abs=1e-3)
    assert spec["is_structurally_balanced"] is True
    assert (tmp_path / "graph_metrics.json").exists()
    assert (tmp_path / "relationship_strengths.json").exists()

    path = ga.find_shortest_path("Alice", "NodeB")
    assert path == ["Alice", "NodeB"] or len(path) >= 2
    store.close()


def test_ising_spin_glass_frustration_triangle(tmp_path: Path):
    """
    Verify Toulouse / Heider frustrated spin-glass triangle:
    A --(+1)--> B, B --(+1)--> C, and A --(-1: CONTRADICTS)--> C.
    A 3-cycle with two ferromagnetic (+J) bonds and one antiferromagnetic (-J) bond
    cannot satisfy all three bonds simultaneously, resulting in frustrated_bonds_count >= 1
    and structural_frustration_index > 0.
    """
    store = GraphStore("Target_Ising", vault_dir=tmp_path)
    entities = [
        {"name": "Concept_A", "type": "Concept"},
        {"name": "Concept_B", "type": "Concept"},
        {"name": "Concept_C", "type": "Concept"},
    ]
    relationships = [
        {"source": "Concept_A", "target": "Concept_B", "relation": "SUPPORTS", "weight": 0.9},
        {"source": "Concept_B", "target": "Concept_C", "relation": "SUPPORTS", "weight": 0.9},
        {"source": "Concept_A", "target": "Concept_C", "relation": "CONTRADICTS", "weight": 0.9, "dissonance_score": 0.9},
    ]
    store.store_graph(entities, relationships)

    ga = GraphAnalytics(store, vault_dir=tmp_path)
    metrics = ga.run_analytics()
    ising = metrics["network_physics"]["ising_spin_glass"]
    spec = metrics["network_physics"]["spectral_laplacian"]

    assert ising["frustrated_bonds_count"] >= 1
    assert ising["structural_frustration_index"] > 0.20
    assert ising["annealing_restarts"] == 12
    assert ising["sa_exact_ground_state_recovery_rate"] == 1.0
    assert "replica_overlap_histogram_P_q" in ising
    assert "distinct_ground_states_count" in ising
    # Harary's Structural Balance Theorem & Rayleigh Quotient Bound: F >= n * lambda_min / 4
    assert spec["signed_laplacian_frustration_lambda_min"] > 0.10
    assert spec["rayleigh_bound_satisfied"] is True
    assert ising["frustrated_bond_weight_F"] + 1e-4 >= (3 * spec["signed_laplacian_frustration_lambda_min"]) / 4.0
    assert spec["is_structurally_balanced"] is False
    store.close()


def test_random_gauge_switched_harary_balance(tmp_path: Path):
    """
    Verify Harary's Structural Balance Theorem under random Z2 gauge switching:
    Take a 6-node graph, assign random node gauge signs sigma_i in {-1, +1},
    and set edge sign J_ij = sigma_i * sigma_j (SUPPORTS if +1, CONTRADICTS if -1).
    Even though the graph contains negative (CONTRADICTS) edges, every cycle has an even
    number of negative edges, so lambda_min(L_signed) must vanish (< 1e-5) and F == 0.
    """
    import random
    rng = random.Random(123)
    node_names = [f"N{i}" for i in range(6)]
    sigmas = {n: rng.choice([-1, 1]) for n in node_names}

    entities = [{"name": n, "type": "Concept"} for n in node_names]
    relationships = []
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if rng.random() < 0.7:
                u, v = node_names[i], node_names[j]
                sign = sigmas[u] * sigmas[v]
                rel = "SUPPORTS" if sign > 0 else "CONTRADICTS"
                d_score = 0.0 if sign > 0 else 0.85
                relationships.append({
                    "source": u,
                    "target": v,
                    "relation": rel,
                    "weight": 0.85,
                    "dissonance_score": d_score,
                })

    store = GraphStore("Target_Harary_Gauge", vault_dir=tmp_path)
    store.store_graph(entities, relationships)
    ga = GraphAnalytics(store, vault_dir=tmp_path)
    metrics = ga.run_analytics()

    ising = metrics["network_physics"]["ising_spin_glass"]
    spec = metrics["network_physics"]["spectral_laplacian"]

    assert ising["frustrated_bonds_count"] == 0
    assert ising["frustrated_bond_weight_F"] == pytest.approx(0.0, abs=1e-6)
    assert spec["signed_laplacian_frustration_lambda_min"] == pytest.approx(0.0, abs=1e-5)
    assert spec["is_structurally_balanced"] is True
    store.close()


def test_vector_embedder(tmp_path: Path):
    ve = VectorEmbedder("Alice", chroma_dir=tmp_path)
    docs = [
        {"content": "Decentralized state machines provide Byzantine fault tolerance across untrusted peers.", "source_url": "doc1"},
        {"content": "Zero-knowledge verification allows privacy without sacrificing auditability.", "source_url": "doc2"},
    ]
    count = ve.add_documents(docs)
    assert count >= 2

    results = ve.search_similar("Byzantine fault tolerance", top_k=1)
    assert len(results) == 1
    assert "Byzantine" in results[0]["text"]


def test_mindset_profiler(tmp_path: Path):
    mp = MindsetProfiler("Alice", vault_dir=tmp_path)
    skill = {
        "behavioral_profile": {"cognitive_style": "Calibrated Empirical", "social_orientation": "Collaborative", "calibration_ratio": 1.4},
        "generation_rules": ["Ground all findings in empirical citations."],
        "supported_evidence_models": [{"scientific_basis": "LIWC Word Count", "domain": "Psychology", "citation": "http://doi.org", "applied_insight": "High validity"}],
    }
    graph_metrics = {
        "total_nodes": 5,
        "total_relationships": 6,
        "density": 0.4,
        "top_influential_concepts": [{"node": "Consensus", "type": "Concept", "pagerank": 0.35}],
    }
    quotes = ["We must rely on verifiable proofs, never authorities."]

    res = mp.build_profile(skill, graph_metrics, quotes)

    assert (tmp_path / "character.json").exists()
    assert (tmp_path / "mindset_summary.md").exists()
    assert (tmp_path / "cognitive_evidence_report.md").exists()
    assert (tmp_path / "profile_confidence_report.md").exists()

    char_data = res["character"]
    assert char_data["name"] == "Alice"
    assert "bio" in char_data
    assert "lore" in char_data
    assert "style" in char_data
