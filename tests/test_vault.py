"""
Tests for Module 3: who_vault (Graph Builder, Graph Store, Graph Analytics, Vector Embedder, Mindset Profiler).
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
    assert (tmp_path / "graph_metrics.json").exists()
    assert (tmp_path / "relationship_strengths.json").exists()

    path = ga.find_shortest_path("Alice", "NodeB")
    assert path == ["Alice", "NodeB"] or len(path) >= 2
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

    # Check ElizaOS character format
    char_data = res["character"]
    assert char_data["name"] == "Alice"
    assert "bio" in char_data
    assert "lore" in char_data
    assert "style" in char_data
