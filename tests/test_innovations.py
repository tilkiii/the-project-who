"""
Tests for the 3 Innovative Capabilities:
1. Cognitive Chronomics (Circadian Cycle Analysis)
2. Graph-Driven Cognitive Dissonance ([CONTRADICTS] Edges & Clusters)
3. Epistemic Gaslighting & Manipulation Index
"""

from pathlib import Path
import pytest

from who.learn.chronomics import CognitiveChronomics
from who.learn.feature_extractor import FeatureExtractor
from who.vault.graph_builder import GraphBuilder
from who.vault.graph_store import GraphStore
from who.vault.graph_analytics import GraphAnalytics
from who.engine.explainability_engine import ExplainabilityEngine
from who.vault.mindset_profiler import MindsetProfiler


def test_cognitive_chronomics_parsing_and_drift(tmp_path: Path):
    """Test timestamp extraction, circadian window grouping, and epistemic decay calculation."""
    cc = CognitiveChronomics(output_dir=tmp_path)

    sample_docs = [
        {
            "content": (
                "[12.04.2024, 02:30] Gece düşünceleri: Kesinlikle hiçbir şey eskisi gibi olmayacak, mutlaka yıkılacak.\n"
                "[12.04.2024, 03:15] Herkes biliyor ki bu sistem asla çalışmaz, şüphesiz başarısız olacak.\n"
            ),
            "source_id": "chat_night.txt",
        },
        {
            "content": (
                "[12.04.2024, 11:00] Gündüz toplantısı: Veriler gösteriyor ki yeni protokol stabil görünüyor.\n"
                "[12.04.2024, 14:45] Olası senaryolar üzerinde belki de daha dikkatli durmalıyız, ampirik bulgular umut verici.\n"
            ),
            "source_id": "chat_day.txt",
        },
    ]

    segments = cc.extract_timestamp_segments(sample_docs)
    assert len(segments) >= 4

    windows = {s["window"] for s in segments}
    assert "NIGHT" in windows
    assert "DAY" in windows

    report = cc.analyze_circadian_drift(sample_docs)
    assert "circadian_drift_summary" in report
    assert "probabilistic_statement" in report
    assert "Sirkadyen Sapma" in report["probabilistic_statement"]

    # Verify JSON persistence
    assert (tmp_path / "chronomics.json").exists()


def test_epistemic_manipulation_index(tmp_path: Path):
    """Test dogmatic vs evidence density and risk tier assignment."""
    fe = FeatureExtractor(output_dir=tmp_path)

    dogmatic_texts = [
        "Kesinlikle herkes biliyor ki bu iddia şüphesiz doğrudur ve asla aksini iddia edemezsiniz.",
        "Mutlaka dediklerimiz kabul edilecek, başka hiçbir alternatif asla yoktur.",
    ]
    features_dogmatic = fe.extract_features(dogmatic_texts)
    manip_d = features_dogmatic.get("epistemic_manipulation_index", {})

    assert manip_d["score"] > 0.3
    assert "score_pct" in manip_d
    assert "probabilistic_statement" in manip_d
    assert "Manipülasyon Endeksi" in manip_d["probabilistic_statement"]

    rational_texts = [
        "Veriler gösteriyor ki model olasılıkla tutarlıdır, belki de alternatif hipotezleri göz önünde bulundurmalıyız.",
        "Ampirik gözlemler sonucunda hipotezin plausibly doğru olduğu anlaşılmaktadır.",
    ]
    features_rational = fe.extract_features(rational_texts)
    manip_r = features_rational.get("epistemic_manipulation_index", {})

    assert manip_r["score"] < manip_d["score"]
    assert "EMPIRICALLY_CALIBRATED" in manip_r["risk_tier"]


def test_graph_cognitive_dissonance_and_clustering(tmp_path: Path):
    """Test detection of CONTRADICTS edges and calculation of dissonance clusters."""
    gb = GraphBuilder(target_name="TestPersona")

    entities = [
        {"name": "Kurumsal Gizlilik", "type": "Principle"},
        {"name": "Kurumsal Seffaflik", "type": "Principle"},
    ]
    relationships = [
        {"source": "TestPersona", "target": "Kurumsal Gizlilik", "relation": "ADVOCATES", "quote": "Gizlilik mutlak zorunluluktur."},
        {"source": "TestPersona", "target": "Kurumsal Seffaflik", "relation": "ADVOCATES", "quote": "Her sey seffaf olmalidir."},
    ]

    # Test automatic dissonance detection heuristic
    dissonances = gb._detect_cognitive_dissonance(entities, relationships)
    assert len(dissonances) >= 1
    d = dissonances[0]
    assert d["dissonance_score"] >= 0.7
    assert d["relation"] == "CONTRADICTS"

    # Add detected CONTRADICTS relationship
    relationships.extend(dissonances)

    # Test persistence and analytics
    gs = GraphStore(target_name="TestPersona", vault_dir=tmp_path)
    gs.store_graph(entities, relationships)

    ga = GraphAnalytics(gs, vault_dir=tmp_path)
    metrics = ga.run_analytics()

    assert "cognitive_dissonance" in metrics
    cd = metrics["cognitive_dissonance"]
    assert cd["total_contradiction_edges"] >= 1
    assert len(cd["dissonance_clusters"]) >= 1
    assert "Bilişsel Çelişki Kümesi" in cd["probabilistic_statement"]

    gs.close()


def test_explainability_dissonance_check(tmp_path: Path):
    """Test that ExplainabilityEngine generates cognitive dissonance warnings when claims touch conflict nodes."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True)

    class TargetDirs:
        def __init__(self, base: Path):
            self.base_dir = base
            self.vault_dir = base / "vault"
            self.learned_skills_dir = base / "learned"
            self.chroma_dir = base / "chroma"
            self.raw_data_dir = base / "raw"

    dirs = TargetDirs(tmp_path)
    dirs.learned_skills_dir.mkdir(parents=True)
    dirs.chroma_dir.mkdir(parents=True)

    # Store a conflicting graph edge
    gs = GraphStore("TestPersona", vault_dir=dirs.vault_dir)
    gs.store_graph(
        entities=[
            {"name": "Kurumsal Gizlilik", "type": "Concept"},
            {"name": "Kurumsal Seffaflik", "type": "Concept"},
        ],
        relationships=[
            {
                "source": "Kurumsal Gizlilik",
                "target": "Kurumsal Seffaflik",
                "relation": "CONTRADICTS",
                "dissonance_score": 0.85,
                "summary": "Tutum gerilimi",
            }
        ],
    )
    ga = GraphAnalytics(gs, vault_dir=dirs.vault_dir)
    ga.run_analytics()
    gs.close()

    engine = ExplainabilityEngine("TestPersona", dirs)
    dissonance_res = engine._check_cognitive_dissonance("Kurumsal gizlilik ve kurumsal şeffaflık dengesi")

    assert dissonance_res["has_dissonance_tension"] is True
    assert "Bilişsel Çelişki Uyarısı" in dissonance_res["probabilistic_warning"]


def test_mindset_summary_includes_new_capabilities(tmp_path: Path):
    """Test that MindsetProfiler injects the 3 capability sections into mindset_summary.md."""
    mp = MindsetProfiler("TestPersona", vault_dir=tmp_path)

    features = {
        "chronomics": {
            "circadian_drift_summary": {"epistemic_filter_decay_pct": 18, "lexical_diversity_decay_pct": 10},
            "probabilistic_statement": "Sirkadyen Sapma: Hedefin gece metinlerinde rasyonel süzgeç %18 azalmaktadır.",
        },
        "epistemic_manipulation_index": {
            "score_pct": "22%",
            "risk_tier": "DÜŞÜK / RASYONEL",
            "dogmatic_density_per_k": 3.2,
            "evidence_density_per_k": 14.5,
            "probabilistic_statement": "Manipülasyon Endeksi: Kanıtsız iddialarda mutlak kesinlik dili kullanımı %22 seviyesindedir.",
        },
    }

    graph_metrics = {
        "top_influential_concepts": [{"node": "Demokrasi", "pagerank": 0.3}],
        "cognitive_dissonance": {
            "dissonance_clusters": [
                {
                    "cluster_id": 1,
                    "concepts": ["Gizlilik", "Seffaflik"],
                    "mean_dissonance_score": 0.80,
                    "primary_conflict": "Kurumsal gerilim",
                }
            ],
            "probabilistic_statement": "Bilişsel Çelişki Kümesi: [Gizlilik ile Seffaflik] arasında %80 olasılıkla çelişen inanç saptanmıştır.",
        },
    }

    res = mp.build_profile(
        mapping_skill={"behavioral_profile": {"cognitive_style": "Empirical", "social_orientation": "Diplomatic", "calibration_ratio": 1.4}},
        graph_metrics=graph_metrics,
        sample_quotes=["Her iddia kanıtlanmalıdır."],
        features=features,
    )

    summary_md = res["mindset_summary"]
    assert "Cognitive Chronomics" in summary_md
    assert "Sirkadyen Sapma" in summary_md
    assert "Graph-Driven Cognitive Dissonance" in summary_md
    assert "Bilişsel Çelişki Kümesi" in summary_md
    assert "Epistemic Gaslighting & Manipulation İndeksi" in summary_md
    assert "Manipülasyon Endeksi" in summary_md
