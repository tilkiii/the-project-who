"""
Tests for Module 2: who_learn (Feature Extractor, Topic Mapper, Academic Researcher, Skill Synthesizer).
"""

from pathlib import Path
import pytest

from src.learn.academic_researcher import AcademicResearcher
from src.learn.feature_extractor import FeatureExtractor
from src.learn.skill_synthesizer import SkillSynthesizer
from src.learn.topic_mapper import TopicMapper


def test_feature_extractor(tmp_path: Path):
    fe = FeatureExtractor(output_dir=tmp_path)
    texts = [
        "I believe we must test hypotheses rigorously because empirical validation ensures robust outcomes.",
        "Perhaps the system is likely to stabilize, though we definitely observed initial volatility.",
        "Why do centralized models persist? Because power concentrates unless constrained by protocols!"
    ]
    features = fe.extract_features(texts)

    assert "corpus_stats" in features
    assert features["corpus_stats"]["total_words"] > 0
    assert "pronoun_ratios" in features
    assert "epistemic_markers" in features
    assert features["epistemic_markers"]["epistemic_calibration_ratio"] > 0
    assert (tmp_path / "features.json").exists()


def test_topic_mapper(tmp_path: Path):
    tm = TopicMapper(output_dir=tmp_path)
    docs = [
        {"content": "Decentralized consensus protocols ensure Byzantine fault tolerance across nodes."},
        {"content": "Cryptographic zero-knowledge proofs enable privacy-preserving state verification."},
        {"content": "Quadratic voting algorithms prevent governance capture in distributed protocols."},
        {"content": "Byzantine agreement algorithms and consensus protocol scaling mechanisms."},
    ]
    res = tm.extract_topics(docs, nr_topics=2)

    assert "topics" in res
    assert len(res["topics"]) > 0
    assert "topic_evolution" in res
    assert (tmp_path / "topics.json").exists()
    assert (tmp_path / "topic_evolution.json").exists()


def test_academic_researcher(tmp_path: Path):
    ar = AcademicResearcher(output_dir=tmp_path)
    features = {
        "epistemic_markers": {"epistemic_calibration_ratio": 1.6},
        "pronoun_ratios": {"first_person_singular_pct": 2.5},
    }
    topics = [{"keywords": ["cryptography", "consensus"]}]
    papers = ar.research_markers(features, topics, max_papers_per_query=1)

    assert len(papers) >= 1
    assert "title" in papers[0]
    assert "doi_or_url" in papers[0]
    assert (tmp_path / "retrieved_academic_literature.json").exists()


def test_skill_synthesizer(tmp_path: Path):
    ss = SkillSynthesizer(output_dir=tmp_path)
    features = {
        "epistemic_markers": {"epistemic_calibration_ratio": 1.4},
        "pronoun_ratios": {"first_person_singular_pct": 2.0},
        "corpus_stats": {"type_token_ratio": 0.55},
    }
    topics = [{"name": "Cryptographic Verification", "keywords": ["proof", "verification"]}]
    lit = [{"title": "Linguistic Inquiry in Psychology", "doi_or_url": "https://doi.org/10.1177/0261927X09351676", "domain": "Psychology"}]

    skill = ss.synthesize_mapping_skill("Alice", features, topics, lit)

    assert skill["target_name"] == "Alice"
    assert "behavioral_profile" in skill
    assert len(skill["generation_rules"]) > 0
    assert (tmp_path / "generated_mapping_skill.json").exists()
