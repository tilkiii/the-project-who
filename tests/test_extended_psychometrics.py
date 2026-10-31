"""
Unit and integration tests for Extended Psychometrics, AI Scientific Referee,
Knowledge Bank, and Boundary Synthesizer.
"""

from pathlib import Path
import json
import pytest

from src.learn.feature_extractor import FeatureExtractor
from src.learn.scientific_referee import ScientificReferee, fisher_z, inv_fisher_z
from src.learn.boundary_synthesizer import BoundarySynthesizer
from src.learn.autonomous_researcher import AutonomousAcademicResearcher
from src.learn.skill_synthesizer import SkillSynthesizer


def test_feature_extractor_extended_psychometrics(tmp_path: Path):
    """Test calculation of 5 extended psychometric dimensions across Turkish & English text."""
    fe = FeatureExtractor(output_dir=tmp_path)
    corpus = [
        "Bizim için stratejik menfaat ve pragmatik fayda esastır. Bu kozu kullanarak avantaj sağlamalıyız.",
        "Kural ve netlik olmadan kaos çıkar; kesin bir karar ve mutlak düzen talep ediyorum.",
        "Herhangi bir zarar, tehlike veya kayıp ihtimali bizi felakete sürükleyebilir; zararı önlemek şart.",
        "Bu dehşet verici durum karşısında öfke ve hiddetle hareket etmeyeceğiz, tam bir sükûnet ve soğukkanlılıkla yaklaşacağız.",
        "Mazlumların hakkı ve adalet her şeyin üstündedir; bireysel şefkat ve hakkaniyet vazgeçilmezimizdir."
    ]

    features = fe.extract_features(corpus)

    # 1. Machiavellianism
    assert "machiavellianism_index" in features
    mach = features["machiavellianism_index"]
    assert "score" in mach
    assert "tier" in mach
    assert mach["machiavellian_hits"] > 0
    assert "Makyavelizm" in mach["probabilistic_statement"]

    # 2. Need for Closure
    assert "need_for_closure_index" in features
    closure = features["need_for_closure_index"]
    assert "score" in closure
    assert closure["closure_hits"] > 0
    assert "Bilişsel Kapanma" in closure["probabilistic_statement"]

    # 3. Loss Aversion
    assert "loss_aversion_index" in features
    loss = features["loss_aversion_index"]
    assert "loss_to_gain_ratio" in loss
    assert loss["loss_hits"] > 0
    assert "Prospekt Risk Yönelimi" in loss["probabilistic_statement"]

    # 4. Affective Volatility
    assert "affective_volatility_index" in features
    volatility = features["affective_volatility_index"]
    assert "volatility_score" in volatility
    assert "composure_score" in volatility
    assert "Duygusal Denge" in volatility["probabilistic_statement"]

    # 5. Moral Foundations
    assert "moral_foundations_profile" in features
    moral = features["moral_foundations_profile"]
    assert "dominant_foundation" in moral
    assert moral["individualizing_hits"] > 0

    assert (tmp_path / "features.json").exists()


def test_scientific_referee_scoring_and_ranking(tmp_path: Path):
    """Test AI Peer-Review rigor scoring, filtering, and consensus pooling."""
    referee = ScientificReferee(profiles_dir=tmp_path)

    # Study 1: Large sample, peer reviewed top journal -> Gold Standard
    gold_paper = {
        "title": "Large-Scale Computational Psycholinguistics of Strategic Behavior",
        "abstract": "We analyze N = 4,500 participants and find significant correlation r = 0.44 (p < 0.001) between Machiavellianism and instrumental utility language.",
        "source": "nature",
        "sample_size": 4500,
        "r": 0.44,
        "p_value": 0.0001,
        "venue": "nature human behaviour",
        "methodology": "multicenter longitudinal cohort",
        "url": "https://doi.org/10.1038/s41562-024-001",
    }

    # Study 2: Low sample, non-significant preprint -> Low rigor / rejected
    weak_paper = {
        "title": "Exploratory pilot observation of 15 students",
        "abstract": "Small exploratory sample N = 15 shows weak r = 0.05, p = 0.42.",
        "source": "arxiv",
        "sample_size": 15,
        "r": 0.05,
        "p_value": 0.42,
        "venue": "arxiv",
        "methodology": "pilot survey",
        "url": "https://arxiv.org/abs/2401.99999",
    }

    # Score individual papers
    gold_eval = referee.score_paper_rigor(gold_paper, marker="machiavellianism_index", prior_mean_r=0.40)
    weak_eval = referee.score_paper_rigor(weak_paper, marker="machiavellianism_index", prior_mean_r=0.40)

    assert gold_eval["rigor_tier"] == "GOLD_STANDARD"
    assert gold_eval["scores"]["total_rigor_score"] >= 70.0
    assert gold_eval["scores"]["sample_size_score"] > 25.0
    assert gold_eval["scores"]["venue_score"] == 20.0

    assert weak_eval["rigor_tier"] in ("REJECTED_LOW_RIGOR", "EXPLORATORY_PREPRINT")
    assert weak_eval["scores"]["total_rigor_score"] < gold_eval["scores"]["total_rigor_score"]

    # Evaluate corpus
    corpus_result = referee.evaluate_and_rank_corpus([weak_paper, gold_paper], marker="machiavellianism_index")
    assert corpus_result["total_evaluated"] == 2
    assert corpus_result["approved_count"] >= 1
    assert len(corpus_result["gold_standard_papers"]) == 1
    assert corpus_result["gold_standard_papers"][0]["title"] == gold_paper["title"]

    # Test Knowledge Bank persistence
    kb_path = referee.maintain_knowledge_bank(
        marker="machiavellianism_index",
        evaluation_result=corpus_result,
        synthesized_boundaries={"norm": "calibrated"}
    )
    assert kb_path.exists()
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    assert "machiavellianism_index" in kb_data["metrics"]
    assert len(kb_data["metrics"]["machiavellianism_index"]["gold_standard_studies"]) == 1


def test_fisher_z_transformation():
    """Verify Fisher z transformation and inverse round-trip accuracy."""
    test_r = 0.45
    z = fisher_z(test_r)
    r_recovered = inv_fisher_z(z)
    assert round(r_recovered, 4) == test_r


def test_boundary_synthesizer_rules(tmp_path: Path):
    """Test boundary synthesis and behavioral simulation rule derivation."""
    bs = BoundarySynthesizer(profiles_dir=tmp_path)

    # 1. Decision Boundaries
    boundaries = bs.synthesize_metric_boundaries(
        marker="machiavellianism_index",
        consensus_r=0.44,
        gold_papers=[{"title": "Test"}]
    )
    assert "calibrated_distribution" in boundaries
    assert "classification_tiers" in boundaries
    assert boundaries["calibrated_distribution"]["extreme_cutoff"] > 0

    # 2. Rule Derivation for high Machiavellianism and high Need for Closure
    test_features = {
        "machiavellianism_index": {"score": 0.65, "tier": "YÜKSEK / MAKYAVELİST"},
        "need_for_closure_index": {"score": 0.58, "tier": "YÜKSEK KAPALILIK"},
        "loss_aversion_index": {"loss_to_gain_ratio": 1.85, "orientation": "KAYIPTAN KAÇINMA"},
        "affective_volatility_index": {"volatility_score": 0.15, "composure_score": 0.85},
        "moral_foundations_profile": {"dominant_foundation": "BİREY & HAKKANİYET ODAKLI (INDIVIDUALIZING_FAIRNESS)"},
        "epistemic_manipulation_index": {"score": 0.48},
    }

    rules = bs.generate_behavioral_rules(test_features)
    rule_ids = [r["rule_id"] for r in rules]

    assert "TACTICAL_TRANSACTIONALISM" in rule_ids
    assert "RIGID_COGNITIVE_CLOSURE" in rule_ids
    assert "ASYMMETRIC_LOSS_PREVENTION" in rule_ids
    assert "STOIC_EQUANIMITY" in rule_ids
    assert "INDIVIDUAL_RIGHTS_PRIMACY" in rule_ids
    assert "RHETORICAL_ASSERTION_DOMINANCE" in rule_ids

    # Save to profile
    target_dir = tmp_path / "test_target"
    out_file = bs.save_target_boundaries_and_rules(target_dir, {"mach": boundaries}, rules)
    assert out_file.exists()


def test_autonomous_researcher_cycle_integration(tmp_path: Path, monkeypatch):
    """Integration test of autonomous researcher pipeline with referee and boundaries."""
    target_dir = tmp_path / "test_profile"
    researcher = AutonomousAcademicResearcher(output_dir=target_dir, profiles_dir=tmp_path)

    # Mock external network queries for fast, deterministic testing
    monkeypatch.setattr(researcher, "query_arxiv", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_pubmed", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_semantic_scholar", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_crossref", lambda q, max_results=1: [])

    test_features = {
        "pronoun_ratios": {"first_person_singular_pct": 4.2, "first_person_plural_pct": 0.8, "i_to_we_ratio": 3.5},
        "epistemic_markers": {"epistemic_calibration_ratio": 0.45, "certainty_density": 2.5},
        "emotional_cadence": {"critical_density": 1.8},
        "epistemic_manipulation_index": {"score": 0.52},
        "chronomics": {"circadian_drift_summary": {"epistemic_filter_decay_pct": -18.0}},
        "machiavellianism_index": {"score": 0.60, "tier": "YÜKSEK / MAKYAVELİST"},
        "need_for_closure_index": {"score": 0.56, "tier": "YÜKSEK KAPALILIK"},
        "loss_aversion_index": {"loss_to_gain_ratio": 1.75},
        "affective_volatility_index": {"volatility_score": 0.48, "composure_score": 0.52},
        "moral_foundations_profile": {"dominant_foundation": "BİREY & HAKKANİYET ODAKLI"},
    }

    # 1. Anomaly detection should detect all relevant dimensions
    anomalies = researcher.detect_behavioral_anomalies(test_features, [])
    marker_names = [a["marker"] for a in anomalies]

    assert "machiavellian_strategic_orientation" in marker_names
    assert "high_need_for_cognitive_closure" in marker_names
    assert "heightened_loss_aversion_threat_vigilance" in marker_names
    assert "affective_volatility_reactivity" in marker_names
    assert "moral_foundation_individual_fairness" in marker_names

    # 2. Query generation should generate triads
    queries = researcher.generate_scientific_queries(anomalies)
    assert len(queries) == len(anomalies)
    assert len(queries[0]["queries"]) == 3

    # 3. Master research cycle
    papers = researcher.research_markers(test_features, [], max_papers_per_query=1)
    assert len(papers) > 0

    # Verify generated profile artifacts
    assert (target_dir / "retrieved_academic_literature.json").exists()
    assert (target_dir / "behavioral_boundaries.json").exists()
    assert (tmp_path / "scientific_knowledge_bank.json").exists()

    # 4. Verify skill synthesizer includes boundaries and operational rules
    ss = SkillSynthesizer(output_dir=target_dir)
    skill = ss.synthesize_mapping_skill("Test Target", test_features, [], papers)
    assert (target_dir / "generated_mapping_skill.json").exists()
    assert "extended_psychometrics" in skill["behavioral_profile"]
    assert any("[TACTICAL_TRANSACTIONALISM]" in r for r in skill["generation_rules"])
