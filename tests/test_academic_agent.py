"""
Tests for Layer 2: Autonomous Academic Research Agent (who_learn).
Tests scoping, dynamic query generation, statistical metric extraction,
and Bayesian updating with Fisher's z-transformation.
"""

from pathlib import Path
import json
import pytest

from who.learn.autonomous_researcher import (
    AutonomousAcademicResearcher,
    AcademicResearcher,
    fisher_z,
    inv_fisher_z,
)


def test_fisher_z_transformation():
    """Verify bidirectional mathematical fidelity of Fisher's r-to-z transformation."""
    r_val = 0.42
    z_val = fisher_z(r_val)
    r_rec = inv_fisher_z(z_val)
    assert abs(r_rec - r_val) < 1e-4

    # Extreme bounds test
    assert abs(fisher_z(0.999)) < 10.0
    assert abs(fisher_z(-0.999)) < 10.0


def test_anomaly_detection_and_query_generation(tmp_path: Path):
    """Test anomaly detection from linguistic features and generation of academic queries."""
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=tmp_path / "profiles")

    features = {
        "pronoun_ratios": {
            "first_person_singular_pct": 4.5,
            "first_person_plural_pct": 1.2,
            "i_to_we_ratio": 3.75,
        },
        "epistemic_markers": {
            "epistemic_calibration_ratio": 0.35,
            "certainty_density": 2.8,
        },
        "chronomics": {
            "circadian_drift_summary": {
                "epistemic_filter_decay_pct": 28.0,
                "lexical_diversity_decay_pct": 12.0,
            }
        },
        "epistemic_manipulation_index": {
            "score": 0.65,
            "risk_tier": "YÜKSEK (HIGH_MANIPULATION_RISK)",
        },
    }
    topics = [{"name": "Institutional Power", "keywords": ["masonluk", "hiyerarsi"]}]

    anomalies = agent.detect_behavioral_anomalies(features, topics)
    markers = [a["marker"] for a in anomalies]

    assert "high_pronoun_density_I" in markers
    assert "epistemic_overconfidence_dogmatism" in markers
    assert "circadian_cognitive_drift_night" in markers
    assert "epistemic_gaslighting_dogmatism" in markers

    # Verify query generation
    query_objects = agent.generate_scientific_queries(anomalies)
    assert len(query_objects) >= 3

    all_queries = []
    for qo in query_objects:
        all_queries.extend(qo["queries"])

    # Check for expected academic query prefixes
    assert any("digital phenotyping" in q for q in all_queries)
    assert any("computational personality recognition" in q for q in all_queries)


def test_statistical_information_extraction(tmp_path: Path):
    """Test heuristic and structured extraction of correlation coefficients and sample sizes."""
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=tmp_path / "profiles")

    sample_papers = [
        {
            "title": "Computational Analysis of First-Person Singular Pronouns in Emotional Distress",
            "year": "2024",
            "doi": "DOI:10.1016/j.chb.2024.108123",
            "doi_or_url": "https://doi.org/10.1016/j.chb.2024.108123",
            "abstract": (
                "In a large-scale corpus of 1,850 participants (N=1850), we found that elevated first-person "
                "pronoun usage exhibited a statistically significant correlation with neuroticism (r = 0.41, p < 0.001). "
                "Methodology relied on natural language processing and cross-sectional psychometric questionnaires."
            ),
            "repository": "Computers in Human Behavior",
            "query_matched": "digital phenotyping first person singular pronouns",
        }
    ]

    anomalies = [{"marker": "high_pronoun_density_I", "phenomenon": "Self-focus", "academic_keywords": ["pronouns", "neuroticism"]}]
    findings = agent.extract_metrics_from_papers(sample_papers, anomalies)

    assert len(findings) >= 1
    f = findings[0]
    assert f["behavioral_marker"] == "high_pronoun_density_I"
    assert abs(f["correlation_coefficient_r"] - 0.41) < 0.05
    assert "<" in f["p_value"] or "0.001" in f["p_value"]
    assert f["sample_size_n"] >= 1000
    assert "DOI:" in f["evidence_source"]
    assert "Ajanın Bilimsel Muhakeme Notu" in f["agent_reasoning_note"]


def test_bayesian_weight_updating_and_persistence(tmp_path: Path):
    """Test Bayesian updating using Fisher's z pooling and persistence in scientific_weights.json."""
    profiles_dir = tmp_path / "profiles"
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=profiles_dir)

    # 1. Initial State has benchmark weight for high_pronoun_density_I
    initial_weights = agent._load_current_weights()["calibrated_weights"]
    prior_entry = initial_weights["high_pronoun_density_I::Neuroticism"]
    prior_r = prior_entry["correlation_coefficient_r"]
    prior_n = prior_entry["sample_size_total"]

    # 2. Simulate new empirical finding with r=0.36, N=1500
    new_findings = [
        {
            "behavioral_marker": "high_pronoun_density_I",
            "correlated_psychometric": "Neuroticism",
            "correlation_coefficient_r": 0.36,
            "p_value": "< 0.001",
            "sample_size_n": 1500,
            "evidence_source": "DOI:10.1037/xge0009999",
            "doi_url": "https://doi.org/10.1037/xge0009999",
            "confidence_score": 0.90,
            "agent_reasoning_note": "Ajanın Bilimsel Muhakeme Notu: N=1500 çalışmada r=0.36 bulunmuştur.",
        }
    ]

    updated_catalog = agent.update_scientific_weights(new_findings)
    calibrated_entry = updated_catalog["calibrated_weights"]["high_pronoun_density_I::Neuroticism"]

    # Verify Bayesian updating
    assert calibrated_entry["sample_size_total"] == prior_n + 1500
    assert calibrated_entry["bayesian_updates_count"] >= 2
    # The updated r should be pooled between 0.36 and 0.42
    assert 0.36 <= calibrated_entry["correlation_coefficient_r"] <= 0.42
    assert "DOI:10.1037/xge0009999" in calibrated_entry["evidence_sources"]
    assert "Bayesyen Fisher-z" in calibrated_entry["agent_reasoning_note"]

    # Verify JSON files are persisted
    assert (profiles_dir / "scientific_weights.json").exists()
    assert (tmp_path / "scientific_weights.json").exists()

    # Verify schema compliance
    with open(profiles_dir / "scientific_weights.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "calibrated_weights" in data
    assert "meta" in data


def test_academic_researcher_canonical_exports():
    """Verify that canonical imports from who.learn are completely functional."""
    from who.learn.autonomous_researcher import (
        AutonomousAcademicResearcher,
        AcademicResearcher,
        fisher_z,
        inv_fisher_z,
    )

    assert AutonomousAcademicResearcher is not None
    assert AcademicResearcher is not None
    assert callable(fisher_z)
    assert callable(inv_fisher_z)


def test_research_topic_and_literature_review(tmp_path: Path):
    """Test topic-based academic literature retrieval and review synthesis."""
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=tmp_path / "profiles")
    res = agent.research_topic("epistemic calibration", sources=["arxiv"], limit=1)

    assert "papers" in res
    assert "literature_review_markdown" in res
    assert len(res["papers"]) >= 1
    assert "Scientific Literature Review" in res["literature_review_markdown"]


def test_ingest_academic_corpus(tmp_path: Path):
    """Test ingesting retrieved academic papers into a target profile's raw_data corpus."""
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=tmp_path / "profiles")
    sample_papers = [{
        "title": "Quantum Game Theory and Social Dilemmas",
        "repository": "arXiv",
        "year": "2024",
        "doi": "arXiv:2401.99999",
        "doi_or_url": "https://arxiv.org/abs/2401.99999",
        "abstract": "We explore quantum strategies in non-cooperative games.",
        "statistical_metrics": {"correlation_coefficient_r": 0.45, "p_value": 0.001, "sample_size_n": 500},
        "agent_reasoning_note": "Confirms quantum coordination advantage.",
    }]

    files = agent.ingest_academic_corpus("test_scholar", sample_papers)
    assert len(files) == 1
    p = Path(files[0])
    assert p.exists()
    assert "Quantum Game Theory" in p.read_text(encoding="utf-8")


def test_cli_research_command():
    """Verify who research CLI subcommand execution."""
    from typer.testing import CliRunner
    from who.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["research", "--topic", "epistemic humility", "--limit", "1", "--sources", "arxiv"])
    assert result.exit_code == 0
    assert "Peer-Reviewed Literature" in result.output
