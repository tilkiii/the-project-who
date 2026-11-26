"""
Milestone 3 verification suite:
1. Honest psycholinguistics (lexical indicator framing, INSUFFICIENT_DATA thresholds,
   morphological stemming, and privative suffix guards).
2. Strict 3-tier academic grounding (QUANTITATIVE_EXTRACTED vs QUALITATIVE_ABSTRACT_MATCH
   vs REFERENCE_BASELINE_PRIOR, zero invented r/N/DOI/citation counts).
3. Corpus- and knowledge-graph-conditioned scenario simulation with explicit provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from who.engine.simulator import Simulator
from who.learn.autonomous_researcher import AutonomousAcademicResearcher
from who.learn.feature_extractor import FeatureExtractor
from who.learn.scientific_referee import ScientificReferee


def test_feature_extractor_insufficient_data_and_lexical_framing(tmp_path: Path):
    """Short or neutral corpora must emit INSUFFICIENT_DATA instead of confident trait scores."""
    fe = FeatureExtractor(output_dir=tmp_path)

    short_feats = fe.extract_features(["Hello world today."])
    meth = short_feats["measurement_methodology"]
    assert meth["signal_type"] == "LEXICAL_BEHAVIORAL_INDICATOR"
    assert meth["clinical_diagnostic_validity"] is False
    assert meth["corpus_sufficiency"] == "INSUFFICIENT_DATA"

    assert short_feats["epistemic_manipulation_index"]["measurement_type"] == "lexical_indicator"
    assert short_feats["epistemic_manipulation_index"]["sufficient_signal"] is False
    assert short_feats["epistemic_manipulation_index"]["risk_tier"] == "INSUFFICIENT_DATA"
    assert short_feats["epistemic_manipulation_index"]["confidence_score"] <= 0.35

    assert short_feats["machiavellianism_index"]["tier"] == "INSUFFICIENT_DATA"
    assert short_feats["need_for_closure_index"]["tier"] == "INSUFFICIENT_DATA"
    assert short_feats["loss_aversion_index"]["orientation"] == "INSUFFICIENT_DATA"
    assert short_feats["loss_aversion_index"]["loss_to_gain_ratio"] == 1.0
    assert short_feats["affective_volatility_index"]["tier"] == "INSUFFICIENT_DATA"
    assert short_feats["moral_foundations_profile"]["dominant_foundation"] == "INSUFFICIENT_DATA"

    # Empty corpus matches same clinical non-diagnostic contract
    empty_feats = fe.extract_features([])
    assert empty_feats["measurement_methodology"]["corpus_sufficiency"] == "INSUFFICIENT_DATA"
    assert empty_feats["measurement_methodology"]["clinical_diagnostic_validity"] is False
    assert empty_feats["loss_aversion_index"]["loss_to_gain_ratio"] == 1.0

    # Longer signal corpus triggers Lexical Indicator framing
    rich_feats = fe.extract_features([
        "Strategic leverage, pragmatic utility, and careful risk management shape our long-term plan. "
        "Yet cooperation, fairness, empathy, and transparent evidence suggest balanced collaboration across partners."
    ])
    assert "(Lexical Indicator)" in rich_feats["machiavellianism_index"]["probabilistic_statement"]
    assert rich_feats["machiavellianism_index"]["sufficient_signal"] is True


def test_feature_extractor_morphological_stems_and_privative_guards():
    """Verify Turkish/English inflections match roots while privatives ('adaletsiz', 'fearless') do not."""
    fe = FeatureExtractor()

    inflected_loss = fe.extract_features([
        "Repeated felakete, serious tehditler, severe threats, and heavy losses demand protective caution."
    ])
    assert inflected_loss["loss_aversion_index"]["loss_hits"] >= 4

    inflected_coop = fe.extract_features([
        "Stronger işbirliğini, lasting fırsatlar, mutual opportunities, and cooperating teams built value."
    ])
    assert inflected_coop["machiavellianism_index"]["cooperation_hits"] >= 2
    assert inflected_coop["loss_aversion_index"]["gain_hits"] >= 2

    privative_text = fe.extract_features([
        "Bu durum adaletsiz, güvensiz, risksiz ve zararsız; they remain fearless, harmless, and careless."
    ])
    assert privative_text["moral_foundations_profile"]["individualizing_hits"] == 0
    assert privative_text["machiavellianism_index"]["cooperation_hits"] == 0
    assert privative_text["loss_aversion_index"]["loss_hits"] == 0


def test_academic_researcher_three_tier_evidence_separation(tmp_path: Path):
    """Separate QUANTITATIVE_EXTRACTED, QUALITATIVE_ABSTRACT_MATCH, and REFERENCE_BASELINE_PRIOR."""
    agent = AutonomousAcademicResearcher(output_dir=tmp_path, profiles_dir=tmp_path / "profiles")

    quant_paper = {
        "title": "Self-Referential Pronouns and Neuroticism",
        "year": "2024",
        "doi": "DOI:10.1016/j.chb.2024.108123",
        "doi_or_url": "https://doi.org/10.1016/j.chb.2024.108123",
        "abstract": "In a sample of N=1850 participants, I-talk correlated with neuroticism (r = 0.41, p < 0.001).",
        "repository": "Computers in Human Behavior",
        "query_matched": "digital phenotyping first person singular pronouns",
    }
    qual_paper = {
        "title": "Theoretical Reflections on Narrative Pronouns",
        "year": "2023",
        "doi": "arXiv:2309.00001",
        "doi_or_url": "https://arxiv.org/abs/2309.00001",
        "abstract": "We discuss conceptual theories of self-reference in philosophy without reporting any correlation or cohort size.",
        "repository": "arXiv",
        "query_matched": "digital phenotyping first person singular pronouns",
    }
    anomalies = [
        {
            "marker": "high_pronoun_density_I",
            "phenomenon": "Self-focus",
            "academic_keywords": ["pronouns", "neuroticism"],
        }
    ]

    findings = agent.extract_metrics_from_papers([quant_paper, qual_paper], anomalies)
    assert len(findings) == 2

    f_quant, f_qual = findings[0], findings[1]
    assert f_quant["evidence_type"] == "QUANTITATIVE_EXTRACTED"
    assert f_quant["eligible_for_bayesian_update"] is True
    assert abs(f_quant["correlation_coefficient_r"] - 0.41) < 0.05
    assert f_quant["sample_size_n"] == 1850

    assert f_qual["evidence_type"] == "QUALITATIVE_ABSTRACT_MATCH"
    assert f_qual["eligible_for_bayesian_update"] is False
    assert f_qual["correlation_coefficient_r"] is None
    assert f_qual["sample_size_n"] is None

    # Qualitative findings must be skipped by Bayesian weight updater
    catalog = agent.update_scientific_weights([f_qual])
    assert catalog["meta"]["skipped_qualitative_findings"] == 1

    # Canonical benchmarks must be tagged as reference baseline priors
    benchmarks = agent._canonical_psychological_benchmarks()
    assert benchmarks
    assert all(b["evidence_type"] == "REFERENCE_BASELINE_PRIOR" for b in benchmarks)
    assert all(b["is_reference_prior"] is True for b in benchmarks)

    # ScientificReferee must not fabricate r=0.25 or N on qualitative abstracts
    ref = ScientificReferee(profiles_dir=tmp_path / "profiles")
    scored_qual = ref.score_paper_rigor(qual_paper, marker="high_pronoun_density_I", prior_mean_r=0.42)
    assert scored_qual["inferred_r"] is None
    assert scored_qual["effect_size_imputed"] is True
    assert scored_qual["evidence_type"] == "QUALITATIVE_ABSTRACT_MATCH"

    ranked = ref.evaluate_and_rank_corpus([qual_paper], marker="high_pronoun_density_I", prior_mean_r=0.42)
    assert ranked["consensus_source"] == "REFERENCE_BASELINE_PRIOR"
    assert ranked["observed_r_studies"] == 0


def _make_profile_dirs(base: Path, name: str) -> SimpleNamespace:
    root = base / name
    dirs = SimpleNamespace(
        base_dir=root,
        vault_dir=root / "vault",
        learned_skills_dir=root / "learned_skills",
        raw_data_dir=root / "raw_data",
        chroma_dir=root / "chroma",
    )
    for d in (dirs.vault_dir, dirs.learned_skills_dir, dirs.raw_data_dir, dirs.chroma_dir):
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def test_simulator_corpus_and_kg_binding_with_transparency(tmp_path: Path):
    """Distinct observed corpora and KG relations must shift simulation distributions and surface provenance."""
    def_dirs = _make_profile_dirs(tmp_path, "defensive_target")
    (def_dirs.learned_skills_dir / "features.json").write_text(
        json.dumps({
            "corpus_stats": {"total_words": 450},
            "loss_aversion_index": {"estimated_score": 2.25},
            "machiavellianism_index": {"estimated_score": 0.78},
            "need_for_closure_index": {"estimated_score": 0.72},
            "affective_volatility_index": {"volatility_score": 0.25},
            "epistemic_markers": {"epistemic_calibration_ratio": 0.45},
            "epistemic_manipulation_index": {"score": 0.60},
            "moral_foundations_profile": {"dominant_foundation": "BINDING_AUTHORITY"},
            "stated_vs_revealed_preference": {
                "stated_cooperation_polarity": 0.25,
                "revealed_cooperation_polarity": 0.15,
            },
        }),
        encoding="utf-8",
    )
    (def_dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [{"name": "Algorithmic Governance", "type": "Concept"}],
            "relationships": [
                {"source": "defensive_target", "target": "Algorithmic Governance", "relation": "CRITIQUES"},
                {"source": "Algorithmic Governance", "target": "Centralized Control", "relation": "CONTRADICTS"},
            ],
        }),
        encoding="utf-8",
    )

    coop_dirs = _make_profile_dirs(tmp_path, "cooperative_target")
    (coop_dirs.learned_skills_dir / "features.json").write_text(
        json.dumps({
            "corpus_stats": {"total_words": 450},
            "loss_aversion_index": {"estimated_score": 0.90},
            "machiavellianism_index": {"estimated_score": 0.10},
            "need_for_closure_index": {"estimated_score": 0.28},
            "affective_volatility_index": {"volatility_score": 0.10},
            "epistemic_markers": {"epistemic_calibration_ratio": 1.85},
            "epistemic_manipulation_index": {"score": 0.12},
            "moral_foundations_profile": {"dominant_foundation": "INDIVIDUALIZING_FAIRNESS"},
            "stated_vs_revealed_preference": {
                "stated_cooperation_polarity": 0.85,
                "revealed_cooperation_polarity": 0.88,
            },
        }),
        encoding="utf-8",
    )
    (coop_dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [{"name": "Algorithmic Governance", "type": "Concept"}],
            "relationships": [
                {"source": "cooperative_target", "target": "Algorithmic Governance", "relation": "ADVOCATES"},
                {"source": "Algorithmic Governance", "target": "Open Audits", "relation": "SUPPORTS"},
            ],
        }),
        encoding="utf-8",
    )

    fixed_ranges = {
        "counterparty_leverage": (0.45, 0.45),
        "deadline_pressure_days": (7.0, 7.0),
        "information_asymmetry": (0.30, 0.30),
        "reputation_risk": (0.30, 0.30),
    }
    scenario = "Proposal for joint Algorithmic Governance parity council"

    def_res = Simulator("defensive_target", def_dirs).simulate_monte_carlo(
        scenario, iterations=10, parameter_ranges=fixed_ranges, random_seed=42
    )
    coop_res = Simulator("cooperative_target", coop_dirs).simulate_monte_carlo(
        scenario, iterations=10, parameter_ranges=fixed_ranges, random_seed=42
    )

    assert coop_res["branch_probabilities"]["SUBGAME_EQUILIBRIUM_ACCEPT"] > def_res["branch_probabilities"]["SUBGAME_EQUILIBRIUM_ACCEPT"]
    assert (
        def_res["branch_probabilities"]["REJECT_AND_COUNTER"] + def_res["branch_probabilities"]["DEADLOCK_WALKAWAY"]
        > coop_res["branch_probabilities"]["REJECT_AND_COUNTER"] + coop_res["branch_probabilities"]["DEADLOCK_WALKAWAY"]
    )

    for res in (def_res, coop_res):
        tr = res["model_transparency"]
        assert tr["model_type"] == "corpus_conditioned_scenario_model"
        assert tr["is_trained_behavioral_predictor"] is False
        assert tr["corpus_grounded"] is True
        assert "Algorithmic Governance" in tr["matched_knowledge_graph_anchors"]
        assert tr["corpus_field_source"] == "observed_features_and_graph"

    # Empty ghost profile must honestly disclose default priors
    empty_dirs = _make_profile_dirs(tmp_path, "empty_ghost")
    empty_res = Simulator("empty_ghost", empty_dirs).simulate_scenario(scenario)
    assert empty_res["model_transparency"]["corpus_grounded"] is False
    assert "default_prior_fallback" in empty_res["model_transparency"]["parameter_provenance"].values()
