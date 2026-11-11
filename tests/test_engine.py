"""
Tests for Module 4: who_engine (Oracle, Simulator, Composer, Psych Insights, Explainability, Evaluator, Twin Chat).
Includes tests for Temporal Holdout Brier Score Backtesting, Stylometric JSD Verification,
Context-Conditional CAPS Matrix, and Dual-Layer Stated vs. Revealed Preference Simulation.
"""

from pathlib import Path
import pytest

from who.config import settings
from who.engine.composer import Composer
from who.engine.evaluator import ProfileEvaluator, compute_jensen_shannon_divergence
from who.engine.explainability_engine import ExplainabilityEngine
from who.engine.oracle import Oracle
from who.engine.psych_insights import PsychInsights
from who.engine.simulator import Simulator
from who.engine.twin_chat import TwinChat


@pytest.fixture
def target_dirs(tmp_path: Path):
    dirs = settings.get_target_dirs("Alice")
    (dirs.vault_dir / "mindset_summary.md").write_text("# Mindset\nEmpirical, calibrated.", encoding="utf-8")
    (dirs.vault_dir / "character.json").write_text('{"name": "Alice", "bio": ["Calibrated researcher"], "postExamples": ["Empirical evidence is crucial."]}', encoding="utf-8")
    (dirs.learned_skills_dir / "features.json").write_text('{"epistemic_markers": {"epistemic_calibration_ratio": 1.2}, "top_characteristic_terms": ["consensus", "evidence"]}', encoding="utf-8")
    (dirs.learned_skills_dir / "retrieved_academic_literature.json").write_text('[{"title": "Cognitive Reflection in Linguistics", "doi_or_url": "https://doi.org/10.1037", "repository": "Psychology"}]', encoding="utf-8")
    (dirs.raw_data_dir / "doc1.md").write_text(
        "Empirical evidence is crucial for Byzantine agreement and probable consensus.\n\n"
        "We suggest a plausible framework for cooperative governance and transparent dialogue.\n\n"
        "Perhaps probabilistic verification suggests better institutional alignment.\n\n"
        "Collaborative research indicates that dialogue and consensus remain essential.",
        encoding="utf-8",
    )
    return dirs


def test_oracle(target_dirs):
    oracle = Oracle("Alice", target_dirs)
    res = oracle.ask("What is your view on decentralized governance?")

    assert "answer" in res
    assert "confidence_score" in res
    assert res["confidence_score"] > 0.0


def test_simulator(target_dirs):
    sim = Simulator("Alice", target_dirs)
    res = sim.simulate_scenario("Proposing a closed-source proprietary governance council during an urgent crisis")

    assert "simulation_analysis" in res
    assert "confidence_score" in res
    assert "caps_situational_context" in res
    assert res["caps_situational_context"]["active_context"] == "HIGH_STAKES_CRISIS"
    assert "dual_layer_prediction" in res
    assert "stated_public_reaction" in res["dual_layer_prediction"]
    assert "revealed_strategic_action" in res["dual_layer_prediction"]
    assert "calibration_metrics" in res
    assert "brier_score" in res["calibration_metrics"]
    assert "epistemic_qualification" in res


def test_composer(target_dirs):
    composer = Composer("Alice", target_dirs)
    res = composer.compose_outreach("Request a technical collaboration on verifiable proofs")

    assert "initial_draft" in res
    assert "adversarial_rejection_analysis" in res
    assert "optimized_final_pitch" in res


def test_psych_insights(target_dirs):
    pi = PsychInsights("Alice", target_dirs)
    res = pi.generate_report()

    assert "report_markdown" in res
    assert "academic_citations" in res
    assert len(res["academic_citations"]) > 0


def test_explainability_engine(target_dirs):
    ee = ExplainabilityEngine("Alice", target_dirs)
    res = ee.explain_claim("Determine exactly how the target thinks about centralized control")

    assert "infer likely reasoning patterns" in res["calibrated_probabilistic_statement"]
    assert "confidence_score" in res
    assert "required_phrasing" in res


def test_profile_evaluator_and_temporal_backtest(target_dirs):
    evaluator = ProfileEvaluator("Alice", target_dirs)
    res = evaluator.evaluate_profile()

    assert "overall_evaluation_score" in res
    assert "metrics" in res
    assert "citation_validity" in res["metrics"]
    assert "profile_consistency" in res["metrics"]
    assert "hallucination_rate" in res["metrics"]
    assert "temporal_holdout_backtest" in res["metrics"]

    bt = evaluator.run_temporal_backtest(holdout_ratio=0.25)
    assert "brier_score" in bt
    assert 0.0 <= bt["brier_score"] <= 1.0
    assert "empirical_confidence_score" in bt
    assert bt["empirical_confidence_score"] == pytest.approx(max(0.50, min(0.98, 1.0 - bt["brier_score"])), abs=0.01)


def test_stylometric_jsd_and_referee_gate(target_dirs):
    # Identical distributions must yield JSD == 0.0
    assert compute_jensen_shannon_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0
    # Divergent distributions must yield JSD > 0.0
    assert compute_jensen_shannon_divergence([0.9, 0.05, 0.05], [0.05, 0.05, 0.9]) > 0.2

    evaluator = ProfileEvaluator("Alice", target_dirs)
    audit = evaluator.verify_stylometric_and_epistemic_fidelity(
        "Empirical evidence is crucial for probable consensus and transparent governance."
    )
    assert "stylometric_jsd" in audit
    assert "stylometric_fidelity_score" in audit
    assert "evidence_ledger" in audit
    assert len(audit["evidence_ledger"]) >= 1
    assert audit["referee_gate_passed"] is True


def test_twin_chat(target_dirs):
    chat = TwinChat("Alice", target_dirs, mode="formal")
    resp_formal = chat.respond("Could you present your theoretical position on verifiable systems?")
    assert len(resp_formal) > 0
    assert "stylometric_jsd" in chat.last_turn_diagnostics
    assert "active_caps_context" in chat.last_turn_diagnostics

    chat_informal = TwinChat("Alice", target_dirs, mode="informal")
    resp_informal = chat_informal.respond("Hey Alice, what's up?")
    assert len(resp_informal) > 0


def test_simulator_monte_carlo(target_dirs):
    sim = Simulator("Alice", target_dirs)
    res = sim.simulate_monte_carlo("Counterparty demands unilateral board veto under 48h deadline", iterations=15, random_seed=42)

    assert "dominant_branch" in res
    assert "branch_probabilities" in res
    assert len(res["branch_probabilities"]) == 4
    assert sum(res["branch_probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert "expected_nash_payoff" in res
    assert "critical_sensitivity" in res
    assert "caps_situational_context" in res
    assert "dual_layer_prediction" in res
    assert "calibration_metrics" in res
    assert "strategic_synthesis" in res
    assert res["dominant_branch"] in [
        "REJECT_AND_COUNTER",
        "STRATEGIC_CONCESSION",
        "SUBGAME_EQUILIBRIUM_ACCEPT",
        "DEADLOCK_WALKAWAY",
    ]
