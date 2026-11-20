"""
Tests for Module 4: who_engine (Oracle, Simulator, Composer, Psych Insights, Explainability, Evaluator, Twin Chat).
Includes tests for Temporal Holdout Brier Score Backtesting, Stylometric JSD Verification,
Context-Conditional CAPS Matrix, Dual-Layer Stated vs. Revealed Preference Simulation,
and Boltzmann-Gibbs Statistical Mechanics / Quantal Response Equilibrium (QRE).
"""

import json
from pathlib import Path
import pytest

from who.config import settings
from who.engine.composer import Composer
from who.engine.evaluator import ProfileEvaluator, compute_jensen_shannon_divergence
from who.engine.explainability_engine import ExplainabilityEngine
from who.engine.oracle import Oracle
from who.engine.psych_insights import PsychInsights
from who.engine.simulator import Simulator, compute_boltzmann_gibbs_distribution
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


def test_boltzmann_gibbs_statistical_mechanics():
    utilities = {
        "REJECT_AND_COUNTER": 0.75,
        "STRATEGIC_CONCESSION": 0.30,
        "SUBGAME_EQUILIBRIUM_ACCEPT": 0.45,
        "DEADLOCK_WALKAWAY": 0.15,
    }
    low_beta = compute_boltzmann_gibbs_distribution(utilities, beta=0.8)
    high_beta = compute_boltzmann_gibbs_distribution(utilities, beta=8.0)

    assert sum(low_beta["probabilities"].values()) == pytest.approx(1.0, abs=0.005)
    assert sum(high_beta["probabilities"].values()) == pytest.approx(1.0, abs=0.005)
    # Higher inverse temperature beta (lower cognitive temperature T) concentrates probability on the optimal utility branch
    assert high_beta["probabilities"]["REJECT_AND_COUNTER"] > low_beta["probabilities"]["REJECT_AND_COUNTER"]
    # And lowers Shannon decision entropy S
    assert high_beta["shannon_entropy_S"] < low_beta["shannon_entropy_S"]


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
    assert "statistical_mechanics" in res
    assert "inverse_temperature_beta" in res["statistical_mechanics"]
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
    assert compute_jensen_shannon_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0
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


def test_twin_chat_dynamic_character_voice(target_dirs):
    casual_dirs = settings.get_target_dirs("CasualPersona")
    casual_character = {
        "name": "CasualPersona",
        "bio": ["Günlük konuşma dilini kullanan sentetik test profili."],
        "lore": ["Kısa ve samimi mesajlaşma ritmiyle yanıt verir."],
        "adjectives": ["casual", "concise"],
        "style": {
            "all": ["küçük harfli samimi Türkçe yazım", "sıfır ünlem işareti"],
            "chat": ["kısa mesaj patlamaları", "doğal sohbet tonu"],
        },
        "postExamples": ["kahve icmeye cikicam birazdan hava baya guzel"],
        "messageExamples": [
            [
                {"user": "{{user1}}", "content": {"text": "yarınki toplantı kaçta başlıyor"}},
                {"user": "CasualPersona", "content": {"text": "sanirim 10 gibi basliyo ama tekrar bakarim birazdan"}},
            ]
        ],
    }
    (casual_dirs.vault_dir / "character.json").write_text(
        json.dumps(casual_character, ensure_ascii=False), encoding="utf-8"
    )
    (casual_dirs.raw_data_dir / "doc1.md").write_text(
        "kahve icmeye cikicam birazdan hava baya guzel\n\nsanirim 10 gibi basliyo ama tekrar bakarim birazdan",
        encoding="utf-8",
    )

    chat_casual = TwinChat("CasualPersona", casual_dirs, mode="autodetect")
    msg = "yarın akşamki etkinlik için saat kaçta buluşsak daha iyi olur"
    prompt = chat_casual.build_system_prompt(msg)

    assert "Efendim, haklısınız" not in prompt
    assert "Şöyle izah edeyim" not in prompt
    assert "vakarını" not in prompt
    assert "küçük harfli samimi Türkçe yazım" in prompt
    assert "kahve icmeye cikicam birazdan hava baya guzel" in prompt
    assert "sanirim 10 gibi basliyo ama tekrar bakarim birazdan" in prompt

    assert chat_casual._resolve_tone(msg) == "informal"
    resp = chat_casual.respond(msg)
    assert len(resp) > 0
    assert chat_casual.last_turn_diagnostics["active_tone"] == "informal"

    chat_formal = TwinChat("Alice", target_dirs, mode="autodetect")
    assert (
        chat_formal._resolve_tone("Could you present your theoretical position on verifiable systems?")
        == "formal"
    )


def test_simulator_monte_carlo(target_dirs):
    sim = Simulator("Alice", target_dirs)
    res = sim.simulate_monte_carlo("Counterparty demands unilateral board veto under 48h deadline", iterations=15, random_seed=42)

    assert "dominant_branch" in res
    assert "branch_probabilities" in res
    assert "boltzmann_branch_probabilities" in res
    assert len(res["branch_probabilities"]) == 4
    assert sum(res["branch_probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert sum(res["boltzmann_branch_probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert "statistical_mechanics" in res
    assert "shannon_decision_entropy_S" in res["statistical_mechanics"]
    assert "helmholtz_free_energy_F" in res["statistical_mechanics"]
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
