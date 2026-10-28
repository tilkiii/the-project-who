"""
Tests for Module 4: who_engine (Oracle, Simulator, Composer, Psych Insights, Explainability, Evaluator, Twin Chat).
"""

from pathlib import Path
import pytest

from src.config import settings
from src.engine.composer import Composer
from src.engine.evaluator import ProfileEvaluator
from src.engine.explainability_engine import ExplainabilityEngine
from src.engine.oracle import Oracle
from src.engine.psych_insights import PsychInsights
from src.engine.simulator import Simulator
from src.engine.twin_chat import TwinChat


@pytest.fixture
def target_dirs(tmp_path: Path):
    dirs = settings.get_target_dirs("Alice")
    # Write sample mock files
    (dirs.vault_dir / "mindset_summary.md").write_text("# Mindset\nEmpirical, calibrated.", encoding="utf-8")
    (dirs.vault_dir / "character.json").write_text('{"name": "Alice", "bio": ["Calibrated researcher"], "postExamples": ["Empirical evidence is crucial."]}', encoding="utf-8")
    (dirs.learned_skills_dir / "features.json").write_text('{"epistemic_markers": {"epistemic_calibration_ratio": 1.2}, "top_characteristic_terms": ["consensus", "evidence"]}', encoding="utf-8")
    (dirs.learned_skills_dir / "retrieved_academic_literature.json").write_text('[{"title": "Cognitive Reflection in Linguistics", "doi_or_url": "https://doi.org/10.1037", "repository": "Psychology"}]', encoding="utf-8")
    (dirs.raw_data_dir / "doc1.md").write_text("Empirical evidence is crucial for Byzantine agreement.", encoding="utf-8")
    return dirs


def test_oracle(target_dirs):
    oracle = Oracle("Alice", target_dirs)
    res = oracle.ask("What is your view on decentralized governance?")

    assert "answer" in res
    assert "confidence_score" in res
    assert res["confidence_score"] > 0.0


def test_simulator(target_dirs):
    sim = Simulator("Alice", target_dirs)
    res = sim.simulate_scenario("Proposing a closed-source proprietary governance council")

    assert "simulation_analysis" in res
    assert "confidence_score" in res
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


def test_profile_evaluator(target_dirs):
    evaluator = ProfileEvaluator("Alice", target_dirs)
    res = evaluator.evaluate_profile()

    assert "overall_evaluation_score" in res
    assert "metrics" in res
    assert "citation_validity" in res["metrics"]
    assert "profile_consistency" in res["metrics"]
    assert "hallucination_rate" in res["metrics"]


def test_twin_chat(target_dirs):
    chat = TwinChat("Alice", target_dirs, mode="formal")
    resp_formal = chat.respond("Could you present your theoretical position on verifiable systems?")
    assert len(resp_formal) > 0

    chat_informal = TwinChat("Alice", target_dirs, mode="informal")
    resp_informal = chat_informal.respond("Hey Alice, what's up?")
    assert len(resp_informal) > 0
