"""
Unit and integration tests for who?? Personal Agent SDK.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.agent.sdk import (
    AcademicResearcher,
    CognitiveTwin,
    get_target,
    list_targets,
)
from src.config import settings


@pytest.fixture
def populated_target():
    target = "test_target"
    dirs = settings.get_target_dirs(target)
    dirs.ensure_exists()

    # Raw doc
    (dirs.raw_data_dir / "statement.md").write_text("# Statement\nEmpirical observation is paramount.", encoding="utf-8")

    # Vault files
    (dirs.vault_dir / "mindset_summary.md").write_text("# Mindset\nRigorous and probabilistic.", encoding="utf-8")
    (dirs.vault_dir / "knowledge_graph.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    (dirs.vault_dir / "character.json").write_text(json.dumps({
        "name": "Test Target",
        "bio": ["Probabilistic empirical researcher"],
        "style": {"all": ["probabilistic", "evidence-based"]}
    }), encoding="utf-8")

    # Learned skills
    skill_file = dirs.learned_skills_dir / "generated_mapping_skill.json"
    skill_file.write_text(json.dumps({
        "behavioral_profile": {
            "extended_psychometrics": {
                "machiavellianism": {"score": 0.8, "tier": "STRATEGIC"}
            }
        },
        "empirical_decision_boundaries": {},
        "generation_rules": [
            "[TACTICAL_TRANSACTIONALISM] Hedef fayda optimizasyonu yapar."
        ]
    }), encoding="utf-8")

    return target


def test_list_targets(populated_target):
    targets = list_targets()
    assert isinstance(targets, list)
    assert len(targets) > 0
    slugs = [t["slug"] for t in targets]
    assert populated_target in slugs


def test_cognitive_twin_get_profile(populated_target):
    twin = CognitiveTwin(populated_target)
    profile = twin.get_profile()

    assert profile["name"] == populated_target
    assert "psychometrics" in profile
    assert "boundaries" in profile
    assert "operational_rules" in profile
    assert len(profile["operational_rules"]) > 0

    rules = twin.get_operational_rules()
    assert isinstance(rules, list)
    assert any("[TACTICAL_TRANSACTIONALISM]" in r for r in rules)


def test_cognitive_twin_chat(populated_target):
    twin = CognitiveTwin(populated_target)
    reply = twin.chat("Merhaba, nasılsınız?")
    assert isinstance(reply, str)
    assert len(reply) > 15
    assert "Corpus Kanıt Analizi" not in reply
    assert "Aboneliğiniz başarıyla tamamlandı" not in reply


def test_cognitive_twin_simulate_scenario(populated_target):
    twin = CognitiveTwin(populated_target)
    sim = twin.simulate_scenario("Kendisine kurumsal sadakatini bozması teklif ediliyor")
    assert isinstance(sim, dict)
    assert "simulation_analysis" in sim
    assert "confidence_score" in sim
    assert sim["confidence_score"] > 0


def test_academic_researcher_evaluate():
    researcher = AcademicResearcher()
    sample_paper = {
        "title": "Sleep and Cognitive Performance in Decision Making",
        "authors": ["A. Smith", "B. Taylor"],
        "year": 2024,
        "abstract": "We conducted a randomized trial with N = 450 participants investigating sleep deprivation and cognitive closure. The correlation was r = 0.38 (p < 0.001) in peer-reviewed Nature Human Behaviour.",
        "url": "https://doi.org/10.1038/s41562-024",
        "journal": "Nature Human Behaviour",
    }
    evaluation = researcher.evaluate_papers([sample_paper], marker="sleep_deprivation")
    assert isinstance(evaluation, dict)
    assert "ranked_papers" in evaluation
    assert len(evaluation["ranked_papers"]) == 1
    assert evaluation["ranked_papers"][0]["scores"]["total_rigor_score"] > 40
