"""
High-Level Personal Agent SDK for who?? Cognitive Digital Twin & OSINT Framework.
Provides an idiomatic Python interface for AI agents, multi-agent frameworks, and autonomous workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.engine.composer import Composer
from src.engine.evaluator import ProfileEvaluator
from src.engine.explainability_engine import ExplainabilityEngine
from src.engine.oracle import Oracle
from src.engine.psych_insights import PsychInsights
from src.engine.simulator import Simulator
from src.engine.twin_chat import TwinChat
from src.learn.autonomous_researcher import AutonomousAcademicResearcher
from src.learn.scientific_referee import ScientificReferee
from src.utils.text_cleaner import sanitize_slug


def list_targets() -> List[Dict[str, Any]]:
    """List all profiled targets currently available in the vault."""
    profiles_dir = Path(settings.profiles_dir)
    if not profiles_dir.exists():
        return []

    targets = []
    for item in sorted(profiles_dir.iterdir()):
        if item.is_dir() and item.name not in ("social_graphs", "arena_debates", "__pycache__"):
            raw_dir = item / "raw_data"
            vault_dir = item / "vault"
            skills_dir = item / "learned_skills"

            raw_count = len(list(raw_dir.glob("*.md"))) if raw_dir.exists() else 0
            has_mindset = (vault_dir / "mindset_summary.md").exists()
            has_graph = (vault_dir / "knowledge_graph.json").exists()
            has_skills = (skills_dir / "generated_mapping_skill.json").exists()

            targets.append({
                "slug": item.name,
                "name": item.name.replace("_", " ").title(),
                "raw_documents_count": raw_count,
                "is_ready": bool(has_mindset and has_graph),
                "has_skills": has_skills,
                "path": str(item),
            })
    return targets


class CognitiveTwin:
    """Personal Agent interface to a target's cognitive digital twin."""

    def __init__(self, target_name: str, mode: str = "autodetect"):
        self.target_name = target_name
        self.slug = sanitize_slug(target_name)
        self.dirs = settings.get_target_dirs(self.target_name)
        self.mode = mode

        # Core engines
        self._chat_engine: Optional[TwinChat] = None
        self._simulator: Optional[Simulator] = None
        self._explain_engine: Optional[ExplainabilityEngine] = None
        self._composer: Optional[Composer] = None
        self._oracle: Optional[Oracle] = None

    @property
    def chat_engine(self) -> TwinChat:
        if self._chat_engine is None:
            self._chat_engine = TwinChat(self.target_name, self.dirs, mode=self.mode)
        return self._chat_engine

    @property
    def simulator(self) -> Simulator:
        if self._simulator is None:
            self._simulator = Simulator(self.target_name, self.dirs)
        return self._simulator

    @property
    def explain_engine(self) -> ExplainabilityEngine:
        if self._explain_engine is None:
            self._explain_engine = ExplainabilityEngine(self.target_name, self.dirs)
        return self._explain_engine

    @property
    def composer(self) -> Composer:
        if self._composer is None:
            self._composer = Composer(self.target_name, self.dirs)
        return self._composer

    @property
    def oracle(self) -> Oracle:
        if self._oracle is None:
            self._oracle = Oracle(self.target_name, self.dirs)
        return self._oracle

    def chat(self, message: str, tone: Optional[str] = None) -> str:
        """
        Chat directly in-character with the target's cognitive digital twin.
        Adheres strictly to the target's persona, verified memory, and active decision boundaries.
        """
        if tone and tone in ("formal", "informal", "autodetect"):
            self.chat_engine.mode = tone
        return self.chat_engine.respond(message)

    def simulate_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Simulate target's plausible counter-move and reaction to a hypothetical or adversarial proposition.
        Returns simulation analysis, hypothesis, confidence score, and primary quotes.
        """
        return self.simulator.simulate_scenario(scenario)

    def explain(self, claim: str) -> Dict[str, Any]:
        """
        Trace any claim back to primary corpus quotes, graph paths, and epistemic confidence.
        """
        return self.explain_engine.explain_claim(claim)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask an analytical question about the target's beliefs, stances, or principles.
        """
        return self.oracle.answer_question(question)

    def compose_outreach(self, goal: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate hyper-tailored communication drafts optimized for the target with adversarial rejection stress-testing.
        """
        return self.composer.compose_outreach(goal=goal, context=context)

    def get_profile(self) -> Dict[str, Any]:
        """
        Get target's comprehensive psychometric profile, empirical boundaries, active rules, and lore.
        """
        profile: Dict[str, Any] = {
            "name": self.target_name,
            "slug": self.slug,
            "psychometrics": {},
            "boundaries": {},
            "operational_rules": [],
            "character_lore": {},
            "mindset_summary": "",
        }

        # 1. Load generated mapping skills & psychometrics
        skill_file = self.dirs.learned_skills_dir / "generated_mapping_skill.json"
        if skill_file.exists():
            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    skill_data = json.load(f)
                    b_prof = skill_data.get("behavioral_profile", {})
                    profile["psychometrics"] = b_prof.get("extended_psychometrics", {})
                    profile["boundaries"] = skill_data.get("empirical_decision_boundaries", {})
                    profile["operational_rules"] = [
                        r for r in skill_data.get("generation_rules", [])
                        if r.startswith("[") or "Ground all" not in r
                    ]
            except Exception:
                pass

        # 2. Load character JSON
        char_file = self.dirs.vault_dir / "character.json"
        if char_file.exists():
            try:
                with open(char_file, "r", encoding="utf-8") as f:
                    profile["character_lore"] = json.load(f)
            except Exception:
                pass

        # 3. Load mindset summary
        mindset_file = self.dirs.vault_dir / "mindset_summary.md"
        if mindset_file.exists():
            try:
                profile["mindset_summary"] = mindset_file.read_text(encoding="utf-8")
            except Exception:
                pass

        return profile

    def get_operational_rules(self) -> List[str]:
        """Retrieve active empirical behavioral rules (e.g. [TACTICAL_TRANSACTIONALISM])."""
        return self.get_profile().get("operational_rules", [])

    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Retrieve the target's cognitive knowledge graph nodes and relationships."""
        graph_file = self.dirs.vault_dir / "knowledge_graph.json"
        if graph_file.exists():
            try:
                with open(graph_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"nodes": [], "edges": []}


class AcademicResearcher:
    """Autonomous Academic Researcher and AI Scientific Referee interface."""

    def __init__(self, target_name: Optional[str] = None):
        self.target_name = target_name
        output_dir = None
        if target_name:
            target_dirs = settings.get_target_dirs(target_name)
            output_dir = target_dirs.learned_skills_dir

        self.researcher = AutonomousAcademicResearcher(output_dir=output_dir)
        self.referee = ScientificReferee()

    def research_topic(
        self,
        topic: str,
        sources: Optional[List[str]] = None,
        limit: int = 5,
        ingest: bool = False,
    ) -> Dict[str, Any]:
        """
        Search arXiv, PubMed, Semantic Scholar, and OpenAlex for peer-reviewed papers.
        Evaluates each paper with the 0-100 Scientific Rigor Index.
        """
        srcs = sources or ["arxiv", "pubmed", "semantic_scholar", "crossref"]
        res = self.researcher.research_topic(topic, sources=srcs, limit=limit)

        if ingest and self.target_name and res.get("papers"):
            self.researcher.ingest_academic_corpus(self.target_name, res["papers"])

        return res

    def evaluate_papers(
        self,
        papers: List[Dict[str, Any]],
        marker: str = "general_cognition",
        prior_mean_r: float = 0.25,
    ) -> Dict[str, Any]:
        """Run AI Scientific Referee evaluation on a collection of papers."""
        return self.referee.evaluate_and_rank_corpus(papers, marker=marker, prior_mean_r=prior_mean_r)

    def generate_review(self, papers: List[Dict[str, Any]]) -> str:
        """Synthesize an academic literature review grounded in validated papers."""
        target = self.target_name or "Target Profile"
        return self.researcher.generate_literature_review(target, papers)


def get_target(target_name: str) -> CognitiveTwin:
    """Convenience factory to get a CognitiveTwin instance."""
    return CognitiveTwin(target_name)
