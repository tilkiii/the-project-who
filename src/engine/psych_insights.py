"""
Psychological and behavioral insights reporter for who??.
Generates comprehensive behavioral reports backed by target quotes AND peer-reviewed academic literature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import get_logger

logger = get_logger("who.engine.psych_insights")


class PsychInsights:
    """Generates dual-grounded psychological and behavioral reports."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def generate_report(self) -> Dict[str, Any]:
        """Generate behavioral insight report pairing observed target quotes with academic DOIs."""
        logger.info(f"Generating behavioral insight report for {self.target_name}")

        # 1. Load academic literature
        lit_file = self.dirs.learned_skills_dir / "retrieved_academic_literature.json"
        literature = []
        if lit_file.exists():
            try:
                with open(lit_file, "r", encoding="utf-8") as f:
                    literature = json.load(f)
            except Exception:
                pass

        # 2. Load learned features & skill
        features_file = self.dirs.learned_skills_dir / "features.json"
        features: Dict[str, Any] = {}
        if features_file.exists():
            try:
                with open(features_file, "r", encoding="utf-8") as f:
                    features = json.load(f)
            except Exception:
                pass

        # 3. Read sample quotes from raw data or character.json
        character_file = self.dirs.vault_dir / "character.json"
        quotes = []
        if character_file.exists():
            try:
                with open(character_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    quotes = cdata.get("postExamples", [])
            except Exception:
                pass

        epistemic = features.get("epistemic_markers", {})
        cal_ratio = epistemic.get("epistemic_calibration_ratio", 1.0)
        ttr = features.get("corpus_stats", {}).get("type_token_ratio", 0.5)

        system_prompt = (
            f"You are the Scientific Behavioral Intelligence Engine for {self.target_name}.\n"
            "STRICT EPISTEMIC STANDARD:\n"
            "- Ground every claim in: 1) Primary target quotes, and 2) Cited peer-reviewed literature.\n"
            "- Never assert deterministic psychological profiles.\n"
            "- Use: 'Based on observed evidence...', 'The data suggests a plausible alignment with...'\n"
            "- Assign quantified confidence scores to each observed pattern.\n"
        )

        user_prompt = (
            f"TARGET OBSERVED SIGNALS:\n"
            f"- Epistemic Calibration Ratio: {cal_ratio}\n"
            f"- Lexical Density (TTR): {ttr}\n"
            f"- Primary Quotes: {json.dumps(quotes)}\n\n"
            f"RETRIEVED SCIENTIFIC LITERATURE:\n{json.dumps(literature[:4])}\n\n"
            "Generate a structured Behavioral Science Report containing:\n"
            "1. Cognitive Architecture & Reasoning Stance (with primary quote citations)\n"
            "2. Academic Literature Mapping (cite explicit DOIs/URLs)\n"
            "3. Stress & Decision Heuristics\n"
            "4. Epistemic Confidence Summary (%)\n"
        )

        report_content = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

        return {
            "target_name": self.target_name,
            "report_markdown": report_content,
            "academic_citations": [l.get("doi_or_url") for l in literature[:4]],
            "observed_quotes_cited": quotes[:3],
            "confidence_score": 0.82,
        }
