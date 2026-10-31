"""
Skill synthesizer for who??.
Compiles retrieved academic literature, linguistic markers, and topic trajectories
into an evidence-based generated_mapping_skill.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("who.learn.skill_synthesizer")


class SkillSynthesizer:
    """Synthesizes dynamic, evidence-backed behavioral mapping skills from observed data & research."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def synthesize_mapping_skill(
        self,
        target_name: str,
        features: Dict[str, Any],
        topics: List[Dict[str, Any]],
        literature: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compile an evidence-based behavioral skill definition."""
        logger.info(f"Synthesizing scientific behavioral mapping skill for: {target_name}")

        epistemic = features.get("epistemic_markers", {})
        pronouns = features.get("pronoun_ratios", {})
        stats = features.get("corpus_stats", {})

        # Compute empirical behavioral heuristics
        cal_ratio = epistemic.get("epistemic_calibration_ratio", 1.0)
        i_ratio = pronouns.get("first_person_singular_pct", 0.0)
        ttr = stats.get("type_token_ratio", 0.5)

        cognitive_style = "Probabilistic / Calibrated" if cal_ratio > 1.2 else (
            "Decisive / Assertive" if cal_ratio < 0.6 else "Balanced Empirical"
        )
        social_orientation = "Individualist / First-Person Centric" if i_ratio > 3.5 else "Collaborative / Collective"
        lexical_richness = "High Conceptual Density" if ttr > 0.4 else "Direct / Operational"

        rules = [
            f"Ground all statements in verifiable observations. Target demonstrates {cognitive_style.lower()} reasoning patterns.",
            f"Reflect observed social orientation ({social_orientation.lower()}) in communicative framing.",
            "Enforce strict epistemic humility: avoid deterministic psychological claims; frame findings with explicit confidence levels.",
        ]

        if topics:
            top_themes = [t.get("name", "") for t in topics[:3]]
            rules.append(f"Primary thematic focus centers upon: {', '.join(top_themes)}.")

        # Check for empirical behavioral boundaries & operational rules
        boundaries_payload = {}
        if self.output_dir and (self.output_dir / "behavioral_boundaries.json").exists():
            try:
                with open(self.output_dir / "behavioral_boundaries.json", "r", encoding="utf-8") as f:
                    boundaries_payload = json.load(f)
                    derived_rules = boundaries_payload.get("operational_rules", [])
                    for dr in derived_rules:
                        rule_text = f"[{dr.get('rule_id', 'RULE')}] {dr.get('directive', '')}"
                        if rule_text not in rules:
                            rules.append(rule_text)
            except Exception as e:
                logger.debug(f"Could not load behavioral_boundaries.json: {e}")

        academic_grounding = []
        for lit in literature[:5]:
            academic_grounding.append({
                "scientific_basis": lit.get("title"),
                "citation": lit.get("doi_or_url"),
                "domain": lit.get("repository", "Academic"),
                "applied_insight": f"Supports {cognitive_style} inference based on observed linguistic markers.",
            })

        skill_payload = {
            "target_name": target_name,
            "version": "4.0.0",
            "epistemic_framework": "Probabilistic Reasoning & Quantified Uncertainty",
            "behavioral_profile": {
                "cognitive_style": cognitive_style,
                "social_orientation": social_orientation,
                "lexical_richness": lexical_richness,
                "calibration_ratio": cal_ratio,
                "extended_psychometrics": {
                    "machiavellianism": features.get("machiavellianism_index", {}),
                    "need_for_closure": features.get("need_for_closure_index", {}),
                    "loss_aversion": features.get("loss_aversion_index", {}),
                    "affective_volatility": features.get("affective_volatility_index", {}),
                    "moral_foundations": features.get("moral_foundations_profile", {}),
                },
            },
            "empirical_decision_boundaries": boundaries_payload.get("boundaries", {}),
            "generation_rules": rules,
            "supported_evidence_models": academic_grounding,
            "required_output_phrasings": [
                "Based on observed evidence from [Source X]...",
                "The data suggests a plausible alignment with [Pattern Y]...",
                "Confidence Score: X% (Derived from [N] supporting primary quotes)",
            ],
            "forbidden_phrasings": [
                "Determine exactly how the target thinks",
                "Exact psychological mapping",
                "Predict future behavior with certainty",
            ],
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / "generated_mapping_skill.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(skill_payload, f, indent=2)
            logger.info(f"Saved generated_mapping_skill.json to {out_file}")

        return skill_payload
