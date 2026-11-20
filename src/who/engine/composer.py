"""
Adversarial outreach composer for who??.
Generates tailored communications calibrated to target rhetoric with adversarial rejection testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from who.config import call_llm
from who.utils.logger import get_logger

logger = get_logger("who.engine.composer")


class Composer:
    """Composes hyper-tailored outreach drafts with simulated adversarial rejection stress-testing."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def compose_outreach(self, goal: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Draft an outreach letter tailored to target cognitive style,
        adversarially simulate target's rejection reflex, and optimize.
        """
        logger.info(f"Composing tailored outreach for {self.target_name} to achieve: '{goal}'")

        # Load learned features & skill
        features_file = self.dirs.learned_skills_dir / "features.json"
        features: Dict[str, Any] = {}
        if features_file.exists():
            try:
                with open(features_file, "r", encoding="utf-8") as f:
                    features = json.load(f)
            except Exception:
                pass

        terms = features.get("top_characteristic_terms", [])
        epistemic = features.get("epistemic_markers", {})
        cal_ratio = epistemic.get("epistemic_calibration_ratio", 1.0)

        # Step 1: Initial Draft Generation
        draft_prompt = (
            f"Write an outreach message/email to {self.target_name}.\n"
            f"GOAL: {goal}\n"
            f"ADDITIONAL CONTEXT: {context or 'Direct professional engagement'}\n"
            f"TARGET RHETORIC PROFILE:\n"
            f"- Characteristic terms to resonate with: {', '.join(terms[:5])}\n"
            f"- Epistemic style: {'High uncertainty calibration and empirical rigor' if cal_ratio > 1.0 else 'Direct, decisive and focused'}\n\n"
            "Produce a compelling, concise outreach letter."
        )

        initial_draft = call_llm(draft_prompt, temperature=0.3)

        # Step 2: Adversarial Rejection Stress-Test
        adversarial_prompt = (
            f"You are simulating the critical internal perspective of {self.target_name}.\n"
            "Review this incoming outreach proposal:\n"
            f"\"\"\"\n{initial_draft}\n\"\"\"\n\n"
            "Critically analyze:\n"
            "1. Why would you immediately delete or reject this?\n"
            "2. What buzzwords, assumptions, or friction points trigger skepticism?\n"
            "3. What exact adjustment would make it compelling?"
        )

        adversarial_critique = call_llm(adversarial_prompt, temperature=0.3)

        # Step 3: Hardened Final Optimization
        refine_prompt = (
            f"You are finalizing the outreach message for {self.target_name}.\n"
            f"INITIAL DRAFT:\n{initial_draft}\n\n"
            f"ADVERSARIAL CRITIQUE / REJECTION RISKS:\n{adversarial_critique}\n\n"
            "Rewrite and produce the hardened, optimized final version that neutralizes every identified friction point."
        )

        final_outreach = call_llm(refine_prompt, temperature=0.2)

        return {
            "target_name": self.target_name,
            "goal": goal,
            "initial_draft": initial_draft,
            "adversarial_rejection_analysis": adversarial_critique,
            "optimized_final_pitch": final_outreach,
            "calibrated_terms_used": terms[:5],
        }
