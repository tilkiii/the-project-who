"""
Scenario simulation engine for who??.
Simulates plausible reactions and strategic counter-moves under explicit assumptions.
Adheres strictly to probabilistic reasoning without deterministic claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import get_logger

logger = get_logger("who.engine.simulator")


class Simulator:
    """Simulates plausible behavioral reactions and counter-moves under explicit scenario assumptions."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def simulate_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Simulate plausible target reaction, decision heuristics, and potential counter-moves.
        """
        logger.info(f"Simulating reaction for {self.target_name} on scenario: '{scenario[:60]}...'")

        # Load mindset summary and mapping skill
        mindset_path = self.dirs.vault_dir / "mindset_summary.md"
        mindset_text = mindset_path.read_text(encoding="utf-8") if mindset_path.exists() else ""

        skill_path = self.dirs.learned_skills_dir / "generated_mapping_skill.json"
        skill_data: Dict[str, Any] = {}
        if skill_path.exists():
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    skill_data = json.load(f)
            except Exception:
                pass

        cog_style = skill_data.get("behavioral_profile", {}).get("cognitive_style", "Empirical")

        system_prompt = (
            f"You are the Behavioral Simulation Engine for {self.target_name}.\n"
            "EPISTEMOLOGICAL RULES:\n"
            "- You MUST NOT claim to predict the future or know secret intentions with certainty.\n"
            "- You MUST simulate plausible reactions under explicit assumptions.\n"
            "- Use probabilistic phrasing: 'Under the assumption that [X], the data suggests a plausible alignment with [Y]...'\n"
            "- Outline plausible alternative responses (Primary Reaction vs Alternative Scenario).\n"
            "- Provide explicit Confidence Rating (0.0 to 1.0).\n"
        )

        user_prompt = (
            f"SCENARIO PROPOSITION:\n\"{scenario}\"\n\n"
            f"OBSERVED COGNITIVE PROFILE:\n{mindset_text[:1000]}\n\n"
            "Simulate:\n"
            "1. Most Plausible Reaction (Primary Hypothesis)\n"
            "2. Expected Counter-Moves / Objections\n"
            "3. Key Friction Points\n"
            "4. Plausible Alternative Reaction Under Divergent Context\n"
            "5. Explicit Confidence Score with justification."
        )

        simulation_output = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.3)

        # Baseline probabilistic confidence
        confidence = 0.76 if mindset_text else 0.60

        return {
            "target_name": self.target_name,
            "scenario": scenario,
            "simulation_analysis": simulation_output,
            "confidence_score": confidence,
            "cognitive_style_assumed": cog_style,
            "epistemic_qualification": "Simulated under explicit assumptions based on observed corpus signals. Not a deterministic prediction.",
        }
