"""
Prompt template compiler for who??.
Generates structured target-grounded prompt templates for Oracle Q&A, scenario simulation, and chat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from who.utils.logger import get_logger

logger = get_logger("who.learn.prompt_optimizer")


class PromptOptimizer:
    """Compiles evidence-grounded prompt templates from target behavioral skills and guardrails."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def optimize_prompts(
        self,
        target_name: str,
        mapping_skill: Dict[str, Any],
        training_examples: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Compile deterministic prompt templates for profiling, Oracle Q&A, and twin simulation."""
        logger.info(f"Compiling target prompt templates for: {target_name}")

        rules = mapping_skill.get("generation_rules", [])
        rules_text = "\n".join(f"- {r}" for r in rules)
        forbidden = "\n".join(f"- {f}" for f in mapping_skill.get("forbidden_phrasings", []))

        oracle_prompt_template = (
            f"You are the OSINT Evidence Engine for {target_name}.\n"
            "STRICT SCIENTIFIC EPISTEMOLOGY:\n"
            f"{rules_text}\n\n"
            "FORBIDDEN CLAIMS:\n"
            f"{forbidden}\n\n"
            "EVALUATION CRITERIA:\n"
            "1. Ground all claims in primary quotes from corpus.\n"
            "2. State confidence score explicitly (0-100%).\n"
            "3. Cite verified graph relationships.\n"
        )

        simulation_prompt_template = (
            f"You are simulating plausible reactions for {target_name}.\n"
            "REASONING CONSTRAINTS:\n"
            "- Simulate plausible reactions under explicit assumptions.\n"
            "- Never claim certainty about hidden motives.\n"
            f"- Reflect observed cognitive style: {mapping_skill.get('behavioral_profile', {}).get('cognitive_style', 'Empirical')}.\n"
        )

        optimized_package = {
            "target_name": target_name,
            "compiler_mode": "deterministic_template",
            "optimization_metric": "epistemic_calibration_and_hallucination_minimization",
            "compiled_templates": {
                "oracle_system_prompt": oracle_prompt_template,
                "simulation_system_prompt": simulation_prompt_template,
                "chat_formal_prompt": f"Embody {target_name} in formal/public analytical discourse.\n{rules_text}",
                "chat_informal_prompt": f"Embody {target_name} in direct 1-on-1 collaborative dialogue.\n{rules_text}",
            },
            "evaluation_thresholds": {
                "min_quote_citations": 2,
                "min_confidence_score": 0.65,
                "max_hallucination_rate": 0.05,
            },
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / "optimized_prompts.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(optimized_package, f, indent=2)
            logger.info(f"Saved optimized_prompts.json to {out_file}")

        return optimized_package
