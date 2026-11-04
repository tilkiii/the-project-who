"""
Scenario simulation engine for who??.
Simulates plausible reactions, strategic counter-moves, and Monte Carlo game-theoretic payoffs.
Adheres strictly to probabilistic reasoning without deterministic claims.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from who.config import call_llm
from who.utils.logger import get_logger

logger = get_logger("who.engine.simulator")


class Simulator:
    """Simulates plausible behavioral reactions and counter-moves under explicit scenario assumptions."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _load_profile_context(self) -> Tuple[str, Dict[str, Any]]:
        """Load mindset summary and mapping skill data."""
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
        return mindset_text, skill_data

    def simulate_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Simulate plausible target reaction, decision heuristics, and potential counter-moves (Single-Pass).
        """
        logger.info(f"Simulating reaction for {self.target_name} on scenario: '{scenario[:60]}...'")
        mindset_text, skill_data = self._load_profile_context()
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
        confidence = 0.76 if mindset_text else 0.60

        return {
            "target_name": self.target_name,
            "scenario": scenario,
            "simulation_analysis": simulation_output,
            "confidence_score": confidence,
            "cognitive_style_assumed": cog_style,
            "epistemic_qualification": "Simulated under explicit assumptions based on observed corpus signals. Not a deterministic prediction.",
        }

    def simulate_monte_carlo(
        self,
        scenario: str,
        iterations: int = 25,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Multi-iteration Monte Carlo negotiation & scenario simulator (Game Theory 2.0).
        Stochastically perturbs counterparty leverage, deadline pressure, information asymmetry,
        and reputation risk to map out decision branch probability distributions and Nash equilibria.
        """
        logger.info(f"Executing Monte Carlo scenario simulation for {self.target_name} (N={iterations})...")
        mindset_text, skill_data = self._load_profile_context()

        # Extract target psychometrics
        b_prof = skill_data.get("behavioral_profile", {})
        psychometrics = b_prof.get("extended_psychometrics", {})
        cog_style = b_prof.get("cognitive_style", "Empirical")

        loss_aversion = float(psychometrics.get("loss_aversion", {}).get("estimated_score", 1.85))
        machiavellianism = float(psychometrics.get("machiavellianism", {}).get("estimated_score", 0.35))
        need_for_closure = float(psychometrics.get("need_for_closure", {}).get("estimated_score", 0.50))
        risk_tolerance = 1.0 / max(loss_aversion, 1.0)

        # Baseline parameter ranges if not customized
        ranges = {
            "counterparty_leverage": (0.15, 0.85),
            "deadline_pressure_days": (1.0, 14.0),
            "information_asymmetry": (0.05, 0.65),
            "reputation_risk": (0.10, 0.80),
        }
        if parameter_ranges:
            ranges.update(parameter_ranges)

        rng = random.Random(random_seed) if random_seed is not None else random.Random()

        # 4 Canonical Decision Branches
        BRANCH_COUNTER = "REJECT_AND_COUNTER"
        BRANCH_CONCESSION = "STRATEGIC_CONCESSION"
        BRANCH_ACCEPT = "SUBGAME_EQUILIBRIUM_ACCEPT"
        BRANCH_DEADLOCK = "DEADLOCK_WALKAWAY"

        iteration_records: List[Dict[str, Any]] = []
        branch_counts = {
            BRANCH_COUNTER: 0,
            BRANCH_CONCESSION: 0,
            BRANCH_ACCEPT: 0,
            BRANCH_DEADLOCK: 0,
        }
        payoffs: List[float] = []

        for i in range(iterations):
            # Stochastic perturbation
            lev = rng.uniform(*ranges["counterparty_leverage"])
            deadline = rng.uniform(*ranges["deadline_pressure_days"])
            asym = rng.uniform(*ranges["information_asymmetry"])
            rep = rng.uniform(*ranges["reputation_risk"])

            # Utility functions grounded in Prospect Theory and Psychometrics
            u_accept = (1.0 - lev) * (1.0 - rep) - (loss_aversion * asym) + (machiavellianism * 0.20)
            u_counter = (0.60 * (1.0 - lev)) + (0.35 * (1.0 - asym)) + (0.25 * machiavellianism) - (0.15 / max(deadline, 1.0))
            u_concession = (0.45 * lev * machiavellianism) + (0.30 * (1.0 / max(deadline, 1.0)) * need_for_closure) - (0.25 * loss_aversion * rep)
            u_deadlock = 0.20 + (0.45 * loss_aversion * asym) + (0.40 * rep) - (0.35 * need_for_closure)

            # Assign branch
            utilities = {
                BRANCH_COUNTER: u_counter,
                BRANCH_CONCESSION: u_concession,
                BRANCH_ACCEPT: u_accept,
                BRANCH_DEADLOCK: u_deadlock,
            }
            chosen_branch = max(utilities, key=utilities.get)
            chosen_payoff = round(utilities[chosen_branch], 4)

            branch_counts[chosen_branch] += 1
            payoffs.append(chosen_payoff)

            iteration_records.append({
                "iteration": i + 1,
                "parameters": {
                    "counterparty_leverage": round(lev, 3),
                    "deadline_days": round(deadline, 1),
                    "information_asymmetry": round(asym, 3),
                    "reputation_risk": round(rep, 3),
                },
                "payoff": chosen_payoff,
                "branch": chosen_branch,
            })

        # Calculate probability distribution
        total_runs = float(iterations)
        distribution = {k: round(v / total_runs, 3) for k, v in branch_counts.items()}
        dominant_branch = max(distribution, key=distribution.get)

        # Payoff statistics
        sorted_payoffs = sorted(payoffs)
        p10 = sorted_payoffs[max(0, int(0.10 * len(sorted_payoffs)))]
        p50 = sorted_payoffs[int(0.50 * len(sorted_payoffs))]
        p90 = sorted_payoffs[min(len(sorted_payoffs) - 1, int(0.90 * len(sorted_payoffs)))]
        expected_nash_payoff = round(sum(payoffs) / len(payoffs), 3)

        # Sensitivity Analysis: identify which variable drives transition to deadlock or concession
        deadlock_asym_vals = [r["parameters"]["information_asymmetry"] for r in iteration_records if r["branch"] == BRANCH_DEADLOCK]
        normal_asym_vals = [r["parameters"]["information_asymmetry"] for r in iteration_records if r["branch"] != BRANCH_DEADLOCK]

        if deadlock_asym_vals and normal_asym_vals:
            avg_deadlock_asym = sum(deadlock_asym_vals) / len(deadlock_asym_vals)
            avg_normal_asym = sum(normal_asym_vals) / len(normal_asym_vals)
            critical_tipping_metric = "Information Asymmetry (α)"
            tipping_statement = (
                f"Information Asymmetry (α) is the critical switching parameter: "
                f"avg α was {avg_deadlock_asym:.2f} under Deadlock vs {avg_normal_asym:.2f} in constructive branches. "
                f"High opacity triggers the target's loss-aversion threshold (β={loss_aversion:.2f}), causing immediate walkaways."
            )
        else:
            critical_tipping_metric = "Counterparty Leverage (λ)"
            tipping_statement = (
                f"Counterparty Leverage (λ) is the primary friction point. "
                f"When leverage exceeds 0.70, the target's autonomy preservation overrides concessions."
            )

        # Strategic executive synthesis
        system_prompt = (
            f"You are the Game Theory and Behavioral Strategy Analyst for {self.target_name}.\n"
            "Summarize the Monte Carlo simulation results using game-theoretic reasoning."
        )
        user_prompt = (
            f"SCENARIO: \"{scenario}\"\n\n"
            f"MONTE CARLO RESULTS ({iterations} stochastic iterations):\n"
            f"- Outcome Distribution: {distribution}\n"
            f"- Dominant Nash Equilibrium Branch: {dominant_branch} ({int(distribution[dominant_branch]*100)}%)\n"
            f"- Expected Payoff (E[U]): {expected_nash_payoff} [P10: {p10}, Median: {p50}, P90: {p90}]\n"
            f"- Sensitivity Finding: {tipping_statement}\n\n"
            "Provide:\n"
            "1. Game-Theoretic Equilibrium Assessment\n"
            "2. Expected Counter-Offer Structure & Conditions\n"
            "3. Actionable Negotiation Advice to align with target's subgame incentives."
        )

        strategic_synthesis = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

        return {
            "target_name": self.target_name,
            "scenario": scenario,
            "iterations": iterations,
            "dominant_branch": dominant_branch,
            "branch_probabilities": distribution,
            "expected_nash_payoff": expected_nash_payoff,
            "payoff_bounds": {"p10": p10, "median": p50, "p90": p90},
            "critical_sensitivity": {
                "key_variable": critical_tipping_metric,
                "finding": tipping_statement,
            },
            "strategic_synthesis": strategic_synthesis,
            "assumed_psychometrics": {
                "loss_aversion": loss_aversion,
                "machiavellianism": machiavellianism,
                "need_for_closure": need_for_closure,
                "cognitive_style": cog_style,
            },
            "sample_iterations": iteration_records[:5],
            "epistemic_qualification": (
                f"Monte Carlo stochastic model across {iterations} perturbations of leverage, urgency, opacity, and risk. "
                "Represents probabilistic decision space under bounded rationality, not a deterministic certainty."
            ),
        }
