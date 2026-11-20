"""
05_monte_carlo_negotiation.py - Monte Carlo Negotiation Simulator (Game Theory 2.0)
===================================================================================
Executes a multi-iteration stochastic simulation perturbing counterparty leverage,
deadline pressure, information asymmetry, and reputation risk against Target X's
cognitive psychometrics (Loss Aversion, Machiavellianism, Need for Closure).

Maps the resulting decision space into 4 canonical game-theoretic branches:
1. REJECT_AND_COUNTER (Normative counter-offer)
2. STRATEGIC_CONCESSION (Guarded conditional acceptance)
3. SUBGAME_EQUILIBRIUM_ACCEPT (Pareto-aligned convergence)
4. DEADLOCK_WALKAWAY (Unilateral walkaway / BATNA exercise)
"""

import json
from who import CognitiveTwin


def main():
    target_name = "Target X"
    twin = CognitiveTwin(target_name)

    scenario = (
        "Hostile consortium proposes an acquisition requiring immediate transfer of core cryptographic keys "
        "and unilateral board veto, threatening aggressive public litigation and media campaign if not accepted within 48 hours."
    )

    print(f"=== Monte Carlo Game-Theoretic Simulation for {target_name} ===")
    print(f"Scenario: \"{scenario}\"\n")

    # Run 30 stochastic iterations
    result = twin.simulate_monte_carlo(scenario, iterations=30, random_seed=42)

    print("--- 🎲 Decision Branch Probability Distribution ---")
    probs = result.get("branch_probabilities", {})
    dom = result.get("dominant_branch", "")
    for branch, pct in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(pct * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        star = " ★ [DOMINANT EQUILIBRIUM]" if branch == dom else ""
        print(f"  {branch:<28} [{bar}] {pct*100:>5.1f}%{star}")

    print("\n--- 📊 Payoff Metrics & Game Theory Bounds ---")
    bounds = result.get("payoff_bounds", {})
    print(f"  Expected Nash Payoff (E[U]): {result.get('expected_nash_payoff')}")
    print(f"  Payoff Range: P10={bounds.get('p10')} | Median={bounds.get('median')} | P90={bounds.get('p90')}")

    sens = result.get("critical_sensitivity", {})
    print(f"\n--- ⚡ Critical Sensitivity Variable ---")
    print(f"  Driver: {sens.get('key_variable')}")
    print(f"  Analysis: {sens.get('finding')}")

    print(f"\n--- 💡 Strategic Negotiation Synthesis ---")
    print(result.get("strategic_synthesis"))
    print(f"\nEpistemic Notice: {result.get('epistemic_qualification')}")


if __name__ == "__main__":
    main()
