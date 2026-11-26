"""
02_simulate_scenario.py - Strategic Scenario Simulation
======================================================
Runs an adversarial or hypothetical scenario simulation against a Target's
cognitive model to predict counter-moves, strategic stance, and calibrated risk.
"""

import json
import sys
from who import CognitiveTwin


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    target_name = "Target X"
    twin = CognitiveTwin(target_name)

    scenario = (
        "Counterpart consortium demands an immediate 40% equity share with unilateral "
        "veto power over protocol upgrades, threatening to fork the network if rejected within 24 hours."
    )

    print(f"=== Simulating Scenario against {target_name} ===")
    print(f"Scenario Proposition:\n{scenario}\n")

    result = twin.simulate_scenario(scenario)

    print("--- Simulation Output ---")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)


if __name__ == "__main__":
    main()
