"""
01_quick_chat.py - Connect and Chat with a Cognitive Digital Twin
================================================================
Demonstrates how to initialize a CognitiveTwin instance, inspect its
calibrated personality and guardrails, and execute an interactive query.
"""

from who import CognitiveTwin, list_targets


def main():
    print("=== Available Targets in Local Vault ===")
    targets = list_targets()
    for t in targets:
        status = "READY" if t["is_ready"] else "PARTIAL"
        print(f" - {t['name']} ({t['slug']}): [{status}] ({t['raw_documents_count']} documents)")

    # Initialize Cognitive Twin for Target X
    target_name = "Target X"
    print(f"\nInitializing Cognitive Twin: {target_name}...")
    twin = CognitiveTwin(target_name, mode="formal")

    print("\n--- Target Information ---")
    print("Character Lore:", twin.chat_engine.character.get("bio", ["N/A"])[0])
    print("Active Guardrails:", twin.chat_engine.operational_rules)

    # Execute conversational query
    query = "What is the equilibrium strategy when negotiating with an asymmetric counterpart?"
    print(f"\nUser Query: {query}")
    print("\nTwin Response:")
    response = twin.chat(query)
    print(response)


if __name__ == "__main__":
    main()
