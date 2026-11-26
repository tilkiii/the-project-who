"""
Offline / deterministic fallback provider for who??.
Used when no API keys are provided or when network is unavailable.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional

from who.providers.base import BaseLLMProvider


def hash_embedding(text: str, dims: int = 128) -> List[float]:
    """Deterministic pseudo-embedding using SHA-256 for testing and offline runs."""
    vec = []
    for i in range(dims):
        h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OfflineProvider(BaseLLMProvider):
    """Deterministic offline fallback provider."""

    provider_id: str = "offline"
    display_name: str = "Offline / Local Corpus Synthesizer"
    default_model: str = "offline/local-synthesizer"
    default_embedding_model: str = "offline/sha256-hash"
    env_keys: List[str] = []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        p_lower = prompt.lower()

        # Structured JSON extraction tasks
        if "extract entities" in p_lower or ("entity" in p_lower and "json" in p_lower):
            # Return empty sets so GraphBuilder deterministically extracts real entities/quotes from the corpus text.
            return '{"entities": [], "relationships": []}'
        elif "character.json" in p_lower or "elizaos" in p_lower:
            return (
                '{"name": "Target Model", "bio": ["Evidence-focused researcher with high empirical standards"], '
                '"lore": ["Known for rigorous analysis and probabilistic reasoning"], '
                '"style": {"all": ["probabilistic", "evidence-based"], "chat": ["direct", "analytical"]}}'
            )
        elif (
            ("mindset_summary" in p_lower or "cognitive" in p_lower)
            and "question:" not in p_lower
            and "scenario" not in p_lower
            and "simulate" not in p_lower
        ):
            return (
                "# Cognitive & Mindset Profile\n\n"
                "## 1. Core Reasoning Heuristics\n"
                "Based on observed evidence from primary corpus, the data suggests a plausible alignment "
                "with probabilistic epistemic frameworks.\n\n"
                "Confidence Score: 85% (Derived from primary corpus analysis)."
            )

        # Interactive Twin Chat conversational fallback
        combined_ctx = f"{system_prompt or ''}\n{prompt}"
        user_chat_match = re.search(r"(?:USER|Kullanıcı):\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        target_name_match = (
            re.search(r"digital twin of\s+([^\n\.]+)", combined_ctx, re.IGNORECASE)
            or re.search(r"bilişsel dijital ikizisin[:\s]+([^\n\.]+)", combined_ctx, re.IGNORECASE)
        )
        twin_name = target_name_match.group(1).strip() if target_name_match else "Digital Twin"

        if user_chat_match and "SCENARIO PROPOSITION" not in prompt:
            user_msg = user_chat_match.group(1).strip()
            from who.utils.text_cleaner import is_boilerplate_or_junk
            raw_quotes = re.findall(r'(?:Quotation|Alıntı):\s*"([^"]{20,})"', prompt)
            clean_quotes = [q.strip() for q in raw_quotes if not is_boilerplate_or_junk(q.strip())]

            lines = [
                f"Hello. I am {twin_name}.",
                f'I hear your point: "{user_msg}".',
                "",
                "In my framing, analytical clarity, grounded reasoning, and open dialogue are essential. I evaluate proposals carefully against verifiable evidence.",
            ]
            if clean_quotes:
                lines.append(f'\nAs reflected in my prior statements: "{clean_quotes[0]}"')
            lines.append("\n(Offline mode: Configure an API key in .env for full model generation.)")
            return "\n".join(lines)

        # Outreach Composer fallback (Initial Draft / Adversarial Critique / Final Refinement)
        if "write an outreach message/email to" in p_lower:
            tgt_m = re.search(r"write an outreach message/email to[ \t]+([^\r\n\.]+)", prompt, re.IGNORECASE)
            goal_m = re.search(r"GOAL:[ \t]*([^\r\n]*)", prompt)
            ctx_m = re.search(r"ADDITIONAL CONTEXT:[ \t]*([^\r\n]*)", prompt)
            terms_m = re.search(r"Characteristic terms to resonate with:[ \t]*([^\r\n]*)", prompt)
            tgt = tgt_m.group(1).strip() if tgt_m else "Recipient"
            goal = (goal_m.group(1).strip() if goal_m else "") or "Research collaboration"
            ctx = (ctx_m.group(1).strip() if ctx_m else "") or "Direct professional engagement"
            terms_line = terms_m.group(1).strip() if terms_m else ""
            res_note = f", aligning around {terms_line}" if terms_line else ""
            return (
                f"Subject: {goal}\n\n"
                f"Hello {tgt},\n\n"
                f"I am reaching out regarding {goal} ({ctx}). Rather than sending a broad pitch, I wanted to share "
                f"clear empirical premises, explicit methodology, and reciprocal commitments{res_note}.\n\n"
                "If useful, we can start with a concrete agenda and verifiable assumptions.\n\n"
                "(Offline mode: Configure an API key in .env for full model generation.)"
            )
        if "review this incoming outreach proposal" in p_lower:
            return (
                "1. Likely Friction Triggers: Unverified assumptions, vague deliverables, or asymmetric commitments.\n"
                "2. Rhetoric & Evidence Audit: Claims should cite concrete evidence bounds rather than broad enthusiasm.\n"
                "3. Recommended Revision: Lead with explicit scope, falsifiable metrics, and reciprocal terms.\n\n"
                "(Offline mode: Configure an API key in .env for full model generation.)"
            )
        if "you are finalizing the outreach message for" in p_lower:
            tgt_m = re.search(r"finalizing the outreach message for\s+([^\n\.]+)", prompt, re.IGNORECASE)
            tgt = tgt_m.group(1).strip() if tgt_m else "Recipient"
            draft_goal_m = re.search(r"Subject:\s*(.+?)(?:\n|$)", prompt)
            subj = draft_goal_m.group(1).strip() if draft_goal_m else "Focused Evidence-Based Proposal"
            return (
                f"Subject: {subj} — Bounded Scope & Evidence\n\n"
                f"Hello {tgt},\n\n"
                f"To keep this strictly actionable: the objective is {subj}, scoped around verifiable evidence, "
                "explicit evaluation criteria, and balanced mutual opt-out boundaries.\n\n"
                "Let me know which verifiable details would help evaluate fit cleanly.\n\n"
                "(Offline mode: Configure an API key in .env for full model generation.)"
            )

        # Monte Carlo Game-Theoretic synthesis fallback
        if "monte carlo & boltzmann-gibbs results" in p_lower:
            mc_scenario = re.search(r'SCENARIO:\s*"?(.+?)"?\s*(?:\n|$)', prompt)
            dom_match = re.search(r"Dominant Nash Equilibrium Branch:\s*(.+?)(?:\n|$)", prompt)
            payoff_match = re.search(r"Expected Payoff \(E\[U\]\):\s*(.+?)(?:\n|$)", prompt)
            anchors_match = re.search(r"MATCHED KNOWLEDGE GRAPH ANCHORS:\s*(.+?)(?:\n|$)", prompt)
            caps_match = re.search(r"ACTIVE CAPS CONTEXT:\s*(.+?)(?:\n|$)", prompt)
            sens_match = re.search(r"Sensitivity Finding:\s*(.+?)(?:\n|$)", prompt)
            div_match = re.search(r"Stated vs\. Revealed Divergence:\s*(.+?)(?:\n|$)", prompt)

            scenario_val = mc_scenario.group(1).strip() if mc_scenario else "Submitted strategic proposition"
            dom_val = dom_match.group(1).strip() if dom_match else "UNDETERMINED"
            payoff_val = payoff_match.group(1).strip() if payoff_match else "N/A"
            anchors_val = anchors_match.group(1).strip() if anchors_match else "None matched in local graph"
            caps_val = caps_match.group(1).strip() if caps_match else "DEFAULT"
            sens_val = sens_match.group(1).strip() if sens_match else ""
            div_val = div_match.group(1).strip() if div_match else ""

            mc_lines = [
                "**Monte Carlo Scenario Synthesis (Corpus-Conditioned Model)**",
                f"- **Scenario:** {scenario_val}",
                f"- **Dominant Branch:** {dom_val}",
                f"- **Expected Payoff:** {payoff_val}",
                f"- **CAPS Context & KG Anchors:** {caps_val} | Anchors: {anchors_val}",
            ]
            if sens_val:
                mc_lines.append(f"- **Primary Sensitivity Driver:** {sens_val}")
            if div_val:
                mc_lines.append(f"- **Stated vs. Revealed Alignment:** {div_val}")
            mc_lines.append("\n(Offline mode: Configure an API key in .env for full model generation.)")
            return "\n".join(mc_lines)

        # Oracle Q&A or Scenario simulation fallback
        question_match = re.search(r"(?:QUESTION|SORU):\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        scenario_match = re.search(r'SCENARIO PROPOSITION:\s*["\']?(.+?)["\']?\n', prompt, re.IGNORECASE)
        is_scenario = bool(scenario_match)
        question = (
            scenario_match.group(1).strip()
            if is_scenario
            else (question_match.group(1).strip() if question_match else (user_chat_match.group(1).strip() if user_chat_match else "this topic"))
        )

        from who.utils.text_cleaner import is_boilerplate_or_junk
        raw_quotes = re.findall(r'(?:Quotation|Alıntı):\s*"([^"]{20,})"', prompt)
        usable_quotes = [q.strip() for q in raw_quotes if not is_boilerplate_or_junk(q.strip()) and len(q.strip()) > 30][:3]
        graph_edges = re.findall(r"Graph Edge:\s*(.+?)(?:\n|$)", prompt)

        label = "**Scenario:**" if is_scenario else "**Question:**"
        lines = [f"{label} {question}\n"]

        if is_scenario:
            lines.append("**Plausible Reaction Simulation (Corpus-Grounded):**\n")
            lines.append(
                "**1. Primary Hypothesis:** Based on observed corpus signals, the subject profile is plausibly "
                "inclined to approach this scenario from an analytical and principle-driven perspective.\n"
            )
            if usable_quotes:
                lines.append("**Supporting Corpus Evidence:**")
                for i, q in enumerate(usable_quotes, 1):
                    lines.append(f'> [{i}] "{q[:200].replace(chr(10), " ")}"')
            lines.append(
                "\n**2. Likely Counter-Considerations:** Epistemic caution is expected regarding speculative "
                "or unverified framing.\n"
                "**3. Confidence Score:** 76% (Estimated from corpus coverage)"
            )
        else:
            if usable_quotes:
                lines.append("**Corpus Evidence Analysis:**\n")
                lines.append(
                    "Direct evidence retrieved from the primary source corpus:\n"
                )
                for i, q in enumerate(usable_quotes, 1):
                    lines.append(f'> [{i}] "{q.replace(chr(10), " ")}"\n')
                lines.append(
                    "\nBased on these observed quotes, the corpus suggests a consistent analytical stance on this topic."
                )
            else:
                lines.append(
                    "No direct primary quotes matching this question were found in the current corpus. "
                    "Adding more source material is recommended."
                )

        if graph_edges:
            lines.append(f"\n**Graph Connections:** {'  |  '.join(graph_edges[:3])}")

        lines.append(
            "\n\n(Offline mode: Configure an API key in .env for full model generation.)"
        )

        return "\n".join(lines)

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        return hash_embedding(text)
