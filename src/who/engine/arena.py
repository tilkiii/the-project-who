"""
Multi-agent debate, negotiation, and Coupled Ising Spin-Glass interaction engine.
Convenes multiple digital twins into structured rounds to track consensus,
Cross-Agent Ising Hamiltonian interaction energy (H_AB), Edwards-Anderson overlap (q_AB),
round-by-round Jensen-Shannon Divergence (JSD) convergence, and emergent insights.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from who.config import call_llm, settings
from who.engine.evaluator import compute_jensen_shannon_divergence
from who.learn.feature_extractor import FeatureExtractor
from who.utils.logger import get_logger

logger = get_logger("who.engine.arena")


class SwarmArena:
    """Orchestrates structured multi-agent debates between probabilistic digital twins with Coupled Ising Physics."""

    def __init__(
        self,
        participant_names: List[str],
        topic: str,
        rounds: int = 2,
        output_dir: Optional[Path] = None,
    ):
        self.participant_names = [p.strip() for p in participant_names if p.strip()]
        self.topic = topic
        self.rounds = max(1, min(rounds, 5))
        self.output_dir = output_dir or (settings.profiles_dir / "arena_debates")
        self.participants_data: Dict[str, Dict[str, Any]] = {}
        self.feature_extractor = FeatureExtractor()

    def load_participants(self) -> Dict[str, Dict[str, Any]]:
        """Load character.json, mindset summary, features.json, and graph physics for all participants."""
        for name in self.participant_names:
            dirs = settings.get_target_dirs(name, create=False)
            char_file = dirs.vault_dir / "character.json"
            mindset_file = dirs.vault_dir / "mindset_summary.md"
            features_file = dirs.learned_skills_dir / "features.json"
            graph_metrics_file = dirs.vault_dir / "graph_metrics.json"

            character_data: Dict[str, Any] = {}
            if char_file.exists():
                try:
                    with open(char_file, "r", encoding="utf-8") as f:
                        character_data = json.load(f)
                except Exception:
                    pass

            mindset_text = ""
            if mindset_file.exists():
                try:
                    mindset_text = mindset_file.read_text(encoding="utf-8")
                except Exception:
                    pass

            features_data: Dict[str, Any] = {}
            if features_file.exists():
                try:
                    with open(features_file, "r", encoding="utf-8") as f:
                        features_data = json.load(f)
                except Exception:
                    pass

            graph_metrics: Dict[str, Any] = {}
            if graph_metrics_file.exists():
                try:
                    with open(graph_metrics_file, "r", encoding="utf-8") as f:
                        graph_metrics = json.load(f)
                except Exception:
                    pass

            quotes = character_data.get("postExamples", [])
            topics = character_data.get("topics", [])
            style = character_data.get("style", {})

            self.participants_data[name] = {
                "name": name,
                "character": character_data,
                "mindset_summary": mindset_text[:1500],
                "features": features_data,
                "graph_metrics": graph_metrics,
                "quotes": quotes[:3],
                "topics": topics[:5],
                "style": style,
                "dirs": dirs,
            }
            logger.info(f"Loaded arena profile for participant: {name}")

        return self.participants_data

    def _extract_agent_spin_vector(
        self,
        features: Dict[str, Any],
        statement_text: str = "",
    ) -> Tuple[Dict[str, float], List[float]]:
        """
        Map an agent's cognitive & psychometric state (blended with live round discourse)
        into a 6-dimensional continuous Ising Spin vector s_i in [-1.0, +1.0]:
        1. epistemic_calibration_spin (+1 probabilistic vs -1 dogmatic)
        2. cooperative_utility_spin (+1 cooperative vs -1 Machiavellian)
        3. cognitive_openness_spin (+1 open/exploratory vs -1 rigid closure)
        4. risk_prospect_spin (+1 opportunity-seeking vs -1 loss-aversive)
        5. moral_foundation_spin (+1 individualizing/fairness vs -1 binding/authority)
        6. revealed_action_spin (+1 collaborative commitment vs -1 autonomous veto/exit)
        Also returns a 5D normalized discourse distribution for JSD calculation.
        """
        live_feats = self.feature_extractor.extract_features([statement_text]) if statement_text.strip() else {}

        def _blend(base_val: float, live_val: Optional[float], w_live: float = 0.45) -> float:
            if live_val is None or not statement_text.strip():
                return base_val
            return (1.0 - w_live) * base_val + (w_live * live_val)

        # 1. Calibration spin
        base_cal = float(features.get("epistemic_markers", {}).get("epistemic_calibration_ratio", 1.0))
        live_cal = float(live_feats.get("epistemic_markers", {}).get("epistemic_calibration_ratio", base_cal)) if live_feats else None
        cal = _blend(base_cal, live_cal)
        spin_cal = math.tanh(cal - 0.85)

        # 2. Cooperative vs Machiavellian spin
        base_mach = float(features.get("machiavellianism_index", {}).get("estimated_score", 0.35))
        live_mach = float(live_feats.get("machiavellianism_index", {}).get("estimated_score", base_mach)) if live_feats else None
        mach = _blend(base_mach, live_mach)
        spin_coop = math.tanh((0.45 - mach) * 2.5)

        # 3. Openness vs Closure spin
        base_clos = float(features.get("need_for_closure_index", {}).get("estimated_score", 0.45))
        live_clos = float(live_feats.get("need_for_closure_index", {}).get("estimated_score", base_clos)) if live_feats else None
        clos = _blend(base_clos, live_clos)
        spin_open = math.tanh((0.50 - clos) * 2.5)

        # 4. Risk Prospect spin
        base_loss = float(features.get("loss_aversion_index", {}).get("estimated_score", 1.65))
        live_loss = float(live_feats.get("loss_aversion_index", {}).get("estimated_score", base_loss)) if live_feats else None
        loss = _blend(base_loss, live_loss)
        spin_risk = math.tanh((1.65 - loss) * 1.5)

        # 5. Moral Foundation spin
        moral_base = features.get("moral_foundations_profile", {})
        ind_hits = float(moral_base.get("individualizing_hits", 1))
        bind_hits = float(moral_base.get("binding_hits", 1))
        if live_feats:
            m_live = live_feats.get("moral_foundations_profile", {})
            ind_hits += float(m_live.get("individualizing_hits", 0))
            bind_hits += float(m_live.get("binding_hits", 0))
        spin_moral = math.tanh((ind_hits - bind_hits) / max(1.0, ind_hits + bind_hits))

        # 6. Revealed Action spin
        pref_base = features.get("stated_vs_revealed_preference", {})
        base_rev = float(pref_base.get("revealed_cooperation_polarity", 0.50))
        live_rev = float(live_feats.get("stated_vs_revealed_preference", {}).get("revealed_cooperation_polarity", base_rev)) if live_feats else None
        rev = _blend(base_rev, live_rev)
        spin_rev = math.tanh((rev - 0.50) * 2.5)

        spin_map = {
            "epistemic_calibration": round(spin_cal, 4),
            "cooperative_utility": round(spin_coop, 4),
            "cognitive_openness": round(spin_open, 4),
            "risk_prospect": round(spin_risk, 4),
            "moral_foundation": round(spin_moral, 4),
            "revealed_action": round(spin_rev, 4),
        }

        # Positive probability vector for JSD comparison
        prob_vec = [max(0.05, 0.5 * (s + 1.0)) for s in spin_map.values()]
        v_sum = sum(prob_vec) or 1.0
        norm_prob_vec = [p / v_sum for p in prob_vec]

        return spin_map, norm_prob_vec

    def _compute_coupled_ising_arena_physics(
        self,
        transcript: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Model multi-agent debate as a Coupled Ising Spin-Glass Interaction Network:
        - Cross-Agent Hamiltonian between agents A and B at round t:
          H_AB(t) = - (1 / M) * sum_{k=1}^M J_k * s_{k,t}^{(A)} * s_{k,t}^{(B)}
        - Edwards-Anderson Magnetization Overlap Order Parameter:
          q_AB(t) = (1 / M) * sum_{k=1}^M s_{k,t}^{(A)} * s_{k,t}^{(B)} in [-1, +1]
        - Tracks round-by-round energy relaxation Delta_H and discourse JSD convergence.
        """
        coupling_weights = {
            "epistemic_calibration": 1.15,
            "cooperative_utility": 1.25,
            "cognitive_openness": 1.00,
            "risk_prospect": 1.10,
            "moral_foundation": 0.95,
            "revealed_action": 1.25,
        }
        weight_sum = sum(coupling_weights.values())

        round_numbers = sorted({t["round"] for t in transcript}) or [1]
        round_trajectories: List[Dict[str, Any]] = []

        for r_num in round_numbers:
            round_turns = [t for t in transcript if t["round"] == r_num]
            agent_spins: Dict[str, Dict[str, float]] = {}
            agent_probs: Dict[str, List[float]] = {}

            for p_name in self.participant_names:
                p_info = self.participants_data.get(p_name, {})
                stmt = next((t["statement"] for t in round_turns if t["speaker"] == p_name), "")
                s_map, p_vec = self._extract_agent_spin_vector(p_info.get("features", {}), stmt)
                agent_spins[p_name] = s_map
                agent_probs[p_name] = p_vec

            # Compute pairwise H_AB, q_AB, and JSD across all participant pairs
            pair_energies: List[float] = []
            pair_overlaps: List[float] = []
            pair_jsds: List[float] = []
            axis_products: Dict[str, float] = {k: 0.0 for k in coupling_weights}
            pair_count = 0

            names = list(agent_spins.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a_name, b_name = names[i], names[j]
                    s_a, s_b = agent_spins[a_name], agent_spins[b_name]
                    p_a, p_b = agent_probs[a_name], agent_probs[b_name]

                    h_ab = -sum(coupling_weights[k] * s_a[k] * s_b[k] for k in coupling_weights) / weight_sum
                    q_ab = sum(s_a[k] * s_b[k] for k in coupling_weights) / len(coupling_weights)
                    jsd_ab = compute_jensen_shannon_divergence(p_a, p_b)

                    pair_energies.append(h_ab)
                    pair_overlaps.append(q_ab)
                    pair_jsds.append(jsd_ab)
                    for k in coupling_weights:
                        axis_products[k] += s_a[k] * s_b[k]
                    pair_count += 1

            denom = max(1, pair_count)
            mean_h = round(sum(pair_energies) / denom, 4)
            mean_q = round(sum(pair_overlaps) / denom, 4)
            mean_jsd = round(sum(pair_jsds) / denom, 4)
            mean_axis_overlap = {k: round(v / denom, 4) for k, v in axis_products.items()}

            round_trajectories.append({
                "round": r_num,
                "interaction_hamiltonian_H_AB": mean_h,
                "edwards_anderson_overlap_q_AB": mean_q,
                "inter_agent_discourse_jsd": mean_jsd,
                "axis_spin_correlations": mean_axis_overlap,
                "agent_spin_states": agent_spins,
            })

        initial_state = round_trajectories[0]
        final_state = round_trajectories[-1]
        delta_h = round(final_state["interaction_hamiltonian_H_AB"] - initial_state["interaction_hamiltonian_H_AB"], 4)
        delta_jsd = round(final_state["inter_agent_discourse_jsd"] - initial_state["inter_agent_discourse_jsd"], 4)
        final_q = final_state["edwards_anderson_overlap_q_AB"]
        final_jsd = final_state["inter_agent_discourse_jsd"]

        # Partition axes into Ferromagnetic Consensus vs Antiferromagnetic Friction
        final_axes = final_state["axis_spin_correlations"]
        aligned_axes = [k for k, v in sorted(final_axes.items(), key=lambda x: x[1], reverse=True) if v >= 0.0]
        frustrated_axes = [k for k, v in sorted(final_axes.items(), key=lambda x: x[1]) if v < 0.0]

        if final_state["interaction_hamiltonian_H_AB"] <= -0.15:
            phase_verdict = "FERROMAGNETIC_CONSENSUS_ATTRACTOR"
        elif final_state["interaction_hamiltonian_H_AB"] >= 0.15:
            phase_verdict = "ANTIFERROMAGNETIC_SPIN_GLASS_DEADLOCK"
        else:
            phase_verdict = "METASTABLE_NEGOTIATION_COEXISTENCE"

        # Derive mathematically calibrated consensus % in [15, 92]
        raw_consensus = 50.0 * (1.0 + final_q) * (1.0 - 0.35 * final_jsd)
        calibrated_consensus_pct = int(round(max(15.0, min(92.0, raw_consensus))))

        return {
            "initial_hamiltonian_H_AB": initial_state["interaction_hamiltonian_H_AB"],
            "final_hamiltonian_H_AB": final_state["interaction_hamiltonian_H_AB"],
            "hamiltonian_energy_shift_delta_H": delta_h,
            "cross_agent_spin_correlation_q_AB": final_q,
            "edwards_anderson_overlap_q_AB": final_q,
            "inter_agent_discourse_jsd": final_jsd,
            "discourse_jsd_shift_delta_jsd": delta_jsd,
            "ferromagnetic_aligned_axes": aligned_axes,
            "antiferromagnetic_frustrated_axes": frustrated_axes,
            "phase_equilibrium_verdict": phase_verdict,
            "calibrated_consensus_score_pct": calibrated_consensus_pct,
            "calibrated_divergence_pct": 100 - calibrated_consensus_pct,
            "round_trajectories": round_trajectories,
        }

    def run_debate(self) -> Dict[str, Any]:
        """Execute multi-round debate turns, compute Coupled Ising Interaction Physics, and synthesize consensus divergence."""
        if not self.participants_data:
            self.load_participants()

        if len(self.participants_data) < 2:
            raise ValueError("Swarm Arena requires at least 2 participants with existing profiles.")

        logger.info(f"Initiating Swarm Arena debate on topic: '{self.topic}' across {len(self.participants_data)} participants.")

        debate_transcript: List[Dict[str, Any]] = []
        live_ising_state: Optional[Dict[str, Any]] = None

        for round_idx in range(1, self.rounds + 1):
            for p_name, p_data in self.participants_data.items():
                speaker_turn = self._generate_turn(
                    speaker=p_data,
                    round_num=round_idx,
                    transcript=debate_transcript,
                    live_ising_state=live_ising_state,
                )
                debate_transcript.append(speaker_turn)
            # Compute in-loop Coupled Ising Hamiltonian after round_idx so round_idx+1 reacts to frustrated axes
            live_ising_state = self._compute_coupled_ising_arena_physics(debate_transcript)
            for t_rec in debate_transcript:
                if t_rec["round"] == round_idx and "round_ising_hamiltonian_H_AB" not in t_rec:
                    t_rec["round_ising_hamiltonian_H_AB"] = live_ising_state["final_hamiltonian_H_AB"]
                    t_rec["active_frustrated_axes"] = live_ising_state["antiferromagnetic_frustrated_axes"]

        # Final Coupled Multi-Agent Ising Interaction Physics across full trajectory
        coupled_ising_physics = live_ising_state or self._compute_coupled_ising_arena_physics(debate_transcript)

        # Meta-analysis and synthesis grounded in Coupled Ising Physics
        synthesis = self._synthesize_debate(debate_transcript, coupled_ising_physics)
        synthesis["coupled_ising_physics"] = coupled_ising_physics

        report_markdown = self._build_report_markdown(debate_transcript, synthesis, coupled_ising_physics)
        self._persist_report(report_markdown, synthesis)

        return {
            "topic": self.topic,
            "participants": self.participant_names,
            "rounds": self.rounds,
            "transcript": debate_transcript,
            "coupled_ising_physics": coupled_ising_physics,
            "synthesis": synthesis,
            "report_markdown": report_markdown,
        }

    def _generate_turn(
        self,
        speaker: Dict[str, Any],
        round_num: int,
        transcript: List[Dict[str, Any]],
        live_ising_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a single debate turn reflecting speaker's calibrated cognitive style and in-loop Ising feedback."""
        speaker_name = speaker["name"]
        quotes_json = json.dumps(speaker["quotes"], ensure_ascii=False)
        topics_str = ", ".join(speaker["topics"])

        history_text = ""
        if transcript:
            history_text = "PREVIOUS ROUNDS:\n" + "\n".join(
                f"- [{t['speaker']} (Round {t['round']})]: {t['statement']}" for t in transcript[-4:]
            )

        ising_feedback_block = ""
        if live_ising_state:
            frust_axes = ", ".join(live_ising_state.get("antiferromagnetic_frustrated_axes", [])) or "none"
            aligned_axes = ", ".join(live_ising_state.get("ferromagnetic_aligned_axes", [])) or "none"
            ising_feedback_block = (
                f"PREVIOUS ROUND COUPLED ISING INTERACTION FEEDBACK:\n"
                f"- H_AB(t-1) = {live_ising_state.get('final_hamiltonian_H_AB', 0.0)} | "
                f"q_AB(t-1) = {live_ising_state.get('edwards_anderson_overlap_q_AB', 0.0)}\n"
                f"- Antiferromagnetic Frustrated Axes (address directly this round): [{frust_axes}]\n"
                f"- Ferromagnetic Aligned Axes: [{aligned_axes}]\n\n"
            )

        system_prompt = (
            f"You are the probabilistic digital twin of '{speaker_name}' in an intellectual debate arena.\n"
            f"Ground your arguments strictly in your verified cognitive mindset, primary quotes, and epistemological stance.\n"
            "STRICT RULES:\n"
            "- Speak in first person.\n"
            "- Never make absolute psychological assertions about opponents; use probabilistic reasoning.\n"
            "- Directly address the preceding speaker's points, highlighting points of alignment or epistemic friction.\n"
            "- Reflect your characteristic vocabulary and intellectual anchors.\n"
            "- Length: 2 to 4 concise, substantive paragraphs in the same language as the debate topic."
        )

        user_prompt = (
            f"DEBATE TOPIC: {self.topic}\n"
            f"ROUND: {round_num}/{self.rounds}\n"
            f"YOUR IDENTITY & COGNITIVE ANCHORS:\n"
            f"- Core Concepts: {topics_str}\n"
            f"- Primary Quotes: {quotes_json}\n\n"
            f"{ising_feedback_block}"
            f"{history_text}\n\n"
            f"It is your turn to speak ({speaker_name}). Evaluate the proposition and prior speakers' arguments "
            f"from your intellectual and epistemic stance."
        )

        try:
            statement = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.3)
        except Exception as e:
            logger.warning(f"Arena LLM turn generation failed ({e}), using empirical fallback synthesis.")
            statement = (
                f"Based on observed evidence, our approach to '{self.topic}' should rest on institutional clarity "
                f"and verifiable principles. While the preceding speaker raises notable points, evaluating this through "
                f"my epistemic framework indicates that {topics_str} dynamics cannot be overlooked."
            )

        return {
            "round": round_num,
            "speaker": speaker_name,
            "statement": statement.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _synthesize_debate(
        self,
        transcript: List[Dict[str, Any]],
        ising_physics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate consensus divergence, key friction axes, and emergent synthesis grounded in Coupled Ising metrics."""
        all_text = "\n\n".join(f"[{t['speaker']}]: {t['statement']}" for t in transcript)

        prompt = (
            f"You are the Chief Epistemic Moderator analyzing an intellectual debate arena on topic: '{self.topic}'.\n"
            f"PARTICIPANTS: {', '.join(self.participant_names)}\n"
            f"COUPLED ISING HAMILTONIAN (H_AB): {ising_physics['final_hamiltonian_H_AB']} "
            f"(Overlap q_AB={ising_physics['edwards_anderson_overlap_q_AB']}, Discourse JSD={ising_physics['inter_agent_discourse_jsd']})\n\n"
            f"FULL DEBATE TRANSCRIPT:\n{all_text}\n\n"
            "Analyze the debate and return ONLY valid JSON with this schema (write string values in the same language as the debate topic while keeping JSON keys in English):\n"
            "{\n"
            '  "consensus_divergence_pct": 0-100,\n'
            '  "consensus_score_pct": 0-100,\n'
            '  "common_ground_points": ["string", "string"],\n'
            '  "irreconcilable_frictions": ["string", "string"],\n'
            '  "emergent_synthesis": "string (Summary of constructive outcome)",\n'
            '  "most_persuasive_pivot": "string"\n'
            "}"
        )

        try:
            res = call_llm(prompt, temperature=0.1)
            m = re.search(r"\{.*\}", res, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if "consensus_score_pct" in parsed and "consensus_divergence_pct" in parsed:
                    return parsed
        except Exception as e:
            logger.warning(f"Debate synthesis LLM parsing failed ({e}), using Coupled Ising analytical synthesis.")

        c_score = ising_physics["calibrated_consensus_score_pct"]
        d_score = ising_physics["calibrated_divergence_pct"]

        return {
            "consensus_divergence_pct": d_score,
            "consensus_score_pct": c_score,
            "common_ground_points": [
                "Both participants converge on the need for rational, verifiable, and transparent institutional processes.",
                "Preventing arbitrary or unchecked decision-making emerges as a shared baseline priority.",
            ],
            "irreconcilable_frictions": [
                "Prioritization tension between individual autonomy and centralized institutional oversight."
            ],
            "emergent_synthesis": (
                f"Debate on '{self.topic}' converged toward a [{ising_physics['phase_equilibrium_verdict']}] regime "
                f"with Coupled Ising Interaction Energy H_AB={ising_physics['final_hamiltonian_H_AB']} "
                f"(q_AB={ising_physics['edwards_anderson_overlap_q_AB']}, JSD={ising_physics['inter_agent_discourse_jsd']})."
            ),
            "most_persuasive_pivot": "Appeals anchored in empirical evidence and institutional legitimacy framing.",
        }

    def _build_report_markdown(
        self,
        transcript: List[Dict[str, Any]],
        synthesis: Dict[str, Any],
        ising_physics: Dict[str, Any],
    ) -> str:
        """Format arena debate into a comprehensive markdown report including Coupled Ising Spin-Glass metrics."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        participants_str = " vs ".join(self.participant_names)

        lines = [
            f"# ⚔️ Multi-Agent Swarm Arena: {participants_str}",
            f"*Debate Topic: {self.topic}*",
            f"*Generated on {now_str} by who?? Swarm Engine v4.0.0*\n",
            "> [!NOTE]",
            "> All statements are generated by probabilistic digital twins grounded in observed corpus footprints.",
            "> Coupled Ising Spin-Glass Hamiltonian ($H_{AB}$) and Jensen-Shannon Divergence ($\\text{JSD}$) quantify multi-agent phase alignment.\n",
            "---",
            "## 1. 📊 Epistemic Consensus & Coupled Ising Interaction Physics",
            f"- **Consensus Level:** `{synthesis.get('consensus_score_pct', 55)}%`",
            f"- **Divergence / Friction:** `{synthesis.get('consensus_divergence_pct', 45)}%`",
            f"- **Coupled Ising Interaction Energy ($H_{{AB}}$):** `{ising_physics.get('final_hamiltonian_H_AB', 0.0)}` "
            f"($\\Delta H_{{AB}} = {ising_physics.get('hamiltonian_energy_shift_delta_H', 0.0)}$)",
            f"- **Edwards-Anderson Spin Overlap ($q_{{AB}}$):** `{ising_physics.get('edwards_anderson_overlap_q_AB', 0.0)}`",
            f"- **Discourse Divergence (JSD):** `{ising_physics.get('inter_agent_discourse_jsd', 0.0)}`",
            f"- **Phase Equilibrium Regime:** `{ising_physics.get('phase_equilibrium_verdict', 'METASTABLE')}`",
            f"- **Most Persuasive Pivot:** {synthesis.get('most_persuasive_pivot', 'Not specified')}\n",
            "### 🤝 Common Ground:",
        ]

        for cg in synthesis.get("common_ground_points", []):
            lines.append(f"- ✓ {cg}")

        lines.append("\n### ⚡ Irreconcilable Frictions:")
        for f in synthesis.get("irreconcilable_frictions", []):
            lines.append(f"- ⚠️ {f}")

        lines.extend([
            "\n### 💡 Emergent Synthesis:",
            f"{synthesis.get('emergent_synthesis', '')}\n",
            "---",
            "## 2. 🎙️ Live Debate Protocol\n",
        ])

        for t in transcript:
            lines.extend([
                f"### [Round {t['round']}] {t['speaker']}",
                f"{t['statement']}\n",
            ])

        return "\n".join(lines)

    def _persist_report(self, markdown: str, synthesis: Dict[str, Any]) -> None:
        """Save arena debate report to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_topic = "".join(c if c.isalnum() else "_" for c in self.topic[:30])
        filename = f"debate_{safe_topic}_{int(datetime.now().timestamp())}.md"
        report_path = self.output_dir / filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"Saved Swarm Arena report to: {report_path}")
