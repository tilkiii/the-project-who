"""
Temporal Cognitive Drift Engine for who??.
Quantifies mathematical shifts in epistemic calibration, dogmatism density,
lexical diversity (TTR), and concept centrality over time using:
- Ornstein-Uhlenbeck / Fokker-Planck Mean-Reverting Stochastic State Dynamics
  (dx_t = theta * (mu - x_t) dt + sigma dW_t)
- Jensen-Shannon Information Divergence (JSD) between baseline and recent state distributions.
Adheres strictly to probabilistic reasoning without deterministic claims.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Set

from who.config import call_llm
from who.learn.feature_extractor import FeatureExtractor
from who.utils.logger import get_logger

logger = get_logger("who.learn.drift")


def _compute_jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Compute normalized Jensen-Shannon Divergence JSD(P || Q) in [0, 1] (base-2 log)."""
    if not p or not q or len(p) != len(q):
        return 0.0
    eps = 1e-9
    sum_p = sum(p) or 1.0
    sum_q = sum(q) or 1.0
    p_n = [max(eps, x / sum_p) for x in p]
    q_n = [max(eps, x / sum_q) for x in q]
    m = [0.5 * (pi + qi) for pi, qi in zip(p_n, q_n)]
    kl_pm = sum(pi * math.log2(pi / mi) for pi, mi in zip(p_n, m))
    kl_qm = sum(qi * math.log2(qi / mi) for qi, mi in zip(q_n, m))
    return round(min(1.0, max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)), 4)


class TemporalDriftDetector:
    """Detects and quantifies longitudinal cognitive and epistemic drift against baseline profiles."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs
        self.feature_extractor = FeatureExtractor()

    def _load_baseline_metrics(self) -> Dict[str, Any]:
        """Load target baseline features, knowledge graph concepts, and mindset summary."""
        features_file = self.dirs.learned_skills_dir / "features.json"
        baseline_features: Dict[str, Any] = {}
        if features_file.exists():
            try:
                with open(features_file, "r", encoding="utf-8") as f:
                    baseline_features = json.load(f)
            except Exception:
                pass

        graph_concepts: Set[str] = set()
        for fname in ("knowledge_graph.json", "concept_network.json"):
            graph_file = self.dirs.vault_dir / fname
            if graph_file.exists():
                try:
                    with open(graph_file, "r", encoding="utf-8") as f:
                        graph_data = json.load(f)
                        for node in graph_data.get("nodes", []) or graph_data.get("entities", []):
                            concept_name = node.get("name") or node.get("id") or node.get("label") or ""
                            clean_c = str(concept_name).strip().lower()
                            if clean_c and clean_c != self.target_name.lower():
                                graph_concepts.add(clean_c)
                except Exception:
                    pass

        epistemic = baseline_features.get("epistemic_markers", {})
        manip = baseline_features.get("epistemic_manipulation_index", {})
        stats = baseline_features.get("corpus_stats", baseline_features.get("lexical_diversity", {}))
        pronouns = baseline_features.get("pronoun_ratios", baseline_features.get("pronoun_distribution", {}))
        loss_av = baseline_features.get("loss_aversion_index", {})

        raw_terms = baseline_features.get("top_characteristic_terms", [])
        top_terms = [
            (t[0].lower() if isinstance(t, (list, tuple)) else str(t).lower())
            for t in raw_terms
            if t
        ]

        if not graph_concepts and top_terms:
            graph_concepts = set(top_terms[:15])

        return {
            "calibration_ratio": float(epistemic.get("epistemic_calibration_ratio", 1.10)),
            "dogmatic_density": float(manip.get("dogmatic_density_per_k", manip.get("dogmatic_density", 0.45))),
            "type_token_ratio": float(stats.get("type_token_ratio", 0.65)),
            "pronoun_i_pct": float(pronouns.get("first_person_singular_pct", pronouns.get("i_percent", 2.5))),
            "loss_aversion_beta": float(loss_av.get("estimated_score", 1.50)),
            "baseline_concepts": graph_concepts,
            "top_terms": top_terms,
        }

    @staticmethod
    def _compute_ornstein_uhlenbeck_dynamics(
        mu_vec: List[float],
        x_obs_vec: List[float],
        theta: float = 1.25,
        sigma: float = 0.28,
    ) -> Dict[str, Any]:
        """
        Model the cognitive state trajectory under an Ornstein-Uhlenbeck / Fokker-Planck process:
          dx_t = theta * (mu_baseline - x_t) dt + sigma * dW_t
        - Stationary Variance: D_stat = sigma^2 / (2 * theta)
        - Effective Potential Energy Barrier: U_eff(x) = 0.5 * theta * ||x - mu||^2
        - Kramers / Peclet Escape Ratio: Pe = U_eff / D_stat = (theta^2 / sigma^2) * ||x - mu||^2
        """
        deltas = [x - m for m, x in zip(mu_vec, x_obs_vec)]
        l2_disp = math.sqrt(sum(d * d for d in deltas))
        stationary_var = (sigma ** 2) / (2.0 * theta)
        potential_u = 0.5 * theta * (l2_disp ** 2)
        peclet_ratio = round(potential_u / max(stationary_var, 1e-6), 4)

        # Normalize positive state vectors for JSD information divergence
        p_base = [max(0.05, abs(m)) for m in mu_vec]
        q_obs = [max(0.05, abs(x)) for x in x_obs_vec]
        state_jsd = _compute_jensen_shannon_divergence(p_base, q_obs)

        if peclet_ratio >= 2.5 or state_jsd >= 0.08:
            dynamical_regime = "KRAMERS_BARRIER_PHASE_SHIFT"
        elif peclet_ratio >= 0.8:
            dynamical_regime = "NONEQUILIBRIUM_DRIFT_TRANSIENT"
        else:
            dynamical_regime = "OU_MEAN_REVERTING_EQUILIBRIUM"

        return {
            "phase_space_displacement_l2": round(l2_disp, 4),
            "ornstein_uhlenbeck_potential_U": round(potential_u, 4),
            "stationary_diffusion_D": round(stationary_var, 4),
            "peclet_escape_ratio": peclet_ratio,
            "state_distribution_jsd": state_jsd,
            "dynamical_regime": dynamical_regime,
        }

    def detect_drift(
        self,
        new_texts: Optional[List[str]] = None,
        new_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Quantify longitudinal cognitive drift between baseline corpus and incoming text footprint.
        Computes Delta-Calibration, Delta-Dogmatism, Delta-TTR, Concept Centrality Shifts,
        and Ornstein-Uhlenbeck / Fokker-Planck Stochastic State Evolution metrics.
        """
        logger.info(f"Computing temporal cognitive drift for {self.target_name}...")
        baseline = self._load_baseline_metrics()

        observed_corpus: List[str] = []
        if new_texts:
            observed_corpus.extend(new_texts)
        elif new_documents:
            observed_corpus.extend([d.get("content", "") for d in new_documents if d.get("content")])
        else:
            raw_files = list(self.dirs.raw_data_dir.glob("*.md")) + list(self.dirs.raw_data_dir.glob("*.txt"))
            if raw_files:
                raw_files_sorted = sorted(raw_files, key=lambda p: p.stat().st_mtime)
                recent_slice = raw_files_sorted[-max(1, len(raw_files_sorted) // 3):]
                for rf in recent_slice:
                    try:
                        observed_corpus.append(rf.read_text(encoding="utf-8"))
                    except Exception:
                        pass

        if not observed_corpus:
            observed_corpus = [f"Baseline discourse overview for {self.target_name}."]

        obs_features = self.feature_extractor.extract_features(observed_corpus)
        obs_epistemic = obs_features.get("epistemic_markers", {})
        obs_manip = obs_features.get("epistemic_manipulation_index", {})
        obs_stats = obs_features.get("corpus_stats", obs_features.get("lexical_diversity", {}))
        obs_pronouns = obs_features.get("pronoun_ratios", obs_features.get("pronoun_distribution", {}))
        obs_loss = obs_features.get("loss_aversion_index", {})

        obs_cal = float(obs_epistemic.get("epistemic_calibration_ratio", 1.0))
        obs_dogma = float(obs_manip.get("dogmatic_density_per_k", obs_manip.get("dogmatic_density", 0.40)))
        obs_ttr = float(obs_stats.get("type_token_ratio", 0.60))
        obs_i_pct = float(obs_pronouns.get("first_person_singular_pct", obs_pronouns.get("i_percent", 2.0)))
        obs_loss_beta = float(obs_loss.get("estimated_score", 1.50))

        # Calculate mathematical Deltas
        d_cal = round(obs_cal - baseline["calibration_ratio"], 3)
        d_dogma = round(obs_dogma - baseline["dogmatic_density"], 3)
        d_ttr = round(obs_ttr - baseline["type_token_ratio"], 3)
        d_i_pct = round(obs_i_pct - baseline["pronoun_i_pct"], 3)

        # Ornstein-Uhlenbeck / Fokker-Planck phase-space dynamics
        mu_vec = [
            baseline["calibration_ratio"],
            baseline["dogmatic_density"],
            baseline["type_token_ratio"],
            baseline["pronoun_i_pct"] / 10.0,
            baseline["loss_aversion_beta"],
        ]
        x_obs_vec = [
            obs_cal,
            obs_dogma,
            obs_ttr,
            obs_i_pct / 10.0,
            obs_loss_beta,
        ]
        ou_dynamics = self._compute_ornstein_uhlenbeck_dynamics(mu_vec, x_obs_vec)

        # Concept Drift & Salience Analysis
        raw_obs_terms = obs_features.get("top_characteristic_terms", [])
        obs_terms = [
            (t[0].lower() if isinstance(t, (list, tuple)) else str(t).lower())
            for t in raw_obs_terms
            if t
        ]
        base_concepts = baseline["baseline_concepts"]

        surging_concepts = [w for w in obs_terms[:10] if w not in base_concepts]
        retained_concepts = [w for w in obs_terms[:10] if w in base_concepts]

        if base_concepts:
            concept_stability_pct = round(len(retained_concepts) / max(1, min(len(obs_terms[:10]), len(base_concepts))) * 100, 1)
        else:
            concept_stability_pct = 85.0

        # Classify Epistemic Phase Shift
        if d_cal <= -0.40 or d_dogma >= 0.45:
            verdict = "DOGMATIC_HARDENING"
            verdict_desc = "Significant surge in absolutist/unhedged assertions; target exhibiting increased cognitive rigidity or ideological defensiveness."
        elif d_cal >= 0.40:
            verdict = "EPISTEMIC_DIFFUSION"
            verdict_desc = "Elevation in tentative qualifiers and uncertainty markers; indicates increased strategic caution or exploratory hedging."
        elif concept_stability_pct < 40.0:
            verdict = "CONCEPTUAL_PIVOT"
            verdict_desc = "Core thematic vocabulary has shifted significantly from baseline knowledge graph anchors; reflects domain transition or strategic agenda shift."
        else:
            verdict = "STABLE_EQUILIBRIUM"
            verdict_desc = "Cognitive markers, epistemic calibration, and thematic anchors remain closely bounded within learned historical variance."

        def _trend_tag(delta: float, threshold: float = 0.15) -> str:
            if delta > threshold:
                return "▲ SURGING"
            elif delta < -threshold:
                return "▼ DECAYING"
            return "● STABLE"

        metrics_delta = {
            "epistemic_calibration": {
                "baseline": baseline["calibration_ratio"],
                "observed": obs_cal,
                "delta": d_cal,
                "trend": _trend_tag(d_cal, 0.20),
            },
            "dogmatic_density": {
                "baseline": baseline["dogmatic_density"],
                "observed": obs_dogma,
                "delta": d_dogma,
                "trend": _trend_tag(d_dogma, 0.25),
            },
            "lexical_diversity_ttr": {
                "baseline": baseline["type_token_ratio"],
                "observed": obs_ttr,
                "delta": d_ttr,
                "trend": _trend_tag(d_ttr, 0.08),
            },
            "self_focus_pronouns": {
                "baseline": baseline["pronoun_i_pct"],
                "observed": obs_i_pct,
                "delta": d_i_pct,
                "trend": _trend_tag(d_i_pct, 1.0),
            },
        }

        system_prompt = (
            f"You are the Longitudinal Cognitive Drift Analyst for {self.target_name}.\n"
            "Assess cognitive drift scientifically without sensationalism or deterministic overclaims."
        )
        user_prompt = (
            f"TARGET: {self.target_name}\n"
            f"LONGITUDINAL DRIFT AUDIT:\n"
            f"- Epistemic Phase Shift: {verdict} ({verdict_desc})\n"
            f"- Ornstein-Uhlenbeck Dynamical Regime: {ou_dynamics['dynamical_regime']} "
            f"(Péclet Ratio={ou_dynamics['peclet_escape_ratio']}, State JSD={ou_dynamics['state_distribution_jsd']})\n"
            f"- Δ Calibration Ratio: {d_cal} (Baseline: {baseline['calibration_ratio']} -> Observed: {obs_cal})\n"
            f"- Δ Dogmatic Density: {d_dogma} (Baseline: {baseline['dogmatic_density']} -> Observed: {obs_dogma})\n"
            f"- Δ Lexical Diversity (TTR): {d_ttr} (Baseline: {baseline['type_token_ratio']} -> Observed: {obs_ttr})\n"
            f"- Concept Stability: {concept_stability_pct}%\n"
            f"- Newly Surging Concepts: {surging_concepts[:5]}\n\n"
            "Provide:\n"
            "1. Epistemic Trajectory & Phase-Space Analysis\n"
            "2. Strategic Interpretation (Why is this drift occurring?)\n"
            "3. Digital Twin Calibration Guidance (Does the twin model require boundary retraining?)."
        )
        drift_synthesis = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

        return {
            "target_name": self.target_name,
            "verdict": verdict,
            "verdict_description": verdict_desc,
            "concept_stability_pct": concept_stability_pct,
            "metrics_delta": metrics_delta,
            "stochastic_drift_physics": ou_dynamics,
            "conceptual_shifts": {
                "surging_concepts": surging_concepts[:6],
                "retained_concepts": retained_concepts[:6],
            },
            "drift_synthesis": drift_synthesis,
            "epistemic_qualification": (
                "Temporal cognitive drift represents mathematical divergence (Ornstein-Uhlenbeck / JSD) between "
                "baseline corpus features and recent observable footprint. Not a permanent psychological diagnosis."
            ),
        }
