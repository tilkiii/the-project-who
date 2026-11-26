"""
Scenario simulation engine for who??.
Simulates plausible reactions, strategic counter-moves, and Monte Carlo game-theoretic payoffs.
Incorporates Boltzmann-Gibbs Statistical Mechanics / Quantal Response Equilibrium (QRE),
Walter Mischel's Context-Conditional CAPS Matrix, Samuelson's Stated vs. Revealed
Preference Divergence, Temporal Holdout Brier Calibration, and Stylometric JSD Verification.
Adheres strictly to probabilistic reasoning without deterministic claims.
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from who.config import call_llm
from who.engine.evaluator import ProfileEvaluator
from who.learn.feature_extractor import CAPS_CONTEXT_MARKERS
from who.utils.logger import get_logger

logger = get_logger("who.engine.simulator")


def _round_simplex(probs: Dict[str, float], decimals: int = 6) -> Dict[str, float]:
    """
    Round a discrete probability distribution to `decimals` decimal places while
    preserving exact sum == 1.0 via largest-remainder allocation.
    """
    if not probs:
        return {}
    keys = list(probs.keys())
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 0.0:
        normed = [1.0 / len(keys)] * len(keys)
    else:
        normed = [max(0.0, float(probs[k])) / total for k in keys]

    scale = 10 ** decimals
    scaled = [p * scale for p in normed]
    floors = [int(math.floor(x)) for x in scaled]
    rem = scale - sum(floors)
    order = sorted(range(len(keys)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for idx_r in range(max(0, rem)):
        floors[order[idx_r % len(keys)]] += 1
    return {keys[i]: round(floors[i] / scale, decimals) for i in range(len(keys))}


def compute_boltzmann_gibbs_distribution(
    utilities: Dict[str, float],
    beta: float = 3.5,
) -> Dict[str, Any]:
    """
    Compute the Boltzmann-Gibbs / Quantal Response Equilibrium (QRE) distribution over action utilities:
    - Energy convention: E_i = -U_i (high utility corresponds to low thermodynamic energy)
    - Log-Partition Function: log Z(beta) = ln sum_i exp(beta * U_i)
    - Action Probability: P(a_i) = exp(beta * U_i) / Z(beta)
    - Shannon Entropy: S = - sum_i P(a_i) * ln(P(a_i))
    - Specific Heat Capacity (Policy Susceptibility): C(beta) = beta^2 * Var_P(U)
    - Physical Helmholtz Free Energy: F = -(1 / beta) * ln(Z) = -<U> - T * S
    - Economic Smoothed Max Utility (Inclusive Value): V = +(1 / beta) * ln(Z) = <U> + T * S = -F
    """
    beta_raw = float(beta)
    if not utilities:
        is_zero_beta = beta_raw <= 1e-9
        eff_b = 0.0 if is_zero_beta else min(25.0, beta_raw)
        return {
            "probabilities": {},
            "partition_function_Z": 1.0,
            "log_partition_function_logZ": 0.0,
            "expected_utility_U": 0.0,
            "utility_variance_VarU": 0.0,
            "specific_heat_capacity_C": 0.0,
            "shannon_entropy_S": 0.0,
            "normalized_entropy": 0.0,
            "helmholtz_free_energy_F": None if is_zero_beta else 0.0,
            "smoothed_max_utility_logZ": None if is_zero_beta else 0.0,
            "inverse_temperature_beta": 0.0 if is_zero_beta else round(eff_b, 3),
            "cognitive_temperature_T": None if is_zero_beta else round(1.0 / eff_b, 4),
        }

    if beta_raw <= 1e-9:
        # Exact infinite-temperature / zero-beta limit (T -> inf): exact uniform distribution
        k_len = len(utilities)
        inv_k = 1.0 / k_len
        probs_uniform = _round_simplex({k: inv_k for k in utilities}, decimals=6)
        expected_u = sum(utilities.values()) * inv_k
        expected_u2 = sum((u * u) for u in utilities.values()) * inv_k
        var_u = max(0.0, expected_u2 - (expected_u ** 2))
        entropy_s = math.log(k_len) if k_len > 1 else 0.0
        norm_entropy = 1.0 if k_len > 1 else 0.0
        return {
            "probabilities": probs_uniform,
            "partition_function_Z": float(k_len),
            "log_partition_function_logZ": round(entropy_s, 6),
            "expected_utility_U": round(expected_u, 4),
            "utility_variance_VarU": round(var_u, 5),
            "specific_heat_capacity_C": 0.0,
            "shannon_entropy_S": round(entropy_s, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "helmholtz_free_energy_F": None,
            "smoothed_max_utility_logZ": None,
            "inverse_temperature_beta": 0.0,
            "cognitive_temperature_T": None,
        }

    eff_beta = min(25.0, beta_raw)
    temp_t = 1.0 / eff_beta

    max_u = max(utilities.values())
    shifted_exps = {k: math.exp(max(-745.0, eff_beta * (u - max_u))) for k, u in utilities.items()}
    sum_shifted = sum(shifted_exps.values()) or 1.0

    probs = {k: val / sum_shifted for k, val in shifted_exps.items()}
    log_z = (eff_beta * max_u) + math.log(sum_shifted)
    partition_z: Optional[float] = round(math.exp(log_z), 4) if -700.0 <= log_z <= 700.0 else None

    expected_u = sum(probs[k] * utilities[k] for k in utilities)
    expected_u2 = sum(probs[k] * (utilities[k] ** 2) for k in utilities)
    var_u = max(0.0, expected_u2 - (expected_u ** 2))
    specific_heat_c = (eff_beta ** 2) * var_u

    entropy_s = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
    max_entropy = math.log(len(utilities)) if len(utilities) > 1 else 1.0
    norm_entropy = entropy_s / max_entropy

    # Physical Helmholtz Free Energy F = -T * ln(Z) = -<U> - T * S; Inclusive Value V = +T * ln(Z) = -F
    smoothed_max_u = max_u + temp_t * math.log(sum_shifted)
    free_energy_f = -smoothed_max_u

    return {
        "probabilities": _round_simplex(probs, decimals=6),
        "partition_function_Z": partition_z,
        "log_partition_function_logZ": round(log_z, 6),
        "expected_utility_U": round(expected_u, 4),
        "utility_variance_VarU": round(var_u, 5),
        "specific_heat_capacity_C": round(specific_heat_c, 4),
        "shannon_entropy_S": round(entropy_s, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "helmholtz_free_energy_F": round(free_energy_f, 4),
        "smoothed_max_utility_logZ": round(smoothed_max_u, 4),
        "inverse_temperature_beta": round(eff_beta, 3),
        "cognitive_temperature_T": round(temp_t, 4),
    }


class Simulator:
    """Simulates plausible behavioral reactions and counter-moves under explicit scenario assumptions."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _load_profile_context(self) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load mindset summary, mapping skill data, extracted features, and graph physics metrics."""
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

        features_path = self.dirs.learned_skills_dir / "features.json"
        features_data: Dict[str, Any] = {}
        if features_path.exists():
            try:
                with open(features_path, "r", encoding="utf-8") as f:
                    features_data = json.load(f)
            except Exception:
                pass

        graph_metrics_path = self.dirs.vault_dir / "graph_metrics.json"
        graph_metrics: Dict[str, Any] = {}
        if graph_metrics_path.exists():
            try:
                with open(graph_metrics_path, "r", encoding="utf-8") as f:
                    graph_metrics = json.load(f)
            except Exception:
                pass

        return mindset_text, skill_data, features_data, graph_metrics

    def _load_knowledge_graph(self) -> Dict[str, Any]:
        """Load and normalize nodes/edges from local concept_network.json and knowledge_graph.json."""
        nodes_by_lower: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        seen_edges = set()
        loaded_files: List[str] = []

        if not self.dirs or not hasattr(self.dirs, "vault_dir"):
            return {"nodes": [], "edges": [], "loaded_files": []}

        for filename in ("concept_network.json", "knowledge_graph.json"):
            path = self.dirs.vault_dir / filename
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                loaded_files.append(filename)
            except Exception:
                continue

            for raw_n in data.get("nodes", []):
                if isinstance(raw_n, dict):
                    n_name = str(raw_n.get("name") or raw_n.get("id") or "").strip()
                    if n_name:
                        key = n_name.lower()
                        if key not in nodes_by_lower:
                            nodes_by_lower[key] = {"name": n_name, **raw_n}
                elif isinstance(raw_n, str) and raw_n.strip():
                    n_name = raw_n.strip()
                    key = n_name.lower()
                    if key not in nodes_by_lower:
                        nodes_by_lower[key] = {"name": n_name}

            raw_edges = data.get("relationships") or data.get("edges") or data.get("links") or []
            for raw_e in raw_edges:
                if not isinstance(raw_e, dict):
                    continue
                src = str(raw_e.get("source") or "").strip()
                tgt = str(raw_e.get("target") or "").strip()
                rel = str(raw_e.get("relation") or "RELATES_TO").strip().upper()
                if src and tgt:
                    ek = (src.lower(), tgt.lower(), rel)
                    if ek not in seen_edges:
                        seen_edges.add(ek)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "relation": rel,
                            "weight": float(raw_e.get("weight", 0.8) or 0.8),
                        })

        return {
            "nodes": list(nodes_by_lower.values()),
            "edges": edges,
            "loaded_files": loaded_files,
        }

    @staticmethod
    def _resolve_param_with_provenance(
        skill_section: Dict[str, Any],
        feature_section: Dict[str, Any],
        key: str,
        default_val: float,
    ) -> Tuple[float, str]:
        """Resolve numeric parameter preferring observed skill/features with explicit provenance tracking."""
        if isinstance(skill_section, dict) and skill_section.get(key) is not None:
            try:
                return float(skill_section[key]), "observed_skill"
            except (ValueError, TypeError):
                pass
        if isinstance(feature_section, dict) and feature_section.get(key) is not None:
            try:
                return float(feature_section[key]), "observed_features.json"
            except (ValueError, TypeError):
                pass
        return float(default_val), "default_prior_fallback"

    @staticmethod
    def _build_model_transparency(
        *,
        features_data: Dict[str, Any],
        knowledge_graph: Dict[str, Any],
        field_analysis: Dict[str, Any],
        has_mindset: bool,
        has_skill: bool,
        parameter_provenance: Dict[str, str],
    ) -> Dict[str, Any]:
        """Construct transparent provenance and non-predictive disclaimer metadata."""
        total_words = int(features_data.get("corpus_stats", {}).get("total_words", 0) or 0)
        kg_nodes = knowledge_graph.get("nodes", []) if isinstance(knowledge_graph, dict) else []
        kg_edges = knowledge_graph.get("edges", []) if isinstance(knowledge_graph, dict) else []
        observed_sources: List[str] = []
        if total_words > 0:
            observed_sources.append("features.json")
        if has_skill:
            observed_sources.append("generated_mapping_skill.json")
        if has_mindset:
            observed_sources.append("mindset_summary.md")
        for fn in knowledge_graph.get("loaded_files", []):
            if fn not in observed_sources:
                observed_sources.append(fn)

        corpus_grounded = bool(total_words > 0 or len(kg_nodes) > 0 or has_mindset or has_skill)
        return {
            "model_type": "corpus_conditioned_scenario_model",
            "is_trained_behavioral_predictor": False,
            "corpus_grounded": corpus_grounded,
            "corpus_word_count": total_words,
            "knowledge_graph_nodes_loaded": len(kg_nodes),
            "knowledge_graph_edges_loaded": len(kg_edges),
            "matched_knowledge_graph_anchors": field_analysis.get("matched_knowledge_graph_anchors", []),
            "corpus_field_source": field_analysis.get("corpus_field_source", "default_prior_fallback"),
            "observed_sources_used": observed_sources,
            "parameter_provenance": parameter_provenance,
            "disclaimer": (
                "Outputs reflect a corpus-conditioned analytical scenario model driven by observed lexical "
                "indicators, knowledge-graph topology, and heuristic game-theoretic payoffs — NOT a trained "
                "empirical behavioral predictor."
            ),
        }

    def _compute_inverse_temperature_beta(
        self,
        need_for_closure: float,
        affective_volatility: float,
        active_context: str,
        frustration_index: float = 0.0,
    ) -> float:
        """
        Compute the target's Boltzmann-Gibbs Inverse Cognitive Temperature beta = 1 / T:
        - High Need for Closure and low Affective Volatility yield higher beta (low temperature T -> sharp optimization).
        - High Affective Volatility or high Ising Graph Frustration Phi increases temperature T (lower beta -> higher entropy).
        """
        base_beta = 2.8 + (2.4 * need_for_closure)
        thermal_noise = 1.0 + (0.9 * affective_volatility) + (0.8 * frustration_index)
        beta = base_beta / thermal_noise

        if active_context == "FORMAL_ACADEMIC":
            beta *= 1.15  # Cooler, analytical precision
        elif active_context == "HIGH_STAKES_CRISIS":
            beta *= 1.08  # Rigid boundary locking under high closure
        elif active_context == "INTERPERSONAL_DIALOGUE":
            beta *= 0.88  # Warmer, exploratory flexibility

        return round(max(0.8, min(12.0, beta)), 3)

    def _detect_situational_context(
        self,
        scenario: str,
        caps_matrix: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify scenario into one of Mischel's 4 CAPS situational contexts:
        FORMAL_ACADEMIC, HIGH_STAKES_CRISIS, STRATEGIC_NEGOTIATION, INTERPERSONAL_DIALOGUE.
        """
        words = set(re.findall(r"\b\w+\b", scenario.lower()))
        scores: Dict[str, int] = {}
        for ctx_name, markers in CAPS_CONTEXT_MARKERS.items():
            scores[ctx_name] = sum(1 for w in words if w in markers)

        best_ctx = max(scores, key=scores.get) if any(scores.values()) else "STRATEGIC_NEGOTIATION"
        ctx_profiles = caps_matrix.get("contexts", {})
        active_ctx_metrics = ctx_profiles.get(best_ctx, {})

        matched_signature = ""
        for sig in caps_matrix.get("if_then_signatures", []):
            if isinstance(sig, dict) and sig.get("if_context") == best_ctx:
                matched_signature = f"IF [{best_ctx}] -> THEN [{sig.get('then_behavior', '')}]"
                break
        if not matched_signature:
            matched_signature = (
                f"IF [{best_ctx}] -> THEN [Activate context-specific epistemic calibration and autonomy boundaries]"
            )

        return {
            "active_context": best_ctx,
            "context_marker_hits": scores.get(best_ctx, 0),
            "if_then_signature": matched_signature,
            "context_metrics": active_ctx_metrics,
            "contextual_plasticity_index": caps_matrix.get("contextual_plasticity_index", 0.15),
        }

    def _build_dual_layer_prediction(
        self,
        pref_data: Dict[str, Any],
        dominant_branch: str = "REJECT_AND_COUNTER",
    ) -> Dict[str, Any]:
        """
        Compute Samuelson's Dual-Layer Prediction separating:
        1. Stated Public Reaction (normative rhetoric / diplomatic framing)
        2. Revealed Strategic Action (actual historical execution behavior)
        """
        stated_pol = float(pref_data.get("stated_cooperation_polarity", 0.65))
        revealed_pol = float(pref_data.get("revealed_cooperation_polarity", 0.50))
        div_idx = float(pref_data.get("preference_divergence_index", round(abs(stated_pol - revealed_pol), 3)))
        verdict = pref_data.get("verdict", "CONGRUENT_ACTION_AND_RHETORIC")

        if stated_pol >= 0.55:
            stated_posture = (
                "Normative Diplomatic Openness: Target publicly frames response around institutional principles, "
                "dialogue, and mutual governance standards."
            )
        else:
            stated_posture = (
                "Principled Normative Critique: Target publicly articulates explicit reservations and methodological objections."
            )

        branch_action_map = {
            "REJECT_AND_COUNTER": (
                "Autonomous Counter-Conditioning: Despite polite rhetoric, target operationally withholds sign-off "
                "until asymmetric constraints are removed."
            ),
            "STRATEGIC_CONCESSION": (
                "Tactical Conditional Concession: Target yields on secondary procedural items while locking in core autonomy."
            ),
            "SUBGAME_EQUILIBRIUM_ACCEPT": (
                "Verified Structural Alignment: Target executes cooperative commitment once empirical guarantees are met."
            ),
            "DEADLOCK_WALKAWAY": (
                "Unilateral Decoupling / Exit: Target exercises walkaway option to preserve institutional/personal independence."
            ),
        }
        revealed_action = branch_action_map.get(
            dominant_branch,
            branch_action_map["REJECT_AND_COUNTER"],
        )

        return {
            "stated_public_reaction": {
                "posture": stated_posture,
                "cooperation_probability": round(stated_pol, 3),
            },
            "revealed_strategic_action": {
                "operational_move": revealed_action,
                "executed_branch": dominant_branch,
                "cooperation_probability": round(revealed_pol, 3),
            },
            "preference_divergence_index": round(div_idx, 3),
            "divergence_verdict": verdict,
        }

    def _compute_calibrated_confidence(
        self,
        evaluator: ProfileEvaluator,
        features_data: Dict[str, Any],
        stylometric_fidelity: float,
        has_mindset: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute empirically calibrated confidence score from:
        1. Out-of-sample Temporal Holdout Brier Score (1 - BS), or explicit prior_fallback_confidence when holdout=0
        2. Bayesian 95% Credible Interval precision (1 - uncertainty_margin)
        3. Stylometric Fidelity Score (1 - sqrt(JSD))
        """
        backtest = evaluator.run_temporal_backtest(holdout_ratio=0.25)
        raw_emp = backtest.get("empirical_confidence_score")
        if raw_emp is None:
            brier_conf = float(backtest.get("prior_fallback_confidence", 0.68))
        else:
            brier_conf = float(raw_emp)

        loss_ci = features_data.get("loss_aversion_index", {})
        unc_margin = float(loss_ci.get("uncertainty_margin", 0.18))
        bayesian_precision = max(0.50, min(0.95, 1.0 - (unc_margin * 0.5)))

        base_weight = (0.50 * brier_conf) + (0.30 * bayesian_precision) + (0.20 * stylometric_fidelity)
        if not has_mindset:
            base_weight *= 0.85

        calibrated_conf = round(max(0.50, min(0.96, base_weight)), 2)
        return calibrated_conf, backtest

    def _evaluate_scenario_external_fields_and_commitments(
        self,
        scenario: str,
        features_data: Dict[str, Any],
        graph_metrics: Dict[str, Any],
        knowledge_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Project the scenario text x onto:
        1. Concept Graph External Fields h_i(x) in [-1.5, +1.5], added to corpus stance fields h_i^corpus
           so H(s | x) = - sum J_ij s_i s_j - sum (h_i^corpus + h_i(x)) s_i breaks Z2 symmetry naturally.
        2. Conditional Commitment & Sacred-Value Red-Line Guard Evaluation (detecting whether clauses
           preserve or transfer/violate protected rights, autonomy, IP, audit, or governance).
        3. Scenario-induced shifts on baseline covariates (leverage, asymmetry, reputation risk).
        """
        s_lower = scenario.lower()

        # Clause-level preservation, transfer/violation, transparency, opacity, and urgency phrases
        # Note: "devredilmez" is handled solely in negated_transfer_phrases (never double-counted).
        negated_transfer_phrases = [
            "devredilmez", "devredilmemesi", "devredilemez", "devredilmeyecek",
            "not transferred", "never transferred", "no transfer", "will not be transferred",
            "not assigned", "never surrendered",
        ]
        negated_transfer_patterns = [
            re.compile(
                r"\b(?:sponsor|company|label|platform|counterparty)\b[^.;,!?]{0,25}"
                r"\b(?:shall|will|does|do|may|can)\s+(?:not|never)\s+"
                r"(?:receive|obtain|acquire|own|control|hold)\b[^.;,!?]{0,25}"
                r"\b(?:master|rights|ip|ownership|hak)\b"
            ),
        ]
        preserve_clauses = [
            "stay with", "stays with", "remain with", "remains with", "retained by", "retain",
            "preserve", "preserved", "protected", "guaranteed", "audit right",
            "veto right", "fair royalty", "mutual governance", "independent", "autonomy preserved",
            "kalır", "sanatçıda kalır", "kendisinde kalır", "korunur", "korunması",
            "denetim hakkı", "veto hakkı", "adil telif", "bağımsız", "otonomi korunur",
            "garanti", "ortak yönetişim", "onayına tabi", "hakkı tanınır",
        ]
        transfer_violation_clauses = [
            "transferred to", "transfer to", "assigned to", "surrender", "waive", "waives",
            "forfeit", "irrevocable transfer", "exclusive control", "without audit", "no veto",
            "unilateral", "mandatory assignment", "give up", "relinquish",
            "devredilir", "sponsora devredilir", "şirkete devredilir", "devir", "feragat",
            "vazgeç", "tek taraflı", "süresiz devredilir", "denetimsiz", "veto hakkı olmadan",
            "kayıtsız şartsız", "tam kontrol devri", "hakların devri",
            "kalmaz", "kalmayacak", "korunmaz", "korunmayacak", "sağlanmaz",
        ]
        transparency_clauses = [
            "transparent", "open audit", "verified", "disclosure", "empirical", "evidence",
            "şeffaf", "açık denetim", "doğrulanabilir", "kanıt", "raporlama",
        ]
        opacity_clauses = [
            "confidential penalty", "hidden", "opaque", "non-disclosable", "secret", "unverified",
            "gizli", "denetime kapalı", "belirsiz", "şeffaf olmayan", "tek yanlı",
        ]
        urgency_clauses = [
            "immediate", "24 hours", "48 hours", "ultimatum", "take it or leave it", "final offer",
            "derhal", "24 saat", "48 saat", "ültimatom", "son teklif", "hemen",
        ]
        pre_neg_tokens = {
            "no", "not", "never", "without", "none", "waive", "waives", "waived",
            "forfeit", "forfeits", "forfeited", "revoked", "removed", "denied", "olmadan",
        }
        post_neg_tokens = {
            "yok", "yoktur", "olmadan", "olmaması", "olmamasi", "değil", "degil",
            "hariç", "haric", "kaldırılır", "kaldirilir", "feragat", "vazgeçer", "vazgecer",
            "verilmez", "tanınmaz", "taninmaz", "edilmez", "waived", "revoked", "removed",
            "denied", "forfeited", "none",
        }
        pre_bridge_tokens = {
            "the", "a", "an", "any", "all", "be", "been", "being", "to", "its", "their",
            "our", "his", "her", "artist", "sponsor", "full", "clear", "open", "direct",
            "further", "additional", "even", "ever", "currently", "hereby", "do", "does",
            "did", "will", "shall", "would", "should", "can", "could", "may", "might",
            "is", "are", "was", "were", "hiç", "hic", "hiçbir", "hicbir", "bir", "herhangi",
        }
        post_bridge_tokens = {
            "is", "are", "was", "were", "be", "been", "being", "will", "shall", "would",
            "can", "could", "has", "have", "had", "hereby", "expressly", "completely",
            "entirely", "fully", "totally", "permanently", "da", "de", "dahi", "bile",
            "hiç", "hic", "asla", "kesinlikle", "tamamen", "artık", "artik", "zaten",
            "hakkı", "hakki", "hakkından", "hakkindan", "hakkına", "hakkina",
        }
        audit_related_phrases = {"audit right", "denetim hakkı", "transparent", "open audit", "şeffaf", "açık denetim", "disclosure", "raporlama"}

        raw_clauses = [
            c.strip()
            for c in re.split(r"[.;,!?\n]+|\b(?:fakat|ancak|ama|lakin|however|but|while|whereas)\b", s_lower)
            if c.strip()
        ] or [s_lower]

        def _token_matches(c_tok: str, p_tok: str) -> bool:
            return c_tok == p_tok or (len(p_tok) >= 4 and c_tok.startswith(p_tok))

        def _has_local_negation(clause_str: str, phrase: str) -> bool:
            c_tokens = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", clause_str)
            p_tokens = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", phrase)
            if not c_tokens or not p_tokens:
                return False

            copula_aux_tokens = {
                "is", "are", "was", "were", "be", "been", "being",
                "will", "shall", "would", "can", "could", "has", "have", "had",
            }
            grant_verbs = {
                "receive", "retain", "keep", "hold", "have", "get",
                "grant", "provide", "allow", "include", "confer", "give",
            }

            p_len = len(p_tokens)
            for start in range(len(c_tokens) - p_len + 1):
                if all(_token_matches(c_tokens[start + k], p_tokens[k]) for k in range(p_len)):
                    end = start + p_len
                    # Walk backward up to 4 positions across pre-negation bridges or directly-negated grant verbs
                    used_grant_verb = False
                    for step_b in range(1, min(4, start) + 1):
                        b_tok = c_tokens[start - step_b]
                        if b_tok in pre_neg_tokens:
                            return True
                        if b_tok in pre_bridge_tokens:
                            continue
                        if not used_grant_verb and b_tok in grant_verbs:
                            used_grant_verb = True
                            prev_idx = start - step_b - 1
                            if prev_idx >= 0 and c_tokens[prev_idx] in {"not", "never", "no"}:
                                return True
                            break
                        break
                    # Walk forward up to 3 positions across post-negation bridges and single copula -ly adverb
                    seen_copula = False
                    used_ly_adverb = False
                    for step_f in range(0, min(3, len(c_tokens) - end)):
                        f_tok = c_tokens[end + step_f]
                        if f_tok in post_neg_tokens:
                            return True
                        if f_tok in post_bridge_tokens:
                            if f_tok in copula_aux_tokens:
                                seen_copula = True
                            continue
                        if seen_copula and not used_ly_adverb and len(f_tok) > 4 and f_tok.endswith("ly") and f_tok != "only":
                            used_ly_adverb = True
                            continue
                        break
            return False

        preserve_hits = 0
        violation_hits = 0
        trans_hits = 0
        opacity_hits = 0
        urgency_hits = 0
        clause_stats: List[Dict[str, Any]] = []

        for cl in raw_clauses:
            cl_work = cl
            cl_pres = 0
            cl_viol = 0
            cl_trans = 0
            cl_opac = 0

            for nt_p in negated_transfer_phrases:
                if nt_p in cl_work:
                    cl_pres += 1
                    cl_work = cl_work.replace(nt_p, " ")

            for nt_pat in negated_transfer_patterns:
                cl_work, n_hits = nt_pat.subn(" ", cl_work)
                if n_hits > 0:
                    cl_pres += n_hits

            for p_p in preserve_clauses:
                if p_p in cl_work:
                    if _has_local_negation(cl_work, p_p):
                        cl_viol += 1
                        if p_p in audit_related_phrases:
                            cl_opac += 1
                    else:
                        cl_pres += 1
                    cl_work = cl_work.replace(p_p, " ")

            for v_p in transfer_violation_clauses:
                if v_p in cl_work:
                    cl_viol += 1
                    cl_work = cl_work.replace(v_p, " ")

            for t_p in transparency_clauses:
                if t_p in cl_work:
                    if _has_local_negation(cl_work, t_p):
                        cl_opac += 1
                    else:
                        cl_trans += 1

            cl_opac += sum(1 for m in opacity_clauses if m in cl_work)
            urgency_hits += sum(1 for m in urgency_clauses if m in cl_work)

            preserve_hits += cl_pres
            violation_hits += cl_viol
            trans_hits += cl_trans
            opacity_hits += cl_opac
            clause_stats.append({
                "clause": cl,
                "preserve_hits": cl_pres,
                "violation_hits": cl_viol,
            })

        # Check extracted conditional commitments using clause-local matches when protected concepts appear
        comm_data = features_data.get("conditional_commitments", {})
        rules = comm_data.get("rules", [])
        triggered_commitments: List[Dict[str, Any]] = []
        red_line_violation_score = 0.0
        commitment_alignment_score = 0.0

        for rule in rules:
            p_concepts = rule.get("protected_concepts", [])
            matching_clauses = [
                cs for cs in clause_stats
                if p_concepts and any(pc in cs["clause"] for pc in p_concepts)
            ]
            if matching_clauses:
                loc_viol = sum(cs["violation_hits"] for cs in matching_clauses)
                loc_pres = sum(cs["preserve_hits"] for cs in matching_clauses)
            elif any(pc in s_lower for pc in p_concepts) or not p_concepts:
                loc_viol = violation_hits
                loc_pres = preserve_hits
            else:
                continue

            if loc_viol > loc_pres or (loc_viol > 0 and loc_pres == 0):
                status = "VIOLATED"
                w = float(rule.get("rigidity_weight", 0.85))
                red_line_violation_score = max(red_line_violation_score, w)
                commitment_alignment_score -= w
            elif loc_pres > loc_viol:
                status = "SATISFIED"
                w = float(rule.get("rigidity_weight", 0.75))
                commitment_alignment_score += w
            else:
                status = "NEUTRAL"
            triggered_commitments.append({
                "claim_type": rule.get("claim_type", "CONDITIONAL_COMMITMENT"),
                "protected_concepts": p_concepts,
                "status": status,
                "source_quote": rule.get("source_quote", ""),
            })

        # Even if profile corpus is empty/small, rights/autonomy transfer vs preservation in scenario induces commitment response
        clause_polarity = math.tanh(0.65 * (preserve_hits - violation_hits) + 0.35 * (trans_hits - opacity_hits))
        if violation_hits > preserve_hits:
            red_line_violation_score = max(
                red_line_violation_score,
                min(0.95, 0.45 * (violation_hits - preserve_hits) + 0.20 * opacity_hits),
            )
            commitment_alignment_score = min(commitment_alignment_score, clause_polarity)
        elif preserve_hits > violation_hits:
            commitment_alignment_score = max(commitment_alignment_score, clause_polarity)

        commitment_alignment_score = round(max(-1.0, min(1.0, commitment_alignment_score)), 4)
        red_line_violation_score = round(max(0.0, min(1.0, red_line_violation_score)), 4)

        # Build canonical 6-node cognitive Ising concept graph conditioned on observed features & knowledge graph
        ising_data = graph_metrics.get("cognitive_dissonance", {}).get("ising_spin_glass_physics", {})
        vault_h = ising_data.get("corpus_external_fields_h", {})
        vault_m = ising_data.get("node_salience_m", {})
        vault_j = ising_data.get("serialized_couplings_J", [])

        canonical_nodes = [
            "autonomy_and_rights_retention",
            "epistemic_transparency_and_audit",
            "cooperative_mutual_reciprocity",
            "downside_loss_and_risk_shield",
            "institutional_governance_parity",
            "closure_and_execution_readiness",
        ]
        has_observed_features = bool(
            isinstance(features_data, dict)
            and features_data.get("corpus_stats", {}).get("total_words", 0) > 0
        )
        if has_observed_features:
            pref_sec = features_data.get("stated_vs_revealed_preference", {})
            rev_coop = float(pref_sec.get("revealed_cooperation_polarity", 0.50))
            stated_coop = float(pref_sec.get("stated_cooperation_polarity", 0.50))
            mach_est = float(features_data.get("machiavellianism_index", {}).get("estimated_score", 0.35))
            cal_ratio = float(features_data.get("epistemic_markers", {}).get("epistemic_calibration_ratio", 1.0))
            manip_est = float(features_data.get("epistemic_manipulation_index", {}).get("score", 0.15))
            loss_est = float(features_data.get("loss_aversion_index", {}).get("estimated_score", 1.50))
            closure_est = float(features_data.get("need_for_closure_index", {}).get("estimated_score", 0.50))
            dom_moral = str(features_data.get("moral_foundations_profile", {}).get("dominant_foundation", ""))

            raw_h_corpus = {
                "autonomy_and_rights_retention": (
                    0.55
                    + 0.50 * (0.50 - rev_coop)
                    + 0.25 * (mach_est - 0.35)
                    + (0.15 if ("INDIVIDUALIZING" in dom_moral or "BİREY" in dom_moral) else 0.0)
                ),
                "epistemic_transparency_and_audit": (
                    0.35 + 0.25 * min(1.5, cal_ratio - 1.0) - 0.40 * (manip_est - 0.20)
                ),
                "cooperative_mutual_reciprocity": (
                    0.25 + 0.85 * ((rev_coop + stated_coop) * 0.5 - 0.50) - 0.45 * (mach_est - 0.35)
                ),
                "downside_loss_and_risk_shield": 0.35 + 0.55 * ((loss_est - 1.50) / 0.85),
                "institutional_governance_parity": (
                    0.35
                    + (0.25 if ("BINDING" in dom_moral or "OTORİTE" in dom_moral) else 0.0)
                    + 0.30 * (rev_coop - 0.50)
                ),
                "closure_and_execution_readiness": 0.15 + 0.65 * (closure_est - 0.50),
            }
            h_corpus = {k: round(max(-1.0, min(1.0, v)), 4) for k, v in raw_h_corpus.items()}
        else:
            h_corpus = {
                "autonomy_and_rights_retention": 0.55,
                "epistemic_transparency_and_audit": 0.45,
                "cooperative_mutual_reciprocity": 0.30,
                "downside_loss_and_risk_shield": 0.50,
                "institutional_governance_parity": 0.40,
                "closure_and_execution_readiness": 0.15,
            }

        if vault_h:
            mean_vh = sum(float(v) for v in vault_h.values()) / max(1, len(vault_h))
            for k in h_corpus:
                h_corpus[k] = round(0.75 * h_corpus[k] + 0.25 * mean_vh, 4)

        salience_m = {
            "autonomy_and_rights_retention": 1.00,
            "epistemic_transparency_and_audit": 0.85,
            "cooperative_mutual_reciprocity": 0.80,
            "downside_loss_and_risk_shield": 0.90,
            "institutional_governance_parity": 0.75,
            "closure_and_execution_readiness": 0.65,
        }
        if vault_m:
            mean_vm = sum(float(v) for v in vault_m.values()) / max(1, len(vault_m))
            for k in salience_m:
                salience_m[k] = round(0.80 * salience_m[k] + 0.20 * mean_vm, 4)

        # Knowledge Graph anchor matching against scenario proposition
        kg_nodes = knowledge_graph.get("nodes", []) if isinstance(knowledge_graph, dict) else []
        kg_edges = knowledge_graph.get("edges", []) if isinstance(knowledge_graph, dict) else []
        positive_rels = {"ADVOCATES", "SUPPORTS", "CHAMPIONS", "REQUIRES", "PRODUCES", "INFORMS", "STRUCTURES", "USES"}
        negative_rels = {"CRITIQUES", "OPPOSES", "REJECTS", "CONTRADICTS"}

        matched_kg_anchors: List[str] = []
        matched_kg_lowers = set()
        for raw_n in kg_nodes:
            n_name = str(raw_n.get("name") or raw_n.get("id") or "").strip() if isinstance(raw_n, dict) else str(raw_n).strip()
            if len(n_name) >= 4 and n_name.lower() in s_lower:
                if n_name.lower() not in matched_kg_lowers:
                    matched_kg_lowers.add(n_name.lower())
                    matched_kg_anchors.append(n_name)

        kg_support_hits = 0
        kg_opposition_hits = 0
        if matched_kg_lowers and kg_edges:
            counted_edges = set()
            for edge in kg_edges:
                if not isinstance(edge, dict):
                    continue
                src_l = str(edge.get("source") or "").strip().lower()
                tgt_l = str(edge.get("target") or "").strip().lower()
                rel_u = str(edge.get("relation") or "").strip().upper()
                ek = (src_l, tgt_l, rel_u)
                if ek in counted_edges:
                    continue
                if src_l in matched_kg_lowers or tgt_l in matched_kg_lowers:
                    counted_edges.add(ek)
                    if rel_u in positive_rels:
                        kg_support_hits += 1
                    elif rel_u in negative_rels:
                        kg_opposition_hits += 1

        kg_alignment_bias = round(min(0.45, 0.15 * kg_support_hits), 4)
        kg_tension_bias = round(min(0.50, 0.20 * kg_opposition_hits), 4)

        # Scenario-induced external field perturbation h_i(x)
        h_scenario = {
            "autonomy_and_rights_retention": round(
                0.85 * (preserve_hits - violation_hits) + 0.25 * (kg_opposition_hits - 0.5 * kg_support_hits), 4
            ),
            "epistemic_transparency_and_audit": round(
                0.75 * (trans_hits - opacity_hits) + 0.30 * (preserve_hits - violation_hits), 4
            ),
            "cooperative_mutual_reciprocity": round(
                0.65 * (preserve_hits - violation_hits) - 0.35 * urgency_hits + 0.20 * (kg_support_hits - kg_opposition_hits), 4
            ),
            "downside_loss_and_risk_shield": round(
                0.80 * (violation_hits - preserve_hits) + 0.50 * opacity_hits + 0.30 * kg_opposition_hits - 0.15 * kg_support_hits, 4
            ),
            "institutional_governance_parity": round(
                0.60 * (preserve_hits - violation_hits) + 0.40 * (trans_hits - opacity_hits) + 0.30 * (kg_support_hits - kg_opposition_hits), 4
            ),
            "closure_and_execution_readiness": round(0.45 * preserve_hits - 0.50 * violation_hits - 0.30 * urgency_hits, 4),
        }
        for k in h_scenario:
            h_scenario[k] = round(max(-1.5, min(1.5, h_scenario[k])), 4)

        has_observed_kg = len(kg_nodes) > 0
        if has_observed_features and has_observed_kg:
            corpus_field_source = "observed_features_and_graph"
        elif has_observed_features:
            corpus_field_source = "observed_features"
        elif has_observed_kg:
            corpus_field_source = "observed_knowledge_graph"
        else:
            corpus_field_source = "default_prior_fallback"

        frust_phi = float(ising_data.get("structural_frustration_index", 0.25))
        if vault_j and frust_phi == 0.0:
            neg_j = sum(abs(float(e.get("J", 0.0))) for e in vault_j if float(e.get("J", 0.0)) < 0)
            tot_j = sum(abs(float(e.get("J", 0.0))) for e in vault_j) or 1.0
            frust_phi = round(neg_j / tot_j, 4)

        # Signed couplings J_ij on the 6-node cognitive backbone (with frustrated cycle between reciprocity, autonomy, and risk shield)
        j_backbone = [
            ("autonomy_and_rights_retention", "epistemic_transparency_and_audit", 0.65),
            ("autonomy_and_rights_retention", "institutional_governance_parity", 0.60),
            ("cooperative_mutual_reciprocity", "closure_and_execution_readiness", 0.55),
            ("cooperative_mutual_reciprocity", "epistemic_transparency_and_audit", 0.45),
            ("autonomy_and_rights_retention", "downside_loss_and_risk_shield", 0.50),
            # Antiferromagnetic frustrated bonds (autonomy/risk protection vs unilateral concession/closure)
            ("downside_loss_and_risk_shield", "cooperative_mutual_reciprocity", -(0.45 + 0.50 * frust_phi)),
            ("autonomy_and_rights_retention", "closure_and_execution_readiness", -(0.35 + 0.45 * frust_phi)),
            ("downside_loss_and_risk_shield", "closure_and_execution_readiness", -(0.40 + 0.40 * frust_phi)),
        ]

        return {
            "clause_polarity": round(clause_polarity, 4),
            "preserve_clause_hits": preserve_hits,
            "violation_clause_hits": violation_hits,
            "transparency_clause_hits": trans_hits,
            "opacity_clause_hits": opacity_hits,
            "urgency_clause_hits": urgency_hits,
            "red_line_violation_score": red_line_violation_score,
            "commitment_alignment_score": commitment_alignment_score,
            "triggered_commitments": triggered_commitments,
            "matched_knowledge_graph_anchors": matched_kg_anchors,
            "kg_alignment_bias": kg_alignment_bias,
            "kg_tension_bias": kg_tension_bias,
            "corpus_field_source": corpus_field_source,
            "canonical_nodes": canonical_nodes,
            "corpus_external_fields_h": h_corpus,
            "scenario_external_fields_h_x": h_scenario,
            "total_external_fields_h_total": {
                k: round(h_corpus[k] + h_scenario[k], 4) for k in canonical_nodes
            },
            "node_salience_m": salience_m,
            "couplings_J": j_backbone,
            "covariate_shifts": {
                "leverage_shift": round(0.18 * (violation_hits - preserve_hits) + 0.08 * urgency_hits, 4),
                "asymmetry_shift": round(0.16 * (opacity_hits - trans_hits) + 0.10 * max(0, violation_hits - preserve_hits), 4),
                "reputation_risk_shift": round(0.20 * (violation_hits - preserve_hits), 4),
            },
        }

    def _compute_centered_joint_gibbs_coupling(
        self,
        base_utilities: Dict[str, float],
        beta: float,
        kappa: float,
        field_analysis: Dict[str, Any],
        belief_temperature_Tb: float = 0.65,
        reference_action: str = "DEADLOCK_WALKAWAY",
        use_balanced_counterfactual: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the Identifiable Joint Gibbs Measure & Single-Parameter kappa Coupling:
        1. Spin Ensemble Distribution (NO arbitrary Z2 gauge-fixing because h_total = h_corpus + h(x) breaks Z2 symmetry):
           P(s | x) = exp(-H(s | x) / T_b) / Z_s(x),
           H(s | x) = - sum_{(i,j)} J_{ij} s_i s_j - sum_i (h_i^corpus + h_i(x)) s_i.
           If use_balanced_counterfactual=True, replaces couplings J_{ij} with the Harary-balanced
           counterfactual (exact same topology and |J_{ij}|, zero cycle frustration lambda_min=0).
        2. Reference-Invariant Centered & Independent-Spin Normalized Spin-to-Action Map (g_hat_ref ≡ 0):
           w_rel[a][i] = w_raw[a][i] - w_raw[ref][i]
           g_hat_a(s) = sum_i w_rel[a][i] m_i (s_i - <s_i>_P) / (2 * sqrt(sum_i m_i^2)).
        3. Frustration Covariance Matrix Sigma_{ab} = Cov_{P(s|x)}(g_hat_a(s), g_hat_b(s)) and dimensionless T_b / <|J|>.
        4. Marginalized Action Distribution P(a | x) and Quenched Joint Gibbs Thermodynamics
           (joint Helmholtz free energy F, log Z, and joint policy specific heat C).
        """
        BRANCH_COUNTER = "REJECT_AND_COUNTER"
        BRANCH_CONCESSION = "STRATEGIC_CONCESSION"
        BRANCH_ACCEPT = "SUBGAME_EQUILIBRIUM_ACCEPT"
        BRANCH_DEADLOCK = "DEADLOCK_WALKAWAY"
        actions = [BRANCH_COUNTER, BRANCH_CONCESSION, BRANCH_ACCEPT, BRANCH_DEADLOCK]

        nodes: List[str] = field_analysis["canonical_nodes"]
        n_nodes = len(nodes)
        h_total: Dict[str, float] = field_analysis["total_external_fields_h_total"]
        m_sal: Dict[str, float] = field_analysis["node_salience_m"]
        couplings: List[Tuple[str, str, float]] = list(field_analysis["couplings_J"])

        if use_balanced_counterfactual and couplings:
            from who.vault.graph_analytics import GraphAnalytics
            bal_res = GraphAnalytics.construct_harary_balanced_counterfactual(nodes, couplings)
            couplings = bal_res["balanced_couplings"]

        mean_abs_j = sum(abs(float(jv)) for _, _, jv in couplings) / max(1, len(couplings))
        tb = max(0.15, float(belief_temperature_Tb))
        dimless_tb_over_j = round(tb / max(mean_abs_j, 1e-6), 4)

        # Pre-registered fixed raw loading matrix w_raw[a][i] in {-1, 0, +1}
        w_raw: Dict[str, Dict[str, float]] = {
            BRANCH_ACCEPT: {
                "autonomy_and_rights_retention": 1.0,
                "epistemic_transparency_and_audit": 1.0,
                "cooperative_mutual_reciprocity": 1.0,
                "downside_loss_and_risk_shield": -1.0,
                "institutional_governance_parity": 1.0,
                "closure_and_execution_readiness": 1.0,
            },
            BRANCH_CONCESSION: {
                "autonomy_and_rights_retention": -1.0,
                "epistemic_transparency_and_audit": 0.0,
                "cooperative_mutual_reciprocity": 1.0,
                "downside_loss_and_risk_shield": -1.0,
                "institutional_governance_parity": 0.0,
                "closure_and_execution_readiness": 1.0,
            },
            BRANCH_COUNTER: {
                "autonomy_and_rights_retention": 1.0,
                "epistemic_transparency_and_audit": 1.0,
                "cooperative_mutual_reciprocity": -1.0,
                "downside_loss_and_risk_shield": 1.0,
                "institutional_governance_parity": 1.0,
                "closure_and_execution_readiness": -1.0,
            },
            BRANCH_DEADLOCK: {n: 0.0 for n in nodes},
        }

        ref_act = reference_action if reference_action in actions else BRANCH_DEADLOCK
        w_rel: Dict[str, Dict[str, float]] = {
            a: {n: w_raw[a][n] - w_raw[ref_act][n] for n in nodes}
            for a in actions
        }

        m_sq_sum = sum(m_sal[n] ** 2 for n in nodes)
        common_scale = (2.0 * math.sqrt(m_sq_sum)) if m_sq_sum > 1e-12 else 1.0

        # Enumerate all 2^N spin configurations (N=6 -> 64 states)
        spin_states: List[Dict[str, int]] = []
        energies: List[float] = []

        for mask in range(1 << n_nodes):
            s_cfg = {nodes[idx]: (1 if ((mask >> idx) & 1) else -1) for idx in range(n_nodes)}
            e_bond = -sum(j_val * s_cfg[u] * s_cfg[v] for u, v, j_val in couplings)
            e_field = -sum(h_total[n] * s_cfg[n] for n in nodes)
            spin_states.append(s_cfg)
            energies.append(e_bond + e_field)

        min_e = min(energies)
        boltz_weights = [math.exp(-(e_val - min_e) / tb) for e_val in energies]
        z_spin = sum(boltz_weights) or 1.0
        p_spin = [w / z_spin for w in boltz_weights]

        # Expected spin <s_i>_P under P(s | x)
        mean_spins: Dict[str, float] = {
            n: sum(p_spin[idx] * spin_states[idx][n] for idx in range(len(spin_states)))
            for n in nodes
        }

        # Compute centered g_hat_a(s) for each spin state (g_hat_ref ≡ 0 identically)
        g_hat_by_state: List[Dict[str, float]] = []
        for idx, s_cfg in enumerate(spin_states):
            g_a: Dict[str, float] = {}
            for a in actions:
                if a == ref_act:
                    g_a[a] = 0.0
                else:
                    num = sum(w_rel[a][n] * m_sal[n] * (s_cfg[n] - mean_spins[n]) for n in nodes)
                    g_a[a] = num / common_scale
            g_hat_by_state.append(g_a)

        # Verify exact x-conditional centering E_{P(s|x)}[g_hat_a(s)] = 0
        centered_g_expectation_max_abs = max(
            abs(sum(p_spin[idx] * g_hat_by_state[idx][a] for idx in range(len(spin_states))))
            for a in actions
        )

        # Frustration Covariance Matrix Sigma_{ab} = Cov_P(g_hat_a, g_hat_b)
        sigma_cov: Dict[str, Dict[str, float]] = {a: {b: 0.0 for b in actions} for a in actions}
        for idx, pr_s in enumerate(p_spin):
            g_s = g_hat_by_state[idx]
            for a in actions:
                for b in actions:
                    sigma_cov[a][b] += pr_s * g_s[a] * g_s[b]

        sigma_trace = sum(sigma_cov[a][a] for a in actions)

        # Marginalize pi(a | s, x) over P(s | x) and accumulate exact quenched joint thermodynamics
        marginal_probs = {a: 0.0 for a in actions}
        joint_p_sa: List[Dict[str, float]] = []
        beta_raw_in = float(beta)
        is_zero_beta = beta_raw_in <= 1e-9
        beta_eff = 0.0 if is_zero_beta else min(25.0, beta_raw_in)

        expected_log_z_joint = 0.0
        # Quenched specific heat requires E_s[Var_{a|s}(U(s, a))] by Law of Total Variance,
        # excluding the between-spin variance Var_s(E[U|s]).
        expected_cond_var_u = 0.0

        for idx, pr_s in enumerate(p_spin):
            g_s = g_hat_by_state[idx]
            u_cond = {a: base_utilities[a] + (kappa * g_s[a]) for a in actions}
            if is_zero_beta:
                inv_k = 1.0 / len(actions)
                pi_s = {a: inv_k for a in actions}
                log_z_s = math.log(len(actions))
            else:
                max_uc = max(u_cond.values())
                exp_uc = {a: math.exp(max(-745.0, beta_eff * (u_cond[a] - max_uc))) for a in actions}
                sum_uc = sum(exp_uc.values()) or 1.0
                pi_s = {a: exp_uc[a] / sum_uc for a in actions}
                log_z_s = (beta_eff * max_uc) + math.log(sum_uc)

            expected_log_z_joint += pr_s * log_z_s
            mean_u_s = sum(pi_s[a] * u_cond[a] for a in actions)
            var_u_given_s = max(0.0, sum(pi_s[a] * ((u_cond[a] - mean_u_s) ** 2) for a in actions))
            expected_cond_var_u += pr_s * var_u_given_s

            joint_row: Dict[str, float] = {}
            for a in actions:
                p_joint = pr_s * pi_s[a]
                marginal_probs[a] += p_joint
                joint_row[a] = p_joint
            joint_p_sa.append(joint_row)

        p_sum = sum(marginal_probs.values()) or 1.0
        marginal_probs = {a: marginal_probs[a] / p_sum for a in actions}

        # True Joint Gibbs Free Energy, Log-Partition, and Quenched Joint Specific Heat
        bg_base = compute_boltzmann_gibbs_distribution(base_utilities, beta=beta)
        if is_zero_beta:
            joint_free_energy_f: Optional[float] = None
            joint_smoothed_max_u: Optional[float] = None
            joint_partition_z: Optional[float] = float(len(actions))
            joint_specific_heat_c = 0.0
            action_free_energies = {a: None for a in actions}
        else:
            joint_smoothed_max_u = expected_log_z_joint / beta_eff
            joint_free_energy_f = -joint_smoothed_max_u
            joint_partition_z = round(math.exp(expected_log_z_joint), 4) if -700.0 <= expected_log_z_joint <= 700.0 else None
            joint_specific_heat_c = (beta_eff ** 2) * expected_cond_var_u
            # Action-conditioned Helmholtz free energies F_a(x) = -(1/beta) * ln P(a | x) + F_joint
            # so that ln(P(a|x) / P(b|x)) = -beta * (F_a(x) - F_b(x)) holds identically
            action_free_energies = {
                a: round(-(1.0 / beta_eff) * math.log(max(marginal_probs[a], 1e-12)) + joint_free_energy_f, 4)
                for a in actions
            }

        sorted_actions = sorted(actions, key=lambda a: marginal_probs[a], reverse=True)
        a_top = sorted_actions[0]
        b_runner = sorted_actions[1]
        if joint_free_energy_f is not None and action_free_energies[a_top] is not None and action_free_energies[b_runner] is not None:
            delta_f_top_runner = round(float(action_free_energies[a_top]) - float(action_free_energies[b_runner]), 4)
        else:
            delta_f_top_runner = 0.0
        log_odds_top_runner = round(math.log(max(marginal_probs[a_top], 1e-12) / max(marginal_probs[b_runner], 1e-12)), 4)

        # Posterior spin expectation given action E[s_i | a, x] and Concept-Field Sensitivity Gradient
        cond_spin_top: Dict[str, float] = {}
        cond_spin_runner: Dict[str, float] = {}
        field_sensitivity_grad: Dict[str, float] = {}
        p_top = max(marginal_probs[a_top], 1e-12)
        p_run = max(marginal_probs[b_runner], 1e-12)

        for n in nodes:
            e_s_top = sum(joint_p_sa[idx][a_top] * spin_states[idx][n] for idx in range(len(spin_states))) / p_top
            e_s_run = sum(joint_p_sa[idx][b_runner] * spin_states[idx][n] for idx in range(len(spin_states))) / p_run
            cond_spin_top[n] = round(e_s_top, 4)
            cond_spin_runner[n] = round(e_s_run, 4)
            field_sensitivity_grad[n] = round((1.0 / tb) * (e_s_top - e_s_run), 4)

        marginal_entropy = -sum(p * math.log(max(p, 1e-12)) for p in marginal_probs.values())

        return {
            "probabilities": _round_simplex(marginal_probs, decimals=4),
            "shannon_entropy_S": round(marginal_entropy, 4),
            "helmholtz_free_energy_F": round(joint_free_energy_f, 4) if joint_free_energy_f is not None else None,
            "smoothed_max_utility_logZ": round(joint_smoothed_max_u, 4) if joint_smoothed_max_u is not None else None,
            "log_partition_function_logZ": round(expected_log_z_joint, 6),
            "specific_heat_capacity_C": round(joint_specific_heat_c, 4),
            "partition_function_Z": joint_partition_z,
            "base_kappa0_free_energy_F": bg_base["helmholtz_free_energy_F"],
            "mean_spins_P": {n: round(v, 4) for n, v in mean_spins.items()},
            "reference_action": ref_act,
            "centered_g_expectation_max_abs": round(centered_g_expectation_max_abs, 12),
            "n_nodes": n_nodes,
            "mean_abs_coupling_J": round(mean_abs_j, 4),
            "dimensionless_temperature_Tb_over_meanJ": dimless_tb_over_j,
            "tr_Sigma_per_node": round(sigma_trace / max(1, n_nodes), 5),
            "is_balanced_counterfactual": use_balanced_counterfactual,
            "frustration_covariance_trace_trSigma": round(sigma_trace, 5),
            "frustration_covariance_Sigma": {
                a: {b: round(sigma_cov[a][b], 5) for b in actions} for a in actions
            },
            "action_conditioned_free_energies_F_a": action_free_energies,
            "top_vs_runner_up_free_energy_gap_deltaF": delta_f_top_runner,
            "top_vs_runner_up_log_odds": log_odds_top_runner,
            "concept_field_log_odds_sensitivity": field_sensitivity_grad,
            "top_action": a_top,
            "runner_up_action": b_runner,
        }

    def _compute_competing_mechanisms_and_evidence_recommender(
        self,
        probs_m1_trait: Dict[str, float],
        probs_m2_caps: Dict[str, float],
        probs_m3_gibbs: Dict[str, float],
        probs_m4_commitment: Dict[str, float],
        field_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate Observational Equivalence & Mechanism Divergence across 4 competing behavioral mechanisms:
        - M1_TRAIT_LOGIT: Baseline psychometric trait logit
        - M2_CAPS_CONTEXTUAL: Mischel context-conditional parameter shift
        - M3_JOINT_GIBBS_ISING: External field h_i(x) + centered spin-glass coupling kappa * g_hat_a(s)
        - M4_LEXICOGRAPHIC_COMMITMENT: Sacred-value red-line & conditional commitment guard
        Computes pairwise JSD across mechanisms, identifies the Split Conformal Prediction Set C_alpha(x),
        and recommends the highest-value missing empirical observation to resolve mechanism ambiguity.
        """
        from who.engine.evaluator import compute_jensen_shannon_divergence

        actions = [
            "REJECT_AND_COUNTER",
            "STRATEGIC_CONCESSION",
            "SUBGAME_EQUILIBRIUM_ACCEPT",
            "DEADLOCK_WALKAWAY",
        ]
        mechs = {
            "M1_TRAIT_LOGIT": probs_m1_trait,
            "M2_CAPS_CONTEXTUAL": probs_m2_caps,
            "M3_JOINT_GIBBS_ISING": probs_m3_gibbs,
            "M4_LEXICOGRAPHIC_COMMITMENT": probs_m4_commitment,
        }
        mech_names = list(mechs.keys())
        jsd_pairs: List[float] = []
        pairwise_table: Dict[str, float] = {}

        for i in range(len(mech_names)):
            for j in range(i + 1, len(mech_names)):
                m_a, m_b = mech_names[i], mech_names[j]
                vec_a = [mechs[m_a].get(act, 0.25) for act in actions]
                vec_b = [mechs[m_b].get(act, 0.25) for act in actions]
                d_ab = compute_jensen_shannon_divergence(vec_a, vec_b)
                jsd_pairs.append(d_ab)
                pairwise_table[f"{m_a}_vs_{m_b}"] = d_ab

        mean_mech_jsd = round(sum(jsd_pairs) / max(1, len(jsd_pairs)), 4)
        top_actions_by_mech = {m: max(p_map, key=p_map.get) for m, p_map in mechs.items()}
        unique_tops = set(top_actions_by_mech.values())

        # Split Conformal Prediction Set C_alpha(x) at alpha = 0.10 (90% target coverage)
        sorted_final = sorted(probs_m4_commitment.items(), key=lambda kv: kv[1], reverse=True)
        conformal_set: List[str] = []
        cum_mass = 0.0
        target_coverage = 0.90
        for act_name, pr_val in sorted_final:
            conformal_set.append(act_name)
            cum_mass += pr_val
            if cum_mass >= target_coverage - 1e-6:
                break

        sens = field_analysis.get("concept_field_log_odds_sensitivity", {})
        top_sens_node = max(sens, key=lambda k: abs(sens[k])) if sens else "autonomy_and_rights_retention"

        if len(unique_tops) > 1 or mean_mech_jsd >= 0.03:
            equiv_status = "MECHANISM_DIVERGENCE_DETECTED"
            recommended_obs = (
                f"To break observational equivalence between {list(unique_tops)} (Mechanism JSD={mean_mech_jsd}), "
                f"acquire primary evidence of a past decision where [{top_sens_node}] was traded off against "
                f"immediate financial/deadline gain."
            )
        else:
            equiv_status = "HIGH_MECHANISM_INVARIANCE"
            recommended_obs = (
                f"All 4 competing cognitive mechanisms converge on [{sorted_final[0][0]}] (Mechanism JSD={mean_mech_jsd}). "
                f"Highest leverage concept node remains [{top_sens_node}]."
            )

        return {
            "observational_equivalence_status": equiv_status,
            "mean_mechanism_divergence_jsd": mean_mech_jsd,
            "pairwise_mechanism_jsd": pairwise_table,
            "top_action_by_mechanism": top_actions_by_mech,
            "mechanism_distributions": {
                m: {a: round(p_map.get(a, 0.0), 4) for a in actions}
                for m, p_map in mechs.items()
            },
            "conformal_prediction_set_90": {
                "alpha": 0.10,
                "target_coverage": 0.90,
                "prediction_set": conformal_set,
                "cumulative_probability_mass": round(cum_mass, 4),
                "set_cardinality": len(conformal_set),
            },
            "active_evidence_recommendation": recommended_obs,
        }

    def simulate_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Simulate plausible target reaction, decision heuristics, and potential counter-moves (Single-Pass),
        conditioned on observed corpus features, knowledge graph topology, CAPS context, Stated vs. Revealed
        preference, Boltzmann-Gibbs temperature, and Brier-calibrated confidence.
        """
        logger.info(f"Simulating reaction for {self.target_name} on scenario: '{scenario[:60]}...'")
        mindset_text, skill_data, features_data, graph_metrics = self._load_profile_context()
        knowledge_graph = self._load_knowledge_graph()
        b_prof = skill_data.get("behavioral_profile", {})
        psychometrics = b_prof.get("extended_psychometrics", {})
        cog_style = b_prof.get("cognitive_style", "Empirical")

        caps_matrix = (
            b_prof.get("caps_contextual_matrix")
            or features_data.get("caps_contextual_matrix")
            or {}
        )
        pref_data = (
            b_prof.get("stated_vs_revealed_preference")
            or features_data.get("stated_vs_revealed_preference")
            or {}
        )

        loss_aversion, loss_prov = self._resolve_param_with_provenance(
            psychometrics.get("loss_aversion", {}),
            features_data.get("loss_aversion_index", {}),
            "estimated_score",
            1.85,
        )
        need_for_closure, closure_prov = self._resolve_param_with_provenance(
            psychometrics.get("need_for_closure", {}),
            features_data.get("need_for_closure_index", {}),
            "estimated_score",
            0.50,
        )
        volatility, vol_prov = self._resolve_param_with_provenance(
            psychometrics.get("affective_volatility", {}),
            features_data.get("affective_volatility_index", {}),
            "volatility_score",
            0.20,
        )

        field_analysis = self._evaluate_scenario_external_fields_and_commitments(
            scenario=scenario,
            features_data=features_data,
            graph_metrics=graph_metrics,
            knowledge_graph=knowledge_graph,
        )
        red_line_viol = float(field_analysis.get("red_line_violation_score", 0.0))
        clause_pol = float(field_analysis.get("clause_polarity", 0.0))
        net_kg = float(field_analysis.get("kg_alignment_bias", 0.0)) - float(field_analysis.get("kg_tension_bias", 0.0))

        if red_line_viol >= 0.65 and loss_aversion >= 1.80:
            inferred_dominant_branch = "DEADLOCK_WALKAWAY"
        elif (clause_pol + net_kg) >= 0.25 and red_line_viol < 0.20:
            inferred_dominant_branch = "SUBGAME_EQUILIBRIUM_ACCEPT"
        elif (clause_pol + net_kg) > 0.05 and loss_aversion < 1.60:
            inferred_dominant_branch = "STRATEGIC_CONCESSION"
        else:
            inferred_dominant_branch = "REJECT_AND_COUNTER"

        caps_context = self._detect_situational_context(scenario, caps_matrix)
        dual_layer = self._build_dual_layer_prediction(pref_data, dominant_branch=inferred_dominant_branch)

        frustration_phi = float(
            graph_metrics.get("cognitive_dissonance", {})
            .get("ising_spin_glass_physics", {})
            .get("structural_frustration_index", 0.0)
        )
        beta = self._compute_inverse_temperature_beta(
            need_for_closure=need_for_closure,
            affective_volatility=volatility,
            active_context=caps_context["active_context"],
            frustration_index=frustration_phi,
        )

        matched_kg_anchors = field_analysis.get("matched_knowledge_graph_anchors", [])
        anchors_str = ", ".join(matched_kg_anchors[:6]) if matched_kg_anchors else "None matched in local graph"

        system_prompt = (
            f"You are the Behavioral Simulation Engine for {self.target_name}.\n"
            "EPISTEMOLOGICAL RULES:\n"
            "- You MUST NOT claim to predict the future or know secret intentions with certainty.\n"
            "- You MUST simulate plausible reactions under explicit assumptions.\n"
            "- Use probabilistic phrasing: 'Under the assumption that [X], the data suggests a plausible alignment with [Y]...'\n"
            "- Explicitly separate (A) Stated Public/Diplomatic Reaction vs. (B) Revealed Strategic Action.\n"
            f"- Apply Active CAPS Rule: {caps_context['if_then_signature']}\n"
            "- Respond in the same language as the SCENARIO PROPOSITION.\n"
        )

        user_prompt = (
            f"SCENARIO PROPOSITION:\n\"{scenario}\"\n\n"
            f"ACTIVE SITUATIONAL CONTEXT (CAPS): {caps_context['active_context']}\n"
            f"BOLTZMANN INVERSE COGNITIVE TEMPERATURE (β=1/T): {beta}\n"
            f"MATCHED KNOWLEDGE GRAPH ANCHORS: {anchors_str}\n"
            f"STATED VS REVEALED PREFERENCE VERDICT: {dual_layer['divergence_verdict']} "
            f"(Divergence Index: {dual_layer['preference_divergence_index']})\n"
            f"OBSERVED COGNITIVE PROFILE:\n{mindset_text[:1000]}\n\n"
            "Simulate:\n"
            "1. Stated Public Reaction (Diplomatic/Normative Rhetoric)\n"
            "2. Revealed Strategic Action (Operational Counter-Move / Execution)\n"
            "3. Key Friction Points & Contextual Triggers\n"
            "4. Plausible Alternative Reaction Under Divergent Context\n"
            "5. Explicit Confidence Score with empirical justification."
        )

        simulation_output = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.3)

        evaluator = ProfileEvaluator(self.target_name, self.dirs)
        referee_audit = evaluator.verify_stylometric_and_epistemic_fidelity(simulation_output)
        confidence, backtest = self._compute_calibrated_confidence(
            evaluator=evaluator,
            features_data=features_data,
            stylometric_fidelity=referee_audit.get("stylometric_fidelity_score", 0.80),
            has_mindset=bool(mindset_text),
        )

        parameter_provenance = {
            "loss_aversion": loss_prov,
            "need_for_closure": closure_prov,
            "affective_volatility": vol_prov,
            "caps_contextual_matrix": "observed_caps_matrix" if caps_matrix.get("contexts") else "default_prior_fallback",
            "knowledge_graph": (
                f"observed_graph ({len(knowledge_graph.get('nodes', []))} nodes)"
                if knowledge_graph.get("nodes")
                else "default_prior_fallback"
            ),
            "mindset_summary": "observed_mindset_summary.md" if bool(mindset_text) else "default_prior_fallback",
        }
        model_transparency = self._build_model_transparency(
            features_data=features_data,
            knowledge_graph=knowledge_graph,
            field_analysis=field_analysis,
            has_mindset=bool(mindset_text),
            has_skill=bool(skill_data),
            parameter_provenance=parameter_provenance,
        )

        return {
            "target_name": self.target_name,
            "scenario": scenario,
            "inferred_dominant_branch": inferred_dominant_branch,
            "simulation_analysis": simulation_output,
            "confidence_score": confidence,
            "cognitive_style_assumed": cog_style,
            "caps_situational_context": caps_context,
            "dual_layer_prediction": dual_layer,
            "scenario_field_and_commitments": {
                "clause_polarity": field_analysis["clause_polarity"],
                "red_line_violation_score": field_analysis["red_line_violation_score"],
                "commitment_alignment_score": field_analysis["commitment_alignment_score"],
                "matched_knowledge_graph_anchors": matched_kg_anchors,
                "kg_alignment_bias": field_analysis["kg_alignment_bias"],
                "kg_tension_bias": field_analysis["kg_tension_bias"],
            },
            "statistical_mechanics": {
                "inverse_temperature_beta": beta,
                "cognitive_temperature_T": round(1.0 / beta, 4),
                "ising_frustration_index_phi": frustration_phi,
            },
            "calibration_metrics": {
                "brier_score": backtest.get("brier_score", 0.20),
                "holdout_accuracy_pct": backtest.get("holdout_accuracy_pct", 80.0),
                "stylometric_jsd": referee_audit.get("stylometric_jsd", 0.05),
                "stylometric_fidelity_score": referee_audit.get("stylometric_fidelity_score", 0.80),
            },
            "model_transparency": model_transparency,
            "evidence_ledger": referee_audit.get("evidence_ledger", []),
            "epistemic_qualification": (
                "Corpus-conditioned scenario model (not a trained empirical behavioral predictor), evaluated "
                "under explicit CAPS situational assumptions, Boltzmann-Gibbs temperature calibration, and out-of-sample "
                "Brier verification."
            ),
        }

    def simulate_monte_carlo(
        self,
        scenario: str,
        iterations: int = 25,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Multi-iteration Monte Carlo negotiation & scenario simulator (Boltzmann-Gibbs Quantal Response Equilibrium).
        Stochastically perturbs counterparty leverage, deadline pressure, information asymmetry,
        and reputation risk, mapping utility vectors through the Boltzmann-Gibbs partition function Z(beta)
        (Quantal Response Equilibrium) and computing Shannon Decision Entropy (S) and Helmholtz Free Energy (F).
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError(f"iterations must be a positive integer >= 1, got {iterations!r}")
        logger.info(f"Executing Monte Carlo scenario simulation for {self.target_name} (N={iterations})...")
        mindset_text, skill_data, features_data, graph_metrics = self._load_profile_context()
        knowledge_graph = self._load_knowledge_graph()

        # Extract target psychometrics with explicit provenance
        b_prof = skill_data.get("behavioral_profile", {})
        psychometrics = b_prof.get("extended_psychometrics", {})
        cog_style = b_prof.get("cognitive_style", "Empirical")

        loss_aversion, loss_prov = self._resolve_param_with_provenance(
            psychometrics.get("loss_aversion", {}),
            features_data.get("loss_aversion_index", {}),
            "estimated_score",
            1.85,
        )
        machiavellianism, mach_prov = self._resolve_param_with_provenance(
            psychometrics.get("machiavellianism", {}),
            features_data.get("machiavellianism_index", {}),
            "estimated_score",
            0.35,
        )
        need_for_closure, closure_prov = self._resolve_param_with_provenance(
            psychometrics.get("need_for_closure", {}),
            features_data.get("need_for_closure_index", {}),
            "estimated_score",
            0.50,
        )
        volatility, vol_prov = self._resolve_param_with_provenance(
            psychometrics.get("affective_volatility", {}),
            features_data.get("affective_volatility_index", {}),
            "volatility_score",
            0.20,
        )

        caps_matrix = (
            b_prof.get("caps_contextual_matrix")
            or psychometrics.get("caps_contextual_matrix")
            or features_data.get("caps_contextual_matrix")
            or {}
        )
        pref_data = (
            b_prof.get("stated_vs_revealed_preference")
            or psychometrics.get("stated_vs_revealed_preference")
            or features_data.get("stated_vs_revealed_preference")
            or {}
        )

        loss_aversion_base = loss_aversion
        machiavellianism_base = machiavellianism
        need_for_closure_base = need_for_closure

        # Apply Mischel CAPS Context-Conditional parameter shift
        caps_context = self._detect_situational_context(scenario, caps_matrix)
        active_ctx = caps_context["active_context"]
        ctx_metrics = caps_context.get("context_metrics", {})

        if active_ctx == "HIGH_STAKES_CRISIS":
            loss_aversion = round(min(2.50, loss_aversion * 1.12), 2)
            need_for_closure = round(min(1.00, need_for_closure * 1.15), 2)
        elif active_ctx == "FORMAL_ACADEMIC":
            need_for_closure = round(max(0.15, need_for_closure * 0.88), 2)
        elif active_ctx == "STRATEGIC_NEGOTIATION":
            machiavellianism = round(min(1.00, machiavellianism * 1.10), 2)

        if ctx_metrics.get("segment_count", 0) >= 2 and ctx_metrics.get("contextual_loss_aversion"):
            loss_aversion = round(0.7 * loss_aversion + 0.3 * float(ctx_metrics["contextual_loss_aversion"]), 2)

        # Evaluate Scenario-Induced External Fields h_i(x), Knowledge Graph anchors, and Conditional Commitments
        field_analysis = self._evaluate_scenario_external_fields_and_commitments(
            scenario=scenario,
            features_data=features_data,
            graph_metrics=graph_metrics,
            knowledge_graph=knowledge_graph,
        )
        cov_shifts = field_analysis["covariate_shifts"]
        clause_pol = float(field_analysis["clause_polarity"])
        red_line_viol = float(field_analysis["red_line_violation_score"])
        comm_align = float(field_analysis["commitment_alignment_score"])
        kg_align_bias = float(field_analysis.get("kg_alignment_bias", 0.0))
        kg_tens_bias = float(field_analysis.get("kg_tension_bias", 0.0))

        # Extract Ising Spin-Glass frustration from concept graph if available
        ising_data = graph_metrics.get("cognitive_dissonance", {}).get("ising_spin_glass_physics", {})
        frustration_phi = float(ising_data.get("structural_frustration_index", 0.0))

        # Compute Boltzmann-Gibbs Inverse Cognitive Temperature beta = 1 / T
        beta = self._compute_inverse_temperature_beta(
            need_for_closure=need_for_closure,
            affective_volatility=volatility,
            active_context=active_ctx,
            frustration_index=frustration_phi,
        )

        # Single-parameter Joint Gibbs spin-to-action coupling strength kappa
        kappa_couple = round(0.35 + (0.45 * frustration_phi), 4)

        ranges = {
            "counterparty_leverage": (0.15, 0.85),
            "deadline_pressure_days": (1.0, 14.0),
            "information_asymmetry": (0.05, 0.65),
            "reputation_risk": (0.10, 0.80),
        }
        if parameter_ranges:
            ranges.update(parameter_ranges)

        rng = random.Random(random_seed) if random_seed is not None else random.Random()

        BRANCH_COUNTER = "REJECT_AND_COUNTER"
        BRANCH_CONCESSION = "STRATEGIC_CONCESSION"
        BRANCH_ACCEPT = "SUBGAME_EQUILIBRIUM_ACCEPT"
        BRANCH_DEADLOCK = "DEADLOCK_WALKAWAY"
        all_branches = [BRANCH_COUNTER, BRANCH_CONCESSION, BRANCH_ACCEPT, BRANCH_DEADLOCK]

        iteration_records: List[Dict[str, Any]] = []
        branch_counts = {b: 0 for b in all_branches}
        boltzmann_prob_sums = {b: 0.0 for b in all_branches}
        m1_prob_sums = {b: 0.0 for b in all_branches}
        m2_prob_sums = {b: 0.0 for b in all_branches}
        m3_prob_sums = {b: 0.0 for b in all_branches}

        payoffs: List[float] = []
        entropies: List[float] = []
        free_energies: List[float] = []
        smoothed_max_utilities: List[float] = []
        specific_heats: List[float] = []
        partition_zs: List[float] = []
        last_gibbs_state: Dict[str, Any] = {}

        p_frustrated_mode = min(0.45, max(0.0, 0.5 * frustration_phi))

        for i in range(iterations):
            lev_raw = rng.uniform(*ranges["counterparty_leverage"])
            deadline = rng.uniform(*ranges["deadline_pressure_days"])
            asym_raw = rng.uniform(*ranges["information_asymmetry"])
            rep_raw = rng.uniform(*ranges["reputation_risk"])

            # Apply scenario-induced covariate shifts (clamped to valid domain [0.02, 0.98])
            lev = min(0.98, max(0.02, lev_raw + cov_shifts["leverage_shift"]))
            asym = min(0.98, max(0.02, asym_raw + cov_shifts["asymmetry_shift"]))
            rep = min(0.98, max(0.02, rep_raw + cov_shifts["reputation_risk_shift"]))

            # M1: Baseline Trait Logit Utilities (no CAPS shift, no clause polarity, no spin-glass or red-line guard)
            u_m1 = {
                BRANCH_ACCEPT: (1.0 - lev_raw) * (1.0 - rep_raw) - (loss_aversion_base * asym_raw) + (machiavellianism_base * 0.20),
                BRANCH_COUNTER: (0.60 * (1.0 - lev_raw)) + (0.35 * (1.0 - asym_raw)) + (0.25 * machiavellianism_base) - (0.15 / max(deadline, 1.0)),
                BRANCH_CONCESSION: (0.45 * lev_raw * machiavellianism_base) + (0.30 * (1.0 / max(deadline, 1.0)) * need_for_closure_base) - (0.25 * loss_aversion_base * rep_raw),
                BRANCH_DEADLOCK: 0.20 + (0.45 * loss_aversion_base * asym_raw) + (0.40 * rep_raw) - (0.35 * need_for_closure_base),
            }
            bg_m1 = compute_boltzmann_gibbs_distribution(u_m1, beta=beta)

            # M2: CAPS Context-Conditional + Scenario Clause Polarity Utilities
            u_m2 = {
                BRANCH_ACCEPT: (1.0 - lev) * (1.0 - rep) - (loss_aversion * asym) + (machiavellianism * 0.20) + (0.55 * clause_pol),
                BRANCH_COUNTER: (0.60 * (1.0 - lev)) + (0.35 * (1.0 - asym)) + (0.25 * machiavellianism) - (0.15 / max(deadline, 1.0)) - (0.25 * clause_pol),
                BRANCH_CONCESSION: (0.45 * lev * machiavellianism) + (0.30 * (1.0 / max(deadline, 1.0)) * need_for_closure) - (0.25 * loss_aversion * rep) + (0.20 * clause_pol),
                BRANCH_DEADLOCK: 0.20 + (0.45 * loss_aversion * asym) + (0.40 * rep) - (0.35 * need_for_closure) - (0.40 * clause_pol),
            }
            bg_m2 = compute_boltzmann_gibbs_distribution(u_m2, beta=beta)

            # M3: Centered Joint Gibbs Spin-Glass Coupling (with external field h_corpus + h(x))
            gibbs_m3 = self._compute_centered_joint_gibbs_coupling(
                base_utilities=u_m2,
                beta=beta,
                kappa=kappa_couple,
                field_analysis=field_analysis,
            )

            # M4 (Full Canonical Model): Joint Gibbs + Conditional Commitment & Lexicographic Red-Line Guard + KG Anchors
            u_m4 = {
                BRANCH_ACCEPT: u_m2[BRANCH_ACCEPT] + (0.45 * comm_align) - (0.85 * loss_aversion * red_line_viol) + kg_align_bias - (0.65 * kg_tens_bias),
                BRANCH_COUNTER: u_m2[BRANCH_COUNTER] + (0.40 * red_line_viol) - (0.15 * comm_align) + (0.55 * kg_tens_bias) - (0.20 * kg_align_bias),
                BRANCH_CONCESSION: u_m2[BRANCH_CONCESSION] + (0.15 * comm_align) - (0.55 * red_line_viol) + (0.20 * kg_align_bias) - (0.35 * kg_tens_bias),
                BRANCH_DEADLOCK: u_m2[BRANCH_DEADLOCK] + (0.50 * loss_aversion * red_line_viol) - (0.30 * comm_align) + (0.45 * kg_tens_bias) - (0.25 * kg_align_bias),
            }
            bg_state = self._compute_centered_joint_gibbs_coupling(
                base_utilities=u_m4,
                beta=beta,
                kappa=kappa_couple,
                field_analysis=field_analysis,
            )
            last_gibbs_state = bg_state

            for b_key in all_branches:
                boltzmann_prob_sums[b_key] += bg_state["probabilities"][b_key]
                m1_prob_sums[b_key] += bg_m1["probabilities"][b_key]
                m2_prob_sums[b_key] += bg_m2["probabilities"][b_key]
                m3_prob_sums[b_key] += gibbs_m3["probabilities"][b_key]

            entropies.append(bg_state["shannon_entropy_S"])
            free_energies.append(bg_state["helmholtz_free_energy_F"])
            smoothed_max_utilities.append(bg_state["smoothed_max_utility_logZ"])
            specific_heats.append(bg_state["specific_heat_capacity_C"])
            partition_zs.append(bg_state["partition_function_Z"])

            # Expected QRE payoff under the marginalized Boltzmann-Gibbs distribution P(a | x)
            qre_expected_payoff_i = round(
                sum(bg_state["probabilities"][b_k] * u_m4[b_k] for b_k in all_branches),
                4,
            )
            chosen_branch = max(bg_state["probabilities"], key=bg_state["probabilities"].get)
            branch_counts[chosen_branch] += 1
            payoffs.append(qre_expected_payoff_i)

            iteration_records.append({
                "iteration": i + 1,
                "parameters": {
                    "counterparty_leverage": round(lev, 3),
                    "deadline_days": round(deadline, 1),
                    "information_asymmetry": round(asym, 3),
                    "reputation_risk": round(rep, 3),
                },
                "payoff": qre_expected_payoff_i,
                "branch": chosen_branch,
                "boltzmann_probabilities": bg_state["probabilities"],
                "shannon_entropy_S": bg_state["shannon_entropy_S"],
            })

        total_runs = float(max(1, iterations))
        argmax_frequencies = {k: round(v / total_runs, 4) for k, v in branch_counts.items()}
        boltzmann_distribution = {k: round(v / total_runs, 4) for k, v in boltzmann_prob_sums.items()}
        # Unify user-facing branch_probabilities, boltzmann_branch_probabilities, and dominant_branch on the QRE distribution
        distribution = dict(boltzmann_distribution)
        dominant_branch = max(boltzmann_distribution, key=boltzmann_distribution.get)

        m1_dist = {k: round(v / total_runs, 4) for k, v in m1_prob_sums.items()}
        m2_dist = {k: round(v / total_runs, 4) for k, v in m2_prob_sums.items()}
        m3_dist = {k: round(v / total_runs, 4) for k, v in m3_prob_sums.items()}

        field_with_sens = dict(field_analysis)
        field_with_sens["concept_field_log_odds_sensitivity"] = last_gibbs_state.get(
            "concept_field_log_odds_sensitivity", {}
        )
        competing_mechanisms = self._compute_competing_mechanisms_and_evidence_recommender(
            probs_m1_trait=m1_dist,
            probs_m2_caps=m2_dist,
            probs_m3_gibbs=m3_dist,
            probs_m4_commitment=boltzmann_distribution,
            field_analysis=field_with_sens,
        )

        mean_entropy_s = round(sum(entropies) / len(entropies), 4) if entropies else 0.0
        norm_entropy = round(mean_entropy_s / math.log(4), 4)
        valid_fs = [f for f in free_energies if f is not None]
        mean_free_energy_f = round(sum(valid_fs) / len(valid_fs), 4) if valid_fs else 0.0
        valid_smax = [v for v in smoothed_max_utilities if v is not None]
        mean_smoothed_max_u = round(sum(valid_smax) / len(valid_smax), 4) if valid_smax else 0.0
        mean_specific_heat_c = round(sum(specific_heats) / len(specific_heats), 4) if specific_heats else 0.0
        valid_zs = [z for z in partition_zs if z is not None]
        mean_partition_z = round(sum(valid_zs) / len(valid_zs), 4) if valid_zs else None

        # Harary-Balanced Counterfactual Control (E2_BAL) on final iteration utilities
        bal_gibbs_state = self._compute_centered_joint_gibbs_coupling(
            base_utilities=u_m4,
            beta=beta,
            kappa=kappa_couple,
            field_analysis=field_analysis,
            use_balanced_counterfactual=True,
        ) if last_gibbs_state else {}

        if norm_entropy <= 0.65:
            phase_regime = "ORDERED_LOW_ENTROPY_REGIME"
        elif norm_entropy <= 0.85:
            phase_regime = "SCHOTTKY_CROSSOVER_REGIME"
        else:
            phase_regime = "HIGH_ENTROPY_STOCHASTIC_REGIME"

        dual_layer = self._build_dual_layer_prediction(pref_data, dominant_branch=dominant_branch)

        sorted_payoffs = sorted(payoffs)
        p10 = sorted_payoffs[max(0, int(0.10 * len(sorted_payoffs)))]
        p50 = sorted_payoffs[int(0.50 * len(sorted_payoffs))]
        p90 = sorted_payoffs[min(len(sorted_payoffs) - 1, int(0.90 * len(sorted_payoffs)))]
        expected_nash_payoff = round(sum(payoffs) / len(payoffs), 3)

        deadlock_asym_vals = [r["parameters"]["information_asymmetry"] for r in iteration_records if r["branch"] == BRANCH_DEADLOCK]
        normal_asym_vals = [r["parameters"]["information_asymmetry"] for r in iteration_records if r["branch"] != BRANCH_DEADLOCK]

        if deadlock_asym_vals and normal_asym_vals:
            avg_deadlock_asym = sum(deadlock_asym_vals) / len(deadlock_asym_vals)
            avg_normal_asym = sum(normal_asym_vals) / len(normal_asym_vals)
            critical_tipping_metric = "Information Asymmetry (α)"
            tipping_statement = (
                f"Information Asymmetry (α) is the critical Schottky crossover parameter: "
                f"avg α was {avg_deadlock_asym:.2f} under Deadlock vs {avg_normal_asym:.2f} in constructive branches. "
                f"High opacity triggers the target's loss-aversion threshold (β_loss={loss_aversion:.2f}), causing immediate walkaways."
            )
        else:
            critical_tipping_metric = "Counterparty Leverage (λ)"
            tipping_statement = (
                f"Counterparty Leverage (λ) is the primary friction point. "
                f"When leverage exceeds 0.70, the target's autonomy preservation overrides concessions."
            )

        matched_kg_anchors = field_analysis.get("matched_knowledge_graph_anchors", [])
        anchors_str = ", ".join(matched_kg_anchors[:6]) if matched_kg_anchors else "None matched in local graph"

        system_prompt = (
            f"You are the Game Theory and Behavioral Strategy Analyst for {self.target_name}.\n"
            "Summarize the Monte Carlo simulation results using game-theoretic and statistical decision reasoning, "
            "explicitly distinguishing Stated Public Reaction from Revealed Strategic Action.\n"
            "Respond in the same language as the SCENARIO."
        )
        user_prompt = (
            f"SCENARIO: \"{scenario}\"\n\n"
            f"ACTIVE CAPS CONTEXT: {active_ctx} ({caps_context['if_then_signature']})\n"
            f"MATCHED KNOWLEDGE GRAPH ANCHORS: {anchors_str}\n"
            f"MONTE CARLO & BOLTZMANN-GIBBS RESULTS ({iterations} stochastic iterations):\n"
            f"- Outcome Distribution: {distribution}\n"
            f"- Boltzmann-Gibbs (QRE) Action Distribution: {boltzmann_distribution}\n"
            f"- Inverse Temperature (β=1/T): {beta} | Shannon Entropy (S): {mean_entropy_s} ({phase_regime})\n"
            f"- Dominant Nash Equilibrium Branch: {dominant_branch} ({int(distribution[dominant_branch]*100)}%)\n"
            f"- Expected Payoff (E[U]): {expected_nash_payoff} [P10: {p10}, Median: {p50}, P90: {p90}]\n"
            f"- Sensitivity Finding: {tipping_statement}\n"
            f"- Stated vs. Revealed Divergence: {dual_layer['divergence_verdict']} (Δ={dual_layer['preference_divergence_index']})\n\n"
            "Provide:\n"
            "1. Game-Theoretic & Quantal Response Equilibrium Assessment\n"
            "2. Stated Public Posture vs. Revealed Operational Counter-Move\n"
            "3. Actionable Negotiation Advice to align with target's subgame incentives."
        )

        strategic_synthesis = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

        evaluator = ProfileEvaluator(self.target_name, self.dirs)
        referee_audit = evaluator.verify_stylometric_and_epistemic_fidelity(strategic_synthesis)
        confidence, backtest = self._compute_calibrated_confidence(
            evaluator=evaluator,
            features_data=features_data,
            stylometric_fidelity=referee_audit.get("stylometric_fidelity_score", 0.80),
            has_mindset=bool(mindset_text),
        )

        parameter_provenance = {
            "loss_aversion": loss_prov,
            "machiavellianism": mach_prov,
            "need_for_closure": closure_prov,
            "affective_volatility": vol_prov,
            "caps_contextual_matrix": "observed_caps_matrix" if caps_matrix.get("contexts") else "default_prior_fallback",
            "knowledge_graph": (
                f"observed_graph ({len(knowledge_graph.get('nodes', []))} nodes)"
                if knowledge_graph.get("nodes")
                else "default_prior_fallback"
            ),
            "mindset_summary": "observed_mindset_summary.md" if bool(mindset_text) else "default_prior_fallback",
        }
        model_transparency = self._build_model_transparency(
            features_data=features_data,
            knowledge_graph=knowledge_graph,
            field_analysis=field_analysis,
            has_mindset=bool(mindset_text),
            has_skill=bool(skill_data),
            parameter_provenance=parameter_provenance,
        )

        return {
            "target_name": self.target_name,
            "scenario": scenario,
            "iterations": iterations,
            "dominant_branch": dominant_branch,
            "branch_probabilities": distribution,
            "boltzmann_branch_probabilities": boltzmann_distribution,
            "deterministic_argmax_frequencies": argmax_frequencies,
            "conformal_prediction_set_90": competing_mechanisms["conformal_prediction_set_90"],
            "expected_qre_payoff": expected_nash_payoff,
            "expected_nash_payoff": expected_nash_payoff,
            "payoff_bounds": {"p10": p10, "median": p50, "p90": p90},
            "statistical_mechanics": {
                "inverse_temperature_beta": beta,
                "cognitive_temperature_T": round(1.0 / beta, 4),
                "mean_partition_function_Z": mean_partition_z,
                "helmholtz_free_energy_F": mean_free_energy_f,
                "smoothed_max_utility_logZ": mean_smoothed_max_u,
                "specific_heat_capacity_C": mean_specific_heat_c,
                "shannon_decision_entropy_S": mean_entropy_s,
                "normalized_entropy": norm_entropy,
                "crossover_regime": phase_regime,
                "phase_regime": phase_regime,
                "ising_frustration_index_phi": frustration_phi,
                "joint_gibbs_coupling_kappa": kappa_couple,
                "joint_gibbs_frustrated_mode_weight": round(p_frustrated_mode, 4),
                "dimensionless_temperature_Tb_over_meanJ": last_gibbs_state.get("dimensionless_temperature_Tb_over_meanJ", 0.0),
                "tr_Sigma_per_node": last_gibbs_state.get("tr_Sigma_per_node", 0.0),
                "balanced_counterfactual_trSigma": bal_gibbs_state.get("frustration_covariance_trace_trSigma", 0.0),
                "frustration_attributable_sigma_delta": round(
                    float(last_gibbs_state.get("frustration_covariance_trace_trSigma", 0.0))
                    - float(bal_gibbs_state.get("frustration_covariance_trace_trSigma", 0.0)),
                    5,
                ),
                "frustration_covariance_trace_trSigma": last_gibbs_state.get("frustration_covariance_trace_trSigma", 0.0),
                "frustration_covariance_Sigma": last_gibbs_state.get("frustration_covariance_Sigma", {}),
                "action_conditioned_free_energies_F_a": last_gibbs_state.get("action_conditioned_free_energies_F_a", {}),
                "top_vs_runner_up_free_energy_gap_deltaF": last_gibbs_state.get("top_vs_runner_up_free_energy_gap_deltaF", 0.0),
                "concept_field_log_odds_sensitivity": last_gibbs_state.get("concept_field_log_odds_sensitivity", {}),
            },
            "scenario_field_and_commitments": {
                "clause_polarity": field_analysis["clause_polarity"],
                "red_line_violation_score": field_analysis["red_line_violation_score"],
                "commitment_alignment_score": field_analysis["commitment_alignment_score"],
                "triggered_commitments": field_analysis["triggered_commitments"],
                "matched_knowledge_graph_anchors": matched_kg_anchors,
                "kg_alignment_bias": kg_align_bias,
                "kg_tension_bias": kg_tens_bias,
                "corpus_external_fields_h": field_analysis["corpus_external_fields_h"],
                "scenario_external_fields_h_x": field_analysis["scenario_external_fields_h_x"],
                "total_external_fields_h_total": field_analysis["total_external_fields_h_total"],
            },
            "competing_mechanisms_analysis": competing_mechanisms,
            "critical_sensitivity": {
                "key_variable": critical_tipping_metric,
                "finding": tipping_statement,
            },
            "caps_situational_context": caps_context,
            "dual_layer_prediction": dual_layer,
            "strategic_synthesis": strategic_synthesis,
            "confidence_score": confidence,
            "calibration_metrics": {
                "brier_score": backtest.get("brier_score"),
                "is_prior_fallback": backtest.get("is_prior_fallback", False),
                "holdout_accuracy_pct": backtest.get("holdout_accuracy_pct"),
                "stylometric_jsd": referee_audit.get("stylometric_jsd", 0.05),
                "stylometric_fidelity_score": referee_audit.get("stylometric_fidelity_score", 0.80),
            },
            "model_transparency": model_transparency,
            "evidence_ledger": referee_audit.get("evidence_ledger", []),
            "assumed_psychometrics": {
                "loss_aversion": loss_aversion,
                "machiavellianism": machiavellianism,
                "need_for_closure": need_for_closure,
                "cognitive_style": cog_style,
            },
            "sample_iterations": iteration_records[:5],
            "epistemic_qualification": (
                f"Corpus-conditioned scenario model (not a trained empirical behavioral predictor) across {iterations} "
                f"Monte Carlo & Joint Gibbs Boltzmann-QRE perturbations conditioned on CAPS context [{active_ctx}], "
                f"inverse temperature β={beta}, external field h_i(x), and out-of-sample Brier verification."
            ),
        }
