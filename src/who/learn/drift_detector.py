"""
Temporal Cognitive Drift Engine for who??.
Quantifies mathematical shifts in epistemic calibration, dogmatism density,
lexical diversity (TTR), and concept centrality over time.
Adheres strictly to probabilistic reasoning without deterministic claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from who.config import call_llm
from who.learn.feature_extractor import FeatureExtractor
from who.utils.logger import get_logger

logger = get_logger("who.learn.drift")


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

        graph_file = self.dirs.vault_dir / "knowledge_graph.json"
        graph_concepts: Set[str] = set()
        if graph_file.exists():
            try:
                with open(graph_file, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                    for node in graph_data.get("nodes", []):
                        label = node.get("label") or node.get("id") or ""
                        if label:
                            graph_concepts.add(label.lower().strip())
            except Exception:
                pass

        # Extract calibrated baseline floats
        epistemic = baseline_features.get("epistemic_markers", {})
        manip = baseline_features.get("epistemic_manipulation_index", {})
        lexical = baseline_features.get("lexical_diversity", {})
        pronouns = baseline_features.get("pronoun_distribution", {})

        top_terms = [t[0].lower() for t in baseline_features.get("top_characteristic_terms", []) if isinstance(t, (list, tuple))]

        # If graph is empty, fallback to top baseline terms
        if not graph_concepts and top_terms:
            graph_concepts = set(top_terms[:15])

        return {
            "calibration_ratio": float(epistemic.get("epistemic_calibration_ratio", 1.10)),
            "dogmatic_density": float(manip.get("dogmatic_density", 0.45)),
            "type_token_ratio": float(lexical.get("type_token_ratio", 0.65)),
            "pronoun_i_pct": float(pronouns.get("i_percent", 2.5)),
            "baseline_concepts": graph_concepts,
            "top_terms": top_terms,
        }

    def detect_drift(
        self,
        new_texts: Optional[List[str]] = None,
        new_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Quantify longitudinal cognitive drift between baseline corpus and incoming text footprint.
        Computes Delta-Calibration, Delta-Dogmatism, Delta-TTR, and Concept Centrality Shifts.
        """
        logger.info(f"Computing temporal cognitive drift for {self.target_name}...")
        baseline = self._load_baseline_metrics()

        # Gather observed texts
        observed_corpus: List[str] = []
        if new_texts:
            observed_corpus.extend(new_texts)
        elif new_documents:
            observed_corpus.extend([d.get("content", "") for d in new_documents if d.get("content")])
        else:
            # Load raw data from disk
            raw_files = list(self.dirs.raw_data_dir.glob("*.md")) + list(self.dirs.raw_data_dir.glob("*.txt"))
            if raw_files:
                # Use latest files or full corpus
                raw_files_sorted = sorted(raw_files, key=lambda p: p.stat().st_mtime)
                # Take recent slice
                recent_slice = raw_files_sorted[-max(1, len(raw_files_sorted) // 3):]
                for rf in recent_slice:
                    try:
                        observed_corpus.append(rf.read_text(encoding="utf-8"))
                    except Exception:
                        pass

        if not observed_corpus:
            observed_corpus = [f"Baseline discourse overview for {self.target_name}."]

        # Extract features from observed footprint
        obs_features = self.feature_extractor.extract_features(observed_corpus)
        obs_epistemic = obs_features.get("epistemic_markers", {})
        obs_manip = obs_features.get("epistemic_manipulation_index", {})
        obs_lexical = obs_features.get("lexical_diversity", {})
        obs_pronouns = obs_features.get("pronoun_distribution", {})

        obs_cal = float(obs_epistemic.get("epistemic_calibration_ratio", 1.0))
        obs_dogma = float(obs_manip.get("dogmatic_density", 0.40))
        obs_ttr = float(obs_lexical.get("type_token_ratio", 0.60))
        obs_i_pct = float(obs_pronouns.get("i_percent", 2.0))

        # Calculate mathematical Deltas
        d_cal = round(obs_cal - baseline["calibration_ratio"], 3)
        d_dogma = round(obs_dogma - baseline["dogmatic_density"], 3)
        d_ttr = round(obs_ttr - baseline["type_token_ratio"], 3)
        d_i_pct = round(obs_i_pct - baseline["pronoun_i_pct"], 3)

        # Concept Drift & Salience Analysis
        obs_terms = [t[0].lower() for t in obs_features.get("top_characteristic_terms", []) if isinstance(t, (list, tuple))]
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

        # Executive LLM synthesis
        system_prompt = (
            f"You are the Longitudinal Cognitive Drift Analyst for {self.target_name}.\n"
            "Assess cognitive drift scientifically without sensationalism or deterministic overclaims."
        )
        user_prompt = (
            f"TARGET: {self.target_name}\n"
            f"LONGITUDINAL DRIFT AUDIT:\n"
            f"- Epistemic Phase Shift: {verdict} ({verdict_desc})\n"
            f"- Δ Calibration Ratio: {d_cal} (Baseline: {baseline['calibration_ratio']} -> Observed: {obs_cal})\n"
            f"- Δ Dogmatic Density: {d_dogma} (Baseline: {baseline['dogmatic_density']} -> Observed: {obs_dogma})\n"
            f"- Δ Lexical Diversity (TTR): {d_ttr} (Baseline: {baseline['type_token_ratio']} -> Observed: {obs_ttr})\n"
            f"- Concept Stability: {concept_stability_pct}%\n"
            f"- Newly Surging Concepts: {surging_concepts[:5]}\n\n"
            "Provide:\n"
            "1. Epistemic Trajectory Analysis\n"
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
            "conceptual_shifts": {
                "surging_concepts": surging_concepts[:6],
                "retained_concepts": retained_concepts[:6],
            },
            "drift_synthesis": drift_synthesis,
            "epistemic_qualification": (
                "Temporal cognitive drift represents mathematical divergence between baseline corpus features "
                "and recent observable footprint. Not a permanent psychological diagnosis."
            ),
        }
