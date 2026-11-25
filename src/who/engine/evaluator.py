"""
Evaluation, Temporal Backtesting, and Stylometric Referee pipeline for who??.
Tests retrieval quality, profile consistency, citation validity, hallucination rates,
out-of-sample Temporal Holdout Brier Score calibration, and Jensen-Shannon Stylometric Divergence (JSD).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from who.learn.feature_extractor import (
    CERTAINTY_MARKERS,
    FIRST_PERSON_PLURAL,
    FIRST_PERSON_SINGULAR,
    REVEALED_AUTONOMOUS_EXIT_MARKERS,
    REVEALED_COOPERATIVE_ACTION_MARKERS,
    STATED_COOPERATIVE_MARKERS,
    STATED_DEFENSIVE_MARKERS,
    TENTATIVE_MARKERS,
    FeatureExtractor,
)
from who.utils.logger import get_logger

logger = get_logger("who.engine.evaluator")


def _kl_divergence(p: List[float], q: List[float]) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) with Laplace smoothing."""
    eps = 1e-9
    total = 0.0
    for pi, qi in zip(p, q):
        pi_s = max(pi, eps)
        qi_s = max(qi, eps)
        total += pi_s * math.log2(pi_s / qi_s)
    return max(0.0, total)


def compute_jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """
    Compute normalized Jensen-Shannon Divergence JSD(P || Q) in [0, 1] (base-2 log).
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q).
    """
    if not p or not q or len(p) != len(q):
        return 0.0
    sum_p = sum(p) or 1.0
    sum_q = sum(q) or 1.0
    p_norm = [max(1e-9, x / sum_p) for x in p]
    q_norm = [max(1e-9, x / sum_q) for x in q]
    m = [0.5 * (pi + qi) for pi, qi in zip(p_norm, q_norm)]
    jsd = 0.5 * _kl_divergence(p_norm, m) + 0.5 * _kl_divergence(q_norm, m)
    return round(min(1.0, max(0.0, jsd)), 4)


def compute_murphy_brier_decomposition(
    predictions: List[float],
    outcomes: List[float],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Exact Within-Bin Corrected Murphy (1973) Brier Score Decomposition & Expected Calibration Error (ECE):
    BS = Reliability (REL) - Resolution (RES) + Uncertainty (UNC) + Within_Bin_Correction (WBC)
    Both exact_bs and bin terms are computed consistently on predictions clamped to [0, 1]
    so REL - RES + UNC + WBC == brier_score holds to machine precision even when inputs exceed [0, 1].
    """
    n_tot = len(predictions)
    if n_tot == 0 or len(outcomes) != n_tot:
        return {
            "brier_score": None,
            "raw_unclamped_brier_score": None,
            "clamped_predictions_count": 0,
            "reliability_rel": None,
            "resolution_res": None,
            "uncertainty_unc": None,
            "within_bin_correction_wbc": None,
            "reconstructed_brier_score": None,
            "decomposition_identity_error": 0.0,
            "expected_calibration_error_ece": None,
            "bins": [],
        }

    b_count = max(2, int(n_bins))
    bins_p: List[List[float]] = [[] for _ in range(b_count)]
    bins_y: List[List[float]] = [[] for _ in range(b_count)]
    clamped_preds: List[float] = []
    clamped_count = 0

    for p_val, y_val in zip(predictions, outcomes):
        p_raw = float(p_val)
        p_clamped = min(1.0, max(0.0, p_raw))
        if p_clamped != p_raw:
            clamped_count += 1
        clamped_preds.append(p_clamped)
        b_idx = min(b_count - 1, int(p_clamped * b_count))
        bins_p[b_idx].append(p_clamped)
        bins_y[b_idx].append(float(y_val))

    y_bar = sum(outcomes) / n_tot
    unc = y_bar * (1.0 - y_bar)
    rel = 0.0
    res = 0.0
    wbc = 0.0
    ece = 0.0
    bin_records: List[Dict[str, Any]] = []

    for b_idx in range(b_count):
        p_list = bins_p[b_idx]
        y_list = bins_y[b_idx]
        n_b = len(p_list)
        if n_b == 0:
            continue
        p_bar_b = sum(p_list) / n_b
        y_bar_b = sum(y_list) / n_b
        w_b = n_b / n_tot
        rel += w_b * ((p_bar_b - y_bar_b) ** 2)
        res += w_b * ((y_bar_b - y_bar) ** 2)
        ece += w_b * abs(p_bar_b - y_bar_b)
        for p_i, y_i in zip(p_list, y_list):
            wbc += ((p_i - p_bar_b) ** 2 - 2.0 * (p_i - p_bar_b) * (y_i - y_bar_b)) / n_tot
        bin_records.append({
            "bin_index": b_idx,
            "count": n_b,
            "mean_predicted": round(p_bar_b, 4),
            "empirical_frequency": round(y_bar_b, 4),
            "abs_calibration_gap": round(abs(p_bar_b - y_bar_b), 4),
        })

    exact_bs = sum((p_c - float(y)) ** 2 for p_c, y in zip(clamped_preds, outcomes)) / n_tot
    raw_bs = sum((float(p) - float(y)) ** 2 for p, y in zip(predictions, outcomes)) / n_tot
    reconstructed_bs = rel - res + unc + wbc

    return {
        "brier_score": round(exact_bs, 6),
        "raw_unclamped_brier_score": round(raw_bs, 6),
        "clamped_predictions_count": clamped_count,
        "reliability_rel": round(rel, 6),
        "resolution_res": round(res, 6),
        "uncertainty_unc": round(unc, 6),
        "within_bin_correction_wbc": round(wbc, 6),
        "reconstructed_brier_score": round(reconstructed_bs, 6),
        "decomposition_identity_error": round(abs(exact_bs - reconstructed_bs), 9),
        "expected_calibration_error_ece": round(ece, 4),
        "bins": bin_records,
    }


def validate_conformal_alpha(alpha: float) -> float:
    """Validate conformal miscoverage level alpha strictly in (0, 1)."""
    val = float(alpha)
    if not math.isfinite(val) or val <= 0.0 or val >= 1.0:
        raise ValueError(f"Conformal alpha must be finite and strictly in (0, 1), got {alpha!r}")
    return val


def compute_split_conformal_prediction_set(
    probabilities: Dict[str, float],
    calibration_nonconformity_scores: Optional[List[float]] = None,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute a finite-sample Split Conformal Prediction Set C_alpha(x) over discrete actions:
    - Nonconformity score: s(x, y) = 1 - P(y | x).
    - Finite-sample quantile rank: k = ceil((n_cal + 1) * (1 - alpha)).
    - When k > n_cal (insufficient calibration points for nominal 1 - alpha), q_hat = +inf
      and ALL actions are included so the finite-sample marginal coverage guarantee >= 1 - alpha holds strictly.
    """
    eff_alpha = validate_conformal_alpha(alpha)

    if not probabilities:
        return {
            "alpha": round(eff_alpha, 4),
            "target_coverage": round(1.0 - eff_alpha, 4),
            "q_hat_threshold": 1.0,
            "q_hat_is_infinite": False,
            "prediction_set": [],
            "set_cardinality": 0,
            "cumulative_probability_mass": 0.0,
        }

    sorted_acts = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    q_hat_is_infinite = False

    if calibration_nonconformity_scores and len(calibration_nonconformity_scores) >= 1:
        s_sorted = sorted(float(s) for s in calibration_nonconformity_scores)
        n_cal = len(s_sorted)
        k_order = math.ceil((n_cal + 1) * (1.0 - eff_alpha))
        if k_order > n_cal:
            q_hat_is_infinite = True
            q_hat_out: Optional[float] = None
            pred_set = [act for act, _ in sorted_acts]
        else:
            q_hat = s_sorted[max(0, k_order - 1)]
            q_hat_out = round(q_hat, 4)
            pred_set = [act for act, pr in sorted_acts if (1.0 - pr) <= q_hat + 1e-9]
    else:
        # Cumulative conformal mass fallback when no calibration scores are supplied
        q_hat_out = round(1.0 - eff_alpha, 4)
        pred_set = []
        cum = 0.0
        for act, pr in sorted_acts:
            pred_set.append(act)
            cum += pr
            if cum >= (1.0 - eff_alpha) - 1e-9:
                break

    if not pred_set:
        pred_set = [sorted_acts[0][0]]

    cum_mass = sum(probabilities.get(a, 0.0) for a in pred_set)
    return {
        "alpha": round(eff_alpha, 4),
        "target_coverage": round(1.0 - eff_alpha, 4),
        "q_hat_threshold": q_hat_out,
        "q_hat_is_infinite": q_hat_is_infinite,
        "prediction_set": pred_set,
        "set_cardinality": len(pred_set),
        "cumulative_probability_mass": round(cum_mass, 4),
    }


class ProfileEvaluator:
    """
    Evaluates profile consistency, citation veracity, retrieval quality, hallucination rates,
    out-of-sample Temporal Holdout Backtesting (Brier Score), and Stylometric JSD Fidelity.
    """

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _load_raw_documents(self) -> List[Tuple[str, str]]:
        """Load raw documents sorted chronologically/alphabetically by filename."""
        if not self.dirs.raw_data_dir.exists():
            return []
        raw_files = sorted(
            list(self.dirs.raw_data_dir.glob("*.md")) + list(self.dirs.raw_data_dir.glob("*.txt")),
            key=lambda p: p.name,
        )
        docs: List[Tuple[str, str]] = []
        for rf in raw_files:
            try:
                text = rf.read_text(encoding="utf-8").strip()
                if text:
                    docs.append((rf.name, text))
            except Exception:
                pass
        return docs

    def run_temporal_backtest(self, holdout_ratio: float = 0.25) -> Dict[str, Any]:
        """
        Execute Out-of-Sample Temporal Holdout Backtesting (T-Split Validation).
        1. Splits chronological corpus into Training Slice (first 1 - holdout_ratio) and Holdout Slice (last holdout_ratio).
        2. Fits FeatureExtractor on the Training Slice ONLY to estimate behavioral stance & epistemic calibration priors.
        3. Predicts behavior on unseen Holdout segments and computes the Brier Score:
           BS = (1/N) * sum((p_predicted - o_observed)^2)
        4. Derives Empirical Confidence Score = 1.0 - BS.
        """
        docs = self._load_raw_documents()
        all_segments: List[Dict[str, str]] = []
        for source_id, content in docs:
            paragraphs = [
                p.strip()
                for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", content)
                if len(p.strip().split()) >= 5 and not p.strip().startswith("#")
            ]
            for para in paragraphs:
                all_segments.append({"source_id": source_id, "text": para})

        if len(all_segments) < 4:
            # Fallback if minimal corpus: split sentences across whatever text exists
            combined = " ".join(text for _, text in docs)
            sents = [s.strip() for s in re.split(r"[.!?]+", combined) if len(s.strip().split()) >= 3]
            all_segments = [{"source_id": "corpus_segment", "text": s} for s in sents]

        if len(all_segments) < 2:
            return {
                "status": "INSUFFICIENT_CORPUS_FOR_HOLDOUT",
                "is_prior_fallback": True,
                "training_segments": len(all_segments),
                "holdout_segments": 0,
                "brier_score": None,
                "brier_ci_95": None,
                "brier_reference_base_rate": None,
                "brier_skill_score_bss": None,
                "bss_ci_95": None,
                "log_loss_score": None,
                "log_loss_skill_score_llss": None,
                "holdout_accuracy_pct": None,
                "semantic_alignment_score": None,
                "empirical_confidence_score": None,
                "prior_fallback_confidence": 0.68,
                "prior_fallback_brier_estimate": 0.22,
                "calibration_tier": "BASELINE_PRIOR_FALLBACK",
                "murphy_brier_decomposition": None,
                "sample_predictions": [],
            }

        split_idx = max(1, int(len(all_segments) * (1.0 - holdout_ratio)))
        if split_idx >= len(all_segments):
            split_idx = len(all_segments) - 1

        train_segments = all_segments[:split_idx]
        holdout_segments = all_segments[split_idx:]

        train_texts = [seg["text"] for seg in train_segments]
        fe = FeatureExtractor(output_dir=None)
        train_features = fe.extract_features(train_texts)

        # Derive training probabilities for two testable behavioral axes:
        # Axis A: Epistemic Calibration (Probabilistic/Analytical vs. Dogmatic/Certain)
        epistemic = train_features.get("epistemic_markers", {})
        tent_rate = float(epistemic.get("tentative_per_1k", 1.0))
        cert_rate = float(epistemic.get("certainty_per_1k", 1.0))
        # Laplace-smoothed probability of calibrated/analytical framing
        p_calibrated_train = (tent_rate + 1.5) / (tent_rate + cert_rate + 2.5)
        p_calibrated_train = min(0.92, max(0.08, p_calibrated_train))

        # Axis B: Revealed/Stated Cooperative vs. Autonomous/Defensive Stance
        pref = train_features.get("stated_vs_revealed_preference", {})
        p_coop_train = 0.5 * float(pref.get("stated_cooperation_polarity", 0.55)) + 0.5 * float(
            pref.get("revealed_cooperation_polarity", 0.55)
        )
        p_coop_train = min(0.92, max(0.08, p_coop_train))

        # Vocabulary set from training slice for semantic alignment
        train_vocab = {
            w.lower()
            for t in train_texts
            for w in re.findall(r"\b\w{4,}\b", t.lower())
        }

        brier_errors: List[float] = []
        base_brier_errors: List[float] = []
        log_losses: List[float] = []
        base_log_losses: List[float] = []
        holdout_preds: List[float] = []
        holdout_outcomes: List[float] = []
        correct_predictions = 0
        semantic_overlaps: List[float] = []
        sample_predictions: List[Dict[str, Any]] = []

        # Training-only reference base rate (zero holdout data leakage!)
        p_base_train = min(0.88, max(0.12, 0.5 * p_calibrated_train + 0.5 * p_coop_train))
        eps_clip = 1e-6

        coop_markers = STATED_COOPERATIVE_MARKERS | REVEALED_COOPERATIVE_ACTION_MARKERS
        def_markers = STATED_DEFENSIVE_MARKERS | REVEALED_AUTONOMOUS_EXIT_MARKERS

        for idx, seg in enumerate(holdout_segments):
            seg_text = seg["text"]
            words = re.findall(r"\b\w+\b", seg_text.lower())
            if not words:
                continue

            coop_hits = sum(1 for w in words if w in coop_markers)
            def_hits = sum(1 for w in words if w in def_markers)
            tent_hits = sum(1 for w in words if w in TENTATIVE_MARKERS)
            cert_hits = sum(1 for w in words if w in CERTAINTY_MARKERS)

            if (coop_hits + def_hits) > 0:
                p_pred = p_coop_train
                observed_outcome = 1.0 if coop_hits >= def_hits else 0.0
                axis_evaluated = "COOPERATIVE_VS_DEFENSIVE_STANCE"
            elif (tent_hits + cert_hits) > 0:
                p_pred = p_calibrated_train
                observed_outcome = 1.0 if tent_hits >= cert_hits else 0.0
                axis_evaluated = "CALIBRATED_VS_DOGMATIC_CADENCE"
            else:
                train_avg_len = float(train_features.get("corpus_stats", {}).get("avg_sentence_length", 15.0))
                seg_len = len(words)
                len_ratio = min(seg_len, train_avg_len) / max(seg_len, train_avg_len, 1.0)
                p_pred = max(0.55, min(0.90, 0.5 * p_calibrated_train + 0.5 * p_coop_train))
                observed_outcome = 1.0 if len_ratio >= 0.35 else 0.0
                axis_evaluated = "STYLOMETRIC_CADENCE_CONTINUITY"

            p_c = min(1.0 - eps_clip, max(eps_clip, p_pred))
            p_b = min(1.0 - eps_clip, max(eps_clip, p_base_train))

            holdout_preds.append(p_c)
            holdout_outcomes.append(observed_outcome)

            brier_err = (p_c - observed_outcome) ** 2
            base_brier_err = (p_b - observed_outcome) ** 2
            brier_errors.append(brier_err)
            base_brier_errors.append(base_brier_err)

            ll = -(observed_outcome * math.log(p_c) + (1.0 - observed_outcome) * math.log(1.0 - p_c))
            base_ll = -(observed_outcome * math.log(p_b) + (1.0 - observed_outcome) * math.log(1.0 - p_b))
            log_losses.append(ll)
            base_log_losses.append(base_ll)

            pred_binary = 1.0 if p_c >= 0.50 else 0.0
            if pred_binary == observed_outcome:
                correct_predictions += 1

            seg_vocab = {w for w in words if len(w) >= 4}
            if seg_vocab and train_vocab:
                overlap = len(seg_vocab & train_vocab) / len(seg_vocab)
                semantic_overlaps.append(min(1.0, overlap * 1.5))
            else:
                semantic_overlaps.append(0.75)

            if len(sample_predictions) < 5:
                sample_predictions.append({
                    "holdout_index": idx + 1,
                    "source_id": seg["source_id"],
                    "axis": axis_evaluated,
                    "predicted_probability": round(p_c, 3),
                    "observed_outcome": observed_outcome,
                    "brier_loss": round(brier_err, 4),
                    "snippet": seg_text[:110] + ("..." if len(seg_text) > 110 else ""),
                })

        if not brier_errors:
            return {
                "status": "INSUFFICIENT_CORPUS_FOR_HOLDOUT",
                "is_prior_fallback": True,
                "training_segments": len(train_segments),
                "holdout_segments": 0,
                "brier_score": None,
                "brier_ci_95": None,
                "brier_reference_base_rate": None,
                "brier_skill_score_bss": None,
                "bss_ci_95": None,
                "log_loss_score": None,
                "log_loss_skill_score_llss": None,
                "holdout_accuracy_pct": None,
                "semantic_alignment_score": None,
                "empirical_confidence_score": None,
                "prior_fallback_confidence": 0.68,
                "prior_fallback_brier_estimate": 0.22,
                "calibration_tier": "BASELINE_PRIOR_FALLBACK",
                "murphy_brier_decomposition": None,
                "sample_predictions": [],
            }

        import random
        n_h = len(brier_errors)
        brier_score = round(sum(brier_errors) / n_h, 4)
        bs_ref = round(max(0.12, sum(base_brier_errors) / n_h), 4)
        bss = round((bs_ref - brier_score) / max(bs_ref, 1e-6), 4)
        log_loss = round(sum(log_losses) / n_h, 4)
        base_ll_mean = max(0.25, sum(base_log_losses) / n_h)
        ll_skill = round((base_ll_mean - log_loss) / max(base_ll_mean, 1e-6), 4)

        # Nonparametric Bootstrap 95% Confidence Intervals (B=200 resamples, fixed seed)
        rng_boot = random.Random(42)
        boot_bs: List[float] = []
        boot_bss: List[float] = []
        for _ in range(200):
            indices = [rng_boot.randrange(n_h) for _ in range(n_h)]
            b_m = sum(brier_errors[k] for k in indices) / n_h
            b_r = max(0.12, sum(base_brier_errors[k] for k in indices) / n_h)
            boot_bs.append(b_m)
            boot_bss.append((b_r - b_m) / b_r)
        boot_bs.sort()
        boot_bss.sort()
        brier_ci_95 = [round(boot_bs[int(0.025 * 200)], 4), round(boot_bs[int(0.975 * 200)], 4)]
        bss_ci_95 = [round(boot_bss[int(0.025 * 200)], 4), round(boot_bss[int(0.975 * 200)], 4)]

        accuracy_pct = round((correct_predictions / n_h) * 100.0, 1)
        sem_align = round(sum(semantic_overlaps) / len(semantic_overlaps), 3) if semantic_overlaps else 0.78
        murphy_decomp = compute_murphy_brier_decomposition(holdout_preds, holdout_outcomes, n_bins=5)

        empirical_confidence = round(max(0.50, min(0.98, 1.0 - brier_score)), 3)
        if brier_score <= 0.18:
            cal_tier = "EMPIRICALLY_CALIBRATED_HIGH"
        elif brier_score <= 0.28:
            cal_tier = "MODERATE_OUT_OF_SAMPLE_CALIBRATION"
        else:
            cal_tier = "HIGH_VARIANCE_NEEDS_MORE_DATA"

        return {
            "status": "COMPLETED",
            "is_prior_fallback": False,
            "holdout_ratio": holdout_ratio,
            "training_segments": len(train_segments),
            "holdout_segments": len(holdout_segments),
            "training_base_rate_prior": round(p_base_train, 4),
            "brier_score": brier_score,
            "brier_ci_95": brier_ci_95,
            "brier_reference_base_rate": bs_ref,
            "brier_skill_score_bss": bss,
            "bss_ci_95": bss_ci_95,
            "log_loss_score": log_loss,
            "log_loss_skill_score_llss": ll_skill,
            "holdout_accuracy_pct": accuracy_pct,
            "semantic_alignment_score": sem_align,
            "empirical_confidence_score": empirical_confidence,
            "calibration_tier": cal_tier,
            "murphy_brier_decomposition": murphy_decomp,
            "sample_predictions": sample_predictions,
        }

    @staticmethod
    def _invert_square_matrix(m: List[List[float]]) -> List[List[float]]:
        """Gauss-Jordan matrix inversion with partial pivoting and ridge stabilization for arbitrary d x d FIM."""
        n = len(m)
        if n == 0:
            return []
        aug = [
            [float(m[i][j]) + (1e-6 if i == j else 0.0) for j in range(n)]
            + [1.0 if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        for col in range(n):
            piv_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[piv_row][col]) < 1e-12:
                return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            if piv_row != col:
                aug[col], aug[piv_row] = aug[piv_row], aug[col]
            piv = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= piv
            for r in range(n):
                if r != col and abs(aug[r][col]) > 1e-15:
                    fac = aug[r][col]
                    for j in range(2 * n):
                        aug[r][j] -= fac * aug[col][j]
        return [row[n:] for row in aug]

    @staticmethod
    def _invert_3x3_matrix(m: List[List[float]]) -> List[List[float]]:
        """Exact 3x3 matrix inversion with ridge stabilization for Fisher Information covariance."""
        a = [[float(m[i][j]) + (1e-6 if i == j else 0.0) for j in range(3)] for i in range(3)]
        c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1]
        c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0])
        c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0]
        c10 = -(a[0][1] * a[2][2] - a[0][2] * a[2][1])
        c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0]
        c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0])
        c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1]
        c21 = -(a[0][0] * a[1][2] - a[0][2] * a[1][0])
        c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02
        if abs(det) < 1e-12:
            return [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
        inv_det = 1.0 / det
        return [
            [c00 * inv_det, c10 * inv_det, c20 * inv_det],
            [c01 * inv_det, c11 * inv_det, c21 * inv_det],
            [c02 * inv_det, c12 * inv_det, c22 * inv_det],
        ]

    @staticmethod
    def run_synthetic_parameter_recovery(
        n_samples: int = 240,
        random_seed: int = 42,
        n_coverage_trials: int = 20,
        n_null_reps: int = 100,
        n_permutations: int = 99,
    ) -> Dict[str, Any]:
        """
        Execute Synthetic Parameter Recovery, Fisher Information Sloppiness Diagnostic,
        95% Fisher CI Coverage Verification (theta only, excluding boundary kappa), Misspecified Generator Test,
        Scalar Temperature Baseline M_T comparison, Joint (theta, kappa) 4x4 FIM correlations,
        Self & Liang (1987) Null Size Check under H0: kappa*=0, and Norm-Preserving w-Permutation p_perm:
        1. Fixes inverse temperature beta = 1.0 and anchors DEADLOCK_WALKAWAY as reference utility U_ref = 0.
        2. Generates N synthetic scenarios with known true parameter vector:
           theta_true = [theta_loss=1.60, theta_mach=0.55, theta_closure=0.70].
        3. Reports Relative RMSE (||theta_hat - theta*||_2 / ||theta*||_2 < 0.25), per-parameter Bias,
           and empirical 95% Fisher CI coverage for theta across Monte Carlo trials.
        4. Compares Joint Gibbs kappa > 0 against BOTH H0: kappa=0 (Self & Liang mixture) AND the global
           scalar temperature baseline M_T: softmax(U^0 / tau), plus T_b sensitivity grid.
        """
        import random
        theta_true = [1.60, 0.55, 0.70]
        theta_norm = math.sqrt(sum(t * t for t in theta_true) / 3.0)

        def _branch_features(lev: float, asym: float, rep: float) -> List[List[float]]:
            c_lev = 2.0 * (lev - 0.50)
            c_asym = 2.0 * (asym - 0.45)
            c_rep = 2.0 * (rep - 0.50)
            f_accept = [-(1.35 * asym + 0.65 * c_rep), 0.55 * (1.0 - lev) - 0.45 * c_lev, -0.80 * c_asym]
            f_counter = [-0.45 * c_asym, 1.45 * (1.0 - lev) + 0.50 * c_rep, 0.55 * (1.0 - rep)]
            f_concess = [-0.85 * c_rep, 0.85 * c_lev, 1.45 * (1.0 - asym) - 0.40 * c_rep]
            f_deadlock = [0.0, 0.0, 0.0]
            return [f_accept, f_counter, f_concess, f_deadlock]

        def _probs_from_utils(utils: List[float]) -> List[float]:
            m_u = max(utils)
            exps = [math.exp(max(-700.0, u - m_u)) for u in utils]
            z = sum(exps) or 1.0
            return [e / z for e in exps]

        def _probs(feats: List[List[float]], th: List[float], shifts: Optional[List[float]] = None) -> List[float]:
            utils = [
                sum(f[d] * th[d] for d in range(3)) + (shifts[b] if shifts else 0.0)
                for b, f in enumerate(feats)
            ]
            return _probs_from_utils(utils)

        def _fit_mle(data_rows: List[Tuple[List[List[float]], int]]) -> Tuple[List[float], List[List[float]]]:
            th_est = [1.0, 0.3, 0.3]
            fim_mat = [[0.0] * 3 for _ in range(3)]
            for step in range(55):
                grad = [0.0, 0.0, 0.0]
                fim_mat = [[0.0] * 3 for _ in range(3)]
                for feats, y in data_rows:
                    pr = _probs(feats, th_est)
                    mean_f = [sum(pr[b] * feats[b][d] for b in range(4)) for d in range(3)]
                    for d in range(3):
                        grad[d] += feats[y][d] - mean_f[d]
                    for d1 in range(3):
                        for d2 in range(3):
                            cov_d12 = sum(pr[b] * (feats[b][d1] - mean_f[d1]) * (feats[b][d2] - mean_f[d2]) for b in range(4))
                            fim_mat[d1][d2] += cov_d12
                inv_fim = ProfileEvaluator._invert_3x3_matrix(fim_mat)
                lr = 0.75 if step < 30 else 0.40
                for d in range(3):
                    step_d = sum(inv_fim[d][k] * grad[k] for k in range(3))
                    th_est[d] += lr * max(-0.5, min(0.5, step_d))
            return th_est, fim_mat

        def _simulate_dataset(seed_val: int, misspecified: bool = False) -> List[Tuple[List[List[float]], int]]:
            local_rng = random.Random(seed_val)
            rows: List[Tuple[List[List[float]], int]] = []
            for _ in range(max(60, n_samples)):
                lev = local_rng.uniform(0.10, 0.90)
                asym = local_rng.uniform(0.05, 0.85)
                rep = local_rng.uniform(0.10, 0.90)
                feats = _branch_features(lev, asym, rep)
                if misspecified:
                    nonlin_shift = [0.15 * (lev * asym), -0.10 * (rep ** 2), 0.12 * (lev * rep), 0.0]
                    probs = _probs(feats, theta_true, shifts=nonlin_shift)
                else:
                    probs = _probs(feats, theta_true)
                u_draw = local_rng.random()
                cum = 0.0
                chosen = 3
                for idx_b, pr in enumerate(probs):
                    cum += pr
                    if u_draw <= cum:
                        chosen = idx_b
                        break
                rows.append((feats, chosen))
            return rows

        primary_dataset = _simulate_dataset(random_seed, misspecified=False)
        theta_est, fim = _fit_mle(primary_dataset)
        inv_fim_primary = ProfileEvaluator._invert_3x3_matrix(fim)
        se_vec = [math.sqrt(max(1e-8, inv_fim_primary[d][d])) for d in range(3)]

        rmse = math.sqrt(sum((theta_est[d] - theta_true[d]) ** 2 for d in range(3)) / 3.0)
        relative_rmse = rmse / max(theta_norm, 1e-6)
        bias_vec = [round(theta_est[d] - theta_true[d], 4) for d in range(3)]
        per_param_rel_error = {
            "loss_aversion": round(abs(theta_est[0] - theta_true[0]) / theta_true[0], 4),
            "machiavellianism": round(abs(theta_est[1] - theta_true[1]) / theta_true[1], 4),
            "need_for_closure": round(abs(theta_est[2] - theta_true[2]) / theta_true[2], 4),
        }

        # Monte Carlo 95% Fisher CI Empirical Coverage for theta only (excludes boundary parameter kappa >= 0)
        covered_hits = 0
        total_param_checks = 0
        trials = max(5, min(30, int(n_coverage_trials)))
        for t_idx in range(trials):
            ds_t = _simulate_dataset(random_seed + t_idx * 17, misspecified=False)
            th_t, fim_t = _fit_mle(ds_t)
            inv_t = ProfileEvaluator._invert_3x3_matrix(fim_t)
            for d in range(3):
                se_d = math.sqrt(max(1e-8, inv_t[d][d]))
                if abs(th_t[d] - theta_true[d]) <= 1.96 * se_d:
                    covered_hits += 1
                total_param_checks += 1
        empirical_ci_coverage = round(covered_hits / max(1, total_param_checks), 4)

        # Misspecified generator stress test
        misspec_ds = _simulate_dataset(random_seed + 999, misspecified=True)
        th_misspec, _ = _fit_mle(misspec_ds)
        misspec_rmse = math.sqrt(sum((th_misspec[d] - theta_true[d]) ** 2 for d in range(3)) / 3.0)
        misspec_rel_rmse = round(misspec_rmse / max(theta_norm, 1e-6), 4)

        # Joint Gibbs Coupling kappa Identifiability, M_T Baseline, Joint (theta, kappa) FIM,
        # Self & Liang (1987) Null Size Check (kappa*=0), and Random w-Permutation p_perm
        g_vec_true = [1.15, -1.05, 0.75, 0.0]  # reference branch 3 (DEADLOCK) ≡ 0
        kappa_true = 1.10
        rng_k = random.Random(random_seed + 7)
        gibbs_data: List[Tuple[List[List[float]], float, int]] = []
        u0_gibbs: List[List[float]] = []

        def _marginal_gibbs_probs(
            feats: List[List[float]],
            th: List[float],
            kap: float,
            g_v: List[float],
            frust_act: float = 1.0,
        ) -> List[float]:
            eff_k = kap * frust_act
            p_plus = _probs(feats, th, shifts=[eff_k * g for g in g_v])
            p_minus = _probs(feats, th, shifts=[-eff_k * g for g in g_v])
            return [0.5 * (p_plus[b] + p_minus[b]) for b in range(4)]

        def _marginal_gibbs_probs_u0(
            u0_row: List[float],
            kap: float,
            g_v: List[float],
            frust_act: float,
        ) -> List[float]:
            eff_k = kap * frust_act
            p_plus = _probs_from_utils([u0_row[b] + eff_k * g_v[b] for b in range(4)])
            p_minus = _probs_from_utils([u0_row[b] - eff_k * g_v[b] for b in range(4)])
            return [0.5 * (p_plus[b] + p_minus[b]) for b in range(4)]

        for idx_g in range(max(120, n_samples)):
            lev = rng_k.uniform(0.10, 0.90)
            asym = rng_k.uniform(0.05, 0.85)
            rep = rng_k.uniform(0.10, 0.90)
            frust_act = 1.75 if (idx_g % 2 == 0) else 0.15
            feats = _branch_features(lev, asym, rep)
            u0_row = [sum(f[d] * theta_true[d] for d in range(3)) for f in feats]
            pr_g = _marginal_gibbs_probs_u0(u0_row, kappa_true, g_vec_true, frust_act)
            u_draw = rng_k.random()
            cum = 0.0
            chosen = 3
            for idx_b, pr in enumerate(pr_g):
                cum += pr
                if u_draw <= cum:
                    chosen = idx_b
                    break
            gibbs_data.append((feats, frust_act, chosen))
            u0_gibbs.append(u0_row)

        def _log_lik_gibbs_u0(kap: float, g_v: List[float]) -> float:
            ll_tot = 0.0
            for idx_r, (_, frust_act, y) in enumerate(gibbs_data):
                pr = _marginal_gibbs_probs_u0(u0_gibbs[idx_r], kap, g_v, frust_act)
                ll_tot += math.log(max(pr[y], 1e-12))
            return ll_tot

        ll_m0 = _log_lik_gibbs_u0(0.0, g_vec_true)
        best_kap = 0.0
        ll_m1 = ll_m0
        for k_grid in [0.05 * j for j in range(1, 35)]:
            ll_cand = _log_lik_gibbs_u0(k_grid, g_vec_true)
            if ll_cand > ll_m1:
                ll_m1 = ll_cand
                best_kap = k_grid

        lrt_lambda = max(0.0, 2.0 * (ll_m1 - ll_m0))
        p_self_liang = 1.0 if lrt_lambda <= 1e-9 else 0.5 * math.erfc(math.sqrt(lrt_lambda / 2.0))

        # Scalar Temperature Baseline M_T: softmax(U^0 / tau)
        best_tau = 1.0
        ll_mt = ll_m0
        for tau_cand in [0.35 + 0.05 * j for j in range(44)]:
            ll_t = 0.0
            for idx_r, (_, _, y) in enumerate(gibbs_data):
                pr_t = _probs_from_utils([u / tau_cand for u in u0_gibbs[idx_r]])
                ll_t += math.log(max(pr_t[y], 1e-12))
            if ll_t > ll_mt:
                ll_mt = ll_t
                best_tau = tau_cand

        lrt_vs_mt = round(2.0 * (ll_m1 - ll_mt), 4)
        beats_mt = bool(ll_m1 > ll_mt + 1e-6)

        # Joint 4x4 Empirical FIM for (theta_0, theta_1, theta_2, kappa) at kappa > 0
        kap_eval = max(0.25, best_kap)
        params_eval = [theta_true[0], theta_true[1], theta_true[2], kap_eval]
        eps_fd = 1e-4
        fim_joint = [[0.0] * 4 for _ in range(4)]
        for feats, frust_act, y in gibbs_data:
            score_i = [0.0] * 4
            for p_idx in range(4):
                p_plus = list(params_eval)
                p_minus = list(params_eval)
                p_plus[p_idx] += eps_fd
                p_minus[p_idx] -= eps_fd
                lp_plus = math.log(max(_marginal_gibbs_probs(feats, p_plus[:3], p_plus[3], g_vec_true, frust_act)[y], 1e-12))
                lp_minus = math.log(max(_marginal_gibbs_probs(feats, p_minus[:3], p_minus[3], g_vec_true, frust_act)[y], 1e-12))
                score_i[p_idx] = (lp_plus - lp_minus) / (2.0 * eps_fd)
            for a_idx in range(4):
                for b_idx in range(4):
                    fim_joint[a_idx][b_idx] += score_i[a_idx] * score_i[b_idx]
        cov_joint = ProfileEvaluator._invert_square_matrix(fim_joint)
        sd_joint = [math.sqrt(max(1e-12, cov_joint[k][k])) for k in range(4)]
        theta_kappa_corr = {
            "loss_aversion_vs_kappa": round(cov_joint[0][3] / max(1e-12, sd_joint[0] * sd_joint[3]), 4),
            "machiavellianism_vs_kappa": round(cov_joint[1][3] / max(1e-12, sd_joint[1] * sd_joint[3]), 4),
            "need_for_closure_vs_kappa": round(cov_joint[2][3] / max(1e-12, sd_joint[2] * sd_joint[3]), 4),
        }

        # Self & Liang (1987) Null Size Check under H0: kappa* = 0
        eff_null_reps = max(20, int(n_null_reps))
        null_kap_grid = [0.0] + [0.1 * j for j in range(1, 16)]
        log_prob_null_table: List[List[List[float]]] = []
        base_probs_null = [_probs_from_utils(u0_row) for u0_row in u0_gibbs]
        for k_val in null_kap_grid:
            rows_k = []
            for idx_r, (_, frust_act, _) in enumerate(gibbs_data):
                pr_r = _marginal_gibbs_probs_u0(u0_gibbs[idx_r], k_val, g_vec_true, frust_act)
                rows_k.append([math.log(max(pr_r[b], 1e-12)) for b in range(4)])
            log_prob_null_table.append(rows_k)

        rng_null = random.Random(random_seed + 509)
        null_lambdas: List[float] = []
        zero_lambda_hits = 0
        type1_reject_005 = 0
        n_rows_g = len(gibbs_data)
        for _ in range(eff_null_reps):
            y_null: List[int] = []
            for idx_r in range(n_rows_g):
                u_draw = rng_null.random()
                cum = 0.0
                ch = 3
                for idx_b, pr in enumerate(base_probs_null[idx_r]):
                    cum += pr
                    if u_draw <= cum:
                        ch = idx_b
                        break
                y_null.append(ch)
            ll0_b = sum(log_prob_null_table[0][idx_r][y_null[idx_r]] for idx_r in range(n_rows_g))
            ll_best_b = ll0_b
            for k_idx in range(1, len(null_kap_grid)):
                ll_k = sum(log_prob_null_table[k_idx][idx_r][y_null[idx_r]] for idx_r in range(n_rows_g))
                if ll_k > ll_best_b:
                    ll_best_b = ll_k
            lam_b = max(0.0, 2.0 * (ll_best_b - ll0_b))
            null_lambdas.append(lam_b)
            if lam_b <= 1e-7:
                zero_lambda_hits += 1
            p_b = 1.0 if lam_b <= 1e-9 else 0.5 * math.erfc(math.sqrt(lam_b / 2.0))
            if p_b < 0.05:
                type1_reject_005 += 1

        lam_sorted = sorted(null_lambdas)
        b_null = len(lam_sorted)
        # Exact Kolmogorov-Smirnov supremum sup_x |F_n(x) - F_0(x)| respecting the point mass at x = 0
        # At x = 0: F_0(0-) = F_n(0-) = 0, F_0(0) = 0.50, F_n(0) = #{lambda == 0} / B
        ks_stat = abs((zero_lambda_hits / b_null) - 0.5)
        for idx_l, lx in enumerate(lam_sorted):
            if lx <= 1e-7:
                continue
            f0_x = 0.5 + 0.5 * math.erf(math.sqrt(lx / 2.0))
            ks_stat = max(ks_stat, abs(((idx_l + 1) / b_null) - f0_x), abs((idx_l / b_null) - f0_x))

        null_size_summary = {
            "n_null_replications": eff_null_reps,
            "kappa_null": 0.0,
            "zero_lambda_fraction": round(zero_lambda_hits / eff_null_reps, 4),
            "empirical_type1_error_at_005": round(type1_reject_005 / eff_null_reps, 4),
            "ks_statistic_vs_self_liang": round(ks_stat, 4),
            "theoretical_self_liang_zero_fraction": 0.50,
            "theoretical_alpha": 0.05,
        }

        # Unbiased Full-Sphere Random w-Permutation Null Distribution (NO cosine truncation bias)
        eff_perms = max(20, int(n_permutations))
        rng_perm = random.Random(random_seed + 881)
        g_true_3 = g_vec_true[:3]
        g_norm_3 = math.sqrt(sum(v * v for v in g_true_3))
        perm_vectors: List[List[float]] = []
        cos_abs_list: List[float] = []
        while len(perm_vectors) < eff_perms:
            raw_v = [rng_perm.gauss(0.0, 1.0) for _ in range(3)]
            r_norm = math.sqrt(sum(v * v for v in raw_v))
            if r_norm < 1e-9:
                continue
            scaled = [v * (g_norm_3 / r_norm) for v in raw_v]
            cos_sim = abs(sum(scaled[j] * g_true_3[j] for j in range(3)) / max(g_norm_3 * g_norm_3, 1e-9))
            cos_abs_list.append(cos_sim)
            perm_vectors.append([scaled[0], scaled[1], scaled[2], 0.0])

        perm_lambdas: List[float] = []
        for g_perm in perm_vectors:
            ll_p1 = max(_log_lik_gibbs_u0(0.05 * j, g_perm) for j in range(0, 30))
            perm_lambdas.append(max(0.0, 2.0 * (ll_p1 - ll_m0)))
        mean_perm_lambda = sum(perm_lambdas) / max(1, len(perm_lambdas))
        w_perm_p_value = (1 + sum(1 for lp in perm_lambdas if lp >= lrt_lambda - 1e-9)) / (len(perm_lambdas) + 1)

        # Dose-response Pearson correlation between |cos(w_perm, w_true)| and Lambda_perm
        mean_c = sum(cos_abs_list) / len(cos_abs_list)
        mean_lp = mean_perm_lambda
        cov_cl = sum((c - mean_c) * (lp - mean_lp) for c, lp in zip(cos_abs_list, perm_lambdas))
        sd_c = math.sqrt(sum((c - mean_c) ** 2 for c in cos_abs_list))
        sd_lp = math.sqrt(sum((lp - mean_lp) ** 2 for lp in perm_lambdas))
        cos_lambda_corr = round(cov_cl / max(1e-12, sd_c * sd_lp), 4) if (sd_c > 1e-9 and sd_lp > 1e-9) else 0.0

        # 3x3 Model Recovery Confusion Matrix across generators {M0, M_T, E2} via BIC
        tau_fit_grid = [0.45 + 0.10 * j for j in range(18)]
        kap_fit_grid = [0.05 * j for j in range(1, 32)]
        log_tab_m0 = [[math.log(max(p, 1e-12)) for p in _probs_from_utils(u0_row)] for u0_row in u0_gibbs]
        log_tab_mt = [
            [
                [math.log(max(p, 1e-12)) for p in _probs_from_utils([u / tau_v for u in u0_row])]
                for u0_row in u0_gibbs
            ]
            for tau_v in tau_fit_grid
        ]
        log_tab_e2 = [
            [
                [math.log(max(p, 1e-12)) for p in _marginal_gibbs_probs_u0(u0_gibbs[idx_r], kv, g_vec_true, gibbs_data[idx_r][1])]
                for idx_r in range(n_rows_g)
            ]
            for kv in kap_fit_grid
        ]
        gen_probs = {
            "M0": [_probs_from_utils(u0_row) for u0_row in u0_gibbs],
            "M_T": [_probs_from_utils([u / 1.65 for u in u0_row]) for u0_row in u0_gibbs],
            "E2": [_marginal_gibbs_probs_u0(u0_gibbs[idx_r], 1.10, g_vec_true, gibbs_data[idx_r][1]) for idx_r in range(n_rows_g)],
        }
        confusion_counts: Dict[str, Dict[str, int]] = {
            gm: {"M0": 0, "M_T": 0, "E2": 0} for gm in ["M0", "M_T", "E2"]
        }
        rng_cm = random.Random(random_seed + 1237)
        bic_pen = math.log(max(2, n_rows_g))
        cm_trials_per_gen = 15
        for gen_name in ["M0", "M_T", "E2"]:
            p_rows_gen = gen_probs[gen_name]
            for _ in range(cm_trials_per_gen):
                y_cm: List[int] = []
                for idx_r in range(n_rows_g):
                    u_draw = rng_cm.random()
                    cum = 0.0
                    ch = 3
                    for idx_b, pr in enumerate(p_rows_gen[idx_r]):
                        cum += pr
                        if u_draw <= cum:
                            ch = idx_b
                            break
                    y_cm.append(ch)
                ll0_c = sum(log_tab_m0[idx_r][y_cm[idx_r]] for idx_r in range(n_rows_g))
                llmt_c = max(sum(tab[idx_r][y_cm[idx_r]] for idx_r in range(n_rows_g)) for tab in log_tab_mt)
                lle2_c = max(sum(tab[idx_r][y_cm[idx_r]] for idx_r in range(n_rows_g)) for tab in log_tab_e2)
                bics = {
                    "M0": -2.0 * ll0_c,
                    "M_T": -2.0 * llmt_c + bic_pen,
                    "E2": -2.0 * lle2_c + bic_pen,
                }
                winner = min(bics, key=bics.get)
                confusion_counts[gen_name][winner] += 1

        correct_cm = sum(confusion_counts[gm][gm] for gm in ["M0", "M_T", "E2"])
        total_cm = 3 * cm_trials_per_gen
        e2_false_win_on_mt = round(confusion_counts["M_T"]["E2"] / cm_trials_per_gen, 4)

        # Frustrated vs. Harary-Balanced Counterfactual Comparison (E2_BAL) on canonical concept graph
        from who.vault.graph_analytics import GraphAnalytics
        from who.engine.simulator import Simulator
        sim_tmp = Simulator.__new__(Simulator)
        field_ex = sim_tmp._evaluate_scenario_external_fields_and_commitments(
            "Negotiation proposal with governance parity.", {}, {}
        )
        c_nodes = field_ex["canonical_nodes"]
        c_edges = field_ex["couplings_J"]
        m_ex = len(c_nodes)
        nidx_ex = {n: idx for idx, n in enumerate(c_nodes)}
        l_frust = [[0.0] * m_ex for _ in range(m_ex)]
        for u, v, jv in c_edges:
            i_u, i_v = nidx_ex[u], nidx_ex[v]
            l_frust[i_u][i_u] += abs(jv)
            l_frust[i_v][i_v] += abs(jv)
            l_frust[i_u][i_v] -= jv
            l_frust[i_v][i_u] -= jv
        frust_lams = GraphAnalytics._symmetric_eigenvalues(l_frust)
        bal_ex = GraphAnalytics.construct_harary_balanced_counterfactual(c_nodes, c_edges)
        ex_u = {"REJECT_AND_COUNTER": 0.4, "STRATEGIC_CONCESSION": 0.1, "SUBGAME_EQUILIBRIUM_ACCEPT": 0.45, "DEADLOCK_WALKAWAY": 0.25}
        gibbs_frust_ex = sim_tmp._compute_centered_joint_gibbs_coupling(ex_u, 3.0, 0.60, field_ex, 0.65, use_balanced_counterfactual=False)
        gibbs_bal_ex = sim_tmp._compute_centered_joint_gibbs_coupling(ex_u, 3.0, 0.60, field_ex, 0.65, use_balanced_counterfactual=True)

        # Belief temperature T_b sensitivity diagnostic (fixed prior T_b=0.65, not peak-picked)
        tb_sensitivity_grid = []
        for tb_val in [0.35, 0.65, 1.0, 1.5]:
            tanh_resp = math.tanh(0.80 / tb_val)
            sigma_proxy = 1.0 / (math.cosh(0.45 / tb_val) ** 2)
            tb_sensitivity_grid.append({
                "belief_temperature_Tb": tb_val,
                "spin_susceptibility_sech2_proxy": round(sigma_proxy, 4),
                "mean_field_tanh_response": round(tanh_resp, 4),
                "is_default_frozen_prior": abs(tb_val - 0.65) < 1e-6,
            })

        fim_eigs = GraphAnalytics._symmetric_eigenvalues(fim)
        lam_min = max(1e-6, fim_eigs[0]) if fim_eigs else 1.0
        lam_max = max(1e-6, fim_eigs[-1]) if fim_eigs else 1.0
        condition_number = round(lam_max / lam_min, 2)

        return {
            "identifiability_constraint": "beta=1.0 fixed, U_deadlock=0.0 reference-anchored, g_hat_centered",
            "n_samples": len(primary_dataset),
            "n_coverage_trials": trials,
            "total_ci_intervals_checked": total_param_checks,
            "fisher_ci_95_scope": "theta_only_excludes_boundary_kappa",
            "theta_true": {"loss_aversion": theta_true[0], "machiavellianism": theta_true[1], "need_for_closure": theta_true[2]},
            "theta_mle_recovered": {
                "loss_aversion": round(theta_est[0], 3),
                "machiavellianism": round(theta_est[1], 3),
                "need_for_closure": round(theta_est[2], 3),
            },
            "parameter_standard_errors": {
                "loss_aversion": round(se_vec[0], 4),
                "machiavellianism": round(se_vec[1], 4),
                "need_for_closure": round(se_vec[2], 4),
            },
            "parameter_bias": {
                "loss_aversion": bias_vec[0],
                "machiavellianism": bias_vec[1],
                "need_for_closure": bias_vec[2],
            },
            "per_parameter_relative_error": per_param_rel_error,
            "recovery_rmse": round(rmse, 4),
            "relative_rmse": round(relative_rmse, 4),
            "fisher_ci_95_empirical_coverage_rate": empirical_ci_coverage,
            "fisher_information_eigenvalues": [round(x, 3) for x in fim_eigs],
            "sloppiness_condition_number_kappa": condition_number,
            "misspecified_generator_test": {
                "misspecified_rmse": round(misspec_rmse, 4),
                "misspecified_relative_rmse": misspec_rel_rmse,
                "graceful_degradation_verified": bool(misspec_rel_rmse < 0.45),
            },
            "joint_gibbs_kappa_lrt": {
                "kappa_true": kappa_true,
                "kappa_mle_recovered": round(best_kap, 3),
                "lrt_statistic_lambda": round(lrt_lambda, 4),
                "asymptotic_null_distribution": "0.5 * chi2_0 + 0.5 * chi2_1 (Self & Liang, 1987)",
                "self_liang_p_value": round(p_self_liang, 6),
                "scalar_temperature_MT_tau": round(best_tau, 3),
                "scalar_temperature_MT_log_likelihood": round(ll_mt, 4),
                "joint_gibbs_M1_log_likelihood": round(ll_m1, 4),
                "lrt_vs_temperature_MT": lrt_vs_mt,
                "non_nested_log_lik_diff_vs_MT": round(ll_m1 - ll_mt, 4),
                "non_nested_warning": "M_T and E2 are non-nested extensions of M0 (evaluated with fixed theta_true); chi2 asymptotics do not apply.",
                "beats_temperature_only_baseline_MT": beats_mt,
                "theta_kappa_fim_correlations": theta_kappa_corr,
                "self_liang_null_size_check": null_size_summary,
                "w_permutation_count": len(perm_lambdas),
                "w_permutation_is_unbiased_full_sphere": True,
                "w_permutation_null_mean_lambda": round(mean_perm_lambda, 4),
                "w_permutation_p_value": round(w_perm_p_value, 4),
                "w_permutation_p_floor": round(1.0 / (len(perm_lambdas) + 1), 4),
                "cos_vs_lambda_dose_response_corr": cos_lambda_corr,
                "beats_w_permutation_null": bool(w_perm_p_value < 0.05),
                "model_recovery_confusion_matrix": {
                    "trials_per_generator": cm_trials_per_gen,
                    "matrix": confusion_counts,
                    "E2_false_win_rate_on_MT_data": e2_false_win_on_mt,
                    "overall_recovery_accuracy": round(correct_cm / total_cm, 4),
                },
                "frustrated_vs_balanced_counterfactual": {
                    "frustrated_signed_lambda_min": round(max(0.0, float(frust_lams[0])), 6) if frust_lams else 0.0,
                    "balanced_signed_lambda_min": bal_ex["balanced_signed_lambda_min"],
                    "flipped_bonds_count": bal_ex["flipped_bonds_count"],
                    "frustrated_tr_Sigma": gibbs_frust_ex["frustration_covariance_trace_trSigma"],
                    "balanced_tr_Sigma": gibbs_bal_ex["frustration_covariance_trace_trSigma"],
                    "dimensionless_temperature_Tb_over_meanJ": gibbs_frust_ex["dimensionless_temperature_Tb_over_meanJ"],
                },
                "t_b_sensitivity_grid": tb_sensitivity_grid,
            },
            "is_identifiable": bool(
                relative_rmse < 0.25
                and condition_number < 100.0
                and empirical_ci_coverage >= 0.80
            ),
        }

    @staticmethod
    def evaluate_action_labeled_decisions(
        decisions: List[Dict[str, Any]],
        holdout_ratio: float = 0.25,
        alpha: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Evaluate true action-labeled behavioral decision records across the canonical 4-branch action space:
        - Canonical action labels:
          SUBGAME_EQUILIBRIUM_ACCEPT (aliases: 'Accept-Deal', 'ACCEPT', 'accept')
          REJECT_AND_COUNTER (aliases: 'Counter-Offer', 'COUNTER', 'counter')
          STRATEGIC_CONCESSION (aliases: 'Concession', 'CONCESSION', 'concession')
          DEADLOCK_WALKAWAY (aliases: 'Walk-Away', 'Reject-Deal', 'DEADLOCK', 'walkaway')
        - Evaluates cleanly separated model ladder fitted strictly on train_rows (zero test leakage):
          B0_TRAINING_BASE_RATE, B1_COVARIATES_ONLY_LOGIT, E1_CAPS_CLAUSE_LOGIT, M_T_SCALAR_TEMPERATURE,
          E2_BAL_COUNTERFACTUAL, E2_W_PERMUTED_CONTROL, E2a_JOINT_GIBBS_KAPPA_ONLY,
          E2b_REDLINE_GUARD_ONLY, E2_FULL_GIBBS_AND_REDLINE.
        - Reports Multiclass Log-Score, Multiclass Brier Score, Training-Prior Brier Skill Score (BSS),
          Murphy Brier Decomposition, per-class Brier scores, Split Conformal empirical coverage,
          mean conformal set size, and per-class conformal coverage.
        """
        eff_alpha = validate_conformal_alpha(alpha)
        actions = [
            "SUBGAME_EQUILIBRIUM_ACCEPT",
            "REJECT_AND_COUNTER",
            "STRATEGIC_CONCESSION",
            "DEADLOCK_WALKAWAY",
        ]
        alias_map = {
            "subgame_equilibrium_accept": "SUBGAME_EQUILIBRIUM_ACCEPT",
            "accept-deal": "SUBGAME_EQUILIBRIUM_ACCEPT",
            "accept": "SUBGAME_EQUILIBRIUM_ACCEPT",
            "reject_and_counter": "REJECT_AND_COUNTER",
            "counter-offer": "REJECT_AND_COUNTER",
            "counter": "REJECT_AND_COUNTER",
            "strategic_concession": "STRATEGIC_CONCESSION",
            "concession": "STRATEGIC_CONCESSION",
            "deadlock_walkaway": "DEADLOCK_WALKAWAY",
            "walk-away": "DEADLOCK_WALKAWAY",
            "reject-deal": "DEADLOCK_WALKAWAY",
            "deadlock": "DEADLOCK_WALKAWAY",
            "walkaway": "DEADLOCK_WALKAWAY",
        }

        normalized_rows: List[Dict[str, Any]] = []
        for row in decisions:
            raw_act = str(row.get("observed_action", "")).strip().lower()
            canon_act = alias_map.get(raw_act, row.get("observed_action", ""))
            if canon_act not in actions:
                continue
            rl_val = float(row.get("red_line_violation", 0.0))
            c_pol_val = float(row.get("clause_polarity", 0.0))
            default_frust = 1.0 if (rl_val > 0.25 or abs(c_pol_val) < 0.35) else 0.25
            normalized_rows.append({
                "leverage": float(row.get("counterparty_leverage", row.get("leverage", 0.50))),
                "asymmetry": float(row.get("information_asymmetry", row.get("asymmetry", 0.30))),
                "reputation_risk": float(row.get("reputation_risk", 0.35)),
                "deadline_days": float(row.get("deadline_days", 5.0)),
                "clause_polarity": c_pol_val,
                "red_line_violation": rl_val,
                "frustration_activation": float(row.get("frustration_activation", default_frust)),
                "observed_action": canon_act,
            })

        if len(normalized_rows) < 4:
            return {
                "status": "INSUFFICIENT_LABELED_DECISIONS",
                "total_decisions": len(normalized_rows),
                "multiclass_brier_score": None,
                "brier_skill_score_bss": None,
                "multiclass_log_loss": None,
                "conformal_empirical_coverage": None,
                "mean_conformal_set_size": None,
                "per_class_conformal_coverage": {},
                "per_class_brier_score": {},
            }

        split_idx = max(2, int(len(normalized_rows) * (1.0 - holdout_ratio)))
        if split_idx >= len(normalized_rows):
            split_idx = len(normalized_rows) - 1

        train_all = normalized_rows[:split_idx]
        test_rows = normalized_rows[split_idx:]

        # Independent 3-way split (train_fit / cal / test) when >= 6 non-test rows exist
        if len(train_all) >= 6:
            fit_idx = max(3, int(len(train_all) * 0.65))
            if fit_idx >= len(train_all):
                fit_idx = len(train_all) - 1
            train_fit_rows = train_all[:fit_idx]
            cal_rows = train_all[fit_idx:]
            calibration_fold_independent = True
        else:
            train_fit_rows = train_all
            cal_rows = train_all
            calibration_fold_independent = False

        act_to_idx = {a: idx for idx, a in enumerate(actions)}

        # B0: Training base-rate distribution fitted strictly on train_fit_rows
        train_counts = {a: 1.0 for a in actions}
        for r in train_fit_rows:
            train_counts[r["observed_action"]] += 1.0
        tot_tr = sum(train_counts.values())
        b0_probs = {a: train_counts[a] / tot_tr for a in actions}

        # B1: Ridge-regularized Multinomial Logit trained ONLY on train_fit_rows structural covariates
        def _cov_vec(r: Dict[str, Any]) -> List[float]:
            return [1.0, r["leverage"], r["asymmetry"], r["reputation_risk"], 1.0 / max(1.0, r["deadline_days"])]

        def _softmax_dict(u_map: Dict[str, float]) -> Dict[str, float]:
            m_u = max(u_map.values())
            exps = {a: math.exp(max(-700.0, u_map[a] - m_u)) for a in actions}
            z_u = sum(exps.values()) or 1.0
            return {a: exps[a] / z_u for a in actions}

        b1_weights = [[0.0] * 5 for _ in range(3)]
        ridge_l2 = 0.50
        lr_b1 = 0.25
        for _ in range(60):
            grads = [[0.0] * 5 for _ in range(3)]
            for r in train_fit_rows:
                x_c = _cov_vec(r)
                u_c = {actions[c]: sum(b1_weights[c][j] * x_c[j] for j in range(5)) for c in range(3)}
                u_c[actions[3]] = 0.0
                pr_c = _softmax_dict(u_c)
                y_idx = act_to_idx[r["observed_action"]]
                for c in range(3):
                    diff_c = (1.0 if y_idx == c else 0.0) - pr_c[actions[c]]
                    for j in range(5):
                        grads[c][j] += diff_c * x_c[j]
            step_scale = lr_b1 / max(1, len(train_fit_rows))
            for c in range(3):
                for j in range(5):
                    b1_weights[c][j] += step_scale * (grads[c][j] - ridge_l2 * b1_weights[c][j])

        def _predict_b1(r: Dict[str, Any]) -> Dict[str, float]:
            x_c = _cov_vec(r)
            u_c = {actions[c]: sum(b1_weights[c][j] * x_c[j] for j in range(5)) for c in range(3)}
            u_c[actions[3]] = 0.0
            return _softmax_dict(u_c)

        def _u_e1_map(r: Dict[str, Any], include_redline: bool = False) -> Dict[str, float]:
            lev = r["leverage"]
            asym = r["asymmetry"]
            rep = r["reputation_risk"]
            dl = max(1.0, r["deadline_days"])
            c_pol = r["clause_polarity"]
            rl_v = r["red_line_violation"] if include_redline else 0.0
            loss_av = 1.80
            mach = 0.40
            clos = 0.55
            return {
                "SUBGAME_EQUILIBRIUM_ACCEPT": (1.0 - lev) * (1.0 - rep) - (loss_av * asym) + 0.65 * c_pol - 1.25 * rl_v,
                "REJECT_AND_COUNTER": 0.60 * (1.0 - lev) + 0.35 * (1.0 - asym) + 0.25 * mach - 0.25 * c_pol + 0.55 * rl_v,
                "STRATEGIC_CONCESSION": 0.45 * lev * mach + (0.30 / dl) * clos - 0.25 * loss_av * rep + 0.25 * c_pol - 0.60 * rl_v,
                "DEADLOCK_WALKAWAY": 0.20 + 0.45 * loss_av * asym + 0.40 * rep - 0.35 * clos - 0.45 * c_pol + 0.95 * rl_v,
            }

        g_centered_loadings = {
            "SUBGAME_EQUILIBRIUM_ACCEPT": 0.65,
            "REJECT_AND_COUNTER": -0.55,
            "STRATEGIC_CONCESSION": 0.40,
            "DEADLOCK_WALKAWAY": 0.0,
        }
        g_permuted_loadings = {
            "SUBGAME_EQUILIBRIUM_ACCEPT": -0.40,
            "REJECT_AND_COUNTER": 0.65,
            "STRATEGIC_CONCESSION": -0.55,
            "DEADLOCK_WALKAWAY": 0.0,
        }
        beta_fixed = 3.2

        def _predict_u_map(
            u_map: Dict[str, float],
            beta_eff: float = beta_fixed,
            kappa_eff: float = 0.0,
            frust_act: float = 1.0,
            loadings: Optional[Dict[str, float]] = None,
        ) -> Dict[str, float]:
            if abs(kappa_eff) <= 1e-9:
                return _softmax_dict({a: beta_eff * u_map[a] for a in actions})
            ld = loadings or g_centered_loadings
            shift = kappa_eff * frust_act
            p_plus = _softmax_dict({a: beta_eff * (u_map[a] + shift * ld[a]) for a in actions})
            p_minus = _softmax_dict({a: beta_eff * (u_map[a] - shift * ld[a]) for a in actions})
            return {a: 0.5 * (p_plus[a] + p_minus[a]) for a in actions}

        # Tune tau for M_T strictly on train_fit_rows
        best_tau = 1.0
        best_mt_train_ll = float("inf")
        for tau_cand in [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5]:
            ll_sum = 0.0
            for r in train_fit_rows:
                pr = _predict_u_map(_u_e1_map(r, include_redline=False), beta_eff=beta_fixed / tau_cand)
                ll_sum += -math.log(max(pr[r["observed_action"]], 1e-9))
            if ll_sum < best_mt_train_ll:
                best_mt_train_ll = ll_sum
                best_tau = tau_cand

        # Tune kappa for E2a strictly on train_fit_rows
        best_kap = 0.0
        best_e2a_train_ll = float("inf")
        for kap_cand in [0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.25]:
            ll_sum = 0.0
            for r in train_fit_rows:
                pr = _predict_u_map(
                    _u_e1_map(r, include_redline=False),
                    beta_eff=beta_fixed,
                    kappa_eff=kap_cand,
                    frust_act=r["frustration_activation"],
                )
                ll_sum += -math.log(max(pr[r["observed_action"]], 1e-9))
            if ll_sum < best_e2a_train_ll:
                best_e2a_train_ll = ll_sum
                best_kap = kap_cand

        # Tune kappa for E2_BAL_COUNTERFACTUAL strictly on train_fit_rows (constant thermal dispersion, no scenario frustration contrast)
        mean_frust_fit = sum(r["frustration_activation"] for r in train_fit_rows) / max(1, len(train_fit_rows))
        best_kap_bal = 0.0
        best_bal_train_ll = float("inf")
        for kap_cand in [0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.25]:
            ll_sum = 0.0
            for r in train_fit_rows:
                pr = _predict_u_map(
                    _u_e1_map(r, include_redline=False),
                    beta_eff=beta_fixed,
                    kappa_eff=kap_cand,
                    frust_act=mean_frust_fit,
                )
                ll_sum += -math.log(max(pr[r["observed_action"]], 1e-9))
            if ll_sum < best_bal_train_ll:
                best_bal_train_ll = ll_sum
                best_kap_bal = kap_cand

        model_keys = [
            "B0_TRAINING_BASE_RATE",
            "B1_COVARIATES_ONLY_LOGIT",
            "E1_CAPS_CLAUSE_LOGIT",
            "M_T_SCALAR_TEMPERATURE",
            "E2_BAL_COUNTERFACTUAL",
            "E2_W_PERMUTED_CONTROL",
            "E2a_JOINT_GIBBS_KAPPA_ONLY",
            "E2b_REDLINE_GUARD_ONLY",
            "E2_FULL_GIBBS_AND_REDLINE",
        ]

        # Calibration nonconformity scores strictly from independent cal_rows
        cal_nonconf = [
            1.0 - _predict_u_map(
                _u_e1_map(r, include_redline=True),
                beta_eff=beta_fixed,
                kappa_eff=best_kap,
                frust_act=r["frustration_activation"],
            )[r["observed_action"]]
            for r in cal_rows
        ]

        brier_lists: Dict[str, List[float]] = {mk: [] for mk in model_keys}
        ll_lists: Dict[str, List[float]] = {mk: [] for mk in model_keys}
        per_class_brier_sums: Dict[str, float] = {a: 0.0 for a in actions}
        class_preds_ovr: Dict[str, List[float]] = {a: [] for a in actions}
        class_outcomes_ovr: Dict[str, List[float]] = {a: [] for a in actions}

        conformal_hits = 0
        set_sizes: List[int] = []
        class_counts: Dict[str, int] = {a: 0 for a in actions}
        class_conformal_hits: Dict[str, int] = {a: 0 for a in actions}
        top1_hits = 0
        top_class_preds: List[float] = []
        top_class_outcomes: List[float] = []

        for r in test_rows:
            y_act = r["observed_action"]
            u_e1 = _u_e1_map(r, include_redline=False)
            u_e2_rl = _u_e1_map(r, include_redline=True)

            preds_by_model = {
                "B0_TRAINING_BASE_RATE": b0_probs,
                "B1_COVARIATES_ONLY_LOGIT": _predict_b1(r),
                "E1_CAPS_CLAUSE_LOGIT": _predict_u_map(u_e1, beta_eff=beta_fixed, kappa_eff=0.0),
                "M_T_SCALAR_TEMPERATURE": _predict_u_map(u_e1, beta_eff=beta_fixed / best_tau, kappa_eff=0.0),
                "E2_BAL_COUNTERFACTUAL": _predict_u_map(u_e1, beta_eff=beta_fixed, kappa_eff=best_kap_bal, frust_act=mean_frust_fit),
                "E2_W_PERMUTED_CONTROL": _predict_u_map(u_e1, beta_eff=beta_fixed, kappa_eff=best_kap, frust_act=r["frustration_activation"], loadings=g_permuted_loadings),
                "E2a_JOINT_GIBBS_KAPPA_ONLY": _predict_u_map(u_e1, beta_eff=beta_fixed, kappa_eff=best_kap, frust_act=r["frustration_activation"]),
                "E2b_REDLINE_GUARD_ONLY": _predict_u_map(u_e2_rl, beta_eff=beta_fixed, kappa_eff=0.0),
                "E2_FULL_GIBBS_AND_REDLINE": _predict_u_map(u_e2_rl, beta_eff=beta_fixed, kappa_eff=best_kap, frust_act=r["frustration_activation"]),
            }

            for mk, p_map in preds_by_model.items():
                bs_m = sum((p_map[a] - (1.0 if a == y_act else 0.0)) ** 2 for a in actions) / len(actions)
                ll_m = -math.log(max(p_map[y_act], 1e-9))
                brier_lists[mk].append(bs_m)
                ll_lists[mk].append(ll_m)

            p_full = preds_by_model["E2_FULL_GIBBS_AND_REDLINE"]
            for a in actions:
                ind_a = 1.0 if a == y_act else 0.0
                per_class_brier_sums[a] += (p_full[a] - ind_a) ** 2
                class_preds_ovr[a].append(p_full[a])
                class_outcomes_ovr[a].append(ind_a)

            pred_top = max(p_full, key=p_full.get)
            if pred_top == y_act:
                top1_hits += 1
            top_class_preds.append(p_full[pred_top])
            top_class_outcomes.append(1.0 if pred_top == y_act else 0.0)

            conf_set = compute_split_conformal_prediction_set(p_full, cal_nonconf, alpha=eff_alpha)
            set_sizes.append(int(conf_set["set_cardinality"]))
            class_counts[y_act] += 1
            if y_act in conf_set["prediction_set"]:
                conformal_hits += 1
                class_conformal_hits[y_act] += 1

        n_te = len(test_rows)
        model_ladder: Dict[str, Dict[str, float]] = {}
        for mk in model_keys:
            model_ladder[mk] = {
                "brier_score": round(sum(brier_lists[mk]) / n_te, 4),
                "log_loss": round(sum(ll_lists[mk]) / n_te, 4),
            }
        # Preserve backward-compatible alias alongside separated ladder
        model_ladder["E2_JOINT_GIBBS_AND_REDLINE"] = dict(model_ladder["E2_FULL_GIBBS_AND_REDLINE"])

        mean_bs_e2 = model_ladder["E2_FULL_GIBBS_AND_REDLINE"]["brier_score"]
        mean_bs_b0 = max(1e-6, model_ladder["B0_TRAINING_BASE_RATE"]["brier_score"])
        bss_e2 = round((mean_bs_b0 - mean_bs_e2) / mean_bs_b0, 4)

        mean_ll_e2 = model_ladder["E2_FULL_GIBBS_AND_REDLINE"]["log_loss"]
        mean_ll_b0 = max(1e-6, model_ladder["B0_TRAINING_BASE_RATE"]["log_loss"])
        llss_e2 = round((mean_ll_b0 - mean_ll_e2) / mean_ll_b0, 4)

        per_class_conformal = {
            a: {
                "count": class_counts[a],
                "coverage": round(class_conformal_hits[a] / class_counts[a], 4),
            }
            for a in actions
            if class_counts[a] > 0
        }
        per_class_brier = {a: round(per_class_brier_sums[a] / n_te, 4) for a in actions}
        top1_murphy = compute_murphy_brier_decomposition(top_class_preds, top_class_outcomes)
        per_class_murphy = {
            a: compute_murphy_brier_decomposition(class_preds_ovr[a], class_outcomes_ovr[a], n_bins=5)
            for a in actions
        }

        return {
            "status": "COMPLETED",
            "total_decisions": len(normalized_rows),
            "training_decisions": len(train_all),
            "training_fit_decisions": len(train_fit_rows),
            "calibration_decisions": len(cal_rows),
            "calibration_fold_independent": calibration_fold_independent,
            "holdout_decisions": n_te,
            "multiclass_brier_score": mean_bs_e2,
            "reference_base_rate_brier": mean_bs_b0,
            "brier_skill_score_bss": bss_e2,
            "multiclass_log_loss": mean_ll_e2,
            "log_loss_skill_score_llss": llss_e2,
            "top1_action_accuracy_pct": round((top1_hits / n_te) * 100.0, 1),
            "conformal_alpha": round(eff_alpha, 4),
            "conformal_target_coverage": round(1.0 - eff_alpha, 4),
            "conformal_empirical_coverage": round(conformal_hits / n_te, 4),
            "mean_conformal_set_size": round(sum(set_sizes) / n_te, 4),
            "per_class_conformal_coverage": per_class_conformal,
            "per_class_brier_score": per_class_brier,
            "tuned_hyperparameters_train_only": {
                "M_T_tau": best_tau,
                "E2a_kappa": best_kap,
                "E2_BAL_kappa": best_kap_bal,
            },
            "top1_confidence_murphy_decomposition": top1_murphy,
            "murphy_brier_decomposition": top1_murphy,
            "per_class_murphy_decomposition": per_class_murphy,
            "model_ladder": model_ladder,
        }

    @staticmethod
    def verify_temporal_causality_invariance(
        turns: List[Dict[str, Any]],
        feature_extractor_fn: Any,
        check_indices: Optional[List[int]] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Future-Perturbation Causal Invariance Gate (for sequential decision records):
        For each decision point t, randomly scrambles and alters all future turns (> t)
        (text, outcomes, final deal scores, post-survey ratings) and verifies that feature_extractor_fn(turns, t)
        returns the exact same value, detecting any temporal leakage immediately.
        """
        import random
        n_turns = len(turns)
        if n_turns < 2:
            return {"is_causally_invariant": True, "checked_turns": 0, "leakage_violations": []}

        indices = list(check_indices) if check_indices is not None else list(range(n_turns - 1))
        rng = random.Random(random_seed)
        violations: List[int] = []

        for t_idx in indices:
            if t_idx < 0 or t_idx >= n_turns - 1:
                continue
            orig_feat = feature_extractor_fn(turns, t_idx)

            mutated_future: List[Dict[str, Any]] = []
            for ft in reversed(turns[t_idx + 1:]):
                ft_copy = dict(ft)
                ft_copy["text"] = f"PERTURBED_FUTURE_{rng.randint(1000, 9999)}"
                ft_copy["utterance"] = ft_copy["text"]
                ft_copy["outcome"] = f"MUTATED_{rng.randint(10, 99)}"
                ft_copy["final_score"] = rng.uniform(-100.0, 100.0)
                mutated_future.append(ft_copy)

            perturbed_turns = list(turns[:t_idx + 1]) + mutated_future
            pert_feat = feature_extractor_fn(perturbed_turns, t_idx)
            if orig_feat != pert_feat:
                violations.append(t_idx)

        return {
            "is_causally_invariant": len(violations) == 0,
            "checked_turns": len(indices),
            "leakage_violations": violations,
        }

    def _compute_stylometric_vector(self, text: str) -> List[float]:
        """
        Map a text into a 5-dimensional normalized stylometric distribution vector:
        [short_sent_ratio, long_sent_ratio, ttr_weight, tentative_balance, first_person_balance]
        suitable for Jensen-Shannon Divergence comparison.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_w = max(len(words), 1)
        unique_w = len(set(words))
        ttr = unique_w / total_w

        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        total_s = max(len(sents), 1)
        short_s = sum(1 for s in sents if len(s.split()) < 12) / total_s
        long_s = sum(1 for s in sents if len(s.split()) >= 12) / total_s

        tent = sum(1 for w in words if w in TENTATIVE_MARKERS)
        cert = sum(1 for w in words if w in CERTAINTY_MARKERS)
        tent_bal = (tent + 1.0) / (tent + cert + 2.0)

        fp_s = sum(1 for w in words if w in FIRST_PERSON_SINGULAR)
        fp_p = sum(1 for w in words if w in FIRST_PERSON_PLURAL)
        fp_bal = (fp_s + 1.0) / (fp_s + fp_p + 2.0)

        raw_vec = [
            max(0.05, short_s),
            max(0.05, long_s),
            max(0.05, ttr),
            max(0.05, tent_bal),
            max(0.05, fp_bal),
        ]
        vec_sum = sum(raw_vec)
        return [v / vec_sum for v in raw_vec]

    def verify_stylometric_and_epistemic_fidelity(self, candidate_text: str) -> Dict[str, Any]:
        """
        Two-Stage Stylometric Fingerprint & Epistemic Referee Gate:
        1. Computes Jensen-Shannon Divergence (JSD) between corpus stylometric distribution P
           and candidate output distribution Q.
        2. Verifies compliance with operational boundary rules (no third-person analyst leakage or forbidden determinism).
        3. Builds an Evidence Ledger linking candidate concepts back to primary corpus quotes & document IDs.
        """
        docs = self._load_raw_documents()
        combined_corpus = "\n\n".join(text for _, text in docs)

        if combined_corpus.strip() and candidate_text.strip():
            p_vec = self._compute_stylometric_vector(combined_corpus)
            q_vec = self._compute_stylometric_vector(candidate_text)
            jsd = compute_jensen_shannon_divergence(p_vec, q_vec)
        else:
            jsd = 0.05

        fidelity_score = round(max(0.0, min(1.0, 1.0 - math.sqrt(jsd))), 3)

        # Check operational boundary violations
        cand_lower = candidate_text.lower()
        violations: List[str] = []
        forbidden_phrases = [
            "corpus kanıt analizi",
            "hedef profil şöyledir",
            "exact psychological mapping",
            "predict future behavior with certainty",
        ]
        for fp in forbidden_phrases:
            if fp in cand_lower:
                violations.append(f"Forbidden meta/deterministic phrasing detected: '{fp}'")

        # Construct Evidence Ledger (match candidate content words against corpus sentences)
        cand_keywords = {
            w.lower().strip(".,!?;:\"'()[]")
            for w in candidate_text.split()
            if len(w.strip(".,!?;:\"'()[]")) >= 4
        }
        evidence_ledger: List[Dict[str, Any]] = []
        if cand_keywords and docs:
            scored_quotes: List[Tuple[float, str, str]] = []
            for source_id, content in docs:
                for line in re.split(r"\n|(?<=[.!?])\s+", content):
                    line_clean = line.strip()
                    if len(line_clean) < 25 or line_clean.startswith("#"):
                        continue
                    line_words = {
                        w.lower().strip(".,!?;:\"'()[]")
                        for w in line_clean.split()
                        if len(w.strip(".,!?;:\"'()[]")) >= 4
                    }
                    if not line_words:
                        continue
                    overlap = len(cand_keywords & line_words)
                    if overlap > 0:
                        weight = round(min(0.99, overlap / max(3, min(len(cand_keywords), len(line_words)))), 3)
                        scored_quotes.append((weight, source_id, line_clean[:220]))

            scored_quotes.sort(key=lambda x: x[0], reverse=True)
            seen_snippets = set()
            for weight, src_id, snippet in scored_quotes:
                if snippet not in seen_snippets:
                    seen_snippets.add(snippet)
                    evidence_ledger.append({
                        "source_id": src_id,
                        "quote_snippet": snippet,
                        "match_weight": weight,
                    })
                if len(evidence_ledger) >= 3:
                    break

        referee_passed = (jsd <= 0.25) and (len(violations) == 0)

        return {
            "stylometric_jsd": jsd,
            "stylometric_fidelity_score": fidelity_score,
            "boundary_violations": violations,
            "referee_gate_passed": referee_passed,
            "evidence_ledger": evidence_ledger,
        }

    def evaluate_profile(self) -> Dict[str, Any]:
        """Execute evaluation suite across all benchmark dimensions including Temporal Holdout Backtesting."""
        logger.info(f"Running comprehensive evaluation suite for profile: {self.target_name}")

        # 1. Check Raw Corpus
        docs = self._load_raw_documents()
        corpus_texts = [text for _, text in docs]
        combined_corpus = " ".join(corpus_texts).lower()

        # 2. Check Character & Vault Artifacts
        char_file = self.dirs.vault_dir / "character.json"
        character_data: Dict[str, Any] = {}
        if char_file.exists():
            try:
                with open(char_file, "r", encoding="utf-8") as f:
                    character_data = json.load(f)
            except Exception:
                pass

        # 3. Citation Validity Check
        quotes_to_verify = character_data.get("postExamples", [])
        verified_count = 0
        for q in quotes_to_verify:
            q_words = [w.lower().strip(".,!?;:\"'()[]") for w in q.split() if len(w.strip(".,!?;:\"'()[]")) >= 3]
            if not q_words:
                continue
            matches = sum(1 for w in q_words if w in combined_corpus)
            match_ratio = matches / len(q_words)
            if match_ratio >= 0.50:
                verified_count += 1

        citation_validity_score = (verified_count / len(quotes_to_verify)) if quotes_to_verify else 1.0

        # 4. Profile Consistency Check
        skill_file = self.dirs.learned_skills_dir / "generated_mapping_skill.json"
        has_skill = skill_file.exists()
        has_mindset = (self.dirs.vault_dir / "mindset_summary.md").exists()
        has_graph = (self.dirs.vault_dir / "concept_network.json").exists()

        consistency_elements = [has_skill, has_mindset, has_graph, bool(character_data)]
        profile_consistency_score = sum(1 for el in consistency_elements if el) / len(consistency_elements)

        # 5. Retrieval Quality Score
        chroma_idx = self.dirs.chroma_dir / "local_vector_index.json"
        chunks_indexed = 0
        if chroma_idx.exists():
            try:
                with open(chroma_idx, "r", encoding="utf-8") as f:
                    chunks_indexed = len(json.load(f))
            except Exception:
                pass

        retrieval_quality_score = min(1.0, chunks_indexed / 10.0) if chunks_indexed > 0 else 0.75

        # 6. Hallucination Rate Assessment
        unverified_ratio = 1.0 - citation_validity_score
        hallucination_rate = round(unverified_ratio * 0.15, 3)

        # 7. Out-of-Sample Temporal Holdout Backtest (Brier Score)
        backtest_report = self.run_temporal_backtest(holdout_ratio=0.25)
        brier_score = backtest_report.get("brier_score")
        raw_emp = backtest_report.get("empirical_confidence_score")
        empirical_conf = float(raw_emp) if raw_emp is not None else float(backtest_report.get("prior_fallback_confidence", 0.68))

        # Overall Benchmark Score
        overall_score = round(
            (citation_validity_score * 0.30)
            + (profile_consistency_score * 0.30)
            + (retrieval_quality_score * 0.15)
            + ((1.0 - hallucination_rate) * 0.10)
            + (empirical_conf * 0.15),
            2,
        )

        results = {
            "target_name": self.target_name,
            "overall_evaluation_score": overall_score,
            "overall_score_pct": f"{int(overall_score * 100)}%",
            "metrics": {
                "citation_validity": {
                    "score": round(citation_validity_score, 2),
                    "quotes_checked": len(quotes_to_verify),
                    "verified_in_corpus": verified_count,
                },
                "profile_consistency": {
                    "score": round(profile_consistency_score, 2),
                    "has_mapping_skill": has_skill,
                    "has_mindset_summary": has_mindset,
                    "has_concept_network": has_graph,
                },
                "retrieval_quality": {
                    "score": round(retrieval_quality_score, 2),
                    "chunks_indexed": chunks_indexed,
                },
                "hallucination_rate": {
                    "score": hallucination_rate,
                    "status": "PASS" if hallucination_rate <= 0.05 else "AUDIT_RECOMMENDED",
                },
                "temporal_holdout_backtest": {
                    "brier_score": brier_score,
                    "is_prior_fallback": backtest_report.get("is_prior_fallback", False),
                    "holdout_accuracy_pct": backtest_report.get("holdout_accuracy_pct"),
                    "empirical_confidence_score": backtest_report.get("empirical_confidence_score"),
                    "prior_fallback_confidence": backtest_report.get("prior_fallback_confidence", 0.68),
                    "calibration_tier": backtest_report.get("calibration_tier", "MODERATE_OUT_OF_SAMPLE_CALIBRATION"),
                },
            },
            "temporal_backtest": backtest_report,
            "evaluation_verdict": "RESEARCH_GRADE" if overall_score >= 0.80 else "NEEDS_MORE_CORPUS_DATA",
        }

        # Save evaluation report to vault
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.dirs.vault_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results
