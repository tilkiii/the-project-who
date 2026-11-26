"""
Scientific Referee & Meta-Analytic AI Evaluator for who??.
Rigorously evaluates academic papers across empirical dimensions:
- Sample Size (N) & Statistical Power
- Effect Size Magnitude (|r|, Cohen's d) & Significance (p-values)
- Venue & Peer-Review Credibility (Top journals vs preprints)
- Consensus & Replicability Alignment vs Outlier Divergence
- Inverse-Variance Fisher Information Pooling (I_k = N_k - 3), 95% Fisher-z Confidence Intervals,
  and Cochran's Q / Higgins' I^2 Heterogeneity metrics.

Filters out low-rigor or non-replicable studies and selects Gold Standard benchmarks
for Bayesian cognitive parameter calibration. Maintains the persistent knowledge bank at
profiles/scientific_knowledge_bank.json.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from who.utils.logger import get_logger

logger = get_logger("who.learn.scientific_referee")


def fisher_z(r: float) -> float:
    """Fisher's r-to-z transformation for meta-analytic correlation pooling."""
    r_clamped = max(-0.995, min(0.995, float(r)))
    return 0.5 * math.log((1.0 + r_clamped) / (1.0 - r_clamped))


def inv_fisher_z(z: float) -> float:
    """Inverse Fisher z transformation (z to r)."""
    return math.tanh(z)


class ScientificReferee:
    """
    Automated Peer-Reviewer and Meta-Analytic Evaluator.
    Evaluates harvested papers on a 0-100 scientific rigor index, computes Inverse-Variance
    Fisher Information pooling, and manages the canonical knowledge repository of empirical benchmarks.
    """

    TIER1_VENUES = {
        "nature", "science", "pnas", "cell", "journal of personality and social psychology",
        "jpsp", "computers in human behavior", "ieee", "acm", "american psychologist",
        "psychological science", "plos one", "frontiers in psychology", "springer",
        "elsevier", "oxford university press", "cambridge university press", "wiley",
        "behaviour & information technology", "information processing & management"
    }

    PREPRINT_VENUES = {
        "arxiv", "biorxiv", "medrxiv", "ssrn", "osf preprints", "research square", "psyarxiv"
    }

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or Path("profiles")
        self.knowledge_bank_file = self.profiles_dir / "scientific_knowledge_bank.json"
        self._ensure_knowledge_bank()

    def _ensure_knowledge_bank(self) -> None:
        """Ensure profiles/scientific_knowledge_bank.json exists."""
        if not self.knowledge_bank_file.exists():
            try:
                self.profiles_dir.mkdir(parents=True, exist_ok=True)
                initial_data = {
                    "schema_version": "1.0",
                    "description": "Enterprise OSINT & Psychometric Scientific Knowledge Bank. Curated peer-reviewed empirical papers, consensus effect sizes, and operational rules.",
                    "created_at": datetime.datetime.now(timezone.utc).isoformat(),
                    "total_studies_indexed": 0,
                    "metrics": {},
                }
                with open(self.knowledge_bank_file, "w", encoding="utf-8") as f:
                    json.dump(initial_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Initialized Scientific Knowledge Bank at {self.knowledge_bank_file}")
            except Exception as e:
                logger.debug(f"Could not initialize knowledge bank: {e}")

    # -------------------------------------------------------------------------
    # 1. Rigor Scoring Framework (0 - 100 Scale)
    # -------------------------------------------------------------------------

    def score_paper_rigor(
        self,
        paper: Dict[str, Any],
        marker: str,
        prior_mean_r: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate paper on 4 rigorous axes:
        1. S_N: Sample Size & Power Score (0-35)
        2. S_r: Effect Size & Significance Score (0-25)
        3. S_v: Venue & Peer-Review Credibility Score (0-20)
        4. S_c: Consensus & Replicability Alignment Score (0-20)
        Also computes asymptotic Fisher Information I_k = max(4, N_k - 3) and variance 1 / I_k.
        """
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        methodology = (paper.get("methodology") or "").lower()
        venue = (paper.get("venue") or paper.get("source") or "").lower()

        n_raw = paper.get("sample_size")
        n = self._extract_or_infer_sample_size(n_raw, abstract)

        r_raw = paper.get("r")
        r = self._extract_or_infer_correlation(r_raw, abstract)

        p_raw = paper.get("p_value")
        p = self._extract_or_infer_p_value(p_raw, abstract)

        is_meta_analysis = "meta-analysis" in methodology or "systematic review" in methodology or "meta-analytic" in abstract.lower()
        if is_meta_analysis:
            s_n = 35.0
        elif n and n > 0:
            s_n = min(35.0, max(5.0, 10.0 * math.log10(max(10, n))))
        else:
            s_n = 12.0

        abs_r = abs(r)
        base_r_score = 20.0 * min(1.0, abs_r / 0.40)
        if p is not None:
            if p < 0.001:
                sig_adj = 5.0
            elif p < 0.01:
                sig_adj = 3.0
            elif p < 0.05:
                sig_adj = 1.0
            else:
                sig_adj = -10.0
        else:
            sig_adj = 2.0

        s_r = max(0.0, min(25.0, base_r_score + sig_adj))

        s_v = 6.0
        for t1 in self.TIER1_VENUES:
            if t1 in venue or t1 in paper.get("source", "").lower():
                s_v = 20.0
                break
        if s_v != 20.0:
            for prep in self.PREPRINT_VENUES:
                if prep in venue or prep in paper.get("source", "").lower():
                    s_v = 10.0
                    break
            else:
                if "journal" in venue or "conference" in venue or "transactions" in venue or paper.get("source") in ("pubmed", "crossref"):
                    s_v = 15.0

        delta_r = abs(r - prior_mean_r)
        if delta_r <= 0.10:
            s_c = 20.0
        elif delta_r <= 0.20:
            s_c = 15.0
        elif delta_r <= 0.35:
            s_c = 10.0
        else:
            s_c = 4.0

        total_rigor = round(s_n + s_r + s_v + s_c, 1)

        if total_rigor >= 70.0:
            rigor_tier = "GOLD_STANDARD"
            tier_desc = "Peer-reviewed, large-sample, replicable gold-standard empirical study."
        elif total_rigor >= 50.0:
            rigor_tier = "SOLID_EMPIRICAL"
            tier_desc = "Reliable empirical methodology and statistical significance."
        elif total_rigor >= 35.0:
            tier_desc = "Exploratory or preprint stage; interpret with caution."
            rigor_tier = "EXPLORATORY_PREPRINT"
        else:
            rigor_tier = "REJECTED_LOW_RIGOR"
            tier_desc = "Filtered out due to small sample size, weak effect size, or methodological inconsistency."

        sample_size_imputed = not (n is not None and int(n) >= 10)
        if not sample_size_imputed:
            eff_n: Optional[int] = int(n)
            fisher_info_i = float(max(4, eff_n - 3))
            asymptotic_var_z: Optional[float] = round(1.0 / fisher_info_i, 6)
        else:
            # Never fabricate observed Fisher information when real N is missing or < 10
            eff_n = None
            fisher_info_i = 0.0
            asymptotic_var_z = None

        return {
            "title": title,
            "abstract": abstract,
            "source": paper.get("source", "unknown"),
            "url": paper.get("url", ""),
            "doi": paper.get("doi", ""),
            "published": paper.get("published", ""),
            "inferred_n": n,
            "effective_n": eff_n,
            "sample_size_imputed": sample_size_imputed,
            "inferred_r": round(r, 3),
            "inferred_p": p,
            "is_meta_analysis": is_meta_analysis,
            "fisher_information_I": fisher_info_i,
            "asymptotic_variance_z": asymptotic_var_z,
            "scores": {
                "sample_size_score": round(s_n, 1),
                "effect_size_score": round(s_r, 1),
                "venue_score": round(s_v, 1),
                "consensus_score": round(s_c, 1),
                "total_rigor_score": total_rigor,
            },
            "rigor_tier": rigor_tier,
            "tier_description": tier_desc,
            "confidence_weight": round(total_rigor / 100.0, 2),
        }

    # -------------------------------------------------------------------------
    # 2. Multi-Paper Evaluation & Inverse-Variance Fisher Information Pooling
    # -------------------------------------------------------------------------

    def evaluate_and_rank_corpus(
        self,
        papers: List[Dict[str, Any]],
        marker: str,
        prior_mean_r: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate a full candidate corpus for a specific behavioral marker.
        - Scores every paper and filters out REJECTED_LOW_RIGOR studies.
        - Computes Inverse-Variance Fisher Information pooled effect size consensus_r,
          95% confidence interval [ci_low_r, ci_high_r], Cochran's Q, and Higgins' I^2 heterogeneity.
        """
        if not papers:
            return {
                "marker": marker,
                "total_evaluated": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "consensus_r": prior_mean_r,
                "fisher_meta_analysis": {
                    "pooled_z": round(fisher_z(prior_mean_r), 4),
                    "pooled_se_z": 0.15,
                    "ci_95_r": [round(max(-0.99, prior_mean_r - 0.15), 3), round(min(0.99, prior_mean_r + 0.15), 3)],
                    "total_fisher_information": 0.0,
                    "observed_n_studies": 0,
                    "imputed_n_studies_excluded_from_fisher": 0,
                    "cochran_q": 0.0,
                    "higgins_i2_pct": 0.0,
                },
                "ranked_papers": [],
                "gold_standard_papers": [],
            }

        evaluated: List[Dict[str, Any]] = []
        for p in papers:
            ev = self.score_paper_rigor(p, marker=marker, prior_mean_r=prior_mean_r)
            evaluated.append(ev)

        evaluated.sort(key=lambda x: x["scores"]["total_rigor_score"], reverse=True)

        approved = [p for p in evaluated if p["rigor_tier"] != "REJECTED_LOW_RIGOR"]
        rejected = [p for p in evaluated if p["rigor_tier"] == "REJECTED_LOW_RIGOR"]
        gold_standards = [p for p in approved if p["rigor_tier"] == "GOLD_STANDARD"]

        if approved:
            observed_approved = [
                ap for ap in approved
                if not ap.get("sample_size_imputed", False) and float(ap.get("fisher_information_I", 0.0)) > 0.0
            ]
            imputed_excluded = len(approved) - len(observed_approved)

            if observed_approved:
                z_vals: List[float] = []
                fisher_infos: List[float] = []
                quality_weights: List[float] = []

                for ap in observed_approved:
                    z_k = fisher_z(ap["inferred_r"])
                    i_k = float(ap["fisher_information_I"])
                    w_qual = float(ap["confidence_weight"]) * i_k
                    z_vals.append(z_k)
                    fisher_infos.append(i_k)
                    quality_weights.append(w_qual)

                # Exact Inverse-Variance Fisher Information Pooling over observed-N studies only: w_k = I_k = N_k - 3
                total_fisher_info = sum(fisher_infos) or 1.0
                consensus_z = sum(z * i_k for z, i_k in zip(z_vals, fisher_infos)) / total_fisher_info
                consensus_r = round(inv_fisher_z(consensus_z), 3)

                qual_sum = sum(quality_weights) or 1.0
                quality_weighted_z = sum(z * w for z, w in zip(z_vals, quality_weights)) / qual_sum
                quality_weighted_r = round(inv_fisher_z(quality_weighted_z), 3)

                se_z = 1.0 / math.sqrt(max(4.0, total_fisher_info))
                ci_low_r = round(inv_fisher_z(consensus_z - 1.96 * se_z), 3)
                ci_high_r = round(inv_fisher_z(consensus_z + 1.96 * se_z), 3)

                # Cochran's Q and Higgins' I^2 heterogeneity across observed-N studies (exact under w_k = I_k)
                cochran_q = sum(i_k * ((z_k - consensus_z) ** 2) for i_k, z_k in zip(fisher_infos, z_vals))
                df = max(1, len(observed_approved) - 1)
                higgins_i2 = max(0.0, ((cochran_q - df) / cochran_q) * 100.0) if cochran_q > df else 0.0

                meta_stats = {
                    "pooled_z": round(consensus_z, 4),
                    "pooled_se_z": round(se_z, 4),
                    "ci_95_r": [ci_low_r, ci_high_r],
                    "quality_weighted_r": quality_weighted_r,
                    "total_fisher_information": round(total_fisher_info, 1),
                    "observed_n_studies": len(observed_approved),
                    "imputed_n_studies_excluded_from_fisher": imputed_excluded,
                    "all_sample_sizes_imputed": False,
                    "cochran_q": round(cochran_q, 3),
                    "higgins_i2_pct": round(higgins_i2, 1),
                }
            else:
                z_vals = [fisher_z(ap["inferred_r"]) for ap in approved]
                consensus_z = sum(z_vals) / len(z_vals)
                consensus_r = round(inv_fisher_z(consensus_z), 3)
                se_z = 0.25
                ci_low_r = round(inv_fisher_z(consensus_z - 1.96 * se_z), 3)
                ci_high_r = round(inv_fisher_z(consensus_z + 1.96 * se_z), 3)
                meta_stats = {
                    "pooled_z": round(consensus_z, 4),
                    "pooled_se_z": se_z,
                    "ci_95_r": [ci_low_r, ci_high_r],
                    "quality_weighted_r": consensus_r,
                    "total_fisher_information": 0.0,
                    "observed_n_studies": 0,
                    "imputed_n_studies_excluded_from_fisher": imputed_excluded,
                    "all_sample_sizes_imputed": True,
                    "cochran_q": 0.0,
                    "higgins_i2_pct": 0.0,
                }
        else:
            consensus_r = prior_mean_r
            meta_stats = {
                "pooled_z": round(fisher_z(prior_mean_r), 4),
                "pooled_se_z": 0.15,
                "ci_95_r": [round(max(-0.99, prior_mean_r - 0.15), 3), round(min(0.99, prior_mean_r + 0.15), 3)],
                "total_fisher_information": 0.0,
                "observed_n_studies": 0,
                "imputed_n_studies_excluded_from_fisher": 0,
                "cochran_q": 0.0,
                "higgins_i2_pct": 0.0,
            }

        return {
            "marker": marker,
            "total_evaluated": len(evaluated),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "consensus_r": consensus_r,
            "fisher_meta_analysis": meta_stats,
            "ranked_papers": evaluated,
            "approved_papers": approved,
            "gold_standard_papers": gold_standards,
        }

    # -------------------------------------------------------------------------
    # 3. Persistent Knowledge Bank Maintenance
    # -------------------------------------------------------------------------

    def maintain_knowledge_bank(
        self,
        marker: str,
        evaluation_result: Dict[str, Any],
        synthesized_boundaries: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Incrementally persist gold standards, empirical consensus metrics,
        and synthesized decision boundaries into profiles/scientific_knowledge_bank.json.
        """
        self._ensure_knowledge_bank()
        try:
            with open(self.knowledge_bank_file, "r", encoding="utf-8") as f:
                kb = json.load(f)
        except Exception:
            kb = {"metrics": {}}

        metrics_map = kb.setdefault("metrics", {})
        metric_record = metrics_map.setdefault(marker, {
            "marker_name": marker,
            "first_indexed_at": datetime.datetime.now(timezone.utc).isoformat(),
            "total_evaluations": 0,
            "consensus_r": evaluation_result.get("consensus_r", 0.25),
            "gold_standard_studies": [],
            "decision_boundaries": {},
        })

        metric_record["last_updated_at"] = datetime.datetime.now(timezone.utc).isoformat()
        metric_record["total_evaluations"] += evaluation_result.get("total_evaluated", 0)
        metric_record["consensus_r"] = evaluation_result.get("consensus_r", metric_record.get("consensus_r", 0.25))
        if "fisher_meta_analysis" in evaluation_result:
            metric_record["fisher_meta_analysis"] = evaluation_result["fisher_meta_analysis"]

        existing_urls = {s.get("url") for s in metric_record.get("gold_standard_studies", []) if s.get("url")}
        existing_titles = {s.get("title").strip().lower() for s in metric_record.get("gold_standard_studies", []) if s.get("title")}

        for gs in evaluation_result.get("gold_standard_papers", []):
            url = gs.get("url")
            title_clean = gs.get("title", "").strip().lower()
            if (url and url in existing_urls) or (title_clean and title_clean in existing_titles):
                continue
            study_summary = {
                "title": gs.get("title"),
                "url": url,
                "source": gs.get("source"),
                "published": gs.get("published"),
                "sample_size": gs.get("inferred_n"),
                "effect_size_r": gs.get("inferred_r"),
                "p_value": gs.get("inferred_p"),
                "fisher_information_I": gs.get("fisher_information_I"),
                "rigor_score": gs["scores"]["total_rigor_score"],
                "rigor_tier": gs["rigor_tier"],
                "indexed_at": datetime.datetime.now(timezone.utc).isoformat(),
            }
            metric_record.setdefault("gold_standard_studies", []).append(study_summary)
            if url:
                existing_urls.add(url)
            if title_clean:
                existing_titles.add(title_clean)

        if synthesized_boundaries:
            metric_record["decision_boundaries"] = synthesized_boundaries

        total_studies = sum(len(m.get("gold_standard_studies", [])) for m in metrics_map.values())
        kb["total_studies_indexed"] = total_studies
        kb["last_sync"] = datetime.datetime.now(timezone.utc).isoformat()

        try:
            with open(self.knowledge_bank_file, "w", encoding="utf-8") as f:
                json.dump(kb, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated Scientific Knowledge Bank for marker '{marker}' ({len(metric_record['gold_standard_studies'])} gold studies).")
        except Exception as e:
            logger.warning(f"Failed to persist Scientific Knowledge Bank: {e}")

        return self.knowledge_bank_file

    # -------------------------------------------------------------------------
    # Helper Extractors
    # -------------------------------------------------------------------------

    def _extract_or_infer_sample_size(self, explicit_n: Any, text: str) -> Optional[int]:
        if explicit_n is not None:
            try:
                val = int(explicit_n)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass

        patterns = [
            r"[Nn]\s*=\s*([0-9,]+)",
            r"([0-9,]+)\s*(?:participants|subjects|individuals|tweets|users|posts)",
            r"sample\s*(?:of|size\s*of)\s*([0-9,]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                raw_str = match.group(1).replace(",", "")
                try:
                    num = int(raw_str)
                    if 10 <= num <= 10_000_000:
                        return num
                except ValueError:
                    continue
        return None

    def _extract_or_infer_correlation(self, explicit_r: Any, text: str) -> float:
        if explicit_r is not None:
            try:
                val = float(explicit_r)
                if -1.0 <= val <= 1.0:
                    return val
            except (ValueError, TypeError):
                pass

        match = re.search(r"\br\s*=\s*([+-]?(?:0?\.\d+|[01]))\b", text)
        if match:
            try:
                val = float(match.group(1))
                if -1.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return 0.25

    def _extract_or_infer_p_value(self, explicit_p: Any, text: str) -> Optional[float]:
        if explicit_p is not None:
            try:
                val = float(explicit_p)
                if 0.0 <= val <= 1.0:
                    return val
            except (ValueError, TypeError):
                pass

        match = re.search(r"\bp\s*(?:<|=)\s*([0-9.]+)\b", text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return None
