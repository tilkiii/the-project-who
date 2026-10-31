"""
Scientific Referee & Meta-Analytic AI Evaluator for who??.
Rigorously evaluates academic papers across empirical dimensions:
- Sample Size (N) & Statistical Power
- Effect Size Magnitude (|r|, Cohen's d) & Significance (p-values)
- Venue & Peer-Review Credibility (Top journals vs preprints)
- Consensus & Replicability Alignment vs Outlier Divergence

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
    Evaluates harvested papers on a 0-100 scientific rigor index and manages the
    canonical knowledge repository of empirical psychometric benchmarks.
    """

    # High-impact peer-reviewed publishers & journals (Tier 1: 20 pts)
    TIER1_VENUES = {
        "nature", "science", "pnas", "cell", "journal of personality and social psychology",
        "jpsp", "computers in human behavior", "ieee", "acm", "american psychologist",
        "psychological science", "plos one", "frontiers in psychology", "springer",
        "elsevier", "oxford university press", "cambridge university press", "wiley",
        "behaviour & information technology", "information processing & management"
    }

    # Preprint repositories (Tier 3: 10 pts)
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

        Returns enriched dictionary with itemized scores and rigor tier.
        """
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        methodology = (paper.get("methodology") or "").lower()
        venue = (paper.get("venue") or paper.get("source") or "").lower()

        # Parse or infer sample size N
        n_raw = paper.get("sample_size")
        n = self._extract_or_infer_sample_size(n_raw, abstract)

        # Parse or infer r and p-value
        r_raw = paper.get("r")
        r = self._extract_or_infer_correlation(r_raw, abstract)

        p_raw = paper.get("p_value")
        p = self._extract_or_infer_p_value(p_raw, abstract)

        # 1. S_N: Sample Size Score (0 - 35)
        is_meta_analysis = "meta-analysis" in methodology or "systematic review" in methodology or "meta-analytic" in abstract.lower()
        if is_meta_analysis:
            s_n = 35.0
        elif n and n > 0:
            # S_N = min(35, max(5, 10 * log10(max(10, N))))
            s_n = min(35.0, max(5.0, 10.0 * math.log10(max(10, n))))
        else:
            s_n = 12.0  # Conservative default for unspecified N

        # 2. S_r: Effect Size & Significance Score (0 - 25)
        abs_r = abs(r)
        # Base correlation contribution (up to 20 pts for r >= 0.40)
        base_r_score = 20.0 * min(1.0, abs_r / 0.40)
        # Significance adjustment
        if p is not None:
            if p < 0.001:
                sig_adj = 5.0
            elif p < 0.01:
                sig_adj = 3.0
            elif p < 0.05:
                sig_adj = 1.0
            else:
                sig_adj = -10.0  # Heavy penalty for non-significant p >= 0.05
        else:
            sig_adj = 2.0  # Neutral mild bonus when p is assumed significant in publication

        s_r = max(0.0, min(25.0, base_r_score + sig_adj))

        # 3. S_v: Venue & Credibility Score (0 - 20)
        s_v = 6.0  # default
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
                # Other peer-reviewed / indexed conference or journal
                if "journal" in venue or "conference" in venue or "transactions" in venue or paper.get("source") in ("pubmed", "crossref"):
                    s_v = 15.0

        # 4. S_c: Consensus Alignment Score (0 - 20)
        delta_r = abs(r - prior_mean_r)
        if delta_r <= 0.10:
            s_c = 20.0
        elif delta_r <= 0.20:
            s_c = 15.0
        elif delta_r <= 0.35:
            s_c = 10.0
        else:
            s_c = 4.0  # Outlier divergence penalty

        # Total Rigor Score
        total_rigor = round(s_n + s_r + s_v + s_c, 1)

        # Tier Assignment
        if total_rigor >= 70.0:
            rigor_tier = "GOLD_STANDARD"
            tier_desc = "Hakemli, yüksek örneklemli ve replike edilebilir altın standart ampirik çalışma."
        elif total_rigor >= 50.0:
            rigor_tier = "SOLID_EMPIRICAL"
            tier_desc = "Güvenilir ampirik metodoloji ve istatistiksel anlamlılık."
        elif total_rigor >= 35.0:
            tier_desc = "Keşifsel veya önbaskı aşamasında; dikkatle değerlendirilmeli."
            rigor_tier = "EXPLORATORY_PREPRINT"
        else:
            rigor_tier = "REJECTED_LOW_RIGOR"
            tier_desc = "Düşük örneklem, zayıf etki büyüklüğü veya yöntemsel tutarsızlık nedeniyle elendi."

        return {
            "title": title,
            "abstract": abstract,
            "source": paper.get("source", "unknown"),
            "url": paper.get("url", ""),
            "doi": paper.get("doi", ""),
            "published": paper.get("published", ""),
            "inferred_n": n,
            "inferred_r": round(r, 3),
            "inferred_p": p,
            "is_meta_analysis": is_meta_analysis,
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
    # 2. Multi-Paper Evaluation & Ranking
    # -------------------------------------------------------------------------

    def evaluate_and_rank_corpus(
        self,
        papers: List[Dict[str, Any]],
        marker: str,
        prior_mean_r: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate a full candidate corpus for a specific behavioral marker.
        - Scores every paper.
        - Filters out REJECTED_LOW_RIGOR studies.
        - Ranks papers descending by total rigor score.
        - Computes pooled meta-analytic consensus r across approved studies via Fisher z.
        """
        if not papers:
            return {
                "marker": marker,
                "total_evaluated": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "consensus_r": prior_mean_r,
                "ranked_papers": [],
                "gold_standard_papers": [],
            }

        evaluated: List[Dict[str, Any]] = []
        for p in papers:
            ev = self.score_paper_rigor(p, marker=marker, prior_mean_r=prior_mean_r)
            evaluated.append(ev)

        # Sort descending by total rigor
        evaluated.sort(key=lambda x: x["scores"]["total_rigor_score"], reverse=True)

        approved = [p for p in evaluated if p["rigor_tier"] != "REJECTED_LOW_RIGOR"]
        rejected = [p for p in evaluated if p["rigor_tier"] == "REJECTED_LOW_RIGOR"]
        gold_standards = [p for p in approved if p["rigor_tier"] == "GOLD_STANDARD"]

        # Meta-analytic pooling across approved papers
        if approved:
            z_sum = 0.0
            w_sum = 0.0
            for ap in approved:
                w = ap["confidence_weight"]
                z = fisher_z(ap["inferred_r"])
                z_sum += z * w
                w_sum += w
            consensus_z = z_sum / (w_sum or 1.0)
            consensus_r = round(inv_fisher_z(consensus_z), 3)
        else:
            consensus_r = prior_mean_r

        return {
            "marker": marker,
            "total_evaluated": len(evaluated),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "consensus_r": consensus_r,
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

        # Merge gold standard studies uniquely by URL / title
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

        # Update global study counter
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

        # Regex heuristics on abstract text: e.g. "N = 1,240", "sample of 500 participants"
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

        # Regex for r = 0.28, r = -0.19, etc.
        match = re.search(r"\br\s*=\s*([+-]?(?:0?\.\d+|[01]))\b", text)
        if match:
            try:
                val = float(match.group(1))
                if -1.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return 0.25  # Standard empirical baseline correlation

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
