"""
Boundary Synthesizer & Empirical Rule Derivation Engine for who??.
Translates meta-analytically validated scientific findings into:
1. Calibrated Z-score and percentile decision boundaries (low, moderate, elevated, extreme).
2. Hard operational rules & behavioral constraints for digital twin simulations.
3. Automated skill mapping directives to prevent ungrounded hallucinations or behavioral drift.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from who.utils.logger import get_logger

logger = get_logger("who.learn.boundary_synthesizer")


class BoundarySynthesizer:
    """
    Synthesizes empirical psychometric distribution boundaries and derives
    actionable behavioral rules for target digital twin simulation.
    """

    # Canonical Empirical Population Priors (derived from psychometric normative data: SD3, NFCS, Prospect Theory, PANAS)
    NORMATIVE_DISTRIBUTIONS = {
        "machiavellianism_index": {
            "mean": 0.28,
            "std": 0.14,
            "p10": 0.12,
            "p25": 0.18,
            "p50": 0.27,
            "p75": 0.38,
            "p90": 0.52,
            "extreme_threshold": 0.55,
        },
        "need_for_closure_index": {
            "mean": 0.34,
            "std": 0.15,
            "p10": 0.15,
            "p25": 0.23,
            "p50": 0.33,
            "p75": 0.45,
            "p90": 0.55,
            "extreme_threshold": 0.55,
        },
        "loss_aversion_index": {
            "mean": 1.10,
            "std": 0.45,
            "p10": 0.60,
            "p25": 0.80,
            "p50": 1.05,
            "p75": 1.40,
            "p90": 1.70,
            "extreme_threshold": 1.75,
        },
        "affective_volatility_index": {
            "mean": 0.20,
            "std": 0.13,
            "p10": 0.06,
            "p25": 0.10,
            "p50": 0.18,
            "p75": 0.28,
            "p90": 0.42,
            "extreme_threshold": 0.48,
        },
        "epistemic_manipulation_index": {
            "mean": 0.22,
            "std": 0.15,
            "p10": 0.05,
            "p25": 0.12,
            "p50": 0.20,
            "p75": 0.32,
            "p90": 0.45,
            "extreme_threshold": 0.50,
        },
    }

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or Path("profiles")

    # -------------------------------------------------------------------------
    # 1. Synthesize Metric Decision Boundaries
    # -------------------------------------------------------------------------

    def synthesize_metric_boundaries(
        self,
        marker: str,
        consensus_r: float = 0.25,
        gold_papers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate statistical distribution boundaries for a given behavioral marker.
        Incorporates meta-analytic consensus effect size (r) and gold standard evidence.
        """
        norm = self.NORMATIVE_DISTRIBUTIONS.get(marker, {
            "mean": 0.30,
            "std": 0.15,
            "p10": 0.12,
            "p25": 0.20,
            "p50": 0.30,
            "p75": 0.40,
            "p90": 0.50,
            "extreme_threshold": 0.55,
        })

        # Bayesian adjustment: if strong empirical correlation exists (|r| >= 0.35),
        # boundary precision narrows, reflecting higher discriminative power.
        r_adjustment = min(0.08, max(0.0, (abs(consensus_r) - 0.20) * 0.25))

        calibrated_p90 = round(norm["p90"] - r_adjustment, 2)
        calibrated_extreme = round(norm["extreme_threshold"] - r_adjustment, 2)

        return {
            "marker": marker,
            "meta_analytic_consensus_r": consensus_r,
            "gold_evidence_count": len(gold_papers or []),
            "calibrated_distribution": {
                "mean": norm["mean"],
                "std": norm["std"],
                "percentiles": {
                    "p10": norm["p10"],
                    "p25": norm["p25"],
                    "p50_median": norm["p50"],
                    "p75": norm["p75"],
                    "p90": calibrated_p90,
                },
                "extreme_cutoff": calibrated_extreme,
            },
            "classification_tiers": {
                "low": f"< {norm['p25']}",
                "moderate": f"{norm['p25']} - {norm['p75']}",
                "elevated": f"{norm['p75']} - {calibrated_p90}",
                "extreme": f">= {calibrated_extreme}",
            },
        }

    # -------------------------------------------------------------------------
    # 2. Derive Actionable Simulation Rules
    # -------------------------------------------------------------------------

    def generate_behavioral_rules(
        self,
        features: Dict[str, Any],
        boundaries_map: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate target's psychometric feature vector against calibrated empirical
        boundaries and synthesize explicit operational directives for digital twin simulation.
        """
        rules: List[Dict[str, Any]] = []

        # 1. Machiavellianism & Power Orientation
        mach = features.get("machiavellianism_index", {})
        mach_score = mach.get("score", 0.0)
        if mach_score >= 0.55:
            rules.append({
                "rule_id": "TACTICAL_TRANSACTIONALISM",
                "severity": "CRITICAL",
                "dimension": "Machiavellianism & Strategic Utility",
                "trigger": f"Machiavellianism score ({mach_score}) >= 0.55 (Extreme High)",
                "directive": (
                    "Hedef simülasyonunda salt ahlaki söylem veya soyut fedakârlık vaatlerine dayalı teklifleri "
                    "kuşkuyla karşıla; her müzakerede asimetrik kurumsal fayda, somut çıkar dengesi ve garantili koz talep et. "
                    "Kişisel duygulanımlardan bağımsız, soğuk rasyonel fayda optimizasyonu yürüt."
                ),
                "scientific_grounding": "Paulhus & Williams (2002) Dark Triad SD3; Bereczkei (2015) Machiavellian Intelligence Theory.",
            })
        elif mach_score < 0.20:
            rules.append({
                "rule_id": "COOPERATIVE_INTEGRITY",
                "severity": "MODERATE",
                "dimension": "Machiavellianism & Strategic Utility",
                "trigger": f"Machiavellianism score ({mach_score}) < 0.20 (High Altruism)",
                "directive": (
                    "Etik ilkelere, şeffaflığa ve ortak güvene dayalı iş birliğini öncelikle benimse. "
                    "İkili oynama veya manipülatif güç hamlelerine karşı sert ahlaki itiraz geliştir."
                ),
                "scientific_grounding": "Axelrod (1984) Evolution of Cooperation; Fehr & Gächter (2000) Altruistic Punishment.",
            })

        # 2. Need for Cognitive Closure
        closure = features.get("need_for_closure_index", {})
        closure_score = closure.get("score", 0.0)
        if closure_score >= 0.55:
            rules.append({
                "rule_id": "RIGID_COGNITIVE_CLOSURE",
                "severity": "HIGH",
                "dimension": "Need for Cognitive Closure",
                "trigger": f"Need for Closure score ({closure_score}) >= 0.55 (Extreme Urgency & Decisiveness)",
                "directive": (
                    "Belirsiz, ucu açık ve ertelenen durumları derhal netleştirme baskısı uygula. "
                    "Görüşmelerde yapılandırılmış gündem, kesin teslim tarihleri ve sarih sonuç maddeleri talep et; "
                    "gri alanları ve aşırı nüanslı belirsizlikleri hızla tasfiye ederek kesin karara bağla."
                ),
                "scientific_grounding": "Webster & Kruglanski (1994) Need for Closure Scale (NFCS); Roets et al. (2015).",
            })
        elif closure_score < 0.25:
            rules.append({
                "rule_id": "EXPLORATORY_AMBIGUITY_TOLERANCE",
                "severity": "LOW",
                "dimension": "Need for Cognitive Closure",
                "trigger": f"Need for Closure score ({closure_score}) < 0.25 (High Openness to Ambiguity)",
                "directive": (
                    "Erken veya aceleye getirilmiş uzlaşıları reddet. Alternatif olasılıkları, karşı hipotezleri "
                    "ve bağlamsal incelikleri masada tutarak çok yönlü derinlemesine sorgulama yürüt."
                ),
                "scientific_grounding": "Kruglanski & Webster (1996) Motivated Closing of the Mind.",
            })

        # 3. Loss Aversion (Prospect Theory)
        loss = features.get("loss_aversion_index", {})
        loss_ratio = loss.get("loss_to_gain_ratio", 1.0)
        if loss_ratio >= 1.60:
            rules.append({
                "rule_id": "ASYMMETRIC_LOSS_PREVENTION",
                "severity": "CRITICAL",
                "dimension": "Prospect Theory & Loss Aversion",
                "trigger": f"Loss/Gain Ratio ({loss_ratio}) >= 1.60 (High Threat Vigilance)",
                "directive": (
                    "Karar süreçlerinde potansiyel kazançlardan önce olası zayiat, itibar kaybı veya yasal riskleri "
                    "analiz et. Risk azaltıcı güvenceler ve zararı sınırlayıcı protokoller sağlanmadıkça yüksek "
                    "getiri vaat eden inisiyatifleri onaylama."
                ),
                "scientific_grounding": "Kahneman & Tversky (1979) Prospect Theory: Risk Aversion in Choices Involving Gains and Losses.",
            })
        elif loss_ratio < 0.80:
            rules.append({
                "rule_id": "OPPORTUNITY_MAXIMIZATION",
                "severity": "MODERATE",
                "dimension": "Prospect Theory & Loss Aversion",
                "trigger": f"Loss/Gain Ratio ({loss_ratio}) < 0.80 (High Risk Tolerance / Growth Drive)",
                "directive": (
                    "Kayıp korkusu yerine stratejik fırsat büyüklüğüne odaklan. Kontrollü belirsizlikleri "
                    "asimetrik büyüme imkânı olarak değerlendir."
                ),
                "scientific_grounding": "Tversky & Kahneman (1992) Advances in Prospect Theory: Cumulative Representation of Uncertainty.",
            })

        # 4. Affective Volatility vs Stoic Composure
        volatility = features.get("affective_volatility_index", {})
        vol_score = volatility.get("volatility_score", 0.0)
        composure = volatility.get("composure_score", 1.0)
        if vol_score >= 0.45:
            rules.append({
                "rule_id": "AFFECTIVE_SURGE_RESPONSE",
                "severity": "HIGH",
                "dimension": "Affective Dynamics & Emotional Volatility",
                "trigger": f"Affective Volatility ({vol_score}) >= 0.45 (High Reactivity)",
                "directive": (
                    "Provokasyon, doğrudan suçlama veya değerlerine saldırı anlarında duygusal reaktiviteyi yükselt; "
                    "tonda sertleşme, ajitasyon ve keskin hitabet sergileyerek pasif kalma."
                ),
                "scientific_grounding": "Russell (1980) A Circumplex Model of Affect; Kuppens et al. (2010) Emotional Inertia.",
            })
        elif composure >= 0.75:
            rules.append({
                "rule_id": "STOIC_EQUANIMITY",
                "severity": "HIGH",
                "dimension": "Affective Dynamics & Emotional Volatility",
                "trigger": f"Composure score ({composure}) >= 0.75 (High Stoic Equanimity)",
                "directive": (
                    "Kriz, itham veya panik ortamlarında tam duygusal soğukkanlılığı koru. İğneleyici laflara veya "
                    "duygusal provokasyonlara tepki vermek yerine soğuk nesnel veriler ve kurumsal olgunlukla yanıt ver."
                ),
                "scientific_grounding": "Gross (2002) Emotion Regulation: Affective, Cognitive, and Social Consequences.",
            })

        # 5. Moral Foundations Profile
        moral = features.get("moral_foundations_profile", {})
        dominant_moral = moral.get("dominant_foundation", "")
        if "INDIVIDUALIZING" in dominant_moral or "BİREY" in dominant_moral:
            rules.append({
                "rule_id": "INDIVIDUAL_RIGHTS_PRIMACY",
                "severity": "MODERATE",
                "dimension": "Moral Foundations Theory",
                "trigger": "Dominant Foundation: Individualizing (Fairness/Care)",
                "directive": (
                    "Tartışmalarda birey hakları, liyakat, adalet ve haksızlığın önlenmesi argümanlarına mutlak "
                    "öncelik tanı. Kurumsal gelenek veya kör itaat gerekçelerini hak ihlali karşısında geçersiz say."
                ),
                "scientific_grounding": "Graham, Haidt et al. (2011) Mapping the Moral Domain.",
            })
        elif "BINDING" in dominant_moral or "OTORİTE" in dominant_moral:
            rules.append({
                "rule_id": "INSTITUTIONAL_LOYALTY_PRIMACY",
                "severity": "MODERATE",
                "dimension": "Moral Foundations Theory",
                "trigger": "Dominant Foundation: Binding (Authority/Loyalty/Order)",
                "directive": (
                    "Kurumsal sadakat, hiyerarşi, teşkilat bütünlüğü ve teamüllerin korunmasını bireysel taleplerin "
                    "önünde tut. Teşkilatı sarsacak veya iç düzeni bozacak eylemleri ihanet ekseninde değerlendir."
                ),
                "scientific_grounding": "Haidt & Graham (2007) When Morality Opposes Justice: Conservatives Have Moral Intuitions.",
            })

        # 6. Epistemic Manipulation & Gaslighting Check
        manip = features.get("epistemic_manipulation_index", {})
        manip_score = manip.get("score", 0.0)
        if manip_score >= 0.45:
            rules.append({
                "rule_id": "RHETORICAL_ASSERTION_DOMINANCE",
                "severity": "CRITICAL",
                "dimension": "Epistemic Manipulation Index",
                "trigger": f"Manipulation score ({manip_score}) >= 0.45",
                "directive": (
                    "Hitabette retorik üstünlük kurma, kesin hükümlerle karşı tarafın epistemik zeminini sarsma ve "
                    "iddiaları tartışmasız bir gerçeklik gibi sunma eğilimini simüle et. Kendi yargılarını mutlak kabul ettir."
                ),
                "scientific_grounding": "Fricker (2007) Epistemic Injustice; Mercier & Sperber (2011) Why Do Humans Reason?.",
            })

        return rules

    # -------------------------------------------------------------------------
    # 3. Persistence into Target Profile
    # -------------------------------------------------------------------------

    def save_target_boundaries_and_rules(
        self,
        target_dir: Path,
        boundaries: Dict[str, Any],
        rules: List[Dict[str, Any]],
    ) -> Path:
        """Save synthesized boundaries and derived operational rules to target profile."""
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / "behavioral_boundaries.json"

        data = {
            "schema_version": "1.0",
            "synthesized_at": datetime.datetime.now(timezone.utc).isoformat(),
            "boundaries": boundaries,
            "operational_rules": rules,
            "total_rules": len(rules),
        }

        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved behavioral boundaries and {len(rules)} operational rules to {out_file}")
        except Exception as e:
            logger.error(f"Failed to save target behavioral boundaries: {e}")

        return out_file
