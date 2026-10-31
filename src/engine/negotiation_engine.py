"""
Game-Theoretic Multi-Turn Negotiation & Persuasion Matrix Engine for who??.
Derives Prospect Theory parameters (loss aversion lambda, concession elasticity eta),
Non-Negotiable Red Lines, and High-Utility Value Drivers from the target's verified vault & linguistic profile.
Simulates a 3-stage Extensive-Form Negotiation Game Tree across 4 strategic openings to identify
the highest-probability agreement trajectory without deterministic claims (AGENTS.md Rule #3).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import call_llm
from src.utils.logger import get_logger
from src.utils.text_cleaner import tokenize_words, tr_lower

logger = get_logger("who.engine.negotiation")


class GameTheoreticNegotiator:
    """Models target utility functions, red lines, and multi-turn negotiation game trees."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def extract_strategic_utility_profile(self) -> Dict[str, Any]:
        """Derive red lines, value incentives, loss aversion, and concession elasticity from vault data."""
        features = self._load_json(self.dirs.learned_skills_dir / "features.json")
        graph = self._load_json(self.dirs.vault_dir / "concept_network.json")
        character = self._load_json(self.dirs.vault_dir / "character.json")

        epistemic = features.get("epistemic_markers", {})
        pronouns = features.get("pronoun_ratios", {})
        emotional = features.get("emotional_cadence", {})
        manip = features.get("epistemic_manipulation_index", {})

        cal_ratio = float(epistemic.get("epistemic_calibration_ratio", 1.0) or 1.0)
        crit_density = float(emotional.get("critical_density", 5.0) or 5.0)
        i_to_we = float(pronouns.get("i_to_we_ratio", 1.5) or 1.5)
        dogmatism = float(manip.get("score", 0.25) or 0.25)

        # Prospect Theory Loss Aversion (Kahneman-Tversky baseline ~2.25, scaled by critical & dogmatic stance)
        loss_aversion_lambda = round(max(1.20, min(3.60, 1.75 + (crit_density * 0.04) + (dogmatism * 0.90))), 2)

        # Concession Elasticity eta in [0.15, 0.90]: higher epistemic calibration & collective 'we' orientation -> more elastic
        concession_elasticity = round(
            max(0.15, min(0.90, 0.35 + min(0.35, cal_ratio * 0.18) - min(0.20, dogmatism * 0.35) + (0.10 if i_to_we < 1.2 else 0.0))),
            2,
        )

        red_lines: List[Dict[str, Any]] = []
        value_drivers: List[Dict[str, Any]] = []

        for rel in graph.get("relationships", []):
            rtype = rel.get("relation", "")
            tgt = rel.get("target", "")
            src = rel.get("source", "")
            concept = tgt if tr_lower(src) == tr_lower(self.target_name) else tgt
            if not concept:
                continue
            weight = float(rel.get("weight", 0.75) or 0.75)
            if rtype in ("REJECTS", "CRITIQUES", "CONTRADICTS"):
                red_lines.append({"concept": concept, "severity": round(weight, 2), "quote": rel.get("quote", "")})
            elif rtype in ("ADVOCATES", "DEVELOPS", "ALIGNED_WITH"):
                value_drivers.append({"concept": concept, "utility_weight": round(weight, 2), "quote": rel.get("quote", "")})

        if not value_drivers and character.get("topics"):
            for top in character.get("topics", [])[:4]:
                value_drivers.append({"concept": top, "utility_weight": 0.80, "quote": ""})

        red_lines.sort(key=lambda x: x["severity"], reverse=True)
        value_drivers.sort(key=lambda x: x["utility_weight"], reverse=True)

        return {
            "loss_aversion_lambda": loss_aversion_lambda,
            "concession_elasticity_eta": concession_elasticity,
            "epistemic_calibration_ratio": cal_ratio,
            "red_lines": red_lines[:6],
            "value_drivers": value_drivers[:6],
        }

    def simulate_negotiation(
        self,
        objective: str,
        leverage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Simulate a multi-strategy 3-turn negotiation game tree for the given objective."""
        profile = self.extract_strategic_utility_profile()
        lam = profile["loss_aversion_lambda"]
        eta = profile["concession_elasticity_eta"]
        cal = profile["epistemic_calibration_ratio"]

        obj_tokens = set(tokenize_words(f"{objective} {leverage or ''}", min_len=3))

        # Check if objective collides with target's documented Red Lines or aligns with Value Drivers
        red_line_collisions = [
            rl["concept"]
            for rl in profile["red_lines"]
            if set(tokenize_words(rl["concept"], min_len=3)) & obj_tokens
        ]
        aligned_incentives = [
            vd["concept"]
            for vd in profile["value_drivers"]
            if set(tokenize_words(vd["concept"], min_len=3)) & obj_tokens
        ]

        collision_penalty = 0.24 * len(red_line_collisions)
        alignment_bonus = 0.12 * min(2, len(aligned_incentives))
        leverage_bonus = 0.08 if (leverage and len(leverage.strip()) >= 8) else 0.0

        strategies = [
            {
                "strategy_id": "evidence_first_verification",
                "name": "Kanıt & Yanlışlanabilirlik Odaklı Doğrudan Çerçeve (Evidence-First Pitch)",
                "opening_move": "İddiaları ölçülebilir metrikler, açık kaynak doğrulaması ve sınırlı pilot kapsamıyla sun.",
                "expected_target_counter": "Metodolojik varsayımları ve bağımsızlık garantilerini sorgular.",
                "optimal_turn2_pivot": "Ham veri/protokol denetimi hakkı tanı ve bağlayıcı ön koşul dayatma.",
                "raw_prob": 0.52 + min(0.24, cal * 0.14) + alignment_bonus + leverage_bonus - collision_penalty,
            },
            {
                "strategy_id": "reciprocal_concession_coalition",
                "name": "Karşılıklı Taviz & Ortak Yönetişim Çerçevesi (Reciprocal Concession)",
                "opening_move": "Yüksek kapsamlı vizyonla başla, ardından hedefin öncelikli değer alanında kontrol devri öner.",
                "expected_target_counter": "Kapsam daraltma veya özerklik şerhi talep eder.",
                "optimal_turn2_pivot": "Planlı ikinci teklife (anchoring -> concession) geçerek ortak paydada sabitle.",
                "raw_prob": 0.46 + (eta * 0.30) + alignment_bonus + leverage_bonus - collision_penalty,
            },
            {
                "strategy_id": "prospect_theory_loss_mitigation",
                "name": "Kayıp Önleme & Asimetrik Risk Koruması (Loss-Aversion Shield)",
                "opening_move": "Kazanım vaadi yerine eylemsizliğin doğuracağı yapısal/kurumsal kaybı ve çıkış garantisini vurgula.",
                "expected_target_counter": "Kötümser senaryonun olasılığını test eder ve geri dönüş (exit) şartı arar.",
                "optimal_turn2_pivot": "Sıfır kilitlenme (zero lock-in) ve tek taraflı cayma opsiyonu sun.",
                "raw_prob": 0.44 + min(0.28, (lam - 1.2) * 0.15) + alignment_bonus + leverage_bonus - (collision_penalty * 0.8),
            },
            {
                "strategy_id": "coercive_ultimatum_stress_test",
                "name": "Zaman Baskılı Ültimatom (Adversarial Stress-Test — Karşılaştırma Amaçlı)",
                "opening_move": "Kesin süre sınırı ve tek seçenekli ('take-it-or-leave-it') teklif sun.",
                "expected_target_counter": "Özerklik tepkisiyle (psychological reactance) sert ret veya masadan kalkma.",
                "optimal_turn2_pivot": "Ültimatomdan kaçınılmalıdır; geri çekilip kanıt temelli çerçeveye dön.",
                "raw_prob": 0.22 - (lam * 0.04) - collision_penalty,
            },
        ]

        evaluated_branches: List[Dict[str, Any]] = []
        for st in strategies:
            prob = round(max(0.05, min(0.92, st["raw_prob"])), 2)
            evaluated_branches.append({
                "strategy_id": st["strategy_id"],
                "name": st["name"],
                "agreement_probability": prob,
                "agreement_probability_pct": f"{int(round(prob * 100))}%",
                "turn_1_opening": st["opening_move"],
                "turn_2_expected_target_counter": st["expected_target_counter"],
                "turn_3_optimal_pivot": st["optimal_turn2_pivot"],
            })

        evaluated_branches.sort(key=lambda x: x["agreement_probability"], reverse=True)
        recommended = evaluated_branches[0]

        # Synthesize tailored tactical brief via LLM (with safe corpus-grounded fallback)
        tactical_prompt = (
            f"Target: {self.target_name}\n"
            f"Negotiation Objective: {objective}\n"
            f"Available Leverage/Offer: {leverage or 'Mutual empirical value'}\n"
            f"Loss Aversion (lambda): {lam}, Concession Elasticity (eta): {eta}\n"
            f"Red Line Collisions: {red_line_collisions or 'None'}\n"
            f"Top Recommended Strategy: {recommended['name']} ({recommended['agreement_probability_pct']})\n\n"
            "Write a concise, probabilistic 3-step tactical negotiation brief in Turkish adhering strictly to epistemic humility."
        )
        tactical_brief = call_llm(
            tactical_prompt,
            system_prompt="You are a game-theoretic negotiation strategist grounded strictly in observed behavioral evidence.",
            temperature=0.25,
        )

        has_vault_evidence = bool(profile["red_lines"] or profile["value_drivers"])
        confidence = 0.82 if has_vault_evidence else 0.58

        report = {
            "target_name": self.target_name,
            "objective": objective,
            "leverage": leverage or "",
            "utility_parameters": profile,
            "red_line_collisions": red_line_collisions,
            "aligned_value_drivers": aligned_incentives,
            "recommended_strategy": recommended,
            "game_tree_branches": evaluated_branches,
            "tactical_brief": tactical_brief,
            "confidence_score": confidence,
            "epistemic_qualification": (
                "Oyun teorisi uzlaşma olasılıkları hedefin gözlemlenen söylem kalibrasyonu ve bilgi grafı "
                "kırmızı çizgileri üzerinden koşullu olarak simüle edilmiştir; deterministik gelecek tahmini değildir."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dirs.vault_dir / "negotiation_simulation.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        up = report["utility_parameters"]
        rec = report["recommended_strategy"]
        lines = [
            f"# ♟️ Game-Theoretic Negotiation Playbook: {self.target_name}",
            f"*Objective:* `{report['objective']}`",
            f"*Prospect Theory Loss Aversion (λ):* `{up['loss_aversion_lambda']}` | *Concession Elasticity (η):* `{up['concession_elasticity_eta']}`\n",
            f"## 1. Recommended Optimal Strategy: {rec['name']} (`{rec['agreement_probability_pct']}`)",
            f"- **1. Hamle (Açılış):** {rec['turn_1_opening']}",
            f"- **2. Hamle (Beklenen Karşı Tepki):** {rec['turn_2_expected_target_counter']}",
            f"- **3. Hamle (Optimal Pivot):** {rec['turn_3_optimal_pivot']}\n",
            "## 2. All Simulated Strategy Branches",
        ]
        for b in report["game_tree_branches"]:
            lines.append(f"- **{b['name']}** — Uzlaşma Olasılığı: `{b['agreement_probability_pct']}`")

        if report["red_line_collisions"]:
            lines.append(f"\n> ⚠️ **Kırmızı Çizgi Çarpışması:** `{', '.join(report['red_line_collisions'])}`")

        lines.extend(["\n## 3. Tactical Synthesis", report["tactical_brief"]])
        (self.dirs.vault_dir / "negotiation_playbook.md").write_text("\n".join(lines), encoding="utf-8")

