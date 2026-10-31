"""
Toulmin Argumentation Miner & Fuzzy Cognitive Map (Causal Belief Matrix) Engine for who??.
Models WHY a target holds specific positions by extracting structured Toulmin chains:
  [Grounds / Evidence] --(via Warrant / Hidden Assumption)--> [Qualifier + Claim] (unless [Rebuttal])
Constructs a causal Fuzzy Cognitive Map (FCM) and runs counterfactual perturbation to find
the target's 'Keystone Warrants' — underlying assumptions whose falsification maximally shifts downstream stances.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.text_cleaner import BILINGUAL_STOPWORDS, extract_target_primary_text, tokenize_words, tr_lower

logger = get_logger("who.vault.argument_miner")


class ToulminArgumentMiner:
    """Extracts Toulmin argumentation structures and simulates Fuzzy Cognitive Map (FCM) belief propagation."""

    # Bilingual discourse markers separating Claim and Grounds/Warrant
    CAUSAL_MARKERS = (
        " çünkü ", " zira ", " nedeniyle ", " sebebiyle ", " sayesinde ",
        " because ", " since ", " given that ", " due to ",
    )
    CONSEQUENTIAL_MARKERS = (
        " dolayısıyla ", " bu yüzden ", " bu nedenle ", " sonuç olarak ", " böylece ",
        " therefore ", " consequently ", " thus ", " hence ", " implies that ",
    )
    REBUTTAL_MARKERS = (
        " ancak ", " fakat ", " yalnız ", " olmadıkça ", " hariç ", " rağmen ",
        " unless ", " except ", " however ", " provided that ", " although ",
    )
    NEGATIVE_CAUSAL_VERBS = {
        "azaltır", "engeller", "zayıflatır", "bozar", "tehdit", "ortadan kaldırır", "önler",
        "reduces", "prevents", "eliminates", "undermines", "threatens", "blocks", "inhibits",
    }

    def __init__(self, target_name: str, vault_dir: Optional[Path] = None):
        self.target_name = target_name
        self.vault_dir = vault_dir

    def _extract_key_concept(self, clause: str) -> str:
        """Distill a 1-3 word salient concept label from an argument clause (Turkish & English aware)."""
        tokens = [w for w in tokenize_words(clause, min_len=4) if w not in BILINGUAL_STOPWORDS]
        if not tokens:
            tokens = tokenize_words(clause, min_len=3)
        if not tokens:
            return "Core Proposition"
        selected = tokens[:2]
        return " ".join(w.capitalize() for w in selected)

    def _detect_qualifier(self, text: str) -> str:
        """Classify epistemic qualifier strength of a claim."""
        low = tr_lower(text)
        if any(w in low for w in ("kesinlikle", "mutlaka", "şüphesiz", "zorunludur", "definitely", "must", "certainly", "always")):
            return "STRONG_NECESSITY (Kesin/Zorunlu)"
        if any(w in low for w in ("olası", "muhtemelen", "belki", "eğiliminde", "plausible", "likely", "might", "tends to")):
            return "PROBABILISTIC_CALIBRATED (Olasılıksal)"
        return "ASSERTIVE_EMPIRICAL (Doğrudan İddia)"

    def mine_arguments(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract Toulmin argument chains and construct a causal Fuzzy Cognitive Map across corpus documents."""
        toulmin_chains: List[Dict[str, Any]] = []
        causal_edges: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            src = doc.get("source_id") or doc.get("file_name") or f"doc_{idx}"
            primary = extract_target_primary_text(doc.get("content", ""))
            if not primary:
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", primary) if len(s.strip()) >= 25]
            for sent in sentences:
                padded = f" {tr_lower(sent)} "
                claim_part = ""
                grounds_part = ""
                rebuttal_part = ""

                # Check for Rebuttal / Exception clause
                working_sent = sent
                for rm in self.REBUTTAL_MARKERS:
                    pos = padded.find(rm)
                    if pos != -1 and 10 < pos < len(sent) - 10:
                        rebuttal_part = sent[pos:].strip(" ,.;:-")
                        working_sent = sent[:pos].strip()
                        padded = f" {tr_lower(working_sent)} "
                        break

                # Pattern A: [CLAIM] because/çünkü [GROUNDS]
                matched_causal = False
                for cm in self.CAUSAL_MARKERS:
                    pos = padded.find(cm)
                    if pos != -1:
                        claim_part = working_sent[:pos].strip(" ,;:-")
                        grounds_part = working_sent[pos + len(cm) - 1 :].strip(" ,.;:-")
                        matched_causal = True
                        break

                # Pattern B: [GROUNDS] therefore/dolayısıyla [CLAIM]
                if not matched_causal:
                    for cqm in self.CONSEQUENTIAL_MARKERS:
                        pos = padded.find(cqm)
                        if pos != -1:
                            grounds_part = working_sent[:pos].strip(" ,;:-")
                            claim_part = working_sent[pos + len(cqm) - 1 :].strip(" ,.;:-")
                            matched_causal = True
                            break

                if not (claim_part and grounds_part and len(claim_part) >= 10 and len(grounds_part) >= 10):
                    continue

                cause_concept = self._extract_key_concept(grounds_part)
                effect_concept = self._extract_key_concept(claim_part)
                qualifier = self._detect_qualifier(claim_part)

                is_negative = any(nv in tr_lower(sent) for nv in self.NEGATIVE_CAUSAL_VERBS)
                polarity = -0.80 if is_negative else 0.85

                warrant = (
                    f"Eğer '{cause_concept}' geçerliyse, nedensel olarak '{effect_concept}' "
                    f"{'zayıflar/engellenir' if is_negative else 'güçlenir/gerçekleşir'}."
                )

                chain_id = f"arg_{len(toulmin_chains) + 1:02d}"
                toulmin_chains.append({
                    "argument_id": chain_id,
                    "source_id": src,
                    "claim": claim_part,
                    "grounds": grounds_part,
                    "warrant": warrant,
                    "qualifier": qualifier,
                    "rebuttal": rebuttal_part or "Açık istisna koşulu belirtilmemiş",
                    "cause_concept": cause_concept,
                    "effect_concept": effect_concept,
                    "causal_weight": polarity,
                    "verbatim_quote": sent,
                })

                if cause_concept != effect_concept:
                    causal_edges.append({
                        "source": cause_concept,
                        "target": effect_concept,
                        "weight": polarity,
                        "argument_id": chain_id,
                    })

        fcm_analysis = self._analyze_fuzzy_cognitive_map(causal_edges)
        confidence = round(min(0.94, 0.55 + len(toulmin_chains) * 0.06), 2) if toulmin_chains else 0.0

        report = {
            "target_name": self.target_name,
            "status": "ARGUMENTS_MINED" if toulmin_chains else "NO_EXPLICIT_CAUSAL_CHAINS",
            "total_arguments_extracted": len(toulmin_chains),
            "toulmin_arguments": toulmin_chains[:25],
            "fuzzy_cognitive_map": fcm_analysis,
            "confidence_score": confidence,
            "probabilistic_statement": (
                f"Toulmin Argümantasyon & Nedensel İnanç Analizi: '{self.target_name}' korpusundan "
                f"{len(toulmin_chains)} yapılandırılmış [Kanıt -> Dayanak Varsayım -> İddia] zinciri ve "
                f"{len(fcm_analysis.get('nodes', []))} düğümlü Bulanık Bilişsel Harita (FCM) çıkarılmıştır "
                f"(Güven Skoru: %{int(confidence * 100)})."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _analyze_fuzzy_cognitive_map(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build Fuzzy Cognitive Map adjacency matrix W and simulate counterfactual belief perturbation:
        For each root/driver concept, invert its activation (-1.0) and propagate A^(t+1) = tanh(A^(t) * W)
        for 4 steps to measure total downstream stance shift (Keystone Warrant Leverage).
        """
        if not edges:
            return {"nodes": [], "causal_edges": [], "keystone_assumptions": []}

        node_set = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
        idx_map = {n: i for i, n in enumerate(node_set)}
        n_nodes = len(node_set)

        # Accumulate and clamp adjacency matrix W in [-1.0, 1.0]
        w_mat = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        for e in edges:
            i, j = idx_map[e["source"]], idx_map[e["target"]]
            w_mat[i][j] = max(-1.0, min(1.0, w_mat[i][j] + float(e["weight"])))

        # Perturbation simulation per node
        keystones: List[Dict[str, Any]] = []
        for candidate in node_set:
            c_idx = idx_map[candidate]
            out_degree = sum(1 for j in range(n_nodes) if abs(w_mat[c_idx][j]) > 0.01)
            if out_degree == 0:
                continue

            # Baseline activation (+1 on candidate) vs Falsified activation (-1 on candidate)
            pos_state = self._propagate_fcm(w_mat, c_idx, initial_val=1.0, steps=4)
            neg_state = self._propagate_fcm(w_mat, c_idx, initial_val=-1.0, steps=4)

            downstream_delta = sum(
                abs(pos_state[j] - neg_state[j]) for j in range(n_nodes) if j != c_idx
            )
            affected = [
                {
                    "concept": node_set[j],
                    "shift_magnitude": round(abs(pos_state[j] - neg_state[j]), 3),
                }
                for j in range(n_nodes)
                if j != c_idx and abs(pos_state[j] - neg_state[j]) > 0.15
            ]
            affected.sort(key=lambda x: x["shift_magnitude"], reverse=True)

            keystones.append({
                "keystone_concept": candidate,
                "direct_causal_links": out_degree,
                "systemic_perturbation_leverage": round(downstream_delta, 3),
                "impacted_stances": affected[:4],
            })

        keystones.sort(key=lambda x: x["systemic_perturbation_leverage"], reverse=True)

        return {
            "nodes": node_set,
            "causal_edges": edges,
            "keystone_assumptions": keystones[:6],
        }

    def _propagate_fcm(
        self,
        w_mat: List[List[float]],
        clamped_idx: int,
        initial_val: float,
        steps: int = 4,
    ) -> List[float]:
        """Run hyperbolic-tangent Fuzzy Cognitive Map state propagation with clamped source intervention."""
        n = len(w_mat)
        state = [0.0] * n
        state[clamped_idx] = initial_val

        for _ in range(steps):
            next_state = [0.0] * n
            for j in range(n):
                if j == clamped_idx:
                    next_state[j] = initial_val
                else:
                    incoming = sum(state[i] * w_mat[i][j] for i in range(n))
                    next_state[j] = math.tanh(incoming)
            state = next_state
        return state

    def _persist_report(self, report: Dict[str, Any]) -> None:
        """Persist Toulmin argumentation chains and FCM causal map to vault."""
        if not self.vault_dir:
            return
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        with open(self.vault_dir / "toulmin_arguments.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        lines = [
            f"# ⚖️ Toulmin Argumentation & Causal Belief Map: {self.target_name}",
            f"*Arguments Extracted:* `{report['total_arguments_extracted']}` | *Confidence:* `{int(report['confidence_score'] * 100)}%`\n",
            f"> {report['probabilistic_statement']}\n",
            "## 1. Keystone Assumptions (Fikrini Değiştirebilecek Temel Dayanak Noktaları)",
        ]
        keystones = report.get("fuzzy_cognitive_map", {}).get("keystone_assumptions", [])
        if keystones:
            for ks in keystones:
                impacted_str = ", ".join(f"{i['concept']} (Δ={i['shift_magnitude']})" for i in ks["impacted_stances"])
                lines.append(
                    f"- **{ks['keystone_concept']}** — Sistemik Etki Kaldıracı: `{ks['systemic_perturbation_leverage']}` "
                    f"-> Etkilenen Duruşlar: {impacted_str or 'Doğrudan sonuç'}"
                )
        else:
            lines.append("- Çok adımlı nedensel kaldıraç düğümü saptanmadı.")

        lines.append("\n## 2. Structured Toulmin Chains (İddia — Kanıt — Dayanak — İstisna)")
        for arg in report.get("toulmin_arguments", [])[:12]:
            lines.extend([
                f"### [{arg['argument_id']}] `{arg['cause_concept']}` -> `{arg['effect_concept']}` ({arg['qualifier']})",
                f"- **İddia (Claim):** {arg['claim']}",
                f"- **Kanıt (Grounds):** {arg['grounds']}",
                f"- **Dayanak Varsayım (Warrant):** {arg['warrant']}",
                f"- **İstisna / Çürütme Koşulu (Rebuttal):** {arg['rebuttal']}",
                f"- *Birincil Alıntı (`{arg['source_id']}`):* \"{arg['verbatim_quote']}\"\n",
            ])

        (self.vault_dir / "argumentation_map.md").write_text("\n".join(lines), encoding="utf-8")

