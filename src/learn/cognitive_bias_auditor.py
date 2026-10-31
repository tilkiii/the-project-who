"""
Cognitive Blindspot, Reasoning Bias & Rhetorical Fallacy Auditor for who??.
Scans verified [TARGET_PRIMARY] statements for empirically defined cognitive heuristics and rhetorical patterns
(False Dilemma, Absolutist Overgeneralization, Sunk-Cost Anchoring, Appeal to Authority, Causal Reductionism,
Status-Quo Defensiveness), balanced against Cognitive Safeguards (Steelmanning, Base-Rate Hedging).
Strictly grounded in verbatim primary quotes — zero fabricated claims (AGENTS.md Rule #3).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.text_cleaner import extract_target_primary_text, tokenize_words, tr_lower

logger = get_logger("who.learn.cognitive_bias")


class CognitiveBiasAuditor:
    """Detects evidence-backed cognitive biases, rhetorical fallacies, and epistemic safeguards in target discourse."""

    BIAS_PATTERNS: Dict[str, Dict[str, Any]] = {
        "false_dilemma": {
            "label_tr": "Yanlış İkilem / İkili Kutuplaştırma (False Dilemma)",
            "description": "Karmaşık olasılık uzayını yalnızca iki uç seçeneğe indirgeme eğilimi.",
            "regexes": [
                r"\bya\s+[^.!?,]{3,60}\s+ya\s+da\b",
                r"\b(?:tek\s+yol|tek\s+seçenek|başka\s+çare\s+yok|başka\s+alternatif\s+yok)\b",
                r"\beither\s+[^.!?,]{3,60}\s+or\b",
                r"\b(?:the\s+only\s+choice|no\s+other\s+alternative|the\s+only\s+way\s+forward)\b",
            ],
        },
        "absolutist_overgeneralization": {
            "label_tr": "Mutlakiyetçi Aşırı Genelleme (Overgeneralization)",
            "description": "İstisnaları dışlayarak evrensel ve koşulsuz genellemeler kurma örüntüsü.",
            "regexes": [
                r"\b(?:istisnasız\s+herkes|her\s+zaman\s+böyle|hiçbir\s+koşulda|asla\s+değişmez|tamamen\s+imkansız)\b",
                r"\b(?:without\s+exception|everyone\s+always|under\s+no\s+circumstances|never\s+works)\b",
            ],
        },
        "appeal_to_unquestioned_authority": {
            "label_tr": "Sorgulanamaz Otoriteye / Ortak Akla Başvurma",
            "description": "Ampirik mekanizma yerine tartışılmaz konsensüs veya otorite iddiasıyla argüman kapatma.",
            "regexes": [
                r"\b(?:tartışmaya\s+kapalı|herkes\s+biliyor\s+ki|tüm\s+uzmanlar\s+hemfikir|inkar\s+edilemez\s+gerçek)\b",
                r"\b(?:beyond\s+any\s+debate|everyone\s+knows\s+that|experts\s+unanimously|undeniable\s+truth)\b",
            ],
        },
        "sunk_cost_anchoring": {
            "label_tr": "Batık Maliyet & Geçmiş Yatırım Çapası (Sunk-Cost Anchoring)",
            "description": "Gelecekteki beklenen değer yerine geçmişte harcanan emek/maliyete tutunma eğilimi.",
            "regexes": [
                r"\b(?:bu\s+kadar\s+yatırımdan\s+sonra|geri\s+dönülemez\s+nokta|bunca\s+emekten\s+sonra|vazgeçmek\s+için\s+çok\s+geç)\b",
                r"\b(?:after\s+all\s+we\s+invested|too\s+far\s+to\s+turn\s+back|point\s+of\s+no\s+return|cannot\s+abandon\s+now)\b",
            ],
        },
        "causal_reductionism": {
            "label_tr": "Tek Nedenli İndirgemecilik (Single-Cause Fallacy)",
            "description": "Çok değişkenli sistem sonuçlarını tek bir kök nedene bağlama kısayolu.",
            "regexes": [
                r"\b(?:tek\s+sebebi|yegane\s+nedeni|tamamen\s+\w+\s+yüzünden|sadece\s+bundan\s+kaynaklanıyor)\b",
                r"\b(?:the\s+sole\s+reason|entirely\s+caused\s+by|the\s+single\s+cause|purely\s+because\s+of)\b",
            ],
        },
        "status_quo_defensiveness": {
            "label_tr": "Statüko & Kurumsal Refleks Savunusu (Status-Quo Bias)",
            "description": "Mevcut düzenin değişim maliyetini abartıp yapısal reformları risk olarak çerçeveleme.",
            "regexes": [
                r"\b(?:mevcut\s+düzeni\s+bozmak|geleneksel\s+yapıyı\s+korumak\s+zorundayız|denenmemiş\s+macera)\b",
                r"\b(?:proven\s+tradition\s+must\s+not|destabilize\s+existing\s+order|reckless\s+experiment)\b",
            ],
        },
    }

    SAFEGUARD_PATTERNS: List[Tuple[str, str]] = [
        ("steelmanning_concession", r"\b(?:karşı\s+argümanın\s+haklı|her\s+ne\s+kadar|bununla\s+birlikte|yanılıyor\s+olabilirim|although|admittedly|counterargument|i\s+might\s+be\s+wrong|steelman)\b"),
        ("empirical_falsifiability", r"\b(?:yanlışlanabilir|hipotez|ampirik\s+kanıt|veri\s+değişirse|kontrol\s+grubu|falsifiable|empirical\s+evidence|base\s+rate|confidence\s+interval|null\s+hypothesis)\b"),
        ("conditional_framing", r"\b(?:varsayımı\s+altında|koşuluna\s+bağlı|bağlama\s+göre|under\s+the\s+assumption|conditional\s+on|depends\s+on\s+context)\b"),
    ]

    def __init__(self, output_dir: Optional[Path] = None, vault_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.vault_dir = vault_dir

    def audit_corpus(self, target_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scan verified target corpus for cognitive blindspots and epistemic safeguards."""
        sentences_with_source: List[Tuple[str, str]] = []
        total_words = 0

        for idx, doc in enumerate(documents):
            src = doc.get("source_id") or doc.get("file_name") or f"doc_{idx}"
            primary = extract_target_primary_text(doc.get("content", ""))
            if not primary:
                continue
            total_words += len(tokenize_words(primary))
            for s in re.split(r"(?<=[.!?])\s+|\n+", primary):
                s_clean = s.strip()
                if len(s_clean) >= 18:
                    sentences_with_source.append((src, s_clean))

        if total_words == 0:
            return {
                "target_name": target_name,
                "status": "INSUFFICIENT_CORPUS",
                "total_words_analyzed": 0,
                "detected_biases": [],
                "epistemic_safeguards": [],
                "probabilistic_statement": "Bilişsel Kör Nokta Analizi: İncelenecek doğrulanmış birincil metin bulunamadı.",
            }

        detected_biases: List[Dict[str, Any]] = []
        total_bias_hits = 0

        for bias_id, spec in self.BIAS_PATTERNS.items():
            matched_quotes: List[Dict[str, str]] = []
            compiled = [re.compile(pat, re.IGNORECASE) for pat in spec["regexes"]]

            for src, sent in sentences_with_source:
                sent_norm = tr_lower(sent)
                if any(rx.search(sent_norm) for rx in compiled):
                    matched_quotes.append({"source_id": src, "quote": sent})

            if matched_quotes:
                hits = len(matched_quotes)
                total_bias_hits += hits
                rate_per_k = round((hits / total_words) * 1000.0, 2)
                conf = round(min(0.92, 0.55 + (hits * 0.10) + min(0.20, total_words / 2000.0)), 2)
                detected_biases.append({
                    "bias_id": bias_id,
                    "label": spec["label_tr"],
                    "description": spec["description"],
                    "occurrences": hits,
                    "density_per_1k_words": rate_per_k,
                    "confidence_score": conf,
                    "evidence_quotes": matched_quotes[:4],
                })

        detected_biases.sort(key=lambda x: x["occurrences"], reverse=True)

        safeguards: List[Dict[str, Any]] = []
        total_safeguard_hits = 0
        for sg_id, pat in self.SAFEGUARD_PATTERNS:
            rx = re.compile(pat, re.IGNORECASE)
            sg_quotes = [
                {"source_id": src, "quote": sent}
                for src, sent in sentences_with_source
                if rx.search(tr_lower(sent))
            ]
            if sg_quotes:
                total_safeguard_hits += len(sg_quotes)
                safeguards.append({
                    "safeguard_id": sg_id,
                    "occurrences": len(sg_quotes),
                    "density_per_1k_words": round((len(sg_quotes) / total_words) * 1000.0, 2),
                    "evidence_quotes": sg_quotes[:3],
                })

        # Net Cognitive Resilience Score (0-100): rewards safeguards, penalizes unhedged bias density
        bias_density = (total_bias_hits / total_words) * 1000.0
        safeguard_density = (total_safeguard_hits / total_words) * 1000.0
        raw_resilience = 75.0 + (safeguard_density * 4.5) - (bias_density * 7.5)
        resilience_score = int(round(max(10.0, min(98.0, raw_resilience))))
        overall_conf = round(min(0.93, 0.55 + min(0.38, total_words / 1800.0)), 2)

        report = {
            "target_name": target_name,
            "status": "AUDIT_COMPLETED",
            "total_words_analyzed": total_words,
            "total_sentences_analyzed": len(sentences_with_source),
            "cognitive_resilience_score_pct": resilience_score,
            "total_bias_markers_found": total_bias_hits,
            "total_safeguard_markers_found": total_safeguard_hits,
            "detected_biases": detected_biases,
            "epistemic_safeguards": safeguards,
            "confidence_score": overall_conf,
            "probabilistic_statement": (
                f"Bilişsel Kör Nokta & Muhakeme Denetimi: {total_words} kelimelik birincil söylemde "
                f"{len(detected_biases)} farklı bilişsel kısayol kategorisinde toplam {total_bias_hits} birincil kanıt cümlesi "
                f"ve {total_safeguard_hits} rasyonel öz-denetim (epistemic safeguard) işareti saptanmıştır. "
                f"Bilişsel Esneklik & Safsata Direnci Skoru: **%{resilience_score}** (Güven Skoru: %{int(overall_conf * 100)})."
            ),
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        """Persist cognitive blindspot JSON and Markdown audit reports."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "cognitive_blindspots.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        md_dir = self.vault_dir or self.output_dir
        if md_dir:
            md_dir.mkdir(parents=True, exist_ok=True)
            lines = [
                f"# 🧠 Cognitive Blindspot & Rhetorical Heuristic Audit: {report['target_name']}",
                f"*Cognitive Resilience Score:* `%{report['cognitive_resilience_score_pct']}` | *Words Analyzed:* `{report['total_words_analyzed']}`\n",
                f"> {report['probabilistic_statement']}\n",
                "## 1. Observed Cognitive Biases & Heuristics (Birincil Alıntı Kanıtlı)",
            ]
            if report["detected_biases"]:
                for b in report["detected_biases"]:
                    lines.append(
                        f"### ⚠️ {b['label']} (`{b['occurrences']}` kanıt, `{b['density_per_1k_words']}/1k kelime`)"
                    )
                    lines.append(f"*{b['description']}*")
                    for q in b["evidence_quotes"]:
                        lines.append(f"- `[{q['source_id']}]`: \"{q['quote']}\"")
            else:
                lines.append("- İncelenen korpusta belirgin bir retorik safsata veya katı bilişsel kısayol kalıbı saptanmadı.")

            lines.append("\n## 2. Observed Epistemic Safeguards (Rasyonel Öz-Denetim Kanıtları)")
            if report["epistemic_safeguards"]:
                for sg in report["epistemic_safeguards"]:
                    lines.append(f"- **{sg['safeguard_id']}** (`{sg['occurrences']}` kanıt)")
                    for q in sg["evidence_quotes"][:2]:
                        lines.append(f"  - `[{q['source_id']}]`: \"{q['quote']}\"")
            else:
                lines.append("- Açık karşı-argüman kabulü (steelmanning) veya yanlışlanabilirlik vurgusu saptanmadı.")

            (md_dir / "cognitive_blindspots_report.md").write_text("\n".join(lines), encoding="utf-8")

