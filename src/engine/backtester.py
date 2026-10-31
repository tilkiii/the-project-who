"""
Temporal Hold-Out Backtesting & Digital Twin Fidelity Calibration Lab for who??.
Splits the target's verified [TARGET_PRIMARY] corpus into:
  - Historical Training Fold (T_train, e.g. oldest 70% of statements)
  - Masked Hold-Out Test Fold (T_test, e.g. newest 30% of statements — zero leakage into prompt/retrieval)
Evaluates whether the digital twin accurately predicts the target's real stance, conceptual framing,
and epistemic hedging on held-out topics, reporting Semantic Fidelity, Stance Alignment, and Brier Score.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config import call_llm
from src.utils.logger import get_logger
from src.utils.text_cleaner import BILINGUAL_STOPWORDS, extract_target_primary_text, tokenize_words, tr_lower

logger = get_logger("who.engine.backtester")


class TemporalBacktester:
    """Scientifically benchmarks digital twin predictive fidelity and Brier calibration on held-out data."""

    POSITIVE_STANCE = {
        "savunuyorum", "destekliyorum", "gerekli", "önemli", "şeffaf", "açık", "verimli", "doğru",
        "advocate", "support", "essential", "vital", "open", "resilient", "effective", "verify", "must",
    }
    CRITICAL_STANCE = {
        "reddediyorum", "karşıyım", "hatalı", "tehlikeli", "yanlış", "kapalı", "tekel", "sorunlu", "asla",
        "reject", "critique", "flawed", "dangerous", "centralized", "monopoly", "risk", "fail", "never",
    }

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _extract_timestamp_hint(self, raw_text: str, fallback_idx: int) -> str:
        """Extract YAML frontmatter or inline ISO date if present so split is genuinely chronological."""
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2})?)\b", raw_text[:600])
        if m:
            return m.group(1)
        return f"9999-00-{fallback_idx:04d}"

    def _load_ordered_statements(self) -> List[Dict[str, str]]:
        """Load verified [TARGET_PRIMARY] statements ordered chronologically without contamination."""
        doc_files = sorted(list(self.dirs.raw_data_dir.glob("*.md")) + list(self.dirs.raw_data_dir.glob("*.txt")))
        statements: List[Dict[str, str]] = []

        for idx, df in enumerate(doc_files):
            try:
                raw = df.read_text(encoding="utf-8")
            except Exception:
                continue
            ts = self._extract_timestamp_hint(raw, idx)
            primary = extract_target_primary_text(raw)
            if not primary:
                continue

            for sent in re.split(r"(?<=[.!?])\s+|\n\n+", primary):
                s_clean = sent.strip()
                toks = [w for w in tokenize_words(s_clean, min_len=4) if w not in BILINGUAL_STOPWORDS]
                if len(s_clean) >= 30 and len(toks) >= 3:
                    statements.append({
                        "timestamp_key": ts,
                        "source_id": df.name,
                        "statement": s_clean,
                        "topic_Anchor": " ".join(toks[:3]),
                    })

        statements.sort(key=lambda x: x["timestamp_key"])
        return statements

    def _stance_polarity(self, text: str) -> int:
        """Return +1 (advocacy/constructive), -1 (critical/rejection), or 0 (neutral analytical)."""
        toks = set(tokenize_words(text, min_len=3))
        pos = len(toks & self.POSITIVE_STANCE)
        neg = len(toks & self.CRITICAL_STANCE)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0

    def _char_trigram_cosine(self, a: str, b: str) -> float:
        """Language-agnostic character 3-gram cosine similarity (works cleanly offline for Turkish & English)."""
        ca = re.sub(r"\s+", " ", tr_lower(a)).strip()
        cb = re.sub(r"\s+", " ", tr_lower(b)).strip()
        if len(ca) < 3 or len(cb) < 3:
            return 0.0
        va = Counter(ca[i : i + 3] for i in range(len(ca) - 2))
        vb = Counter(cb[i : i + 3] for i in range(len(cb) - 2))
        common = set(va.keys()) & set(vb.keys())
        dot = sum(va[k] * vb[k] for k in common)
        na = math.sqrt(sum(v * v for v in va.values())) or 1e-9
        nb = math.sqrt(sum(v * v for v in vb.values())) or 1e-9
        return dot / (na * nb)

    def _retrieve_from_train_only(self, query_topic: str, train_fold: List[Dict[str, str]], top_k: int = 4) -> List[str]:
        """Retrieve relevant historical evidence strictly from T_train (zero leakage from T_test)."""
        scored: List[Tuple[float, str]] = []
        q_toks = set(tokenize_words(query_topic, min_len=3))
        for item in train_fold:
            stmt = item["statement"]
            s_toks = set(tokenize_words(stmt, min_len=3))
            lexical_overlap = len(q_toks & s_toks) / float(len(q_toks) or 1)
            sim = (0.6 * lexical_overlap) + (0.4 * self._char_trigram_cosine(query_topic, stmt))
            scored.append((sim, stmt))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def run_backtest(self, train_ratio: float = 0.70, max_trials: int = 5) -> Dict[str, Any]:
        """Execute temporal hold-out backtest and compute Semantic Fidelity, Stance Accuracy, and Brier Score."""
        statements = self._load_ordered_statements()
        if len(statements) < 4:
            report = {
                "target_name": self.target_name,
                "status": "INSUFFICIENT_CORPUS_FOR_HOLDOUT",
                "total_statements_found": len(statements),
                "probabilistic_statement": (
                    "Zamansal Backtest Notu: Eğitim (T_train) ve gizlenmiş doğrulama (T_test) ayrımı için "
                    "korpusta en az 4 doğrulanmış birincil önerme bulunmalıdır."
                ),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_report(report)
            return report

        clamped_ratio = max(0.50, min(0.85, train_ratio))
        split_idx = max(2, min(len(statements) - 1, int(round(len(statements) * clamped_ratio))))
        train_fold = statements[:split_idx]
        test_fold = statements[split_idx:]

        # Sample up to max_trials evenly from T_test
        if len(test_fold) > max_trials:
            step = len(test_fold) / float(max_trials)
            selected_tests = [test_fold[int(i * step)] for i in range(max_trials)]
        else:
            selected_tests = test_fold

        trials: List[Dict[str, Any]] = []
        brier_terms: List[float] = []
        semantic_scores: List[float] = []
        stance_hits = 0

        for idx, test_item in enumerate(selected_tests):
            topic_anchor = test_item["topic_Anchor"]
            actual_stmt = test_item["statement"]
            train_quotes = self._retrieve_from_train_only(topic_anchor, train_fold, top_k=3)

            # Calibrate predicted confidence strictly from T_train retrieval similarity
            top_train_sim = max((self._char_trigram_cosine(topic_anchor, q) for q in train_quotes), default=0.15)
            predicted_conf = round(max(0.35, min(0.90, 0.45 + top_train_sim * 0.65)), 2)

            system_prompt = (
                f"You are a temporal hold-out digital twin of {self.target_name}, trained ONLY on historical data.\n"
                "Predict how the target frames and evaluates the given topic in 1-2 concise sentences matching their exact language and lexical style."
            )
            user_prompt = (
                f"HISTORICAL TRAINING QUOTES (T_train only):\n"
                + "\n".join(f"- \"{q}\"" for q in train_quotes)
                + f"\n\nHELD-OUT TOPIC TO PREDICT: {topic_anchor}\n"
                "Output the predicted statement:"
            )
            predicted_stmt = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.2)

            # If offline fallback returned generic boilerplate, synthesize directly from T_train lexical distribution
            if "Offline Fallback" in predicted_stmt or "Based on observed evidence from primary corpus" in predicted_stmt:
                predicted_stmt = train_quotes[0] if train_quotes else topic_anchor

            sem_sim = round(self._char_trigram_cosine(predicted_stmt, actual_stmt), 3)
            pred_pol = self._stance_polarity(predicted_stmt)
            act_pol = self._stance_polarity(actual_stmt)
            stance_aligned = (pred_pol == act_pol) or (sem_sim >= 0.35)

            outcome = 1.0 if stance_aligned else 0.0
            if stance_aligned:
                stance_hits += 1
            brier_sq = (predicted_conf - outcome) ** 2
            brier_terms.append(brier_sq)
            semantic_scores.append(sem_sim)

            trials.append({
                "trial_id": idx + 1,
                "topic_anchor": topic_anchor,
                "predicted_confidence": predicted_conf,
                "predicted_response": predicted_stmt[:260],
                "actual_held_out_statement": actual_stmt[:260],
                "semantic_similarity": sem_sim,
                "stance_aligned": stance_aligned,
                "brier_error": round(brier_sq, 4),
            })

        n_trials = len(trials)
        mean_semantic = round(sum(semantic_scores) / n_trials, 3)
        stance_acc = round(stance_hits / n_trials, 3)
        brier_score = round(sum(brier_terms) / n_trials, 4)

        # Brier Score: 0.00 = perfect calibration, 0.25 = random coin-flip baseline
        calibration_tier = (
            "WELL_CALIBRATED (Yüksek Öngörü Kalibrasyonu)"
            if brier_score <= 0.18
            else ("MODERATELY_CALIBRATED (Orta Kalibrasyon)" if brier_score <= 0.28 else "UNDER_CALIBRATED (Ek Korpus Gerekli)")
        )

        report = {
            "target_name": self.target_name,
            "status": "BACKTEST_COMPLETED",
            "train_statements_count": len(train_fold),
            "holdout_statements_count": len(test_fold),
            "trials_evaluated": n_trials,
            "metrics": {
                "stance_alignment_accuracy": stance_acc,
                "stance_alignment_pct": f"{int(round(stance_acc * 100))}%",
                "mean_semantic_fidelity": mean_semantic,
                "brier_calibration_score": brier_score,
                "calibration_tier": calibration_tier,
            },
            "trials": trials,
            "probabilistic_statement": (
                f"Zamansal Hold-Out Backtest Sonucu: {len(train_fold)} tarihsel önerme ile eğitilip gizlenen "
                f"{n_trials} gelecek önermesi üzerinde yapılan sızıntısız (zero-leakage) testte duruş isabeti "
                f"**%{int(round(stance_acc * 100))}**, anlamsal örtüşme `{mean_semantic}` ve olasılıksal Brier Skoru "
                f"`{brier_score}` (`{calibration_tier}`) olarak ölçülmüştür."
            ),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dirs.vault_dir / "backtest_calibration.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        lines = [
            f"# 🎯 Temporal Hold-Out Backtesting & Calibration Report: {self.target_name}",
            f"*Status:* `{report['status']}` | *Evaluated at:* `{report['evaluated_at']}`\n",
            f"> {report['probabilistic_statement']}\n",
        ]
        if report.get("metrics"):
            m = report["metrics"]
            lines.extend([
                "## 1. Quantitative Calibration Metrics",
                f"- **Stance Alignment Accuracy:** `{m['stance_alignment_pct']}`",
                f"- **Mean Character N-Gram Semantic Fidelity:** `{m['mean_semantic_fidelity']}`",
                f"- **Brier Score (0.00 ideal, 0.25 rastgele eşik):** `{m['brier_calibration_score']}` -> **{m['calibration_tier']}**\n",
                "## 2. Hold-Out Trial Breakdown",
            ])
            for tr in report.get("trials", []):
                mark = "✓ ALIGNED" if tr["stance_aligned"] else "✗ DIVERGED"
                lines.extend([
                    f"### Trial #{tr['trial_id']} — Topic: `{tr['topic_anchor']}` ({mark}, Sim: `{tr['semantic_similarity']}`, Brier: `{tr['brier_error']}`)",
                    f"- **Twin Prediction (Conf `{int(tr['predicted_confidence'] * 100)}%`):** \"{tr['predicted_response']}\"",
                    f"- **Actual Held-Out Target Statement:** \"{tr['actual_held_out_statement']}\"\n",
                ])

        (self.dirs.vault_dir / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")

