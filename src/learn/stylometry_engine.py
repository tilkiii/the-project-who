"""
Forensic Stylometry & Authorship Attribution Engine for who??.
Implements Burrows' Classic Delta, Cosine Delta, Koppel's Monte-Carlo Feature-Subsampling Verification,
and Sentence Burstiness analysis (detecting ghostwritten or LLM-synthesized outliers inside a target corpus).
Bilingual Turkish & English aware. Adheres strictly to AGENTS.md Rule #3 (Epistemic Honesty).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.text_cleaner import compute_mattr, extract_target_primary_text, tokenize_words, tr_lower

logger = get_logger("who.learn.stylometry")


class ForensicStylometryEngine:
    """Extracts authorial style fingerprints, verifies questioned texts, and detects ghostwriter/synthetic outliers."""

    FUNCTION_WORDS: Tuple[str, ...] = (
        # Turkish function words, conjunctions & epistemic particles
        "ve", "ile", "için", "ama", "fakat", "ancak", "lakin", "çünkü", "dolayısıyla",
        "olarak", "gibi", "kadar", "daha", "çok", "ise", "hem", "veya", "ki", "değil",
        "rağmen", "örneğin", "özellikle", "aslında", "belki", "kesinlikle", "böylece",
        # English function words & discourse connectors
        "the", "and", "of", "to", "in", "that", "for", "with", "as", "but", "however",
        "therefore", "because", "although", "while", "perhaps", "specifically", "which",
    )

    PUNCTUATION_MARKS: Tuple[str, ...] = (",", ";", ":", "?", "!", "—", "-", "(", ")")

    def __init__(self, output_dir: Optional[Path] = None, vault_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.vault_dir = vault_dir

    def _extract_doc_vector(self, text: str, ngram_vocab: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract normalized stylometric measurements from a single text sample."""
        cleaned = extract_target_primary_text(text) or text.strip()
        lowered = tr_lower(cleaned)
        words = tokenize_words(cleaned, min_len=1)
        total_words = len(words)
        if total_words == 0:
            return {"total_words": 0}

        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        sent_lengths = [len(tokenize_words(s, min_len=1)) for s in sentences]
        sent_lengths = [l for l in sent_lengths if l > 0] or [total_words]

        mean_sent = sum(sent_lengths) / len(sent_lengths)
        variance_sent = (
            sum((l - mean_sent) ** 2 for l in sent_lengths) / len(sent_lengths)
            if len(sent_lengths) > 1
            else 0.0
        )
        std_sent = math.sqrt(variance_sent)
        # Burstiness (coefficient of variation of sentence length): humans exhibit higher burstiness (>0.40),
        # whereas heavily edited PR prose or uniform synthetic LLM text exhibits flat burstiness (<0.22).
        burstiness = round(std_sent / (mean_sent or 1.0), 3)

        word_counts = Counter(words)
        func_freqs = {
            f"fw::{fw}": round((word_counts.get(fw, 0) / total_words) * 1000.0, 4)
            for fw in self.FUNCTION_WORDS
        }

        punct_freqs = {
            f"punct::{p}": round((cleaned.count(p) / total_words) * 1000.0, 4)
            for p in self.PUNCTUATION_MARKS
        }

        # Character 3-grams (captures Turkish agglutinative morphology & sub-word rhythm)
        compact = re.sub(r"\s+", " ", lowered)
        char_ngrams = Counter(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))
        total_ngrams = sum(char_ngrams.values()) or 1

        if ngram_vocab is None:
            top_ngrams = [ng for ng, _ in char_ngrams.most_common(80)]
        else:
            top_ngrams = ngram_vocab

        ngram_freqs = {
            f"c3g::{ng}": round((char_ngrams.get(ng, 0) / total_ngrams) * 1000.0, 4)
            for ng in top_ngrams
        }

        avg_word_len = round(sum(len(w) for w in words) / total_words, 3)
        mattr = compute_mattr(words, window_size=50)

        features: Dict[str, float] = {
            "syntax::mean_sentence_len": round(mean_sent, 3),
            "syntax::avg_word_len": avg_word_len,
            "syntax::mattr": mattr,
            "syntax::burstiness": burstiness,
            **func_freqs,
            **punct_freqs,
            **ngram_freqs,
        }

        return {
            "total_words": total_words,
            "sentence_count": len(sent_lengths),
            "mean_sentence_length": round(mean_sent, 2),
            "sentence_std": round(std_sent, 2),
            "burstiness_index": burstiness,
            "mattr": mattr,
            "avg_word_length": avg_word_len,
            "raw_char_ngrams": char_ngrams,
            "features": features,
        }

    def build_author_fingerprint(
        self,
        target_name: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a statistical Burrows' Delta fingerprint (feature means & standard deviations)
        across all verified [TARGET_PRIMARY] documents and audit internal outliers.
        """
        valid_docs: List[Tuple[str, str]] = []
        global_ngrams: Counter[str] = Counter()

        for idx, doc in enumerate(documents):
            src = doc.get("source_id") or doc.get("file_name") or f"doc_{idx}"
            primary_txt = extract_target_primary_text(doc.get("content", ""))
            if len(tokenize_words(primary_txt)) >= 12:
                valid_docs.append((src, primary_txt))
                compact = re.sub(r"\s+", " ", tr_lower(primary_txt))
                global_ngrams.update(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))

        if not valid_docs:
            return {
                "target_name": target_name,
                "status": "INSUFFICIENT_CORPUS",
                "documents_analyzed": 0,
                "probabilistic_statement": "Adli Stilometri Notu: Yazarlık parmak izi çıkarmak için yeterli doğrulanmış birincil metin bulunamadı.",
            }

        # If only 1 large document exists, segment it into pseudo-blocks (>=40 words each) so feature variance is measurable
        if len(valid_docs) == 1:
            only_src, only_txt = valid_docs[0]
            paras = [p.strip() for p in re.split(r"\n\n+|(?<=[.!?])\s+", only_txt) if p.strip()]
            blocks: List[Tuple[str, str]] = []
            buf: List[str] = []
            buf_words = 0
            for p in paras:
                buf.append(p)
                buf_words += len(tokenize_words(p))
                if buf_words >= 35:
                    blocks.append((f"{only_src}#part{len(blocks) + 1}", " ".join(buf)))
                    buf, buf_words = [], 0
            if buf and buf_words >= 12:
                blocks.append((f"{only_src}#part{len(blocks) + 1}", " ".join(buf)))
            if len(blocks) >= 2:
                valid_docs = blocks

        top_ngram_vocab = [ng for ng, _ in global_ngrams.most_common(45)]
        doc_extractions = [(src, self._extract_doc_vector(txt, ngram_vocab=top_ngram_vocab)) for src, txt in valid_docs]

        feature_keys = sorted(doc_extractions[0][1]["features"].keys())
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}

        n_docs = len(doc_extractions)
        total_words_all = sum(ext["total_words"] for _, ext in doc_extractions)
        mean_doc_words = max(20.0, total_words_all / float(n_docs))

        for fk in feature_keys:
            vals = [ext["features"].get(fk, 0.0) for _, ext in doc_extractions]
            mu = sum(vals) / n_docs
            var = sum((v - mu) ** 2 for v in vals) / n_docs if n_docs > 1 else 0.0
            # Include Poisson finite-sample noise floor for per-1000 rate features so short texts don't over-penalize sparse n-grams
            if fk.startswith(("c3g::", "fw::", "punct::")):
                poisson_floor = math.sqrt(max(mu, 0.5) * (120.0 / mean_doc_words))
                sigma = max(math.sqrt(var), poisson_floor, abs(mu) * 0.35, 0.50)
            else:
                sigma = max(math.sqrt(var), abs(mu) * 0.25, 0.25)
            means[fk] = round(mu, 4)
            stds[fk] = round(sigma, 4)

        # Internal Corpus Outlier Audit (Ghostwriter / PR / Synthetic LLM Detection)
        doc_audits: List[Dict[str, Any]] = []
        outliers: List[Dict[str, Any]] = []
        mean_burstiness = round(sum(ext["burstiness_index"] for _, ext in doc_extractions) / n_docs, 3)
        mean_mattr = round(sum(ext["mattr"] for _, ext in doc_extractions) / n_docs, 3)

        for src, ext in doc_extractions:
            delta = self._burrows_delta(ext["features"], means, stds, feature_keys)
            burst = ext["burstiness_index"]
            is_synthetic_cadence = ext["sentence_count"] >= 5 and burst < 0.18
            is_style_outlier = n_docs >= 3 and delta > 1.45

            flag = "CONSISTENT_PRIMARY_VOICE"
            if is_style_outlier and is_synthetic_cadence:
                flag = "GHOSTWRITER_OR_SYNTHETIC_LLM_SUSPECT"
            elif is_style_outlier:
                flag = "STYLOMETRIC_DIVERGENCE_OUTLIER"
            elif is_synthetic_cadence:
                flag = "LOW_BURSTINESS_UNIFORM_CADENCE"

            record = {
                "source_id": src,
                "word_count": ext["total_words"],
                "burrows_delta_from_centroid": round(delta, 3),
                "burstiness_index": burst,
                "mattr": ext["mattr"],
                "audit_flag": flag,
            }
            doc_audits.append(record)
            if flag != "CONSISTENT_PRIMARY_VOICE":
                outliers.append(record)

        confidence = round(min(0.95, 0.55 + min(0.40, total_words_all / 2500.0)), 2)

        fingerprint = {
            "target_name": target_name,
            "status": "FINGERPRINT_READY",
            "documents_analyzed": n_docs,
            "total_words_analyzed": total_words_all,
            "signature_summary": {
                "mean_sentence_length": round(sum(ext["mean_sentence_length"] for _, ext in doc_extractions) / n_docs, 2),
                "mean_burstiness_index": mean_burstiness,
                "mean_mattr": mean_mattr,
                "avg_word_length": round(sum(ext["avg_word_length"] for _, ext in doc_extractions) / n_docs, 2),
            },
            "ngram_vocab": top_ngram_vocab,
            "feature_keys": feature_keys,
            "feature_means": means,
            "feature_stds": stds,
            "document_consistency_audit": doc_audits,
            "flagged_outliers": outliers,
            "confidence_score": confidence,
            "probabilistic_statement": (
                f"Adli Stilometri Parmak İzi: '{target_name}' için {total_words_all} kelimelik doğrulanmış korpustan "
                f"üslup imzası çıkarılmıştır (MATTR: {mean_mattr}, Cümle Ritmi Dalgalanması/Burstiness: {mean_burstiness}). "
                f"Korpus içi tutarlılık taramasında {len(outliers)} belgede üslup sapması işaretlenmiştir "
                f"(Güven Skoru: %{int(confidence * 100)})."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_fingerprint(fingerprint)
        return fingerprint

    def verify_questioned_text(
        self,
        questioned_text: str,
        fingerprint: Dict[str, Any],
        iterations: int = 80,
    ) -> Dict[str, Any]:
        """
        Test whether an anonymous or questioned text plausibly belongs to the target author
        using Burrows' Delta + Koppel's Monte-Carlo Feature-Subsampling consistency test.
        """
        if fingerprint.get("status") != "FINGERPRINT_READY":
            return {
                "status": "FINGERPRINT_UNAVAILABLE",
                "authorship_probability": 0.0,
                "verdict": "INSUFFICIENT_REFERENCE_CORPUS",
            }

        vocab = fingerprint["ngram_vocab"]
        feature_keys = fingerprint["feature_keys"]
        means = fingerprint["feature_means"]
        stds = fingerprint["feature_stds"]

        q_ext = self._extract_doc_vector(questioned_text, ngram_vocab=vocab)
        if q_ext.get("total_words", 0) < 8:
            return {
                "status": "QUESTIONED_TEXT_TOO_SHORT",
                "authorship_probability": 0.0,
                "verdict": "INSUFFICIENT_SAMPLE_LENGTH",
                "probabilistic_statement": "Şüpheli metin istatistiksel yazarlık testi için çok kısadır (en az 8 kelime gereklidir).",
            }

        q_feats = q_ext["features"]
        full_delta = self._burrows_delta(q_feats, means, stds, feature_keys)
        cosine_sim = self._zscore_cosine_similarity(q_feats, means, stds, feature_keys)

        # Deterministic Monte-Carlo Feature Subsampling (Koppel's verification principle):
        # In each trial, sample 50% of stylometric features and check if Delta stays within authorial acceptance band (<1.15)
        rng = random.Random(42)
        k_sub = max(10, len(feature_keys) // 2)
        sub_passes = 0
        for _ in range(iterations):
            sampled_keys = rng.sample(feature_keys, k_sub)
            sub_delta = self._burrows_delta(q_feats, means, stds, sampled_keys)
            if sub_delta <= 1.15:
                sub_passes += 1

        subsample_consistency = sub_passes / float(iterations)
        # Logistic transformation of Burrows' Delta combined with subsample consistency & cosine alignment
        delta_prob = 1.0 / (1.0 + math.exp(2.6 * (full_delta - 1.05)))
        raw_prob = (0.50 * delta_prob) + (0.35 * subsample_consistency) + (0.15 * max(0.0, cosine_sim))

        # Penalize certainty when questioned text is short (<60 words)
        length_calibration = min(1.0, q_ext["total_words"] / 60.0)
        calibrated_prob = round((raw_prob * length_calibration) + (0.50 * (1.0 - length_calibration)), 3)

        if calibrated_prob >= 0.72 and full_delta <= 1.05:
            verdict = "VERIFIED_AUTHORSHIP_PLAUSIBLE"
        elif calibrated_prob <= 0.38 or full_delta >= 1.45:
            verdict = "AUTHORSHIP_UNLIKELY_DIVERGENT_STYLE"
        else:
            verdict = "INCONCLUSIVE_MODERATE_SIMILARITY"

        return {
            "target_name": fingerprint.get("target_name"),
            "status": "VERIFICATION_COMPLETED",
            "questioned_word_count": q_ext["total_words"],
            "burrows_delta": round(full_delta, 3),
            "zscore_cosine_similarity": round(cosine_sim, 3),
            "monte_carlo_consistency_pct": round(subsample_consistency * 100.0, 1),
            "questioned_burstiness": q_ext["burstiness_index"],
            "authorship_probability": calibrated_prob,
            "authorship_probability_pct": f"{int(round(calibrated_prob * 100))}%",
            "verdict": verdict,
            "probabilistic_statement": (
                f"Adli Yazarlık Doğrulaması: İncelenen metnin '{fingerprint.get('target_name')}' üslup parmak iziyle "
                f"Burrows' Delta mesafesi `{round(full_delta, 3)}`, Monte-Carlo alt-örneklem tutarlılığı "
                f"`%{round(subsample_consistency * 100.0, 1)}` olarak ölçülmüştür. "
                f"Kalibre edilmiş yazarlık uyumu olasılığı: **%{int(round(calibrated_prob * 100))}** (`{verdict}`)."
            ),
        }

    def _burrows_delta(
        self,
        doc_features: Dict[str, float],
        means: Dict[str, float],
        stds: Dict[str, float],
        keys: List[str],
    ) -> float:
        """Compute Classic Burrows' Delta (mean absolute z-score difference across feature set)."""
        if not keys:
            return 2.0
        total = 0.0
        for k in keys:
            z = abs(doc_features.get(k, 0.0) - means.get(k, 0.0)) / (stds.get(k, 1.0) or 1.0)
            total += min(z, 5.0)
        return total / len(keys)

    def _zscore_cosine_similarity(
        self,
        doc_features: Dict[str, float],
        means: Dict[str, float],
        stds: Dict[str, float],
        keys: List[str],
    ) -> float:
        """Compute normalized proximity in z-score space (1 / (1 + L2_z_mean))."""
        if not keys:
            return 0.0
        sq = sum(
            min(5.0, abs(doc_features.get(k, 0.0) - means.get(k, 0.0)) / (stds.get(k, 1.0) or 1.0)) ** 2
            for k in keys
        )
        rms = math.sqrt(sq / len(keys))
        return 1.0 / (1.0 + rms)

    def _persist_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        """Persist stylometry JSON fingerprint and Markdown forensic audit report."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "stylometry_fingerprint.json", "w", encoding="utf-8") as f:
                json.dump(fingerprint, f, indent=2, ensure_ascii=False)

        report_dir = self.vault_dir or self.output_dir
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            sig = fingerprint.get("signature_summary", {})
            lines = [
                f"# 🖋️ Forensic Stylometry & Authorship Profile: {fingerprint.get('target_name')}",
                f"*Documents Analyzed:* `{fingerprint.get('documents_analyzed')}` | *Words:* `{fingerprint.get('total_words_analyzed')}`\n",
                f"> {fingerprint.get('probabilistic_statement')}\n",
                "## 1. Core Stylometric Signature",
                f"- **Mean Sentence Length:** `{sig.get('mean_sentence_length')} words`",
                f"- **Sentence Burstiness (Human Rhythm Variance):** `{sig.get('mean_burstiness_index')}`",
                f"- **Moving-Average Lexical Diversity (MATTR):** `{sig.get('mean_mattr')}`",
                f"- **Mean Word Length:** `{sig.get('avg_word_length')} chars`\n",
                "## 2. Internal Corpus Consistency & Ghostwriter/LLM Audit",
            ]
            for doc in fingerprint.get("document_consistency_audit", [])[:15]:
                lines.append(
                    f"- `{doc['source_id']}` ({doc['word_count']} words) — "
                    f"Delta: `{doc['burrows_delta_from_centroid']}`, Burstiness: `{doc['burstiness_index']}` -> **{doc['audit_flag']}**"
                )
            (report_dir / "stylometry_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

