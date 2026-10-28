"""
Linguistic and behavioral feature extraction for who??.
Analyzes pronoun density, lexical diversity, epistemic stance, and emotional cadence.
Corresponds to Layer 4 (Behavioral Signal Memory).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("who.learn.features")


class FeatureExtractor:
    """Extracts quantitative linguistic, cognitive, and rhetorical markers from target texts."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def extract_features(self, texts: List[str]) -> Dict[str, Any]:
        """Compute comprehensive linguistic and psychological metrics across a corpus."""
        combined_text = " ".join(texts)
        if not combined_text.strip():
            return self._empty_feature_set()

        words = re.findall(r"\b[a-zA-Z']+\b", combined_text.lower())
        total_words = len(words)
        if total_words == 0:
            return self._empty_feature_set()

        sentences = [s.strip() for s in re.split(r"[.!?]+", combined_text) if s.strip()]
        total_sentences = len(sentences) or 1

        # 1. Pronoun analysis
        i_pronouns = {"i", "me", "my", "mine", "myself"}
        we_pronouns = {"we", "us", "our", "ours", "ourselves"}
        you_pronouns = {"you", "your", "yours", "yourself", "yourselves"}
        they_pronouns = {"they", "them", "their", "theirs", "themselves", "he", "him", "his", "she", "her", "hers"}

        i_count = sum(1 for w in words if w in i_pronouns)
        we_count = sum(1 for w in words if w in we_pronouns)
        you_count = sum(1 for w in words if w in you_pronouns)
        they_count = sum(1 for w in words if w in they_pronouns)

        # 2. Epistemic stance words
        certainty_lexicon = {"definitely", "certainly", "undoubtedly", "always", "never", "obvious", "clear", "absolute", "guaranteed"}
        tentative_lexicon = {"likely", "plausible", "perhaps", "maybe", "possible", "suggests", "probable", "might", "could", "seems"}
        causal_lexicon = {"because", "therefore", "thus", "hence", "consequently", "implies", "since", "results"}

        certainty_count = sum(1 for w in words if w in certainty_lexicon)
        tentative_count = sum(1 for w in words if w in tentative_lexicon)
        causal_count = sum(1 for w in words if w in causal_lexicon)

        # 3. Emotional valence indicators (lexical approximation)
        positive_lexicon = {"great", "excellent", "superb", "constructive", "insightful", "promising", "breakthrough", "success", "effective", "valuable"}
        critical_lexicon = {"flawed", "erroneous", "inefficient", "risk", "failure", "deceptive", "critical", "vulnerable", "concern", "problem"}

        pos_count = sum(1 for w in words if w in positive_lexicon)
        crit_count = sum(1 for w in words if w in critical_lexicon)

        # 4. Lexical diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / total_words

        # 5. Rhetorical syntax metrics
        avg_sentence_len = total_words / total_sentences
        question_count = combined_text.count("?")
        exclamation_count = combined_text.count("!")

        # 6. Top keywords (excluding common stop words)
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "that", "this", "these", "those", "it", "its", "as", "by", "from", "about"
        }
        word_freq: Dict[str, int] = {}
        for w in words:
            if len(w) > 3 and w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        features = {
            "corpus_stats": {
                "total_words": total_words,
                "total_sentences": total_sentences,
                "unique_words": unique_words,
                "type_token_ratio": round(ttr, 3),
                "avg_sentence_length": round(avg_sentence_len, 2),
            },
            "pronoun_ratios": {
                "first_person_singular_pct": round((i_count / total_words) * 100, 2),
                "first_person_plural_pct": round((we_count / total_words) * 100, 2),
                "second_person_pct": round((you_count / total_words) * 100, 2),
                "third_person_pct": round((they_count / total_words) * 100, 2),
                "i_to_we_ratio": round(i_count / (we_count or 1), 2),
            },
            "epistemic_markers": {
                "certainty_density": round((certainty_count / total_words) * 1000, 2),
                "tentative_density": round((tentative_count / total_words) * 1000, 2),
                "causal_density": round((causal_count / total_words) * 1000, 2),
                "epistemic_calibration_ratio": round(tentative_count / (certainty_count or 1), 2),
            },
            "emotional_cadence": {
                "positive_density": round((pos_count / total_words) * 1000, 2),
                "critical_density": round((crit_count / total_words) * 1000, 2),
                "sentiment_ratio": round(pos_count / (crit_count or 1), 2),
            },
            "rhetorical_syntax": {
                "inquisitiveness_ratio": round((question_count / total_sentences) * 100, 2),
                "intensity_ratio": round((exclamation_count / total_sentences) * 100, 2),
            },
            "top_characteristic_terms": [k for k, _ in sorted_keywords[:10]],
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / "features.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved behavioral signal features to {out_file}")

        return features

    def _empty_feature_set(self) -> Dict[str, Any]:
        return {
            "corpus_stats": {"total_words": 0, "total_sentences": 0, "type_token_ratio": 0.0},
            "pronoun_ratios": {"first_person_singular_pct": 0.0, "i_to_we_ratio": 0.0},
            "epistemic_markers": {"epistemic_calibration_ratio": 1.0},
            "top_characteristic_terms": [],
        }
