"""
Linguistic, psychometric, and behavioral feature extraction.
Computes metrics including negation-aware semantic psychometrics, Bayesian Beta-Binomial
credible intervals, Walter Mischel's Context-Conditional CAPS Matrix, Stated vs. Revealed
Preference divergence, and Epistemic Manipulation indices.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from who.utils.logger import get_logger

logger = get_logger("who.learn.features")

MIN_WORDS_FOR_INDICATORS = 20
MIN_WORDS_RECOMMENDED = 100

_TURKISH_PRIVATIVE_SUFFIXES: Tuple[str, ...] = (
    "siz", "sız", "suz", "süz",
    "sizlik", "sızlık", "suzluk", "süzlük",
    "sizce", "sızca", "sizdir", "sızdır",
    "suzca", "süzce", "suzluğu", "süzlüğü",
)

NEGATION_TOKENS: Set[str] = {
    "not", "never", "no", "neither", "nor", "without", "hardly",
    "cannot", "cant", "can't", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "wont", "won't", "wouldnt", "wouldn't", "shouldnt", "shouldn't", "couldnt", "couldn't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "havent", "haven't", "hasnt", "hasn't", "hadnt", "hadn't", "mustnt", "mustn't",
    "neednt", "needn't", "aint", "ain't",
    "değil", "degil", "yok", "yoktur", "asla", "hiç", "hic", "hiçbir", "hicbir",
    "olmayan", "olmaz", "imkansız", "imkansiz", "hayır", "hayir",
    "etmem", "etmeyiz", "vermem", "vermeyiz", "olmam", "olmayız", "olmayiz",
    "istemem", "istemeyiz",
}

TURKISH_NEG_SUFFIXES: Tuple[str, ...] = (
    "meme", "mama", "mez", "maz", "miyor", "mıyor", "muyor", "müyor",
    "medi", "madı", "medik", "madık", "meyecek", "mayacak",
    "meyeceğim", "mayacağım", "meyeceğiz", "mayacağız",
    "miyorum", "mıyorum", "miyoruz", "mıyoruz",
    "medim", "madım",
    "etmez", "kalmaz", "korunmaz",
)

_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;:,\n]+|\b(?:but|however|yet|though|although|whereas|ama|fakat|ancak|lakin)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b")
_CLAUSE_BARRIERS: Set[str] = {
    "but", "however", "yet", "though", "although", "whereas", "unless", "until",
    "ama", "fakat", "ancak", "lakin",
}
_PRE_NEG_BRIDGES: Set[str] = {
    "a", "an", "the", "any", "all", "this", "that", "these", "those", "such",
    "so", "very", "too", "quite", "more", "most", "much", "at", "of", "to", "in", "for",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might",
    "have", "has", "had", "i'm", "we're", "it's", "that's", "there's",
    "afraid", "scared", "feel", "seem", "look", "sound", "get", "go", "come",
    "take", "make", "give",
    "bir", "hiçbir", "hicbir", "herhangi", "zaman", "kadar", "çok", "cok",
    "bu", "şu", "su", "o", "pek", "fazla",
}
_POST_NEG_BRIDGES: Set[str] = {
    "is", "are", "was", "were", "be", "been", "being",
    "at", "all", "anymore", "either", "too",
    "da", "de", "dahi", "bile", "hiç", "hic", "artık", "artik",
    "bir", "durum", "halinde", "olarak", "hareket", "verici", "mümkün", "mumkun", "uygun",
}

FIRST_PERSON_SINGULAR: Set[str] = {
    "i", "me", "my", "mine", "myself", "ben", "bana", "beni", "benim", "kendim"
}

FIRST_PERSON_PLURAL: Set[str] = {
    "we", "us", "our", "ours", "ourselves", "biz", "bize", "bizi", "bizim", "kendimiz"
}

CERTAINTY_MARKERS: Set[str] = {
    "definitely", "certainly", "undoubtedly", "always", "never", "obvious", "clear", "absolute", "guaranteed",
    "kesinlikle", "asla", "şüphesiz", "suphesiz", "mutlaka", "apaçık", "apack", "tartışmasız", "tartismasiz"
}

TENTATIVE_MARKERS: Set[str] = {
    "likely", "plausible", "perhaps", "maybe", "possible", "suggests", "probable", "might", "could", "seems",
    "olası", "olasi", "muhtemel", "belki", "sanırım", "sanirim", "görünüyor", "gorunuyor", "işaret", "varsayım"
}

CAPS_CONTEXT_MARKERS: Dict[str, Set[str]] = {
    "FORMAL_ACADEMIC": {
        "theory", "theorem", "empirical", "research", "analysis", "framework", "methodology", "evidence", "protocol",
        "teori", "akademik", "bilimsel", "analiz", "yöntem", "kavram", "makale", "literatür", "epistemik", "sistem",
        "teorik", "metodoloji", "hipotez",
    },
    "HIGH_STAKES_CRISIS": {
        "crisis", "threat", "attack", "emergency", "collapse", "danger", "hostile", "ultimatum", "deadline", "risk", "urgent",
        "kriz", "tehdit", "saldırı", "saldiri", "acil", "çöküş", "tehlike", "baskı", "baski", "dayatma", "risk", "kaos",
    },
    "STRATEGIC_NEGOTIATION": {
        "negotiate", "deal", "proposal", "counter", "leverage", "veto", "concession", "agreement", "alliance", "terms", "contract",
        "offer", "price", "buyer", "seller", "bargain", "budget",
        "müzakere", "muzakere", "teklif", "pazarlık", "pazarlik", "fiyat", "uzlaşı", "uzlasi", "veto", "taviz", "anlaşma", "mutabakat", "kontrat",
    },
    "INTERPERSONAL_DIALOGUE": {
        "friend", "hello", "hi", "thank", "thanks", "please", "sorry", "welcome", "conversation", "personal", "feel", "together", "chat", "hey",
        "merhaba", "dost", "teşekkür", "teşekkürler", "rica", "sohbet", "samimi", "birlikte", "kişisel", "insan", "gönül", "selam",
    },
}

STATED_COOPERATIVE_MARKERS: Set[str] = {
    "consensus", "democratic", "open", "shared", "cooperate", "inclusive", "compromise", "trust", "dialogue", "tolerance",
    "uzlaşı", "demokratik", "açık", "ortak", "işbirliği", "kapsayıcı", "taviz", "güven", "istişare", "diyalog", "hoşgörü", "mutabakat",
}

STATED_DEFENSIVE_MARKERS: Set[str] = {
    "control", "veto", "unilateral", "reject", "strict", "authority", "enforce", "autonomous", "independent",
    "kontrol", "veto", "tek taraflı", "red", "katı", "otorite", "bağımsız", "özerk", "merkezi",
}

REVEALED_COOPERATIVE_ACTION_MARKERS: Set[str] = {
    "signed", "agreed", "partnered", "joined", "approved", "compromised", "collaborated",
    "imzaladı", "imzaladık", "kabul etti", "onayladı", "uzlaştı", "katıldı", "destekledi",
}

REVEALED_AUTONOMOUS_EXIT_MARKERS: Set[str] = {
    "rejected", "vetoed", "walked", "terminated", "blocked", "withdrew", "canceled", "refused",
    "reddetti", "reddettik", "çekildi", "veto", "iptal", "sonlandırdı", "sonlandırdık", "bloke",
}


class FeatureExtractor:
    """Extracts quantitative linguistic, cognitive, Bayesian psychometric, and CAPS behavioral markers."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    @staticmethod
    def _is_negation_token(word: str) -> bool:
        if word in NEGATION_TOKENS:
            return True
        if len(word) > 5 and word.endswith(TURKISH_NEG_SUFFIXES):
            return True
        return False

    @staticmethod
    def _split_clauses(text: str) -> List[List[str]]:
        """Split text on clause punctuation and contrast conjunctions, then tokenize each clause."""
        clauses: List[List[str]] = []
        for piece in _CLAUSE_SPLIT_RE.split(text.lower()):
            toks = _TOKEN_RE.findall(piece)
            if toks:
                clauses.append(toks)
        return clauses

    @classmethod
    def _is_negated_context(
        cls,
        words: List[str],
        idx: int,
        window_before: int = 4,
        window_after: int = 3,
    ) -> bool:
        """
        Check whether token at `idx` is governed by a negation marker via clause-bounded
        directional bridge walking (e.g. "don't want risk", "never afraid of risk", "fiyat makul değil").
        """
        for step in range(1, window_before + 1):
            j = idx - step
            if j < 0:
                break
            w = words[j]
            if w in _CLAUSE_BARRIERS:
                break
            if w == "not" and j + 1 < len(words) and words[j + 1] in {"only", "just", "merely"}:
                break
            if (
                w == "no"
                and j + 2 < len(words)
                and words[j + 1] in {"less", "more"}
                and words[j + 2] == "than"
            ):
                break
            if cls._is_negation_token(w):
                return True
            if w in _PRE_NEG_BRIDGES or (len(w) > 4 and w.endswith("ly")):
                continue
            break

        for step in range(1, window_after + 1):
            j = idx + step
            if j >= len(words):
                break
            w = words[j]
            if w in _CLAUSE_BARRIERS:
                break
            if cls._is_negation_token(w):
                return True
            if w in _POST_NEG_BRIDGES or (len(w) > 4 and w.endswith("ly")):
                continue
            break

        return False

    @classmethod
    def _matches_lexicon(cls, word: str, lexicon: Set[str]) -> bool:
        """
        Match `word` against `lexicon` using exact lookup plus bounded, privative-safe
        English inflection and Turkish agglutinative suffix normalization.
        """
        if word in lexicon:
            return True
        if cls._is_negation_token(word) or len(word) < 5:
            return False

        # Never map privative/negated adjectives ("adaletsiz", "fearless") to positive roots
        if word.endswith(_TURKISH_PRIVATIVE_SUFFIXES) or (len(word) > 6 and word.endswith("less")):
            return False

        # Safe English inflection normalization
        candidates: List[str] = []
        if word.endswith(("ies", "ied")) and len(word) > 4:
            candidates.append(word[:-3] + "y")
        if word.endswith("ing") and len(word) > 6:
            stem = word[:-3]
            candidates.append(stem)
            candidates.append(stem + "e")
        if word.endswith("ed") and len(word) > 5:
            candidates.append(word[:-2])
            candidates.append(word[:-1])
        if word.endswith("es") and len(word) > 5:
            candidates.append(word[:-2])
        if word.endswith("s") and not word.endswith("ss") and len(word) > 5:
            candidates.append(word[:-1])
        if word.endswith("y") and len(word) > 5:
            candidates.append(word[:-1])
        if word.endswith("ness") and len(word) > 7:
            candidates.append(word[:-4])
        if word.endswith("ly") and len(word) > 6:
            ly_base = word[:-2]
            if len(ly_base) >= 5:
                candidates.append(ly_base)

        for base in candidates:
            if len(base) >= 4 and base in lexicon:
                return True

        # Safe Turkish agglutinative suffix matching on content roots (len >= 5)
        if not word.endswith(TURKISH_NEG_SUFFIXES):
            for root in lexicon:
                if len(root) >= 5 and not cls._is_negation_token(root) and word.startswith(root):
                    tail = word[len(root):]
                    if 1 <= len(tail) <= 7 and not tail.startswith(("siz", "sız", "suz", "süz")):
                        return True

        return False

    @classmethod
    def _count_negation_aware(
        cls,
        words: List[str],
        primary_lexicon: Set[str],
        opposite_lexicon: Optional[Set[str]] = None,
    ) -> Tuple[int, int, int]:
        """
        Count hits for `primary_lexicon` and `opposite_lexicon` with syntactic negation reversal.
        Returns (effective_primary_hits, effective_opposite_hits, negated_reversals_count).
        """
        opp_lex = opposite_lexicon or set()
        primary_hits = 0
        opposite_hits = 0
        reversals = 0

        for idx, w in enumerate(words):
            if cls._matches_lexicon(w, primary_lexicon):
                if not cls._is_negation_token(w) and cls._is_negated_context(words, idx):
                    opposite_hits += 1
                    reversals += 1
                else:
                    primary_hits += 1
            elif cls._matches_lexicon(w, opp_lex):
                if not cls._is_negation_token(w) and cls._is_negated_context(words, idx):
                    primary_hits += 1
                    reversals += 1
                else:
                    opposite_hits += 1

        return primary_hits, opposite_hits, reversals

    @classmethod
    def _count_negation_aware_clauses(
        cls,
        clauses: Sequence[List[str]],
        primary_lexicon: Set[str],
        opposite_lexicon: Optional[Set[str]] = None,
    ) -> Tuple[int, int, int]:
        """Run negation-aware counting strictly bounded within each clause."""
        p_tot = 0
        o_tot = 0
        r_tot = 0
        for clause_words in clauses:
            p, o, r = cls._count_negation_aware(clause_words, primary_lexicon, opposite_lexicon)
            p_tot += p
            o_tot += o
            r_tot += r
        return p_tot, o_tot, r_tot

    @staticmethod
    def _compute_bayesian_credible_interval(
        positive_hits: float,
        opposing_hits: float,
        total_words: int,
        prior_alpha: float = 1.5,
        prior_beta: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Compute Bayesian Beta-Binomial posterior mean and 95% Credible Interval.
        Incorporates corpus sample size scaling so small corpora yield wide uncertainty intervals
        and large corpora yield tight, high-precision intervals.
        """
        effective_evidence_scale = max(1.0, math.log1p(total_words / 50.0))
        alpha = prior_alpha + (positive_hits * effective_evidence_scale)
        beta = prior_beta + (opposing_hits * effective_evidence_scale)
        total_ab = alpha + beta

        mean = alpha / total_ab
        variance = (alpha * beta) / ((total_ab ** 2) * (total_ab + 1.0))
        std_dev = math.sqrt(max(1e-9, variance))

        margin_95 = min(0.45, 1.96 * std_dev)
        ci_low = round(max(0.0, mean - margin_95), 3)
        ci_high = round(min(1.0, mean + margin_95), 3)
        posterior_precision = round(max(0.0, 1.0 - (margin_95 * 2.0)), 3)

        if total_words >= 800 and (positive_hits + opposing_hits) >= 8:
            density_tier = "HIGH_EVIDENCE_DENSITY"
        elif total_words >= 200 and (positive_hits + opposing_hits) >= 3:
            density_tier = "MODERATE_EVIDENCE_DENSITY"
        else:
            density_tier = "LOW_SAMPLE_WIDE_PRIOR"

        return {
            "posterior_mean": round(mean, 3),
            "credible_interval_95": [ci_low, ci_high],
            "uncertainty_margin": round(margin_95, 3),
            "posterior_precision": posterior_precision,
            "evidence_density_tier": density_tier,
        }

    def extract_features(
        self,
        texts: List[str],
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive linguistic, Bayesian psychometric, CAPS, and Revealed Preference metrics."""
        combined_text = " ".join(texts)
        if not combined_text.strip():
            return self._empty_feature_set()

        words = _TOKEN_RE.findall(combined_text.lower())
        total_words = len(words)
        if total_words == 0:
            return self._empty_feature_set()

        clauses = self._split_clauses(combined_text) or [words]
        sentences = [s.strip() for s in re.split(r"[.!?]+", combined_text) if s.strip()]
        total_sentences = len(sentences) or 1

        # 1. Pronoun analysis (Turkish + English)
        you_pronouns = {"you", "your", "yours", "yourself", "yourselves", "sen", "siz", "sana", "size", "senin", "sizin"}
        they_pronouns = {
            "they", "them", "their", "theirs", "themselves", "he", "him", "his", "she", "her", "hers",
            "onlar", "onlara", "onlari", "onları", "onlarin", "onların", "o", "ona", "onu", "onun"
        }

        i_count = sum(1 for w in words if w in FIRST_PERSON_SINGULAR)
        we_count = sum(1 for w in words if w in FIRST_PERSON_PLURAL)
        you_count = sum(1 for w in words if w in you_pronouns)
        they_count = sum(1 for w in words if w in they_pronouns)

        # 2. Epistemic stance words (Clause-bounded negation-aware)
        causal_lexicon = {
            "because", "therefore", "thus", "hence", "consequently", "implies", "since", "results",
            "çünkü", "cunku", "dolayısıyla", "dolayisiyla", "böylece", "boylece", "nedeniyle", "sonucunda"
        }

        certainty_count, tentative_count, epistemic_reversals = self._count_negation_aware_clauses(
            clauses, CERTAINTY_MARKERS, TENTATIVE_MARKERS
        )
        causal_count = sum(1 for w in words if w in causal_lexicon)

        # 3. Emotional valence indicators (Clause-bounded negation-aware)
        positive_lexicon = {
            "great", "excellent", "superb", "constructive", "insightful", "promising", "breakthrough", "success", "effective", "valuable",
            "harika", "mükemmel", "mukemmel", "başarılı", "basarili", "faydalı", "faydali", "önemli", "onemli", "değerli", "degerli"
        }
        critical_lexicon = {
            "flawed", "erroneous", "inefficient", "risk", "failure", "deceptive", "critical", "vulnerable", "concern", "problem",
            "hatalı", "hatali", "yanlış", "yanlis", "başarısız", "basarisiz", "risk", "tehlike", "sorun", "kriz", "çelişki", "celiski"
        }

        pos_count, crit_count, valence_reversals = self._count_negation_aware_clauses(
            clauses, positive_lexicon, critical_lexicon
        )

        # 4. Lexical diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / total_words

        # 5. Rhetorical syntax metrics
        avg_sentence_len = total_words / total_sentences
        sentence_lengths = [len(_TOKEN_RE.findall(s)) for s in sentences]
        sent_len_variance = (
            sum((sl - avg_sentence_len) ** 2 for sl in sentence_lengths) / len(sentence_lengths)
            if sentence_lengths else 0.0
        )
        question_count = combined_text.count("?")
        exclamation_count = combined_text.count("!")

        # 6. Top characteristic keywords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "that", "this", "these", "those", "it", "its", "as", "by", "from", "about",
            "bir", "ve", "veya", "ile", "de", "da", "bu", "şu", "su", "o", "için", "icin", "gibi",
            "kadar", "daha", "en", "çok", "cok", "her", "tüm", "tum", "ise", "ama", "fakat", "lakin"
        }
        word_freq: Dict[str, int] = {}
        for w in words:
            if len(w) > 3 and w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        # 7. Epistemic Gaslighting & Manipulation Index
        manipulation_metrics = self._calculate_manipulation_index(combined_text, words, total_words)

        # 8. Optional Cognitive Chronomics Integration
        chronomics_data: Dict[str, Any] = {}
        if documents:
            try:
                from who.learn.chronomics import CognitiveChronomics
                cc = CognitiveChronomics(output_dir=self.output_dir)
                chronomics_data = cc.analyze_circadian_drift(documents)
            except Exception as e:
                logger.debug(f"Chronomics execution note: {e}")

        # 9. Extended Psychometrics (with Clause-Bounded Negation & Bayesian 95% Credible Intervals)
        machiavellianism_data = self._calculate_machiavellianism(words, total_words, clauses=clauses)
        closure_data = self._calculate_need_for_closure(words, total_words, clauses=clauses)
        loss_aversion_data = self._calculate_loss_aversion(words, total_words, clauses=clauses)
        volatility_data = self._calculate_affective_volatility(sentences, words, total_words, clauses=clauses)
        moral_foundations_data = self._calculate_moral_foundations(words, total_words, clauses=clauses)

        # 10. Context-Conditional CAPS Matrix (Walter Mischel If-Then Behavioral Signatures)
        caps_matrix = self._calculate_caps_contextual_matrix(sentences)

        # 11. Stated vs. Revealed Preference Divergence Analysis
        preference_divergence = self._calculate_stated_vs_revealed_preference(sentences)

        # 12. Conditional Commitments & Sacred-Value Red-Line Guard Extraction
        conditional_commitments = self._extract_conditional_commitments(sentences)

        features = {
            "measurement_methodology": {
                "signal_type": "LEXICAL_BEHAVIORAL_INDICATOR",
                "clinical_diagnostic_validity": False,
                "corpus_word_count": total_words,
                "min_words_for_indicators": MIN_WORDS_FOR_INDICATORS,
                "min_words_recommended": MIN_WORDS_RECOMMENDED,
                "corpus_sufficiency": (
                    "SUFFICIENT" if total_words >= MIN_WORDS_RECOMMENDED
                    else ("LOW_VOLUME" if total_words >= MIN_WORDS_FOR_INDICATORS else "INSUFFICIENT_DATA")
                ),
                "disclaimer": (
                    "Lexicon-based behavioral indicators measure surface linguistic patterns, not clinical "
                    "psychological traits. Interpret scores alongside corpus size and 95% credible intervals."
                ),
            },
            "corpus_stats": {
                "total_words": total_words,
                "total_sentences": total_sentences,
                "unique_words": unique_words,
                "type_token_ratio": round(ttr, 3),
                "avg_sentence_length": round(avg_sentence_len, 2),
                "sentence_length_std": round(math.sqrt(sent_len_variance), 2),
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
                "certainty_per_1k": round((certainty_count / total_words) * 1000, 2),
                "tentative_per_1k": round((tentative_count / total_words) * 1000, 2),
                "causal_density": round((causal_count / total_words) * 1000, 2),
                "epistemic_calibration_ratio": round(tentative_count / (certainty_count or 1), 2),
                "negated_reversals_detected": epistemic_reversals + valence_reversals,
            },
            "epistemic_manipulation_index": manipulation_metrics,
            "machiavellianism_index": machiavellianism_data,
            "need_for_closure_index": closure_data,
            "loss_aversion_index": loss_aversion_data,
            "affective_volatility_index": volatility_data,
            "moral_foundations_profile": moral_foundations_data,
            "caps_contextual_matrix": caps_matrix,
            "stated_vs_revealed_preference": preference_divergence,
            "conditional_commitments": conditional_commitments,
            "emotional_cadence": {
                "positive_density": round((pos_count / total_words) * 1000, 2),
                "critical_density": round((crit_count / total_words) * 1000, 2),
                "sentiment_ratio": round(pos_count / (crit_count or 1), 2),
            },
            "rhetorical_syntax": {
                "inquisitiveness_ratio": round((question_count / total_sentences) * 100, 2),
                "intensity_ratio": round((exclamation_count / total_sentences) * 100, 2),
            },
            "chronomics": chronomics_data,
            "top_characteristic_terms": [k for k, _ in sorted_keywords[:10]],
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / "features.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2, default=str)
            logger.info(f"Saved behavioral signal features to {out_file}")

        return features

    def _calculate_manipulation_index(
        self,
        full_text: str,
        words: List[str],
        total_words: int,
    ) -> Dict[str, Any]:
        """Quantify the Epistemic Manipulation & Gaslighting Index with Bayesian Credible Intervals."""
        lower_text = full_text.lower()

        dogmatic_terms = [
            "kesinlikle", "asla", "şüphesiz", "suphesiz", "herkes biliyor ki", "herkes bilir ki",
            "apaçık ortada", "apack ortada", "tartışmasız", "tartismasiz", "hiç kimse inkar edemez",
            "hic kimse inkar edemez", "tek gerçek", "tek gercek", "mutlaka", "başka yolu yok",
            "obviously", "undoubtedly", "everyone knows that", "unquestionable", "indisputable",
            "absolute truth", "beyond question", "without any doubt", "purely", "guaranteed"
        ]

        evidence_terms = [
            "veri gösteriyor ki", "veri gosteriyor ki", "kanıtlar ışığında", "kanitlar isiginda",
            "araştırmalara göre", "arastirmalara gore", "gözlemlerle uyumlu", "gozlemlerle uyumlu",
            "olasılıkla", "olasilikla", "plausibly", "belki", "varsayım altında", "varsayim altinda",
            "evidence suggests", "data indicates", "according to findings", "empirically",
            "statistically", "probabilistically", "based on observed", "tentatively"
        ]

        dogmatic_matches = sum(lower_text.count(dt) for dt in dogmatic_terms)
        evidence_matches = sum(lower_text.count(et) for et in evidence_terms)

        if dogmatic_matches == 0 or evidence_matches == 0:
            single_dogmatic = {"kesinlikle", "asla", "şüphesiz", "mutlaka", "definitely", "certainly"}
            single_evidence = {"olası", "muhtemel", "belki", "plausible", "likely", "suggests"}
            d_hits, e_hits, _ = self._count_negation_aware(words, single_dogmatic, single_evidence)
            dogmatic_matches = max(dogmatic_matches, d_hits)
            evidence_matches = max(evidence_matches, e_hits)

        dogmatic_density = round((dogmatic_matches / total_words) * 1000, 2)
        evidence_density = round((evidence_matches / total_words) * 1000, 2)

        raw_index = (dogmatic_matches * 1.5) / (dogmatic_matches + evidence_matches + 2.0)
        manipulation_index = round(min(1.0, max(0.0, raw_index)), 2)

        bayes = self._compute_bayesian_credible_interval(dogmatic_matches * 1.5, evidence_matches + 0.5, total_words)
        hits_total = dogmatic_matches + evidence_matches
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)

        if not sufficient_signal:
            risk_tier = "INSUFFICIENT_DATA"
            confidence = round(min(0.35, total_words / 200.0), 2)
            prob_stmt = (
                f"Epistemic Manipulation Index (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if manipulation_index >= 0.70:
                risk_tier = "HIGH (HIGH_MANIPULATION_RISK)"
                risk_desc = "Pronounced reliance on unsubstantiated, absolutist framing."
            elif manipulation_index >= 0.40:
                risk_tier = "MODERATE (MODERATE_DOGMATIC_TENDENCY)"
                risk_desc = "Occasional absolutist assertions balanced by rational epistemic hedging."
            else:
                risk_tier = "LOW / CALIBRATED (EMPIRICALLY_CALIBRATED)"
                risk_desc = "Observed statements ground claims with probabilistic and evidence-based caution."

            confidence = min(0.95, 0.60 + (total_words / 1500.0) * 0.3)
            prob_stmt = (
                f"Epistemic Manipulation Index (Lexical Indicator): Absolutist unsubstantiated framing tendency estimated at "
                f"{int(manipulation_index * 100)}% (±{int(bayes['uncertainty_margin']*100)}%) ({risk_desc} Confidence: {int(confidence * 100)}%)."
            )

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "score": manipulation_index,
            "estimated_score": manipulation_index,
            "score_pct": f"{int(manipulation_index * 100)}%",
            "credible_interval_95": bayes["credible_interval_95"],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "evidence_density_tier": bayes["evidence_density_tier"],
            "dogmatic_density_per_k": dogmatic_density,
            "evidence_density_per_k": evidence_density,
            "risk_tier": risk_tier,
            "confidence_score": round(confidence, 2),
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_machiavellianism(
        self,
        words: List[str],
        total_words: int,
        clauses: Optional[Sequence[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Quantify Machiavellianism with negation reversal and Bayesian credible intervals."""
        mach_lexicon = {
            "manipulate", "exploitation", "leverage", "advantage", "subordinate", "opportunism", "deceit",
            "ruthless", "dominate", "maneuver", "outmaneuver", "instrumental", "pragmatic", "utility",
            "firm", "lowest", "cheapest", "overpriced", "bluff", "squeeze", "margin", "resell",
            "insist", "demand", "strictly",
            "kullanmak", "fayda", "menfaat", "çıkar", "cikar", "koz", "üstünlük", "ustunluk",
            "sömürü", "somuru", "araçsallaştırma", "baskınlık", "zaaf", "zaafı", "fırsatçılık",
            "kâr", "pazarlık", "pazarlik", "baskı", "baski",
        }
        coop_lexicon = {
            "cooperate", "mutual", "altruism", "empathy", "ethics", "integrity", "solidarity", "compassion",
            "fair", "reasonable", "halfway", "split", "share", "agree", "agreed", "appreciate",
            "thanks", "thank", "please", "help", "honest", "together",
            "işbirliği", "isbirligi", "dayanışma", "dayanisma", "empati", "ahlak", "dürüstlük", "durustluk",
            "karşılıklı", "karsilikli", "güven", "guven", "adalet", "hakkaniyet", "şefkat", "sefkat",
            "makul", "anlaşalım", "anlasalim", "teşekkür", "tesekkur", "rica", "birlikte", "adil",
        }

        if clauses is not None:
            m_count, c_count, reversals = self._count_negation_aware_clauses(clauses, mach_lexicon, coop_lexicon)
        else:
            m_count, c_count, reversals = self._count_negation_aware(words, mach_lexicon, coop_lexicon)

        raw_score = (m_count * 2.2) / (m_count + c_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)
        bayes = self._compute_bayesian_credible_interval(m_count * 2.2, c_count + 1.5, total_words)
        hits_total = m_count + c_count
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)

        if not sufficient_signal:
            tier = "INSUFFICIENT_DATA"
            prob_stmt = (
                f"Machiavellianism & Strategic Utility (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if score >= 0.55:
                tier = "HIGH / STRATEGIC_MACHIAVELLIAN"
                desc = "High pragmatic utility and strategic leverage framing, prioritizing power dynamics over moral sentiment."
            elif score >= 0.25:
                tier = "BALANCED / BALANCED_PRAGMATIST"
                desc = "Open to cooperation while safeguarding strategic interests and institutional rationality."
            else:
                tier = "LOW / COOPERATIVE_ALTRUIST"
                desc = "High empathy, ethical commitment, and collective solidarity framing predominate."
            prob_stmt = f"Machiavellianism & Strategic Utility (Lexical Indicator): {int(score * 100)}% (95% CI: {bayes['credible_interval_95']}) ({desc})."

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "score": score,
            "estimated_score": score,
            "score_pct": f"{int(score * 100)}%",
            "credible_interval_95": bayes["credible_interval_95"],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "evidence_density_tier": bayes["evidence_density_tier"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "tier": tier,
            "machiavellian_hits": m_count,
            "cooperation_hits": c_count,
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_need_for_closure(
        self,
        words: List[str],
        total_words: int,
        clauses: Optional[Sequence[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Quantify Need for Cognitive Closure (Webster & Kruglanski, 1994) with negation & Bayesian CI."""
        closure_lexicon = {
            "order", "structure", "predictable", "definite", "final", "decide", "closure", "certainty",
            "settle", "conclusive", "rigid", "fixed", "firm", "lowest", "highest", "minimum", "maximum",
            "bottom", "strict", "must", "exact", "non-negotiable",
            "düzen", "duzen", "kural", "kesin", "net", "karar", "sonlandırma", "sabit", "kapanma", "tayin",
            "son",
        }
        openness_lexicon = {
            "explore", "adapt", "open-ended", "ambiguous", "alternative", "revisit", "evolve", "flexible",
            "nuance", "complexity", "maybe", "might", "could", "negotiable", "willing", "open", "consider",
            "depends", "discuss",
            "esnek", "açık", "acik", "keşif", "kesif", "belirsiz", "alternatif", "değişken", "degisken", "nüans",
            "belki", "olabilir", "düşünebilirim", "dusunebilirim",
        }

        if clauses is not None:
            cl_count, op_count, reversals = self._count_negation_aware_clauses(clauses, closure_lexicon, openness_lexicon)
        else:
            cl_count, op_count, reversals = self._count_negation_aware(words, closure_lexicon, openness_lexicon)

        raw_score = (cl_count * 1.8) / (cl_count + op_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)
        bayes = self._compute_bayesian_credible_interval(cl_count * 1.8, op_count + 1.5, total_words)
        hits_total = cl_count + op_count
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)

        if not sufficient_signal:
            tier = "INSUFFICIENT_DATA"
            prob_stmt = (
                f"Need for Cognitive Closure (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if score >= 0.65:
                tier = "HIGH / HIGH_NEED_FOR_CLOSURE"
                desc = "Low tolerance for ambiguity; strong preference for rapid decisive judgment and fixed rules."
            elif score >= 0.35:
                tier = "BALANCED / CALIBRATED_CLOSURE"
                desc = "Maintains structural order while remaining adaptable to new evidence."
            else:
                tier = "OPEN / HIGH_AMBIGUITY_TOLERANCE"
                desc = "Comfortable operating under ambiguity; exploratory and open to alternative hypotheses."
            prob_stmt = f"Need for Cognitive Closure (Lexical Indicator): {int(score * 100)}% (95% CI: {bayes['credible_interval_95']}) ({desc})."

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "score": score,
            "estimated_score": score,
            "score_pct": f"{int(score * 100)}%",
            "credible_interval_95": bayes["credible_interval_95"],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "evidence_density_tier": bayes["evidence_density_tier"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "tier": tier,
            "closure_hits": cl_count,
            "openness_hits": op_count,
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_loss_aversion(
        self,
        words: List[str],
        total_words: int,
        clauses: Optional[Sequence[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Quantify Prospect Theory Loss Aversion vs Opportunity Seeking (Kahneman & Tversky)."""
        loss_lexicon = {
            "loss", "risk", "threat", "cost", "failure", "collapse", "danger", "damage", "crisis", "vulnerable", "afraid",
            "budget", "expensive", "overpriced", "broken", "damaged", "scratched", "worn", "missing", "defect", "repair", "fix", "tight",
            "kayıp", "kayip", "risk", "tehdit", "zarar", "çöküş", "cokus", "tehlike", "kriz", "felaket", "hasar", "korku", "yıkım",
            "pahalı", "pahali", "bütçe", "butce", "hasarlı", "hasarli", "bozuk", "masraf",
        }
        gain_lexicon = {
            "gain", "opportunity", "profit", "growth", "upside", "win", "breakthrough", "benefit", "advance", "reward",
            "afford", "affordable", "new", "mint", "clean", "excellent", "perfect", "bargain", "worth", "value", "bonus", "extra", "free", "included", "save", "savings",
            "kazanç", "kazanc", "fırsat", "firsat", "kâr", "kar", "büyüme", "buyume", "atılım", "avantaj", "getiri", "başarı",
            "temiz", "yeni", "değer", "deger", "indirim",
        }

        if clauses is not None:
            l_count, g_count, reversals = self._count_negation_aware_clauses(clauses, loss_lexicon, gain_lexicon)
        else:
            l_count, g_count, reversals = self._count_negation_aware(words, loss_lexicon, gain_lexicon)
        hits_total = l_count + g_count
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)
        ratio = 1.0 if hits_total == 0 else round(l_count / (g_count or 1), 2)
        bayes = self._compute_bayesian_credible_interval(l_count, g_count, total_words)
        # Map posterior mean [0, 1] into Prospect Theory beta [0.80, 2.50]
        estimated_beta = round(0.80 + (bayes["posterior_mean"] * 1.70), 2)
        ci_low_beta = round(0.80 + (bayes["credible_interval_95"][0] * 1.70), 2)
        ci_high_beta = round(0.80 + (bayes["credible_interval_95"][1] * 1.70), 2)

        if not sufficient_signal:
            orientation = "INSUFFICIENT_DATA"
            prob_stmt = (
                f"Prospect Risk Orientation (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if ratio >= 1.7:
                orientation = "LOSS_AVERSIVE_DEFENSIVE"
                desc = "Risk avoidance, defensive posture, and protection of existing positions predominate."
            elif ratio >= 0.7:
                orientation = "CALIBRATED_RISK"
                desc = "Balanced risk appetite weighing potential losses and gains probabilistically."
            else:
                orientation = "OPPORTUNITY_SEEKING"
                desc = "High risk tolerance focused on potential growth and upside opportunities."
            prob_stmt = f"Prospect Risk Orientation (Lexical Indicator): Loss/Gain ratio {ratio} (β={estimated_beta}, 95% CI: {[ci_low_beta, ci_high_beta]}) ({desc})."

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "loss_to_gain_ratio": ratio,
            "estimated_score": estimated_beta,
            "credible_interval_95": [ci_low_beta, ci_high_beta],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "evidence_density_tier": bayes["evidence_density_tier"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "loss_hits": l_count,
            "gain_hits": g_count,
            "orientation": orientation,
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_affective_volatility(
        self,
        sentences: List[str],
        words: List[str],
        total_words: int,
        clauses: Optional[Sequence[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Quantify Emotional Volatility vs Stoic Composure across sentence transitions."""
        arousal_lexicon = {
            "fury", "panic", "rage", "ecstatic", "terrifying", "devastating", "catastrophic", "shocking",
            "ridiculous", "crazy", "insane", "insult", "insulting", "absurd", "outrageous", "awful",
            "terrible", "horrible", "desperate", "urgent", "hurry", "asap", "frustrated", "annoying", "angry", "mad",
            "öfke", "ofke", "panik", "dehşet", "dehset", "hiddet", "çıldırma", "rezalet", "coşku", "cosku",
            "saçma", "sacma", "sinir", "acil", "berbat",
        }
        calm_lexicon = {
            "serene", "calm", "steady", "balanced", "composed", "tranquil", "objective",
            "understand", "understood", "respect", "patient", "polite", "fine", "okay", "ok", "sure", "alright",
            "sakin", "soğukkanlı", "sogukkanli", "itidalli", "dengeli", "ölçülü", "olculu", "nesnel", "sükûnet",
            "anlayış", "anlayis", "tamam", "olur",
        }

        if clauses is not None:
            arousal_count, calm_count, reversals = self._count_negation_aware_clauses(clauses, arousal_lexicon, calm_lexicon)
        else:
            arousal_count, calm_count, reversals = self._count_negation_aware(words, arousal_lexicon, calm_lexicon)

        volatility_score = round(min(1.0, (arousal_count * 2.5) / (total_words / 100.0 + calm_count + 1.0)), 2)
        composure_score = round(1.0 - volatility_score, 2)
        bayes = self._compute_bayesian_credible_interval(arousal_count * 2.0, calm_count + 2.0, total_words)
        hits_total = arousal_count + calm_count
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)

        if not sufficient_signal:
            tier = "INSUFFICIENT_DATA"
            prob_stmt = (
                f"Affective Balance & Reactivity (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if volatility_score >= 0.50:
                tier = "HIGH_AFFECTIVE_VOLATILITY"
                desc = "Elevated agitation and sharp affective swings observed under pressure or crisis."
            elif volatility_score >= 0.20:
                tier = "BALANCED_REACTIVITY"
                desc = "Emotional expressions stay within a context-appropriate, controllable reactivity range."
            else:
                tier = "STOIC_COMPOSURE"
                desc = "High emotional self-regulation; maintains calm, objective tone even under pressure."
            prob_stmt = f"Affective Balance & Reactivity (Lexical Indicator): {tier} ({desc})."

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "volatility_score": volatility_score,
            "composure_score": composure_score,
            "credible_interval_95": bayes["credible_interval_95"],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "tier": tier,
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_moral_foundations(
        self,
        words: List[str],
        total_words: int,
        clauses: Optional[Sequence[List[str]]] = None,
    ) -> Dict[str, Any]:
        """Quantify Moral Foundations profile (Haidt & Graham: Fairness/Care vs Authority/Loyalty)."""
        individualizing_lexicon = {
            "care", "harm", "protect", "suffer", "justice", "fair", "rights", "equality", "oppression",
            "hak", "hakkı", "adalet", "koruma", "şefkat", "sefkat", "eşitlik", "esitlik", "mazlum", "mazlumların", "zulüm", "özgürlük", "hakkaniyet"
        }
        binding_lexicon = {
            "authority", "obey", "duty", "loyalty", "respect", "tradition", "order", "honor", "treason",
            "otorite", "itaat", "görev", "gorev", "sadakat", "saygı", "saygi", "gelenek", "nizam", "hiyerarşi"
        }

        if clauses is not None:
            ind_count, bind_count, reversals = self._count_negation_aware_clauses(
                clauses, individualizing_lexicon, binding_lexicon
            )
        else:
            ind_count, bind_count, reversals = self._count_negation_aware(
                words, individualizing_lexicon, binding_lexicon
            )
        bayes = self._compute_bayesian_credible_interval(ind_count, bind_count, total_words)
        hits_total = ind_count + bind_count
        sufficient_signal = (hits_total >= 1 and total_words >= MIN_WORDS_FOR_INDICATORS) or (hits_total >= 2)

        if not sufficient_signal:
            foundation = "INSUFFICIENT_DATA"
            prob_stmt = (
                f"Moral Foundations Matrix (Lexical Indicator): INSUFFICIENT_DATA — "
                f"insufficient lexical signal in corpus (words={total_words}, matches={hits_total})."
            )
        else:
            if ind_count > bind_count * 1.5:
                foundation = "INDIVIDUALIZING_FAIRNESS"
                desc = "Individual rights, fairness, and care/protection principles outweigh institutional authority."
            elif bind_count > ind_count * 1.5:
                foundation = "BINDING_AUTHORITY"
                desc = "Institutional loyalty, hierarchy, tradition, and order take precedence."
            else:
                foundation = "BALANCED_FOUNDATIONS"
                desc = "Balances individual fairness/care alongside institutional order and tradition."
            prob_stmt = f"Moral Foundations Matrix (Lexical Indicator): {foundation} ({desc})."

        return {
            "measurement_type": "lexical_indicator",
            "sufficient_signal": sufficient_signal,
            "individualizing_hits": ind_count,
            "binding_hits": bind_count,
            "credible_interval_95": bayes["credible_interval_95"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "dominant_foundation": foundation,
            "probabilistic_statement": prob_stmt,
        }

    def _calculate_caps_contextual_matrix(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Model Walter Mischel's Cognitive-Affective Personality System (CAPS):
        Extracts context-conditional behavioral shifts (`IF [Context] -> THEN [Behavioral Signature]`)
        across 4 canonical situational domains.
        """
        rigidity_terms = {
            "never", "must", "unacceptable", "reject", "refuse", "non-negotiable", "strictly", "always", "definitely",
            "asla", "kesinlikle", "reddediyorum", "kabul edilemez", "tavizsiz", "şarttır", "mutlaka", "mecbur", "kesin", "net", "kural"
        }
        flexibility_terms = {
            "perhaps", "open", "explore", "collaborate", "adapt", "consider", "mutual", "dialogue", "possible",
            "belki", "açık", "esnek", "işbirliği", "değerlendirebiliriz", "ortak", "diyalog", "olası", "mümkün", "olasılıksal"
        }

        buckets: Dict[str, List[str]] = {k: [] for k in CAPS_CONTEXT_MARKERS}
        for s in sentences:
            s_words = set(re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", s.lower()))
            matched_any = False
            for ctx_name, markers in CAPS_CONTEXT_MARKERS.items():
                if s_words & markers:
                    buckets[ctx_name].append(s)
                    matched_any = True
            if not matched_any:
                buckets["FORMAL_ACADEMIC"].append(s)

        context_profiles: Dict[str, Dict[str, Any]] = {}
        rigidity_scores: List[float] = []
        signatures: List[Dict[str, Any]] = []

        for ctx_name, ctx_sents in buckets.items():
            ctx_words = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", " ".join(ctx_sents).lower())
            rig_hits, flex_hits, _ = self._count_negation_aware(ctx_words, rigidity_terms, flexibility_terms)
            base_prior = {
                "FORMAL_ACADEMIC": 0.48,
                "HIGH_STAKES_CRISIS": 0.72,
                "STRATEGIC_NEGOTIATION": 0.64,
                "INTERPERSONAL_DIALOGUE": 0.28,
            }[ctx_name]

            if (rig_hits + flex_hits) > 0:
                observed_rigidity = rig_hits / float(rig_hits + flex_hits)
                weight = min(0.85, (rig_hits + flex_hits) / 6.0)
                ctx_rigidity = round((weight * observed_rigidity) + ((1.0 - weight) * base_prior), 2)
            else:
                ctx_rigidity = base_prior

            rigidity_scores.append(ctx_rigidity)

            if ctx_rigidity >= 0.65:
                posture = "DEFENSIVE_BOUNDARY_ENFORCEMENT"
                then_behavior = "Enforce strict normative boundaries, high loss-aversion veto, and zero unilateral concession."
            elif ctx_rigidity >= 0.40:
                posture = "ANALYTICAL_RECIPROCITY"
                then_behavior = "Demand empirical verification and conditional tit-for-tat reciprocity."
            else:
                posture = "CONSTRUCTIVE_OPENNESS"
                then_behavior = "Maintain warm diplomatic composure and exploratory synthesis."

            rule = f"IF [{ctx_name}] -> THEN {then_behavior}"
            context_profiles[ctx_name] = {
                "sample_sentences": len(ctx_sents),
                "segment_count": len(ctx_sents),
                "rigidity_score": ctx_rigidity,
                "flexibility_score": round(1.0 - ctx_rigidity, 2),
                "contextual_loss_aversion": round(1.0 + ctx_rigidity * 1.4, 2),
                "dominant_posture": posture,
                "if_then_signature": rule,
            }
            signatures.append({
                "if_context": ctx_name,
                "then_behavior": then_behavior,
                "rigidity_score": ctx_rigidity,
                "rule_string": rule,
            })

        mean_rig = sum(rigidity_scores) / len(rigidity_scores)
        ctx_variance = round(math.sqrt(sum((r - mean_rig) ** 2 for r in rigidity_scores) / len(rigidity_scores)), 3)

        return {
            "theory": "Mischel & Shoda (1995) Cognitive-Affective Personality System (CAPS)",
            "contextual_plasticity_index": ctx_variance,
            "contexts": context_profiles,
            "if_then_signatures": signatures,
        }

    def _calculate_stated_vs_revealed_preference(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Econometric & Behavioral Stated vs. Revealed Preference Divergence (Samuelson, 1938):
        Compares normative self-descriptions ('Stated Preference') against documented past decisions/actions
        ('Revealed Preference') to detect strategic dissonance or rhetoric-action gaps.
        """
        stated_markers = {
            "believe", "principle", "value", "advocate", "ideal", "should", "ought", "philosophy", "always support", "declare",
            "inanıyorum", "ilkemiz", "ilkelerine", "değerimiz", "savunuyorum", "olmalıdır", "felsefemiz", "her zaman", "vizyonumuz", "beyan", "bağlılığımızı"
        }
        revealed_markers = {
            "decided", "rejected", "vetoed", "enforced", "signed", "walked", "terminated", "executed", "implemented", "blocked", "withdrew",
            "reddetti", "reddettik", "veto", "uyguladık", "uygulamada", "imzaladı", "imzaladık", "karar", "iptal", "sonlandırdık", "bloke", "çekildi", "yaptık", "seçtik"
        }

        coop_words = STATED_COOPERATIVE_MARKERS | REVEALED_COOPERATIVE_ACTION_MARKERS
        control_words = STATED_DEFENSIVE_MARKERS | REVEALED_AUTONOMOUS_EXIT_MARKERS

        stated_sents: List[str] = []
        revealed_sents: List[str] = []

        for s in sentences:
            s_lower = s.lower()
            if any(m in s_lower for m in revealed_markers):
                revealed_sents.append(s)
            elif any(m in s_lower for m in stated_markers):
                stated_sents.append(s)

        stated_tokens = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", " ".join(stated_sents).lower())
        revealed_tokens = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", " ".join(revealed_sents).lower())

        s_coop, s_ctrl, _ = self._count_negation_aware(stated_tokens, coop_words, control_words)
        r_coop, r_ctrl, _ = self._count_negation_aware(revealed_tokens, coop_words, control_words)

        # Symmetric Beta(1, 1) prior so zero-evidence neutral text yields 0.50 / 0.50 and 0.00 divergence
        stated_coop_score = round((s_coop + 1.0) / (s_coop + s_ctrl + 2.0), 2)
        revealed_coop_score = round((r_coop + 1.0) / (r_coop + r_ctrl + 2.0), 2)

        if len(stated_sents) == 0 and len(revealed_sents) == 0:
            divergence = 0.0
            verdict = "INSUFFICIENT_PREFERENCE_EVIDENCE"
            explanation = (
                "Insufficient Preference Evidence: No explicit normative statements or documented past actions "
                "found in corpus; retaining symmetric Beta(1,1) prior (divergence=0.0)."
            )
        else:
            divergence = round(abs(stated_coop_score - revealed_coop_score), 2)
            if divergence >= 0.35:
                verdict = "STRATEGIC_DISSONANCE_DETECTED"
                explanation = (
                    "High Rhetoric-Action Divergence: Public rhetoric emphasizes collaborative openness while "
                    "documented actions favor autonomy preservation and strict veto control."
                )
            elif divergence >= 0.15:
                verdict = "PRAGMATIC_BOUNDARY_GUARDING"
                explanation = (
                    "Partial Pragmatic Divergence: Rhetoric signals openness to collaboration, "
                    "but execution avoids delegating authority without structural guarantees."
                )
            else:
                verdict = "HIGH_EPISTEMIC_CONGRUENCE"
                explanation = (
                    "Rhetoric-Action Congruence: Stated normative principles closely align with "
                    "documented past decisions."
                )

        return {
            "stated_cooperation_polarity": stated_coop_score,
            "revealed_cooperation_polarity": revealed_coop_score,
            "preference_divergence_index": divergence,
            "stated_sample_count": len(stated_sents),
            "revealed_sample_count": len(revealed_sents),
            "verdict": verdict,
            "diagnosis": explanation,
        }

    def _extract_conditional_commitments(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Extract Typed Conditional Commitments (CONDITIONAL_COMMITMENT) and
        Lexicographic Sacred-Value Red-Line Vetoes (NECESSARY_CONDITION / RED_LINE_VETO)
        from corpus sentences.
        """
        red_line_markers = {
            "never", "non-negotiable", "red line", "unacceptable", "under no circumstances",
            "will not transfer", "refuse to waive", "cannot compromise", "must remain",
            "asla", "kırmızı çizgi", "pazarlık konusu olamaz", "kabul edilemez", "devredilemez",
            "vazgeçilemez", "taviz", "veto", "kesinlikle reddederiz",
        }
        conditional_markers = {
            "only if", "unless", "provided that", "on condition", "as long as", "prerequisite",
            "ancak", "şartıyla", "koşuluyla", "olmadıkça", "sağlanmadan", "garanti edilmedikçe",
            "kalması şartıyla", "korunması halinde",
        }
        concept_anchors = [
            "master rights", "kayıt hakları", "intellectual property", "fikri mülkiyet",
            "autonomy", "otonomi", "bağımsızlık", "independence", "veto", "governance",
            "yönetişim", "equity", "hisse", "source code", "kaynak kodu", "editorial control",
            "yayın hakları", "creative control", "sanatsal kontrol", "audit", "denetim",
            "transparency", "şeffaflık", "royalty", "telif", "data ownership", "veri sahipliği",
        ]

        commitments: List[Dict[str, Any]] = []
        protected_concepts_set: List[str] = []

        for sent in sentences:
            s_lower = sent.lower()
            is_red_line = any(m in s_lower for m in red_line_markers)
            is_cond = any(m in s_lower for m in conditional_markers)
            if not (is_red_line or is_cond):
                continue

            matched_concepts = [c for c in concept_anchors if c in s_lower]
            if not matched_concepts:
                tokens = [
                    w for w in re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{5,}\b", s_lower)
                    if w not in {"ancak", "asla", "never", "unless", "provided", "condition", "kesinlikle"}
                ]
                matched_concepts = tokens[:3]

            for mc in matched_concepts:
                if mc not in protected_concepts_set:
                    protected_concepts_set.append(mc)

            commitments.append({
                "claim_type": "NECESSARY_CONDITION_RED_LINE" if is_red_line else "CONDITIONAL_COMMITMENT",
                "protected_concepts": matched_concepts[:4],
                "rigidity_weight": 0.90 if is_red_line else 0.70,
                "source_quote": sent.strip()[:220],
            })

        return {
            "total_commitments_extracted": len(commitments),
            "red_line_count": sum(1 for c in commitments if c["claim_type"] == "NECESSARY_CONDITION_RED_LINE"),
            "conditional_count": sum(1 for c in commitments if c["claim_type"] == "CONDITIONAL_COMMITMENT"),
            "protected_concepts": protected_concepts_set[:10],
            "rules": commitments[:8],
        }

    def _empty_feature_set(self) -> Dict[str, Any]:
        return {
            "measurement_methodology": {
                "signal_type": "LEXICAL_BEHAVIORAL_INDICATOR",
                "clinical_diagnostic_validity": False,
                "corpus_word_count": 0,
                "min_words_for_indicators": MIN_WORDS_FOR_INDICATORS,
                "min_words_recommended": MIN_WORDS_RECOMMENDED,
                "corpus_sufficiency": "INSUFFICIENT_DATA",
                "disclaimer": (
                    "Lexicon-based behavioral indicators measure surface linguistic patterns, not clinical "
                    "psychological traits. Interpret scores alongside corpus size and 95% credible intervals."
                ),
            },
            "corpus_stats": {"total_words": 0, "total_sentences": 0, "type_token_ratio": 0.0, "avg_sentence_length": 0.0},
            "pronoun_ratios": {"first_person_singular_pct": 0.0, "i_to_we_ratio": 0.0},
            "epistemic_markers": {"epistemic_calibration_ratio": 1.0, "negated_reversals_detected": 0},
            "epistemic_manipulation_index": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "score": 0.0,
                "estimated_score": 0.0,
                "score_pct": "0%",
                "credible_interval_95": [0.0, 0.5],
                "uncertainty_margin": 0.25,
                "risk_tier": "INSUFFICIENT_DATA",
                "confidence_score": 0.0,
                "probabilistic_statement": "Epistemic Manipulation Index (Lexical Indicator): INSUFFICIENT_DATA — insufficient corpus volume.",
            },
            "machiavellianism_index": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "score": 0.0,
                "estimated_score": 0.0,
                "credible_interval_95": [0.0, 0.5],
                "tier": "INSUFFICIENT_DATA",
            },
            "need_for_closure_index": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "score": 0.0,
                "estimated_score": 0.0,
                "credible_interval_95": [0.0, 0.5],
                "tier": "INSUFFICIENT_DATA",
            },
            "loss_aversion_index": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "loss_to_gain_ratio": 1.0,
                "estimated_score": 1.5,
                "credible_interval_95": [1.0, 2.0],
                "orientation": "INSUFFICIENT_DATA",
            },
            "affective_volatility_index": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "volatility_score": 0.0,
                "credible_interval_95": [0.0, 0.5],
                "tier": "INSUFFICIENT_DATA",
            },
            "moral_foundations_profile": {
                "measurement_type": "lexical_indicator",
                "sufficient_signal": False,
                "dominant_foundation": "INSUFFICIENT_DATA",
                "credible_interval_95": [0.0, 0.5],
            },
            "caps_contextual_matrix": {"contextual_plasticity_index": 0.0, "contexts": {}, "if_then_signatures": []},
            "stated_vs_revealed_preference": {
                "stated_cooperation_polarity": 0.5,
                "revealed_cooperation_polarity": 0.5,
                "preference_divergence_index": 0.0,
                "stated_sample_count": 0,
                "revealed_sample_count": 0,
                "verdict": "INSUFFICIENT_DATA",
            },
            "conditional_commitments": {
                "total_commitments_extracted": 0,
                "red_line_count": 0,
                "conditional_count": 0,
                "protected_concepts": [],
                "rules": [],
            },
            "top_characteristic_terms": [],
        }
