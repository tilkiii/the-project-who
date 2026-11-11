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
from typing import Any, Dict, List, Optional, Set, Tuple

from who.utils.logger import get_logger

logger = get_logger("who.learn.features")

NEGATION_TOKENS: Set[str] = {
    "not", "never", "no", "neither", "nor", "without", "hardly", "rarely", "cannot", "cant", "won't", "dont", "doesn't",
    "değil", "degil", "yok", "yoktur", "asla", "hiç", "hic", "hiçbir", "hicbir", "olmayan", "olmaz", "imkansız", "imkansiz",
}

TURKISH_NEG_SUFFIXES: Tuple[str, ...] = (
    "meme", "mama", "mez", "maz", "miyor", "mıyor", "muyor", "müyor",
    "medi", "madı", "medik", "madık", "meyecek", "mayacak",
)

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
        "müzakere", "muzakere", "teklif", "pazarlık", "pazarlik", "uzlaşı", "uzlasi", "veto", "taviz", "anlaşma", "mutabakat", "kontrat",
    },
    "INTERPERSONAL_DIALOGUE": {
        "friend", "hello", "thank", "welcome", "conversation", "personal", "feel", "together", "chat", "hey",
        "merhaba", "dost", "teşekkür", "sohbet", "samimi", "birlikte", "kişisel", "insan", "gönül", "selam",
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
    def _is_negated_context(words: List[str], idx: int, window_before: int = 3, window_after: int = 2) -> bool:
        """
        Check whether token at `idx` is governed by a negation marker within its syntactic window
        (e.g. 'kesinlikle doğru değil', 'not certain at all', 'risk yok').
        """
        start = max(0, idx - window_before)
        end = min(len(words), idx + window_after + 1)
        for j in range(start, end):
            if j == idx:
                continue
            w = words[j]
            if w in NEGATION_TOKENS:
                return True
            if len(w) > 5 and w.endswith(TURKISH_NEG_SUFFIXES):
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
            if w in primary_lexicon:
                if w not in NEGATION_TOKENS and cls._is_negated_context(words, idx):
                    opposite_hits += 1
                    reversals += 1
                else:
                    primary_hits += 1
            elif w in opp_lex:
                if w not in NEGATION_TOKENS and cls._is_negated_context(words, idx):
                    primary_hits += 1
                    reversals += 1
                else:
                    opposite_hits += 1

        return primary_hits, opposite_hits, reversals

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

        words = re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", combined_text.lower())
        total_words = len(words)
        if total_words == 0:
            return self._empty_feature_set()

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

        # 2. Epistemic stance words (Negation-aware)
        causal_lexicon = {
            "because", "therefore", "thus", "hence", "consequently", "implies", "since", "results",
            "çünkü", "cunku", "dolayısıyla", "dolayisiyla", "böylece", "boylece", "nedeniyle", "sonucunda"
        }

        certainty_count, tentative_count, epistemic_reversals = self._count_negation_aware(
            words, CERTAINTY_MARKERS, TENTATIVE_MARKERS
        )
        causal_count = sum(1 for w in words if w in causal_lexicon)

        # 3. Emotional valence indicators (Negation-aware)
        positive_lexicon = {
            "great", "excellent", "superb", "constructive", "insightful", "promising", "breakthrough", "success", "effective", "valuable",
            "harika", "mükemmel", "mukemmel", "başarılı", "basarili", "faydalı", "faydali", "önemli", "onemli", "değerli", "degerli"
        }
        critical_lexicon = {
            "flawed", "erroneous", "inefficient", "risk", "failure", "deceptive", "critical", "vulnerable", "concern", "problem",
            "hatalı", "hatali", "yanlış", "yanlis", "başarısız", "basarisiz", "risk", "tehlike", "sorun", "kriz", "çelişki", "celiski"
        }

        pos_count, crit_count, valence_reversals = self._count_negation_aware(
            words, positive_lexicon, critical_lexicon
        )

        # 4. Lexical diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / total_words

        # 5. Rhetorical syntax metrics
        avg_sentence_len = total_words / total_sentences
        sentence_lengths = [len(re.findall(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ']+\b", s)) for s in sentences]
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

        # 9. Extended Psychometrics (with Negation Awareness & Bayesian 95% Credible Intervals)
        machiavellianism_data = self._calculate_machiavellianism(words, total_words)
        closure_data = self._calculate_need_for_closure(words, total_words)
        loss_aversion_data = self._calculate_loss_aversion(words, total_words)
        volatility_data = self._calculate_affective_volatility(sentences, words, total_words)
        moral_foundations_data = self._calculate_moral_foundations(words, total_words)

        # 10. Context-Conditional CAPS Matrix (Walter Mischel If-Then Behavioral Signatures)
        caps_matrix = self._calculate_caps_contextual_matrix(sentences)

        # 11. Stated vs. Revealed Preference Divergence Analysis
        preference_divergence = self._calculate_stated_vs_revealed_preference(sentences)

        features = {
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

        if manipulation_index >= 0.70:
            risk_tier = "YÜKSEK (HIGH_MANIPULATION_RISK)"
            risk_desc = "Kanıtsız ve mutlakiyetçi dil kullanımı belirgin düzeydedir."
        elif manipulation_index >= 0.40:
            risk_tier = "ORTA (MODERATE_DOGMATIC_TENDENCY)"
            risk_desc = "Ara sıra mutlakiyetçi ifadelere rastlanmakta, ancak rasyonel süzgeç dengelenmektedir."
        else:
            risk_tier = "DÜŞÜK / RASYONEL (EMPIRICALLY_CALIBRATED)"
            risk_desc = "Gözlemlenen beyanlarda iddialar olasılıksal ve kanıta dayalı ihtiyatla temellendirilmektedir."

        confidence = min(0.95, 0.60 + (total_words / 1500.0) * 0.3)

        return {
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
            "probabilistic_statement": (
                f"Manipülasyon Endeksi: Kanıtsız iddialarda mutlak kesinlik dili kullanımı eğilimi "
                f"%{int(manipulation_index * 100)} (±%{int(bayes['uncertainty_margin']*100)}) seviyesindedir ({risk_desc} Olasılık Skoru: %{int(confidence * 100)})."
            ),
        }

    def _calculate_machiavellianism(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Machiavellianism with negation reversal and Bayesian credible intervals."""
        mach_lexicon = {
            "manipulate", "exploitation", "leverage", "advantage", "subordinate", "opportunism", "deceit",
            "ruthless", "dominate", "maneuver", "outmaneuver", "instrumental", "pragmatic", "utility",
            "kullanmak", "fayda", "menfaat", "çıkar", "cikar", "koz", "üstünlük", "ustunluk",
            "sömürü", "somuru", "araçsallaştırma", "baskınlık", "zaaf", "zaafı", "fırsatçılık"
        }
        coop_lexicon = {
            "cooperate", "mutual", "altruism", "empathy", "ethics", "integrity", "solidarity", "compassion",
            "işbirliği", "isbirligi", "dayanışma", "dayanisma", "empati", "ahlak", "dürüstlük", "durustluk",
            "karşılıklı", "karsilikli", "güven", "guven", "adalet", "hakkaniyet", "şefkat", "sefkat"
        }

        m_count, c_count, reversals = self._count_negation_aware(words, mach_lexicon, coop_lexicon)

        raw_score = (m_count * 2.2) / (m_count + c_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)
        bayes = self._compute_bayesian_credible_interval(m_count * 2.2, c_count + 1.5, total_words)

        if score >= 0.55:
            tier = "YÜKSEK / MAKYAVELİST (STRATEGIC_MACHIAVELLIAN)"
            desc = "Pragmatik fayda ve stratejik manevra dili yüksek, ahlaki duygulanımdan ziyade güce odaklı."
        elif score >= 0.25:
            tier = "DENGELİ / STRATEJİK PRAGMATİST (BALANCED_PRAGMATIST)"
            desc = "İş birliğine açık fakat stratejik çıkar ve kurumsal rasyonaliteyi gözeten tutum."
        else:
            tier = "DÜŞÜK / ALTRUİST & İŞBİRLİKÇİ (COOPERATIVE_ALTRUIST)"
            desc = "Yüksek empati, etik ilkelere bağlılık ve kolektif dayanışma dili baskın."

        return {
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
            "probabilistic_statement": (
                f"Makyavelizm & Güç Yönelimi: %{int(score * 100)} (95% CI: {bayes['credible_interval_95']}) seviyesinde ({desc})."
            ),
        }

    def _calculate_need_for_closure(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Need for Cognitive Closure (Webster & Kruglanski, 1994) with negation & Bayesian CI."""
        closure_lexicon = {
            "order", "structure", "predictable", "definite", "final", "decide", "closure", "certainty",
            "settle", "conclusive", "rigid", "fixed",
            "düzen", "duzen", "kural", "kesin", "net", "karar", "sonlandırma", "sabit", "kapanma", "tayin"
        }
        openness_lexicon = {
            "explore", "adapt", "open-ended", "ambiguous", "alternative", "revisit", "evolve", "flexible",
            "nuance", "complexity",
            "esnek", "açık", "acik", "keşif", "kesif", "belirsiz", "alternatif", "değişken", "degisken", "nüans"
        }

        cl_count, op_count, reversals = self._count_negation_aware(words, closure_lexicon, openness_lexicon)

        raw_score = (cl_count * 1.8) / (cl_count + op_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)
        bayes = self._compute_bayesian_credible_interval(cl_count * 1.8, op_count + 1.5, total_words)

        if score >= 0.65:
            tier = "KATI / YÜKSEK KAPATMA İHTİYACI (HIGH_NEED_FOR_CLOSURE)"
            desc = "Belirsizliğe tahammül düşük; hızla kesin hükme varma ve sabit kurallar arayışı baskın."
        elif score >= 0.35:
            tier = "DENGELİ BİLİŞSEL DÜZEN (CALIBRATED_CLOSURE)"
            desc = "Karar alırken yapısal çerçeveyi koruyan ancak yeni kanıtlar karşısında esneyebilen tutum."
        else:
            tier = "AÇIK UÇLU & BİLİŞSEL ESNEK (HIGH_AMBIGUITY_TOLERANCE)"
            desc = "Belirsizlikle rahatça çalışabilen, keşifçi ve alternatif hipotezlere açık bilişsel yapı."

        return {
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
            "probabilistic_statement": (
                f"Bilişsel Kapanma İhtiyacı: %{int(score * 100)} (95% CI: {bayes['credible_interval_95']}) seviyesinde ({desc})."
            ),
        }

    def _calculate_loss_aversion(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Prospect Theory Loss Aversion vs Opportunity Seeking (Kahneman & Tversky)."""
        loss_lexicon = {
            "loss", "risk", "threat", "cost", "failure", "collapse", "danger", "damage", "crisis", "vulnerable", "afraid",
            "kayıp", "kayip", "risk", "tehdit", "zarar", "çöküş", "cokus", "tehlike", "kriz", "felaket", "hasar", "korku", "yıkım"
        }
        gain_lexicon = {
            "gain", "opportunity", "profit", "growth", "upside", "win", "breakthrough", "benefit", "advance", "reward",
            "kazanç", "kazanc", "fırsat", "firsat", "kâr", "kar", "büyüme", "buyume", "atılım", "avantaj", "getiri", "başarı"
        }

        l_count, g_count, reversals = self._count_negation_aware(words, loss_lexicon, gain_lexicon)
        ratio = round(l_count / (g_count or 1), 2)
        bayes = self._compute_bayesian_credible_interval(l_count, g_count, total_words)
        # Map posterior mean [0, 1] into Prospect Theory beta [0.80, 2.50]
        estimated_beta = round(0.80 + (bayes["posterior_mean"] * 1.70), 2)
        ci_low_beta = round(0.80 + (bayes["credible_interval_95"][0] * 1.70), 2)
        ci_high_beta = round(0.80 + (bayes["credible_interval_95"][1] * 1.70), 2)

        if ratio >= 1.7:
            orientation = "KAYIPTAN KAÇINMA BASKIN (LOSS_AVERSIVE_DEFENSIVE)"
            desc = "Riskten kaçınma, savunma ve mevcut kazanımları koruma odağı belirleyici."
        elif ratio >= 0.7:
            orientation = "DENGELİ PROSPEKT (CALIBRATED_RISK)"
            desc = "Kayıp ve kazanç senaryolarını rasyonel olasılıklarla tartan dengeli risk iştahı."
        else:
            orientation = "FIRSAT & KAZANÇ ODAKLI (OPPORTUNITY_SEEKING)"
            desc = "Potansiyel büyüme ve yeni fırsatlara odaklanan yüksek risk toleransı."

        return {
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
            "probabilistic_statement": f"Prospekt Risk Yönelimi: Kayıp/Kazanç oranı {ratio} (β={estimated_beta}, 95% CI: {[ci_low_beta, ci_high_beta]}) ({desc}).",
        }

    def _calculate_affective_volatility(self, sentences: List[str], words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Emotional Volatility vs Stoic Composure across sentence transitions."""
        arousal_lexicon = {
            "fury", "panic", "rage", "ecstatic", "terrifying", "devastating", "catastrophic", "shocking",
            "öfke", "ofke", "panik", "dehşet", "dehset", "hiddet", "çıldırma", "rezalet", "coşku", "cosku"
        }
        calm_lexicon = {
            "serene", "calm", "steady", "balanced", "composed", "tranquil", "objective",
            "sakin", "soğukkanlı", "sogukkanli", "itidalli", "dengeli", "ölçülü", "olculu", "nesnel", "sükûnet"
        }

        arousal_count, calm_count, reversals = self._count_negation_aware(words, arousal_lexicon, calm_lexicon)

        volatility_score = round(min(1.0, (arousal_count * 2.5) / (total_words / 100.0 + calm_count + 1.0)), 2)
        composure_score = round(1.0 - volatility_score, 2)
        bayes = self._compute_bayesian_credible_interval(arousal_count * 2.0, calm_count + 2.0, total_words)

        if volatility_score >= 0.50:
            tier = "YÜKSEK VOLATİLİTE / DUYGUSAL TEPKİSELLİK"
            desc = "Kriz veya baskı anlarında yüksek ajitasyon ve keskin duygusal iniş-çıkışlar gözlemlenmektedir."
        elif volatility_score >= 0.20:
            tier = "DENGELİ REAKTİVİTE"
            desc = "Duygusal ifadeler bağlama uygun, kontrol edilebilir bir reaktivite aralığındadır."
        else:
            tier = "STOİK / YÜKSEK SOĞUKKANLILIK (STOIC_COMPOSURE)"
            desc = "Yüksek duygusal öz-denetim, baskı altında dahi sükûnet ve nesnel tonunu korur."

        return {
            "volatility_score": volatility_score,
            "composure_score": composure_score,
            "credible_interval_95": bayes["credible_interval_95"],
            "uncertainty_margin": bayes["uncertainty_margin"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "tier": tier,
            "probabilistic_statement": f"Duygusal Denge & Afektif Denge: {tier} ({desc}).",
        }

    def _calculate_moral_foundations(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Moral Foundations profile (Haidt & Graham: Fairness/Care vs Authority/Loyalty)."""
        individualizing_lexicon = {
            "care", "harm", "protect", "suffer", "justice", "fair", "rights", "equality", "oppression",
            "hak", "hakkı", "adalet", "koruma", "şefkat", "sefkat", "eşitlik", "esitlik", "mazlum", "mazlumların", "zulüm", "özgürlük", "hakkaniyet"
        }
        binding_lexicon = {
            "authority", "obey", "duty", "loyalty", "respect", "tradition", "order", "honor", "treason",
            "otorite", "itaat", "görev", "gorev", "sadakat", "saygı", "saygi", "gelenek", "nizam", "hiyerarşi"
        }

        ind_count, bind_count, reversals = self._count_negation_aware(words, individualizing_lexicon, binding_lexicon)
        bayes = self._compute_bayesian_credible_interval(ind_count, bind_count, total_words)

        if ind_count > bind_count * 1.5:
            foundation = "BİREY & HAKKANİYET ODAKLI (INDIVIDUALIZING_FAIRNESS)"
            desc = "Bireysel haklar, adalet ve şefkat/koruma ilkeleri kurumsal otoritenin üzerinde tutulur."
        elif bind_count > ind_count * 1.5:
            foundation = "OTORİTE & NİZAM ODAKLI (BINDING_AUTHORITY)"
            desc = "Kurumsal sadakat, hiyerarşi, gelenek ve nizamın korunması önceliklidir."
        else:
            foundation = "ÇOK BOYUTLU / DENGELİ AHLAKİ TEMEL (BALANCED_FOUNDATIONS)"
            desc = "Hem bireysel adalet ve hakkaniyeti hem de kurumsal gelenek ve düzeni birlikte gözetir."

        return {
            "individualizing_hits": ind_count,
            "binding_hits": bind_count,
            "credible_interval_95": bayes["credible_interval_95"],
            "negated_reversals_detected": reversals,
            "negated_reversals": reversals,
            "dominant_foundation": foundation,
            "probabilistic_statement": f"Ahlaki Temeller Matrisi: {foundation} ({desc}).",
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

        stated_coop_score = round((s_coop + 1.5) / (s_coop + s_ctrl + 2.5), 2)
        revealed_coop_score = round((r_coop + 1.0) / (r_coop + r_ctrl + 2.5), 2)

        divergence = round(abs(stated_coop_score - revealed_coop_score), 2)

        if divergence >= 0.35:
            verdict = "STRATEGIC_DISSONANCE_DETECTED"
            explanation = (
                "Yüksek Söylem-Eylem Sapması: Kamusal söylemlerde uzlaşıcı/açık bir dil kullanılırken, "
                "gerçekleşen kararlarda otonomi koruma ve katı kontrol/veto davranışı ağır basmaktadır."
            )
        elif divergence >= 0.15:
            verdict = "PRAGMATIC_BOUNDARY_GUARDING"
            explanation = (
                "Kısmi Pragmatik Sapma: Söylem düzeyinde iş birliğine açık olmakla birlikte, "
                "uygulama aşamasında yapısal güvenceler sağlanmadan yetki devri yapılmamaktadır."
            )
        else:
            verdict = "HIGH_EPISTEMIC_CONGRUENCE"
            explanation = (
                "Söylem-Eylem Tutarlılığı: Beyan edilen normatif ilkeler ile geçmişte alınan somut kararlar "
                "yüksek düzeyde örtüşmektedir."
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

    def _empty_feature_set(self) -> Dict[str, Any]:
        return {
            "corpus_stats": {"total_words": 0, "total_sentences": 0, "type_token_ratio": 0.0, "avg_sentence_length": 0.0},
            "pronoun_ratios": {"first_person_singular_pct": 0.0, "i_to_we_ratio": 0.0},
            "epistemic_markers": {"epistemic_calibration_ratio": 1.0, "negated_reversals_detected": 0},
            "epistemic_manipulation_index": {
                "score": 0.0,
                "estimated_score": 0.0,
                "score_pct": "0%",
                "credible_interval_95": [0.0, 0.5],
                "uncertainty_margin": 0.25,
                "risk_tier": "YETERSİZ VERİ",
                "probabilistic_statement": "Yetersiz korpus nedeniyle manipülasyon endeksi hesaplanamadı."
            },
            "machiavellianism_index": {"score": 0.0, "estimated_score": 0.0, "credible_interval_95": [0.0, 0.5], "tier": "YETERSİZ VERİ"},
            "need_for_closure_index": {"score": 0.0, "estimated_score": 0.0, "credible_interval_95": [0.0, 0.5], "tier": "YETERSİZ VERİ"},
            "loss_aversion_index": {"loss_to_gain_ratio": 1.0, "estimated_score": 1.5, "credible_interval_95": [1.0, 2.0], "orientation": "YETERSİZ VERİ"},
            "affective_volatility_index": {"volatility_score": 0.0, "credible_interval_95": [0.0, 0.5], "tier": "YETERSİZ VERİ"},
            "moral_foundations_profile": {"dominant_foundation": "YETERSİZ VERİ", "credible_interval_95": [0.0, 0.5]},
            "caps_contextual_matrix": {"contextual_plasticity_index": 0.0, "contexts": {}, "if_then_signatures": []},
            "stated_vs_revealed_preference": {
                "stated_cooperation_polarity": 0.5,
                "revealed_cooperation_polarity": 0.5,
                "preference_divergence_index": 0.0,
                "verdict": "YETERSİZ VERİ",
            },
            "top_characteristic_terms": [],
        }
