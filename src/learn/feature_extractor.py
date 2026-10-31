"""
Linguistic, psychometric, and behavioral feature extraction.
Computes metrics including pronoun distribution, lexical diversity,
epistemic hedging ratios, and extended psychometric dimensions.
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

    def extract_features(
        self,
        texts: List[str],
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive linguistic, epistemic manipulation, and psychological metrics."""
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
        i_pronouns = {"i", "me", "my", "mine", "myself", "ben", "bana", "beni", "benim", "kendim"}
        we_pronouns = {"we", "us", "our", "ours", "ourselves", "biz", "bize", "bizi", "bizim", "kendimiz"}
        you_pronouns = {"you", "your", "yours", "yourself", "yourselves", "sen", "siz", "sana", "size", "senin", "sizin"}
        they_pronouns = {
            "they", "them", "their", "theirs", "themselves", "he", "him", "his", "she", "her", "hers",
            "onlar", "onlara", "onlari", "onları", "onlarin", "onların", "o", "ona", "onu", "onun"
        }

        i_count = sum(1 for w in words if w in i_pronouns)
        we_count = sum(1 for w in words if w in we_pronouns)
        you_count = sum(1 for w in words if w in you_pronouns)
        they_count = sum(1 for w in words if w in they_pronouns)

        # 2. Epistemic stance words
        certainty_lexicon = {
            "definitely", "certainly", "undoubtedly", "always", "never", "obvious", "clear", "absolute", "guaranteed",
            "kesinlikle", "asla", "şüphesiz", "suphesiz", "mutlaka", "apaçık", "apack", "tartışmasız", "tartismasiz"
        }
        tentative_lexicon = {
            "likely", "plausible", "perhaps", "maybe", "possible", "suggests", "probable", "might", "could", "seems",
            "olası", "olasi", "muhtemel", "belki", "sanırım", "sanirim", "görünüyor", "gorunuyor", "işaret", "varsayım"
        }
        causal_lexicon = {
            "because", "therefore", "thus", "hence", "consequently", "implies", "since", "results",
            "çünkü", "cunku", "dolayısıyla", "dolayisiyla", "böylece", "boylece", "nedeniyle", "sonucunda"
        }

        certainty_count = sum(1 for w in words if w in certainty_lexicon)
        tentative_count = sum(1 for w in words if w in tentative_lexicon)
        causal_count = sum(1 for w in words if w in causal_lexicon)

        # 3. Emotional valence indicators
        positive_lexicon = {
            "great", "excellent", "superb", "constructive", "insightful", "promising", "breakthrough", "success", "effective", "valuable",
            "harika", "mükemmel", "mukemmel", "başarılı", "basarili", "faydalı", "faydali", "önemli", "onemli", "değerli", "degerli"
        }
        critical_lexicon = {
            "flawed", "erroneous", "inefficient", "risk", "failure", "deceptive", "critical", "vulnerable", "concern", "problem",
            "hatalı", "hatali", "yanlış", "yanlis", "başarısız", "basarisiz", "risk", "tehlike", "sorun", "kriz", "çelişki", "celiski"
        }

        pos_count = sum(1 for w in words if w in positive_lexicon)
        crit_count = sum(1 for w in words if w in critical_lexicon)

        # 4. Lexical diversity (Type-Token Ratio)
        unique_words = len(set(words))
        ttr = unique_words / total_words

        # 5. Rhetorical syntax metrics
        avg_sentence_len = total_words / total_sentences
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

        # 7. Epistemic Gaslighting & Manipulation Index (Capability #3)
        manipulation_metrics = self._calculate_manipulation_index(combined_text, words, total_words)

        # 8. Optional Cognitive Chronomics Integration (Capability #1)
        chronomics_data: Dict[str, Any] = {}
        if documents:
            try:
                from src.learn.chronomics import CognitiveChronomics
                cc = CognitiveChronomics(output_dir=self.output_dir)
                chronomics_data = cc.analyze_circadian_drift(documents)
            except Exception as e:
                logger.debug(f"Chronomics execution note: {e}")

        # 9. Extended Psychometrics (Machiavellianism, Need for Closure, Loss Aversion, Affective Volatility, Moral Foundations)
        machiavellianism_data = self._calculate_machiavellianism(words, total_words)
        closure_data = self._calculate_need_for_closure(words, total_words)
        loss_aversion_data = self._calculate_loss_aversion(words, total_words)
        volatility_data = self._calculate_affective_volatility(sentences, words, total_words)
        moral_foundations_data = self._calculate_moral_foundations(words, total_words)

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
            "epistemic_manipulation_index": manipulation_metrics,
            "machiavellianism_index": machiavellianism_data,
            "need_for_closure_index": closure_data,
            "loss_aversion_index": loss_aversion_data,
            "affective_volatility_index": volatility_data,
            "moral_foundations_profile": moral_foundations_data,
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
        """
        Quantify the Epistemic Manipulation & Gaslighting Index.
        Measures dogmatic/absolutist linguistic density vs. rational/evidence-based qualifiers,
        detecting unsupported certainty patterns.
        """
        lower_text = full_text.lower()

        # Dogmatic / Absolutist Lexicon & Phrases
        dogmatic_terms = [
            "kesinlikle", "asla", "şüphesiz", "suphesiz", "herkes biliyor ki", "herkes bilir ki",
            "apaçık ortada", "apack ortada", "tartışmasız", "tartismasiz", "hiç kimse inkar edemez",
            "hic kimse inkar edemez", "tek gerçek", "tek gercek", "mutlaka", "başka yolu yok",
            "obviously", "undoubtedly", "everyone knows that", "unquestionable", "indisputable",
            "absolute truth", "beyond question", "without any doubt", "purely", "guaranteed"
        ]

        # Rational / Evidence-Grounded Qualifiers
        evidence_terms = [
            "veri gösteriyor ki", "veri gosteriyor ki", "kanıtlar ışığında", "kanitlar isiginda",
            "araştırmalara göre", "arastirmalara gore", "gözlemlerle uyumlu", "gozlemlerle uyumlu",
            "olasılıkla", "olasilikla", "plausibly", "belki", "varsayım altında", "varsayim altinda",
            "evidence suggests", "data indicates", "according to findings", "empirically",
            "statistically", "probabilistically", "based on observed", "tentatively"
        ]

        # Multi-word phrase matches + single word counts
        dogmatic_matches = 0
        for dt in dogmatic_terms:
            dogmatic_matches += lower_text.count(dt)

        evidence_matches = 0
        for et in evidence_terms:
            evidence_matches += lower_text.count(et)

        # Basic word-level fallbacks if phrases were not found
        if dogmatic_matches == 0:
            single_dogmatic = {"kesinlikle", "asla", "şüphesiz", "mutlaka", "definitely", "certainly"}
            dogmatic_matches = sum(1 for w in words if w in single_dogmatic)

        if evidence_matches == 0:
            single_evidence = {"olası", "muhtemel", "belki", "plausible", "likely", "suggests"}
            evidence_matches = sum(1 for w in words if w in single_evidence)

        # Density per 1,000 words
        dogmatic_density = round((dogmatic_matches / total_words) * 1000, 2)
        evidence_density = round((evidence_matches / total_words) * 1000, 2)

        # Epistemic manipulation index: ratio of dogmatism to (evidence + 1)
        raw_index = (dogmatic_matches * 1.5) / (dogmatic_matches + evidence_matches + 2.0)
        manipulation_index = round(min(1.0, max(0.0, raw_index)), 2)

        # Qualitative Risk Classification
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
            "score_pct": f"{int(manipulation_index * 100)}%",
            "dogmatic_density_per_k": dogmatic_density,
            "evidence_density_per_k": evidence_density,
            "risk_tier": risk_tier,
            "confidence_score": round(confidence, 2),
            "probabilistic_statement": (
                f"Manipülasyon Endeksi: Kanıtsız iddialarda mutlak kesinlik dili kullanımı eğilimi "
                f"%{int(manipulation_index * 100)} seviyesindedir ({risk_desc} Olasılık Skoru: %{int(confidence * 100)})."
            ),
        }

    def _calculate_machiavellianism(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Machiavellianism, pragmatic utility vs altruism/cooperation (Paulhus & Williams SD3)."""
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

        m_count = sum(1 for w in words if w in mach_lexicon)
        c_count = sum(1 for w in words if w in coop_lexicon)

        # Calibrated Machiavellianism ratio
        raw_score = (m_count * 2.2) / (m_count + c_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)

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
            "score_pct": f"{int(score * 100)}%",
            "tier": tier,
            "machiavellian_hits": m_count,
            "cooperation_hits": c_count,
            "probabilistic_statement": f"Makyavelizm & Güç Yönelimi: %{int(score * 100)} seviyesinde ({desc}).",
        }

    def _calculate_need_for_closure(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Need for Cognitive Closure (Webster & Kruglanski, 1994)."""
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

        cl_count = sum(1 for w in words if w in closure_lexicon)
        op_count = sum(1 for w in words if w in openness_lexicon)

        raw_score = (cl_count * 1.8) / (cl_count + op_count + 3.0)
        score = round(min(1.0, max(0.0, raw_score)), 2)

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
            "score_pct": f"{int(score * 100)}%",
            "tier": tier,
            "closure_hits": cl_count,
            "openness_hits": op_count,
            "probabilistic_statement": f"Bilişsel Kapanma İhtiyacı: %{int(score * 100)} seviyesinde ({desc}).",
        }

    def _calculate_loss_aversion(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Prospect Theory Loss Aversion vs Opportunity Seeking (Kahneman & Tversky)."""
        loss_lexicon = {
            "loss", "risk", "threat", "cost", "failure", "collapse", "danger", "damage", "crisis", "vulnerable",
            "kayıp", "kayip", "risk", "tehdit", "zarar", "çöküş", "cokus", "tehlike", "kriz", "felaket", "hasar"
        }
        gain_lexicon = {
            "gain", "opportunity", "profit", "growth", "upside", "win", "breakthrough", "benefit", "advance",
            "kazanç", "kazanc", "fırsat", "firsat", "kâr", "kar", "büyüme", "buyume", "atılım", "avantaj", "getiri"
        }

        l_count = sum(1 for w in words if w in loss_lexicon)
        g_count = sum(1 for w in words if w in gain_lexicon)
        ratio = round(l_count / (g_count or 1), 2)

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
            "loss_hits": l_count,
            "gain_hits": g_count,
            "orientation": orientation,
            "probabilistic_statement": f"Prospekt Risk Yönelimi: Kayıp/Kazanç oranı {ratio} ({desc}).",
        }

    def _calculate_affective_volatility(self, sentences: List[str], words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Emotional Volatility vs Stoic Composure across sentence transitions."""
        arousal_lexicon = {
            "fury", "panic", "rage", "ecstatic", "terrifying", "devastating", "catastrophic", "shocking",
            "öfke", "ofke", "panik", "dehşet", "dehset", "hiddet", "çıldırma", "rezalet", "coşku", "cosku"
        }
        calm_lexicon = {
            "serene", "calm", "steady", "balanced", "composed", "tranquil", "objective",
            "sakin", "soğukkanlı", "sogukkanli", "itidalli", "dengeli", "ölçülü", "olculu", "nesnel"
        }

        arousal_count = sum(1 for w in words if w in arousal_lexicon)
        calm_count = sum(1 for w in words if w in calm_lexicon)

        # Volatility factor based on emotional extremes
        volatility_score = round(min(1.0, (arousal_count * 2.5) / (total_words / 100.0 + 1.0)), 2)
        composure_score = round(1.0 - volatility_score, 2)

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
            "tier": tier,
            "probabilistic_statement": f"Duygusal Denge & Afektif Denge: {tier} ({desc}).",
        }

    def _calculate_moral_foundations(self, words: List[str], total_words: int) -> Dict[str, Any]:
        """Quantify Moral Foundations profile (Haidt & Graham: Fairness/Care vs Authority/Loyalty)."""
        individualizing_lexicon = {
            "care", "harm", "protect", "suffer", "justice", "fair", "rights", "equality", "oppression",
            "hak", "adalet", "koruma", "şefkat", "sefkat", "eşitlik", "esitlik", "mazlum", "zulüm", "özgürlük"
        }
        binding_lexicon = {
            "authority", "obey", "duty", "loyalty", "respect", "tradition", "order", "honor", "treason",
            "otorite", "itaat", "görev", "gorev", "sadakat", "saygı", "saygi", "gelenek", "nizam", "hiyerarşi"
        }

        ind_count = sum(1 for w in words if w in individualizing_lexicon)
        bind_count = sum(1 for w in words if w in binding_lexicon)

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
            "dominant_foundation": foundation,
            "probabilistic_statement": f"Ahlaki Temeller Matrisi: {foundation} ({desc}).",
        }

    def _empty_feature_set(self) -> Dict[str, Any]:
        return {
            "corpus_stats": {"total_words": 0, "total_sentences": 0, "type_token_ratio": 0.0},
            "pronoun_ratios": {"first_person_singular_pct": 0.0, "i_to_we_ratio": 0.0},
            "epistemic_markers": {"epistemic_calibration_ratio": 1.0},
            "epistemic_manipulation_index": {
                "score": 0.0,
                "score_pct": "0%",
                "risk_tier": "YETERSİZ VERİ",
                "probabilistic_statement": "Yetersiz korpus nedeniyle manipülasyon endeksi hesaplanamadı."
            },
            "machiavellianism_index": {"score": 0.0, "tier": "YETERSİZ VERİ"},
            "need_for_closure_index": {"score": 0.0, "tier": "YETERSİZ VERİ"},
            "loss_aversion_index": {"loss_to_gain_ratio": 1.0, "orientation": "YETERSİZ VERİ"},
            "affective_volatility_index": {"volatility_score": 0.0, "tier": "YETERSİZ VERİ"},
            "moral_foundations_profile": {"dominant_foundation": "YETERSİZ VERİ"},
            "top_characteristic_terms": [],
        }
