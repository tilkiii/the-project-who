"""
Unit and integration tests for Extended Psychometrics, AI Scientific Referee,
Knowledge Bank, Boundary Synthesizer, Negation-Aware Bayesian Psychometrics,
CAPS Contextual Matrix, and Stated vs. Revealed Preference Divergence.
"""

from pathlib import Path
import json
import pytest

from who.learn.feature_extractor import FeatureExtractor
from who.learn.scientific_referee import ScientificReferee, fisher_z, inv_fisher_z
from who.learn.boundary_synthesizer import BoundarySynthesizer
from who.learn.autonomous_researcher import AutonomousAcademicResearcher
from who.learn.skill_synthesizer import SkillSynthesizer


def test_feature_extractor_extended_psychometrics(tmp_path: Path):
    """Test calculation of 5 extended psychometric dimensions across Turkish & English text."""
    fe = FeatureExtractor(output_dir=tmp_path)
    corpus = [
        "Bizim için stratejik menfaat ve pragmatik fayda esastır. Bu kozu kullanarak avantaj sağlamalıyız.",
        "Kural ve netlik olmadan kaos çıkar; kesin bir karar ve mutlak düzen talep ediyorum.",
        "Herhangi bir zarar, tehlike veya kayıp ihtimali bizi felakete sürükleyebilir; zararı önlemek şart.",
        "Bu dehşet verici durum karşısında öfke ve hiddetle hareket etmeyeceğiz, tam bir sükûnet ve soğukkanlılıkla yaklaşacağız.",
        "Mazlumların hakkı ve adalet her şeyin üstündedir; bireysel şefkat ve hakkaniyet vazgeçilmezimizdir."
    ]

    features = fe.extract_features(corpus)

    # 1. Machiavellianism
    assert "machiavellianism_index" in features
    mach = features["machiavellianism_index"]
    assert "score" in mach
    assert "tier" in mach
    assert "credible_interval_95" in mach
    assert len(mach["credible_interval_95"]) == 2
    assert mach["machiavellian_hits"] > 0
    assert "Machiavellianism" in mach["probabilistic_statement"]

    # 2. Need for Closure
    assert "need_for_closure_index" in features
    closure = features["need_for_closure_index"]
    assert "score" in closure
    assert "credible_interval_95" in closure
    assert closure["closure_hits"] > 0
    assert "Need for Cognitive Closure" in closure["probabilistic_statement"]

    # 3. Loss Aversion
    assert "loss_aversion_index" in features
    loss = features["loss_aversion_index"]
    assert "loss_to_gain_ratio" in loss
    assert "credible_interval_95" in loss
    assert loss["loss_hits"] > 0
    assert "Prospect Risk Orientation" in loss["probabilistic_statement"]

    # 4. Affective Volatility
    assert "affective_volatility_index" in features
    volatility = features["affective_volatility_index"]
    assert "volatility_score" in volatility
    assert "composure_score" in volatility
    assert "credible_interval_95" in volatility
    assert "Affective Balance" in volatility["probabilistic_statement"]

    # 5. Moral Foundations
    assert "moral_foundations_profile" in features
    moral = features["moral_foundations_profile"]
    assert "dominant_foundation" in moral
    assert "credible_interval_95" in moral
    assert moral["individualizing_hits"] > 0

    # 6. CAPS Contextual Matrix & Stated vs. Revealed Preference
    assert "caps_contextual_matrix" in features
    assert "stated_vs_revealed_preference" in features

    assert (tmp_path / "features.json").exists()


def test_negation_aware_psychometrics_and_bayesian_ci(tmp_path: Path):
    """Verify syntactic negation reversal and Bayesian 95% credible interval shrinkage."""
    fe = FeatureExtractor(output_dir=tmp_path)

    # Corpus where threat/loss words are explicitly negated ("korku yok", "tehlike değil", "not a threat")
    negated_loss_corpus = [
        "Hiçbir zaman korku yok ve asla tehlike değil; biz daima kazanç, fırsat ve başarı odaklıyız.",
        "We are never afraid of risk and this is not a threat; we pursue opportunity, reward, and growth.",
    ]
    # Corpus where threat/loss words are affirmatively asserted
    affirmative_loss_corpus = [
        "Büyük bir korku, tehlike, zarar ve yıkım tehdidi altındayız; kayıp ve kriz kaçınılmaz.",
        "Severe danger, threat, loss, and crisis are imminent; we must avoid damage and catastrophe.",
    ]

    neg_features = fe.extract_features(negated_loss_corpus)
    aff_features = fe.extract_features(affirmative_loss_corpus)

    neg_loss = neg_features["loss_aversion_index"]
    aff_loss = aff_features["loss_aversion_index"]

    # Negated threat words should result in significantly lower loss aversion than affirmative threat words
    assert neg_loss["negated_reversals"] > 0
    assert neg_loss["estimated_score"] < aff_loss["estimated_score"]

    # Verify Bayesian 95% Credible Interval shrinks as corpus sample size grows
    large_corpus = affirmative_loss_corpus * 25
    large_features = fe.extract_features(large_corpus)
    small_margin = aff_loss["uncertainty_margin"]
    large_margin = large_features["loss_aversion_index"]["uncertainty_margin"]
    assert large_margin < small_margin


def test_caps_matrix_and_stated_vs_revealed_preference(tmp_path: Path):
    """Verify Mischel's CAPS IF-THEN matrix and Samuelson's Stated vs. Revealed Preference Divergence."""
    fe = FeatureExtractor(output_dir=tmp_path)
    corpus = [
        "Teorik metodoloji, bilimsel hipotez ve akademik literatür çerçevesinde olasılıksal analiz şarttır.",
        "Acil kriz, tehdit ve olağanüstü kaos durumunda derhal kesin karar ve net kural uygulanmalıdır.",
        "Müzakere masasında stratejik anlaşma, kontrat ve pazarlık kozlarımızı korumalıyız.",
        "Biz her zaman uzlaşı, diyalog, hoşgörü ve ortak mutabakat ilkelerine bağlılığımızı beyan ederiz.",
        "Ancak uygulamada teklifi reddetti, anlaşmadan çekildi, veto etti ve bağımsız karar alarak iptal etti.",
    ]
    features = fe.extract_features(corpus)

    caps = features["caps_contextual_matrix"]
    assert "contextual_plasticity_index" in caps
    assert "contexts" in caps
    assert "FORMAL_ACADEMIC" in caps["contexts"]
    assert "HIGH_STAKES_CRISIS" in caps["contexts"]
    assert len(caps["if_then_signatures"]) >= 1

    pref = features["stated_vs_revealed_preference"]
    assert "stated_cooperation_polarity" in pref
    assert "revealed_cooperation_polarity" in pref
    assert "preference_divergence_index" in pref
    # Rhetoric is cooperative ("uzlaşı, diyalog, hoşgörü") while action is autonomous exit ("reddetti, çekildi, veto")
    assert pref["stated_cooperation_polarity"] > pref["revealed_cooperation_polarity"]
    assert pref["preference_divergence_index"] > 0.10


def test_scientific_referee_scoring_and_ranking(tmp_path: Path):
    """Test AI Peer-Review rigor scoring, filtering, and consensus pooling."""
    referee = ScientificReferee(profiles_dir=tmp_path)

    gold_paper = {
        "title": "Large-Scale Computational Psycholinguistics of Strategic Behavior",
        "abstract": "We analyze N = 4,500 participants and find significant correlation r = 0.44 (p < 0.001) between Machiavellianism and instrumental utility language.",
        "source": "nature",
        "sample_size": 4500,
        "r": 0.44,
        "p_value": 0.0001,
        "venue": "nature human behaviour",
        "methodology": "multicenter longitudinal cohort",
        "url": "https://doi.org/10.1038/s41562-024-001",
    }

    weak_paper = {
        "title": "Exploratory pilot observation of 15 students",
        "abstract": "Small exploratory sample N = 15 shows weak r = 0.05, p = 0.42.",
        "source": "arxiv",
        "sample_size": 15,
        "r": 0.05,
        "p_value": 0.42,
        "venue": "arxiv",
        "methodology": "pilot survey",
        "url": "https://arxiv.org/abs/2401.99999",
    }

    gold_eval = referee.score_paper_rigor(gold_paper, marker="machiavellianism_index", prior_mean_r=0.40)
    weak_eval = referee.score_paper_rigor(weak_paper, marker="machiavellianism_index", prior_mean_r=0.40)

    assert gold_eval["rigor_tier"] == "GOLD_STANDARD"
    assert gold_eval["scores"]["total_rigor_score"] >= 70.0
    assert gold_eval["scores"]["sample_size_score"] > 25.0
    assert gold_eval["scores"]["venue_score"] == 20.0

    assert weak_eval["rigor_tier"] in ("REJECTED_LOW_RIGOR", "EXPLORATORY_PREPRINT")
    assert weak_eval["scores"]["total_rigor_score"] < gold_eval["scores"]["total_rigor_score"]

    corpus_result = referee.evaluate_and_rank_corpus([weak_paper, gold_paper], marker="machiavellianism_index")
    assert corpus_result["total_evaluated"] == 2
    assert corpus_result["approved_count"] >= 1
    assert len(corpus_result["gold_standard_papers"]) == 1
    assert corpus_result["gold_standard_papers"][0]["title"] == gold_paper["title"]

    kb_path = referee.maintain_knowledge_bank(
        marker="machiavellianism_index",
        evaluation_result=corpus_result,
        synthesized_boundaries={"norm": "calibrated"}
    )
    assert kb_path.exists()
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    assert "machiavellianism_index" in kb_data["metrics"]
    assert len(kb_data["metrics"]["machiavellianism_index"]["gold_standard_studies"]) == 1


def test_fisher_z_transformation():
    """Verify Fisher z transformation and inverse round-trip accuracy."""
    test_r = 0.45
    z = fisher_z(test_r)
    r_recovered = inv_fisher_z(z)
    assert round(r_recovered, 4) == test_r


def test_boundary_synthesizer_rules(tmp_path: Path):
    """Test boundary synthesis and behavioral simulation rule derivation."""
    bs = BoundarySynthesizer(profiles_dir=tmp_path)

    boundaries = bs.synthesize_metric_boundaries(
        marker="machiavellianism_index",
        consensus_r=0.44,
        gold_papers=[{"title": "Test"}]
    )
    assert "calibrated_distribution" in boundaries
    assert "classification_tiers" in boundaries
    assert boundaries["calibrated_distribution"]["extreme_cutoff"] > 0

    test_features = {
        "machiavellianism_index": {"score": 0.65, "tier": "YÜKSEK / MAKYAVELİST"},
        "need_for_closure_index": {"score": 0.58, "tier": "YÜKSEK KAPALILIK"},
        "loss_aversion_index": {"loss_to_gain_ratio": 1.85, "orientation": "KAYIPTAN KAÇINMA"},
        "affective_volatility_index": {"volatility_score": 0.15, "composure_score": 0.85},
        "moral_foundations_profile": {"dominant_foundation": "BİREY & HAKKANİYET ODAKLI (INDIVIDUALIZING_FAIRNESS)"},
        "epistemic_manipulation_index": {"score": 0.48},
    }

    rules = bs.generate_behavioral_rules(test_features)
    rule_ids = [r["rule_id"] for r in rules]

    assert "TACTICAL_TRANSACTIONALISM" in rule_ids
    assert "RIGID_COGNITIVE_CLOSURE" in rule_ids
    assert "ASYMMETRIC_LOSS_PREVENTION" in rule_ids
    assert "STOIC_EQUANIMITY" in rule_ids
    assert "INDIVIDUAL_RIGHTS_PRIMACY" in rule_ids
    assert "RHETORICAL_ASSERTION_DOMINANCE" in rule_ids

    target_dir = tmp_path / "test_target"
    out_file = bs.save_target_boundaries_and_rules(target_dir, {"mach": boundaries}, rules)
    assert out_file.exists()


def test_autonomous_researcher_cycle_integration(tmp_path: Path, monkeypatch):
    """Integration test of autonomous researcher pipeline with referee and boundaries."""
    target_dir = tmp_path / "test_profile"
    researcher = AutonomousAcademicResearcher(output_dir=target_dir, profiles_dir=tmp_path)

    monkeypatch.setattr(researcher, "query_arxiv", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_pubmed", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_semantic_scholar", lambda q, max_results=1: [])
    monkeypatch.setattr(researcher, "query_crossref", lambda q, max_results=1: [])

    test_features = {
        "pronoun_ratios": {"first_person_singular_pct": 4.2, "first_person_plural_pct": 0.8, "i_to_we_ratio": 3.5},
        "epistemic_markers": {"epistemic_calibration_ratio": 0.45, "certainty_density": 2.5},
        "emotional_cadence": {"critical_density": 1.8},
        "epistemic_manipulation_index": {"score": 0.52},
        "chronomics": {"circadian_drift_summary": {"epistemic_filter_decay_pct": -18.0}},
        "machiavellianism_index": {"score": 0.60, "tier": "YÜKSEK / MAKYAVELİST"},
        "need_for_closure_index": {"score": 0.56, "tier": "YÜKSEK KAPALILIK"},
        "loss_aversion_index": {"loss_to_gain_ratio": 1.75},
        "affective_volatility_index": {"volatility_score": 0.48, "composure_score": 0.52},
        "moral_foundations_profile": {"dominant_foundation": "BİREY & HAKKANİYET ODAKLI"},
    }

    anomalies = researcher.detect_behavioral_anomalies(test_features, [])
    marker_names = [a["marker"] for a in anomalies]

    assert "machiavellian_strategic_orientation" in marker_names
    assert "high_need_for_cognitive_closure" in marker_names
    assert "heightened_loss_aversion_threat_vigilance" in marker_names
    assert "affective_volatility_reactivity" in marker_names
    assert "moral_foundation_individual_fairness" in marker_names

    queries = researcher.generate_scientific_queries(anomalies)
    assert len(queries) == len(anomalies)
    assert len(queries[0]["queries"]) == 3

    papers = researcher.research_markers(test_features, [], max_papers_per_query=1)
    assert len(papers) > 0

    assert (target_dir / "retrieved_academic_literature.json").exists()
    assert (target_dir / "behavioral_boundaries.json").exists()
    assert (tmp_path / "scientific_knowledge_bank.json").exists()

    ss = SkillSynthesizer(output_dir=target_dir)
    skill = ss.synthesize_mapping_skill("Test Target", test_features, [], papers)
    assert (target_dir / "generated_mapping_skill.json").exists()
    assert "extended_psychometrics" in skill["behavioral_profile"]
    assert any("[TACTICAL_TRANSACTIONALISM]" in r for r in skill["generation_rules"])


def test_feature_extractor_contraction_and_clause_bounded_negation():
    """Verify English contraction negations stay bounded within their clause and don't leak across 'but'."""
    fe = FeatureExtractor()

    # Contractions must negate within their clause
    neg_deal = fe.extract_features(["I can't afford this overpriced phone and it isn't worth the asking price."])
    pos_deal = fe.extract_features(["I can afford this clean bargain phone and it is definitely worth the price."])
    assert neg_deal["loss_aversion_index"]["negated_reversals"] >= 2
    assert neg_deal["loss_aversion_index"]["estimated_score"] > pos_deal["loss_aversion_index"]["estimated_score"]

    # Contrast conjunction 'but' must block negation leakage from prior clause onto 'fair'
    split_clause = fe.extract_features(["I don't think so, but this is a fair and reasonable deal."])
    assert split_clause["machiavellianism_index"]["negated_reversals"] == 0
    assert split_clause["machiavellianism_index"]["cooperation_hits"] >= 2

    # Turkish clause negation & suffix recognition
    tr_neg = fe.extract_features(["Bu teklifi asla kabul etmem, fiyat makul değil ama teşekkür ederim."])
    assert tr_neg["machiavellianism_index"]["negated_reversals"] >= 1
    assert tr_neg["machiavellianism_index"]["machiavellian_hits"] >= 1
    assert tr_neg["machiavellianism_index"]["cooperation_hits"] >= 1


def test_feature_extractor_negated_stances_and_everyday_lexicons():
    """Verify negation reversal across core psychometric dimensions and zero empty crashes."""
    fe = FeatureExtractor()
    firm_msg = [
        "Hi, thanks for messaging. Wait, this is not fair and I cannot accept an overpriced risk.",
        "Our core policy is strictly firm and fixed under uncertainty.",
    ]
    flex_msg = [
        "Hello! Could you maybe consider an open flexible alternative together?",
        "I am willing to discuss nuance and adapt our approach.",
    ]

    f_firm = fe.extract_features(firm_msg)
    f_flex = fe.extract_features(flex_msg)

    assert f_firm["machiavellianism_index"]["negated_reversals"] >= 1
    assert f_firm["machiavellianism_index"]["machiavellian_hits"] >= 1
    assert f_firm["need_for_closure_index"]["closure_hits"] > 0
    assert f_flex["need_for_closure_index"]["openness_hits"] > 0
    assert f_firm["loss_aversion_index"]["loss_hits"] > 0

    empty_feat = fe.extract_features([])
    assert empty_feat["corpus_stats"]["total_words"] == 0
    assert empty_feat["machiavellianism_index"]["score"] == 0.0


def test_feature_extractor_negation_precision_guards():
    """Verify false-negation guards on verbal nouns, attitude verbs, 'not only', and 'rarely'."""
    fe = FeatureExtractor()

    # Turkish affirmative verbal noun (-mem) must NOT negate following lexicon token
    tr_obl = fe.extract_features(["Zamanında ödemem kazanç sağlar."])
    assert tr_obl["loss_aversion_index"]["negated_reversals"] == 0
    assert tr_obl["loss_aversion_index"]["gain_hits"] >= 1

    # Explicit 1st-person negative verb 'vermem' must negate 'güven'
    tr_refusal = fe.extract_features(["Sözlerine güven vermem."])
    assert tr_refusal["machiavellianism_index"]["negated_reversals"] >= 1
    assert tr_refusal["machiavellianism_index"]["machiavellian_hits"] >= 1

    # Attitude verb 'think' must not bridge negation across embedded content
    en_think = fe.extract_features(["I don't think it is risky."])
    assert en_think["loss_aversion_index"]["negated_reversals"] == 0

    # Focus particle 'not only' and 'no less than' must not invert polarity
    en_not_only = fe.extract_features(["This offer is not only fair, it is also reasonable."])
    assert en_not_only["machiavellianism_index"]["negated_reversals"] == 0
    assert en_not_only["machiavellianism_index"]["cooperation_hits"] >= 2

    en_no_less = fe.extract_features(["No less than a fair settlement is expected."])
    assert en_no_less["machiavellianism_index"]["negated_reversals"] == 0

    # Frequency adverbs 'rarely'/'seldom' must not invert 'risk' into gain
    en_rarely = fe.extract_features(["We rarely face severe risk."])
    assert en_rarely["loss_aversion_index"]["negated_reversals"] == 0
    assert en_rarely["loss_aversion_index"]["loss_hits"] >= 1
