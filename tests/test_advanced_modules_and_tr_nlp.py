"""
Unit & CLI Integration Tests for:
1. Turkish NLP & Unicode Normalization (tr_lower, 'İ' token-split prevention, Turkish slugs, TCKN PII redaction)
2. Zero-Hallucination Cyber Footprint Auditing (no fabricated LinkedIn/Dropbox/Adobe breaches)
3. All 8 Advanced Domain Modules:
   - ForensicStylometryEngine (who stylometry)
   - CryptographicIdentityAnchor (who anchor)
   - ToulminArgumentMiner (who arguments)
   - GameTheoreticNegotiator (who negotiate)
   - CognitiveBiasAuditor (who blindspots)
   - WaybackDigitalArchaeologist (who archaeology)
   - CodeIntelligenceHarvester (who code-osint)
   - TemporalBacktester (who backtest)
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.engine.backtester import TemporalBacktester
from src.engine.negotiation_engine import GameTheoreticNegotiator
from src.harvest.code_intelligence import CodeIntelligenceHarvester
from src.harvest.cyber_footprint import CyberFootprintAuditor
from src.harvest.identity_anchor import CryptographicIdentityAnchor
from src.harvest.wayback_archaeologist import WaybackDigitalArchaeologist
from src.learn.cognitive_bias_auditor import CognitiveBiasAuditor
from src.learn.feature_extractor import FeatureExtractor
from src.learn.stylometry_engine import ForensicStylometryEngine
from src.learn.topic_mapper import TopicMapper
from src.utils.entity_validator import normalize_name
from src.utils.text_cleaner import redact_pii, sanitize_slug, tokenize_words, tr_lower
from src.vault.argument_miner import ToulminArgumentMiner

runner = CliRunner()


def test_turkish_unicode_lowercasing_and_no_fake_english_i_pronoun(tmp_path: Path):
    """Ensure capitalized Turkish 'İ' never splits into 'i' + combining dot or inflates English 'I' pronouns."""
    sample_tr = "İstanbul ve İzmir İçin İnsan Odaklı İnovasyon İlkeleri Kanıtlar Işığında ŞÜPHESİZ İncelendi."
    lowered = tr_lower(sample_tr)
    assert "\u0307" not in lowered
    assert "istanbul" in lowered
    assert "için" in lowered
    assert "ışığında" in lowered
    assert "şüphesiz" in lowered

    tokens = tokenize_words(sample_tr)
    assert "i" not in tokens

    fe = FeatureExtractor(output_dir=tmp_path)
    features = fe.extract_features([sample_tr])
    # Zero first-person singular pronouns exist in sample_tr!
    assert features["pronoun_ratios"]["first_person_singular_pct"] == 0.0

    # English 'I' pronoun must still work cleanly
    en_features = fe.extract_features(["I think I can verify this empirical hypothesis myself."])
    assert en_features["pronoun_ratios"]["first_person_singular_pct"] > 15.0


def test_turkish_slug_and_pii_redaction():
    """Verify Turkish dotless 'ı' and dotted 'İ' transliterate to 'i' in slugs, and TCKN is redacted as ID."""
    assert sanitize_slug("Işık Kılıç") == "isik_kilic"
    assert sanitize_slug("İsmail Çelik") == "ismail_celik"
    assert normalize_name("İSMAİL IŞIK") == "ismail isik"

    masked = redact_pii("TCKN: 12345678902 ve tel +90 532 111 22 33")
    assert "[REDACTED_ID]" in masked
    assert "[REDACTED_PHONE]" in masked


def test_turkish_topic_mapper_preserves_diacritics(tmp_path: Path):
    tm = TopicMapper(output_dir=tmp_path)
    docs = [
        {"content": "Özgürlükçü yönetişim ilkeleri ve bilişsel şeffaflık kurumsal güvenliği güçlendirir."},
        {"content": "Bilişsel şeffaflık sayesinde çoğulcu yönetişim modelleri sürdürülebilir hale gelir."},
        {"content": "Kurumsal şeffaflık ve yönetişim mekanizmaları denetlenebilir olmalıdır."},
    ]
    res = tm.extract_topics(docs, nr_topics=2)
    all_keywords = [kw for t in res["topics"] for kw in t["keywords"]]
    assert any("şeffaflık" in kw or "yönetişim" in kw or "bilişsel" in kw for kw in all_keywords)


def test_cyber_footprint_zero_hallucination_no_fake_breaches(monkeypatch):
    """Without HIBP_API_KEY, CyberFootprintAuditor must NEVER fabricate LinkedIn/Dropbox/Adobe breaches."""
    monkeypatch.delenv("HIBP_API_KEY", raising=False)
    dirs = settings.get_target_dirs("Zero Hallucination User")
    auditor = CyberFootprintAuditor("Zero Hallucination User", dirs)

    report = auditor.run_full_audit("fresh_2026_account@gmail.com")
    breach = report["breach_exposure"]
    assert breach["verification_status"] == "UNVERIFIED_NO_BREACH_API_KEY"
    assert breach["documented_breach_incidents"] == []
    assert breach["estimated_breaches_count"] == 0


def test_forensic_stylometry_engine_attribution(tmp_path: Path):
    engine = ForensicStylometryEngine(output_dir=tmp_path, vault_dir=tmp_path)
    docs = [
        {
            "source_id": "essay_1.md",
            "content": (
                "Ampirik doğrulama olmadan kurulan her kurumsal model belirsizliğe mahkumdur; dolayısıyla "
                "açık protokoller ve yanlışlanabilir hipotezler bizim temel dayanağımızdır. Çünkü şeffaf veri "
                "olmadan rasyonel uzlaşı kurulamaz!"
            ),
        },
        {
            "source_id": "essay_2.md",
            "content": (
                "Açık protokoller kurumsal şeffaflığı büyütür; dolayısıyla yanlışlanabilir hipotezler ve "
                "ampirik doğrulama bilimsel yönetişimin vazgeçilmez zemini olarak korunmalıdır. Çünkü kapalı "
                "sistemler rasyonel denetimi engeller!"
            ),
        },
    ]
    fp = engine.build_author_fingerprint("Stilometri Hedefi", docs)
    assert fp["status"] == "FINGERPRINT_READY"
    assert (tmp_path / "stylometry_fingerprint.json").exists()

    same_author_sample = (
        "Açık protokoller ve yanlışlanabilir hipotezler ampirik doğrulama sağlar; dolayısıyla "
        "şeffaf veri olmadan kurumsal yönetişim kurulamaz!"
    )
    different_author_sample = (
        "Hava bugün sıcak. Eve erken gittim. Kahve içtim. Televizyon izledim. Sonra hemen uyudum."
    )
    verdict_same = engine.verify_questioned_text(same_author_sample, fp)
    verdict_diff = engine.verify_questioned_text(different_author_sample, fp)
    assert verdict_same["authorship_probability"] > 0.55
    assert verdict_same["authorship_probability"] > verdict_diff["authorship_probability"]


def test_identity_anchor_rel_me_and_zero_substitution(monkeypatch):
    dirs = settings.get_target_dirs("Mucteba Özdemir")
    anchor = CryptographicIdentityAnchor("Mucteba Özdemir", dirs)

    def fake_http_get(url: str, accept: str = ""):
        if "mucteba.example" in url:
            return '<html><head><link rel="me" href="https://github.com/mucteba"/></head></html>'
        if url == "https://github.com/mucteba":
            return '<html><body><a rel="me" href="https://mucteba.example">Website</a></body></html>'
        if "api.github.com/users/namesake_handle" in url:
            return json.dumps({"login": "namesake_handle", "name": "Muhammet Mücteba Özdemir"})
        return None

    monkeypatch.setattr(anchor, "_http_get_text", fake_http_get)

    report = anchor.run_anchor_verification(
        website="https://mucteba.example",
        github="namesake_handle",
        pgp_fingerprint="ABCDEF0123456789ABCDEF0123456789ABCDEF01",
    )
    # Bidirectional rel='me' + PGP must be verified, while 'Muhammet Mücteba Özdemir' MUST be rejected!
    assert report["verified_anchors_count"] == 2
    assert report["rejected_namesakes_count"] == 1
    assert report["disambiguation_notice"] is not None


def test_toulmin_argument_miner_and_fcm_keystones(tmp_path: Path):
    miner = ToulminArgumentMiner("Argüman Hedefi", vault_dir=tmp_path)
    docs = [
        {
            "source_id": "discourse.md",
            "content": (
                "Merkeziyetçi tekeller toplumsal inovasyonu zayıflatır çünkü kapalı kapılar ardındaki "
                "kararlar bağımsız denetimi engeller ancak acil kriz dönemleri istisna oluşturabilir. "
                "Açık kaynak protokoller şeffaf denetimi güçlendirir dolayısıyla kurumsal güven kalıcı biçimde artar."
            ),
        }
    ]
    res = miner.mine_arguments(docs)
    assert res["total_arguments_extracted"] >= 2
    assert len(res["fuzzy_cognitive_map"]["keystone_assumptions"]) >= 1
    assert (tmp_path / "toulmin_arguments.json").exists()


def test_cognitive_bias_auditor_quotes_and_safeguards(tmp_path: Path):
    auditor = CognitiveBiasAuditor(output_dir=tmp_path, vault_dir=tmp_path)
    docs = [
        {
            "source_id": "speech.md",
            "content": (
                "Ya bu protokolü bugün tamamen kabul edeceğiz ya da tüm ekosistem çökecek, başka çare yok. "
                "Bununla birlikte ampirik kanıt ve yanlışlanabilir hipotez çerçevesinde yanılıyor olabilirim."
            ),
        }
    ]
    report = auditor.audit_corpus("Bias Hedefi", docs)
    bias_ids = [b["bias_id"] for b in report["detected_biases"]]
    assert "false_dilemma" in bias_ids
    assert report["total_safeguard_markers_found"] >= 1
    assert (tmp_path / "cognitive_blindspots.json").exists()


def test_wayback_digital_archaeologist_diff_and_zero_substitution():
    dirs = settings.get_target_dirs("Mucteba Özdemir")
    arch = WaybackDigitalArchaeologist("Mucteba Özdemir", dirs)

    snapshots = [
        {
            "timestamp": "2022-03-01",
            "content": (
                "Mücteba Özdemir açık kaynak yönetişim araştırmaları yürütmektedir. "
                "Kurumsal danışma kurulunda X Vakfı başdanışmanı olarak görev yapmaktadır. "
                "Merkezi sistemler kesinlikle başarısız olmaya mahkumdur."
            ),
        },
        {
            "timestamp": "2023-06-01",
            "content": "Kararı imzalayan Muhammet Mücteba Özdemir yeni görevine atandı.",
        },
        {
            "timestamp": "2025-10-01",
            "content": (
                "Mücteba Özdemir açık kaynak yönetişim araştırmaları yürütmektedir. "
                "Merkezi sistemler belki belirli koşullarda başarısız olma riski taşımaktadır."
            ),
        },
    ]

    report = arch.compare_snapshots(snapshots, source_url="https://example.org/bio")
    assert report["status"] == "ARCHIVAL_DIFF_COMPLETED"
    # Middle snapshot belonging to 'Muhammet Mücteba Özdemir' must be rejected!
    assert report["rejected_namesakes_count"] == 1
    assert report["snapshots_analyzed"] == 2
    assert any("X Vakfı" in d["deleted_quote"] for d in report["silent_deletions"])
    assert len(report["modified_revisions"]) >= 1


def test_code_intelligence_zero_substitution_gate(monkeypatch):
    dirs = settings.get_target_dirs("Mucteba Özdemir")
    ci = CodeIntelligenceHarvester("Mucteba Özdemir", dirs)

    monkeypatch.setattr(
        ci,
        "_github_get_json",
        lambda endpoint: {"login": "wrong_person", "name": "Muhammet Mücteba Özdemir"},
    )
    rejected = ci.analyze_github_account("wrong_person")
    assert rejected["status"] == "REJECTED_ZERO_SUBSTITUTION"
    assert not (dirs.raw_data_dir / "code_osint_wrong_person.md").exists()


def test_negotiation_and_temporal_backtest_cli():
    target = "Strategic Twin"
    dirs = settings.get_target_dirs(target)
    (dirs.raw_data_dir / "01_early.md").write_text(
        "---\ntimestamp: 2023-01-10\n---\n"
        "[TARGET_PRIMARY]: Açık kaynak protokoller kurumsal şeffaflık için gerekli ve doğru zemindir çünkü bağımsız doğrulama sağlar.\n"
        "[TARGET_PRIMARY]: Kapalı tekelci sistemleri reddediyorum çünkü denetimsiz yapılar tehlikeli riskler üretir.\n"
        "[TARGET_PRIMARY]: Şeffaf yönetişim ve açık protokoller uzun vadeli güven için önemli bir dayanak oluşturur.\n",
        encoding="utf-8",
    )
    (dirs.raw_data_dir / "02_recent.md").write_text(
        "---\ntimestamp: 2025-05-20\n---\n"
        "[TARGET_PRIMARY]: Açık protokoller ve şeffaf yönetişim kurumsal güven için gerekli ve doğru mekanizmadır.\n"
        "[TARGET_PRIMARY]: Tekelci kapalı sistemler hatalı ve tehlikeli olduğu için bu yaklaşımları reddediyorum.\n",
        encoding="utf-8",
    )
    (dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [{"name": "Açık Protokoller"}, {"name": "Kapalı Tekeller"}],
            "relationships": [
                {"source": target, "target": "Açık Protokoller", "relation": "ADVOCATES", "weight": 0.92},
                {"source": target, "target": "Kapalı Tekeller", "relation": "REJECTS", "weight": 0.95},
            ],
        }),
        encoding="utf-8",
    )

    res_stylo = runner.invoke(app, ["stylometry", "--target", target])
    assert res_stylo.exit_code == 0

    res_blind = runner.invoke(app, ["blindspots", "--target", target])
    assert res_blind.exit_code == 0

    res_args = runner.invoke(app, ["arguments", "--target", target])
    assert res_args.exit_code == 0

    res_neg = runner.invoke(app, ["negotiate", "--target", target, "--objective", "Açık protokoller üzerinde araştırma ortaklığı"])
    assert res_neg.exit_code == 0
    assert "Negotiation" in res_neg.output

    res_bt = runner.invoke(app, ["backtest", "--target", target, "--train-ratio", "0.6"])
    assert res_bt.exit_code == 0
    assert "Brier" in res_bt.output

