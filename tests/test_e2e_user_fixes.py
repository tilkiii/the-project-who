"""
Regression test suite for all P0/P1/P2 bugs identified during the real-user E2E audit.
Covers path traversal in purge, outside-CWD profiles resolution, Neo4j rel_type sanitization,
learn concurrency & paper authors, DialogueHarvester iOS/speaker/email parsing,
DocumentIngestor HTML/PDF guards, WebHarvester RSS URL detection, OfflineProvider routing,
honest chronomics/psych insights without fabricated percentages, GraphBuilder/TopicMapper hygiene,
VectorEmbedder source_id preservation, Turkish slugification, CLI ghost/crash guards, and MCP BOM/encoding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

import who
from who import get_tool_schemas, list_targets
from who.cli import app
from who.config import _default_profiles_dir, settings
from who.engine.arena import SwarmArena
from who.engine.psych_insights import PsychInsights
from who.engine.simulator import Simulator
from who.engine.twin_chat import TwinChat
from who.harvest.dialogue_harvester import DialogueHarvester
from who.harvest.document_ingestor import DocumentIngestor
from who.harvest.web_harvester import WebHarvester
from who.learn.autonomous_researcher import AutonomousAcademicResearcher
from who.learn.chronomics import CognitiveChronomics
from who.learn.drift_detector import TemporalDriftDetector
from who.learn.skill_synthesizer import SkillSynthesizer
from who.learn.topic_mapper import TopicMapper
from who.utils.text_cleaner import sanitize_slug
from who.vault.graph_builder import GraphBuilder
from who.vault.graph_store import GraphStore
from who.vault.mindset_profiler import MindsetProfiler
from who.vault.vector_embedder import VectorEmbedder

runner = CliRunner()
_ORIG_QUERY_ARXIV = AutonomousAcademicResearcher.query_arxiv


def test_purge_blocks_path_traversal(tmp_path: Path):
    """who purge must refuse paths escaping profiles_dir (e.g. ../canary_victim)."""
    parent_dir = settings.profiles_dir.parent
    canary = parent_dir / "canary_victim_e2e"
    canary.mkdir(parents=True, exist_ok=True)
    (canary / "keep.txt").write_text("important", encoding="utf-8")

    for malicious in ("../canary_victim_e2e", "..", ".", "sub/../canary_victim_e2e"):
        res = runner.invoke(app, ["purge", "--target", malicious, "--force"])
        assert res.exit_code == 1
        assert canary.exists()
        assert (canary / "keep.txt").exists()


def test_default_profiles_dir_fallback_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When CWD lacks profiles/ and PROFILES_DIR is unset, _default_profiles_dir resolves package root profiles."""
    monkeypatch.delenv("PROFILES_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    resolved = _default_profiles_dir()
    expected_pkg_profiles = Path(who.__file__).resolve().parent.parent.parent / "profiles"
    if expected_pkg_profiles.exists():
        assert resolved == expected_pkg_profiles


def test_cypher_rel_type_sanitization():
    """GraphStore._sanitize_cypher_rel_type must strip Cypher injection payloads."""
    san = GraphStore._sanitize_cypher_rel_type("ADVOCATES]->(x) DETACH DELETE x //")
    assert san.isidentifier() or san.replace("_", "").isalnum()
    assert "]" not in san and "(" not in san and " " not in san and "/" not in san
    assert GraphStore._sanitize_cypher_rel_type("123_EDGE").startswith("REL_")
    assert GraphStore._sanitize_cypher_rel_type("") == "RELATES_TO"


def test_dialogue_harvester_speaker_boundary_ios_unicode_email_and_zero_turns(tmp_path: Path):
    """DialogueHarvester separates Ece vs Ecem, strips iOS marks, parses From:+Subject: emails, skips empty files."""
    # 1. Similar speaker names must not collide
    dh = DialogueHarvester(output_dir=tmp_path / "wa_out", target_alias="Ece")
    assert dh.is_target_speaker("Ece") is True
    assert dh.is_target_speaker("Ece Yılmaz") is True
    assert dh.is_target_speaker("Ecem") is False

    # 2. iOS WhatsApp BOM + \u200e + Telegram Ece vs Ecem
    wa_file = tmp_path / "ios_wa.txt"
    wa_file.write_text(
        "\ufeff\u200e[12.05.2024, 14:22:10] Ali: Başlayalım mı?\n"
        "\u200e[12.05.2024, 14:23:05] Ece: Önce ölçüm yapalım.\n"
        "\u200e[12.05.2024, 14:24:11] Ecem: Hemen geçelim!\n"
        "[12.05.2024, 14:25:00] Ali: Risk analizi?\n"
        "[12.05.2024, 14:26:30] Ece: Pilotla başlayalım.\n",
        encoding="utf-8",
    )
    wa_pairs = dh.parse_file(wa_file)
    assert len(wa_pairs) == 2
    assert all("Hemen geçelim!" not in p["target_response"] for p in wa_pairs)

    tg_file = tmp_path / "tg.json"
    tg_file.write_text(
        json.dumps({
            "messages": [
                {"type": "message", "from": "Can", "text": "Rewrite?", "date": "2024-06-01T10:00:00"},
                {"type": "message", "from": "Ece", "text": "Benchmark first.", "date": "2024-06-01T10:01:00"},
                {"type": "message", "from": "Ecem", "text": "Rewrite everything immediately!", "date": "2024-06-01T10:02:00"},
            ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    tg_pairs = dh.parse_file(tg_file)
    assert len(tg_pairs) == 1
    assert tg_pairs[0]["target_response"] == "Benchmark first."

    # 3. Email thread with From: + Subject: (no To:) must be detected and parsed
    email_file = tmp_path / "mail_thread.txt"
    email_file.write_text(
        "From: Can <can@example.org>\n"
        "Subject: Review\n\n"
        "Skip audits to ship faster?\n\n"
        "From: Ece <ece@example.org>\n"
        "Subject: Re: Review\n\n"
        "Empirical safeguards prevent outages.\n",
        encoding="utf-8",
    )
    email_pairs = dh.parse_file(email_file)
    assert len(email_pairs) == 1
    assert "Empirical safeguards" in email_pairs[0]["target_response"]

    # 4. Zero-turn input must NOT write empty .md / .json files
    empty_dir = tmp_path / "empty_out"
    dh_empty = DialogueHarvester(output_dir=empty_dir, target_alias="Ece")
    junk_chat = tmp_path / "no_turns.txt"
    junk_chat.write_text("just random prose with no timestamps or speakers\n", encoding="utf-8")
    res_empty = dh_empty.parse_file(junk_chat)
    assert res_empty == []
    assert not empty_dir.exists() or list(empty_dir.glob("*")) == []


def test_document_ingestor_html_and_metadata_pdf_guard(tmp_path: Path):
    """HTML ingestion must strip scripts cleanly; binary/metadata-only PDFs must fail cleanly without leaking Producer tags."""
    out_dir = tmp_path / "doc_out"
    di = DocumentIngestor(output_dir=out_dir)

    html_file = tmp_path / "post.html"
    html_file.write_text(
        "<html><head><title>Ece Essay</title><script>var secret='leak';</script></head>"
        "<body><h1>Local Ownership</h1><p>Users control their own behavioral data.</p></body></html>",
        encoding="utf-8",
    )
    h_res = di.ingest_file(html_file)
    assert h_res["status"] == "OK"
    assert "var secret" not in h_res["content"]
    assert "Users control their own behavioral data." in h_res["content"]

    pdf_file = tmp_path / "scanned.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n/Producer (Adobe InDesign 19.0)\n(D:20240501120000)\n%%EOF")
    p_res = di.ingest_file(pdf_file)
    assert p_res["status"] == "EXTRACTION_FAILED"
    assert p_res["content"] == ""


def test_web_harvester_feedback_url_not_treated_as_rss():
    """WebHarvester must not route '/blog/ai-feedback-loops' to parse_rss_feed."""
    assert WebHarvester._is_feed_url("https://example.com/blog/ai-feedback-loops") is False
    assert WebHarvester._is_feed_url("https://example.com/feed") is True
    assert WebHarvester._is_feed_url("https://example.com/blog/rss.xml") is True
    assert WebHarvester._is_feed_url("https://example.com/posts?format=atom") is True


def test_offline_provider_routing_chat_drift_arena_and_simulation_no_76_conflict():
    """OfflineProvider must keep TwinChat in-character even when memory mentions 'cognitive', support Drift/Arena, and avoid '76%'."""
    dirs = settings.get_target_dirs("OfflineCheckTarget")
    (dirs.raw_data_dir / "notes.md").write_text(
        "# Notes\nUsers deserve full ownership of local cognitive data and empirical audits.\n",
        encoding="utf-8",
    )
    ve = VectorEmbedder("OfflineCheckTarget", chroma_dir=dirs.chroma_dir)
    ve.add_documents([{"content": "Users deserve full ownership of local cognitive data and empirical audits.", "file_name": "notes.md"}])

    chat = TwinChat("OfflineCheckTarget", dirs)
    reply = chat.respond("Tell me about local cognitive data ownership")
    assert "# Cognitive & Mindset Profile" not in reply
    assert "OfflineCheckTarget" in reply

    drift = TemporalDriftDetector("OfflineCheckTarget", dirs).detect_drift(
        new_texts=["Certainly everyone knows only opaque control works."]
    )
    assert "# Cognitive & Mindset Profile" not in drift["drift_synthesis"]
    assert "Epistemic Trajectory" in drift["drift_synthesis"]

    sim = Simulator("OfflineCheckTarget", dirs).simulate_scenario("Propose skipping verification")
    assert "Confidence Score:** 76%" not in sim["simulation_analysis"]

    p2_dirs = settings.get_target_dirs("OfflinePartnerTarget")
    (p2_dirs.vault_dir / "character.json").write_text(json.dumps({"name": "OfflinePartnerTarget"}), encoding="utf-8")
    arena = SwarmArena(["OfflineCheckTarget", "OfflinePartnerTarget"], topic="Cognitive liberty & audits", rounds=1)
    debate = arena.run_debate()
    assert "# Cognitive & Mindset Profile" not in debate["transcript"][0]["statement"]


def test_no_fabricated_percentages_or_stale_v4_versions(tmp_path: Path):
    """Chronomics, PsychInsights, and MindsetProfiler must never fabricate 28%/14%/18%/v4.0.0 when evidence is missing."""
    cc = CognitiveChronomics(output_dir=tmp_path / "chron")
    day_only_docs = [
        {"content": "[12.04.2024, 14:00] " + ("Empirical verification requires clear data and careful reasoning. " * 4)}
    ]
    report = cc.analyze_circadian_drift(day_only_docs)
    summary = report["circadian_drift_summary"]
    assert summary["has_direct_night_corpus"] is False
    assert summary["epistemic_filter_decay_pct"] == 0.0
    assert summary["lexical_diversity_decay_pct"] == 0.0
    assert summary["critical_valence_increase_pct"] == 0.0
    assert "Insufficient nocturnal timestamped corpus" in report["probabilistic_statement"]
    assert "28.0%" not in report["probabilistic_statement"]

    target_dirs = settings.get_target_dirs("CleanStatsTarget")
    mp = MindsetProfiler("CleanStatsTarget", vault_dir=target_dirs.vault_dir)
    mp_res = mp.build_profile(
        mapping_skill={"behavioral_profile": {"cognitive_style": "Empirical", "social_orientation": "Balanced", "calibration_ratio": 1.0}},
        graph_metrics={"top_influential_concepts": []},
        sample_quotes=[
            "[INTERLOCUTOR_CONTEXT]: Should we skip audits?",
            "[12.05.2024, 14:26:30] [TARGET_PRIMARY]: We rely strictly on empirical audits.",
        ],
        features={},
    )
    assert f"who?? v{who.__version__}" in mp_res["mindset_summary"]
    assert "v4.0.0" not in mp_res["mindset_summary"]
    assert "14%" not in mp_res["mindset_summary"]
    assert "18%" not in mp_res["mindset_summary"]
    assert mp_res["character"]["postExamples"] == ["We rely strictly on empirical audits."]

    pi = PsychInsights("CleanStatsTarget", target_dirs).generate_report()
    assert "14%" not in pi["chronomics_statement"]
    assert "18%" not in pi["manipulation_statement"]

    ss_res = SkillSynthesizer(output_dir=target_dirs.learned_skills_dir).synthesize_mapping_skill(
        "CleanStatsTarget", {}, [], []
    )
    assert ss_res["version"] == who.__version__


def test_graph_builder_and_topic_mapper_clean_turkish_and_no_scaffolding(tmp_path: Path):
    """GraphBuilder and TopicMapper must exclude diarization tags/stopwords, preserve Turkish words, and avoid self-loops."""
    raw_doc = (
        "---\nfile_name: whatsapp.md\nsource_type: dialogue\n---\n\n"
        "# Diarized Dialogue: whatsapp_ios (whatsapp)\n\n"
        "[INTERLOCUTOR_CONTEXT]: Usually you like Opacity?\n\n"
        "[12.05.2024, 14:26:30] [TARGET_PRIMARY]: However, Distributed Governance and Local-First Design outperform centralized gates [REDACTED_PHONE].\n"
        "Açık kaynak ekosistemlerinde şeffaflık ve ölçülebilir veri büyük ihtimalle uzun vadeli güven oluşturur.\n"
        "Ölçülebilir veri ve şeffaflık kurumsal dayanıklılığı güçlendirir.\n"
    )
    gb = GraphBuilder("Ece Yılmaz")
    elems = gb.extract_graph_elements(raw_doc, source_id="whatsapp.md")
    ent_names = {e["name"] for e in elems["entities"]}
    for junk in ("INTERLOCUTOR_CONTEXT", "TARGET_PRIMARY", "REDACTED_PHONE", "Diarized Dialogue", "Usually", "However", "Opacity"):
        assert junk not in ent_names
    assert all("\n" not in name for name in ent_names)

    # Stance inversion must not emit self-loops source == target
    inv_dissonances = gb._detect_cognitive_dissonance(
        entities=[{"name": "Centralization", "type": "Concept"}],
        relationships=[
            {"source": "Ece Yılmaz", "target": "Centralization", "relation": "ADVOCATES", "quote": "Centralization helps initially."},
            {"source": "Ece Yılmaz", "target": "Centralization", "relation": "REJECTS", "quote": "Centralization fails at scale."},
        ],
    )
    assert len(inv_dissonances) == 1
    assert inv_dissonances[0]["source"] != inv_dissonances[0]["target"]

    tm = TopicMapper(output_dir=tmp_path)
    t_res = tm.extract_topics([{"content": raw_doc}, {"content": raw_doc}])
    all_kws = {kw for t in t_res["topics"] for kw in t["keywords"]}
    assert not (all_kws & {"diarized", "dialogue", "whatsapp", "interlocutor", "target", "primary", "redacted"})
    assert "şeffaflık" in all_kws or "ölçülebilir" in all_kws


def test_vector_embedder_preserves_source_id_and_turkish_slug():
    """VectorEmbedder respects 'source_id' and sanitize_slug maps Turkish characters accurately."""
    assert sanitize_slug("Ece Yılmaz") == "ece_yilmaz"
    assert sanitize_slug("Işık Çağlar") == "isik_caglar"
    assert sanitize_slug("Özgür Şahin") == "ozgur_sahin"

    dirs = settings.get_target_dirs("Ece Yılmaz")
    ve = VectorEmbedder("Ece Yılmaz", chroma_dir=dirs.chroma_dir)
    ve.add_documents([{"content": "Şeffaflık ve ölçülebilir veri temel ilkedir.", "source_id": "turkce_deneme.md"}])
    assert ve.local_chunks[0]["id"].startswith("turkce_deneme.md_chunk_")
    assert ve.local_chunks[0]["metadata"]["source_id"] == "turkce_deneme.md"


def test_cli_ghost_guards_and_crash_prevention(tmp_path: Path):
    """CLI commands on missing profiles or invalid args must fail cleanly without ghost folders or unhandled tracebacks."""
    before = {p.name for p in settings.profiles_dir.iterdir() if p.is_dir()}

    for args in (
        ["evaluate-profile", "--target", "Missing Ghost A"],
        ["monitor", "--target", "Missing Ghost B", "--drift"],
        ["learn", "--target", "Missing Ghost C"],
        ["build-vault", "--target", "Missing Ghost D"],
        ["graph-export", "--target", "Missing Ghost E"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 1
        assert "Traceback" not in res.output

    after = {p.name for p in settings.profiles_dir.iterdir() if p.is_dir()}
    assert after == before

    res_chat = runner.invoke(app, ["harvest", "--target", "MissingChat", "--chat-logs", str(tmp_path / "nope.txt")])
    assert res_chat.exit_code == 1
    assert "not found" in res_chat.output.lower()

    res_iter = runner.invoke(app, ["simulate", "--target", "Target X", "--scenario", "test", "--monte-carlo", "--iterations", "0"])
    assert res_iter.exit_code != 0


def test_mcp_bom_multibyte_content_length_and_sdk_export():
    """MCPServer handles UTF-8 BOM, multi-header multi-byte Content-Length, and SDK exports get_tool_schemas."""
    assert callable(get_tool_schemas)
    assert len(get_tool_schemas("openai")) == 7

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["WHO_PROVIDER"] = "offline"
    env["PYTHONUTF8"] = "0"
    env.setdefault("PROFILES_DIR", str(repo_root / "profiles"))

    body1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode("utf-8")
    body2 = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "who_explain", "arguments": {"target": "Target X", "claim": "şeffaflık — ölçüm"}},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    bom_and_cl = (
        b"\xef\xbb\xbf"
        + f"Content-Length: {len(body1)}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n".encode("ascii")
        + body1
        + f"Content-Length: {len(body2)}\r\n\r\n".encode("ascii")
        + body2
    )
    proc = subprocess.run(
        [sys.executable, "-m", "who.cli", "mcp"],
        input=bom_and_cl,
        capture_output=True,
        cwd=str(repo_root),
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0
    lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert [p["id"] for p in parsed] == [1, 2]
    assert "error" not in parsed[1]


def test_academic_researcher_authors_and_cli_research_honesty(monkeypatch: pytest.MonkeyPatch):
    """query_arxiv parses authors cleanly and CLI research avoids labeling r=None papers as 'Empirical'."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2501.00001v1</id>
        <published>2025-01-10T00:00:00Z</published>
        <title>Bounded Rationality in Networks</title>
        <summary>Qualitative conceptual framework without reported correlation coefficients.</summary>
        <author><name>Ada Lovelace</name></author>
        <author><name>Alan Turing</name></author>
      </entry>
    </feed>"""

    class FakeResp:
        status_code = 200
        text = sample_xml

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return None
        def get(self, url, **kw):
            return FakeResp()

    monkeypatch.setattr("who.learn.autonomous_researcher.httpx.Client", FakeClient)
    ar = AutonomousAcademicResearcher(output_dir=settings.profiles_dir / "res_test", profiles_dir=settings.profiles_dir)
    # Call original unpatched query_arxiv saved at import time (before conftest autouse patch)
    real_papers = _ORIG_QUERY_ARXIV(ar, "bounded rationality", max_results=1)
    assert len(real_papers) == 1
    assert real_papers[0]["authors"] == ["Ada Lovelace", "Alan Turing"]

    monkeypatch.setattr(
        ar,
        "query_arxiv",
        lambda q, max_results=3: real_papers,
    )
    monkeypatch.setattr(
        "who.cli.commands.vault.AcademicResearcher",
        lambda output_dir=None: ar,
    )
    res = runner.invoke(app, ["research", "--topic", "bounded rationality", "--sources", "arxiv", "--limit", "1"])
    assert res.exit_code == 0
    assert "Qual. N/A" in res.output
    assert "Empirical" not in res.output


def test_list_targets_display_name_and_skips_hidden():
    """list_targets() skips dot-directories and prefers character.json name (e.g. 'Ece Yılmaz')."""
    (settings.profiles_dir / ".hidden_meta").mkdir(parents=True, exist_ok=True)
    dirs = settings.get_target_dirs("Ece Yılmaz")
    (dirs.vault_dir / "character.json").write_text(json.dumps({"name": "Ece Yılmaz"}, ensure_ascii=False), encoding="utf-8")
    (dirs.raw_data_dir / "notes.txt").write_text("Ölçülebilir veri.", encoding="utf-8")

    targets = list_targets()
    slugs = [t["slug"] for t in targets]
    assert ".hidden_meta" not in slugs
    ece_entry = next(t for t in targets if t["slug"] == "ece_yilmaz")
    assert ece_entry["name"] == "Ece Yılmaz"
    assert ece_entry["raw_documents_count"] == 1
