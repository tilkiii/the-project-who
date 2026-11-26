"""
v1.0.1 Hardening & Security Regression Suite:
Verifies all P0/P1/P2 audit fixes across MCP stdio framing, LiveMonitor, graph schemas,
security sanitization, zero-placeholder harvesting, Oracle/Explainability fallbacks,
VectorEmbedder deduplication, lazy read-only directories, TwinChat, and offline Composer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from who import CognitiveTwin
from who.cli import app
from who.config import settings
from who.engine.arena import SwarmArena
from who.engine.composer import Composer
from who.engine.explainability_engine import ExplainabilityEngine
from who.engine.oracle import Oracle
from who.engine.twin_chat import TwinChat
from who.harvest.document_ingestor import DocumentIngestor
from who.harvest.live_monitor import LiveMonitor
from who.harvest.social_osint import SocialOSINTHunter
from who.harvest.web_harvester import WebHarvester
from who.learn.drift_detector import TemporalDriftDetector
from who.learn.feature_extractor import FeatureExtractor
from who.vault.graph_store import GraphStore
from who.vault.mindset_profiler import MindsetProfiler
from who.vault.social_graph import SocialGraphAnalyzer
from who.vault.vector_embedder import VectorEmbedder

runner = CliRunner()


def test_mcp_stdio_stdout_pure_json_rpc():
    """Verify real subprocess `who mcp` keeps stdout strictly valid JSON-RPC without log leakage."""
    repo_root = Path(__file__).resolve().parent.parent
    reqs = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "who_explain", "arguments": {"target": "Target X", "claim": "empirical verification"}},
        }),
        json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "who_simulate", "arguments": {"target": "Target X", "scenario": "Transparent mutual governance proposal."}},
        }),
    ]
    env = dict(os.environ)
    env["WHO_PROVIDER"] = "offline"
    env.setdefault("PROFILES_DIR", str(repo_root / "profiles"))

    proc = subprocess.run(
        [sys.executable, "-m", "who.cli", "mcp"],
        input="\n".join(reqs) + "\n",
        text=True,
        capture_output=True,
        cwd=str(repo_root),
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 3
    parsed = [json.loads(ln) for ln in lines]
    assert [p["id"] for p in parsed] == [1, 2, 3]
    assert "result" in parsed[1] and "result" in parsed[2]


def test_live_monitor_and_feature_extractor_string_guard(tmp_path: Path):
    """Ensure passing a raw string instead of List[str] does not split into single characters."""
    fe = FeatureExtractor()
    sample = "Certainly everyone knows that the legacy system has completely failed."
    feats = fe.extract_features(sample)  # type: ignore[arg-type]
    assert feats["corpus_stats"]["total_words"] == 10

    dirs = settings.get_target_dirs("MonitorGuard")
    mon = LiveMonitor("MonitorGuard", dirs)
    anoms = mon.detect_anomalies(
        [{
            "title": "Check",
            "content": "Kesinlikle herkes bilmelidir ki başka hiçbir alternatif asla kabul edilemez ve tamamen yanlıştır.",
            "url": "https://example.org/check",
            "timestamp": "2026-10-30T02:15:00Z",
        }],
        ignore_history=True,
    )
    assert len(anoms) == 1
    assert any(t["type"] == "DOGMATISM_SURGE" for t in anoms[0]["triggers"])


def test_unified_graph_schema_support_id_edges(tmp_path: Path):
    """GraphStore, SocialGraphAnalyzer, and TemporalDriftDetector must support id/edges JSON schemas."""
    dirs = settings.get_target_dirs("SchemaTarget")
    (dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [
                {"id": "Mechanism Design", "label": "Concept", "pagerank": 0.94},
                {"id": "Equilibrium Analysis", "label": "Concept", "pagerank": 0.89},
            ],
            "edges": [
                {"source": "Mechanism Design", "target": "Equilibrium Analysis", "weight": 1.0},
            ],
        }),
        encoding="utf-8",
    )

    gs = GraphStore("SchemaTarget", vault_dir=dirs.vault_dir)
    assert len(gs.get_all_nodes()) == 2
    assert len(gs.get_all_relationships()) == 1
    gs.close()

    td = TemporalDriftDetector("SchemaTarget", dirs)
    bm = td._load_baseline_metrics()
    assert "mechanism design" in bm["baseline_concepts"]
    assert "equilibrium analysis" in bm["baseline_concepts"]
    assert bm["baseline_concepts"] != {"concept"}

    sga = SocialGraphAnalyzer(["SchemaTarget", "EmptySecond"], output_dir=tmp_path)
    net = sga.build_multi_target_network(export_html=False)
    node_ids = {n["id"] for n in net["nodes"]}
    assert "Mechanism Design" in node_ids
    assert list(tmp_path.glob("*.html")) == []


def test_security_path_traversal_and_xss_escaping(tmp_path: Path):
    """Sanitize dossier filenames against path traversal and escape HTML/XSS in D3 HTML exports."""
    dirs = settings.get_target_dirs("SecTarget")
    hunter = SocialOSINTHunter("SecTarget", dirs)
    hunter._ingest_discovered_dossier(
        "../../escape_attempt",
        [{"platform": "GitHub", "category": "Dev", "icon": "🐙", "url": "https://github.com/test", "bio_snippet": "Hi"}],
    )
    saved_files = list(dirs.raw_data_dir.glob("social_osint_dossier_*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].resolve().is_relative_to(dirs.raw_data_dir.resolve())

    # Stored XSS test in SocialGraphAnalyzer HTML output
    xss_target = 'Evil</script><script>alert(1)</script>'
    xss_dirs = settings.get_target_dirs(xss_target)
    (xss_dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({"nodes": [{"name": "</script><img onerror=alert(1)>", "type": "Concept"}], "relationships": []}),
        encoding="utf-8",
    )
    sga = SocialGraphAnalyzer([xss_target], output_dir=tmp_path)
    sga.build_multi_target_network(export_html=True)
    html_files = list(tmp_path.glob("social_network_*.html"))
    assert len(html_files) == 1
    html_body = html_files[0].read_text(encoding="utf-8")
    assert "</script><img" not in html_body
    assert "<script>alert(1)</script>" not in html_body


def test_default_output_dirs_respect_settings_profiles_dir(tmp_path: Path):
    """SocialGraphAnalyzer and SwarmArena default output dirs must follow settings.profiles_dir."""
    sga = SocialGraphAnalyzer(["Target A"])
    arena = SwarmArena(["Target A", "Target B"], topic="Test")
    assert sga.output_dir == settings.profiles_dir / "social_graphs"
    assert arena.output_dir == settings.profiles_dir / "arena_debates"

    res_net = runner.invoke(app, ["network", "--targets", "Target A, Target B", "--no-html"])
    assert res_net.exit_code == 0
    assert list((settings.profiles_dir / "social_graphs").glob("*.html")) == []


def test_no_placeholder_fabrication_across_harvest_and_mindset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No fake baseline_overview.md, alerts, scrape failures, doc failures, or boilerplate character.json."""
    # 1. harvest with no inputs must not create fake baseline_overview.md
    res_h = runner.invoke(app, ["harvest", "--target", "EmptyHarvestTarget"])
    assert res_h.exit_code == 0
    empty_dirs = settings.get_target_dirs("EmptyHarvestTarget", create=False)
    assert not (empty_dirs.raw_data_dir / "baseline_overview.md").exists()

    # 2. monitor without --simulate-feed and without sources must not write alerts
    res_m = runner.invoke(app, ["monitor", "--target", "EmptyMonitorTarget"])
    assert res_m.exit_code == 0
    mon_dirs = settings.get_target_dirs("EmptyMonitorTarget", create=False)
    assert not (mon_dirs.vault_dir / "monitoring_alerts.json").exists()

    # 3. DocumentIngestor failed parse must not save placeholder .md
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"\x00\x01\x02")
    doc_out = tmp_path / "doc_out"
    di = DocumentIngestor(output_dir=doc_out)
    doc_res = di.ingest_file(bad_pdf)
    assert doc_res["status"] == "EXTRACTION_FAILED"
    assert doc_res["content"] == ""
    assert list(doc_out.glob("*.md")) == []

    # 4. WebHarvester HTTP failure must not save "# Scrape Failure" .md
    import asyncio

    class FailClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return None
        async def get(self, url, headers=None):
            raise RuntimeError("Network unreachable")

    monkeypatch.setattr("who.harvest.web_harvester.httpx.AsyncClient", FailClient)
    web_out = tmp_path / "web_out"
    wh = WebHarvester(output_dir=web_out)
    web_res = asyncio.run(wh.scrape_url("https://nonexistent.invalid/post"))
    assert web_res["status"] == "FAILED"
    assert web_res["content"] == ""
    assert list(web_out.glob("*.md")) == []

    # 5. MindsetProfiler with empty quotes must not fabricate boilerplate postExamples or "Confidence Score: 85%"
    mp = MindsetProfiler("QuoteFree", vault_dir=tmp_path / "mp_vault")
    mp_res = mp.build_profile({}, {"top_influential_concepts": []}, sample_quotes=[])
    assert mp_res["character"]["postExamples"] == []
    assert "85%" not in mp_res["character"]["messageExamples"][0][1]["content"]["text"]


def test_oracle_zero_evidence_honesty_and_raw_corpus_fallback():
    """Oracle returns 0.0 when corpus & graph are empty, and falls back to raw_data when unindexed."""
    ghost_dirs = settings.get_target_dirs("TrueGhost", create=False)
    res_empty = Oracle("TrueGhost", ghost_dirs).ask("Any evidence?")
    assert res_empty["confidence_score"] == 0.0

    raw_only_dirs = settings.get_target_dirs("RawOnlyTarget")
    (raw_only_dirs.raw_data_dir / "notes.md").write_text(
        "# Notes\nBilateral consensus requires empirical verification before execution.",
        encoding="utf-8",
    )
    (raw_only_dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [{"id": "Empirical Verification"}, {"id": "Bilateral Consensus"}],
            "edges": [{"source": "Empirical Verification", "target": "Bilateral Consensus"}],
        }),
        encoding="utf-8",
    )
    res_raw = Oracle("RawOnlyTarget", raw_only_dirs).ask("What about empirical verification?")
    assert res_raw["confidence_score"] > 0.0
    assert len(res_raw["primary_quotes"]) >= 1

    exp = ExplainabilityEngine("RawOnlyTarget", raw_only_dirs).explain_claim("Empirical verification")
    assert exp["confidence_score"] > 0.0
    assert len(exp["graph_trace_paths"]) >= 1


def test_vector_embedder_sanitization_and_upsert_dedup():
    """VectorEmbedder sanitizes Turkish target names for Chroma collections and upserts local_chunks."""
    dirs = settings.get_target_dirs("Çağrı Öztürk (Test!)")
    ve = VectorEmbedder("Çağrı Öztürk (Test!)", chroma_dir=dirs.chroma_dir)
    doc = {"content": "Repeated verification quote about open protocols and audits.", "source_url": "docA"}
    ve.add_documents([doc])
    first_count = len(ve.local_chunks)
    ve.add_documents([doc])
    assert len(ve.local_chunks) == first_count


def test_read_only_operations_do_not_create_empty_profile_dirs():
    """Read-only SDK and CLI calls on an unknown target must not pollute profiles/ with empty folders."""
    ghost = "Never Created Ghost 99"
    expected_dir = settings.profiles_dir / "never_created_ghost_99"
    assert not expected_dir.exists()

    twin = CognitiveTwin(ghost)
    _ = twin.get_profile()
    _ = twin.get_knowledge_graph()
    assert not expected_dir.exists()

    res_ask = runner.invoke(app, ["ask", "--target", ghost, "--prompt", "Any stance?"])
    assert res_ask.exit_code == 0
    assert not expected_dir.exists()

    res_exp = runner.invoke(app, ["explain", "--target", ghost, "--claim", "Any stance"])
    assert res_exp.exit_code == 0
    assert not expected_dir.exists()


def test_twin_chat_word_boundary_greeting_and_offline_composer(monkeypatch: pytest.MonkeyPatch):
    """Ensure 'plan'/'size' trigger vector memory search and offline Composer produces outreach drafts."""
    dirs = settings.get_target_dirs("BoundaryChat")
    chat = TwinChat("BoundaryChat", dirs)
    queries = []

    def spy_search(query: str, top_k: int = 4):
        queries.append(query)
        return []

    monkeypatch.setattr(chat.vector_store, "search_similar", spy_search)
    chat.respond("What is the plan?")
    chat.respond("What size is the team?")
    assert queries == ["What is the plan?", "What size is the team?"]

    outreach = Composer("BoundaryChat", dirs).compose_outreach(
        goal="Invite to empirical workshop",
        context="Joint mechanism design review",
    )
    assert "Invite to empirical workshop" in outreach["initial_draft"]
    assert "**Question:** this topic" not in outreach["initial_draft"]
    assert "**Question:** this topic" not in outreach["adversarial_rejection_analysis"]
    assert "**Question:** this topic" not in outreach["optimized_final_pitch"]
