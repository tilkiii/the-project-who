"""
Unit tests for system management CLI commands (targets, purge, providers, ui).
"""

from pathlib import Path
from typer.testing import CliRunner

from src.cli import app
from src.config import settings

runner = CliRunner()


def test_cli_targets_empty(tmp_path: Path):
    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "No digital twin profiles found" in result.output or "Profiles directory" in result.output


def test_cli_targets_with_profile(tmp_path: Path):
    # Setup test profile in isolated settings.profiles_dir
    target_dir = Path(settings.profiles_dir) / "test_scholar"
    vault_dir = target_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "character.json").write_text('{"name": "Test Scholar"}', encoding="utf-8")
    (vault_dir / "concept_network.json").write_text('{"nodes": [], "relationships": []}', encoding="utf-8")

    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "Test Scholar" in result.output
    assert "READY" in result.output


def test_cli_purge_target():
    # Setup test profile
    target_dir = Path(settings.profiles_dir) / "disposable_twin"
    target_dir.mkdir(parents=True, exist_ok=True)
    assert target_dir.exists()

    # Purge with --force
    result = runner.invoke(app, ["purge", "--target", "disposable_twin", "--force"])
    assert result.exit_code == 0
    assert "Successfully purged target profile" in result.output
    assert not target_dir.exists()


def test_cli_purge_nonexistent():
    result = runner.invoke(app, ["purge", "--target", "nonexistent_target_slug", "--force"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_ui_help():
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Launch the interactive Web Dashboard" in result.output


def test_dashboard_api_status_healthcheck():
    """Verify /api/status endpoint used by Docker HEALTHCHECK returns 200 OK."""
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from src.ui.dashboard import DashboardRequestHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"
            assert payload["service"] == "who-engine"
            assert "active_provider" in payload
            assert "targets_count" in payload
    finally:
        server.shutdown()
        server.server_close()


def test_gemini_provider_rest_embedding_fallback(monkeypatch):
    """Verify GeminiProvider.embed uses REST API when google-genai SDK is unavailable."""
    import httpx
    from src.providers.gemini_provider import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-gemini-key")
    called_urls = []

    class DummyPostResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": {"values": [0.12, -0.34, 0.56, 0.78]}}

    def fake_post(self, url, json=None, **kwargs):
        called_urls.append((url, json))
        return DummyPostResp()

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    provider = GeminiProvider()
    provider._client = None  # Force SDK-absent path
    vec = provider.embed("Epistemic calibration test text")

    assert vec == [0.12, -0.34, 0.56, 0.78]
    assert len(called_urls) == 1
    assert "text-embedding-004:embedContent" in called_urls[0][0]


def test_dashboard_v2_contradicts_graph_and_endpoints():
    """Verify Web Dashboard 2.0 UI tabs, [CONTRADICTS] graph metadata, and /api/oracle, /api/compose, /api/arena."""
    import json
    import shutil
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from src.ui.dashboard import DashboardRequestHandler

    slug_a = "dash_v2_alpha"
    slug_b = "dash_v2_beta"
    dir_a = Path(settings.profiles_dir) / slug_a
    dir_b = Path(settings.profiles_dir) / slug_b

    for d, name in [(dir_a, "Dash Alpha"), (dir_b, "Dash Beta")]:
        (d / "vault").mkdir(parents=True, exist_ok=True)
        (d / "learned_skills").mkdir(parents=True, exist_ok=True)
        (d / "raw_data").mkdir(parents=True, exist_ok=True)
        (d / "vault" / "character.json").write_text(
            json.dumps({"name": name, "bio": ["Empirical researcher"], "topics": ["Governance", "Transparency"], "postExamples": ["Evidence matters."]}),
            encoding="utf-8",
        )
        (d / "vault" / "mindset_summary.md").write_text(f"# {name}\nCalibrated epistemic reasoning.", encoding="utf-8")

    # Populate concept_network.json with a [CONTRADICTS] edge on Dash Alpha
    (dir_a / "vault" / "concept_network.json").write_text(
        json.dumps({
            "target_name": "Dash Alpha",
            "nodes": [
                {"name": "Dash Alpha", "type": "Person", "description": "Target"},
                {"name": "Open Protocols", "type": "Concept", "description": "Decentralized systems"},
                {"name": "Centralized Gatekeeping", "type": "Concept", "description": "Top-down control"},
            ],
            "relationships": [
                {"source": "Dash Alpha", "target": "Open Protocols", "relation": "ADVOCATES", "weight": 0.9},
                {
                    "source": "Open Protocols",
                    "target": "Centralized Gatekeeping",
                    "relation": "CONTRADICTS",
                    "weight": 0.88,
                    "dissonance_score": 0.91,
                    "contradiction_summary": "Direct structural tension between open protocols and gatekeeping.",
                    "quote_a": "Open protocols empower verification.",
                    "quote_b": "Gatekeeping ensures compliance.",
                },
            ],
        }),
        encoding="utf-8",
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post_json(endpoint: str, body: dict):
        req = urllib.request.Request(
            f"{base_url}{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    try:
        # 1. Verify UI HTML includes Dashboard 2.0 tabs & [CONTRADICTS] styling
        with urllib.request.urlopen(f"{base_url}/ui", timeout=5) as resp:
            html = resp.read().decode("utf-8")
            assert "Oracle Evidence Q&A" in html
            assert "Outreach Composer" in html
            assert "Swarm Arena" in html
            assert "link-contradicts" in html

        # 2. Verify /api/target/{slug}/graph exposes [CONTRADICTS] dissonance metadata
        with urllib.request.urlopen(f"{base_url}/api/target/{slug_a}/graph", timeout=5) as resp:
            g_data = json.loads(resp.read().decode("utf-8"))
            assert g_data["contradictions_count"] == 1
            contradiction_edges = [l for l in g_data["links"] if l.get("is_contradiction")]
            assert len(contradiction_edges) == 1
            assert contradiction_edges[0]["dissonance_score"] == 0.91
            dissonance_nodes = [n for n in g_data["nodes"] if n.get("has_contradiction")]
            assert len(dissonance_nodes) == 2

        # 3. Verify POST /api/oracle
        status_oracle, oracle_res = post_json("/api/oracle", {"target": slug_a, "query": "What is the stance on Open Protocols?"})
        assert status_oracle == 200
        assert "answer" in oracle_res
        assert "confidence_score" in oracle_res

        # 4. Verify POST /api/compose
        status_comp, comp_res = post_json("/api/compose", {"target": slug_a, "goal": "Invite to open governance panel"})
        assert status_comp == 200
        assert "initial_draft" in comp_res
        assert "adversarial_rejection_analysis" in comp_res
        assert "optimized_final_pitch" in comp_res

        # 5. Verify POST /api/arena (2 participants, 1 round) and HTTP 400 validation
        status_arena, arena_res = post_json(
            "/api/arena",
            {"targets": [slug_a, slug_b], "topic": "Open protocols vs centralized governance", "rounds": 1},
        )
        assert status_arena == 200
        assert len(arena_res["transcript"]) == 2
        assert "synthesis" in arena_res
        assert "consensus_score_pct" in arena_res["synthesis"]

        try:
            post_json("/api/arena", {"targets": [slug_a], "topic": "Invalid single participant"})
            assert False, "Expected HTTP 400 for <2 arena participants"
        except urllib.error.HTTPError as err:
            assert err.code == 400
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(dir_a, ignore_errors=True)
        shutil.rmtree(dir_b, ignore_errors=True)
