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
