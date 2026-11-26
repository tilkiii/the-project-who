"""
Milestone 4 (v1.0.0) Release Gate verification suite:
1. Package, CLI (--version), and MCP version alignment at 1.0.0.
2. Provider integration and schema contracts for Anthropic Claude, OpenAI/ChatGPT,
   Google Gemini, local Ollama, and MCP (100% mocked HTTP, zero external network calls).
3. Offline Monte Carlo strategic synthesis consistency.
4. SDK / CLI target & knowledge graph fallback consistency (concept_network.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

import who
from who import CognitiveTwin, list_targets
from who.cli import app
from who.config import settings
from who.engine.simulator import Simulator
from who.mcp_server import MCPServer, get_tool_schemas
from who.providers.anthropic_provider import AnthropicProvider
from who.providers.gemini_provider import GeminiProvider
from who.providers.manager import provider_manager
from who.providers.openai_provider import OpenAICompatibleProvider

runner = CliRunner()


def test_v1_0_0_version_consistency():
    """Verify package metadata, CLI --version flag, and MCP server handshake all report 1.0.1."""
    assert who.__version__ == "1.0.1"

    mcp = MCPServer()
    init_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["version"] == who.__version__

    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert who.__version__ in res.output


def _seed_sample_target(name: str = "Gate Target") -> Any:
    dirs = settings.get_target_dirs(name)
    dirs.ensure_exists()
    (dirs.raw_data_dir / "note.md").write_text(
        "# Note\nEmpirical evidence and mutual governance audits build reliable consensus.",
        encoding="utf-8",
    )
    (dirs.vault_dir / "mindset_summary.md").write_text("# Mindset\nAnalytical Empirical.", encoding="utf-8")
    (dirs.vault_dir / "concept_network.json").write_text(
        json.dumps({
            "nodes": [{"name": "Mutual Governance", "type": "Concept"}],
            "relationships": [{"source": name, "target": "Mutual Governance", "relation": "ADVOCATES"}],
        }),
        encoding="utf-8",
    )
    return dirs


def test_multi_provider_tool_calling_and_transport_contracts(monkeypatch: pytest.MonkeyPatch):
    """Verify Anthropic, OpenAI, Gemini, Ollama, and MCP schemas & HTTP request formatting."""
    _seed_sample_target("Gate Target")
    mcp = MCPServer()
    calls: List[Dict[str, Any]] = []

    class FakeResp:
        def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> Dict[str, Any]:
            return self._payload

    class FakeHttpClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "FakeHttpClient":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get(self, url: str, headers: Any = None) -> FakeResp:
            calls.append({"method": "GET", "url": url, "headers": headers})
            return FakeResp({"models": [{"name": "llama3.2"}]}, status_code=200)

        def post(self, url: str, json: Any = None, headers: Any = None) -> FakeResp:
            calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
            if "api.anthropic.com" in url:
                return FakeResp({"content": [{"type": "text", "text": "anthropic-ok"}]})
            if "generativelanguage.googleapis.com" in url:
                return FakeResp({"candidates": [{"content": {"parts": [{"text": "gemini-ok"}]}}]})
            if url.endswith("/embeddings"):
                return FakeResp({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
            return FakeResp({"choices": [{"message": {"content": "openai-compat-ok"}}]})

    monkeypatch.setattr("who.providers.anthropic_provider.httpx.Client", FakeHttpClient)
    monkeypatch.setattr("who.providers.openai_provider.httpx.Client", FakeHttpClient)
    monkeypatch.setattr("who.providers.gemini_provider.httpx.Client", FakeHttpClient)

    # 1. Anthropic Claude
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    anth = AnthropicProvider()
    assert anth.is_available() is True
    assert anth.generate("Hello", system_prompt="Sys") == "anthropic-ok"
    anth_call = calls[-1]
    assert anth_call["url"] == "https://api.anthropic.com/v1/messages"
    assert anth_call["headers"]["x-api-key"] == "test-anthropic-key"
    assert anth_call["headers"]["anthropic-version"] == "2023-06-01"
    assert anth_call["json"]["system"] == "Sys"

    anth_tools = get_tool_schemas("anthropic")
    assert len(anth_tools) == 7 and all("input_schema" in t for t in anth_tools)
    res_anth = mcp.dispatch_tool_call("who_list_targets", {})
    assert any(t["slug"] == "gate_target" for t in res_anth)

    # 2. OpenAI / ChatGPT
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    oai = provider_manager.get_provider("openai")
    assert oai.is_available() is True
    assert oai.generate("Hello") == "openai-compat-ok"
    oai_call = calls[-1]
    assert oai_call["url"] == "https://api.openai.com/v1/chat/completions"
    assert oai_call["headers"]["Authorization"] == "Bearer test-openai-key"

    oai_tools = get_tool_schemas("openai")
    assert len(oai_tools) == 7 and all(t["type"] == "function" for t in oai_tools)
    res_oai = mcp.dispatch_tool_call("who_get_profile", json.dumps({"target": "Gate Target"}))
    assert res_oai["slug"] == "gate_target"

    # 3. Google Gemini
    gem = GeminiProvider()
    gem_out = gem._generate_rest("test-gemini-key", "gemini-2.5-flash-lite", "Hello", "Sys", 0.2, 128)
    assert gem_out == "gemini-ok"
    gem_call = calls[-1]
    assert "generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent" in gem_call["url"]
    assert gem_call["json"]["systemInstruction"] == {"parts": [{"text": "Sys"}]}

    gem_tools = get_tool_schemas("gemini")
    assert len(gem_tools) == 7 and all("parameters" in t for t in gem_tools)

    # 4. Ollama local /v1 resolution & tool schema
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    ollama = OpenAICompatibleProvider(
        provider_id="ollama",
        display_name="Ollama",
        env_keys=[],
        base_url="http://localhost:11434/v1",
        default_model="llama3.2",
        default_embedding_model="nomic-embed-text",
        requires_api_key=False,
    )
    assert ollama._get_base_url() == "http://localhost:11434/v1"
    assert ollama.is_available() is True
    assert calls[-1]["url"] == "http://localhost:11434/v1/models"
    assert ollama.generate("Hi") == "openai-compat-ok"
    assert calls[-1]["url"] == "http://localhost:11434/v1/chat/completions"
    assert ollama.embed("Embed me") == [0.1, 0.2, 0.3]
    assert calls[-1]["url"] == "http://localhost:11434/v1/embeddings"

    ollama_tools = get_tool_schemas("ollama")
    assert len(ollama_tools) == 7 and all(t["type"] == "function" for t in ollama_tools)

    # 5. MCP tool metadata cleanliness
    tools_resp = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = tools_resp["result"]["tools"]
    assert len(tools) == 7
    for t in tools:
        assert "OpenAlex" not in t["description"]
        assert "Neo4j" not in t["description"]
        ann = t["annotations"]
        assert all(isinstance(ann[k], bool) for k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"))


def test_offline_monte_carlo_synthesis_and_concept_network_fallback():
    """Verify offline Monte Carlo summary stays scenario-grounded and SDK reads concept_network.json."""
    dirs = _seed_sample_target("Fallback Twin")

    targets = list_targets()
    entry = next(t for t in targets if t["slug"] == "fallback_twin")
    assert entry["is_ready"] is True

    twin = CognitiveTwin("Fallback Twin")
    kg = twin.get_knowledge_graph()
    assert kg.get("nodes") and kg["nodes"][0]["name"] == "Mutual Governance"

    sim = Simulator("Fallback Twin", dirs).simulate_monte_carlo(
        "Joint proposal for Mutual Governance and transparent bilateral audit.",
        iterations=10,
        random_seed=42,
    )
    syn = sim["strategic_synthesis"]
    assert "Monte Carlo Scenario Synthesis" in syn
    assert sim["dominant_branch"] in syn
    assert "**Question:** this topic" not in syn
