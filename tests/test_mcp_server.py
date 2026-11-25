"""
Unit tests for who?? Model Context Protocol (MCP) Server.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from who.config import settings
from who.mcp_server import MCPServer, TOOLS_SCHEMA, get_tool_schemas


@pytest.fixture
def populated_mcp_target():
    target = "test_target"
    dirs = settings.get_target_dirs(target)
    dirs.ensure_exists()

    (dirs.raw_data_dir / "sample.md").write_text("# Doc\nSample text", encoding="utf-8")
    (dirs.vault_dir / "mindset_summary.md").write_text("# Mindset\nEmpirical stance", encoding="utf-8")
    (dirs.vault_dir / "knowledge_graph.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    (dirs.vault_dir / "character.json").write_text(json.dumps({"name": "Test Target", "bio": ["A target"]}), encoding="utf-8")

    skill_file = dirs.learned_skills_dir / "generated_mapping_skill.json"
    skill_file.write_text(json.dumps({
        "behavioral_profile": {
            "extended_psychometrics": {"machiavellianism": {"score": 0.7}}
        },
        "empirical_decision_boundaries": {},
        "generation_rules": ["[TACTICAL_TRANSACTIONALISM] Hedef fayda arar."]
    }), encoding="utf-8")

    return target


def test_mcp_initialize():
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "who-twin-mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_tools_list():
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = server.handle_request(req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "who_chat" in tool_names
    assert "who_simulate" in tool_names
    assert "who_get_profile" in tool_names
    assert "who_explain" in tool_names
    assert "who_academic_research" in tool_names
    assert "who_compose_outreach" in tool_names
    assert "who_list_targets" in tool_names
    assert "who_negotiate_decision" not in tool_names
    assert len(tool_names) == 7
    for tool in tools:
        ann = tool.get("annotations", {})
        assert isinstance(ann.get("readOnlyHint"), bool)
        assert isinstance(ann.get("destructiveHint"), bool)
        assert isinstance(ann.get("idempotentHint"), bool)
        assert isinstance(ann.get("openWorldHint"), bool)


def test_mcp_execute_list_targets(populated_mcp_target):
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "who_list_targets",
            "arguments": {},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert isinstance(data, list)
    assert any(t["slug"] == populated_mcp_target for t in data)


def test_mcp_execute_get_profile(populated_mcp_target):
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "who_get_profile",
            "arguments": {"target": populated_mcp_target},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    data = json.loads(content)
    assert "psychometrics" in data
    assert "operational_rules" in data


def test_mcp_execute_chat(populated_mcp_target):
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "who_chat",
            "arguments": {"target": populated_mcp_target, "message": "Merhaba."},
        },
    }
    resp = server.handle_request(req)
    assert resp is not None
    content = resp["result"]["content"][0]["text"]
    assert isinstance(content, str)
    assert len(content) > 10


def test_mcp_execute_simulate_explain_and_compose(populated_mcp_target):
    server = MCPServer()
    for idx, (t_name, args) in enumerate([
        ("who_simulate", {"target": populated_mcp_target, "scenario": "Test proposal"}),
        ("who_explain", {"target": populated_mcp_target, "claim": "Empirical stance"}),
        ("who_compose_outreach", {"target": populated_mcp_target, "goal": "Research collaboration"}),
    ], start=6):
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": idx,
            "method": "tools/call",
            "params": {"name": t_name, "arguments": args},
        })
        assert resp is not None
        assert "result" in resp


def test_mcp_export_tool_schemas_multi_provider_and_dispatch(populated_mcp_target):
    mcp_tools = get_tool_schemas("mcp")
    openai_tools = get_tool_schemas("openai")
    claude_tools = get_tool_schemas("anthropic")
    gemini_tools = get_tool_schemas("gemini")

    assert len(mcp_tools) == 7
    assert len(openai_tools) == 7
    assert len(claude_tools) == 7
    assert len(gemini_tools) == 7

    assert all(t["type"] == "function" and "parameters" in t["function"] for t in openai_tools)
    assert all("input_schema" in t for t in claude_tools)
    assert all("parameters" in t for t in gemini_tools)

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_tool_schemas("unknown_provider")

    server = MCPServer()
    res_openai = server.dispatch_tool_call("who_get_profile", json.dumps({"target": populated_mcp_target}))
    res_claude = server.dispatch_tool_call("who_get_profile", {"target": populated_mcp_target})
    assert res_openai["slug"] == populated_mcp_target
    assert res_claude["slug"] == populated_mcp_target
