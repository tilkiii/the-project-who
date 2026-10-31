"""
Unit tests for who?? Model Context Protocol (MCP) Server.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.config import settings
from src.mcp_server import MCPServer, TOOLS_SCHEMA


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
    assert "who_academic_research" in tool_names
    assert "who_list_targets" in tool_names


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
