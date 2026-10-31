"""
04_mcp_agent_client.py - Model Context Protocol (MCP) Integration
=================================================================
Demonstrates how external LLMs and agent harnesses (Claude Desktop, Cursor, LangChain)
can interact with `who??` as a standard JSON-RPC 2.0 MCP server.
"""

import json
from who.mcp_server import MCPServer


def run_rpc(server: MCPServer, request_id: int, method: str, params: dict) -> dict:
    req = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    return server.handle_request(req)


def main():
    print("=== Initializing who?? MCP Server ===")
    server = MCPServer()

    # 1. MCP Handshake
    init_res = run_rpc(server, 1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "ExampleMCPClient", "version": "1.0.0"},
    })
    server_info = init_res["result"]["serverInfo"]
    print(f"Connected to MCP Server: {server_info['name']} (v{server_info['version']})")

    # 2. Discover Tools
    tools_res = run_rpc(server, 2, "tools/list", {})
    tools = tools_res["result"]["tools"]
    print(f"\nDiscovered {len(tools)} MCP Tools:")
    for t in tools:
        print(f" - {t['name']}: {t['description']}")

    # 3. Call 'who_get_profile' for Target X
    print("\nCalling 'who_get_profile' for 'Target X' via JSON-RPC...")
    profile_call = run_rpc(server, 3, "tools/call", {
        "name": "who_get_profile",
        "arguments": {"target": "Target X"},
    })
    content = json.loads(profile_call["result"]["content"][0]["text"])
    print(f"Target Name: {content.get('name')}")
    print(f"Cognitive Style: {content.get('behavioral_profile', {}).get('cognitive_style')}")
    print(f"Operational Rules: {content.get('operational_rules')}")

    # 4. Call 'who_chat' tool
    print("\nCalling 'who_chat' tool via JSON-RPC...")
    chat_call = run_rpc(server, 4, "tools/call", {
        "name": "who_chat",
        "arguments": {
            "target": "Target X",
            "message": "Summarize your stance on unilateral contract revisions in one concise sentence.",
        },
    })
    twin_reply = chat_call["result"]["content"][0]["text"]
    print(f"\nResponse from MCP Tool:\n{twin_reply}")


if __name__ == "__main__":
    main()
