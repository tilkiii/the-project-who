# Model Context Protocol (MCP) Server

`who??` implements the official [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) specification over JSON-RPC 2.0 stdio, allowing any LLM client to query cognitive twins as native tools.

## Supported MCP Tools

| Tool Name | Parameters | Description |
|---|---|---|
| `who_list_targets` | `{}` | Lists all profiled targets in the local vault. |
| `who_get_profile` | `{"target": str}` | Returns target's psychometric dimensions and operational rules. |
| `who_chat` | `{"target": str, "message": str, "mode"?: str}` | Executes in-character dialogue query. |
| `who_simulate` | `{"target": str, "scenario": str}` | Simulates counter-moves for strategic dilemmas. |
| `who_explain` | `{"target": str, "claim": str}` | Traces assertions to verified quotes and graph nodes. |
| `who_academic_research` | `{"topic": str, "target"?: str, "sources"?: list, "limit"?: int, "ingest"?: bool}` | Queries arXiv/PubMed with AI referee rigor evaluation. |
| `who_compose_outreach` | `{"target": str, "goal": str, "context"?: str}` | Drafts communication optimized for the target's cognitive profile. |

## Integration with Claude Desktop

Add this block to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "who": {
      "command": "who",
      "args": ["mcp"]
    }
  }
}
```

## Direct Python Usage

```python
from who.mcp_server import MCPServer

server = MCPServer()
response = server.handle_request({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "who_chat",
        "arguments": {
            "target": "Target X",
            "message": "What is our best counter-strategy?"
        }
    }
})
print(response["result"]["content"][0]["text"])
```
