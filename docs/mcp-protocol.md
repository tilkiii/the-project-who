# Model Context Protocol (MCP) & Native Function Calling

`who??` implements the official [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) specification over JSON-RPC 2.0 stdio, and exposes native JSON tool schemas (`who.mcp_server.get_tool_schemas`) for **Claude (Anthropic)**, **ChatGPT (OpenAI)**, **Google Gemini**, **Ollama** (OpenAI-compatible function calling), and **Cursor**.

## Supported MCP Tools

| Tool Name | Parameters | Description |
|---|---|---|
| `who_list_targets` | `{}` | Lists all profiled targets in the local vault. |
| `who_get_profile` | `{"target": str}` | Returns target's psychometric dimensions, CAPS matrix, and operational rules. |
| `who_chat` | `{"target": str, "message": str, "tone"?: str}` | Executes evidence-grounded in-character dialogue query. |
| `who_simulate` | `{"target": str, "scenario": str, "monte_carlo"?: bool, "iterations"?: int}` | Simulates scenario reactions or runs Boltzmann-Gibbs Monte Carlo sweeps. |
| `who_explain` | `{"target": str, "claim": str}` | Traces assertions to verified quotes and knowledge graph paths. |
| `who_academic_research` | `{"topic": str, "target"?: str, "limit"?: int, "ingest"?: bool}` | Queries arXiv, PubMed, Semantic Scholar, and Crossref with 0–100 Scientific Rigor scoring. |
| `who_compose_outreach` | `{"target": str, "goal": str, "context"?: str}` | Drafts communication tailored to the target's cognitive profile. |

---

## 1. Integration with Claude Desktop / Claude Code / Cursor (MCP)

Add this block to `claude_desktop_config.json` (or `.mcp.json` / Cursor MCP config):

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

Or invoke directly via Anthropic SDK tool use:

```python
from anthropic import Anthropic
from who.mcp_server import MCPServer, get_tool_schemas

mcp = MCPServer()
client = Anthropic()
resp = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=512,
    tools=get_tool_schemas("anthropic"),
    messages=[{"role": "user", "content": "Get Target X's cognitive profile and key operational rules."}],
)
for block in resp.content:
    if block.type == "tool_use":
        print(mcp.dispatch_tool_call(block.name, block.input))
```

---

## 2. Integration with ChatGPT / OpenAI Function Calling

```python
from openai import OpenAI
from who.mcp_server import MCPServer, get_tool_schemas

mcp = MCPServer()
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=get_tool_schemas("openai"),
    messages=[{"role": "user", "content": "Ask Target X's digital twin how they evaluate empirical claims."}],
)
tc = resp.choices[0].message.tool_calls[0]
print(mcp.dispatch_tool_call(tc.function.name, tc.function.arguments))
```

---

## 3. Direct Python MCP JSON-RPC Usage

```python
from who.mcp_server import MCPServer

server = MCPServer()
response = server.handle_request({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "who_explain",
        "arguments": {
            "target": "Target X",
            "claim": "Relies on empirical verification",
        },
    },
})
print(response["result"]["content"][0]["text"])
```
