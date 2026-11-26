# Python Agent SDK Reference

The `who` package provides a Python interface designed for AI agents (**Anthropic Claude**, **OpenAI / ChatGPT**, **Google Gemini**, **Ollama**), MCP servers, and offline behavioral analysis workflows.

## Top-Level Exports

```python
from who import (
    CognitiveTwin,
    AcademicResearcher,
    list_targets,
    get_target,
)
```

For native multi-LLM tool calling across OpenAI, Claude, Gemini, and MCP:

```python
from who.mcp_server import MCPServer, get_tool_schemas
```

---

## `list_targets()`

Retrieves all targets currently available in the local vault.

```python
targets = list_targets()
for t in targets:
    print(t["slug"], t["is_ready"], t["raw_documents_count"])
```

---

## `CognitiveTwin`

Interface to an individual target's cognitive digital twin.

### Constructor
```python
twin = CognitiveTwin(target_name: str, mode: str = "autodetect")
```

### Methods

#### `chat(message: str, tone: Optional[str] = None) -> str`
Sends an interactive message and receives an in-character response calibrated to the target's persona, memory, and operational guardrails.

#### `ask(question: str) -> Dict[str, Any]`
Asks an analytical question backed by primary corpus quotes, knowledge graph links, and a quantified confidence score.

#### `simulate_scenario(scenario: str) -> Dict[str, Any]`
Simulates how the target would plausibly react to a hypothetical scenario proposition.

#### `simulate_monte_carlo(scenario: str, iterations: int = 25, ...) -> Dict[str, Any]`
Runs multi-iteration Boltzmann-Gibbs / QRE scenario simulation returning branch probabilities, payoff bounds, CAPS context shifts, and sensitivity findings.

#### `explain(claim: str) -> Dict[str, Any]`
Traces any claim or behavioral assertion back to primary quotes, graph paths, and epistemic confidence.

#### `compose_outreach(goal: str, context: Optional[str] = None) -> Dict[str, Any]`
Generates communication drafts aligned with the target's epistemic framing and stress-tests them against potential objections.

#### `analyze_drift(new_texts=None, new_documents=None) -> Dict[str, Any]`
Measures longitudinal psycholinguistic drift relative to the target's baseline corpus.

#### `backtest(holdout_ratio: float = 0.25) -> Dict[str, Any]`
Runs Temporal Holdout Validation over chronological target texts to report Brier Score and holdout consistency.

#### `verify_response(candidate_text: str) -> Dict[str, Any]`
Checks a candidate response against the target's stylometric profile (Jensen-Shannon Divergence) and boundary rules.

#### `get_profile() -> Dict[str, Any]`
Returns full target metadata including psychometrics, CAPS matrix, stated vs. revealed framing, character lore, and active decision boundaries.

#### `get_operational_rules() -> List[str]`
Returns the active behavioral guardrails (e.g., `["[TACTICAL_TRANSACTIONALISM] ...", "[STOIC_EQUANIMITY] ..."]`).

#### `get_knowledge_graph() -> Dict[str, Any]`
Returns the target's concept knowledge graph nodes and relationships.

---

## `AcademicResearcher`

Autonomous academic literature retrieval and empirical rigor evaluator across arXiv, PubMed, Semantic Scholar, and Crossref.

```python
from who import AcademicResearcher

researcher = AcademicResearcher()
res = researcher.research_topic("cognitive rigidity under stress", limit=5)
```

---

## Native LLM Tool Calling (`get_tool_schemas` + `MCPServer.dispatch_tool_call`)

```python
from who.mcp_server import MCPServer, get_tool_schemas

mcp = MCPServer()

# Supported values: "mcp", "openai", "chatgpt", "anthropic", "claude", "gemini", "google", "ollama", "vllm", "groq", "deepseek", "openrouter", "mistral"
openai_tools = get_tool_schemas("openai")
claude_tools = get_tool_schemas("anthropic")
gemini_tools = get_tool_schemas("gemini")
ollama_tools = get_tool_schemas("ollama")

# Dispatch from OpenAI JSON-string arguments or Claude/Gemini dict arguments:
result = mcp.dispatch_tool_call("who_get_profile", {"target": "Target X"})
```
