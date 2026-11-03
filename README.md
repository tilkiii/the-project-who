<p align="center">
  <img src="assets/logo.svg" alt="who?? logo" width="520" />
</p>

<p align="center">
  <a href="https://github.com/tilkiii/the-project-who/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/tilkiii/the-project-who/ci.yml?branch=main&label=CI&logo=github" alt="CI Status" /></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-reference-teal.svg?logo=googledocs&logoColor=white" alt="Documentation" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" /></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" alt="Python Versions" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT" /></a>
</p>

An open-source scientific framework to construct **empirical, evidence-backed Cognitive Digital Twins** of individuals or organizations based on their real-world verbal and written footprint.

Instead of generic chatbot roleplay, `who??` reverse-engineers a target's cognitive operating system—grounding reasoning, psychometrics, and behavioral boundaries in verified primary discourse and peer-reviewed academic literature.

Designed to serve as a **personal agent toolkit** via the Model Context Protocol (MCP) or direct Python SDK.

---

## Terminal Showcase

<p align="center">
  <img src="assets/terminal_demo.svg" alt="who terminal showcase" width="100%" />
</p>

---

## Key Capabilities

- **Evidence-Grounded Dialogue (`who chat`)**: Converse in-character with the digital twin across formal and 1-on-1 registers. Responses strictly adhere to verified memory and calibrated decision boundaries.
- **Hypothetical Scenario Simulator (`who simulate`)**: Model how the target plausibly reacts to proposals, ethical dilemmas, or adversarial stress-tests before real-world engagement.
- **5-Axis Extended Psychometrics**: Quantifies Machiavellianism (SD3), Need for Cognitive Closure (NFCS), Prospect Loss Aversion, Affective Volatility, and Moral Foundations (MFT).
- **AI Scientific Referee & Rigor Index**: Evaluates retrieved papers across arXiv, PubMed, Semantic Scholar, and OpenAlex using a 0–100 rigor score ($S_N, S_r, S_v, S_c$) to derive empirical decision boundaries.
- **Cognitive Knowledge Graph (NetworkX / Neo4j)**: Models conceptual networks, entity relations, and ideological tensions (cognitive dissonance).
- **Personal Agent Integration (MCP Server & SDK)**: Expose twins, simulations, and research tools directly to Claude Desktop, Cursor, Antigravity, or custom multi-agent pipelines.

---

## System Architecture & Data Flow

```mermaid
flowchart LR
    subgraph Ingestion["1. Multimodal Ingestion & Verification"]
        A[YouTube Transcripts] & B[Web Articles & Speeches] & C[Academic Treatises] --> D[PII Mask & Boilerplate Filter]
        D --> E[Zero-Substitution Gate]
    end

    subgraph Memory["2. Epistemic Memory & Graphs"]
        E --> F[ChromaDB Vector Store]
        E --> G[NetworkX / Neo4j Knowledge Graph]
    end

    subgraph Modeling["3. Cognitive & Psychometric Modeling"]
        F & G --> H[5-Dimension Psychometrics]
        H --> I[Empirical Decision Boundaries]
        I --> J[Autonomous Academic Referee]
    end

    subgraph Interfaces["4. Agent & Protocol Delivery"]
        J --> K[Python SDK]
        J --> L[Model Context Protocol Server]
        J --> M[CLI Interactive Shell]
    end
```

---

## Tech Stack

| Category | Technologies & Tools |
|---|---|
| **Core & Systems** | Python 3.10+, Pydantic v2, Typer, Rich |
| **Agent Protocols** | Model Context Protocol (MCP - JSON-RPC 2.0 stdio) |
| **Vector & Graph Storage** | ChromaDB (Cosine embeddings), NetworkX, Neo4j Graph Data Science |
| **LLM & Inference** | Google Gemini (Native API), OpenAI, Anthropic Claude, Groq, DSPy |
| **Academic Meta-Analysis** | arXiv API, PubMed (E-Utilities), Semantic Scholar, OpenAlex |
| **Quality & CI/CD** | Pytest, Ruff, Pre-commit, GitHub Actions Matrix CI |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/tilkiii/the-project-who.git
cd the-project-who
pip install -e .
```

> **Note for Windows users:** If the `who` command is not recognized in your shell after installation, ensure Python's `Scripts/` directory is in your system PATH, or invoke the CLI directly using `python -m who.cli`.

For development dependencies (testing, linting, docs):
```bash
pip install -e ".[dev]"
```

### 2. Configure Environment (`.env`)

```bash
cp .env.example .env
```

Add your preferred AI provider key:
```ini
# Google Gemini (Default)
GEMINI_API_KEY=your_gemini_api_key_here

# Or other providers:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_claude_key
# GROQ_API_KEY=your_groq_key
```

Verify configured providers:
```bash
who providers
```

---

## Personal Agent Integration

### Option A: Python SDK

```python
from who import CognitiveTwin, AcademicResearcher, list_targets

# 1. Discover available targets in local vault
targets = list_targets()
print([t["name"] for t in targets])

# 2. Connect to digital twin
twin = CognitiveTwin("Target X", mode="formal")

# 3. Interactive dialogue
reply = twin.chat("What is your equilibrium strategy when negotiating with an asymmetric counterpart?")
print(reply)

# 4. Simulate scenario reaction
sim = twin.simulate_scenario("Counterpart demands unilateral veto power under threat of network fork.")
print(sim["simulation_analysis"])

# 5. Autonomous academic research & literature grounding
researcher = AcademicResearcher(target_name="Target X")
results = researcher.research_topic("Bayesian mechanism design equilibrium", limit=3)
```

Explore runnable examples in the [`examples/`](examples/) cookbook:
- [`01_quick_chat.py`](examples/01_quick_chat.py): Connect and chat with a cognitive twin.
- [`02_simulate_scenario.py`](examples/02_simulate_scenario.py): Run adversarial scenario simulations.
- [`03_academic_research.py`](examples/03_academic_research.py): Search peer-reviewed literature across arXiv/PubMed.
- [`04_mcp_agent_client.py`](examples/04_mcp_agent_client.py): Direct JSON-RPC MCP server client.

---

### Option B: Model Context Protocol (MCP Server)

Connect `who??` as a native tool server in Claude Desktop, Cursor, or any MCP client:

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

Or run via CLI:
```bash
who mcp
```

**Available MCP Tools:**
- `who_chat(target, message, mode)`: Direct dialogue with cognitive digital twin.
- `who_simulate(target, scenario)`: Run adversarial / hypothetical counter-move simulation.
- `who_get_profile(target)`: Retrieve psychometrics, boundaries, and active operational rules.
- `who_explain(target, claim)`: Verify claim against primary corpus and knowledge graph paths.
- `who_academic_research(topic, target, limit, ingest)`: Search peer-reviewed literature with AI Referee scoring.
- `who_compose_outreach(target, goal)`: Draft calibrated outreach with rejection stress-testing.
- `who_list_targets()`: List all profiled targets in the local vault.

---

## CLI Reference

```bash
# List available targets
who targets

# Interactive terminal chat with active guardrails
who chat --target "Target X"

# Simulate hypothetical reaction
who simulate --target "Target X" --scenario "Controversial proposal"

# Autonomous literature search
who research --topic "Mechanism design equilibrium" --source arxiv

# Start MCP server
who mcp
```

---

## Documentation & Community

- **Documentation**: Explore guides and architecture specifications in the [`docs/`](docs/) directory.
- **Contributing**: Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
- **Security**: For reporting security vulnerabilities, refer to [SECURITY.md](SECURITY.md).

## License

MIT License. Designed for evidence-grounded research, behavioral simulation, and personal agent tooling.
