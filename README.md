# who?? — Evidence-Grounded Cognitive Digital Twin & Behavioral Modeling Toolkit

A lightweight, scientific framework to construct **empirical, evidence-backed Cognitive Digital Twins** of individuals or organizations based on their real-world verbal and written footprint.

Instead of generic chatbot roleplay, `who??` reverse-engineers a target's cognitive operating system—grounding reasoning, psychometrics, and behavioral boundaries in verified primary discourse and peer-reviewed academic literature.

Designed to serve as a **personal agent toolkit** via the Model Context Protocol (MCP) or direct Python SDK.

---

## Key Capabilities

- **Evidence-Grounded Dialogue (`who chat`)**: Converse in-character with the digital twin across formal keynote and 1-on-1 registers. Responses adhere to verified memory, persona lore, and behavioral boundaries.
- **Hypothetical Scenario Simulator (`who simulate`)**: Model how the target plausibly reacts to proposals, ethical dilemmas, or adversarial stress-tests before real-world engagement.
- **5-Axis Extended Psychometrics**: Quantifies Machiavellianism (SD3), Need for Cognitive Closure (NFCS), Prospect Loss Aversion, Affective Volatility (Circumplex), and Moral Foundations (MFT).
- **AI Scientific Referee & Rigor Index**: Evaluates retrieved papers across arXiv, PubMed, Semantic Scholar, and OpenAlex using a 0–100 rigor score ($S_N, S_r, S_v, S_c$) to derive empirical decision boundaries.
- **Cognitive Knowledge Graph (Neo4j / NetworkX)**: Models conceptual networks, entity relations, and ideological tensions (cognitive dissonance).
- **Personal Agent Integration (MCP Server & SDK)**: Expose twins, simulations, and research tools directly to Claude Desktop, Cursor, Antigravity, or custom multi-agent pipelines.

---

## Architecture: 7-Layer Cognitive Memory

```
Layer 7: Character Simulation & Dialogue Engine (TwinChat, Simulator, Oracle, Arena)
Layer 6: Scientific Referee & Meta-Analytic Synthesis (arXiv, PubMed, OpenAlex, Rigor 0-100)
Layer 5: Psychometrics & Empirical Decision Boundaries (5 Grand Dimensions, Behavioral Rules)
Layer 4: Semantic Topic Trajectories (BERTopic, UMAP, HDBSCAN Evolution)
Layer 3: Epistemic Knowledge Graph & GDS (Neo4j PageRank, Louvain Communities, Dissonance)
Layer 2: Semantic Vector Memory (ChromaDB Cosine Embeddings, Boilerplate Filter, PII Masking)
Layer 1: Multimodal Footprint Acquisition (YouTube Diarization, Web Scrapers, Documents, Social)
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/tilkiii/the-project-who.git
cd the-project-who
pip install -r requirements.txt
pip install -e . --no-deps
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

### Option A: Model Context Protocol (MCP Server)

Connect `who??` as a native tool server in Claude Desktop, Cursor, or any MCP client:

```json
{
  "mcpServers": {
    "who-twin": {
      "command": "python",
      "args": ["-m", "src.mcp_server"]
    }
  }
}
```

Or run via CLI:
```bash
who mcp
```

**Available MCP Tools:**
- `who_chat(target, message, tone)`: Direct dialogue with cognitive digital twin.
- `who_simulate(target, scenario)`: Run adversarial / hypothetical counter-move simulation.
- `who_get_profile(target)`: Retrieve psychometrics, boundaries, and active operational rules.
- `who_explain(target, claim)`: Verify claim against primary corpus and knowledge graph paths.
- `who_academic_research(topic, target, limit, ingest)`: Search peer-reviewed literature with AI Referee scoring.
- `who_compose_outreach(target, goal)`: Draft calibrated outreach with rejection stress-testing.
- `who_list_targets()`: List all profiled targets in the local vault.

---

### Option B: Python SDK

```python
from src import CognitiveTwin, AcademicResearcher, list_targets

# 1. Load digital twin
twin = CognitiveTwin("remzi_sanver")

# 2. Interactive dialogue
reply = twin.chat("What is your view on institutional meritocracy?")
print(reply)

# 3. Simulate scenario reaction
sim = twin.simulate_scenario("An enterprise offers asymmetric benefits in exchange for institutional loyalty")
print(sim["simulation_analysis"])

# 4. Inspect psychometrics and active rules
profile = twin.get_profile()
print(profile["psychometrics"])
print(twin.get_operational_rules())

# 5. Autonomous academic research
researcher = AcademicResearcher(target_name="remzi_sanver")
results = researcher.research_topic("cognitive closure in decision making", limit=3)
```

---

## CLI Reference

```bash
# Harvest public footprint (YouTube, web articles, PDFs)
who harvest --target "Remzi Sanver" --urls "https://example.com/speech"

# Learn topics and extract behavioral features
who learn --target "Remzi Sanver"

# Build knowledge graph and semantic embeddings
who build-vault --target "Remzi Sanver"

# Interactive terminal chat with active guardrails
who chat --target "Remzi Sanver"

# Simulate hypothetical reaction
who simulate --target "Remzi Sanver" --scenario "Controversial proposal"

# Multi-agent debate arena
who arena --targets "Remzi Sanver, Target B" --topic "AI Ethics"

# Start MCP server
who mcp
```

---

## License

MIT License. Designed for evidence-grounded research, behavioral simulation, and personal agent tooling.
