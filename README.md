<div align="center">

<img src="assets/logo.svg" alt="who?? logo" width="120" height="120" />

# who??
### Evidence-Grounded Cognitive Digital Twin, Psycholinguistic & Behavioral Modeling Toolkit

[![CI](https://github.com/tilkiii/the-project-who/actions/workflows/ci.yml/badge.svg)](https://github.com/tilkiii/the-project-who/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Server](https://img.shields.io/badge/MCP-Server%20Ready-purple.svg)](https://modelcontextprotocol.io)
[![M8ven Score](https://m8ven.ai/badge/mcp/tilkiii/the-project-who?variant=verified)](https://m8ven.ai/mcp/tilkiii/the-project-who?s=readme)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

<p align="center">
  <b>Build evidence-grounded cognitive digital twins from public corpora, transcripts, and documents — analyze linguistic markers, knowledge graphs, conceptual contradictions, and counterfactual scenarios across Claude, ChatGPT (OpenAI), Google Gemini, local Ollama, or 100% Offline execution.</b>
</p>

</div>

---

## Overview

**`who??`** is an open-source framework for **evidence-grounded cognitive digital twins and psycholinguistic behavioral analysis**. Instead of prompting a generic role-play chatbot that fabricates opinions, `who??` extracts structured signals directly from a subject's writings, speeches, documents, and dialogue logs:

- **Evidence & Citation First:** Claims, twin chat replies, and simulated reactions are linked back to retrieved primary corpus quotes and concept-graph relationships, paired with explicit probabilistic framing rather than deterministic claims.
- **Universal AI Stack & MCP Ready:** Works out of the box with **Anthropic Claude**, **OpenAI / ChatGPT**, **Google Gemini**, **Groq**, **DeepSeek**, **OpenRouter**, **Mistral**, local **Ollama / vLLM**, and **100% zero-API offline mode**, plus native **MCP (`who mcp`)** and multi-provider JSON tool schemas (`get_tool_schemas`).
- **Strict Zero-Substitution Identity Gate:** Rejects unauthorized name expansions or third-party record merges during academic and OSINT ingestion via strict token matching (`entity_validator`).

```
   Primary Footprint & Corpus          Psycholinguistic & Graph Core         Any LLM, Agent, or Offline CLI
 [Articles / Transcripts / PDFs] ──► [Feature Extractor + Cognitive Vault] ──► [Claude / ChatGPT / Gemini / Ollama]
 [Dialogue Logs / RSS / Papers ] ──► [Vector Memory + Concept Graph      ] ──► [MCP Tools / SDK / CLI Reports    ]
```

> [!TIP]
> **⚡ Instant MCP Setup (Claude Desktop, Claude Code, Cursor):**
> Add `who??` to `claude_desktop_config.json` or your MCP client config:
> ```json
> {
>   "mcpServers": {
>     "who": {
>       "command": "who",
>       "args": ["mcp"]
>     }
>   }
> }
> ```
> *Exposes 7 read/research tools (`who_list_targets`, `who_get_profile`, `who_chat`, `who_simulate`, `who_explain`, `who_academic_research`, `who_compose_outreach`) over JSON-RPC 2.0 stdio.*

---

## Scientific & Mathematical Foundations

| Capability | Methodology & Implemented Engine | What It Does |
|---|---|---|
| **Negation-Aware Bayesian Psychometrics** | Clause-scoped syntactic negation guards (English & Turkish) + Beta-Binomial **95% Credible Intervals** across lexical markers (*Machiavellianism, Need for Closure, Prospect Loss Framing β, Affective Volatility, Moral Foundations, Epistemic Calibration*) | Prevents false keyword hits (e.g., *"no risk"*, *"not only fair"*, *"-mam gerekiyor"* vs *"kabul etmem"*) and scales credible intervals with corpus size |
| **CAPS Context Matrix & Stated vs. Revealed Framing** | Context-conditional keyword bucketing (`IF [Context] -> THEN [Signature]`) + separation of stated ideals vs. conditional red-line commitments (`[NECESSARY_CONDITION_RED_LINE]`) | Maps how a subject's framing shifts across Academic, Crisis, Strategic, and Interpersonal contexts |
| **Knowledge Graph & Structural Balance** | Local JSON/NetworkX + optional Neo4j concept graph extraction (`ADVOCATES`, `CRITIQUES`, `CONTRADICTS`), Signed Graph Laplacian spectrum ($\lambda_{\min}$), and simulated-annealing Ising frustration metrics | Identifies central concepts, recurring tensions, and structural contradictions across a subject's own corpus |
| **Boltzmann-Gibbs Scenario Simulation & Swarms** | Corpus- and knowledge-graph-conditioned Softmax/Boltzmann-Gibbs scenario models, Monte Carlo sensitivity sweeps, and multi-twin `SwarmArena` debates | Models plausible scenario reactions with explicit provenance, entropy, and free-energy diagnostics (non-predictive scenario modeling) |
| **Temporal Drift, Chronomics & Verification** | Circadian timestamp bucketing (`CognitiveChronomics`), Ornstein-Uhlenbeck longitudinal drift analysis (`TemporalDriftDetector`), Temporal Holdout Backtesting (`ProfileEvaluator`), and stylometric Jensen-Shannon Divergence (`JSD`) | Tracks day/night rhetorical shifts, longitudinal vocabulary changes, citation validity, and candidate-response style fidelity |

---

## Quickstart

### 1. Installation

**Core Installation (Python 3.10+):**
```bash
pip install the-project-who
```

Or install from source:
```bash
git clone https://github.com/tilkiii/the-project-who.git
cd the-project-who
pip install -e .
```

**Optional Extras:**
```bash
# Persistent ChromaDB HNSW, Neo4j driver, scholarly, yt-dlp:
pip install -e ".[full]"

# Heavy local multimodal ML (Docling, PaddleOCR, BERTopic, Crawl4AI):
pip install -e ".[ml]"

# Development & testing tools (pytest, ruff, mypy):
pip install -e ".[dev]"
```

> **Note for Windows users:** If the `who` executable is not on your PATH after installation, invoke the CLI directly with `python -m who.cli`.

### 2. Provider Setup (Claude, ChatGPT, Gemini, Local Ollama, or Offline)

All quantitative feature extraction, graph analytics, and offline corpus-grounded synthesis work **without any API key**. For full natural-language generation, copy `.env.example` to `.env` and set **whichever provider you already use**:

```ini
# Pick ONE provider (or omit all to use offline corpus-grounded fallback):

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key_here
# WHO_PROVIDER=anthropic

# OpenAI / ChatGPT
OPENAI_API_KEY=your_openai_key_here
# WHO_PROVIDER=openai

# Google Gemini
GEMINI_API_KEY=your_gemini_key_here
# WHO_PROVIDER=gemini

# Fast Open-Weight / Local Providers (optional)
# GROQ_API_KEY=...
# DEEPSEEK_API_KEY=...
# OPENROUTER_API_KEY=...
# OLLAMA_API_BASE=http://localhost:11434
```

---

## Usage: Python SDK, Multi-LLM Tool Calling, MCP & CLI

### Option A: Direct Python SDK

```python
from who import CognitiveTwin, AcademicResearcher, list_targets

print(list_targets())

twin = CognitiveTwin("Target X", mode="autodetect")

# 1. Profile & active behavioral rules
profile = twin.get_profile()
rules = twin.get_operational_rules()

# 2. In-character grounded dialogue & analytical Q&A
reply = twin.chat("What standards do you require before endorsing an organizational proposal?")
insight = twin.ask("What recurring principles appear in the subject's primary documents?")

# 3. Claim verification & scenario simulation
verification = twin.explain("Subject prefers empirical verification over centralized authority")
sim = twin.simulate_monte_carlo(
    "A counterparty requests exclusive IP transfer with no audit mechanism.",
    iterations=25,
)

# 4. Longitudinal drift & stylometric fidelity check
drift = twin.analyze_drift(["New public statement emphasizing transparency and verifiable audits."])
check = twin.verify_response("Empirical bounds and reciprocal governance remain essential.")
```

### Option B: Connect Directly to ChatGPT (OpenAI), Claude (Anthropic), Gemini, or Ollama Tool Calling

Export native tool schemas for the 7 Cognitive Digital Twin tools (`get_tool_schemas`) and dispatch calls cleanly through `MCPServer().dispatch_tool_call`:

```python
from openai import OpenAI
from who.mcp_server import MCPServer, get_tool_schemas

mcp = MCPServer()

# 1. ChatGPT / OpenAI Function Calling
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Inspect Target X's cognitive profile and summarize their red lines."}],
    tools=get_tool_schemas("openai"),
)
tc = resp.choices[0].message.tool_calls[0]
result = mcp.dispatch_tool_call(tc.function.name, tc.function.arguments)

# 2. Anthropic Claude Tool Use
# tools=get_tool_schemas("anthropic")
# result = mcp.dispatch_tool_call(tool_use_block.name, tool_use_block.input)

# 3. Google Gemini Function Calling
# tools=get_tool_schemas("gemini")
# result = mcp.dispatch_tool_call(function_call.name, dict(function_call.args))

# 4. Local Ollama (OpenAI-Compatible Function Calling)
# tools=get_tool_schemas("ollama")
```

### Option C: Model Context Protocol (MCP) — Claude Desktop, Claude Code, Cursor

Run `who mcp` over JSON-RPC stdio. Available MCP tools (all annotated with safety hints):
- `who_list_targets()`: List all profiled targets in the local vault.
- `who_get_profile(target)`: Retrieve psychometric dimensions, CAPS matrix, stated vs. revealed framing, and operational rules.
- `who_chat(target, message, tone)`: Direct in-character dialogue with the target's cognitive digital twin.
- `who_simulate(target, scenario, monte_carlo, iterations)`: Simulate scenario reactions and Boltzmann-Gibbs Monte Carlo distributions.
- `who_explain(target, claim)`: Verify a claim against primary corpus quotes and knowledge graph paths.
- `who_academic_research(topic, target, limit, ingest)`: Search arXiv, PubMed, Semantic Scholar, and Crossref with 0–100 Scientific Rigor scoring.
- `who_compose_outreach(target, goal, context)`: Draft communication tailored to the profiled subject's epistemic framing.

### Option D: Full CLI Pipeline

```bash
# Diagnostics, configured AI providers & local profiles
who doctor
who providers
who targets

# Ingest -> Learn -> Build Vault -> Audit -> Interact
who harvest --target "Subject A" --docs-dir ./path/to/texts
who learn --target "Subject A"
who build-vault --target "Subject A"
who evaluate-profile --target "Subject A"
who psych-insights --target "Subject A"

# Query, chat, simulate, debate, & track drift (demo profile: Target X)
who ask --target "Target X" --prompt "What criteria drive strategic trust?"
who chat --target "Target X"
who explain --target "Target X" --claim "Prioritizes empirical verification"
who simulate --target "Target X" --scenario "Proposal demanding unilateral veto removal" --monte-carlo --iterations 25
who compose --target "Target X" --goal "Invite to peer-reviewed workshop"
who arena --targets "Target X, Subject A" --topic "Transparency vs institutional autonomy" --rounds 2
who monitor --target "Target X" --drift
```

Explore runnable scripts in [`examples/`](examples/):
- [`01_quick_chat.py`](examples/01_quick_chat.py): Connect and chat with a cognitive twin.
- [`02_simulate_scenario.py`](examples/02_simulate_scenario.py): Run scenario simulations.
- [`03_academic_research.py`](examples/03_academic_research.py): Search peer-reviewed literature across arXiv/PubMed.
- [`04_mcp_agent_client.py`](examples/04_mcp_agent_client.py): Direct JSON-RPC MCP server client.
- [`05_monte_carlo_negotiation.py`](examples/05_monte_carlo_negotiation.py): Multi-iteration Monte Carlo game-theoretic scenario simulation.

---

## Methodological Limitations & Ethical Scope

To maintain strict scientific transparency, users and researchers should account for the following boundaries:

1. **Non-Clinical Scope (Goldwater Rule Alignment):** `who??` models observable linguistic, rhetorical, and decision-framing patterns from text corpora using transparent lexicon/statistical heuristics. It does **not** perform clinical psychological diagnosis, mental health assessment, or deterministic mind-reading.
2. **Heuristic Lexical Indicators & Short-Corpus Thresholds:** Core indices combine curated English/Turkish lexicons, privative/syntactic negation guards, and Beta-Binomial uncertainty intervals. They emit explicit `INSUFFICIENT_DATA` states below 20 words and represent lexical behavioral indicators — not clinical psychometric diagnoses. Scenario simulations are explicitly labeled as corpus-conditioned scenario models rather than trained empirical predictors.
3. **3-Tier Literature Grounding:** Retrieved literature is strictly separated into `QUANTITATIVE_EXTRACTED` (explicit `r` and `N`), `QUALITATIVE_ABSTRACT_MATCH` (excluded from Fisher-z Bayesian updates), and `REFERENCE_BASELINE_PRIOR`. Quantitative effect sizes serve strictly as **Bayesian population priors** and do not establish individual-level causality.
4. **Corpus Quality & Consent:** Public texts reflect public personas; simulated outputs remain strictly bounded by input volume, source diversity, and selection bias. This toolkit is intended for computational social science, public-domain analysis, consented self-twinning, and research workflows — never covert surveillance or non-consensual private profiling (see [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).

---

## Documentation & Community

- **Documentation**: Explore guides and architecture specifications in the [`docs/`](docs/) directory.
- **Contributing**: Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
- **Security**: For reporting security vulnerabilities, refer to [SECURITY.md](SECURITY.md).

## License

MIT License. Designed for evidence-grounded research, behavioral simulation, and personal agent tooling.

## Thanks
We thank Sude Naz Akcabağ for her contributions.
