# Getting Started

## Installation

Install `who??` directly via pip or clone from GitHub:

```bash
git clone https://github.com/tilkiii/the-project-who.git
cd the-project-who
pip install -e .
```

Optional extras:

```bash
# Persistent ChromaDB HNSW, Neo4j driver, scholarly, yt-dlp:
pip install -e ".[full]"

# Development & testing tools (pytest, ruff, mypy):
pip install -e ".[dev]"
```

## Optional LLM Provider Setup

Quantitative feature extraction, graph analytics, and offline corpus-grounded synthesis run **100% offline with zero API keys**. For full natural-language generation, copy `.env.example` to `.env` and configure any provider you prefer:

```ini
ANTHROPIC_API_KEY=...   # Claude
OPENAI_API_KEY=...      # ChatGPT (GPT-4o / GPT-4o-mini)
GEMINI_API_KEY=...      # Google Gemini
# Or Groq / DeepSeek / OpenRouter / Mistral, or local Ollama:
# OLLAMA_API_BASE=http://localhost:11434
```

Verify detected providers and system health at any time:

```bash
who --version
who doctor
who providers
```

## Quick CLI Usage

Once installed, the `who` command (or `python -m who.cli` on Windows) is available in your shell:

### 1. List Available Profiles
```bash
who targets
```

### 2. Interactive Twin Chat
```bash
who chat --target "Target X"
```

### 3. Ask Analytical Evidence Questions
```bash
who ask --target "Target X" --prompt "What principles guide institutional decisions?"
```

### 4. Scenario & Monte Carlo Simulation
```bash
who simulate --target "Target X" --scenario "Proposal demands unilateral control without audit transparency." --monte-carlo --iterations 25
```

### 5. Autonomous Academic Literature Search
```bash
who research --topic "Bayesian mechanism design equilibrium" --source arxiv
```

## Running the MCP Server (Claude Desktop, Claude Code, Cursor)

To expose `who??` as a Model Context Protocol tool provider to Claude Desktop, Claude Code, Cursor, or any MCP client:

```bash
who mcp
```

Add the following to your `claude_desktop_config.json` (or `.mcp.json` / Cursor MCP config):

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
