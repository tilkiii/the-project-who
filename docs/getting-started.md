# Getting Started

## Installation

Install `who??` directly via pip or clone from GitHub:

```bash
git clone https://github.com/tilkiii/the-project-who.git
cd the-project-who
pip install -e .
```

For development dependencies (testing, linting, docs):

```bash
pip install -e ".[dev]"
```

## Quick CLI Usage

Once installed, the `who` command is available in your shell:

### 1. List Available Profiles
```bash
who targets
```

### 2. Interactive Twin Chat
```bash
who chat --target "Target X"
```

### 3. Scenario Simulation
```bash
who simulate --target "Target X" --scenario "Counterpart demands 40% equity or threatens a hostile fork."
```

### 4. Autonomous Academic Literature Search
```bash
who research --topic "Bayesian mechanism design equilibrium" --source arxiv
```

## Running the MCP Server

To expose `who??` as a Model Context Protocol tool provider to Claude Desktop, Cursor, or other AI agents:

```bash
who mcp
```

Add the following to your `claude_desktop_config.json`:

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
