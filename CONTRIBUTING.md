# Contributing to who??

Thank you for your interest in contributing to **who??**! We welcome contributions to our cognitive modeling engine, OSINT harvesting pipelines, and autonomous research agents.

## Code of Conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tilkiii/the-project-who.git
   cd the-project-who
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Guidelines

- **Clean Namespace:** All core library logic must live inside `src/who/`. Import paths must always be `from who import ...` or `from who.submodule import ...`.
- **Epistemic Honesty:** All psychometric and behavioral profiling logic must enforce probabilistic bounds and evidence citation. Deterministic assertions without empirical support are rejected.
- **Code Style:** We use [Ruff](https://astral.sh/ruff) for linting and formatting. Run:
  ```bash
  ruff check src/who tests/ examples/
  ruff format src/who tests/ examples/
  ```
- **Type Annotations:** Annotate public function and method signatures with standard type hints.
- **Testing:** Add pytest test cases under `tests/` for any new engine or utility. Verify test suite passes:
  ```bash
  pytest
  ```

## Submitting Pull Requests

1. Fork the repo and create a topic branch from `main` (e.g. `feature/mcp-streaming` or `fix/vector-filter`).
2. Commit your changes with descriptive, conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`).
3. Ensure all tests and linters pass locally.
4. Push your branch to GitHub and open a Pull Request using the provided PR template.
