"""
Integration tests for who?? Typer CLI subcommands.
"""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Enterprise-Grade Multimodal OSINT Engine" in result.output or "who" in result.output


def test_cli_harvest_and_pipeline(tmp_path: Path):
    target = "Test Target"
    # Create sample doc for harvest
    doc_file = tmp_path / "test_notes.md"
    doc_file.write_text("# Target Philosophy\n\n[TARGET_PRIMARY]: We prioritize empirical proof and open consensus protocols over centralized decree.", encoding="utf-8")

    # 1. Harvest
    res_harvest = runner.invoke(app, ["harvest", "--target", target, "--docs-dir", str(tmp_path)])
    assert res_harvest.exit_code == 0
    assert "Harvest completed" in res_harvest.output

    # 2. Learn
    res_learn = runner.invoke(app, ["learn", "--target", target])
    assert res_learn.exit_code == 0
    assert "Learning phase completed" in res_learn.output

    # 3. Topics
    res_topics = runner.invoke(app, ["topics", "--target", target])
    assert res_topics.exit_code == 0

    # 4. Build Vault
    res_vault = runner.invoke(app, ["build-vault", "--target", target])
    assert res_vault.exit_code == 0
    assert "Vault Built Successfully" in res_vault.output

    # 5. Graph
    res_graph = runner.invoke(app, ["graph", "--target", target])
    assert res_graph.exit_code == 0

    # 6. Graph Export
    res_export = runner.invoke(app, ["graph-export", "--target", target])
    assert res_export.exit_code == 0
    assert "Knowledge Graph exported" in res_export.output

    # 7. Evaluate Profile
    res_eval = runner.invoke(app, ["evaluate-profile", "--target", target])
    assert res_eval.exit_code == 0
    assert "Overall Scientific Integrity Rating" in res_eval.output

    # 8. Explain
    res_explain = runner.invoke(app, ["explain", "--target", target, "--claim", "Target rejects centralized governance"])
    assert res_explain.exit_code == 0
    assert "Observed Pattern" in res_explain.output

    # 9. Psych Insights
    res_psych = runner.invoke(app, ["psych-insights", "--target", target])
    assert res_psych.exit_code == 0

    # 10. Ask
    res_ask = runner.invoke(app, ["ask", "--target", target, "--prompt", "What is Target's view on consensus?"])
    assert res_ask.exit_code == 0

    # 11. Simulate
    res_sim = runner.invoke(app, ["simulate", "--target", target, "--scenario", "An enterprise asks Target to endorse proprietary closed systems"])
    assert res_sim.exit_code == 0

    # 12. Compose
    res_compose = runner.invoke(app, ["compose", "--target", target, "--goal", "Invite Target to keynote on open consensus"])
    assert res_compose.exit_code == 0
