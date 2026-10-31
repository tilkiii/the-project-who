"""
Unit tests for system management CLI commands (targets, purge, providers, ui).
"""

from pathlib import Path
from typer.testing import CliRunner

from who.cli import app
from who.config import settings

runner = CliRunner()


def test_cli_targets_empty(tmp_path: Path):
    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "No digital twin profiles found" in result.output or "Profiles directory" in result.output


def test_cli_targets_with_profile(tmp_path: Path):
    # Setup test profile in isolated settings.profiles_dir
    target_dir = Path(settings.profiles_dir) / "test_scholar"
    vault_dir = target_dir / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "character.json").write_text('{"name": "Test Scholar"}', encoding="utf-8")
    (vault_dir / "concept_network.json").write_text('{"nodes": [], "relationships": []}', encoding="utf-8")

    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    assert "Test Scholar" in result.output
    assert "READY" in result.output


def test_cli_purge_target():
    # Setup test profile
    target_dir = Path(settings.profiles_dir) / "disposable_twin"
    target_dir.mkdir(parents=True, exist_ok=True)
    assert target_dir.exists()

    # Purge with --force
    result = runner.invoke(app, ["purge", "--target", "disposable_twin", "--force"])
    assert result.exit_code == 0
    assert "Successfully purged target profile" in result.output
    assert not target_dir.exists()


def test_cli_purge_nonexistent():
    result = runner.invoke(app, ["purge", "--target", "nonexistent_target_slug", "--force"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_ui_help():
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Launch the interactive Web Dashboard" in result.output
