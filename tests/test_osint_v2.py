"""
Unit tests for who?? OSINT Enhancements:
1. Social Media & Username Intelligence (src.harvest.social_osint)
2. Cyber Footprint & Breach Auditing (src.harvest.cyber_footprint)
3. Multi-Target Social Network & Entity Resolution (src.vault.social_graph)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.harvest.cyber_footprint import CyberFootprintAuditor
from src.harvest.social_osint import SocialOSINTHunter
from src.vault.social_graph import SocialGraphAnalyzer

runner = CliRunner()


def test_social_osint_hunter(tmp_path: Path):
    target = "Test Target"
    dirs = settings.get_target_dirs(target)
    hunter = SocialOSINTHunter(target, dirs)

    # Test single probe with a mock spec
    spec = {
        "url": "https://api.github.com/users/{username}",
        "check_type": "json_api",
        "category": "Code & Development",
        "icon": "🐙",
    }

    # Probing a known public username on GitHub
    res = hunter.probe_single_platform("GitHub", spec, "octocat")
    assert res is not None
    assert res["exists"] is True
    assert "github.com" in res["url"]

    # Test dossier ingestion
    hunter._ingest_discovered_dossier("octocat", [res])
    dossier_file = dirs.raw_data_dir / "social_osint_dossier_octocat.md"
    assert dossier_file.exists()
    content = dossier_file.read_text(encoding="utf-8")
    assert "Social Footprint OSINT Dossier" in content
    assert "GitHub" in content


def test_cyber_footprint_auditor(tmp_path: Path):
    target = "Test Target"
    dirs = settings.get_target_dirs(target)
    auditor = CyberFootprintAuditor(target, dirs)

    test_email = "investigation_test@example.com"
    report = auditor.run_full_audit(test_email)

    assert report["email"] == test_email
    assert report["validation"]["is_valid_format"] is True
    assert report["validation"]["domain"] == "example.com"
    assert "risk_tier" in report["breach_exposure"]
    assert "digital_hygiene_score_pct" in report["breach_exposure"]

    # Verify report persistence
    json_reports = list(dirs.vault_dir.glob("cyber_footprint_*.json"))
    assert len(json_reports) >= 1
    md_report = dirs.vault_dir / "cyber_footprint_report.md"
    assert md_report.exists()
    assert "Cyber Footprint & Breach Intelligence Report" in md_report.read_text(encoding="utf-8")


def test_social_graph_analyzer(tmp_path: Path):
    targets = ["Remzi Sanver", "Test Target"]
    analyzer = SocialGraphAnalyzer(targets, output_dir=tmp_path)
    network = analyzer.build_multi_target_network()

    assert network["total_nodes"] >= 2
    assert "nodes" in network
    assert "edges" in network
    assert "shared_bridges" in network
    assert "broker_nodes_ranking" in network

    # Verify HTML visualizer generation
    html_files = list(tmp_path.glob("social_network_*.html"))
    assert len(html_files) >= 1
    html_content = html_files[0].read_text(encoding="utf-8")
    assert "d3.v7.min.js" in html_content
    assert "Social Network Intelligence Graph" in html_content


def test_cli_hunt_cyber_network_commands():
    # 1. Test CLI hunt
    res_hunt = runner.invoke(app, ["hunt", "--target", "Test Target", "--username", "torvalds"])
    assert res_hunt.exit_code == 0
    assert "Username Hunt" in res_hunt.output

    # 2. Test CLI cyber
    res_cyber = runner.invoke(app, ["cyber", "--target", "Test Target", "--email", "analyst@bilgi.edu.tr"])
    assert res_cyber.exit_code == 0
    assert "Cyber Reconnaissance Audit" in res_cyber.output

    # 3. Test CLI network
    res_net = runner.invoke(app, ["network", "--targets", "Remzi Sanver, Test Target"])
    assert res_net.exit_code == 0
    assert "Social Network Link Analysis" in res_net.output
    assert "Interactive D3.js Social Graph" in res_net.output


def test_social_graph_no_html_flag(tmp_path: Path):
    targets = ["Remzi Sanver", "Test Target"]
    no_html_dir = tmp_path / "no_html_out"
    analyzer = SocialGraphAnalyzer(targets, output_dir=no_html_dir)
    analyzer.build_multi_target_network(export_html=False)

    assert len(list(no_html_dir.glob("social_network_*.json"))) == 1
    assert len(list(no_html_dir.glob("social_network_*.html"))) == 0

    res_cli = runner.invoke(app, ["network", "--targets", "Remzi Sanver, Test Target", "--no-html"])
    assert res_cli.exit_code == 0
    assert "HTML skipped" in res_cli.output
