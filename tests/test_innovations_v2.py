"""
Unit tests for who?? Innovations:
1. Live OSINT Monitor & Cognitive Anomaly Detector (src.harvest.live_monitor)
2. Multi-Agent Swarm Arena Debate Simulation (src.engine.arena)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.engine.arena import SwarmArena
from src.harvest.live_monitor import LiveMonitor

runner = CliRunner()


def test_live_monitor_anomaly_detection(tmp_path: Path):
    target = "Test Target"
    dirs = settings.get_target_dirs(target)

    monitor = LiveMonitor(target, dirs)

    # Simulated candidate item with nocturnal timestamp and dogmatic text
    feed_items = [
        {
            "title": "Midnight Manifesto",
            "content": "Kesinlikle herkes bilmelidir ki, başka hiçbir alternatif asla kabul edilemez. Durum şüphesiz ortadadır.",
            "url": "https://example.org/post1",
            "timestamp": "2026-10-30T02:30:00Z",
            "source_type": "simulated_rss",
        },
        {
            "title": "Afternoon Ultimatum",
            "content": "Kesinlikle hiçbir şüphe yoktur ki bu karar asla tartışılamaz ve mutlaka uygulanacaktır.",
            "url": "https://example.org/post2",
            "timestamp": "Fri, 30 Oct 2026 14:30:00 GMT",
            "source_type": "simulated_rss_rfc2822",
        }
    ]

    anomalies = monitor.detect_anomalies(feed_items, ignore_history=True)
    assert len(anomalies) >= 1

    first_anom = anomalies[0]
    assert first_anom["item_title"] == "Midnight Manifesto"
    trigger_types = [t["type"] for t in first_anom["triggers"]]

    # Expect at least circadian nocturnal or dogmatism trigger
    assert "CIRCADIAN_NOCTURNAL_SPIKE" in trigger_types
    assert "DOGMATISM_SURGE" in trigger_types

    second_anom = anomalies[1]
    assert second_anom["item_title"] == "Afternoon Ultimatum"
    second_triggers = [t["type"] for t in second_anom["triggers"]]
    assert "DOGMATISM_SURGE" in second_triggers
    assert "CIRCADIAN_NOCTURNAL_SPIKE" not in second_triggers

    # Verify alerts file was saved
    assert monitor.alerts_file.exists()


def test_swarm_arena_execution(tmp_path: Path):
    targets = ["Remzi Sanver", "Test Target"]
    topic = "Akademik otonomi ile merkezi kurumsal yönetim arasındaki denge"

    arena = SwarmArena(participant_names=targets, topic=topic, rounds=1, output_dir=tmp_path)
    participants = arena.load_participants()

    assert "Remzi Sanver" in participants
    assert "Test Target" in participants

    debate_res = arena.run_debate()

    assert debate_res["topic"] == topic
    assert len(debate_res["transcript"]) == 2  # 2 participants x 1 round
    assert "synthesis" in debate_res
    assert "consensus_score_pct" in debate_res["synthesis"]
    assert "consensus_divergence_pct" in debate_res["synthesis"]

    # Verify report was generated
    report_files = list(tmp_path.glob("debate_*.md"))
    assert len(report_files) == 1
    content = report_files[0].read_text(encoding="utf-8")
    assert "Multi-Agent Swarm Arena" in content
    assert "Remzi Sanver" in content


def test_cli_monitor_and_arena_commands():
    # 1. Test CLI monitor
    res_monitor = runner.invoke(app, ["monitor", "--target", "Test Target", "--simulate-feed"])
    assert res_monitor.exit_code == 0
    assert "Live OSINT Monitor" in res_monitor.output

    # 2. Test CLI arena
    res_arena = runner.invoke(app, ["arena", "--targets", "Remzi Sanver, Test Target", "--topic", "Oyun teorisi ve konsensüs protokolleri", "--rounds", "1"])
    assert res_arena.exit_code == 0
    assert "Swarm Arena Debate" in res_arena.output
    assert "Konsensüs" in res_arena.output
