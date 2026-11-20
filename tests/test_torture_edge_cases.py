"""
Extreme Torture & Numerical Stability Test Suite for who??.
Stress-tests pathological corpora, numerical overflow/underflow in Boltzmann-Gibbs & JSD,
Fisher-z singularities, degenerate graph topologies in Ising/Spectral physics,
and zero-corpus resilience across all 10 CognitiveTwin SDK methods, SwarmArena, and MCP.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from who.agent.sdk import CognitiveTwin
from who.config import settings
from who.engine.arena import SwarmArena
from who.engine.evaluator import compute_jensen_shannon_divergence
from who.engine.simulator import compute_boltzmann_gibbs_distribution
from who.learn.feature_extractor import FeatureExtractor
from who.learn.scientific_referee import ScientificReferee, fisher_z, inv_fisher_z
from who.mcp_server import MCPServer
from who.vault.graph_analytics import GraphAnalytics
from who.vault.graph_store import GraphStore


def test_torture_feature_extractor_pathological_corpora():
    fe = FeatureExtractor()
    for corpus in [
        [],
        [""],
        ["   \n\t  "],
        ["!!! ??? ... @@@ ### 🚀🔥"],
        ["değil yok asla not never"],
        ["risk tehlike korku değil yok asla kazanç fırsat başarı " * 1000],
    ]:
        res = fe.extract_features(corpus)
        assert "loss_aversion_index" in res
        assert "caps_contextual_matrix" in res
        assert "stated_vs_revealed_preference" in res


def test_torture_boltzmann_and_jsd_numerical_extremes():
    for beta in [-100.0, 0.0, 1e-12, 1.0, 1e6]:
        bg = compute_boltzmann_gibbs_distribution(
            {"A": 1e5, "B": -1e5, "C": 0.0, "D": 1e-9},
            beta=beta,
        )
        assert sum(bg["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    assert compute_jensen_shannon_divergence([], []) == 0.0
    assert compute_jensen_shannon_divergence([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0
    assert compute_jensen_shannon_divergence([-5.0, 10.0, 0.0], [0.0, 0.0, 1.0]) >= 0.0


def test_torture_fisher_z_and_referee_singularities(tmp_path: Path):
    for r_val in [-999.0, -1.0, -0.999999, 0.0, 0.999999, 1.0, 999.0]:
        z = fisher_z(r_val)
        r_back = inv_fisher_z(z)
        assert -1.0 <= r_back <= 1.0

    ref = ScientificReferee(profiles_dir=tmp_path)
    weird_papers = [
        {"title": "", "abstract": "", "sample_size": -50, "r": 5.0, "p_value": -1.0},
        {"title": "Zero N", "abstract": "N=0, r=-1.0, p=99", "sample_size": 0, "r": -1.0, "p_value": 99.0},
    ]
    ev = ref.evaluate_and_rank_corpus(weird_papers, marker="torture_marker")
    assert "fisher_meta_analysis" in ev


def test_torture_graph_analytics_degenerate_topologies(tmp_path: Path):
    cases = [
        ("empty", [], []),
        ("singleton", [{"name": "Lonely", "type": "Concept"}], []),
        (
            "self_loop",
            [{"name": "Loop", "type": "Concept"}],
            [{"source": "Loop", "target": "Loop", "relation": "CONTRADICTS", "weight": 0.0}],
        ),
        (
            "disconnected_frustrated",
            [
                {"name": "A", "type": "C"},
                {"name": "B", "type": "C"},
                {"name": "C", "type": "C"},
                {"name": "X", "type": "C"},
                {"name": "Y", "type": "C"},
            ],
            [
                {"source": "A", "target": "B", "relation": "CONTRADICTS", "weight": 1.0},
                {"source": "B", "target": "C", "relation": "CONTRADICTS", "weight": 1.0},
                {"source": "A", "target": "C", "relation": "CONTRADICTS", "weight": 1.0},
                {"source": "X", "target": "Y", "relation": "SUPPORTS", "weight": 0.0},
            ],
        ),
    ]
    for case_name, ents, rels in cases:
        gs = GraphStore(f"graph_{case_name}", vault_dir=tmp_path / case_name)
        gs.store_graph(ents, rels)
        ga = GraphAnalytics(gs, vault_dir=tmp_path / case_name)
        m = ga.run_analytics()
        assert "network_physics" in m
        gs.close()


def test_torture_zero_corpus_ghost_target_all_engines(tmp_path: Path):
    ghost = CognitiveTwin("Ghost Target Zero")
    assert isinstance(ghost.get_profile(), dict)
    assert isinstance(ghost.chat("Hello from empty vault!"), str)
    assert isinstance(ghost.simulate_scenario("Sudden crisis with zero corpus"), dict)
    assert isinstance(ghost.simulate_monte_carlo("Sudden crisis with zero corpus", iterations=5), dict)
    assert isinstance(ghost.backtest(), dict)
    assert isinstance(ghost.verify_response("Test response on empty corpus"), dict)
    assert isinstance(ghost.analyze_drift(), dict)
    assert isinstance(ghost.explain("Absolute claim without corpus"), dict)
    assert isinstance(ghost.ask("Question on empty corpus?"), dict)
    assert isinstance(ghost.compose_outreach("Outreach on empty corpus"), dict)

    arena = SwarmArena(
        ["Ghost Target Zero", "Ghost Target Two"],
        topic="Zero corpus debate",
        rounds=1,
        output_dir=tmp_path / "arena",
    )
    ar_res = arena.run_debate()
    assert "coupled_ising_physics" in ar_res

    mcp = MCPServer()
    err_resp = mcp.handle_request(
        {"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {"name": "nonexistent_tool", "arguments": {}}}
    )
    assert err_resp is not None and "error" in err_resp
