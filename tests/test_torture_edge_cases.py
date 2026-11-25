"""
Extreme Torture & Numerical Stability Test Suite for who??.
Stress-tests pathological corpora, numerical overflow/underflow in Boltzmann-Gibbs & JSD,
Fisher-z singularities, degenerate graph topologies in Ising/Spectral physics,
and zero-corpus resilience across all 10 CognitiveTwin SDK methods, SwarmArena, and MCP.
"""

from __future__ import annotations

import math
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


def test_music_contract_master_rights_scenario_sensitivity():
    """
    Regression test for scenario-induced external field h_i(x) and conditional commitment red-line guard:
    Two music contracts in the same CAPS bucket ('master rights stay with artist' vs 'transferred to sponsor')
    must produce distinct action probability distributions, with higher ACCEPT when rights stay with the artist
    and unified branch_probabilities == boltzmann_branch_probabilities even when covariates are fixed.
    """
    twin = CognitiveTwin("Ghost Target Zero")
    fixed_ranges = {
        "counterparty_leverage": (0.45, 0.45),
        "deadline_pressure_days": (7.0, 7.0),
        "information_asymmetry": (0.25, 0.25),
        "reputation_risk": (0.30, 0.30),
    }
    offer_keep = (
        "Müzik sözleşmesinde master kayıt hakları sanatçıda kalır, "
        "şeffaf denetim hakkı ve adil telif paylaşımı garanti edilir."
    )
    offer_transfer = (
        "Müzik sözleşmesinde master kayıt hakları süresiz olarak sponsora devredilir, "
        "sanatçı denetim ve veto hakkından feragat eder."
    )

    res_keep = twin.simulator.simulate_monte_carlo(
        offer_keep, iterations=15, parameter_ranges=fixed_ranges, random_seed=42
    )
    res_transfer = twin.simulator.simulate_monte_carlo(
        offer_transfer, iterations=15, parameter_ranges=fixed_ranges, random_seed=42
    )

    # 1. Unified Boltzmann QRE distribution (no 100% argmax vs 59% Boltzmann discrepancy)
    assert res_keep["branch_probabilities"] == res_keep["boltzmann_branch_probabilities"]
    assert res_transfer["branch_probabilities"] == res_transfer["boltzmann_branch_probabilities"]

    # 2. Opposite contract clauses must yield distinct probability distributions
    p_accept_keep = res_keep["branch_probabilities"]["SUBGAME_EQUILIBRIUM_ACCEPT"]
    p_accept_transfer = res_transfer["branch_probabilities"]["SUBGAME_EQUILIBRIUM_ACCEPT"]
    p_deadlock_keep = res_keep["branch_probabilities"]["DEADLOCK_WALKAWAY"]
    p_deadlock_transfer = res_transfer["branch_probabilities"]["DEADLOCK_WALKAWAY"]

    assert p_accept_keep > p_accept_transfer + 0.10
    assert p_deadlock_transfer > p_deadlock_keep + 0.05
    assert "competing_mechanisms_analysis" in res_keep
    assert "conformal_prediction_set_90" in res_keep
    assert len(res_keep["conformal_prediction_set_90"]["prediction_set"]) >= 1


def test_helmholtz_free_energy_sign_and_action_conditioned_log_odds():
    """
    Verify physical Helmholtz Free Energy F = -T * ln(Z) = -0.6931 when U = [0, 0], beta = 1,
    and verify action-conditioned free energy difference Delta_F satisfies log(P(a)/P(b)) = -beta * Delta_F.
    """
    bg = compute_boltzmann_gibbs_distribution({"A": 0.0, "B": 0.0}, beta=1.0)
    assert bg["helmholtz_free_energy_F"] == pytest.approx(-0.6931, abs=1e-3)
    assert bg["smoothed_max_utility_logZ"] == pytest.approx(0.6931, abs=1e-3)

    twin = CognitiveTwin("Ghost Target Zero")
    mc = twin.simulator.simulate_monte_carlo("Strategic partnership with transparent audit right", iterations=5)
    sm = mc["statistical_mechanics"]
    assert "frustration_covariance_Sigma" in sm
    assert "action_conditioned_free_energies_F_a" in sm
    assert "concept_field_log_odds_sensitivity" in sm


def test_zero_evidence_symmetric_prior_and_empty_holdout_null_brier():
    """
    1. Neutral text with 0 stated and 0 revealed samples must yield preference_divergence_index == 0.0 (not 0.20).
    2. Empty profile holdout backtest (0 holdout segments) must return brier_score is None and is_prior_fallback is True.
    """
    fe = FeatureExtractor()
    neutral_feats = fe.extract_features(["Bugün hava oldukça güneşli ve rüzgar hafif esiyor."])
    pref = neutral_feats["stated_vs_revealed_preference"]
    assert pref["stated_sample_count"] == 0
    assert pref["revealed_sample_count"] == 0
    assert pref["stated_cooperation_polarity"] == 0.50
    assert pref["revealed_cooperation_polarity"] == 0.50
    assert pref["preference_divergence_index"] == 0.0

    ghost = CognitiveTwin("Ghost Target Empty HoldoutCheck")
    bt = ghost.backtest()
    assert bt["holdout_segments"] == 0
    assert bt["is_prior_fallback"] is True
    assert bt["brier_score"] is None
    assert bt["brier_skill_score_bss"] is None
    assert bt["holdout_accuracy_pct"] is None
    assert bt["prior_fallback_confidence"] == pytest.approx(0.68, abs=0.05)


def test_murphy_decomposition_conformal_self_liang_and_action_evaluator():
    """
    Verify:
    1. Exact Within-Bin Corrected Murphy (1973) Brier Decomposition identity: REL - RES + UNC + WBC == BS.
    2. Split Conformal Prediction Set finite-sample quantile.
    3. Synthetic Parameter Recovery relative_rmse < 0.25, 95% Fisher CI coverage >= 0.80,
       and Joint Gibbs kappa Self-Liang (1987) boundary LRT beating w-permutation null.
    4. Action-labeled decision evaluator across the 4 canonical response branches.
    """
    from who.engine.evaluator import (
        ProfileEvaluator,
        compute_murphy_brier_decomposition,
        compute_split_conformal_prediction_set,
    )

    preds = [0.12, 0.18, 0.35, 0.62, 0.78, 0.85, 0.91, 0.22, 0.48, 0.73]
    outcomes = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    murphy = compute_murphy_brier_decomposition(preds, outcomes, n_bins=4)
    assert murphy["decomposition_identity_error"] < 1e-7
    assert murphy["reconstructed_brier_score"] == pytest.approx(murphy["brier_score"], abs=1e-6)

    conf = compute_split_conformal_prediction_set(
        {"SUBGAME_EQUILIBRIUM_ACCEPT": 0.55, "REJECT_AND_COUNTER": 0.30, "STRATEGIC_CONCESSION": 0.10, "DEADLOCK_WALKAWAY": 0.05},
        calibration_nonconformity_scores=[0.15, 0.25, 0.35, 0.45, 0.50, 0.68, 0.72, 0.40],
        alpha=0.10,
    )
    assert "SUBGAME_EQUILIBRIUM_ACCEPT" in conf["prediction_set"]
    assert conf["set_cardinality"] >= 1

    rec = ProfileEvaluator.run_synthetic_parameter_recovery(n_samples=200, random_seed=42)
    assert rec["is_identifiable"] is True
    assert rec["relative_rmse"] < 0.25
    assert rec["fisher_ci_95_empirical_coverage_rate"] >= 0.80
    assert rec["fisher_ci_95_scope"] == "theta_only_excludes_boundary_kappa"
    jk = rec["joint_gibbs_kappa_lrt"]
    assert jk["beats_temperature_only_baseline_MT"] is True
    assert jk["beats_w_permutation_null"] is True
    assert jk["w_permutation_p_value"] < 0.05
    assert jk["self_liang_p_value"] < 0.01
    assert "theta_kappa_fim_correlations" in jk
    assert 0.20 <= jk["self_liang_null_size_check"]["zero_lambda_fraction"] <= 0.80
    assert jk["self_liang_null_size_check"]["empirical_type1_error_at_005"] <= 0.15

    action_labeled_decisions = [
        {"leverage": 0.20, "asymmetry": 0.10, "reputation_risk": 0.15, "clause_polarity": 0.85, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.75, "asymmetry": 0.55, "reputation_risk": 0.65, "clause_polarity": -0.70, "red_line_violation": 0.85, "observed_action": "Walk-Away"},
        {"leverage": 0.35, "asymmetry": 0.20, "reputation_risk": 0.25, "clause_polarity": -0.25, "red_line_violation": 0.30, "observed_action": "Counter-Offer"},
        {"leverage": 0.65, "asymmetry": 0.15, "reputation_risk": 0.20, "clause_polarity": 0.35, "red_line_violation": 0.0, "observed_action": "Concession"},
        {"leverage": 0.18, "asymmetry": 0.12, "reputation_risk": 0.10, "clause_polarity": 0.90, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.80, "asymmetry": 0.60, "reputation_risk": 0.75, "clause_polarity": -0.80, "red_line_violation": 0.90, "observed_action": "Walk-Away"},
        {"leverage": 0.30, "asymmetry": 0.25, "reputation_risk": 0.20, "clause_polarity": -0.20, "red_line_violation": 0.35, "observed_action": "Counter-Offer"},
        {"leverage": 0.22, "asymmetry": 0.08, "reputation_risk": 0.12, "clause_polarity": 0.88, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.82, "asymmetry": 0.65, "reputation_risk": 0.70, "clause_polarity": -0.85, "red_line_violation": 0.92, "observed_action": "Walk-Away"},
        {"leverage": 0.28, "asymmetry": 0.18, "reputation_risk": 0.22, "clause_polarity": -0.30, "red_line_violation": 0.40, "observed_action": "Counter-Offer"},
        {"leverage": 0.15, "asymmetry": 0.10, "reputation_risk": 0.10, "clause_polarity": 0.92, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.85, "asymmetry": 0.70, "reputation_risk": 0.80, "clause_polarity": -0.90, "red_line_violation": 0.95, "observed_action": "Walk-Away"},
    ]
    act_eval = ProfileEvaluator.evaluate_action_labeled_decisions(action_labeled_decisions, holdout_ratio=0.33, alpha=0.10)
    assert act_eval["status"] == "COMPLETED"
    assert act_eval["brier_skill_score_bss"] > 0.0
    assert act_eval["conformal_empirical_coverage"] >= 0.75
    assert 1.0 <= act_eval["mean_conformal_set_size"] <= 4.0
    assert "per_class_conformal_coverage" in act_eval
    for expected_key in [
        "B0_TRAINING_BASE_RATE",
        "B1_COVARIATES_ONLY_LOGIT",
        "M_T_SCALAR_TEMPERATURE",
        "E1_CAPS_CLAUSE_LOGIT",
        "E2a_JOINT_GIBBS_KAPPA_ONLY",
        "E2b_REDLINE_GUARD_ONLY",
        "E2_FULL_GIBBS_AND_REDLINE",
    ]:
        assert expected_key in act_eval["model_ladder"]
    assert act_eval["model_ladder"]["E2_JOINT_GIBBS_AND_REDLINE"]["brier_score"] <= act_eval["model_ladder"]["B0_TRAINING_BASE_RATE"]["brier_score"]


def test_boltzmann_beta_zero_exact_uniform_and_referee_imputed_n_guard(tmp_path: Path):
    """
    1. beta=0.0 (infinite cognitive temperature T -> inf) must yield exact uniform 1/K distribution,
       C=0.0, S=ln(K), and helmholtz_free_energy_F=None instead of silently clamping beta to 0.1.
    2. ScientificReferee must NOT fabricate Fisher information (120 / 1200) for papers lacking observed N >= 10.
    """
    bg0 = compute_boltzmann_gibbs_distribution({"A": 5.0, "B": -2.0, "C": 1.0, "D": 0.0}, beta=0.0)
    for k in ["A", "B", "C", "D"]:
        assert bg0["probabilities"][k] == pytest.approx(0.25, abs=1e-4)
    assert bg0["specific_heat_capacity_C"] == 0.0
    assert bg0["shannon_entropy_S"] == pytest.approx(math.log(4), abs=1e-3)
    assert bg0["helmholtz_free_energy_F"] is None
    assert bg0["inverse_temperature_beta"] == 0.0
    assert bg0["cognitive_temperature_T"] is None

    ref = ScientificReferee(profiles_dir=tmp_path)
    papers = [
        {"title": "Observed Sample Study", "abstract": "N = 400, r = 0.40, p < 0.001", "sample_size": 400, "r": 0.40, "p_value": 0.0005, "venue": "nature"},
        {"title": "Missing Sample Study", "abstract": "Qualitative discussion r = 0.35 without reported sample size.", "sample_size": None, "r": 0.35, "p_value": 0.01, "venue": "journal"},
    ]
    res = ref.evaluate_and_rank_corpus(papers, marker="test_marker", prior_mean_r=0.35)
    fma = res["fisher_meta_analysis"]
    assert fma["total_fisher_information"] == pytest.approx(397.0, abs=1e-4)
    assert fma["observed_n_studies"] == 1
    assert fma["imputed_n_studies_excluded_from_fisher"] == 1


def test_joint_gibbs_exact_centering_and_reference_action_invariance():
    """
    Verify:
    1. E_{P(s|x)}[g_hat_a(s)] == 0 to numerical precision (< 1e-9).
    2. Marginalized action probabilities P(a|x) are strictly invariant to changing reference_action
       between DEADLOCK_WALKAWAY and SUBGAME_EQUILIBRIUM_ACCEPT.
    """
    twin = CognitiveTwin("Ghost Target Zero")
    sim = twin.simulator
    field_analysis = sim._evaluate_scenario_external_fields_and_commitments(
        scenario="Contract proposal with transparent audit and balanced governance parity.",
        features_data={},
        graph_metrics={},
    )
    base_u = {
        "REJECT_AND_COUNTER": 0.45,
        "STRATEGIC_CONCESSION": 0.15,
        "SUBGAME_EQUILIBRIUM_ACCEPT": 0.50,
        "DEADLOCK_WALKAWAY": 0.25,
    }
    res_deadlock_ref = sim._compute_centered_joint_gibbs_coupling(
        base_utilities=base_u,
        beta=2.8,
        kappa=0.80,
        field_analysis=field_analysis,
        belief_temperature_Tb=0.65,
        reference_action="DEADLOCK_WALKAWAY",
    )
    res_accept_ref = sim._compute_centered_joint_gibbs_coupling(
        base_utilities=base_u,
        beta=2.8,
        kappa=0.80,
        field_analysis=field_analysis,
        belief_temperature_Tb=0.65,
        reference_action="SUBGAME_EQUILIBRIUM_ACCEPT",
    )

    assert res_deadlock_ref["centered_g_expectation_max_abs"] < 1e-9
    assert res_accept_ref["centered_g_expectation_max_abs"] < 1e-9
    for act in base_u:
        assert res_deadlock_ref["probabilities"][act] == pytest.approx(res_accept_ref["probabilities"][act], abs=1e-4)


def test_scenario_local_negation_and_zero_iterations_guard():
    """
    1. Negated clause 'master kayıt hakları devredilir; denetim hakkı yok' must NOT count
       'denetim hakkı' as a positive preservation hit that cancels the transfer violation.
    2. 'devredilmez' must count once in preserve_clause_hits and zero in violation_clause_hits.
    3. simulate_monte_carlo(iterations=0) must raise ValueError cleanly instead of IndexError.
    """
    twin = CognitiveTwin("Ghost Target Zero")
    sim = twin.simulator

    neg_offer = sim._evaluate_scenario_external_fields_and_commitments(
        scenario="Müzik sözleşmesinde master kayıt hakları devredilir; denetim hakkı yok.",
        features_data={},
        graph_metrics={},
    )
    assert neg_offer["preserve_clause_hits"] == 0
    assert neg_offer["violation_clause_hits"] >= 2
    assert neg_offer["opacity_clause_hits"] >= 1
    assert neg_offer["clause_polarity"] < 0.0
    assert neg_offer["red_line_violation_score"] > 0.5

    no_transfer = sim._evaluate_scenario_external_fields_and_commitments(
        scenario="Haklar devredilmez ve sanatçıda kalır.",
        features_data={},
        graph_metrics={},
    )
    assert no_transfer["preserve_clause_hits"] == 2
    assert no_transfer["violation_clause_hits"] == 0

    with pytest.raises(ValueError, match="iterations"):
        sim.simulate_monte_carlo("Test scenario", iterations=0)


def test_boltzmann_empty_beta_zero_k3_simplex_and_joint_thermodynamics():
    """
    1. Empty utilities with beta=0 returns cognitive_temperature_T=None and inverse_temperature_beta=0.0.
    2. K=3 uniform beta=0 simplex sums to exact 1.0 (not 0.9999).
    3. Large U={A: 100.0, B: 0.0} at beta=1 keeps log_partition_function_logZ unclipped (~100.0)
       so F = -T * ln(Z) consistency holds.
    4. Joint Gibbs F and C respond dynamically to kappa > 0 and report T_b / <|J|>.
    """
    empty_b0 = compute_boltzmann_gibbs_distribution({}, beta=0.0)
    assert empty_b0["inverse_temperature_beta"] == 0.0
    assert empty_b0["cognitive_temperature_T"] is None
    assert empty_b0["helmholtz_free_energy_F"] is None

    k3_b0 = compute_boltzmann_gibbs_distribution({"A": 1.0, "B": 2.0, "C": 3.0}, beta=0.0)
    assert math.isclose(sum(k3_b0["probabilities"].values()), 1.0, abs_tol=1e-9)

    large_u = compute_boltzmann_gibbs_distribution({"A": 100.0, "B": 0.0}, beta=1.0)
    assert large_u["log_partition_function_logZ"] == pytest.approx(100.0, abs=1e-4)
    assert large_u["helmholtz_free_energy_F"] == pytest.approx(-large_u["log_partition_function_logZ"], abs=1e-3)

    twin = CognitiveTwin("Ghost Target Zero")
    sim = twin.simulator
    field_a = sim._evaluate_scenario_external_fields_and_commitments("Neutral contract parity proposal.", {}, {})
    u_base = {"REJECT_AND_COUNTER": 0.4, "STRATEGIC_CONCESSION": 0.1, "SUBGAME_EQUILIBRIUM_ACCEPT": 0.42, "DEADLOCK_WALKAWAY": 0.2}
    g_k0 = sim._compute_centered_joint_gibbs_coupling(u_base, beta=3.0, kappa=0.0, field_analysis=field_a)
    g_k1 = sim._compute_centered_joint_gibbs_coupling(u_base, beta=3.0, kappa=1.2, field_analysis=field_a)
    assert g_k0["helmholtz_free_energy_F"] != g_k1["helmholtz_free_energy_F"]
    assert g_k0["specific_heat_capacity_C"] != g_k1["specific_heat_capacity_C"]
    assert g_k1["dimensionless_temperature_Tb_over_meanJ"] > 0.0
    assert math.isclose(sum(g_k1["probabilities"].values()), 1.0, abs_tol=1e-9)


def test_split_conformal_finite_sample_saturation_and_murphy_clamping():
    """
    1. When ceil((n_cal + 1) * (1 - alpha)) > n_cal (e.g., n_cal=8, alpha=0.10 -> k=9 > 8),
       finite-sample split conformal sets q_hat = +inf and returns all candidate actions.
    2. When n_cal >= 19, alpha=0.10 produces a finite q_hat.
    3. Out-of-range predictions [1.2, -0.1, ...] are clamped consistently before BOTH binning
       and exact_bs so Murphy identity error stays < 1e-7.
    """
    from who.engine.evaluator import (
        compute_murphy_brier_decomposition,
        compute_split_conformal_prediction_set,
    )

    probs = {"SUBGAME_EQUILIBRIUM_ACCEPT": 0.65, "REJECT_AND_COUNTER": 0.25, "STRATEGIC_CONCESSION": 0.07, "DEADLOCK_WALKAWAY": 0.03}
    c_small = compute_split_conformal_prediction_set(probs, calibration_nonconformity_scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.72], alpha=0.10)
    assert c_small["q_hat_is_infinite"] is True
    assert c_small["q_hat_threshold"] is None
    assert c_small["set_cardinality"] == 4

    c_large = compute_split_conformal_prediction_set(probs, calibration_nonconformity_scores=[0.05 * i for i in range(1, 20)], alpha=0.10)
    assert c_large["q_hat_is_infinite"] is False
    assert c_large["q_hat_threshold"] is not None

    murphy_oog = compute_murphy_brier_decomposition([1.2, -0.1, 0.85, 0.15, 0.70, 0.30], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], n_bins=4)
    assert murphy_oog["clamped_predictions_count"] == 2
    assert murphy_oog["decomposition_identity_error"] < 1e-7


def test_harary_balanced_counterfactual_model_recovery_and_causality_gate():
    """
    Verify:
    1. GraphAnalytics.construct_harary_balanced_counterfactual achieves balanced_signed_lambda_min < 1e-5
       while preserving edge magnitudes and topology.
    2. Unbiased full-sphere w-permutation, KS statistic, and 3x3 BIC confusion matrix in synthetic recovery.
    3. Independent 3-way split, E2_BAL_COUNTERFACTUAL, E2_W_PERMUTED_CONTROL, and per-class Murphy decomposition.
    4. verify_temporal_causality_invariance passes causal features and catches future-peeking leakage.
    """
    from who.engine.evaluator import ProfileEvaluator

    nodes = ["A", "B", "C"]
    frust_couplings = [("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", -1.0)]
    bal = GraphAnalytics.construct_harary_balanced_counterfactual(nodes, frust_couplings)
    assert bal["flipped_bonds_count"] == 1
    assert bal["balanced_signed_lambda_min"] < 1e-5
    assert [abs(j) for _, _, j in bal["balanced_couplings"]] == [1.0, 1.0, 1.0]

    rec = ProfileEvaluator.run_synthetic_parameter_recovery(n_samples=200, random_seed=42)
    jk = rec["joint_gibbs_kappa_lrt"]
    assert jk["w_permutation_is_unbiased_full_sphere"] is True
    assert jk["w_permutation_p_value"] < 0.05
    assert jk["self_liang_null_size_check"]["ks_statistic_vs_self_liang"] < 0.20
    assert jk["model_recovery_confusion_matrix"]["E2_false_win_rate_on_MT_data"] <= 0.15
    assert jk["model_recovery_confusion_matrix"]["overall_recovery_accuracy"] >= 0.75
    assert jk["frustrated_vs_balanced_counterfactual"]["balanced_signed_lambda_min"] < 1e-5
    assert jk["frustrated_vs_balanced_counterfactual"]["frustrated_tr_Sigma"] > jk["frustrated_vs_balanced_counterfactual"]["balanced_tr_Sigma"]

    base_block = [
        {"leverage": 0.18, "asymmetry": 0.10, "reputation_risk": 0.12, "clause_polarity": 0.88, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.82, "asymmetry": 0.62, "reputation_risk": 0.72, "clause_polarity": -0.78, "red_line_violation": 0.88, "observed_action": "Walk-Away"},
        {"leverage": 0.32, "asymmetry": 0.22, "reputation_risk": 0.25, "clause_polarity": -0.25, "red_line_violation": 0.32, "observed_action": "Counter-Offer"},
        {"leverage": 0.65, "asymmetry": 0.14, "reputation_risk": 0.20, "clause_polarity": 0.35, "red_line_violation": 0.0, "observed_action": "Concession"},
    ]
    rows_24 = base_block * 6
    ev24 = ProfileEvaluator.evaluate_action_labeled_decisions(rows_24, holdout_ratio=0.25, alpha=0.20)
    assert ev24["calibration_fold_independent"] is True
    assert "E2_BAL_COUNTERFACTUAL" in ev24["model_ladder"]
    assert "E2_W_PERMUTED_CONTROL" in ev24["model_ladder"]
    assert "per_class_murphy_decomposition" in ev24
    for act, m_res in ev24["per_class_murphy_decomposition"].items():
        assert m_res["decomposition_identity_error"] < 1e-7

    turns = [
        {"text": "Initial anchor offer", "outcome": "Counter", "final_score": 10},
        {"text": "Revised proposal with parity", "outcome": "Accept", "final_score": 25},
        {"text": "Final agreement signed", "outcome": "Accept", "final_score": 30},
    ]
    causal_check = ProfileEvaluator.verify_temporal_causality_invariance(
        turns,
        lambda trns, t_idx: {"prefix_len": t_idx + 1, "last_text": trns[t_idx]["text"]},
    )
    leaky_check = ProfileEvaluator.verify_temporal_causality_invariance(
        turns,
        lambda trns, t_idx: {"future_outcome": trns[-1]["outcome"]},
    )
    assert causal_check["is_causally_invariant"] is True
    assert leaky_check["is_causally_invariant"] is False
    assert len(leaky_check["leakage_violations"]) == 2


def test_quenched_joint_specific_heat_matches_logz_second_derivative():
    """Verify quenched specific heat C_v = beta^2 * E_s[Var_{a|s}(U)] equals beta^2 * d^2(E_s[log Z_s])/d(beta^2)."""
    twin = CognitiveTwin("Ghost Target Zero")
    sim = twin.simulator
    field_a = sim._evaluate_scenario_external_fields_and_commitments(
        "Neutral contract parity proposal.", {}, {}
    )
    u_base = {
        "REJECT_AND_COUNTER": 0.4,
        "STRATEGIC_CONCESSION": 0.1,
        "SUBGAME_EQUILIBRIUM_ACCEPT": 0.42,
        "DEADLOCK_WALKAWAY": 0.2,
    }

    # d=0.05 avoids 1e-6 rounding noise in log_partition_function_logZ while keeping truncation error < 2e-3
    d_beta = 0.05
    for beta in (2.0, 3.0):
        res0 = sim._compute_centered_joint_gibbs_coupling(u_base, beta=beta, kappa=1.2, field_analysis=field_a)
        res_p = sim._compute_centered_joint_gibbs_coupling(u_base, beta=beta + d_beta, kappa=1.2, field_analysis=field_a)
        res_m = sim._compute_centered_joint_gibbs_coupling(u_base, beta=beta - d_beta, kappa=1.2, field_analysis=field_a)
        cv_fd = (beta ** 2) * (
            res_p["log_partition_function_logZ"] - 2.0 * res0["log_partition_function_logZ"] + res_m["log_partition_function_logZ"]
        ) / (d_beta ** 2)
        assert res0["specific_heat_capacity_C"] == pytest.approx(cv_fd, abs=3e-3)
        # Quenched derivative assumes spin ensemble P(s | x) is strictly independent of decision beta
        assert res0["mean_spins_P"] == res_p["mean_spins_P"] == res_m["mean_spins_P"]

    # Constant utility across actions (kappa=0) gives zero within-spin variance -> exact C_v = 0.0
    u_const = {a: 2.5 for a in u_base}
    res_const = sim._compute_centered_joint_gibbs_coupling(u_const, beta=3.0, kappa=0.0, field_analysis=field_a)
    assert res_const["specific_heat_capacity_C"] == 0.0


def test_conformal_alpha_validation_and_evaluator_consistency():
    """Verify conformal alpha is validated once and target coverage matches predictor for any alpha in (0, 1)."""
    from who.engine.evaluator import ProfileEvaluator, compute_split_conformal_prediction_set

    probs = {
        "SUBGAME_EQUILIBRIUM_ACCEPT": 0.65,
        "REJECT_AND_COUNTER": 0.25,
        "STRATEGIC_CONCESSION": 0.07,
        "DEADLOCK_WALKAWAY": 0.03,
    }
    cal_scores = [0.05 * i for i in range(1, 20)]
    c80 = compute_split_conformal_prediction_set(probs, calibration_nonconformity_scores=cal_scores, alpha=0.8)
    assert c80["alpha"] == 0.8
    assert c80["target_coverage"] == 0.2

    base_block = [
        {"leverage": 0.18, "asymmetry": 0.10, "reputation_risk": 0.12, "clause_polarity": 0.88, "red_line_violation": 0.0, "observed_action": "Accept-Deal"},
        {"leverage": 0.82, "asymmetry": 0.62, "reputation_risk": 0.72, "clause_polarity": -0.78, "red_line_violation": 0.88, "observed_action": "Walk-Away"},
        {"leverage": 0.32, "asymmetry": 0.22, "reputation_risk": 0.25, "clause_polarity": -0.25, "red_line_violation": 0.32, "observed_action": "Counter-Offer"},
        {"leverage": 0.65, "asymmetry": 0.14, "reputation_risk": 0.20, "clause_polarity": 0.35, "red_line_violation": 0.0, "observed_action": "Concession"},
    ]
    ev80 = ProfileEvaluator.evaluate_action_labeled_decisions(base_block * 6, holdout_ratio=0.25, alpha=0.8)
    assert ev80["conformal_alpha"] == 0.8
    assert ev80["conformal_target_coverage"] == 0.2

    for bad_alpha in (0.0, -0.05, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            compute_split_conformal_prediction_set({}, alpha=bad_alpha)
        with pytest.raises(ValueError):
            ProfileEvaluator.evaluate_action_labeled_decisions([], alpha=bad_alpha)
        with pytest.raises(ValueError):
            compute_split_conformal_prediction_set(probs, calibration_nonconformity_scores=cal_scores, alpha=bad_alpha)
        with pytest.raises(ValueError):
            ProfileEvaluator.evaluate_action_labeled_decisions(base_block * 6, holdout_ratio=0.25, alpha=bad_alpha)


def test_short_clause_local_negation_does_not_bleed_across_phrases():
    """Verify negation attaches directionally to local phrases without tainting sibling preservation clauses."""
    sim = CognitiveTwin("Ghost Target Zero").simulator

    s1 = sim._evaluate_scenario_external_fields_and_commitments(
        "Master stays with artist without audit right", {}, {}
    )
    assert s1["preserve_clause_hits"] >= 1
    assert s1["violation_clause_hits"] >= 1
    assert s1["opacity_clause_hits"] >= 1

    s2 = sim._evaluate_scenario_external_fields_and_commitments(
        "Gecikme yok master sanatçıda kalır", {}, {}
    )
    assert s2["preserve_clause_hits"] >= 1
    assert s2["violation_clause_hits"] == 0

    s3 = sim._evaluate_scenario_external_fields_and_commitments(
        "Master sanatçıda kalır denetim hakkı yok", {}, {}
    )
    assert s3["preserve_clause_hits"] >= 1
    assert s3["violation_clause_hits"] >= 1
    assert s3["opacity_clause_hits"] >= 1

    for negated_audit in ("Denetim hakkı kesinlikle yoktur", "Audit right will be denied"):
        sn = sim._evaluate_scenario_external_fields_and_commitments(negated_audit, {}, {})
        assert sn["violation_clause_hits"] >= 1
        assert sn["opacity_clause_hits"] >= 1

    s_double = sim._evaluate_scenario_external_fields_and_commitments(
        "Audit rights are not denied", {}, {}
    )
    assert s_double["preserve_clause_hits"] >= 1
    assert s_double["violation_clause_hits"] == 0
    assert s_double["opacity_clause_hits"] == 0

    s_not_only = sim._evaluate_scenario_external_fields_and_commitments(
        "Not only does the artist retain the master rights", {}, {}
    )
    assert s_not_only["preserve_clause_hits"] >= 1
    assert s_not_only["violation_clause_hits"] == 0

    for negated_tr_verb in ("Master sanatçıda kalmaz", "Master hakları korunmaz"):
        sv = sim._evaluate_scenario_external_fields_and_commitments(negated_tr_verb, {}, {})
        assert sv["violation_clause_hits"] >= 1
        assert sv["preserve_clause_hits"] == 0
        assert sv["clause_polarity"] < 0.0

    s_no_transfer = sim._evaluate_scenario_external_fields_and_commitments(
        "Haklar devredilmeyecektir ve sanatçıda kalır", {}, {}
    )
    assert s_no_transfer["preserve_clause_hits"] == 2
    assert s_no_transfer["violation_clause_hits"] == 0

    s_clearly_denied = sim._evaluate_scenario_external_fields_and_commitments(
        "Audit rights are clearly denied", {}, {}
    )
    assert s_clearly_denied["violation_clause_hits"] >= 1
    assert s_clearly_denied["opacity_clause_hits"] >= 1
    assert s_clearly_denied["preserve_clause_hits"] == 0

    s_not_clearly = sim._evaluate_scenario_external_fields_and_commitments(
        "Audit rights are not clearly denied", {}, {}
    )
    assert s_not_clearly["preserve_clause_hits"] >= 1
    assert s_not_clearly["violation_clause_hits"] == 0
    assert s_not_clearly["opacity_clause_hits"] == 0

    s_sponsor_no_rx = sim._evaluate_scenario_external_fields_and_commitments(
        "The sponsor shall not receive the master rights", {}, {}
    )
    assert s_sponsor_no_rx["preserve_clause_hits"] >= 1
    assert s_sponsor_no_rx["violation_clause_hits"] == 0
    assert s_sponsor_no_rx["clause_polarity"] > 0.0

    s_artist_no_audit = sim._evaluate_scenario_external_fields_and_commitments(
        "The artist shall not receive the audit right", {}, {}
    )
    assert s_artist_no_audit["violation_clause_hits"] >= 1
    assert s_artist_no_audit["opacity_clause_hits"] >= 1
    assert s_artist_no_audit["preserve_clause_hits"] == 0



