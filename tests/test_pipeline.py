"""Tests for configuration, Monte Carlo, genetic algorithm and ML features."""
import numpy as np
import pandas as pd
import pytest

from core.beam import Beam, BeamCrossSection
from core.material import ALUMINUM_2024_T3
from core.hydraulic_configuration import (create_standard_configuration,
                                          create_dangerous_configuration)


def make_beam(tip=50_000, dist=3_000):
    beam = Beam(4.0, BeamCrossSection(width=0.15, height=0.20),
                ALUMINUM_2024_T3, n_stations=100)
    beam.apply_tip_load(tip)
    beam.apply_distributed_load(dist)
    return beam


# ── hydraulic configuration ─────────────────────────────────────────────
def test_standard_configuration_is_valid():
    a = create_standard_configuration(make_beam()).analyze(make_beam())
    assert a['no_overlap'] and a['separation_ok']
    assert min(r['sf_overall'] for r in a['line_results']) > 1.5


def test_dangerous_configuration_is_flagged():
    beam = make_beam()
    a = create_dangerous_configuration(beam).analyze(beam)
    assert not a['no_overlap']


# ── Monte Carlo ─────────────────────────────────────────────────────────
def test_monte_carlo_is_reproducible():
    from simulation.monte_carlo import MonteCarloConfig, MonteCarloEngine
    beam = make_beam()
    runs = []
    for _ in range(2):
        cfg = MonteCarloConfig(n_iterations=15, seed=7, save_csv=False, verbose=False)
        res = MonteCarloEngine(cfg, create_standard_configuration(beam)).run()
        runs.append(res.to_dataframe() if hasattr(res, 'to_dataframe') else res.records)
    a, b = runs
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    else:
        assert len(a) == len(b)


# ── genetic algorithm ───────────────────────────────────────────────────
def _run_ga(seed, generations=15):
    from optimization.genetic_optimizer import GAConfig, FitnessEvaluator, GeneticOptimizer
    cfg = GAConfig(population_size=20, n_generations=generations, n_lines=3,
                   elitism_count=2, seed=seed)
    return GeneticOptimizer(cfg, FitnessEvaluator(make_beam())).run(verbose=False)


def test_ga_is_reproducible_and_genes_in_bounds():
    r1, r2 = _run_ga(3), _run_ga(3)
    assert r1.best.fitness == r2.best.fitness
    assert np.all(np.abs(r1.best.genes) <= 1.0 + 1e-9)


def test_ga_elitism_never_loses_best():
    r = _run_ga(5)
    best = r.history_df['best_fitness'] if 'best_fitness' in r.history_df else \
        r.history_df.iloc[:, 1]
    assert np.all(np.diff(best.values) >= -1e-9)


def test_score_weights_sum_to_one():
    from core.hydraulic_line import SCORE_WEIGHTS
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


# ── ML features ─────────────────────────────────────────────────────────
def test_features_exclude_outcome_columns():
    from ml.features import FEATURES, OUTCOME_COLUMNS, assert_no_leakage
    assert_no_leakage(FEATURES)
    assert not set(FEATURES) & set(OUTCOME_COLUMNS)


def test_leakage_guard_raises():
    from ml.features import assert_no_leakage, OUTCOME_COLUMNS
    with pytest.raises(Exception):
        assert_no_leakage(['peak_load_kN', sorted(OUTCOME_COLUMNS)[0]])


def test_build_feature_row_matches_model_features():
    from ml.features import FEATURES, build_feature_row
    row = build_feature_row(60.0, 3.0, 0.2, [0.0, 0.05, -0.05], 20.7, 1.0)
    assert list(row.columns) == list(FEATURES)
    assert row.shape == (1, len(FEATURES))
    assert not row.isna().any().any()
