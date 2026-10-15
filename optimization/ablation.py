"""
GA ablation: does the position term in the composite score matter?

Runs the genetic algorithm over seeds 0-9 with
  (A) default weights   sf 0.50 / fatigue 0.30 / position 0.20
  (B) position removed  sf 0.625 / fatigue 0.375 / position 0.0
and scores every winning layout on the SAME physical yardsticks (not the
score it was optimised for): minimum tube SF, mean |y| / (h/2), validity.

Run:  python -m optimization.ablation
"""
import os
import numpy as np
import pandas as pd

from core.beam import Beam, BeamCrossSection
from core.material import ALUMINUM_2024_T3
from core import hydraulic_line
from optimization.genetic_optimizer import GAConfig, FitnessEvaluator, GeneticOptimizer

VARIANTS = {
    'default (position 0.20)': {'sf': 0.50, 'fatigue': 0.30, 'position': 0.20},
    'no position term':        {'sf': 0.625, 'fatigue': 0.375, 'position': 0.0},
}


def _build_beam(tip_n=50_000, dist_n_per_m=3_000):
    beam = Beam(4.0, BeamCrossSection(width=0.15, height=0.20),
                ALUMINUM_2024_T3, n_stations=100)
    beam.apply_tip_load(tip_n)
    beam.apply_distributed_load(dist_n_per_m)
    return beam


def run_ablation(seeds=range(10), tip_n=50_000, dist_n_per_m=3_000,
                 out_csv='data/ablation_ga.csv'):
    beam = _build_beam(tip_n, dist_n_per_m)
    half_h = beam.cs.h / 2
    rows = []
    original = dict(hydraulic_line.SCORE_WEIGHTS)
    try:
        for name, weights in VARIANTS.items():
            hydraulic_line.SCORE_WEIGHTS.update(weights)
            for seed in seeds:
                cfg = GAConfig(population_size=50, n_generations=80, n_lines=3,
                               crossover_rate=0.8, mutation_rate=0.15,
                               mutation_sigma=0.08, tournament_size=3,
                               elitism_count=2, seed=int(seed))
                evaluator = FitnessEvaluator(beam, pressure=20.7e6)
                result = GeneticOptimizer(cfg, evaluator).run(verbose=False)
                config = evaluator.build_configuration(result.best.genes)
                analysis = config.analyze(beam)
                y = np.array([r['y_position'] for r in analysis['line_results']])
                rows.append({
                    'variant': name, 'seed': seed,
                    'min_tube_sf': min(r['sf_overall'] for r in analysis['line_results']),
                    'mean_abs_y_frac': float(np.mean(np.abs(y)) / half_h),
                    'valid': bool(analysis['no_overlap'] and analysis['separation_ok']),
                    'ga_fitness': result.best.fitness,
                })
    finally:
        hydraulic_line.SCORE_WEIGHTS.update(original)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize(df):
    g = df.groupby('variant').agg(
        min_sf_mean=('min_tube_sf', 'mean'), min_sf_std=('min_tube_sf', 'std'),
        abs_y_mean=('mean_abs_y_frac', 'mean'), abs_y_std=('mean_abs_y_frac', 'std'),
        valid_runs=('valid', 'sum'), runs=('seed', 'count'))
    return g


def plot_ablation(df, path='reports/ablation_ga.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in [(axes[0], 'min_tube_sf', 'Minimum tube safety factor'),
                           (axes[1], 'mean_abs_y_frac', 'Mean |y| / (h/2)  (0 = neutral axis)')]:
        data = [df[df.variant == v][col] for v in VARIANTS]
        ax.boxplot(data, tick_labels=list(VARIANTS))
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i), d, alpha=0.6, zorder=3)
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('GA ablation over 10 seeds: removing the position term')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    df = run_ablation()
    print(summarize(df).round(3).to_string())
    plot_ablation(df)
    print('Saved data/ablation_ga.csv and reports/ablation_ga.png')
