# visualization/monte_carlo_plots.py

"""
Monte Carlo Results Visualization
===================================
Engineering-quality plots for Monte Carlo simulation results.

Plot types:
1. Safety factor distribution (histogram + CDF)
2. Stress vs load scatter
3. Failure map (which iterations failed)
4. Configuration comparison
5. Scenario contribution to failure
6. Convergence plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns


def plot_safety_distribution(df: pd.DataFrame, config_id: str = ""):
    """
    Safety factor distribution plot — the primary engineering output.
    
    Shows:
    - Distribution of min_line_sf across all iterations
    - Probability of being below SF=1.5 (unsafe) or SF=1.0 (failure)
    - Comparison with normal distribution fit
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Monte Carlo Safety Analysis — {config_id}\n'
        f'N = {len(df):,} simulated flights',
        fontsize=13, fontweight='bold'
    )
    
    sf_data = df['min_line_sf']
    
    # ── Panel 1: Histogram ─────────────────────────────────────────────
    ax1 = axes[0]
    
    ax1.hist(sf_data, bins=60, density=True,
              color='steelblue', alpha=0.7,
              edgecolor='white', linewidth=0.3)
    
    # Fit lognormal (SF is always positive, often right-skewed)
    shape, loc, scale = stats.lognorm.fit(sf_data, floc=0)
    x_fit = np.linspace(sf_data.min(), sf_data.max(), 300)
    ax1.plot(x_fit, stats.lognorm.pdf(x_fit, shape, loc, scale),
              'r-', linewidth=2, label='Lognormal fit')
    
    # Critical thresholds
    ax1.axvline(1.0, color='darkred', linewidth=2.5, linestyle='-',
                 label='SF = 1.0 (FAILURE)', zorder=5)
    ax1.axvline(1.5, color='red', linewidth=2, linestyle='--',
                 label='SF = 1.5 (aerospace minimum)')
    ax1.axvline(2.0, color='orange', linewidth=1.5, linestyle=':',
                 label='SF = 2.0 (good margin)')
    ax1.axvline(3.0, color='green', linewidth=1.5, linestyle='-.',
                 label='SF = 3.0 (excellent)')
    
    # Shade failure zone
    x_fail = np.linspace(sf_data.min(), 1.0, 100)
    ax1.fill_between(x_fail,
                      stats.lognorm.pdf(x_fail, shape, loc, scale),
                      alpha=0.3, color='red', label='Failure zone')
    
    ax1.set_xlabel('Minimum Line Safety Factor')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Safety Factor Distribution')
    ax1.legend(fontsize=7)
    
    # Statistics box
    p_fail = (sf_data < 1.0).mean()
    p_unsafe = (sf_data < 1.5).mean()
    stats_text = (
        f"μ = {sf_data.mean():.3f}\n"
        f"σ = {sf_data.std():.3f}\n"
        f"5th pct = {sf_data.quantile(0.05):.3f}\n"
        f"P(SF<1.0) = {p_fail:.5f}\n"
        f"P(SF<1.5) = {p_unsafe:.4f}"
    )
    ax1.text(0.97, 0.97, stats_text,
              transform=ax1.transAxes, fontsize=8,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # ── Panel 2: Safety factor vs applied load (scatter) ──────────────
    ax2 = axes[1]
    
    loads = df['peak_load_kN']
    
    # Color by status
    colors = np.where(sf_data < 1.0, 'red',
              np.where(sf_data < 1.5, 'orange',
              np.where(sf_data < 3.0, 'steelblue', 'green')))
    
    ax2.scatter(loads, sf_data, c=colors, alpha=0.4, s=8, linewidths=0)
    ax2.axhline(1.0, color='darkred', linewidth=2,
                 label='Failure (SF=1.0)')
    ax2.axhline(1.5, color='red', linewidth=1.5,
                 linestyle='--', label='Minimum (SF=1.5)')
    ax2.axhline(3.0, color='green', linewidth=1,
                 linestyle='-.', label='Excellent (SF=3.0)')
    
    # Add trend line
    z = np.polyfit(loads, sf_data, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(loads.min(), loads.max(), 200)
    ax2.plot(x_trend, p(x_trend), 'k--', linewidth=1.5,
              label=f'Trend (slope={z[0]:.3f})')
    
    ax2.set_xlabel('Peak Applied Load (kN)')
    ax2.set_ylabel('Minimum Line Safety Factor')
    ax2.set_title('Safety Factor vs Applied Load\n(shows load-SF relationship)')
    ax2.legend(fontsize=8)
    
    # Correlation annotation
    corr = np.corrcoef(loads, sf_data)[0, 1]
    ax2.text(0.05, 0.05,
              f'Pearson r = {corr:.3f}',
              transform=ax2.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # ── Panel 3: Cumulative distribution (exceedance curve) ───────────
    ax3 = axes[2]
    
    sorted_sf = np.sort(sf_data)
    exceedance = 1 - np.arange(1, len(sorted_sf)+1) / len(sorted_sf)
    
    ax3.semilogy(sorted_sf, exceedance, 'b-', linewidth=2,
                  label='Empirical exceedance')
    
    # Lognormal fit CDF
    sf_range = np.linspace(0.5, sorted_sf.max(), 300)
    exceedance_fit = 1 - stats.lognorm.cdf(sf_range, shape, loc, scale)
    ax3.semilogy(sf_range, np.maximum(exceedance_fit, 1e-6),
                  'r--', linewidth=2, label='Lognormal fit')
    
    ax3.axvline(1.0, color='darkred', linewidth=2,
                 label='SF = 1.0 (failure)')
    ax3.axvline(1.5, color='red', linewidth=1.5, linestyle='--',
                 label='SF = 1.5 (min)')
    
    ax3.axhline(1e-4, color='gray', linewidth=1, linestyle=':',
                 label='FAA limit (10⁻⁴)')
    
    ax3.set_xlabel('Safety Factor Threshold')
    ax3.set_ylabel('P(SF < threshold)  [log scale]')
    ax3.set_title('Safety Factor Exceedance Curve\n(probability of being below threshold)')
    ax3.legend(fontsize=8)
    ax3.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_failure_analysis(df: pd.DataFrame):
    """
    Detailed failure analysis — what causes failures and when.
    
    Shows:
    - Which scenarios contribute most to failures
    - How failures correlate with load magnitude
    - Failure frequency over simulation iterations
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Failure Analysis — Monte Carlo Results',
                 fontsize=13, fontweight='bold')
    
    failure_df = df[df['any_failure'] == 1]
    safe_df = df[df['any_failure'] == 0]
    
    p_fail = len(failure_df) / len(df)
    
    # ── Panel 1: Failure rate by scenario ──────────────────────────────
    ax1 = axes[0, 0]
    
    scenario_stats = df.groupby('scenario_name').agg(
        n_total=('any_failure', 'count'),
        n_fail=('any_failure', 'sum')
    ).reset_index()
    scenario_stats['fail_rate'] = (scenario_stats['n_fail'] / 
                                    scenario_stats['n_total'])
    scenario_stats = scenario_stats.sort_values('fail_rate', ascending=True)
    
    colors = ['red' if r > 0.01 else 'orange' if r > 0.001 else 'green'
              for r in scenario_stats['fail_rate']]
    
    bars = ax1.barh(scenario_stats['scenario_name'],
                     scenario_stats['fail_rate'],
                     color=colors, alpha=0.75, edgecolor='black',
                     linewidth=0.8)
    
    ax1.axvline(1e-4, color='gray', linewidth=1.5, linestyle='--',
                 label='FAA target (10⁻⁴)')
    
    for bar, rate in zip(bars, scenario_stats['fail_rate']):
        if rate > 0:
            ax1.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                      f'{rate:.4f}', va='center', fontsize=8)
    
    ax1.set_xlabel('Failure Rate')
    ax1.set_title(f'Failure Rate by Flight Scenario\n'
                   f'Overall P(failure) = {p_fail:.5f}')
    ax1.legend(fontsize=9)
    
    # ── Panel 2: Load histogram — failed vs safe ────────────────────────
    ax2 = axes[0, 1]
    
    bins = np.linspace(df['peak_load_kN'].min(), df['peak_load_kN'].max(), 40)
    
    ax2.hist(safe_df['peak_load_kN'], bins=bins, density=True,
              alpha=0.5, color='green', label=f'Safe ({len(safe_df):,})')
    
    if len(failure_df) > 0:
        ax2.hist(failure_df['peak_load_kN'], bins=bins, density=True,
                  alpha=0.7, color='red',
                  label=f'Failure ({len(failure_df):,})')
    
    ax2.axvline(df['peak_load_kN'].quantile(0.99), color='orange',
                 linestyle='--', linewidth=1.5,
                 label='99th pct load')
    
    ax2.set_xlabel('Peak Applied Load (kN)')
    ax2.set_ylabel('Density')
    ax2.set_title('Load Distribution: Safe vs Failed Flights')
    ax2.legend(fontsize=9)
    
    # ── Panel 3: Stress-SF joint distribution ─────────────────────────
    ax3 = axes[1, 0]
    
    scatter = ax3.scatter(
        df['max_beam_stress_mpa'],
        df['min_line_sf'],
        c=df['any_failure'],
        cmap='RdYlGn_r',
        alpha=0.4, s=8, linewidths=0
    )
    
    ax3.axhline(1.0, color='red', linewidth=2, linestyle='-',
                 label='Failure threshold SF=1.0')
    ax3.axhline(1.5, color='orange', linewidth=1.5, linestyle='--',
                 label='Aerospace minimum SF=1.5')
    
    plt.colorbar(scatter, ax=ax3, label='Failure (1=failed)')
    ax3.set_xlabel('Max Beam Stress (MPa)')
    ax3.set_ylabel('Min Line Safety Factor')
    ax3.set_title('Stress-Safety Factor Joint Distribution')
    ax3.legend(fontsize=8)
    
    # ── Panel 4: Running failure probability ──────────────────────────
    ax4 = axes[1, 1]
    
    failures = df['any_failure'].values
    iterations = np.arange(1, len(failures)+1)
    running_p_fail = np.cumsum(failures) / iterations
    
    ax4.plot(iterations, running_p_fail, 'b-', linewidth=1.5,
              label='Running P(failure)')
    ax4.axhline(p_fail, color='red', linewidth=2, linestyle='--',
                 label=f'Final P(failure) = {p_fail:.5f}')
    ax4.axhline(1e-4, color='gray', linewidth=1.5, linestyle=':',
                 label='FAA target (10⁻⁴)')
    
    ax4.set_xlabel('Monte Carlo Iteration')
    ax4.set_ylabel('Cumulative P(failure)')
    ax4.set_title('Monte Carlo Convergence of Failure Probability')
    ax4.legend(fontsize=9)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_configuration_comparison(comparison_df: pd.DataFrame):
    """
    Visual comparison of multiple hydraulic configurations.
    
    Shows safety scores, failure probabilities, and min safety factors
    side-by-side to help engineers choose the best configuration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Hydraulic Configuration Comparison — Monte Carlo Results',
                 fontsize=13, fontweight='bold')
    
    configs = comparison_df['config_id']
    x = np.arange(len(configs))
    
    bar_colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red'
                  for s in comparison_df['config_score_mean']]
    
    # ── Panel 1: Config score ──────────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(x, comparison_df['config_score_mean'],
                    color=bar_colors, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, comparison_df['config_score_mean']):
        ax1.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.5,
                  f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    ax1.axhline(70, color='green', linestyle='--', linewidth=1.5,
                 label='Good threshold (70)')
    ax1.axhline(50, color='orange', linestyle='--', linewidth=1.5,
                 label='Acceptable (50)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Configuration Score (0-100)')
    ax1.set_title('Overall Safety Score')
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8)
    
    # ── Panel 2: Min line safety factor ───────────────────────────────
    ax2 = axes[1]
    
    sf_means = comparison_df['min_line_sf_mean']
    sf_5ths = comparison_df['min_line_sf_5th']
    
    ax2.bar(x - 0.2, sf_means, 0.35, label='Mean min SF',
             color='steelblue', alpha=0.8, edgecolor='black')
    ax2.bar(x + 0.2, sf_5ths, 0.35, label='5th pct min SF',
             color='lightsalmon', alpha=0.8, edgecolor='black')
    
    ax2.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Aerospace minimum (SF=1.5)')
    ax2.axhline(1.0, color='darkred', linewidth=2, linestyle='-',
                 label='Failure threshold (SF=1.0)')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Minimum Line Safety Factor')
    ax2.set_title('Safety Factor Comparison\n(5th pct = worst 5% of flights)')
    ax2.legend(fontsize=8)
    
    # ── Panel 3: Probability of failure ───────────────────────────────
    ax3 = axes[2]
    
    p_fails = comparison_df['p_failure']
    fail_colors = ['green' if p < 1e-4 else 'orange' if p < 1e-3 else 'red'
                   for p in p_fails]
    
    ax3.bar(x, p_fails, color=fail_colors, alpha=0.8,
             edgecolor='black', linewidth=1.2)
    ax3.axhline(1e-4, color='gray', linewidth=2, linestyle='--',
                 label='FAA maximum (10⁻⁴)')
    
    for xi, (p, color) in enumerate(zip(p_fails, fail_colors)):
        ax3.text(xi, p + max(p_fails) * 0.02,
                  f'{p:.4f}', ha='center', fontsize=9, fontweight='bold',
                  color=color)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel('Probability of Failure')
    ax3.set_title('P(Failure) Comparison\n(Green = passes FAA requirement)')
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_convergence_analysis(result):
    """
    Monte Carlo convergence plot — shows how many iterations are needed.
    
    Plots running statistics vs iteration count.
    When the curve flattens out → simulation has converged.
    """
    df = result.dataframe
    N = len(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Monte Carlo Convergence Analysis (N={N:,})',
                 fontsize=13, fontweight='bold')
    
    iters = np.arange(1, N+1)
    
    metrics = [
        ('min_line_sf', 'Mean Min Line SF', axes[0,0], 'steelblue'),
        ('config_score', 'Mean Config Score', axes[0,1], 'green'),
        ('max_line_vm_mpa', 'Mean Max Von Mises (MPa)', axes[1,0], 'orange'),
        ('any_failure', 'Cumulative P(failure)', axes[1,1], 'red'),
    ]
    
    for col, label, ax, color in metrics:
        values = df[col].values
        running_mean = np.cumsum(values) / iters
        
        # Running standard deviation (for confidence band)
        running_std = np.array([
            np.std(values[:i+1]) / np.sqrt(i+1)  # Standard error
            for i in range(min(N, 500))            # First 500 for speed
        ])
        iters_short = iters[:len(running_std)]
        mean_short = running_mean[:len(running_std)]
        
        ax.plot(iters, running_mean, color=color, linewidth=1.5,
                 label='Running mean')
        ax.fill_between(iters_short,
                         mean_short - 2*running_std,
                         mean_short + 2*running_std,
                         alpha=0.2, color=color, label='95% conf. interval')
        
        # Final value
        final = running_mean[-1]
        ax.axhline(final, color='black', linewidth=1, linestyle='--',
                    label=f'Final value: {final:.4f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.set_title(f'Convergence: {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig