# visualization/analysis_plots.py

"""
Data Science Visualization for Phase 6
========================================
Engineering-quality plots for Monte Carlo dataset analysis.

Plots:
1. Correlation heatmap
2. Load vs safety factor scatter (colored by scenario)
3. Safety factor distribution comparison (std vs dangerous)
4. Per-scenario box plots
5. Stress utilization histogram
6. 3D surface: load × position × safety factor
7. Failure probability by load bin
8. Feature importance preview
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


def plot_correlation_heatmap(df: pd.DataFrame, title: str = ""):
    """
    Correlation heatmap of all numeric engineering variables.
    
    Shows how every variable relates to every other variable.
    Hot colors (red) = strong positive correlation
    Cool colors (blue) = strong negative correlation
    White = no correlation
    
    Engineering insight: if two variables are highly correlated,
    you only need one of them for ML (reduces overfitting).
    """
    # Select relevant engineering columns only
    cols = [
        'peak_load_kN', 'sideslip_deg', 'turbulence_intensity',
        'max_beam_stress_mpa', 'beam_safety_factor',
        'tip_deflection_mm', 'root_moment_kNm',
        'min_line_sf', 'max_line_vm_mpa',
        'config_score', 'any_failure'
    ]
    cols = [c for c in cols if c in df.columns]
    
    corr_matrix = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title(
        f'Engineering Variable Correlation Matrix\n{title}',
        fontsize=13, fontweight='bold', pad=20
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_load_vs_safety(df: pd.DataFrame, title: str = ""):
    """
    Scatter plot: peak load vs safety factor, colored by scenario.
    
    This is the most important engineering plot — shows the
    fundamental trade-off between load and safety.
    
    Key features:
    - Each point is one simulated flight
    - Color = flight scenario
    - Red horizontal lines = critical thresholds
    - Trend line shows load-SF relationship
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Load vs Safety Factor Analysis\n{title}',
                  fontsize=13, fontweight='bold')
    
    scenarios = df['scenario_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    scenario_colors = dict(zip(scenarios, colors))
    
    # ── Panel 1: Load vs Beam SF ───────────────────────────────────────
    ax1 = axes[0]
    
    for scenario in scenarios:
        mask = df['scenario_name'] == scenario
        ax1.scatter(
            df.loc[mask, 'peak_load_kN'],
            df.loc[mask, 'beam_safety_factor'],
            c=[scenario_colors[scenario]],
            alpha=0.4, s=10, linewidths=0,
            label=scenario.replace('_', ' ')
        )
    
    # Threshold lines
    ax1.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Min SF = 1.5')
    ax1.axhline(1.0, color='darkred', linewidth=2,
                 label='Failure SF = 1.0')
    ax1.axhline(3.0, color='green', linewidth=1.5, linestyle='-.',
                 label='Excellent SF = 3.0')
    
    # Trend line
    z = np.polyfit(df['peak_load_kN'], df['beam_safety_factor'], 1)
    x_trend = np.linspace(df['peak_load_kN'].min(),
                           df['peak_load_kN'].max(), 200)
    ax1.plot(x_trend, np.poly1d(z)(x_trend), 'k--',
              linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Peak Applied Load (kN)', fontsize=11)
    ax1.set_ylabel('Beam Safety Factor', fontsize=11)
    ax1.set_title('Load vs Beam Safety Factor')
    ax1.legend(fontsize=7, loc='upper right')
    
    corr = df['peak_load_kN'].corr(df['beam_safety_factor'])
    ax1.text(0.05, 0.05, f'Pearson r = {corr:.3f}',
              transform=ax1.transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ── Panel 2: Load vs Min Line SF ────────────────────────────────────
    ax2 = axes[1]
    
    for scenario in scenarios:
        mask = df['scenario_name'] == scenario
        ax2.scatter(
            df.loc[mask, 'peak_load_kN'],
            df.loc[mask, 'min_line_sf'],
            c=[scenario_colors[scenario]],
            alpha=0.4, s=10, linewidths=0,
            label=scenario.replace('_', ' ')
        )
    
    ax2.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Min SF = 1.5')
    ax2.axhline(1.0, color='darkred', linewidth=2,
                 label='Failure SF = 1.0')
    
    # Mark failure points
    fail_mask = df['any_failure'] == 1
    if fail_mask.sum() > 0:
        ax2.scatter(df.loc[fail_mask, 'peak_load_kN'],
                     df.loc[fail_mask, 'min_line_sf'],
                     c='red', s=80, marker='X', zorder=10,
                     label=f'FAILURE ({fail_mask.sum()} events)')
    
    z2 = np.polyfit(df['peak_load_kN'], df['min_line_sf'], 1)
    ax2.plot(x_trend, np.poly1d(z2)(x_trend), 'k--',
              linewidth=2, label=f'Trend (slope={z2[0]:.4f})')
    
    ax2.set_xlabel('Peak Applied Load (kN)', fontsize=11)
    ax2.set_ylabel('Min Hydraulic Line Safety Factor', fontsize=11)
    ax2.set_title('Load vs Hydraulic Line Safety Factor')
    ax2.legend(fontsize=7, loc='upper right')
    
    corr2 = df['peak_load_kN'].corr(df['min_line_sf'])
    ax2.text(0.05, 0.05, f'Pearson r = {corr2:.3f}',
              transform=ax2.transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_scenario_comparison(df: pd.DataFrame, title: str = ""):
    """
    Box plots comparing safety factors across flight scenarios.
    
    Box plots show:
    - Median (middle line)
    - IQR (box = 25th to 75th percentile)
    - Whiskers (1.5 × IQR)
    - Outliers (dots beyond whiskers)
    
    Engineering use: identify which scenarios need the most attention.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Safety Factor by Flight Scenario\n{title}',
                  fontsize=13, fontweight='bold')
    
    scenario_order = df.groupby('scenario_name')['peak_load_kN'].mean()\
                       .sort_values(ascending=False).index.tolist()
    
    risk_colors = {
        'CRUISE_LIGHT_TURB':  '#2ecc71',
        'CRUISE_MOD_TURB':    '#f39c12',
        'CRUISE_SEVERE_TURB': '#e74c3c',
        'APPROACH_CROSSWIND': '#f39c12',
        'CLIMB_SHEAR':        '#f39c12',
        'MANEUVER_YAW':       '#3498db',
        'EXTREME_GUST':       '#e74c3c',
    }
    
    palette = {s: risk_colors.get(s, '#95a5a6') for s in scenario_order}
    
    # Panel 1: Beam SF
    ax1 = axes[0]
    bp1 = sns.boxplot(
        data=df, x='scenario_name', y='beam_safety_factor',
        order=scenario_order, palette=palette, ax=ax1,
        width=0.6, fliersize=3
    )
    ax1.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Minimum SF = 1.5')
    ax1.axhline(1.0, color='darkred', linewidth=2.5,
                 label='Failure threshold')
    ax1.set_xticklabels([s.replace('_', '\n') for s in scenario_order],
                          fontsize=8)
    ax1.set_xlabel('Flight Scenario')
    ax1.set_ylabel('Beam Safety Factor')
    ax1.set_title('Beam Structural Safety by Scenario')
    ax1.legend(fontsize=9)
    
    # Panel 2: Failure probability per scenario
    ax2 = axes[1]
    scenario_stats = df.groupby('scenario_name').agg(
        p_fail=('any_failure', 'mean'),
        mean_load=('peak_load_kN', 'mean'),
        count=('any_failure', 'count')
    ).reindex(scenario_order)
    
    bar_colors = [risk_colors.get(s, '#95a5a6') for s in scenario_order]
    bars = ax2.bar(range(len(scenario_order)),
                    scenario_stats['p_fail'],
                    color=bar_colors, alpha=0.8,
                    edgecolor='black', linewidth=0.8)
    
    ax2.axhline(1e-4, color='gray', linewidth=2, linestyle='--',
                 label='FAA target (10⁻⁴)')
    
    ax2.set_xticks(range(len(scenario_order)))
    ax2.set_xticklabels([s.replace('_', '\n') for s in scenario_order],
                          fontsize=8)
    ax2.set_xlabel('Flight Scenario')
    ax2.set_ylabel('Probability of Failure')
    ax2.set_title('Failure Rate by Scenario')
    ax2.legend(fontsize=9)
    
    # Annotate bars with values
    for i, (bar, (_, row)) in enumerate(zip(bars, scenario_stats.iterrows())):
        if row['p_fail'] > 0:
            ax2.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.0001,
                      f"{row['p_fail']:.4f}",
                      ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_configuration_comparison(df_std: pd.DataFrame,
                                    df_bad: pd.DataFrame):
    """
    Side-by-side comparison of standard vs dangerous configurations.
    
    4-panel layout:
    1. Safety factor distributions (overlaid histograms)
    2. Config score comparison
    3. Von Mises stress comparison
    4. Failure breakdown by scenario
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Standard vs Dangerous Configuration — Statistical Comparison',
                  fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # ── Panel 1: Safety factor distribution overlay ────────────────────
    bins = np.linspace(
        min(df_std['min_line_sf'].min(), df_bad['min_line_sf'].min()),
        max(df_std['min_line_sf'].max(), df_bad['min_line_sf'].max()),
        50
    )
    
    ax1.hist(df_std['min_line_sf'], bins=bins, density=True,
              alpha=0.6, color='green', label='Standard Config')
    ax1.hist(df_bad['min_line_sf'], bins=bins, density=True,
              alpha=0.6, color='red', label='Dangerous Config')
    
    ax1.axvline(1.5, color='black', linewidth=2, linestyle='--',
                 label='Min required SF')
    ax1.axvline(1.0, color='darkred', linewidth=2,
                 label='Failure threshold')
    
    ax1.set_xlabel('Min Line Safety Factor')
    ax1.set_ylabel('Density')
    ax1.set_title('Safety Factor Distribution Comparison')
    ax1.legend(fontsize=9)
    
    # Statistics annotation
    std_mean = df_std['min_line_sf'].mean()
    bad_mean = df_bad['min_line_sf'].mean()
    ax1.text(0.05, 0.95,
              f'Standard: μ={std_mean:.2f}\nDangerous: μ={bad_mean:.2f}',
              transform=ax1.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ── Panel 2: Config score over load ───────────────────────────────
    ax2.scatter(df_std['peak_load_kN'], df_std['config_score'],
                 alpha=0.3, s=8, color='green', label='Standard')
    ax2.scatter(df_bad['peak_load_kN'], df_bad['config_score'],
                 alpha=0.3, s=8, color='red', label='Dangerous')
    
    ax2.axhline(70, color='green', linestyle='--', linewidth=1.5,
                 label='Good threshold (70)')
    ax2.axhline(50, color='orange', linestyle='--', linewidth=1.5,
                 label='Acceptable (50)')
    
    ax2.set_xlabel('Peak Load (kN)')
    ax2.set_ylabel('Configuration Safety Score')
    ax2.set_title('Config Score vs Applied Load')
    ax2.legend(fontsize=8)
    
    # ── Panel 3: Von Mises stress distributions ────────────────────────
    bins_vm = np.linspace(
        min(df_std['max_line_vm_mpa'].min(), df_bad['max_line_vm_mpa'].min()),
        max(df_std['max_line_vm_mpa'].max(), df_bad['max_line_vm_mpa'].max()),
        50
    )
    ax3.hist(df_std['max_line_vm_mpa'], bins=bins_vm, density=True,
              alpha=0.6, color='green', label='Standard')
    ax3.hist(df_bad['max_line_vm_mpa'], bins=bins_vm, density=True,
              alpha=0.6, color='red', label='Dangerous')
    
    ax3.axvline(345, color='darkred', linewidth=2,
                 linestyle='--', label='Yield strength (345 MPa)')
    ax3.axvline(345/1.5, color='orange', linewidth=1.5,
                 linestyle=':', label='Design limit (230 MPa)')
    
    ax3.set_xlabel('Max Von Mises Stress (MPa)')
    ax3.set_ylabel('Density')
    ax3.set_title('Hydraulic Line Von Mises Stress Distribution')
    ax3.legend(fontsize=8)
    
    # ── Panel 4: Failure breakdown ────────────────────────────────────
    categories = ['Any Failure', 'SF < 1.5', 'SF < 1.0']
    std_rates = [
        df_std['any_failure'].mean(),
        (df_std['min_line_sf'] < 1.5).mean(),
        (df_std['min_line_sf'] < 1.0).mean(),
    ]
    bad_rates = [
        df_bad['any_failure'].mean(),
        (df_bad['min_line_sf'] < 1.5).mean(),
        (df_bad['min_line_sf'] < 1.0).mean(),
    ]
    
    x = np.arange(len(categories))
    w = 0.35
    ax4.bar(x - w/2, std_rates, w, label='Standard', color='green',
             alpha=0.8, edgecolor='black')
    ax4.bar(x + w/2, bad_rates, w, label='Dangerous', color='red',
             alpha=0.8, edgecolor='black')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel('Rate (fraction of flights)')
    ax4.set_title('Failure Rate Comparison')
    ax4.legend(fontsize=9)
    ax4.axhline(1e-4, color='gray', linestyle='--',
                 linewidth=1.5, label='FAA target')
    
    for xi, (sv, bv) in enumerate(zip(std_rates, bad_rates)):
        if sv > 0:
            ax4.text(xi - w/2, sv + 0.0001,
                      f'{sv:.4f}', ha='center', fontsize=8)
        if bv > 0:
            ax4.text(xi + w/2, bv + 0.0001,
                      f'{bv:.4f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_3d_stress_surface(df: pd.DataFrame, title: str = ""):
    """
    3D surface plot: Load × Turbulence → Safety Factor
    
    Shows how two input variables JOINTLY affect safety.
    This is something 2D plots cannot show.
    
    Useful for: identifying the dangerous combinations
    (high load AND high turbulence = most dangerous)
    """
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'3D Safety Factor Surface\n{title}',
                  fontsize=13, fontweight='bold')
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    for ax, y_col, y_label, color in [
        (ax1, 'beam_safety_factor', 'Beam Safety Factor', 'viridis'),
        (ax2, 'min_line_sf', 'Min Line Safety Factor', 'plasma'),
    ]:
        # Create grid from scatter data
        x = df['peak_load_kN'].values
        y = df['turbulence_intensity'].values
        z = df[y_col].values
        
        # Scatter the actual data points
        scatter = ax.scatter(x, y, z, c=z, cmap=color,
                              alpha=0.4, s=5)
        
        ax.set_xlabel('Peak Load (kN)', fontsize=9)
        ax.set_ylabel('Turbulence (m/s)', fontsize=9)
        ax.set_zlabel(y_label, fontsize=9)
        ax.set_title(y_label, fontsize=10)
        
        # Mark failure threshold plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(*xlim, 10),
                               np.linspace(*ylim, 10))
        zz_threshold = np.ones_like(xx) * 1.5
        ax.plot_surface(xx, yy, zz_threshold, alpha=0.15,
                         color='red', label='SF=1.5 threshold')
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
    
    plt.tight_layout()
    return fig


def plot_failure_probability_by_load(df: pd.DataFrame, title: str = ""):
    """
    Failure probability as a function of load magnitude.
    
    This is the key risk curve — shows exactly what load level
    starts causing failures. Aerospace engineers call this
    the "exceedance curve" or "risk curve".
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Failure Probability vs Load\n{title}',
                  fontsize=13, fontweight='bold')
    
    # Bin loads and compute failure rate per bin
    n_bins = 20
    df['load_bin'] = pd.cut(df['peak_load_kN'], bins=n_bins)
    bin_stats = df.groupby('load_bin', observed=True).agg(
        p_fail=('any_failure', 'mean'),
        count=('any_failure', 'count'),
        mean_load=('peak_load_kN', 'mean'),
        mean_sf=('beam_safety_factor', 'mean'),
    ).dropna()
    
    # ── Panel 1: P(failure) vs load bin ───────────────────────────────
    ax1 = axes[0]
    
    colors = ['red' if p > 0.01 else 'orange' if p > 0.001
              else 'green' for p in bin_stats['p_fail']]
    
    ax1.bar(range(len(bin_stats)), bin_stats['p_fail'],
             color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axhline(1e-4, color='gray', linewidth=2, linestyle='--',
                 label='FAA target (10⁻⁴)')
    
    ax1.set_xlabel('Load Bin (increasing →)')
    ax1.set_ylabel('P(failure) in this load range')
    ax1.set_title('Failure Rate by Load Magnitude\n(Green=safe, Orange=marginal, Red=dangerous)')
    ax1.legend(fontsize=9)
    
    # ── Panel 2: Mean SF vs load with confidence band ─────────────────
    ax2 = axes[1]
    
    x_vals = bin_stats['mean_load'].values
    y_vals = bin_stats['mean_sf'].values
    
    ax2.plot(x_vals, y_vals, 'b-o', linewidth=2,
              markersize=6, label='Mean Safety Factor')
    
    ax2.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Minimum SF = 1.5')
    ax2.axhline(1.0, color='darkred', linewidth=2,
                 label='Failure SF = 1.0')
    ax2.axhline(3.0, color='green', linewidth=1.5, linestyle='-.',
                 label='Excellent SF = 3.0')
    
    # Shade the unsafe zone
    ax2.fill_between(
        [df['peak_load_kN'].min(), df['peak_load_kN'].max()],
        [0, 0], [1.5, 1.5],
        alpha=0.1, color='red', label='Unsafe zone'
    )
    
    ax2.set_xlabel('Mean Peak Load in Bin (kN)')
    ax2.set_ylabel('Mean Safety Factor')
    ax2.set_title('Safety Factor vs Load Level\n(Shows which loads become dangerous)')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    return fig