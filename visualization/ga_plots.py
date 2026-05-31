# visualization/ga_plots.py

"""
Genetic Algorithm Visualization
================================
Plots for understanding GA optimization behavior:
1. Convergence curve (fitness over generations)
2. Population diversity over time
3. Optimal vs manual configuration comparison
4. Gene evolution heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_ga_convergence(result, title="Genetic Algorithm Convergence"):
    """
    Plot fitness evolution over generations.

    Shows:
    - Best fitness per generation (should steadily increase)
    - Mean fitness per generation (shows population quality)
    - Std deviation band (shows diversity)

    Convergence is when best fitness flattens out.
    """
    df = result.history_df

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    gens = df['generation'].values

    # ── Panel 1: Fitness convergence ───────────────────────────────────
    ax1 = axes[0]
    ax1.plot(gens, df['best_fitness'], 'g-', linewidth=2.5,
              label='Best fitness', zorder=5)
    ax1.plot(gens, df['mean_fitness'], 'b-', linewidth=1.5,
              alpha=0.7, label='Mean fitness')
    ax1.fill_between(gens,
                      df['mean_fitness'] - df['std_fitness'],
                      df['mean_fitness'] + df['std_fitness'],
                      alpha=0.2, color='blue', label='±1σ (diversity)')

    ax1.axhline(90, color='green', linewidth=1.5, linestyle='--',
                 label='Excellent (90)')
    ax1.axhline(70, color='orange', linewidth=1.5, linestyle=':',
                 label='Good (70)')

    # Mark improvement events
    best_so_far = df['best_fitness'].cummax()
    improvements = df[df['best_fitness'] == best_so_far]
    ax1.scatter(improvements['generation'],
                 improvements['best_fitness'],
                 c='gold', s=30, zorder=10, label='New best found')

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score (0-100)')
    ax1.set_title('Fitness Convergence\n(flattening = convergence)')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, df['generation'].max())
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Safety factor evolution ──────────────────────────────
    ax2 = axes[1]
    ax2.plot(gens, df['best_min_sf'], 'g-', linewidth=2.5,
              label='Best min SF')
    ax2.plot(gens, df['mean_min_sf'], 'b-', linewidth=1.5,
              alpha=0.7, label='Mean min SF')

    ax2.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Aerospace minimum (SF=1.5)')
    ax2.axhline(3.0, color='green', linewidth=1.5, linestyle='-.',
                 label='Excellent (SF=3.0)')

    ax2.fill_between(gens, 0, 1.5, alpha=0.1, color='red',
                      label='Unsafe zone')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Minimum Safety Factor')
    ax2.set_title('Safety Factor Evolution\nAcross Generations')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, df['generation'].max())
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_population_diversity(result):
    """
    Show how the gene distribution evolves across generations.
    Plots gene values at generation 0, 25%, 50%, 75%, 100%.

    Wide spread = high diversity (good exploration)
    Narrow spread = convergence (good exploitation)
    """
    n_gens     = result.config.n_generations
    checkpoints = [0,
                    n_gens//4,
                    n_gens//2,
                    3*n_gens//4,
                    n_gens]
    checkpoints = [min(c, len(result.history)-1) for c in checkpoints]

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(16, 5),
                              sharey=True)
    fig.suptitle('Population Diversity Evolution\n'
                  '(Shows how gene distribution narrows as algorithm converges)',
                  fontsize=12, fontweight='bold')

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(checkpoints)))

    for ax, gen_idx, color in zip(axes, checkpoints, colors):
        stats = result.history[gen_idx]
        gen   = stats['generation']
        best  = stats['best_fitness']

        ax.set_title(f'Gen {gen}\nBest={best:.1f}',
                      fontsize=9, fontweight='bold')

        best_genes = stats['best_genes']
        h_half     = result.evaluator.h_half

        for i, gene in enumerate(best_genes):
            y_mm = gene * h_half * 1000
            ax.barh(i, y_mm, color=color, alpha=0.8,
                     edgecolor='black', linewidth=1.2)
            ax.text(y_mm + 1, i, f'{y_mm:.1f}mm',
                     va='center', fontsize=8)

        ax.axvline(0, color='green', linewidth=2, linestyle='--',
                    label='Neutral axis')
        ax.axvline( h_half*1000*0.9, color='red', linewidth=1.5,
                    linestyle=':', label='90% of h/2')
        ax.axvline(-h_half*1000*0.9, color='red', linewidth=1.5,
                    linestyle=':')

        ax.set_xlim(-h_half*1100, h_half*1100)
        ax.set_yticks(range(len(best_genes)))
        ax.set_yticklabels([f'Line {i+1}' for i in range(len(best_genes))])
        ax.set_xlabel('y-position (mm)')
        ax.grid(True, alpha=0.3, axis='x')

        if ax == axes[0]:
            ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_optimized_vs_manual(beam, result,
                               manual_config, manual_analysis,
                               optimized_config, optimized_analysis):
    """
    Side-by-side comparison of manual vs GA-optimized configuration.

    This is the key result plot — shows what the algorithm found
    vs what a human engineer designed.
    """
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Manual vs GA-Optimized Hydraulic Configuration',
                  fontsize=14, fontweight='bold')

    h      = beam.cs.h
    colors = {'OPTIMAL': 'green', 'SAFE': 'limegreen',
               'MARGINAL': 'orange', 'UNSAFE': 'red', 'CRITICAL': 'darkred'}

    for ax, analysis, title in zip(
        axes[:2],
        [manual_analysis, optimized_analysis],
        ['Manual (Standard Config)', 'GA-Optimized Config']
    ):
        score = analysis['config_score']
        min_sf = analysis['min_safety_factor']
        ax.set_title(f'{title}\nScore: {score:.1f}/100  |  '
                      f'Min SF: {min_sf:.2f}',
                      fontweight='bold', fontsize=10)

        # Stress gradient background
        y_range = np.linspace(-h/2, h/2, 100)
        idx     = len(beam.x) // 2
        M       = beam.moment[idx]
        I       = beam.cs.second_moment_of_area
        for k in range(len(y_range)-1):
            frac  = abs(M * y_range[k] / I) / beam.max_stress()
            color = plt.cm.RdYlGn_r(frac)
            ax.barh(y_range[k]*1000, beam.L, height=(h/100)*1000,
                     left=0, color=color, alpha=0.35)

        ax.axhline(0, color='green', linewidth=2.5, linestyle='--',
                    label='Neutral Axis', zorder=5)

        for r in analysis['line_results']:
            y_mm  = r['y_position'] * 1000
            color = colors.get(r['status'], 'blue')
            rect  = patches.FancyBboxPatch(
                (r['x_start'], y_mm - 7),
                r['x_end'] - r['x_start'], 14,
                boxstyle="round,pad=0.4",
                linewidth=2, edgecolor='black',
                facecolor=color, alpha=0.85, zorder=10
            )
            ax.add_patch(rect)
            ax.text(r['x_center'], y_mm + 12,
                     f"SF={r['sf_overall']:.2f}",
                     fontsize=8, ha='center', fontweight='bold', zorder=11)
            ax.text(r['x_center'], y_mm - 16,
                     f"{r['y_position']*1000:.1f}mm",
                     fontsize=7, ha='center', color='gray', zorder=11)

        ax.set_xlim(0, beam.L)
        ax.set_ylim(-h/2*1000*1.5, h/2*1000*1.5)
        ax.set_xlabel('Spanwise Position (m)')
        ax.set_ylabel('Position from Neutral Axis (mm)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    # ── Panel 3: Metrics comparison ────────────────────────────────────
    ax3 = axes[2]
    metrics = ['Config Score', 'Min SF × 20', 'Mean SF × 10',
                'Max VM Stress\n(inverted)']

    manual_vals = [
        manual_analysis['config_score'],
        min(r['sf_overall'] for r in manual_analysis['line_results']) * 20,
        np.mean([r['sf_overall'] for r in manual_analysis['line_results']]) * 10,
        max(0, 100 - manual_analysis['max_von_mises_mpa'] / 5),
    ]
    opt_vals = [
        optimized_analysis['config_score'],
        min(r['sf_overall'] for r in optimized_analysis['line_results']) * 20,
        np.mean([r['sf_overall'] for r in optimized_analysis['line_results']]) * 10,
        max(0, 100 - optimized_analysis['max_von_mises_mpa'] / 5),
    ]

    x      = np.arange(len(metrics))
    width  = 0.35
    bars1  = ax3.bar(x - width/2, manual_vals, width, label='Manual',
                      color='steelblue', alpha=0.8, edgecolor='black')
    bars2  = ax3.bar(x + width/2, opt_vals,    width, label='GA-Optimized',
                      color='green',    alpha=0.8, edgecolor='black')

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=8)
    ax3.set_ylabel('Score (higher = better)')
    ax3.set_title('Performance Comparison\n(GA vs Manual Design)')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 120)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(list(bars1) + list(bars2),
                         manual_vals + opt_vals):
        ax3.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 1,
                  f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig
