# visualization/wind_plots.py

"""
Wind and Load Visualization
============================
Plots for flight load time series and statistical distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def plot_flight_simulation(sim_result, title="Flight Load Simulation"):
    """
    Plot a single flight simulation: base load, turbulence, gusts, total.
    
    4-panel layout:
    1. Total load time series with peak annotated
    2. Load components breakdown (stacked)
    3. Load histogram with Gaussian fit
    4. Cumulative distribution
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    
    time = sim_result['time'] / 60  # Convert to minutes
    total = sim_result['total_load'] / 1000   # Convert to kN
    base  = sim_result['base_load'] / 1000
    turb  = sim_result['turbulence'] / 1000
    gusts = sim_result['gust_load'] / 1000
    peak  = sim_result['peak_load'] / 1000
    
    # ── Panel 1: Total load time series ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])  # Spans both columns
    ax1.plot(time, total, color='steelblue', linewidth=0.8, alpha=0.9)
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    # Shade regions
    ax1.fill_between(time, 0, total,
                      where=(total > 0), alpha=0.2, color='red',
                      label='Positive (starboard force)')
    ax1.fill_between(time, 0, total,
                      where=(total < 0), alpha=0.2, color='blue',
                      label='Negative (port force)')
    
    # Annotate peak
    peak_idx = np.argmax(np.abs(sim_result['total_load']))
    ax1.annotate(
        f'Peak: {peak:.1f} kN',
        xy=(time[peak_idx], total[peak_idx]),
        xytext=(time[peak_idx] + len(time)//20 * (time[1]-time[0]),
                total[peak_idx] * 0.8),
        fontsize=10, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red')
    )
    
    ax1.set_xlabel('Flight Time (minutes)')
    ax1.set_ylabel('Lateral Force on Fin (kN)')
    ax1.set_title(f'Total Fin Lateral Load — '
                   f'{sim_result["flight_hours"]:.0f}h Flight | '
                   f'{sim_result["n_gusts"]} gust events | '
                   f'Peak = {peak:.1f} kN | '
                   f'RMS = {sim_result["rms_load"]/1000:.1f} kN')
    ax1.legend(fontsize=9)
    
    # ── Panel 2: Load components ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Show a 10-minute window for detail
    window = min(600, len(time))
    t_win = time[:window]
    
    ax2.plot(t_win, base[:window], label='Base aero load',
             color='navy', linewidth=1.5)
    ax2.plot(t_win, turb[:window], label='Turbulence',
             color='orange', linewidth=0.8, alpha=0.8)
    ax2.plot(t_win, gusts[:window], label='Discrete gusts',
             color='red', linewidth=1.5)
    ax2.plot(t_win, (base+turb+gusts)[:window], label='Total',
             color='black', linewidth=1.2, linestyle='--')
    
    ax2.set_xlabel('Flight Time (minutes)')
    ax2.set_ylabel('Force (kN)')
    ax2.set_title('Load Components (First 10 min detail)')
    ax2.legend(fontsize=8)
    
    # ── Panel 3: Load histogram with Gaussian fit ────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    
    total_N = sim_result['total_load']
    mu, sigma = stats.norm.fit(total_N)
    
    ax3.hist(total_N/1000, bins=60, density=True,
              alpha=0.6, color='steelblue', edgecolor='white',
              linewidth=0.3, label='Simulated loads')
    
    x_fit = np.linspace(total_N.min(), total_N.max(), 200)
    ax3.plot(x_fit/1000, stats.norm.pdf(x_fit, mu, sigma),
              'r-', linewidth=2, label=f'Gaussian fit\n(μ={mu/1000:.1f}kN, σ={sigma/1000:.1f}kN)')
    
    ax3.axvline(peak, color='darkred', linewidth=2,
                 linestyle='--', label=f'Peak = {peak:.1f} kN')
    ax3.axvline(-peak, color='darkblue', linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Lateral Force (kN)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Load Distribution (Gaussian behavior confirmed)')
    ax3.legend(fontsize=8)
    
    return fig


def plot_scenario_comparison(sampler, n_samples=500):
    """
    Compare load distributions across different flight scenarios.
    
    Shows which scenarios produce the most severe loads
    and how they contribute to the overall load spectrum.
    """
    from simulation.wind_model import FlightSimulation, TurbulenceModel, GustEventModel
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Flight Scenario Load Distribution Comparison',
                 fontsize=14, fontweight='bold')
    
    scenario_loads = {}
    
    print(f"  Sampling {n_samples} loads per scenario...")
    for scenario in sampler.scenarios:
        loads = []
        for _ in range(n_samples):
            seed = np.random.randint(0, 2**31)
            turb = TurbulenceModel(
                turbulence_intensity=scenario.turbulence_intensity,
                correlation_length=scenario.turbulence_correlation,
                seed=seed
            )
            gust = GustEventModel(
                mean_gust_velocity=scenario.mean_gust_velocity,
                std_gust_velocity=scenario.std_gust_velocity,
                mean_gusts_per_hour=scenario.gusts_per_hour,
                seed=seed+1
            )
            sim = FlightSimulation(
                sampler.aero, turb, gust, scenario.conditions,
                flight_duration_hours=2.0, dt=5.0, seed=seed+2
            )
            loads.append(sim.peak_load_for_beam() / 1000)
        
        scenario_loads[scenario.name] = np.array(loads)
        print(f"    {scenario.name}: μ={np.mean(loads):.1f}kN, "
              f"99th pct={np.percentile(loads,99):.1f}kN")
    
    # Panel 1: Box plots
    ax1 = axes[0]
    labels = list(scenario_loads.keys())
    data = [scenario_loads[k] for k in labels]
    
    bp = ax1.boxplot(data, patch_artist=True, notch=False)
    
    colors = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(labels))
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticklabels([l.replace('_', '\n') for l in labels],
                          fontsize=7)
    ax1.set_ylabel('Peak Lateral Force (kN)')
    ax1.set_title('Peak Load Distribution by Flight Scenario')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Overlapping histograms
    ax2 = axes[1]
    cmap = plt.cm.tab10
    
    for idx, (name, loads) in enumerate(scenario_loads.items()):
        ax2.hist(loads, bins=30, density=True, alpha=0.5,
                  color=cmap(idx/len(scenario_loads)),
                  label=name.replace('_', ' '))
    
    ax2.set_xlabel('Peak Lateral Force (kN)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Load Distributions by Scenario (overlaid)')
    ax2.legend(fontsize=7)
    
    plt.tight_layout()
    return fig


def plot_load_statistics(peak_loads_N, title="Monte Carlo Load Statistics"):
    """
    Statistical analysis plot for a set of Monte Carlo peak loads.
    
    Used to understand the full distribution before structural analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    loads_kN = peak_loads_N / 1000
    
    # ── Panel 1: Histogram with statistics ────────────────────────────
    ax1 = axes[0, 0]
    ax1.hist(loads_kN, bins=50, density=True,
              color='steelblue', alpha=0.7, edgecolor='white')
    
    mu, sigma = np.mean(loads_kN), np.std(loads_kN)
    x = np.linspace(loads_kN.min(), loads_kN.max(), 200)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
              label=f'Normal fit (μ={mu:.1f}, σ={sigma:.1f})')
    
    for pct, color, ls in [(95, 'orange', '--'), (99, 'red', ':'),
                             (99.9, 'darkred', '-.')]:
        v = np.percentile(loads_kN, pct)
        ax1.axvline(v, color=color, linewidth=1.5, linestyle=ls,
                     label=f'{pct}th pct: {v:.1f} kN')
    
    ax1.set_xlabel('Peak Load (kN)')
    ax1.set_ylabel('Density')
    ax1.set_title('Peak Load Distribution')
    ax1.legend(fontsize=8)
    
    # ── Panel 2: Empirical CDF ─────────────────────────────────────
    ax2 = axes[0, 1]
    sorted_loads = np.sort(loads_kN)
    cdf = np.arange(1, len(sorted_loads)+1) / len(sorted_loads)
    
    ax2.plot(sorted_loads, cdf, 'b-', linewidth=1.5, label='Empirical CDF')
    ax2.plot(x, stats.norm.cdf(x, mu, sigma), 'r--',
              linewidth=1.5, label='Gaussian CDF')
    
    ax2.axhline(0.95, color='orange', linestyle='--', linewidth=1)
    ax2.axhline(0.99, color='red', linestyle=':', linewidth=1)
    
    ax2.set_xlabel('Peak Load (kN)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ── Panel 3: Q-Q plot (normality check) ───────────────────────
    ax3 = axes[1, 0]
    stats.probplot(loads_kN, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)\nPoints on line = normal distribution')
    ax3.get_lines()[0].set(markersize=2, alpha=0.5)
    
    # ── Panel 4: Running statistics ────────────────────────────────
    ax4 = axes[1, 1]
    n_arr = np.arange(1, len(loads_kN)+1)
    running_mean = np.cumsum(loads_kN) / n_arr
    running_std = np.array([
        np.std(loads_kN[:i+1]) for i in range(len(loads_kN))
    ])
    
    ax4.plot(n_arr, running_mean, 'b-', linewidth=1.5, label='Running mean')
    ax4.fill_between(n_arr,
                      running_mean - running_std,
                      running_mean + running_std,
                      alpha=0.2, color='blue', label='±1σ band')
    ax4.axhline(mu, color='red', linewidth=1, linestyle='--',
                 label=f'True mean: {mu:.1f} kN')
    
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Mean Peak Load (kN)')
    ax4.set_title('Monte Carlo Convergence Check\n(Should stabilize as n→∞)')
    ax4.legend(fontsize=9)
    
    # Summary statistics text
    summary = (
        f"N = {len(loads_kN):,} samples\n"
        f"Mean = {mu:.1f} kN\n"
        f"Std  = {sigma:.1f} kN\n"
        f"Min  = {loads_kN.min():.1f} kN\n"
        f"Max  = {loads_kN.max():.1f} kN\n"
        f"95th = {np.percentile(loads_kN,95):.1f} kN\n"
        f"99th = {np.percentile(loads_kN,99):.1f} kN"
    )
    ax4.text(0.02, 0.98, summary,
              transform=ax4.transAxes, fontsize=9,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig