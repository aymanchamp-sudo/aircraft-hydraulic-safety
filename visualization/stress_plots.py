# Stress distribution plots
# visualization/stress_plots.py

"""
Stress Visualization Module
============================
Engineering-quality plots for structural analysis results.

Plot types:
1. Bending moment diagram (BMD)
2. Shear force diagram (SFD)
3. Stress distribution along span
4. Deflection curve
5. Cross-section stress profile
6. Safety factor distribution
7. Combined engineering dashboard plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


# Set publication-quality plot style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.dpi': 100,
})


def plot_bending_moment_diagram(beam, ax=None, color='steelblue'):
    """
    Plot the Bending Moment Diagram (BMD).
    
    Convention: plot moment magnitude, shaded from zero line.
    Maximum at root, zero at tip.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    x_norm = beam.x / beam.L  # Normalize to 0-1 (fraction of span)
    moment_kNm = beam.moment / 1000  # Convert to kN·m for readability
    
    ax.fill_between(x_norm, 0, moment_kNm, alpha=0.3, color=color)
    ax.plot(x_norm, moment_kNm, color=color, linewidth=2.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Annotate maximum moment at root
    ax.annotate(
        f'M_max = {moment_kNm[0]:.1f} kN·m',
        xy=(0, moment_kNm[0]),
        xytext=(0.15, moment_kNm[0] * 0.85),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black'),
        color='darkred', fontweight='bold'
    )
    
    ax.set_xlabel('Spanwise Position (fraction of span)')
    ax.set_ylabel('Bending Moment (kN·m)')
    ax.set_title('Bending Moment Diagram — Aircraft Fin')
    ax.set_xlim(0, 1)
    
    # Add root/tip labels
    ax.text(0.02, 0.05, 'ROOT\n(Fixed)', transform=ax.transAxes,
            color='red', fontsize=9, fontweight='bold')
    ax.text(0.92, 0.05, 'TIP\n(Free)', transform=ax.transAxes,
            color='green', fontsize=9, fontweight='bold')
    
    return ax


def plot_stress_distribution(beam, ax=None):
    """
    Plot stress at top and bottom surfaces along the span.
    Shows where the structure is most stressed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    x_norm = beam.x / beam.L
    stress_top_MPa = beam.stress_top / 1e6
    stress_bot_MPa = -beam.stress_bot / 1e6  # Show magnitude
    yield_MPa = beam.material.yield_strength / 1e6
    
    ax.plot(x_norm, np.abs(stress_top_MPa), color='crimson',
            label='Top surface stress', linewidth=2.5)
    ax.plot(x_norm, stress_bot_MPa, color='royalblue',
            label='Bottom surface stress', linewidth=2.5, linestyle='--')
    
    # Yield strength line — crossing this means failure
    ax.axhline(yield_MPa, color='red', linestyle=':', linewidth=2,
               label=f'Yield Strength ({yield_MPa:.0f} MPa) ← FAILURE')
    
    # Design limit line (SF=1.5)
    design_limit = yield_MPa / 1.5
    ax.axhline(design_limit, color='orange', linestyle=':', linewidth=1.5,
               label=f'Design Limit (SF=1.5, {design_limit:.0f} MPa)')
    
    ax.set_xlabel('Spanwise Position (fraction of span)')
    ax.set_ylabel('Stress Magnitude (MPa)')
    ax.set_title(f'Bending Stress Distribution — {beam.material.name}')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    return ax


def plot_deflection_curve(beam, ax=None):
    """
    Plot the deflection (bending) shape of the beam.
    This shows how much the fin tip moves under load.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    x_norm = beam.x / beam.L
    deflection_mm = beam.deflection * 1000  # Convert m to mm
    
    ax.plot(x_norm, deflection_mm, color='darkgreen', linewidth=2.5)
    ax.fill_between(x_norm, 0, deflection_mm, alpha=0.15, color='green')
    
    # Annotate tip deflection
    ax.annotate(
        f'δ_tip = {deflection_mm[-1]:.1f} mm',
        xy=(1.0, deflection_mm[-1]),
        xytext=(0.7, deflection_mm[-1] * 0.7),
        fontsize=10, fontweight='bold', color='darkgreen',
        arrowprops=dict(arrowstyle='->', color='darkgreen')
    )
    
    ax.set_xlabel('Spanwise Position (fraction of span)')
    ax.set_ylabel('Lateral Deflection (mm)')
    ax.set_title('Beam Deflection Curve — Exaggerated Shape')
    ax.set_xlim(0, 1)
    
    return ax


def plot_cross_section_stress_profile(beam, x_positions=None):
    """
    Plot stress distribution ACROSS the cross-section at multiple spanwise locations.
    
    This is the key educational plot — shows that stress is:
    - Maximum at top/bottom surfaces (y = ±h/2)
    - Zero at neutral axis (y = 0)
    - Linear in between (σ = My/I is linear in y)
    
    Also shows WHY hydraulic lines should be near the neutral axis.
    """
    if x_positions is None:
        # Show profiles at root (most stressed) and 25%, 50%, 75% span
        x_positions = [0.0, 0.25, 0.5, 0.75]
    
    n_profiles = len(x_positions)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4*n_profiles, 6),
                              sharey=True)
    
    if n_profiles == 1:
        axes = [axes]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_profiles))
    yield_MPa = beam.material.yield_strength / 1e6
    
    for ax, x_frac, color in zip(axes, x_positions, colors):
        x_pos = x_frac * beam.L
        y_array, sigma_array = beam.get_stress_profile_at_x(x_pos, n_points=100)
        
        y_mm = y_array * 1000  # Convert to mm
        sigma_MPa = sigma_array / 1e6
        
        # Plot the stress profile
        ax.plot(sigma_MPa, y_mm, color=color, linewidth=2.5)
        ax.fill_betweenx(y_mm, 0, sigma_MPa, alpha=0.25, color=color)
        
        # Mark neutral axis
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--',
                   label='Neutral axis')
        ax.axvline(0, color='gray', linewidth=0.8)
        
        # Mark yield strength limits
        ax.axvline(yield_MPa, color='red', linewidth=1, linestyle=':',
                   alpha=0.7, label='Yield strength')
        ax.axvline(-yield_MPa, color='red', linewidth=1, linestyle=':',
                   alpha=0.7)
        
        # Label the neutral axis zone (where hydraulic lines are safest)
        h_mm = beam.cs.h * 1000
        safe_y_range = h_mm * 0.15  # ±15% of height = safe zone
        ax.axhspan(-safe_y_range, safe_y_range, alpha=0.1, color='green',
                   label='Safe zone (near NA)')
        
        ax.set_xlabel('Stress (MPa)', fontsize=9)
        ax.set_title(f'x = {x_frac:.0%} span\nM = {beam.moment[int(x_frac*beam.n)]:.0f} N·m',
                     fontsize=9)
        ax.set_xlim(-yield_MPa*1.3, yield_MPa*1.3)
        
        if ax == axes[0]:
            ax.set_ylabel('Position from Neutral Axis (mm)', fontsize=9)
            ax.legend(fontsize=7, loc='lower right')
        
        # Color-coded zone labels
        ax.text(0.02, 0.98, f'MAX σ =\n{abs(sigma_MPa).max():.0f} MPa',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                color=color, fontweight='bold')
    
    fig.suptitle(
        'Cross-Section Stress Profiles at Different Spanwise Locations\n'
        '(Green zone = safe for hydraulic line placement)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    return fig


def plot_safety_factor_distribution(beam, ax=None):
    """
    Plot safety factor along the span.
    
    Aerospace requirement: SF >= 1.5 everywhere.
    This plot immediately shows which regions are under-designed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    x_norm = beam.x / beam.L
    
    # Avoid division by zero at tip
    stress_abs = np.maximum(np.abs(beam.stress_top), 1e-6)
    sf = beam.material.yield_strength / stress_abs
    
    # Color regions by safety level
    ax.fill_between(x_norm, 0, sf,
                     where=(sf >= 3.0), alpha=0.3, color='green',
                     label='SF ≥ 3.0 (Excellent)')
    ax.fill_between(x_norm, 0, sf,
                     where=((sf >= 1.5) & (sf < 3.0)), alpha=0.3, color='orange',
                     label='1.5 ≤ SF < 3.0 (Acceptable)')
    ax.fill_between(x_norm, 0, sf,
                     where=(sf < 1.5), alpha=0.3, color='red',
                     label='SF < 1.5 (UNSAFE!)')
    
    ax.plot(x_norm, sf, color='black', linewidth=1.5)
    ax.axhline(1.5, color='red', linestyle='--', linewidth=2,
               label='Minimum SF = 1.5 (aerospace requirement)')
    ax.axhline(3.0, color='green', linestyle='--', linewidth=1.5,
               label='SF = 3.0 (excellent)')
    
    ax.set_xlabel('Spanwise Position (fraction of span)')
    ax.set_ylabel('Safety Factor (dimensionless)')
    ax.set_title('Safety Factor Distribution Along Span')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(sf.max() * 1.1, 20))  # Cap at 20 for readability
    
    return ax


def plot_full_engineering_dashboard(beam):
    """
    Combined 6-panel engineering dashboard.
    This is what a real structural analysis report looks like.
    
    Layout:
    ┌─────────────────┬─────────────────┐
    │  Bending Moment │  Shear Force    │
    ├─────────────────┼─────────────────┤
    │  Stress Dist.   │  Deflection     │
    ├─────────────────┼─────────────────┤
    │  Safety Factor  │  Beam Schematic │
    └─────────────────┴─────────────────┘
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'Aircraft Fin Structural Analysis Dashboard\n'
        f'Material: {beam.material.name} | '
        f'L = {beam.L:.1f}m | '
        f'Tip Load = {beam._tip_load/1000:.1f}kN',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Panel 1: Bending Moment
    plot_bending_moment_diagram(beam, ax=ax1)
    
    # Panel 2: Shear Force
    x_norm = beam.x / beam.L
    ax2.fill_between(x_norm, 0, beam.shear/1000, alpha=0.3, color='purple')
    ax2.plot(x_norm, beam.shear/1000, color='purple', linewidth=2)
    ax2.set_xlabel('Spanwise Position (fraction of span)')
    ax2.set_ylabel('Shear Force (kN)')
    ax2.set_title('Shear Force Diagram')
    ax2.set_xlim(0, 1)
    
    # Panel 3: Stress Distribution
    plot_stress_distribution(beam, ax=ax3)
    
    # Panel 4: Deflection
    plot_deflection_curve(beam, ax=ax4)
    
    # Panel 5: Safety Factor
    plot_safety_factor_distribution(beam, ax=ax5)
    
    # Panel 6: Beam Schematic (ASCII art turned into plot)
    ax6.set_xlim(-0.5, beam.L + 0.5)
    ax6.set_ylim(-beam.cs.h * 3, beam.cs.h * 5)
    ax6.set_aspect('auto')
    ax6.set_title('Beam Schematic & Key Results')
    
    # Draw the deflected beam shape
    deflection_scaled = beam.deflection * 15  # Scale for visibility
    beam_bottom = -beam.cs.h / 2 + deflection_scaled
    beam_top = beam.cs.h / 2 + deflection_scaled
    
    ax6.fill_between(beam.x, beam_bottom, beam_top,
                      alpha=0.4, color='steelblue', label='Fin cross-section')
    ax6.plot(beam.x, deflection_scaled, 'r--', linewidth=2, label='Deflection (scaled)')
    
    # Add root wall
    wall = patches.Rectangle((-0.3, -beam.cs.h), 0.3, beam.cs.h*2,
                               linewidth=2, edgecolor='black', facecolor='gray')
    ax6.add_patch(wall)
    
    # Key results text
    sf = beam.material.safety_factor(beam.max_stress())
    results_text = (
        f"Max Stress: {beam.max_stress()/1e6:.1f} MPa\n"
        f"Safety Factor: {sf:.2f}\n"
        f"Tip Deflection: {beam.tip_deflection()*1000:.1f} mm\n"
        f"Natural Freq: {beam.natural_frequency():.2f} Hz\n"
        f"Status: {'✅ SAFE' if sf >= 1.5 else '❌ UNSAFE'}"
    )
    ax6.text(0.05, 0.95, results_text, transform=ax6.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax6.set_xlabel('Spanwise Position (m)')
    ax6.set_ylabel('Position (m)')
    ax6.legend(fontsize=8, loc='lower right')
    
    return fig


if __name__ == "__main__":
    # Test visualization with our beam
    import sys
    sys.path.insert(0, '..')
    
    from core.beam import Beam, BeamCrossSection
    from core.material import ALUMINUM_2024_T3
    
    print("Creating aircraft fin beam...")
    cs = BeamCrossSection(width=0.15, height=0.20)
    beam = Beam(length=4.0, cross_section=cs, material=ALUMINUM_2024_T3, n_stations=200)
    beam.apply_tip_load(50000)
    beam.apply_distributed_load(3000)
    
    beam.structural_summary()
    
    print("\nGenerating engineering dashboard...")
    fig = plot_full_engineering_dashboard(beam)
    plt.savefig('reports/structural_analysis_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Dashboard saved to reports/structural_analysis_dashboard.png")
    
    print("\nGenerating cross-section stress profiles...")
    fig2 = plot_cross_section_stress_profile(beam, x_positions=[0.0, 0.25, 0.5, 0.75])
    plt.savefig('reports/cross_section_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Cross-section profiles saved.")