# Streamlit dashboard application
# dashboard/app.py
"""
AI-Assisted Aircraft Hydraulic Safety Dashboard
=================================================
Interactive Streamlit web application for real-time
structural and hydraulic safety analysis.

Run with:
    streamlit run dashboard/app.py

From the project root directory.
"""

import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Hydraulic Safety System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.8;
    }
    .safe-badge {
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .warning-badge {
        background: #ffc107;
        color: black;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .danger-badge {
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e3a5f;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ────────────────────────────────────────────────
# @st.cache_resource ensures models/data load once, not on every slider move

@st.cache_resource
def load_physics_modules():
    """Load all physics modules once."""
    from core.beam import Beam, BeamCrossSection
    from core.material import (ALUMINUM_2024_T3, ALUMINUM_7075_T6,
                                 TITANIUM_6AL_4V)
    from core.hydraulic_line import (HydraulicLine, HydraulicLineGeometry,
                                      TITANIUM_TUBE)
    from core.hydraulic_configuration import HydraulicConfiguration
    return {
        'Beam': Beam,
        'BeamCrossSection': BeamCrossSection,
        'ALUMINUM_2024_T3': ALUMINUM_2024_T3,
        'ALUMINUM_7075_T6': ALUMINUM_7075_T6,
        'TITANIUM_6AL_4V':  TITANIUM_6AL_4V,
        'HydraulicLine':    HydraulicLine,
        'HydraulicLineGeometry': HydraulicLineGeometry,
        'TITANIUM_TUBE':    TITANIUM_TUBE,
        'HydraulicConfiguration': HydraulicConfiguration,
    }


@st.cache_data
def load_monte_carlo_data():
    """Load Monte Carlo CSV results."""
    # The dangerous-config CSV contains the SAME 2,000 flights, so using both
    # files would double-count every flight. Load the standard run only.
    paths = [
        'data/mc_standard_config.csv',
    ]
    dfs = []
    for path in paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


@st.cache_resource
def load_risk_model():
    """Load the trained failure-risk models (None if they are missing)."""
    try:
        from ml.train_model import FailureRiskModel
        return FailureRiskModel('data')
    except Exception:
        return None


@st.cache_data
def load_ml_metrics():
    """Held-out test metrics written by the training script."""
    import json
    try:
        with open('data/ml_metrics.json') as fh:
            return json.load(fh)
    except Exception:
        return None


def build_beam(modules, beam_length, cross_section_height, material_name):
    """Build beam from slider values."""
    Beam             = modules['Beam']
    BeamCrossSection = modules['BeamCrossSection']
    material_map = {
        'Aluminum 2024-T3': modules['ALUMINUM_2024_T3'],
        'Aluminum 7075-T6': modules['ALUMINUM_7075_T6'],
        'Titanium Ti-6Al-4V': modules['TITANIUM_6AL_4V'],
    }
    material = material_map[material_name]
    cs   = BeamCrossSection(width=0.15, height=cross_section_height)
    beam = Beam(length=beam_length, cross_section=cs,
                material=material, n_stations=100)
    return beam, material


def build_hydraulic_config(modules, beam, y_positions, pressure, wall_mm=1.0):
    """Build hydraulic configuration from slider values."""
    HydraulicLine         = modules['HydraulicLine']
    HydraulicLineGeometry = modules['HydraulicLineGeometry']
    TITANIUM_TUBE         = modules['TITANIUM_TUBE']
    HydraulicConfiguration = modules['HydraulicConfiguration']

    geom = HydraulicLineGeometry(outer_diameter=0.012,
                                  wall_thickness=wall_mm / 1000)
    lines = []
    systems = ['System_A', 'System_B', 'System_C']
    for i, y_pos in enumerate(y_positions):
        line = HydraulicLine(
            line_id=f'HYD-{i+1}',
            geometry=geom,
            material=TITANIUM_TUBE,
            x_start=0.2,
            x_end=beam.L * 0.9,
            y_position=y_pos,
            operating_pressure=pressure,
            system_name=systems[i],
        )
        lines.append(line)
    return HydraulicConfiguration('DASHBOARD_CONFIG', lines)


# ── Plotting functions ─────────────────────────────────────────────────────

def plot_stress_distribution(beam):
    """Compact stress distribution plot for dashboard."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0e1117')

    x_norm = beam.x / beam.L

    # Panel 1: Stress along span
    ax1 = axes[0]
    ax1.set_facecolor('#1e2030')
    stress_mpa = np.abs(beam.stress_top) / 1e6
    ax1.fill_between(x_norm, 0, stress_mpa, alpha=0.4, color='#ff6b6b')
    ax1.plot(x_norm, stress_mpa, color='#ff6b6b', linewidth=2)
    ax1.axhline(beam.material.yield_strength/1e6, color='white',
                 linewidth=1.5, linestyle='--',
                 label=f'Yield ({beam.material.yield_strength/1e6:.0f} MPa)')
    ax1.axhline(beam.material.yield_strength/1e6/1.5, color='#ffd700',
                 linewidth=1.2, linestyle=':', label='Design limit (SF=1.5)')
    ax1.set_xlabel('Span fraction', color='white')
    ax1.set_ylabel('Stress (MPa)', color='white')
    ax1.set_title('Bending Stress Along Span', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')
    ax1.set_xlim(0, 1)
    for spine in ax1.spines.values():
        spine.set_color('#444')

    # Panel 2: Deflection curve
    ax2 = axes[1]
    ax2.set_facecolor('#1e2030')
    defl_mm = beam.deflection * 1000
    ax2.fill_between(beam.x, 0, defl_mm, alpha=0.3, color='#4ecdc4')
    ax2.plot(beam.x, defl_mm, color='#4ecdc4', linewidth=2)
    ax2.annotate(f'Tip: {defl_mm[-1]:.1f} mm',
                  xy=(beam.L, defl_mm[-1]),
                  xytext=(beam.L*0.7, defl_mm[-1]*0.8),
                  fontsize=9, color='#4ecdc4', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#4ecdc4'))
    ax2.set_xlabel('Span position (m)', color='white')
    ax2.set_ylabel('Deflection (mm)', color='white')
    ax2.set_title('Beam Deflection Curve', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')

    plt.tight_layout()
    return fig


def plot_hydraulic_config(beam, config, analysis):
    """Hydraulic configuration visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')

    h = beam.cs.h

    # Panel 1: Configuration view
    ax1 = axes[0]
    ax1.set_facecolor('#1e2030')

    # Stress gradient background
    y_range = np.linspace(-h/2, h/2, 80)
    idx = len(beam.x)//2
    M   = beam.moment[idx]
    I   = beam.cs.second_moment_of_area
    for k in range(len(y_range)-1):
        frac  = abs(M * y_range[k] / I) / max(beam.max_stress(), 1e-6)
        color = plt.cm.RdYlGn_r(frac)
        ax1.barh(y_range[k]*1000, beam.L, height=(h/80)*1000,
                  left=0, color=color, alpha=0.5)

    ax1.axhline(0, color='#00ff88', linewidth=2.5, linestyle='--',
                 label='Neutral Axis (safest)', zorder=5)

    colors_status = {
        'OPTIMAL': '#00cc55', 'SAFE': '#88cc00',
        'MARGINAL': '#ffc107', 'UNSAFE': '#ff6600', 'CRITICAL': '#cc0000'
    }

    for r in analysis['line_results']:
        y_mm  = r['y_position'] * 1000
        color = colors_status.get(r['status'], '#4ecdc4')
        rect  = patches.FancyBboxPatch(
            (r['x_start'], y_mm - 8),
            r['x_end'] - r['x_start'], 16,
            boxstyle='round,pad=0.5',
            linewidth=2, edgecolor='white',
            facecolor=color, alpha=0.9, zorder=10
        )
        ax1.add_patch(rect)
        ax1.text(r['x_center'], y_mm + 14,
                  f"SF={r['sf_overall']:.2f}",
                  fontsize=8, ha='center', color='white',
                  fontweight='bold', zorder=11)

    ax1.set_xlim(0, beam.L)
    ax1.set_ylim(-h/2*1000*1.5, h/2*1000*1.5)
    ax1.set_xlabel('Spanwise Position (m)', color='white')
    ax1.set_ylabel('Position from Neutral Axis (mm)', color='white')
    ax1.set_title('Hydraulic Line Configuration\n(Red=dangerous, Green=safe)',
                   color='white')
    ax1.tick_params(colors='white')
    ax1.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')

    # Panel 2: Safety factor bars per line
    ax2 = axes[1]
    ax2.set_facecolor('#1e2030')

    line_ids = [r['line_id'] for r in analysis['line_results']]
    sfs      = [r['sf_overall'] for r in analysis['line_results']]
    vms      = [r['sigma_von_mises_mpa'] for r in analysis['line_results']]
    bar_colors = [colors_status.get(r['status'], '#4ecdc4')
                   for r in analysis['line_results']]

    bars = ax2.bar(line_ids, sfs, color=bar_colors, alpha=0.85,
                    edgecolor='white', linewidth=1.2)
    ax2.axhline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Minimum SF = 1.5')
    ax2.axhline(3.0, color='#00cc55', linewidth=1.5, linestyle='-.',
                 label='Excellent SF = 3.0')

    for bar, sf, vm in zip(bars, sfs, vms):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f'SF={sf:.2f}\n{vm:.0f}MPa',
                  ha='center', fontsize=8, color='white', fontweight='bold')

    ax2.set_ylabel('Safety Factor', color='white')
    ax2.set_title('Per-Line Safety Factors & Von Mises Stress',
                   color='white')
    ax2.tick_params(colors='white')
    ax2.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')
    ax2.set_ylim(0, max(sfs)*1.3 + 1)
    for spine in ax2.spines.values():
        spine.set_color('#444')

    plt.tight_layout()
    return fig


def plot_mc_summary(df):
    """Monte Carlo results summary chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0e1117')

    for ax in axes:
        ax.set_facecolor('#1e2030')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    # Panel 1: Load distribution
    ax1 = axes[0]
    ax1.hist(df['peak_load_kN'], bins=40, color='#4ecdc4',
              alpha=0.7, edgecolor='white', linewidth=0.3)
    ax1.axvline(df['peak_load_kN'].quantile(0.95), color='orange',
                 linewidth=2, linestyle='--',
                 label=f"95th: {df['peak_load_kN'].quantile(0.95):.0f} kN")
    ax1.set_xlabel('Peak Load (kN)', color='white')
    ax1.set_ylabel('Count', color='white')
    ax1.set_title('Load Distribution', color='white')
    ax1.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')

    # Panel 2: SF distribution
    ax2 = axes[1]
    ax2.hist(df['beam_safety_factor'], bins=40, color='#ff6b6b',
              alpha=0.7, edgecolor='white', linewidth=0.3)
    ax2.axvline(1.5, color='red', linewidth=2, linestyle='--',
                 label='Min SF = 1.5')
    ax2.set_xlabel('Beam Safety Factor', color='white')
    ax2.set_ylabel('Count', color='white')
    ax2.set_title('Safety Factor Distribution', color='white')
    ax2.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')

    # Panel 3: Scenario failure rates
    ax3 = axes[2]
    scen_fail = df.groupby('scenario_name')['any_failure'].mean().sort_values()
    colors    = ['#cc0000' if v > 0.01 else '#ffc107' if v > 0.001
                  else '#00cc55' for v in scen_fail.values]
    ax3.barh([s.replace('_', '\n') for s in scen_fail.index],
              scen_fail.values, color=colors, alpha=0.8, edgecolor='white')
    ax3.set_xlabel('Failure Rate', color='white')
    ax3.set_title('Failure Rate by Scenario', color='white')
    ax3.axvline(1e-4, color='gray', linewidth=1.5, linestyle='--',
                 label='FAA target')
    ax3.legend(fontsize=7, facecolor='#1e2030', labelcolor='white')

    plt.tight_layout()
    return fig


def get_safety_badge(sf):
    """Return HTML badge based on safety factor."""
    if sf >= 3.0:
        return '<span class="safe-badge">OPTIMAL</span>'
    elif sf >= 2.0:
        return '<span class="safe-badge">SAFE</span>'
    elif sf >= 1.5:
        return '<span class="warning-badge">MARGINAL</span>'
    else:
        return '<span class="danger-badge">UNSAFE</span>'


PSI_OPTIONS = ['3000 PSI (Commercial)', '5000 PSI (Military)']

# Every value is a widget key, so choosing a preset sets all the controls.
PRESETS = {
    "Default (standard layout)": dict(
        tip_kn=50.0, dist_kn=3.0, turb=3.0, material='Aluminum 2024-T3',
        span=4.0, spar_cm=20.0, y1_mm=0.0, y2_mm=50.0, y3_mm=-50.0,
        psi=PSI_OPTIONS[0], wall_mm=1.0),
    "Dangerous layout (tubes near the surface)": dict(
        tip_kn=50.0, dist_kn=3.0, turb=3.0, material='Aluminum 2024-T3',
        span=4.0, spar_cm=20.0, y1_mm=84.0, y2_mm=76.0, y3_mm=-80.0,
        psi=PSI_OPTIONS[0], wall_mm=1.0),
    "Stress test (96 kN, 5000 PSI, 0.7 mm wall)": dict(
        tip_kn=96.0, dist_kn=3.0, turb=9.0, material='Aluminum 2024-T3',
        span=4.0, spar_cm=20.0, y1_mm=0.0, y2_mm=80.0, y3_mm=-80.0,
        psi=PSI_OPTIONS[1], wall_mm=0.7),
}


def apply_preset():
    """on_change callback: write the chosen preset into the widget keys."""
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


def compute_verdict(beam_sf, min_line_sf, analysis):
    """Return (label, colour, sentence). The first failing rule decides."""
    if not analysis['no_overlap']:
        return ("DO NOT FLY", "#dc3545",
                "Hydraulic tubes physically overlap, so this layout cannot be built.")
    if beam_sf < 1.0:
        return ("DO NOT FLY", "#dc3545",
                f"The fin spar exceeds its yield strength (safety factor {beam_sf:.2f}).")
    if min_line_sf < 1.0:
        return ("DO NOT FLY", "#dc3545",
                f"The weakest tube exceeds its yield strength (safety factor {min_line_sf:.2f}).")
    if beam_sf < 1.5:
        return ("REVIEW", "#ffc107",
                f"Fin spar safety factor {beam_sf:.2f} is below the 1.5 aerospace minimum.")
    if min_line_sf < 1.5:
        return ("REVIEW", "#ffc107",
                f"Weakest tube safety factor {min_line_sf:.2f} is below the 1.5 minimum.")
    if not analysis['separation_ok']:
        return ("REVIEW", "#ffc107", "Two tubes are closer than the 20 mm separation limit.")
    if not analysis['redundancy_ok']:
        return ("REVIEW", "#ffc107",
                "Redundant hydraulic systems are closer than the 30 mm spacing target.")
    return ("SAFE", "#28a745",
            f"Thinnest margin: fin spar SF {beam_sf:.2f}; weakest tube SF {min_line_sf:.2f}.")


def plot_sf_vs_position(modules, beam, pressure, wall_mm, analysis):
    """Tube safety factor as a function of where the tube sits in the spar."""
    HydraulicLine         = modules['HydraulicLine']
    HydraulicLineGeometry = modules['HydraulicLineGeometry']
    TITANIUM_TUBE         = modules['TITANIUM_TUBE']
    geom = HydraulicLineGeometry(outer_diameter=0.012,
                                 wall_thickness=wall_mm / 1000)
    h = beam.cs.h
    ys = np.linspace(-h / 2, h / 2, 81)
    sfs = [HydraulicLine('probe', geom, TITANIUM_TUBE, 0.2, beam.L * 0.9,
                         float(y), pressure, 'probe').compute_safety_factor(beam)
           for y in ys]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e2030')
    ax.plot(ys * 1000, sfs, color='#4ecdc4', linewidth=2.5,
            label='Tube safety factor at this position')
    ax.axhline(1.5, color='red', linewidth=1.8, linestyle='--', label='Minimum SF = 1.5')
    ax.axhline(3.0, color='#00cc55', linewidth=1.5, linestyle='-.', label='Excellent SF = 3.0')
    for r, c in zip(analysis['line_results'], ['#4ecdc4', '#ffc107', '#a78bfa']):
        ax.scatter([r['y_position'] * 1000], [r['sf_overall']], s=110, color=c,
                   edgecolor='white', zorder=5)
        ax.annotate(f"{r['system_name'].replace('System_', 'Sys ')}  SF {r['sf_overall']:.2f}",
                    (r['y_position'] * 1000, r['sf_overall']),
                    textcoords='offset points', xytext=(0, 12), ha='center',
                    color='white', fontsize=8, fontweight='bold')
    ax.set_xlabel('Tube y-position from neutral axis (mm)', color='white')
    ax.set_ylabel('Tube safety factor', color='white')
    ax.set_title('Does placement matter at this load, pressure and wall thickness?',
                 color='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, max(max(sfs), 3.4) * 1.15)
    ax.legend(fontsize=8, facecolor='#1e2030', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    plt.tight_layout()
    return fig


# ── MAIN DASHBOARD ─────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <h1 style='text-align:center; color:#4ecdc4; font-size:2.2em;'>
        ✈️ AI-Assisted Aircraft Hydraulic Safety System
    </h1>
    <p style='text-align:center; color:#aaa; font-size:1.0em;'>
        Real-time structural analysis | Hydraulic safety evaluation |
        Monte Carlo statistics | ML failure prediction
    </p>
    <hr style='border-color:#333;'>
    """, unsafe_allow_html=True)

    # Load modules
    with st.spinner('Loading physics engine...'):
        modules = load_physics_modules()
        df_mc   = load_monte_carlo_data()
        risk_model = load_risk_model()

    # ── SIDEBAR ────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Simulation Controls")
    st.sidebar.selectbox(
        "Load preset", list(PRESETS), key="preset", on_change=apply_preset,
        help="Sets every control below. You can still change them afterwards."
    )
    st.sidebar.markdown("---")

    # Flight conditions
    st.sidebar.markdown("### ✈️ Flight Conditions")
    tip_load = st.sidebar.slider(
        "Tip Load (kN)", 5.0, 150.0, 50.0, 1.0, key="tip_kn",
        help="Lateral aerodynamic force on fin tip"
    ) * 1000  # Convert to N

    dist_load = st.sidebar.slider(
        "Distributed Load (kN/m)", 0.0, 10.0, 3.0, 0.5, key="dist_kn",
        help="Aerodynamic pressure distributed along fin"
    ) * 1000

    turbulence = st.sidebar.slider(
        "Turbulence Intensity (m/s)", 0.5, 15.0, 3.0, 0.5, key="turb",
        help="RMS turbulence velocity (an input to the ML risk model)"
    )

    st.sidebar.markdown("---")

    # Structural parameters
    st.sidebar.markdown("### 🏗️ Structural Parameters")
    material_name = st.sidebar.selectbox(
        "Material",
        ['Aluminum 2024-T3', 'Aluminum 7075-T6', 'Titanium Ti-6Al-4V'],
        key="material", help="Fin spar material"
    )

    beam_length = st.sidebar.slider(
        "Fin Span (m)", 2.0, 8.0, 4.0, 0.5, key="span",
        help="Height of the vertical fin"
    )

    section_height = st.sidebar.slider(
        "Spar Height (cm)", 10.0, 40.0, 20.0, 2.0, key="spar_cm",
        help="Cross-section height of the fin spar"
    ) / 100  # Convert to meters

    st.sidebar.markdown("---")

    # Hydraulic configuration
    st.sidebar.markdown("### 🔧 Hydraulic Line Positions")
    st.sidebar.caption(
        "0 = Neutral axis (safest) | ±(half spar height) = Surface (dangerous)"
    )

    h_half_mm = section_height / 2 * 1000
    for k in ("y1_mm", "y2_mm", "y3_mm"):      # keep values inside the new range
        if k in st.session_state:
            st.session_state[k] = float(min(max(st.session_state[k], -h_half_mm), h_half_mm))
    y1 = st.sidebar.slider(
        "Line 1 y-position (mm)", -h_half_mm, h_half_mm, 0.0, 1.0, key="y1_mm",
        help="System A: primary flight controls"
    ) / 1000

    y2 = st.sidebar.slider(
        "Line 2 y-position (mm)", -h_half_mm, h_half_mm,
        float(h_half_mm * 0.5), 1.0, key="y2_mm",
        help="System B: secondary controls"
    ) / 1000

    y3 = st.sidebar.slider(
        "Line 3 y-position (mm)", -h_half_mm, h_half_mm,
        float(-h_half_mm * 0.5), 1.0, key="y3_mm",
        help="System C: tertiary systems"
    ) / 1000

    pressure_psi = st.sidebar.selectbox(
        "Hydraulic Pressure", PSI_OPTIONS, key="psi",
    )
    pressure = 20.7e6 if '3000' in pressure_psi else 34.5e6

    wall_mm = st.sidebar.slider(
        "Tube Wall Thickness (mm)", 0.5, 2.0, 1.0, 0.1, key="wall_mm",
        help="12 mm outer diameter tube. Thinner wall = higher pressure stress"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "This dashboard simulates aircraft hydraulic line safety "
        "using beam theory, Monte Carlo analysis, genetic algorithm "
        "optimization, and a machine-learning surrogate model."
    )

    # ── RUN SIMULATION ─────────────────────────────────────────────────
    beam, material = build_beam(
        modules, beam_length, section_height, material_name
    )
    beam.apply_tip_load(tip_load)
    beam.apply_distributed_load(dist_load)

    hyd_config = build_hydraulic_config(
        modules, beam, [y1, y2, y3], pressure, wall_mm
    )
    analysis = hyd_config.analyze(beam)

    max_stress  = beam.max_stress()
    beam_sf     = material.safety_factor(max_stress)
    tip_defl    = beam.tip_deflection() * 1000
    min_line_sf = analysis['min_safety_factor']
    config_score = analysis['config_score']

    # ── TOP METRICS ROW ─────────────────────────────────────────────────
    st.markdown("### 📊 Live Safety Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "#28a745" if beam_sf >= 1.5 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <div class="metric-label">Beam Safety Factor</div>
            <div class="metric-value">{beam_sf:.2f}</div>
            <div>{"✅ SAFE" if beam_sf >= 1.5 else "❌ UNSAFE"}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #ff6b6b;">
            <div class="metric-label">Max Stress</div>
            <div class="metric-value">{max_stress/1e6:.0f}</div>
            <div style="font-size:11px;">MPa (yield: {material.yield_strength/1e6:.0f} MPa)</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        color2 = "#28a745" if min_line_sf >= 1.5 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color2};">
            <div class="metric-label">Min Line SF</div>
            <div class="metric-value">{min_line_sf:.2f}</div>
            <div>{"✅ SAFE" if min_line_sf >= 1.5 else "❌ UNSAFE"}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #4ecdc4;">
            <div class="metric-label">Config Score</div>
            <div class="metric-value">{config_score:.0f}</div>
            <div style="font-size:11px;">out of 100</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #ffd700;">
            <div class="metric-label">Tip Deflection</div>
            <div class="metric-value">{tip_defl:.1f}</div>
            <div style="font-size:11px;">mm ({tip_defl/beam_length/10:.2f}% span)</div>
        </div>""", unsafe_allow_html=True)

    verdict, vcolor, vtext = compute_verdict(beam_sf, min_line_sf, analysis)
    st.markdown(f"""
    <div style="margin-top:12px; padding:12px 18px; border-radius:10px;
                border-left:6px solid {vcolor}; background:#161b2e;">
        <span style="color:{vcolor}; font-weight:800; font-size:18px;">{verdict}</span>
        <span style="color:#cfd8e3; margin-left:12px;">{vtext}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Structural Analysis",
        "🔧 Hydraulic Safety",
        "📊 Monte Carlo Stats",
        "🤖 ML Prediction",
    ])

    # ── Tab 1: Structural Analysis ─────────────────────────────────────
    with tab1:
        st.markdown("#### Structural Analysis Results")

        col_l, col_r = st.columns([2, 1])

        with col_l:
            fig_stress = plot_stress_distribution(beam)
            st.pyplot(fig_stress, width="stretch")
            plt.close('all')

        with col_r:
            st.markdown("**Structural Summary**")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Material | {material.name} |
            | Span | {beam_length:.1f} m |
            | Max Stress | {max_stress/1e6:.1f} MPa |
            | Yield Strength | {material.yield_strength/1e6:.0f} MPa |
            | Safety Factor | **{beam_sf:.2f}** |
            | Tip Deflection | {tip_defl:.1f} mm |
            | Natural Freq | {beam.natural_frequency():.2f} Hz |
            | Status | {get_safety_badge(beam_sf)} |
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Load Applied**")
            st.metric("Tip Load", f"{tip_load/1000:.0f} kN")
            st.metric("Distributed Load", f"{dist_load/1000:.1f} kN/m")
            st.metric("Root Moment", f"{beam.moment[0]/1000:.1f} kN·m")

    # ── Tab 2: Hydraulic Safety ────────────────────────────────────────
    with tab2:
        st.markdown("#### Hydraulic Line Safety Analysis")

        fig_hyd = plot_hydraulic_config(beam, hyd_config, analysis)
        st.pyplot(fig_hyd, width="stretch")
        plt.close('all')

        fig_pos = plot_sf_vs_position(modules, beam, pressure, wall_mm, analysis)
        st.pyplot(fig_pos, width="stretch")
        plt.close('all')
        st.caption(
            "At light loads every position is safe. Try the **Stress test** preset "
            "to see where placement starts to cut the margin."
        )

        st.markdown("**Per-Line Safety Report**")
        line_data = []
        for r in analysis['line_results']:
            line_data.append({
                'Line ID': r['line_id'],
                'System': r['system_name'],
                'y-pos (mm)': f"{r['y_position']*1000:.1f}",
                'Von Mises (MPa)': f"{r['sigma_von_mises_mpa']:.1f}",
                'Safety Factor': f"{r['sf_overall']:.2f}",
                'Fatigue Flights': (
                    '∞' if r['fatigue_flights'] > 1e6
                    else f"{r['fatigue_flights']:.0f}"
                ),
                'Status': r['status'],
            })

        df_lines = pd.DataFrame(line_data)
        st.dataframe(df_lines, width="stretch", hide_index=True)

        col_checks = st.columns(3)
        with col_checks[0]:
            status = "✅ No Overlap" if analysis['no_overlap'] else "❌ Overlap!"
            st.metric("Physical Check 1", status)
        with col_checks[1]:
            status = "✅ Separated" if analysis['separation_ok'] else "❌ Too Close!"
            st.metric("Physical Check 2", status)
        with col_checks[2]:
            status = "✅ Redundancy OK" if analysis['redundancy_ok'] else "❌ Poor Redundancy!"
            st.metric("Physical Check 3", status)

    # ── Tab 3: Monte Carlo ─────────────────────────────────────────────
    with tab3:
        st.markdown("#### Monte Carlo Simulation Statistics")
        st.caption("Based on pre-computed simulation data (2000+ flights)")

        if df_mc is not None:
            fig_mc = plot_mc_summary(df_mc)
            st.pyplot(fig_mc, width="stretch")
            plt.close('all')

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total Flights", f"{len(df_mc):,}")
            col_s2.metric("Mean Peak Load",
                           f"{df_mc['peak_load_kN'].mean():.1f} kN")
            col_s3.metric("P(failure)",
                           f"{df_mc['any_failure'].mean():.4f}")
            col_s4.metric("95th Pct Load",
                           f"{df_mc['peak_load_kN'].quantile(0.95):.0f} kN")

            st.markdown("**Raw Data Preview**")
            display_cols = ['scenario_name', 'peak_load_kN',
                             'beam_safety_factor', 'min_line_sf',
                             'config_score', 'any_failure']
            st.dataframe(
                df_mc[display_cols].head(20),
                width="stretch",
                hide_index=True
            )
        else:
            st.warning(
                "Monte Carlo data not found. "
                "Run `python main.py 5` first to generate data."
            )

    # ── Tab 4: ML Prediction ───────────────────────────────────────────
    with tab4:
        st.markdown("#### Machine Learning Failure-Risk Surrogate")
        st.info(
            "Surrogate model trained on simulated data; for illustration. "
            "Given the load, spar height and material, the physics is deterministic "
            "(the spar either yields or it does not), so this model is a fast "
            "approximation of that rule. Its probability reflects uncertainty near "
            "the failure boundary, not a measured flight-failure rate."
        )

        if risk_model is None:
            st.warning(
                "ML models not found. Run `python main.py 10` then "
                "`python main.py 9` to build the dataset and train them."
            )
        else:
            from ml.features import build_feature_row
            peak_kn = tip_load / 1000 + dist_load / 1000 * beam_length
            row = build_feature_row(
                peak_load_kN=peak_kn, turbulence_intensity=turbulence,
                spar_height_m=section_height, line_y_m=[y1, y2, y3],
                pressure_mpa=pressure / 1e6, wall_thickness_mm=wall_mm,
            )
            pred = risk_model.predict(row)
            issues = risk_model.out_of_range(row)
            if material_name != risk_model.meta['material'] or \
                    abs(beam_length - risk_model.meta['span_m']) > 1e-9:
                issues.insert(0, f"material / span (trained only for "
                                 f"{risk_model.meta['material']}, "
                                 f"{risk_model.meta['span_m']:.0f} m span)")
            if issues:
                st.warning("Outside the range this model was trained on, so treat the "
                           "result as unreliable: " + "; ".join(issues))

            ensemble = pred['ensemble']
            level = pred['level']
            risk_color = {'CRITICAL': '#cc0000', 'HIGH': '#ff6600',
                          'MODERATE': '#ffc107', 'LOW': '#00cc55'}
            st.markdown(f"""
            <div style='text-align:center; padding:20px;
                         background:linear-gradient(135deg,#1e2030,#2d3250);
                         border-radius:15px; border: 2px solid {risk_color[level]};'>
                <h2 style='color:{risk_color[level]};'>{level} RISK</h2>
                <p style='color:#aaa;'>Calibrated probability that the spar or a tube exceeds yield (SF below 1)</p>
                <h1 style='color:white;'>{ensemble*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Random Forest score", f"{pred['rf']*100:.2f}%")
            c2.metric("XGBoost score", f"{pred['xgb']*100:.2f}%")
            c3.metric("Calibrated ensemble", f"{ensemble*100:.2f}%")
            st.caption(
                f"Inputs sent to the model: total load {peak_kn:.1f} kN "
                f"(tip + distributed x span), turbulence {turbulence:.1f} m/s, "
                f"spar height {section_height*100:.0f} cm, tube positions, "
                f"{pressure/1e6:.1f} MPa, {wall_mm:.1f} mm wall. "
                "The two raw scores are not calibrated; the ensemble is."
            )

            st.markdown("**Risk levels:** LOW below 1%, MODERATE 1 to 10%, "
                        "HIGH 10 to 50%, CRITICAL above 50%.")

            metrics = load_ml_metrics()
            if metrics:
                with st.expander("How good is this model? (held-out test set)"):
                    rows = []
                    for name, m in metrics['test_set'].items():
                        rows.append({'Model': name, 'AUC': m['auc_roc'],
                                     'Avg precision': m['avg_precision'],
                                     'Precision': m['precision'],
                                     'Recall': m['recall'], 'F1': m['f1']})
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    st.markdown(
                        f"{metrics['summary']} Test set: {metrics['test_rows']:,} "
                        f"simulated flights, {metrics['test_failures']} failures. "
                        "In this data every failure was the fin spar yielding; no tube "
                        "failed first, so tube position barely changes the prediction."
                    )

            st.markdown("**What the model relies on:**")
            importances = risk_model.rf.feature_importances_
            order = np.argsort(importances)
            fig_imp, ax = plt.subplots(figsize=(8, 4))
            fig_imp.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2030')
            ax.barh([risk_model.feature_names[i] for i in order],
                    importances[order], color='#4ecdc4', alpha=0.8)
            ax.set_xlabel('Importance', color='white')
            ax.set_title('Random Forest feature importance', color='white')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig_imp, width="stretch")
            plt.close('all')

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:0.85em;'>"
        "AI-Assisted Aircraft Hydraulic Safety System | "
        "B.Tech Computer Science Project | "
        "Built with Python, Streamlit, scikit-learn, XGBoost"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()