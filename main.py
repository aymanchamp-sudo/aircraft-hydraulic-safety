import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

def open_image(path):
    """Open a saved image in Windows default viewer."""
    import subprocess
    subprocess.Popen(["start", path], shell=True)

def run_phase2():
    print("\n" + "="*60)
    print("  PHASE 2 - STRUCTURAL SIMULATION")
    print("="*60)
    from core.material import ALUMINUM_2024_T3, ALUMINUM_7075_T6, TITANIUM_6AL_4V
    from core.beam import Beam, BeamCrossSection
    from core.stress_calculator import find_safe_zones, compute_hydraulic_line_stress
    from visualization.stress_plots import plot_full_engineering_dashboard, plot_cross_section_stress_profile
    cs = BeamCrossSection(width=0.15, height=0.20)
    print(cs)
    beam = Beam(length=4.0, cross_section=cs, material=ALUMINUM_2024_T3, n_stations=200)
    beam.apply_tip_load(50000)
    beam.apply_distributed_load(3000)
    beam.structural_summary()
    safe = find_safe_zones(beam, safety_threshold=1.5)
    print(f"\n  Safe fraction of span: {safe['safe_fraction']*100:.1f}%")
    test_positions = [
        (0.5, 0.0,  "Root, Neutral Axis (best)"),
        (0.5, 0.05, "Root, Quarter-height"),
        (0.5, 0.09, "Root, Near surface (dangerous)"),
        (2.0, 0.0,  "Mid-span, Neutral Axis"),
        (2.0, 0.08, "Mid-span, Near surface"),
        (3.5, 0.0,  "Near tip, Neutral Axis"),
    ]
    print(f"\n  {'Position':<35} {'Stress':>10} {'SF':>8} {'Status':>12}")
    print("  " + "-"*68)
    for x_pos, y_pos, label in test_positions:
        r = compute_hydraulic_line_stress(beam, x_pos, y_pos)
        print(f"  {label:<35} {r['stress_mpa']:>8.1f} MPa {r['safety_factor']:>6.2f}  {r['danger_level']:>10}")
    print("\n  Material comparison:")
    for mat in [ALUMINUM_2024_T3, ALUMINUM_7075_T6, TITANIUM_6AL_4V]:
        b = Beam(4.0, cs, mat, n_stations=200)
        b.apply_tip_load(50000)
        b.apply_distributed_load(3000)
        sf = mat.safety_factor(b.max_stress())
        print(f"    {mat.name:<25} SF={sf:.2f}  delta={b.tip_deflection()*1000:.1f}mm  {'OK' if sf>=1.5 else 'FAIL'}")
    fig1 = plot_full_engineering_dashboard(beam)
    p1 = "reports/phase2_structural_dashboard.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close("all")
    fig2 = plot_cross_section_stress_profile(beam, x_positions=[0.0,0.25,0.5,0.75])
    p2 = "reports/phase2_cross_section_profiles.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {p1}")
    print(f"  Saved: {p2}")
    open_image(p1)
    open_image(p2)
    print("  (Images opened in Windows Photo Viewer)")
    print("\n  PHASE 2 COMPLETE")
    return beam

def run_phase3():
    print("\n" + "="*60)
    print("  PHASE 3 - HYDRAULIC SYSTEM MODELING")
    print("="*60)
    from core.material import ALUMINUM_2024_T3
    from core.beam import Beam, BeamCrossSection
    from core.hydraulic_line import HydraulicLine, HydraulicLineGeometry, TITANIUM_TUBE
    from core.hydraulic_configuration import create_standard_configuration, create_dangerous_configuration
    import matplotlib.patches as patches
    cs = BeamCrossSection(width=0.15, height=0.20)
    beam = Beam(4.0, cs, ALUMINUM_2024_T3, n_stations=200)
    beam.apply_tip_load(50000)
    beam.apply_distributed_load(3000)
    beam.structural_summary()
    std_geom = HydraulicLineGeometry(outer_diameter=0.012, wall_thickness=0.001)
    test_lines = [
        HydraulicLine("HYD-NA",      std_geom, TITANIUM_TUBE, 0.2, 3.5, 0.0,   20.7e6, "Sys_A"),
        HydraulicLine("HYD-25PCT",   std_geom, TITANIUM_TUBE, 0.2, 3.5, 0.025, 20.7e6, "Sys_B"),
        HydraulicLine("HYD-50PCT",   std_geom, TITANIUM_TUBE, 0.2, 3.5, 0.05,  20.7e6, "Sys_C"),
        HydraulicLine("HYD-75PCT",   std_geom, TITANIUM_TUBE, 0.2, 3.5, 0.075, 20.7e6, "Sys_A"),
        HydraulicLine("HYD-SURFACE", std_geom, TITANIUM_TUBE, 0.2, 3.5, 0.095, 20.7e6, "Sys_B"),
    ]
    print(f"\n  {'Line ID':<15} {'y-pos':>8} {'sigma_VM':>10} {'SF':>8} {'Status'}")
    print("  " + "-"*55)
    for line in test_lines:
        r = line.full_safety_analysis(beam)
        print(f"  {r['line_id']:<15} {r['y_position']*1000:>6.1f}mm {r['sigma_von_mises_mpa']:>9.1f} {r['sf_overall']:>8.2f} {r['status']}")
    std_config = create_standard_configuration(beam, pressure=20.7e6)
    bad_config  = create_dangerous_configuration(beam, pressure=20.7e6)
    std_analysis = std_config.analyze(beam)
    bad_analysis  = bad_config.analyze(beam)
    std_config.print_report(std_analysis)
    bad_config.print_report(bad_analysis)
    print(f"\n  Config Score  — Standard: {std_analysis['config_score']:.1f}  |  Dangerous: {bad_analysis['config_score']:.1f}")
    print(f"  Min SF        — Standard: {std_analysis['min_safety_factor']:.2f}  |  Dangerous: {bad_analysis['min_safety_factor']:.2f}")
    print(f"  Redundancy OK — Standard: {std_analysis['redundancy_ok']}  |  Dangerous: {bad_analysis['redundancy_ok']}")
    h = beam.cs.h
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 3 - Hydraulic Line Safety Analysis", fontweight="bold")
    for ax, analysis, title in zip(axes, [std_analysis, bad_analysis], ["Standard Config", "Dangerous Config"]):
        ax.set_title(f"{title}  |  Score: {analysis['config_score']:.1f}/100")
        ax.axhline(0, color="green", linewidth=2, linestyle="--", label="Neutral Axis")
        colors_map = {"OPTIMAL":"green","SAFE":"limegreen","MARGINAL":"orange","UNSAFE":"red","CRITICAL":"darkred"}
        for r in analysis["line_results"]:
            y_mm = r["y_position"]*1000
            rect = patches.FancyBboxPatch((r["x_start"], y_mm-6), r["x_end"]-r["x_start"], 12,
                boxstyle="round,pad=0.3", linewidth=2, edgecolor="black",
                facecolor=colors_map.get(r["status"],"blue"), alpha=0.8)
            ax.add_patch(rect)
            ax.text(r["x_center"], y_mm+10, f"SF={r['sf_overall']:.1f}", fontsize=7, ha="center")
        ax.set_xlim(0, beam.L)
        ax.set_ylim(-h/2*1000*1.4, h/2*1000*1.4)
        ax.set_xlabel("Spanwise Position (m)")
        ax.set_ylabel("Position from Neutral Axis (mm)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    p = "reports/phase3_hydraulic_analysis.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {p}")
    open_image(p)
    print("  (Image opened in Windows Photo Viewer)")
    print("  PHASE 3 COMPLETE")
    return beam, std_config, bad_config

def run_phase4():
    print("\n" + "="*60)
    print("  PHASE 4 - WIND & FLIGHT CONDITION SIMULATION")
    print("="*60)
    from simulation.wind_model import AerodynamicModel, TurbulenceModel, GustEventModel, FlightSimulation, CRUISE_CONDITIONS
    from simulation.flight_conditions import build_flight_scenarios, FlightConditionSampler
    from visualization.wind_plots import plot_flight_simulation, plot_load_statistics, plot_scenario_comparison
    aero = AerodynamicModel(fin_area=10.0, fin_span=4.0, lift_curve_slope=3.8, fin_efficiency=0.93)
    print(CRUISE_CONDITIONS)
    turb  = TurbulenceModel(turbulence_intensity=3.0, correlation_length=0.6, seed=42)
    gusts = GustEventModel(mean_gust_velocity=8.0, std_gust_velocity=4.0, mean_gusts_per_hour=5.0, seed=42)
    sim = FlightSimulation(aero, turb, gusts, CRUISE_CONDITIONS, flight_duration_hours=4.0, dt=2.0, seed=42)
    result = sim.run()
    print(f"\n  Peak load:  {result['peak_load']/1000:.2f} kN")
    print(f"  Gust count: {result['n_gusts']}")
    print(f"  RMS load:   {result['rms_load']/1000:.2f} kN")
    p1 = "reports/phase4_flight_simulation.png"
    fig1 = plot_flight_simulation(result, "Phase 4 - Flight Simulation")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close("all")
    scenarios = build_flight_scenarios()
    sampler   = FlightConditionSampler(aero, scenarios, seed=42)
    print("\n  Sampling 500 flights...")
    loads = np.array([sampler.sample_flight_load(4.0)["peak_load"] for _ in range(500)])
    print(f"  Mean: {np.mean(loads)/1000:.1f} kN  |  95th: {np.percentile(loads,95)/1000:.1f} kN  |  Max: {np.max(loads)/1000:.1f} kN")
    p2 = "reports/phase4_load_statistics.png"
    fig2 = plot_load_statistics(loads, "Phase 4 - Load Statistics")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close("all")
    p3 = "reports/phase4_scenario_comparison.png"
    fig3 = plot_scenario_comparison(sampler, n_samples=50)
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  Saved: {p1}")
    print(f"  Saved: {p2}")
    print(f"  Saved: {p3}")
    open_image(p1)
    open_image(p2)
    open_image(p3)
    print("  (Images opened in Windows Photo Viewer)")
    print("  PHASE 4 COMPLETE")
    return sampler, loads

def run_phase5():
    print("\n" + "="*60)
    print("  PHASE 5 - MONTE CARLO SIMULATION ENGINE")
    print("="*60)
    from core.beam import Beam, BeamCrossSection
    from core.material import ALUMINUM_2024_T3
    from core.hydraulic_configuration import create_standard_configuration, create_dangerous_configuration
    from simulation.monte_carlo import MonteCarloConfig, MonteCarloEngine, ConfigurationComparator
    print("\n  Building reference beam...")
    cs = BeamCrossSection(width=0.15, height=0.20)
    ref_beam = Beam(4.0, cs, ALUMINUM_2024_T3)
    ref_beam.apply_tip_load(50000)
    print("  Beam ready")
    std_config = create_standard_configuration(ref_beam, pressure=20.7e6)
    bad_config  = create_dangerous_configuration(ref_beam, pressure=20.7e6)
    print("  Hydraulic configurations ready")
    print("\n  Running Monte Carlo: STANDARD config (2000 iterations)...")
    mc_std = MonteCarloConfig(n_iterations=2000, seed=42, save_csv=True, csv_path="data/mc_standard_config.csv", verbose=True)
    result_std = MonteCarloEngine(mc_std, std_config).run()
    result_std.print_summary()
    print("\n  Running Monte Carlo: DANGEROUS config (2000 iterations)...")
    mc_bad = MonteCarloConfig(n_iterations=2000, seed=42, save_csv=True, csv_path="data/mc_dangerous_config.csv", verbose=True)
    result_bad = MonteCarloEngine(mc_bad, bad_config).run()
    result_bad.print_summary()
    comparator = ConfigurationComparator(mc_std)
    comparator.results["STANDARD_CONFIG"]  = result_std
    comparator.results["DANGEROUS_CONFIG"] = result_bad
    comp_df = comparator.print_comparison()
    comp_df.to_csv("data/configuration_comparison.csv", index=False)
    print("  Saved: data/configuration_comparison.csv")
    try:
        from visualization.monte_carlo_plots import plot_safety_distribution, plot_failure_analysis, plot_configuration_comparison, plot_convergence_analysis
        paths = []
        fig1 = plot_safety_distribution(result_std.dataframe, "STANDARD_CONFIG")
        p1 = "reports/phase5_safety_distribution.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(p1)
        fig2 = plot_failure_analysis(result_bad.dataframe)
        p2 = "reports/phase5_failure_analysis.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(p2)
        fig3 = plot_configuration_comparison(comp_df)
        p3 = "reports/phase5_config_comparison.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(p3)
        fig4 = plot_convergence_analysis(result_std)
        p4 = "reports/phase5_convergence.png"
        fig4.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(p4)
        for p in paths:
            print(f"  Saved: {p}")
            open_image(p)
        print("  (Images opened in Windows Photo Viewer)")
    except Exception as e:
        print(f"  Plot saving failed: {e}")
    print("\n" + "="*60)
    print("  PHASE 5 COMPLETE")
    print("="*60)
    return result_std, result_bad



def run_phase7():
    print("\n" + "="*60)
    print("  PHASE 7 - ADVANCED VISUALIZATION")
    print("="*60)

    import pandas as pd
    from core.beam import Beam, BeamCrossSection
    from core.material import ALUMINUM_2024_T3
    from core.hydraulic_line import HydraulicLine, HydraulicLineGeometry, TITANIUM_TUBE
    from visualization.beam_animation import animate_beam_loading, animate_stress_wave
    from visualization.interactive_plots import save_interactive_plots

    print("\n  Building beam model...")
    cs   = BeamCrossSection(width=0.15, height=0.20)
    beam = Beam(4.0, cs, ALUMINUM_2024_T3, n_stations=150)
    beam.apply_tip_load(50_000)
    beam.apply_distributed_load(3_000)

    std_geom = HydraulicLineGeometry(outer_diameter=0.012, wall_thickness=0.001)
    hyd_lines = [
        HydraulicLine("Sys-A", std_geom, TITANIUM_TUBE, 0.2, 3.8, 0.0,   20.7e6, "A"),
        HydraulicLine("Sys-B", std_geom, TITANIUM_TUBE, 0.2, 3.8, 0.05,  20.7e6, "B"),
        HydraulicLine("Sys-C", std_geom, TITANIUM_TUBE, 0.2, 3.8, -0.05, 20.7e6, "C"),
    ]

    print("\n  Creating beam loading animation...")
    print("  (Takes 30-60 seconds — generating 60 frames)")
    p1 = animate_beam_loading(
        beam_class=Beam,
        cross_section_class=BeamCrossSection,
        material=ALUMINUM_2024_T3,
        hydraulic_lines=hyd_lines,
        max_load=120_000,
        n_frames=60,
        output_path="reports/phase7_beam_loading.gif",
        fps=15
    )
    open_image(p1)

    print("\n  Creating stress wave animation...")
    print("  (Takes 20-40 seconds — generating 50 frames)")
    p2 = animate_stress_wave(
        beam_class=Beam,
        cross_section_class=BeamCrossSection,
        material=ALUMINUM_2024_T3,
        output_path="reports/phase7_stress_wave.gif",
        n_frames=50,
        fps=12
    )
    open_image(p2)

    print("\n  Creating interactive HTML plots...")
    df_std = pd.read_csv("data/mc_standard_config.csv")
    df_std["outcome"] = df_std["any_failure"].map({0: "SAFE", 1: "FAILURE"})

    paths = save_interactive_plots(beam, df_std, output_dir="reports")

    import subprocess
    for p in paths:
        subprocess.Popen(["start", p], shell=True)
        print(f"  Opened in browser: {p}")

    print("\n" + "="*60)
    print("  PHASE 7 COMPLETE")
    print()
    print("  Animations saved:")
    print("    reports/phase7_beam_loading.gif")
    print("    reports/phase7_stress_wave.gif")
    print()
    print("  Interactive HTML (open in browser):")
    print("    reports/phase7_3d_stress_field.html")
    print("    reports/phase7_monte_carlo_3d.html")
    print("    reports/phase7_interactive_dashboard.html")
    print("="*60)

def run_phase6():
    print("\n" + "="*60)
    print("  PHASE 6 - DATA SCIENCE & ANALYSIS")
    print("="*60)

    import pandas as pd
    import numpy as np
    from ml.dataset_builder import SimulationDataset
    from visualization.analysis_plots import (
        plot_correlation_heatmap,
        plot_load_vs_safety,
        plot_scenario_comparison,
        plot_configuration_comparison,
        plot_3d_stress_surface,
        plot_failure_probability_by_load,
    )

    print("\n  Step 1: Loading simulation datasets...")
    ds_std = SimulationDataset("data/mc_standard_config.csv")
    ds_bad = SimulationDataset("data/mc_dangerous_config.csv")

    print("\n  Step 2: Statistical summaries...")
    ds_std.summary()
    ds_bad.summary()

    print("\n  Step 3: Correlation analysis...")
    df_std = ds_std.df
    df_bad = ds_bad.df
    ds_std.correlation_analysis(target="min_line_sf")

    print("\n  Step 4: Key Engineering Findings")
    print("  " + "-"*50)
    fail_loads = df_std[df_std["any_failure"]==1]["peak_load_kN"]
    safe_loads  = df_std[df_std["any_failure"]==0]["peak_load_kN"]
    if len(fail_loads) > 0:
        print(f"\n  FINDING 1: Failures occur above {fail_loads.min():.1f} kN")
        print(f"    Safe mean load:    {safe_loads.mean():.1f} kN")
        print(f"    Failure mean load: {fail_loads.mean():.1f} kN")
    else:
        print(f"\n  FINDING 1: No failures in standard config — well designed!")

    scen_fail = df_std.groupby("scenario_name")["any_failure"].mean()
    worst = scen_fail.idxmax()
    print(f"\n  FINDING 2: Most dangerous scenario: {worst}")
    print(f"    P(failure): {scen_fail[worst]:.4f}")

    print(f"\n  FINDING 3: Configuration comparison")
    print(f"    Standard  — mean SF: {df_std['min_line_sf'].mean():.3f}  P(fail): {df_std['any_failure'].mean():.5f}")
    print(f"    Dangerous — mean SF: {df_bad['min_line_sf'].mean():.3f}  P(fail): {df_bad['any_failure'].mean():.5f}")

    r = df_std["peak_load_kN"].corr(df_std["beam_safety_factor"])
    print(f"\n  FINDING 4: Load-SF Pearson r = {r:.4f}")
    print(f"    Higher load -> lower SF (expected physical behavior)")

    print("\n  Step 5: Generating visualizations...")
    plots_saved = []

    fig1 = plot_correlation_heatmap(df_std, "Standard Config")
    p1 = "reports/phase6_correlation_heatmap.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p1)

    fig2 = plot_load_vs_safety(df_std, "Standard Config")
    p2 = "reports/phase6_load_vs_safety.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p2)

    fig3 = plot_scenario_comparison(df_std, "Standard Config")
    p3 = "reports/phase6_scenario_comparison.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p3)

    fig4 = plot_configuration_comparison(df_std, df_bad)
    p4 = "reports/phase6_config_comparison.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p4)

    fig5 = plot_3d_stress_surface(df_std, "Standard Config")
    p5 = "reports/phase6_3d_surface.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p5)

    fig6 = plot_failure_probability_by_load(df_std, "Standard Config")
    p6 = "reports/phase6_failure_probability.png"
    fig6.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p6)

    for p in plots_saved:
        print(f"  Saved: {p}")
        open_image(p)

    print("\n" + "="*60)
    print("  PHASE 6 COMPLETE")
    print(f"  {len(plots_saved)} plots saved to reports/")
    print("="*60)
    return ds_std, ds_bad


def run_phase8():
    print("\n" + "="*60)
    print("  PHASE 8 - GENETIC ALGORITHM OPTIMIZATION")
    print("="*60)

    from core.beam import Beam, BeamCrossSection
    from core.material import ALUMINUM_2024_T3
    from core.hydraulic_configuration import create_standard_configuration
    from optimization.genetic_optimizer import (GAConfig, FitnessEvaluator,
                                                  GeneticOptimizer)
    from visualization.ga_plots import (plot_ga_convergence,
                                         plot_population_diversity,
                                         plot_optimized_vs_manual)

    print("\n  Building structural model...")
    cs   = BeamCrossSection(width=0.15, height=0.20)
    beam = Beam(4.0, cs, ALUMINUM_2024_T3, n_stations=100)
    beam.apply_tip_load(50_000)
    beam.apply_distributed_load(3_000)
    print("  Beam ready: L=" + str(beam.L) + "m, max_stress=" + str(round(beam.max_stress()/1e6,1)) + " MPa")

    print("\n  Setting up Genetic Algorithm...")
    ga_config = GAConfig(
        population_size=50,
        n_generations=80,
        n_lines=3,
        crossover_rate=0.8,
        mutation_rate=0.15,
        mutation_sigma=0.08,
        tournament_size=3,
        elitism_count=2,
        seed=42,
    )

    evaluator = FitnessEvaluator(beam, pressure=20.7e6)
    optimizer = GeneticOptimizer(ga_config, evaluator)

    print("  Population: " + str(ga_config.population_size))
    print("  Generations: " + str(ga_config.n_generations))
    print("  Optimizing " + str(ga_config.n_lines) + " hydraulic line positions...")
    print("  Expected time: ~60-120 seconds\n")

    result = optimizer.run(verbose=True)
    result.print_summary()

    result.history_df.to_csv("data/ga_optimization_history.csv", index=False)
    print("\n  Saved: data/ga_optimization_history.csv")

    print("\n  Comparing GA result with manual configuration...")
    manual_config   = create_standard_configuration(beam, pressure=20.7e6)
    manual_analysis = manual_config.analyze(beam)
    opt_config      = evaluator.build_configuration(result.best.genes,
                                                     config_id="GA_OPTIMIZED")
    opt_analysis    = opt_config.analyze(beam)

    print("\n  COMPARISON:")
    print("  " + "-"*53)
    for label, m_val, o_val in [
        ("Config Score",
         manual_analysis["config_score"],
         opt_analysis["config_score"]),
        ("Min Safety Factor",
         manual_analysis["min_safety_factor"],
         opt_analysis["min_safety_factor"]),
        ("Max Von Mises MPa",
         manual_analysis["max_von_mises_mpa"],
         opt_analysis["max_von_mises_mpa"]),
    ]:
        better = "better" if o_val > m_val else ""
        if "MPa" in label:
            better = "better" if o_val < m_val else ""
        print("  " + label.ljust(25) + str(round(m_val,2)).rjust(12) +
              str(round(o_val,2)).rjust(14) + "  " + better)

    print("\n  Generating GA visualization plots...")
    plots_saved = []

    fig1 = plot_ga_convergence(result, "GA Convergence - Hydraulic Optimization")
    p1 = "reports/phase8_ga_convergence.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p1)

    fig2 = plot_population_diversity(result)
    p2 = "reports/phase8_population_diversity.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p2)

    fig3 = plot_optimized_vs_manual(beam, result,
                                     manual_config, manual_analysis,
                                     opt_config, opt_analysis)
    p3 = "reports/phase8_optimized_vs_manual.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p3)

    for p in plots_saved:
        print("  Saved: " + p)
        open_image(p)

    print("\n" + "="*60)
    print("  PHASE 8 COMPLETE")
    print("  Best fitness:  " + str(round(result.best.fitness,2)) + "/100")
    print("  Time elapsed:  " + str(round(result.elapsed,1)) + "s")
    print("  History:       data/ga_optimization_history.csv")
    print("="*60)
    return result




def run_phase9():
    print("\n" + "="*60)
    print("  PHASE 9 - MACHINE LEARNING")
    print("="*60)

    from ml.train_model import HydraulicFailurePredictor
    from visualization.ml_plots import (plot_model_comparison,
                                         plot_feature_importance,
                                         plot_failure_probability_landscape)

    print("\n  Step 1: Loading simulation data...")
    predictor = HydraulicFailurePredictor("data/mc_standard_config.csv")
    X, y = predictor.load_and_prepare()

    print("\n  Step 2: Train/test split...")
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)

    print("\n  Step 3: Training models...")
    predictor.train_random_forest()
    predictor.train_xgboost()

    print("\n  Step 4: Evaluating models...")
    rf_result  = predictor.evaluate_model(predictor.rf_model, "Random Forest")
    xgb_result = predictor.evaluate_model(predictor.xgb_model, "XGBoost")

    print("\n  Step 5: Model Comparison Summary")
    print("  " + "-"*45)
    print("  " + "Model".ljust(20) + "AUC-ROC".rjust(10) + "Avg Prec".rjust(10))
    print("  " + "-"*45)
    for name, res in [("Random Forest", rf_result), ("XGBoost", xgb_result)]:
        print("  " + name.ljust(20) +
              str(round(res["auc_roc"],4)).rjust(10) +
              str(round(res["avg_precision"],4)).rjust(10))

    winner = ("Random Forest"
               if rf_result["auc_roc"] >= xgb_result["auc_roc"]
               else "XGBoost")
    print("\n  Best model: " + winner)

    print("\n  Step 6: Sample failure risk prediction...")
    sample = X_test.iloc[0].to_dict()
    prediction = predictor.predict_failure_risk(sample)
    print("    RF probability:      " + str(round(prediction["rf_probability"],4)))
    print("    XGBoost probability: " + str(round(prediction["xgb_probability"],4)))
    print("    Ensemble:            " + str(round(prediction["ensemble"],4)))
    print("    Risk level:          " + prediction["risk_level"])

    print("\n  Step 7: Saving trained models...")
    predictor.save_models("data")

    print("\n  Step 8: Generating ML visualization plots...")
    plots_saved = []

    fig1 = plot_model_comparison(
        {"Random Forest": rf_result, "XGBoost": xgb_result}, y_test)
    p1 = "reports/phase9_model_comparison.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p1)

    fig2 = plot_feature_importance(predictor, top_n=15)
    p2 = "reports/phase9_feature_importance.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p2)

    fig3 = plot_failure_probability_landscape(predictor, predictor.df_raw)
    p3 = "reports/phase9_probability_landscape.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close("all")
    plots_saved.append(p3)

    for p in plots_saved:
        print("  Saved: " + p)
        open_image(p)

    print("\n" + "="*60)
    print("  PHASE 9 COMPLETE")
    print("  Best model:   " + winner)
    print("  AUC-ROC:      " + str(round(max(rf_result["auc_roc"],
                                              xgb_result["auc_roc"]),4)))
    print("  Models saved: data/rf_model.joblib")
    print("                data/xgb_model.joblib")
    print("="*60)
    return predictor



if __name__ == "__main__":
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        if   phase == "2":   run_phase2()
        elif phase == "3":   run_phase3()
        elif phase == "4":   run_phase4()
        elif phase == "5":   run_phase5()
        elif phase == "6":   run_phase6()
        elif phase == "7":   run_phase7()
        elif phase == "8":   run_phase8()
        elif phase == "9":   run_phase9()
        elif phase == "all":
            run_phase2()
            run_phase3()
            run_phase4()
            run_phase5()
        else:
            print(f"Unknown phase '{phase}'. Use: python main.py 2/3/4/5/6/7/8/9/all")
    else:
        run_phase5()



