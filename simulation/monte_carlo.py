# simulation/monte_carlo.py

"""
Monte Carlo Simulation Engine
================================
The computational core of the hydraulic safety optimization system.

Runs thousands of random flight simulations, applies structural loads
to the beam model, evaluates hydraulic line safety for each simulation,
and accumulates complete statistical results.

Architecture:
  MonteCarloConfig   — simulation parameters
  SimulationResult   — single iteration result
  MonteCarloResult   — aggregated N-iteration results
  MonteCarloEngine   — the main simulation runner
  
Performance notes:
  - 5,000 iterations typically completes in 30-120 seconds
  - Progress bar shows real-time completion
  - Results saved incrementally to avoid data loss
  - Parallelizable (see advanced section)
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

# Import our physics engines
from core.beam import Beam, BeamCrossSection
from core.material import ALUMINUM_2024_T3
from core.hydraulic_line import HydraulicLine, HydraulicLineGeometry, TITANIUM_TUBE
from core.hydraulic_configuration import HydraulicConfiguration
from simulation.wind_model import (
    AerodynamicModel, TurbulenceModel, GustEventModel,
    FlightSimulation, CRUISE_CONDITIONS
)
from simulation.flight_conditions import (
    FlightScenario, FlightConditionSampler, build_flight_scenarios
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """
    Complete configuration for a Monte Carlo simulation run.
    
    Having all parameters in one dataclass means:
    - Easy to save/load simulation configs
    - Easy to reproduce results (same seed → same results)
    - Easy to compare runs with different parameters
    """
    # Simulation scale
    n_iterations: int = 5000
    flight_hours_per_sim: float = 4.0
    
    # Random seed (None = random each run, int = reproducible)
    seed: int = 42
    
    # Beam geometry
    beam_length: float = 4.0       # m
    beam_width: float = 0.15       # m
    beam_height: float = 0.20      # m
    
    # Aerodynamic model
    fin_area: float = 10.0         # m²
    lift_curve_slope: float = 3.8  # per radian
    
    # Hydraulic system
    operating_pressure: float = 20.7e6   # Pa (3000 PSI)
    
    # Output
    save_csv: bool = True
    csv_path: str = 'data/monte_carlo_results.csv'
    save_interval: int = 500       # Save progress every N iterations
    verbose: bool = True
    
    def to_dict(self):
        """Convert to dictionary for CSV header metadata."""
        return {
            'n_iterations': self.n_iterations,
            'flight_hours': self.flight_hours_per_sim,
            'seed': self.seed,
            'beam_length': self.beam_length,
            'fin_area': self.fin_area,
            'pressure_mpa': self.operating_pressure / 1e6,
        }


@dataclass
class SimulationRecord:
    """
    Complete record from a single Monte Carlo iteration.
    
    Stores everything needed for Phase 6 data science analysis.
    Each record becomes one row in the output CSV.
    """
    iteration: int
    scenario_name: str
    
    # Applied load
    peak_load_N: float
    sideslip_deg: float
    turbulence_intensity: float
    
    # Structural results
    max_beam_stress_mpa: float
    beam_safety_factor: float
    tip_deflection_mm: float
    root_moment_kNm: float
    
    # Per-line hydraulic results (stored as lists, one per line)
    line_ids: List[str]
    line_y_positions_mm: List[float]
    line_von_mises_mpa: List[float]
    line_safety_factors: List[float]
    line_safety_scores: List[float]
    line_statuses: List[str]
    line_fatigue_flights: List[float]
    
    # Configuration-level results
    config_id: str
    config_score: float
    min_line_sf: float
    max_line_vm_mpa: float
    n_unsafe_lines: int
    any_failure: bool
    
    def to_flat_dict(self):
        """
        Convert to flat dictionary for CSV storage.
        
        Multi-valued fields (per-line results) are stored as
        comma-separated strings in single CSV columns.
        This allows loading back with proper parsing.
        """
        d = {
            'iteration': self.iteration,
            'scenario_name': self.scenario_name,
            'peak_load_kN': self.peak_load_N / 1000,
            'sideslip_deg': round(self.sideslip_deg, 3),
            'turbulence_intensity': self.turbulence_intensity,
            'max_beam_stress_mpa': round(self.max_beam_stress_mpa, 2),
            'beam_safety_factor': round(self.beam_safety_factor, 3),
            'tip_deflection_mm': round(self.tip_deflection_mm, 3),
            'root_moment_kNm': round(self.root_moment_kNm, 2),
            'config_id': self.config_id,
            'config_score': round(self.config_score, 2),
            'min_line_sf': round(self.min_line_sf, 3),
            'max_line_vm_mpa': round(self.max_line_vm_mpa, 2),
            'n_unsafe_lines': self.n_unsafe_lines,
            'any_failure': int(self.any_failure),
        }
        
        # Per-line results as indexed columns (cleaner than CSV strings)
        for i, (lid, ypos, vm, sf, score, status, fatigue) in enumerate(zip(
            self.line_ids, self.line_y_positions_mm, self.line_von_mises_mpa,
            self.line_safety_factors, self.line_safety_scores,
            self.line_statuses, self.line_fatigue_flights
        )):
            prefix = f'line{i+1}'
            d[f'{prefix}_id'] = lid
            d[f'{prefix}_y_mm'] = round(ypos, 2)
            d[f'{prefix}_vm_mpa'] = round(vm, 2)
            d[f'{prefix}_sf'] = round(sf, 3)
            d[f'{prefix}_score'] = round(score, 2)
            d[f'{prefix}_status'] = status
            fatigue_val = min(fatigue, 1e9) if fatigue != float('inf') else 999999
            d[f'{prefix}_fatigue_flights'] = round(fatigue_val, 0)
        
        return d


@dataclass
class MonteCarloResult:
    """
    Aggregated results from a complete Monte Carlo run.
    
    Contains all individual records plus computed statistics.
    This is the final output of the simulation engine.
    """
    config: MonteCarloConfig
    records: List[SimulationRecord]
    dataframe: Optional[pd.DataFrame] = None
    
    # Computed statistics (filled by compute_statistics())
    stats: Dict = field(default_factory=dict)
    
    def compute_statistics(self):
        """
        Compute aggregate statistics from all simulation records.
        
        This is the core analysis step — converting thousands of
        individual results into engineering insights.
        
        Returns:
        --------
        dict : complete statistics dictionary
        """
        if self.dataframe is None:
            self.dataframe = pd.DataFrame(
                [r.to_flat_dict() for r in self.records]
            )
        
        df = self.dataframe
        N = len(df)
        
        stats = {
            # ── Sample size ────────────────────────────────────────────
            'n_iterations': N,
            
            # ── Load statistics ────────────────────────────────────────
            'load_mean_kN':      df['peak_load_kN'].mean(),
            'load_std_kN':       df['peak_load_kN'].std(),
            'load_min_kN':       df['peak_load_kN'].min(),
            'load_max_kN':       df['peak_load_kN'].max(),
            'load_95th_kN':      df['peak_load_kN'].quantile(0.95),
            'load_99th_kN':      df['peak_load_kN'].quantile(0.99),
            
            # ── Beam structural statistics ─────────────────────────────
            'beam_sf_mean':      df['beam_safety_factor'].mean(),
            'beam_sf_std':       df['beam_safety_factor'].std(),
            'beam_sf_min':       df['beam_safety_factor'].min(),
            'beam_sf_5th':       df['beam_safety_factor'].quantile(0.05),
            'max_stress_mean_mpa': df['max_beam_stress_mpa'].mean(),
            'max_stress_99th_mpa': df['max_beam_stress_mpa'].quantile(0.99),
            'deflection_mean_mm':  df['tip_deflection_mm'].mean(),
            'deflection_max_mm':   df['tip_deflection_mm'].max(),
            
            # ── Hydraulic safety statistics ────────────────────────────
            'min_line_sf_mean':  df['min_line_sf'].mean(),
            'min_line_sf_std':   df['min_line_sf'].std(),
            'min_line_sf_5th':   df['min_line_sf'].quantile(0.05),
            'config_score_mean': df['config_score'].mean(),
            'config_score_std':  df['config_score'].std(),
            'config_score_min':  df['config_score'].min(),
            'max_vm_mean_mpa':   df['max_line_vm_mpa'].mean(),
            'max_vm_99th_mpa':   df['max_line_vm_mpa'].quantile(0.99),
            
            # ── Failure statistics ─────────────────────────────────────
            'n_any_failure':     int(df['any_failure'].sum()),
            'p_failure':         df['any_failure'].mean(),
            'n_unsafe_mean':     df['n_unsafe_lines'].mean(),
            
            # ── Scenario breakdown ─────────────────────────────────────
            'scenario_counts':   df['scenario_name'].value_counts().to_dict(),
        }
        
        # Per-scenario failure rates
        scenario_failure = {}
        for scenario in df['scenario_name'].unique():
            mask = df['scenario_name'] == scenario
            scenario_failure[scenario] = df.loc[mask, 'any_failure'].mean()
        stats['scenario_failure_rates'] = scenario_failure
        
        # Per-line statistics (if we have per-line columns)
        line_cols_sf = [c for c in df.columns if c.endswith('_sf')]
        if line_cols_sf:
            all_line_sfs = df[line_cols_sf].values.flatten()
            all_line_sfs = all_line_sfs[~np.isnan(all_line_sfs)]
            stats['all_lines_sf_mean'] = np.mean(all_line_sfs)
            stats['all_lines_sf_min']  = np.min(all_line_sfs)
        
        self.stats = stats
        return stats
    
    def print_summary(self):
        """Print a formatted engineering summary of simulation results."""
        if not self.stats:
            self.compute_statistics()
        
        s = self.stats
        print("\n" + "═"*68)
        print(f"  MONTE CARLO RESULTS SUMMARY")
        print(f"  Configuration: {self.records[0].config_id if self.records else 'N/A'}")
        print("═"*68)
        print(f"  Simulations completed:    {s['n_iterations']:,}")
        print()
        print(f"  AERODYNAMIC LOADS:")
        print(f"    Mean peak load:         {s['load_mean_kN']:.1f} kN")
        print(f"    Std deviation:          {s['load_std_kN']:.1f} kN")
        print(f"    95th percentile:        {s['load_95th_kN']:.1f} kN")
        print(f"    99th percentile:        {s['load_99th_kN']:.1f} kN")
        print(f"    Maximum recorded:       {s['load_max_kN']:.1f} kN")
        print()
        print(f"  STRUCTURAL SAFETY (BEAM):")
        print(f"    Mean safety factor:     {s['beam_sf_mean']:.2f}")
        print(f"    5th percentile SF:      {s['beam_sf_5th']:.2f} "
              f"({'✅' if s['beam_sf_5th'] >= 1.5 else '❌'})")
        print(f"    Minimum SF recorded:    {s['beam_sf_min']:.2f}")
        print(f"    Mean max stress:        {s['max_stress_mean_mpa']:.1f} MPa")
        print(f"    99th pct max stress:    {s['max_stress_99th_mpa']:.1f} MPa")
        print()
        print(f"  HYDRAULIC LINE SAFETY:")
        print(f"    Mean config score:      {s['config_score_mean']:.1f} / 100")
        print(f"    Mean min line SF:       {s['min_line_sf_mean']:.2f}")
        print(f"    5th pct min line SF:    {s['min_line_sf_5th']:.2f} "
              f"({'✅' if s['min_line_sf_5th'] >= 1.5 else '❌'})")
        print()
        print(f"  FAILURE ANALYSIS:")
        print(f"    Flights with failures:  {s['n_any_failure']:,} / {s['n_iterations']:,}")
        print(f"    Probability of failure: {s['p_failure']:.4f} "
              f"({s['p_failure']*100:.2f}%)")
        print(f"    FAA target (≤ 10⁻⁴):   "
              f"{'✅ PASS' if s['p_failure'] <= 1e-4 else '❌ FAIL'}")
        print()
        print(f"  SCENARIO BREAKDOWN:")
        for scen, count in sorted(s['scenario_counts'].items(),
                                   key=lambda x: -x[1]):
            fr = s['scenario_failure_rates'].get(scen, 0)
            print(f"    {scen:<30}: {count:>5} flights, "
                  f"P(fail)={fr:.4f}")
        print("═"*68)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloEngine:
    """
    Main Monte Carlo simulation engine.
    
    Orchestrates:
    1. Building the structural beam model
    2. Setting up the aerodynamic and flight models
    3. Running N iterations of random flight loads
    4. For each iteration: applying load to beam, evaluating hydraulic safety
    5. Accumulating results into SimulationRecord objects
    6. Saving to CSV with progress updates
    7. Computing final statistics
    
    Usage:
        engine = MonteCarloEngine(config, hydraulic_config)
        result = engine.run()
        result.print_summary()
    """
    
    def __init__(
        self,
        config: MonteCarloConfig,
        hydraulic_config: HydraulicConfiguration,
    ):
        """
        Parameters:
        -----------
        config : MonteCarloConfig
            All simulation parameters
        hydraulic_config : HydraulicConfiguration
            The hydraulic routing to analyze
            (Contains HydraulicLine objects with positions)
        """
        self.config = config
        self.hyd_config = hydraulic_config
        self.rng = np.random.default_rng(config.seed)
        
        # Build structural model
        self.beam = self._build_beam()
        
        # Build aerodynamic and flight models
        self.aero_model = self._build_aero_model()
        self.scenarios = build_flight_scenarios()
        self.sampler = FlightConditionSampler(
            self.aero_model,
            self.scenarios,
            seed=config.seed + 1
        )
        
        # Ensure output directory exists
        if config.save_csv:
            os.makedirs(os.path.dirname(config.csv_path), exist_ok=True)
    
    def _build_beam(self):
        """Build the structural beam from config parameters."""
        cs = BeamCrossSection(
            width=self.config.beam_width,
            height=self.config.beam_height
        )
        return Beam(
            length=self.config.beam_length,
            cross_section=cs,
            material=ALUMINUM_2024_T3,
            n_stations=100  # Reduced for speed in MC loop
        )
    
    def _build_aero_model(self):
        """Build the aerodynamic force model."""
        return AerodynamicModel(
            fin_area=self.config.fin_area,
            fin_span=self.config.beam_length,
            lift_curve_slope=self.lift_curve_slope if 
                hasattr(self, 'lift_curve_slope') else 3.8,
            fin_efficiency=0.93
        )
    
    def _run_single_iteration(self, iteration: int, flight_sample: dict):
        """
        Execute one complete Monte Carlo iteration.
        
        This is the inner loop function. Called N times.
        Must be fast — every millisecond here multiplies by N.
        
        Steps:
        1. Apply sampled peak load to beam
        2. Calculate beam structural response
        3. Evaluate all hydraulic lines
        4. Build and return SimulationRecord
        
        Parameters:
        -----------
        iteration : int — iteration number
        flight_sample : dict from FlightConditionSampler.sample_flight_load()
        
        Returns:
        --------
        SimulationRecord
        """
        peak_load = flight_sample['peak_load']
        
        # ── Apply load to beam ─────────────────────────────────────────
        # Peak load is a tip load (simplification — actual load is
        # distributed, but tip load equivalent gives same peak stress)
        # We also add a small distributed load representing fin pressure
        self.beam.apply_tip_load(peak_load * 0.8)     # 80% as tip load
        self.beam.apply_distributed_load(
            peak_load * 0.2 / self.config.beam_length  # 20% as distributed
        )
        
        # ── Structural results ─────────────────────────────────────────
        max_beam_stress = self.beam.max_stress()
        beam_sf = self.beam.material.safety_factor(max_beam_stress)
        tip_def = self.beam.tip_deflection()
        root_moment = self.beam.moment[0]
        
        # ── Hydraulic line analysis ────────────────────────────────────
        config_analysis = self.hyd_config.analyze(self.beam)
        line_results = config_analysis['line_results']
        
        # Extract per-line data
        line_ids = [r['line_id'] for r in line_results]
        line_y_mm = [r['y_position'] * 1000 for r in line_results]
        line_vm = [r['sigma_von_mises_mpa'] for r in line_results]
        line_sfs = [r['sf_overall'] for r in line_results]
        line_scores = [r['safety_score'] for r in line_results]
        line_statuses = [r['status'] for r in line_results]
        line_fatigue = [
            r['fatigue_flights'] if r['fatigue_flights'] != float('inf')
            else 999999
            for r in line_results
        ]
        
        # ── Failure detection ──────────────────────────────────────────
        # Failure = any line SF < 1.0 (stress exceeds yield)
        # OR beam SF < 1.0 (structural failure)
        n_unsafe = sum(1 for sf in line_sfs if sf < 1.5)
        any_failure = (beam_sf < 1.0 or 
                       any(sf < 1.0 for sf in line_sfs))
        
        return SimulationRecord(
            iteration=iteration,
            scenario_name=flight_sample['scenario_name'],
            peak_load_N=peak_load,
            sideslip_deg=flight_sample['sideslip_deg'],
            turbulence_intensity=flight_sample['turbulence_intensity'],
            max_beam_stress_mpa=max_beam_stress / 1e6,
            beam_safety_factor=beam_sf,
            tip_deflection_mm=tip_def * 1000,
            root_moment_kNm=root_moment / 1000,
            line_ids=line_ids,
            line_y_positions_mm=line_y_mm,
            line_von_mises_mpa=line_vm,
            line_safety_factors=line_sfs,
            line_safety_scores=line_scores,
            line_statuses=line_statuses,
            line_fatigue_flights=line_fatigue,
            config_id=self.hyd_config.config_id,
            config_score=config_analysis['config_score'],
            min_line_sf=config_analysis['min_safety_factor'],
            max_line_vm_mpa=config_analysis['max_von_mises_mpa'],
            n_unsafe_lines=n_unsafe,
            any_failure=any_failure,
        )
    
    def run(self):
        """
        Execute the complete Monte Carlo simulation.
        
        This is the main public method. Call this to run everything.
        
        Returns:
        --------
        MonteCarloResult : complete results with statistics
        """
        config = self.config
        records = []
        rows_for_csv = []
        
        if config.verbose:
            print(f"\n{'─'*60}")
            print(f"  Monte Carlo Engine Starting")
            print(f"  Config:      {self.hyd_config.config_id}")
            print(f"  Iterations:  {config.n_iterations:,}")
            print(f"  Seed:        {config.seed}")
            print(f"  Output CSV:  {config.csv_path}")
            print(f"{'─'*60}")
        
        start_time = time.time()
        
        # ── Main simulation loop ───────────────────────────────────────
        # tqdm gives us a progress bar with ETA — essential for long runs
        with tqdm(
            total=config.n_iterations,
            desc=f"  Simulating {self.hyd_config.config_id}",
            unit="flight",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
            disable=not config.verbose
        ) as pbar:
            
            for i in range(config.n_iterations):
                
                # Sample a random flight load
                try:
                    flight_sample = self.sampler.sample_flight_load(
                        config.flight_hours_per_sim
                    )
                except Exception as e:
                    warnings.warn(f"Iteration {i}: load sampling failed: {e}")
                    continue
                
                # Run structural + hydraulic analysis
                try:
                    record = self._run_single_iteration(i, flight_sample)
                    records.append(record)
                    rows_for_csv.append(record.to_flat_dict())
                except Exception as e:
                    warnings.warn(f"Iteration {i}: analysis failed: {e}")
                    continue
                
                # Update progress bar with live statistics
                if (i + 1) % 100 == 0:
                    recent_sfs = [r.min_line_sf for r in records[-100:]]
                    recent_failures = sum(1 for r in records[-100:]
                                          if r.any_failure)
                    pbar.set_postfix({
                        'MinSF': f"{np.mean(recent_sfs):.2f}",
                        'Fail/100': recent_failures,
                    })
                
                # Periodic CSV save
                if (config.save_csv and 
                        (i + 1) % config.save_interval == 0 and
                        rows_for_csv):
                    self._save_csv_checkpoint(rows_for_csv, config.csv_path)
                
                pbar.update(1)
        
        # ── Final CSV save ─────────────────────────────────────────────
        if config.save_csv and rows_for_csv:
            self._save_csv_checkpoint(rows_for_csv, config.csv_path)
            if config.verbose:
                print(f"\n  ✅ Results saved: {config.csv_path}")
                print(f"     Rows: {len(rows_for_csv):,} | "
                      f"Columns: {len(rows_for_csv[0])}")
        
        # ── Build result object ────────────────────────────────────────
        elapsed = time.time() - start_time
        
        if config.verbose:
            print(f"\n  ⏱️  Simulation complete: {elapsed:.1f}s "
                  f"({config.n_iterations/elapsed:.0f} iter/s)")
        
        df = pd.DataFrame(rows_for_csv)
        result = MonteCarloResult(
            config=config,
            records=records,
            dataframe=df,
        )
        result.compute_statistics()
        
        return result
    
    def _save_csv_checkpoint(self, rows: list, path: str):
        """
        Save current results to CSV (overwrites previous checkpoint).
        
        Using pandas to_csv is the cleanest approach.
        The file is always complete (not appended) so partial runs
        are still valid datasets.
        
        Parameters:
        -----------
        rows : list of flat dicts
        path : str — output file path
        """
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, float_format='%.4f')


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-CONFIGURATION COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurationComparator:
    """
    Runs Monte Carlo for multiple hydraulic configurations and compares them.
    
    This is the key tool for understanding which hydraulic routing designs
    are safest across the full flight load spectrum.
    
    In the genetic algorithm (Phase 8), this comparator will be called
    to evaluate each candidate configuration.
    """
    
    def __init__(self, base_config: MonteCarloConfig):
        """
        Parameters:
        -----------
        base_config : MonteCarloConfig
            Template config — each configuration run uses the same
            simulation parameters (N iterations, seed, etc.)
        """
        self.base_config = base_config
        self.results = {}
    
    def add_configuration(
        self,
        hydraulic_config: HydraulicConfiguration,
        n_iterations: Optional[int] = None
    ):
        """
        Run Monte Carlo for one hydraulic configuration and store result.
        
        Parameters:
        -----------
        hydraulic_config : HydraulicConfiguration
        n_iterations : int or None
            Override iteration count (for quick comparison use fewer)
        """
        cfg = MonteCarloConfig(
            n_iterations=n_iterations or self.base_config.n_iterations,
            seed=self.base_config.seed,
            beam_length=self.base_config.beam_length,
            beam_width=self.base_config.beam_width,
            beam_height=self.base_config.beam_height,
            fin_area=self.base_config.fin_area,
            operating_pressure=self.base_config.operating_pressure,
            save_csv=self.base_config.save_csv,
            csv_path=f"data/mc_{hydraulic_config.config_id.lower()}.csv",
            verbose=self.base_config.verbose,
        )
        
        engine = MonteCarloEngine(cfg, hydraulic_config)
        result = engine.run()
        
        self.results[hydraulic_config.config_id] = result
        return result
    
    def comparison_dataframe(self):
        """
        Build a summary DataFrame comparing all configurations.
        
        One row per configuration, key metrics as columns.
        Perfect for a bar chart comparison or ranking.
        
        Returns:
        --------
        pd.DataFrame : comparison table
        """
        rows = []
        for config_id, result in self.results.items():
            s = result.stats
            rows.append({
                'config_id': config_id,
                'n_iterations': s['n_iterations'],
                'config_score_mean': round(s['config_score_mean'], 2),
                'config_score_min': round(s['config_score_min'], 2),
                'min_line_sf_mean': round(s['min_line_sf_mean'], 3),
                'min_line_sf_5th': round(s['min_line_sf_5th'], 3),
                'max_vm_mean_mpa': round(s['max_vm_mean_mpa'], 2),
                'max_vm_99th_mpa': round(s['max_vm_99th_mpa'], 2),
                'p_failure': round(s['p_failure'], 6),
                'beam_sf_mean': round(s['beam_sf_mean'], 3),
                'beam_sf_min': round(s['beam_sf_min'], 3),
                'load_95th_kN': round(s['load_95th_kN'], 1),
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('config_score_mean', ascending=False)
        return df
    
    def print_comparison(self):
        """Print a formatted side-by-side comparison of all configurations."""
        df = self.comparison_dataframe()
        
        print("\n" + "═"*75)
        print("  MULTI-CONFIGURATION MONTE CARLO COMPARISON")
        print("═"*75)
        print(f"  {'Config':<25} {'Score':>8} {'MinSF_μ':>9} "
              f"{'MinSF_5%':>10} {'P(fail)':>10} {'Status':>10}")
        print("  " + "─"*73)
        
        for _, row in df.iterrows():
            is_safe = row['min_line_sf_5th'] >= 1.5
            p_ok = row['p_failure'] <= 1e-4
            status = '✅ PASS' if (is_safe and p_ok) else '❌ FAIL'
            
            print(f"  {row['config_id']:<25} "
                  f"{row['config_score_mean']:>7.1f} "
                  f"{row['min_line_sf_mean']:>9.3f} "
                  f"{row['min_line_sf_5th']:>9.3f} "
                  f"{row['p_failure']:>10.6f} "
                  f"{status:>10}")
        
        print("═"*75)
        return df
