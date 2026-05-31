# Build ML datasets from simulation data
# ml/dataset_builder.py

"""
Dataset Builder and Analysis Module
=====================================
Loads Monte Carlo CSV results and prepares them for:
- Statistical analysis (Phase 6)
- Machine learning (Phase 9)

This module is the bridge between simulation output and data science.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List, Optional


class SimulationDataset:
    """
    Loads, validates, and enriches Monte Carlo simulation CSV data.
    
    Adds derived features that make analysis and ML easier:
    - Load severity categories
    - Safety margin columns
    - Scenario risk levels
    - Normalized features for ML
    """
    
    def __init__(self, csv_path: str):
        """
        Load and validate a Monte Carlo results CSV.
        
        Parameters:
        -----------
        csv_path : str — path to CSV file
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        self.path = csv_path
        self.df_raw = pd.read_csv(csv_path)
        self.df = self._enrich(self.df_raw.copy())
        
        print(f"  Loaded: {csv_path}")
        print(f"  Shape:  {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
    
    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns that make analysis cleaner.
        
        These are calculated fields — not stored in the CSV,
        computed fresh each time from the raw data.
        """
        # ── Load severity categories ───────────────────────────────────
        # Bin the continuous peak_load_kN into categories
        # This makes it easy to group and compare
        df['load_category'] = pd.cut(
            df['peak_load_kN'],
            bins=[0, 15, 30, 50, 75, 200],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Extreme']
        )
        
        # ── Safety margin ──────────────────────────────────────────────
        # How far above the minimum safe SF are we?
        # Positive = safe, negative = unsafe
        df['beam_safety_margin'] = df['beam_safety_factor'] - 1.5
        df['line_safety_margin'] = df['min_line_sf'] - 1.5
        
        # ── Stress utilization ─────────────────────────────────────────
        # Stress / Yield = what fraction of yield we're using
        # 0.67 = safe (SF=1.5), 1.0 = at yield, >1.0 = failure
        YIELD_STRENGTH_MPA = 345.0  # Al 2024-T3
        df['beam_stress_utilization'] = df['max_beam_stress_mpa'] / YIELD_STRENGTH_MPA
        
        # ── Scenario risk level ────────────────────────────────────────
        scenario_risk = {
            'CRUISE_LIGHT_TURB':  'LOW',
            'CRUISE_MOD_TURB':    'MEDIUM',
            'CRUISE_SEVERE_TURB': 'HIGH',
            'APPROACH_CROSSWIND': 'MEDIUM',
            'CLIMB_SHEAR':        'MEDIUM',
            'MANEUVER_YAW':       'MEDIUM',
            'EXTREME_GUST':       'HIGH',
        }
        df['scenario_risk'] = df['scenario_name'].map(scenario_risk).fillna('UNKNOWN')
        
        # ── Flight outcome classification ──────────────────────────────
        df['outcome'] = df['any_failure'].map({0: 'SAFE', 1: 'FAILURE'})
        
        # ── Log-transformed load (useful for ML) ──────────────────────
        df['log_peak_load'] = np.log1p(df['peak_load_kN'])
        
        return df
    
    def summary(self) -> Dict:
        """
        Print and return a complete statistical summary.
        """
        df = self.df
        
        print("\n" + "═"*65)
        print(f"  DATASET SUMMARY: {os.path.basename(self.path)}")
        print("═"*65)
        print(f"  Total simulations:    {len(df):,}")
        print(f"  Config ID:            {df['config_id'].iloc[0]}")
        print()
        
        print("  LOAD DISTRIBUTION:")
        print(f"    Mean peak load:     {df['peak_load_kN'].mean():.1f} kN")
        print(f"    Std deviation:      {df['peak_load_kN'].std():.1f} kN")
        print(f"    Median:             {df['peak_load_kN'].median():.1f} kN")
        print(f"    95th percentile:    {df['peak_load_kN'].quantile(0.95):.1f} kN")
        print(f"    99th percentile:    {df['peak_load_kN'].quantile(0.99):.1f} kN")
        print(f"    Max recorded:       {df['peak_load_kN'].max():.1f} kN")
        print()
        
        print("  STRUCTURAL SAFETY:")
        print(f"    Beam SF — mean:     {df['beam_safety_factor'].mean():.3f}")
        print(f"    Beam SF — min:      {df['beam_safety_factor'].min():.3f}")
        print(f"    Beam SF — 5th pct:  {df['beam_safety_factor'].quantile(0.05):.3f}")
        print(f"    Mean max stress:    {df['max_beam_stress_mpa'].mean():.1f} MPa")
        print(f"    Stress utilization: {df['beam_stress_utilization'].mean():.3f} "
              f"(mean fraction of yield)")
        print()
        
        print("  HYDRAULIC LINE SAFETY:")
        print(f"    Min line SF — mean: {df['min_line_sf'].mean():.3f}")
        print(f"    Min line SF — min:  {df['min_line_sf'].min():.3f}")
        print(f"    Config score mean:  {df['config_score'].mean():.1f}/100")
        print()
        
        print("  FAILURE ANALYSIS:")
        n_fail = df['any_failure'].sum()
        p_fail = df['any_failure'].mean()
        print(f"    Failures:           {n_fail} / {len(df):,}")
        print(f"    P(failure):         {p_fail:.5f} ({p_fail*100:.3f}%)")
        print()
        
        print("  LOAD CATEGORY DISTRIBUTION:")
        cat_counts = df['load_category'].value_counts().sort_index()
        for cat, count in cat_counts.items():
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {str(cat):<12}: {count:>5} ({pct:>5.1f}%)  {bar}")
        
        print()
        print("  SCENARIO BREAKDOWN:")
        scenario_stats = df.groupby('scenario_name').agg(
            count=('any_failure', 'count'),
            p_fail=('any_failure', 'mean'),
            mean_load=('peak_load_kN', 'mean'),
            max_load=('peak_load_kN', 'max'),
        ).sort_values('mean_load', ascending=False)
        
        print(f"  {'Scenario':<28} {'Count':>6} {'P(fail)':>8} "
              f"{'Mean Load':>10} {'Max Load':>10}")
        print("  " + "-"*67)
        for name, row in scenario_stats.iterrows():
            print(f"  {name:<28} {int(row['count']):>6} "
                  f"{row['p_fail']:>8.4f} "
                  f"{row['mean_load']:>8.1f} kN "
                  f"{row['max_load']:>8.1f} kN")
        print("═"*65)
        
        return {
            'n': len(df),
            'p_failure': p_fail,
            'mean_load': df['peak_load_kN'].mean(),
            'mean_beam_sf': df['beam_safety_factor'].mean(),
            'mean_line_sf': df['min_line_sf'].mean(),
            'config_score_mean': df['config_score'].mean(),
        }
    
    def correlation_analysis(self, target='min_line_sf') -> pd.Series:
        """
        Compute Pearson correlations between all numeric features
        and a target variable.
        
        WHY CORRELATION?
        Tells us which inputs most strongly predict our output.
        r = +1.0 → perfect positive relationship
        r = -1.0 → perfect negative relationship  
        r = 0.0  → no linear relationship
        
        Engineering use: if peak_load has r=-0.9 with SF,
        we know that controlling peak load is the key lever.
        
        Parameters:
        -----------
        target : str — column name to correlate against
        
        Returns:
        --------
        pd.Series : correlations sorted by absolute value
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols 
                        if c != target and c != 'any_failure']
        
        corr = self.df[numeric_cols].corrwith(self.df[target])
        corr = corr.dropna().sort_values(key=abs, ascending=False)
        
        print(f"\n  Top correlations with '{target}':")
        print(f"  {'Feature':<35} {'Correlation':>12}")
        print("  " + "-"*50)
        for feat, val in corr.head(15).items():
            bar = "+" * int(abs(val) * 20) if val > 0 else "-" * int(abs(val) * 20)
            direction = "▲" if val > 0 else "▼"
            print(f"  {feat:<35} {val:>+10.4f}  {direction} {bar}")
        
        return corr
    
    def get_ml_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels for machine learning.
        
        Features (X): inputs to the ML model
        Labels (y):   what we want to predict (failure = 1, safe = 0)
        
        Feature engineering choices:
        - Use load, scenario-encoded, stress values
        - Drop ID columns and derived labels
        - One-hot encode categorical scenario names
        
        Returns:
        --------
        X : pd.DataFrame — feature matrix
        y : pd.Series — binary labels (0=safe, 1=failure)
        """
        feature_cols = [
            'peak_load_kN',
            'sideslip_deg',
            'turbulence_intensity',
            'max_beam_stress_mpa',
            'beam_safety_factor',
            'tip_deflection_mm',
            'root_moment_kNm',
            'config_score',
            'min_line_sf',
            'max_line_vm_mpa',
            'n_unsafe_lines',
            'log_peak_load',
            'beam_stress_utilization',
        ]
        
        # Add per-line features if available
        for i in range(1, 4):
            for suffix in ['_vm_mpa', '_sf', '_score']:
                col = f'line{i}{suffix}'
                if col in self.df.columns:
                    feature_cols.append(col)
        
        # Add one-hot encoded scenario
        scenario_dummies = pd.get_dummies(
            self.df['scenario_name'], prefix='scenario'
        )
        
        X = self.df[feature_cols].copy()
        X = pd.concat([X, scenario_dummies], axis=1)
        y = self.df['any_failure'].astype(int)
        
        return X, y