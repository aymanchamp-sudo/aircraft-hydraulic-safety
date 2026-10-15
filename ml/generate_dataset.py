# ml/generate_dataset.py
"""
Build the machine-learning dataset.

WHY THIS FILE EXISTS
--------------------
The original ML data was the same 2,000 simulated flights run twice (once per
tube layout, both with seed 42), so it contained only 2 distinct failures.
A classifier cannot be validated on 2 events.

This generator draws every row independently:

  * its OWN random flight  (scenario, sideslip, turbulence, gusts -> peak load)
  * its OWN random design  (tube positions, pressure, wall thickness, spar height)

Scenarios are sampled uniformly (not by real-world frequency) so that rare,
severe flights appear often enough to produce failures. The real-world weight of
each scenario is stored in `scenario_weight`, so true probabilities can be
recovered by re-weighting (see `real_world_failure_rate`).

Every row has a unique `flight_id`; no flight is ever reused. The row's random
stream is default_rng([seed, flight_id]), so runs are reproducible and rows are
statistically independent.

Run:  python main.py 10      (or: python -m ml.generate_dataset)
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.beam import Beam, BeamCrossSection
from core.material import ALUMINUM_2024_T3
from core.hydraulic_line import (HydraulicLine, HydraulicLineGeometry,
                                  TITANIUM_TUBE)
from core.hydraulic_configuration import HydraulicConfiguration
from simulation.wind_model import AerodynamicModel
from simulation.flight_conditions import (build_flight_scenarios,
                                           FlightConditionSampler)
# Outcome columns (never ML inputs) are defined in ml/features.py

# ── Fixed project geometry (same as the Monte Carlo engine) ────────────────
BEAM_LENGTH = 4.0          # m
BEAM_WIDTH = 0.15          # m
FIN_AREA = 10.0            # m^2
LIFT_SLOPE = 3.8           # per rad
FIN_EFFICIENCY = 0.93
TUBE_OD = 0.012            # m
TIP_FRACTION = 0.8         # 80 % of peak load applied at the tip, 20 % distributed

# ── Design space sampled for every row ─────────────────────────────────────
SPAR_HEIGHT_RANGE = (0.16, 0.26)     # m
TUBE_FRACTION_RANGE = (-0.9, 0.9)    # y / (h/2)
PRESSURES_MPA = (20.7, 34.5)         # 3000 PSI, 5000 PSI
WALL_RANGE_MM = (0.7, 1.3)
STRESS_MULTIPLIER_RANGE = (1.0, 1.6)  # only used if too few failures occur

def _simulate_flight(args):
    """Worker 1: draw one design + one flight. Returns the INPUT columns."""
    i, seed = args
    rng = np.random.default_rng([seed, i])
    scenarios = build_flight_scenarios()
    total_w = sum(s.weight for s in scenarios)

    scenario = scenarios[int(rng.integers(0, len(scenarios)))]
    spar_height = rng.uniform(*SPAR_HEIGHT_RANGE)
    fractions = rng.uniform(*TUBE_FRACTION_RANGE, size=3)
    pressure_mpa = float(rng.choice(PRESSURES_MPA))
    wall_mm = rng.uniform(*WALL_RANGE_MM)
    stress_draw = rng.uniform(*STRESS_MULTIPLIER_RANGE)   # always drawn, used later if needed
    flight_seed = int(rng.integers(0, 2**31 - 1))

    aero = AerodynamicModel(fin_area=FIN_AREA, fin_span=BEAM_LENGTH,
                            lift_curve_slope=LIFT_SLOPE,
                            fin_efficiency=FIN_EFFICIENCY)
    sampler = FlightConditionSampler(aero, [scenario], seed=flight_seed)
    flight = sampler.sample_flight_load(flight_hours=4.0)

    row = {
        "flight_id": i,
        "scenario_name": scenario.name,
        "scenario_weight": scenario.weight / total_w,
        "sideslip_deg": flight["sideslip_deg"],
        "turbulence_intensity": flight["turbulence_intensity"],
        "base_peak_load_kN": flight["peak_load"] / 1000.0,
        "spar_height_m": spar_height,
        "pressure_mpa": pressure_mpa,
        "wall_thickness_mm": wall_mm,
        "stress_draw": stress_draw,
    }
    for k in range(3):
        row[f"line{k+1}_y_frac"] = fractions[k]
    return row


def _evaluate(args):
    """Worker 2: run the beam + tube physics for one row at a load multiplier."""
    row, multiplier = args
    load_n = row["base_peak_load_kN"] * 1000.0 * multiplier
    h = row["spar_height_m"]

    beam = Beam(BEAM_LENGTH, BeamCrossSection(BEAM_WIDTH, h),
                ALUMINUM_2024_T3, n_stations=100)
    beam.apply_tip_load(load_n * TIP_FRACTION)
    beam.apply_distributed_load(load_n * (1 - TIP_FRACTION) / BEAM_LENGTH)

    geom = HydraulicLineGeometry(outer_diameter=TUBE_OD,
                                 wall_thickness=row["wall_thickness_mm"] / 1000.0)
    lines = []
    for k in range(3):
        y = row[f"line{k+1}_y_frac"] * h / 2.0
        lines.append(HydraulicLine(
            line_id=f"HYD-{k+1}", geometry=geom, material=TITANIUM_TUBE,
            x_start=0.2, x_end=BEAM_LENGTH * 0.9, y_position=y,
            operating_pressure=row["pressure_mpa"] * 1e6,
            system_name=f"System_{'ABC'[k]}"))
    analysis = HydraulicConfiguration("ML_SAMPLE", lines).analyze(beam)

    beam_stress = beam.max_stress()
    beam_sf = ALUMINUM_2024_T3.safety_factor(beam_stress)
    line_sfs = [r["sf_overall"] for r in analysis["line_results"]]
    beam_fail, tube_fail = beam_sf < 1.0, min(line_sfs) < 1.0
    cause = ("both" if beam_fail and tube_fail else
             "beam" if beam_fail else "tube" if tube_fail else "none")

    out = dict(row)
    out.pop("stress_draw")
    out.update({
        "stress_multiplier": multiplier,
        "peak_load_kN": load_n / 1000.0,
        "max_beam_stress_mpa": beam_stress / 1e6,
        "beam_safety_factor": beam_sf,
        "min_line_sf": min(line_sfs),
        "max_line_vm_mpa": analysis["max_von_mises_mpa"],
        "config_score": analysis["config_score"],
        "n_unsafe_lines": sum(1 for sf in line_sfs if sf < 1.5),
        "any_failure": int(beam_fail or tube_fail),
        "failure_cause": cause,
    })
    return out


def real_world_failure_rate(df):
    """
    Re-weight the uniform-scenario sample to real-world scenario frequencies:
    P(fail) = sum over scenarios of weight_s * failure_rate_s.
    Only meaningful when stress_multiplier == 1 for every row.
    """
    rates = df.groupby("scenario_name").agg(
        w=("scenario_weight", "first"), p=("any_failure", "mean"))
    return float((rates["w"] * rates["p"]).sum())


def generate_dataset(n_samples=20000, seed=2026, min_failures=300,
                     out_path="data/ml_dataset.csv", workers=None, verbose=True):
    """Generate and save the dataset. Returns the DataFrame."""
    workers = workers or max(1, min(os.cpu_count() or 1, 4))
    t0 = time.time()
    if verbose:
        print(f"  Simulating {n_samples:,} independent flights + designs "
              f"on {workers} worker(s)...")

    with Pool(workers) as pool:
        inputs = pool.map(_simulate_flight,
                          [(i, seed) for i in range(n_samples)], chunksize=200)
        if verbose:
            print(f"  Flights done in {time.time() - t0:.0f}s. Evaluating physics...")

        rows = pool.map(_evaluate, [(r, 1.0) for r in inputs], chunksize=200)
        n_fail = sum(r["any_failure"] for r in rows)
        used_multiplier = False

        if n_fail < min_failures:
            if verbose:
                print(f"  Only {n_fail} failures (< {min_failures}). Applying a "
                      f"stress-test load multiplier drawn from "
                      f"{STRESS_MULTIPLIER_RANGE[0]} to {STRESS_MULTIPLIER_RANGE[1]}.")
            rows = pool.map(_evaluate,
                            [(r, r["stress_draw"]) for r in inputs], chunksize=200)
            used_multiplier = True
            n_fail = sum(r["any_failure"] for r in rows)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = df.round(5)          # keeps the CSV small (about 6 MB)
    df.to_csv(out_path, index=False)

    if verbose:
        print(f"\n  Saved {len(df):,} rows -> {out_path}  "
              f"({time.time() - t0:.0f}s total)")
        print(f"  Unique flight_ids: {df['flight_id'].nunique():,}")
        print(f"  Failures: {n_fail:,} ({n_fail / len(df):.2%})")
        print("  By cause:", df["failure_cause"].value_counts().to_dict())
        per = df.groupby("scenario_name").agg(
            rows=("any_failure", "size"), failures=("any_failure", "sum"),
            rate=("any_failure", "mean"), mean_load_kN=("peak_load_kN", "mean"))
        print("\n" + per.round(3).to_string())
        if not used_multiplier:
            print(f"\n  Real-world failure rate (scenario-weighted): "
                  f"{real_world_failure_rate(df):.5f}")
        if n_fail < min_failures:
            print(f"  WARNING: fewer than {min_failures} failures; "
                  f"increase n_samples or the multiplier range.")
    return df


if __name__ == "__main__":
    generate_dataset()
