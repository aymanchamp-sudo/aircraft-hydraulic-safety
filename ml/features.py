# ml/features.py
"""
The ONE place that defines what the failure-prediction models may see.

The label `any_failure` is computed from the physics OUTPUTS (beam safety factor
below 1, or a tube safety factor below 1). Feeding any of those outputs - or
anything derived from them - to the model leaks the answer and gives a
meaningless AUC of 1.0. The original project made exactly that mistake.

Rule: a feature must be known BEFORE the physics runs, i.e. it is something a
dashboard user can set with a slider.

Deliberately excluded
---------------------
* scenario one-hot : turbulence_intensity already identifies the scenario
  one-to-one, and the dashboard has no scenario input.
* sideslip_deg     : an upstream cause of the load (already captured by
  peak_load_kN) and not a dashboard input.
* stress_multiplier, base_peak_load_kN, scenario_weight, flight_id :
  bookkeeping columns of the dataset generator.
"""

import numpy as np
import pandas as pd

LABEL = "any_failure"

BASE_FEATURES = [
    "peak_load_kN",          # total lateral force on the fin (kN)
    "turbulence_intensity",  # m/s RMS
    "spar_height_m",         # spar cross-section height (m)
    "pressure_mpa",          # hydraulic pressure (MPa)
    "wall_thickness_mm",     # tube wall thickness (mm)
    "line1_y_frac",          # tube positions as a fraction of h/2
    "line2_y_frac",
    "line3_y_frac",
]
ENGINEERED_FEATURES = [
    "load_x_turbulence",
    "log_peak_load",
    "load_squared",
    "max_abs_y_frac",
]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# Columns produced BY the physics (or derived from the label). Never inputs.
OUTCOME_COLUMNS = [
    "max_beam_stress_mpa", "beam_safety_factor", "min_line_sf",
    "max_line_vm_mpa", "config_score", "n_unsafe_lines",
    "any_failure", "failure_cause",
    # historical outcome-derived columns from the old pipeline
    "tip_deflection_mm", "root_moment_kNm", "stress_utilization", "sf_margin",
]
_OUTCOME_PREFIXES = ("beam_", "min_line", "max_line", "config_", "n_unsafe",
                     "any_failure", "failure_", "stress_util", "sf_margin",
                     "tip_defl", "root_moment")


def add_engineered_features(df):
    """Return a copy of df with the engineered feature columns added."""
    out = df.copy()
    out["load_x_turbulence"] = out["peak_load_kN"] * out["turbulence_intensity"]
    out["log_peak_load"] = np.log1p(out["peak_load_kN"])
    out["load_squared"] = out["peak_load_kN"] ** 2
    out["max_abs_y_frac"] = out[["line1_y_frac", "line2_y_frac",
                                 "line3_y_frac"]].abs().max(axis=1)
    return out


def assert_no_leakage(feature_names):
    """Raise if any model input is an outcome of the physics."""
    bad = [f for f in feature_names
           if f in OUTCOME_COLUMNS or f.startswith(_OUTCOME_PREFIXES)
           or (f.startswith("line") and f.endswith(("_vm_mpa", "_sf", "_score",
                                                    "_status", "_fatigue_flights")))]
    if bad:
        raise AssertionError(f"Outcome columns used as ML features: {bad}")
    unknown = [f for f in feature_names if f not in FEATURES]
    if unknown:
        raise AssertionError(f"Features not in the approved list: {unknown}")


def build_feature_row(peak_load_kN, turbulence_intensity, spar_height_m,
                      line_y_m, pressure_mpa, wall_thickness_mm):
    """
    Build ONE model-input row from real dashboard values (no zero filling).

    line_y_m : the three tube y-positions in metres from the neutral axis.
    """
    half = spar_height_m / 2.0
    fr = [y / half for y in line_y_m]
    row = {
        "peak_load_kN": peak_load_kN,
        "turbulence_intensity": turbulence_intensity,
        "spar_height_m": spar_height_m,
        "pressure_mpa": pressure_mpa,
        "wall_thickness_mm": wall_thickness_mm,
        "line1_y_frac": fr[0], "line2_y_frac": fr[1], "line3_y_frac": fr[2],
    }
    return add_engineered_features(pd.DataFrame([row]))[FEATURES]
