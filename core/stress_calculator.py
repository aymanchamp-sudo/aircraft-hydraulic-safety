# Stress and deformation calculations
# core/stress_calculator.py

"""
Stress Calculator Module
========================
Standalone functions for stress calculations and safety analysis.

These functions work on Beam objects and return engineering results.
Keeping calculations separate from the Beam class follows the
Single Responsibility Principle — Beam knows its geometry,
StressCalculator knows how to analyze it.
"""

import numpy as np
from core.beam import Beam


def compute_stress_distribution_along_span(beam):
    """
    Compute stress distribution along the entire beam span.
    
    Returns stress at the top surface (maximum compressive stress)
    at every x-station. This is the critical stress for failure analysis.
    
    Parameters:
    -----------
    beam : Beam object (with loads already applied)
    
    Returns:
    --------
    dict with keys:
        'x'             : spanwise positions (m)
        'moment'        : bending moments (N·m)
        'stress_top'    : stress at top surface (Pa)
        'stress_bot'    : stress at bottom surface (Pa)
        'stress_abs'    : absolute maximum stress at each section (Pa)
        'deflection'    : lateral deflection (m)
        'safety_factor' : local safety factor at each section
    """
    if beam.moment is None:
        raise RuntimeError("No loads applied to beam.")
    
    stress_abs = np.maximum(np.abs(beam.stress_top), np.abs(beam.stress_bot))
    
    # Safety factor at each cross-section
    # Avoid division by zero at tip (where stress = 0)
    sf = np.where(
        stress_abs > 1e-6,
        beam.material.yield_strength / stress_abs,
        float('inf')
    )
    
    return {
        'x': beam.x,
        'moment': beam.moment,
        'stress_top': beam.stress_top,
        'stress_bot': beam.stress_bot,
        'stress_abs': stress_abs,
        'deflection': beam.deflection,
        'safety_factor': sf,
    }


def find_safe_zones(beam, safety_threshold=1.5):
    """
    Find regions of the beam that are structurally safe.
    
    A "safe zone" is where the local safety factor >= threshold.
    
    This is used for hydraulic line placement:
    - SAFE zones → hydraulic lines can be placed here
    - UNSAFE zones → avoid placing hydraulic lines here
    
    Parameters:
    -----------
    beam : Beam (with loads applied)
    safety_threshold : float
        Minimum acceptable safety factor (default 1.5 = aerospace standard)
        
    Returns:
    --------
    dict:
        'safe_mask': boolean array (True = safe at that x station)
        'safe_x_range': (x_start, x_end) of safe region as fraction of L
        'safe_fraction': fraction of span that is safe (0 to 1)
    """
    results = compute_stress_distribution_along_span(beam)
    safe_mask = results['safety_factor'] >= safety_threshold
    
    safe_fraction = np.mean(safe_mask)  # fraction of span that's safe
    
    # Find first and last safe x positions
    safe_indices = np.where(safe_mask)[0]
    if len(safe_indices) > 0:
        x_safe_start = beam.x[safe_indices[0]]
        x_safe_end = beam.x[safe_indices[-1]]
    else:
        x_safe_start = x_safe_end = None
    
    return {
        'safe_mask': safe_mask,
        'x': beam.x,
        'safe_x_start': x_safe_start,
        'safe_x_end': x_safe_end,
        'safe_fraction': safe_fraction,
        'results': results
    }


def compute_hydraulic_line_stress(beam, line_x, line_y):
    """
    Compute the stress experienced by a hydraulic line at position (x, y).
    
    This is the core safety calculation:
    1. Find bending moment at the line's spanwise position x
    2. Apply σ = My/I to find stress at height y
    3. Compare to material yield strength
    
    Parameters:
    -----------
    beam : Beam (with loads applied)
    line_x : float
        Spanwise position of hydraulic line (meters from root)
    line_y : float
        Through-thickness position of line (meters from neutral axis)
        0 = neutral axis (safest), ±h/2 = outer surface (most dangerous)
    
    Returns:
    --------
    dict:
        'stress': stress at line position (Pa)
        'stress_mpa': stress in MPa (easier to read)
        'safety_factor': yield_strength / stress
        'is_safe': True if SF >= 1.5
        'danger_level': 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        'position_fraction': y / (h/2) — how far from neutral axis (0=center, 1=surface)
    """
    stress = beam.get_stress_at_position(line_x, line_y)
    abs_stress = abs(stress)
    
    sf = (beam.material.yield_strength / abs_stress 
          if abs_stress > 1e-6 else float('inf'))
    
    # Position fraction: 0 = exactly at neutral axis, 1 = at surface
    position_fraction = abs(line_y) / (beam.cs.h / 2)
    
    # Classify danger level based on safety factor
    if sf >= 3.0:
        danger = 'LOW'
    elif sf >= 2.0:
        danger = 'MEDIUM'
    elif sf >= 1.5:
        danger = 'ACCEPTABLE'
    elif sf >= 1.0:
        danger = 'HIGH'
    else:
        danger = 'CRITICAL'
    
    return {
        'stress': stress,
        'stress_mpa': abs_stress / 1e6,
        'safety_factor': sf,
        'is_safe': sf >= 1.5,
        'danger_level': danger,
        'position_fraction': position_fraction,
        'line_x': line_x,
        'line_y': line_y,
    }