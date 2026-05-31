# core/hydraulic_configuration.py

"""
Hydraulic Configuration Manager
=================================
Manages a complete set of hydraulic lines inside the aircraft fin.

A "configuration" is the complete routing design — all lines,
their positions, and their collective safety assessment.

This class:
1. Holds multiple HydraulicLine objects
2. Evaluates the entire configuration's safety
3. Checks for physical conflicts (lines overlapping)
4. Checks for redundancy requirements
5. Generates configuration reports
6. Interfaces with the genetic algorithm optimizer

Real aircraft typically have:
- System A: Primary flight control hydraulics
- System B: Secondary/backup flight controls
- System C: Tertiary (brakes, landing gear, spoilers)
All three must be routed so a single failure cannot damage all three.
"""

import numpy as np
from typing import List, Dict, Optional
from core.hydraulic_line import HydraulicLine, LineStatus, TITANIUM_TUBE
from core.hydraulic_line import HydraulicLineGeometry


class HydraulicConfiguration:
    """
    A complete hydraulic routing configuration for the aircraft fin.
    
    Contains multiple hydraulic lines and evaluates their collective safety.
    """
    
    def __init__(self, config_id: str, lines: List[HydraulicLine]):
        """
        Parameters:
        -----------
        config_id : str
            Unique identifier for this configuration
            (e.g., 'CONFIG_001', 'GENETIC_GEN5_IND3')
        lines : list of HydraulicLine
            All hydraulic lines in this configuration
        """
        self.config_id = config_id
        self.lines = lines
        self._analysis_results = None
    
    def analyze(self, beam):
        """
        Perform full safety analysis on all lines in this configuration.
        
        Parameters:
        -----------
        beam : Beam with loads applied
        
        Returns:
        --------
        dict : Complete configuration analysis results
        """
        line_results = []
        for line in self.lines:
            result = line.full_safety_analysis(beam)
            line_results.append(result)
        
        # ── Aggregate metrics ──────────────────────────────────────────
        safety_scores = [r['safety_score'] for r in line_results]
        safety_factors = [r['sf_overall'] for r in line_results]
        vm_stresses = [r['sigma_von_mises_mpa'] for r in line_results]
        
        n_safe = sum(1 for r in line_results if r['is_safe'])
        n_total = len(line_results)
        
        # ── Configuration-level safety checks ─────────────────────────
        separation_ok = self._check_line_separation()
        overlap_ok = self._check_no_overlap()
        redundancy_ok = self._check_redundancy_spacing()
        
        config_result = {
            'config_id': self.config_id,
            'n_lines': n_total,
            'n_safe': n_safe,
            'fraction_safe': n_safe / n_total if n_total > 0 else 0,
            
            # Aggregate stress metrics
            'mean_safety_score': np.mean(safety_scores),
            'min_safety_score': np.min(safety_scores),
            'min_safety_factor': np.min(safety_factors),
            'max_von_mises_mpa': np.max(vm_stresses),
            'mean_von_mises_mpa': np.mean(vm_stresses),
            
            # Configuration-level checks
            'separation_ok': separation_ok,
            'no_overlap': overlap_ok,
            'redundancy_ok': redundancy_ok,
            'all_systems_safe': all(r['is_safe'] for r in line_results),
            
            # Detailed line results
            'line_results': line_results,
        }
        
        # ── Overall configuration score (used by genetic algorithm) ───
        config_result['config_score'] = self._compute_config_score(config_result)
        
        self._analysis_results = config_result
        return config_result
    
    def _check_line_separation(self, min_separation_mm=20):
        """
        Check that lines are separated by at least min_separation_mm.
        
        WHY: Lines too close together create stress concentration
        at their supports. Also, a fitting failure on one line
        must not spray hydraulic fluid onto adjacent lines.
        
        Parameters:
        -----------
        min_separation_mm : float
            Minimum center-to-center separation in mm
            
        Returns:
        --------
        bool : True if all lines are adequately separated
        """
        min_sep = min_separation_mm / 1000  # Convert to meters
        
        for i, line_a in enumerate(self.lines):
            for j, line_b in enumerate(self.lines):
                if i >= j:
                    continue
                separation = abs(line_a.y_position - line_b.y_position)
                if separation < min_sep:
                    return False
        return True
    
    def _check_no_overlap(self):
        """
        Check that no two lines physically overlap in cross-section.
        
        Two lines overlap if their outer surfaces intersect.
        This is a physical impossibility — would mean two tubes
        occupy the same space.
        
        Returns:
        --------
        bool : True if no overlaps exist
        """
        for i, line_a in enumerate(self.lines):
            for j, line_b in enumerate(self.lines):
                if i >= j:
                    continue
                
                # Physical separation (center-to-center)
                separation = abs(line_a.y_position - line_b.y_position)
                
                # Minimum non-overlap distance = sum of outer radii
                min_dist = (line_a.geometry.outer_radius + 
                            line_b.geometry.outer_radius)
                
                if separation < min_dist:
                    return False
        return True
    
    def _check_redundancy_spacing(self, min_spacing_fraction=0.3):
        """
        Check that lines from DIFFERENT hydraulic systems are
        adequately separated (redundancy requirement).
        
        FAA requires that a single bird strike, shrapnel event,
        or structural failure cannot damage all hydraulic systems.
        Lines from different systems must be separated by at least
        30% of the fin height.
        
        Parameters:
        -----------
        min_spacing_fraction : float
            Minimum separation as fraction of fin height
            
        Returns:
        --------
        bool : True if redundancy spacing is adequate
        """
        system_groups = {}
        for line in self.lines:
            sys_name = line.system_name
            if sys_name not in system_groups:
                system_groups[sys_name] = []
            system_groups[sys_name].append(line.y_position)
        
        if len(system_groups) <= 1:
            return True  # Only one system — no cross-system check needed
        
        # Check all pairs of different systems
        systems = list(system_groups.keys())
        for i in range(len(systems)):
            for j in range(i+1, len(systems)):
                sys_a_positions = system_groups[systems[i]]
                sys_b_positions = system_groups[systems[j]]
                
                # Find minimum separation between any line in sys A
                # and any line in sys B
                min_separation = min(
                    abs(ya - yb)
                    for ya in sys_a_positions
                    for yb in sys_b_positions
                )
                
                # This check needs beam height — we'll use a threshold
                # In Phase 8 (genetic algorithm), we'll pass beam.cs.h
                # For now, check absolute separation > 30mm
                if min_separation < 0.030:
                    return False
        
        return True
    
    def _compute_config_score(self, analysis):
        """
        Compute overall configuration score (0-100).
        
        Penalizes:
        - Lines with low safety scores
        - Physical violations (overlap, separation)
        - Poor redundancy
        
        Parameters:
        -----------
        analysis : dict from self.analyze()
        
        Returns:
        --------
        float : configuration score 0-100
        """
        base_score = analysis['mean_safety_score']
        
        # Penalty for minimum safety score (weakest link matters most)
        weakest_link_penalty = (100 - analysis['min_safety_score']) * 0.3
        
        # Hard penalties for physical violations
        overlap_penalty = 0 if analysis['no_overlap'] else 50
        separation_penalty = 0 if analysis['separation_ok'] else 20
        redundancy_penalty = 0 if analysis['redundancy_ok'] else 30
        
        score = base_score - weakest_link_penalty - overlap_penalty - \
                separation_penalty - redundancy_penalty
        
        return max(0.0, min(100.0, round(score, 2)))
    
    def print_report(self, analysis=None):
        """
        Print a formatted engineering report for this configuration.
        """
        if analysis is None:
            raise RuntimeError("Run analyze(beam) first.")
        
        print("\n" + "═"*65)
        print(f"  HYDRAULIC CONFIGURATION REPORT: {self.config_id}")
        print("═"*65)
        print(f"  Lines analyzed:     {analysis['n_lines']}")
        print(f"  Lines safe (SF≥1.5):{analysis['n_safe']} / {analysis['n_lines']}")
        print(f"  Config score:       {analysis['config_score']:.1f} / 100")
        print(f"  Min safety factor:  {analysis['min_safety_factor']:.2f}")
        print(f"  Max Von Mises:      {analysis['max_von_mises_mpa']:.1f} MPa")
        print()
        print(f"  Physical checks:")
        print(f"    No overlap:       {'✅' if analysis['no_overlap'] else '❌'}")
        print(f"    Separation OK:    {'✅' if analysis['separation_ok'] else '❌'}")
        print(f"    Redundancy OK:    {'✅' if analysis['redundancy_ok'] else '❌'}")
        print()
        print(f"  {'Line ID':<15} {'y-pos (mm)':>10} {'σ_VM (MPa)':>12} "
              f"{'SF':>8} {'Score':>8} {'Status':>12}")
        print("  " + "-"*65)
        
        for r in analysis['line_results']:
            print(f"  {r['line_id']:<15} "
                  f"{r['y_position']*1000:>9.1f}mm "
                  f"{r['sigma_von_mises_mpa']:>10.1f} "
                  f"{r['sf_overall']:>8.2f} "
                  f"{r['safety_score']:>7.1f} "
                  f"{r['status']:>12}")
        
        print("═"*65)


def create_standard_configuration(beam, pressure=20.7e6):
    """
    Create a realistic 3-line hydraulic configuration.
    
    Based on typical transport aircraft fin hydraulic routing:
    - System A: Near neutral axis (primary flight controls)
    - System B: Offset toward top (backup)
    - System C: Offset toward bottom (tertiary)
    
    Parameters:
    -----------
    beam : Beam object (used to determine fin geometry)
    pressure : float — hydraulic operating pressure (Pa)
    
    Returns:
    --------
    HydraulicConfiguration
    """
    h = beam.cs.h   # Fin cross-section height
    L = beam.L      # Fin span
    
    # Standard tube geometry (12mm OD titanium)
    std_geometry = HydraulicLineGeometry(
        outer_diameter=0.012,    # 12mm OD
        wall_thickness=0.001     # 1mm wall → 10mm ID
    )
    
    lines = [
        # System A — near neutral axis (safest position)
        HydraulicLine(
            line_id="HYD-A-MAIN",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.2,
            x_end=L * 0.9,
            y_position=0.0,          # Exactly on neutral axis
            operating_pressure=pressure,
            system_name="System_A"
        ),
        
        # System B — offset 25% toward top
        HydraulicLine(
            line_id="HYD-B-MAIN",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.2,
            x_end=L * 0.9,
            y_position=h * 0.25,     # 25% of height above NA
            operating_pressure=pressure,
            system_name="System_B"
        ),
        
        # System C — offset 25% toward bottom
        HydraulicLine(
            line_id="HYD-C-MAIN",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.2,
            x_end=L * 0.9,
            y_position=-h * 0.25,    # 25% of height below NA
            operating_pressure=pressure,
            system_name="System_C"
        ),
    ]
    
    return HydraulicConfiguration("STANDARD_CONFIG", lines)


def create_dangerous_configuration(beam, pressure=20.7e6):
    """
    Create a deliberately dangerous configuration for comparison.
    
    This shows what BAD hydraulic routing looks like:
    - All lines near surfaces (high bending stress)
    - Minimal separation
    - Poor redundancy spacing
    
    Used for educational comparison and to test the optimizer.
    """
    h = beam.cs.h
    L = beam.L
    
    std_geometry = HydraulicLineGeometry(
        outer_diameter=0.012,
        wall_thickness=0.001
    )
    
    lines = [
        HydraulicLine(
            line_id="HYD-A-BAD",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.1,
            x_end=L * 0.95,
            y_position=h * 0.42,    # Very close to top surface — DANGEROUS
            operating_pressure=pressure,
            system_name="System_A"
        ),
        HydraulicLine(
            line_id="HYD-B-BAD",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.1,
            x_end=L * 0.95,
            y_position=h * 0.38,    # Also near top — DANGEROUS + close to A
            operating_pressure=pressure,
            system_name="System_B"
        ),
        HydraulicLine(
            line_id="HYD-C-BAD",
            geometry=std_geometry,
            material=TITANIUM_TUBE,
            x_start=0.1,
            x_end=L * 0.95,
            y_position=-h * 0.40,   # Near bottom — dangerous
            operating_pressure=pressure,
            system_name="System_C"
        ),
    ]
    
    return HydraulicConfiguration("DANGEROUS_CONFIG", lines)