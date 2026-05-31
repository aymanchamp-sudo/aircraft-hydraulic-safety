# Unit tests for beam model
# tests/test_beam.py

"""
Unit tests for the Beam structural model.

WHY UNIT TESTS?
Every engineering simulation must be validated against known solutions.
We test against textbook cantilever beam formulas.
If these tests pass, our code is mathematically correct.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from core.material import ALUMINUM_2024_T3, Material
from core.beam import Beam, BeamCrossSection


class TestBeamCrossSection:
    """Tests for cross-section geometric properties."""
    
    def test_area(self):
        """Area = width × height."""
        cs = BeamCrossSection(width=0.1, height=0.2)
        assert abs(cs.area - 0.02) < 1e-10
    
    def test_second_moment_of_area(self):
        """I = bh³/12. For b=0.1, h=0.2: I = 0.1×0.008/12 = 6.667e-5 m⁴"""
        cs = BeamCrossSection(width=0.1, height=0.2)
        I_expected = (0.1 * 0.2**3) / 12
        assert abs(cs.second_moment_of_area - I_expected) < 1e-15
    
    def test_taller_section_much_stiffer(self):
        """Doubling height should give 8× more I (cubic relationship)."""
        cs1 = BeamCrossSection(width=0.1, height=0.1)
        cs2 = BeamCrossSection(width=0.1, height=0.2)
        ratio = cs2.second_moment_of_area / cs1.second_moment_of_area
        assert abs(ratio - 8.0) < 1e-10


class TestBeamStress:
    """Tests for stress calculations using analytical solutions."""
    
    @pytest.fixture
    def simple_beam(self):
        """Standard test beam: 2m long, 10×10cm, aluminum."""
        cs = BeamCrossSection(width=0.10, height=0.10)
        return Beam(length=2.0, cross_section=cs, material=ALUMINUM_2024_T3,
                    n_stations=1000)
    
    def test_tip_load_moment_at_root(self, simple_beam):
        """For tip load F, root moment should be exactly F×L."""
        F = 10000  # 10 kN
        L = simple_beam.L
        simple_beam.apply_tip_load(F)
        
        M_root_expected = F * L
        M_root_computed = simple_beam.moment[0]
        
        assert abs(M_root_computed - M_root_expected) / M_root_expected < 0.001
    
    def test_tip_moment_is_zero(self, simple_beam):
        """Bending moment at free tip must be exactly zero."""
        simple_beam.apply_tip_load(10000)
        M_tip = simple_beam.moment[-1]
        assert abs(M_tip) < 1.0  # less than 1 N·m (numerical precision)
    
    def test_max_stress_formula(self, simple_beam):
        """σ_max = M_max × (h/2) / I — validate against formula."""
        F = 10000
        simple_beam.apply_tip_load(F)
        
        L = simple_beam.L
        h = simple_beam.cs.h
        I = simple_beam.cs.second_moment_of_area
        
        sigma_expected = (F * L) * (h/2) / I
        sigma_computed = simple_beam.max_stress()
        
        error = abs(sigma_computed - sigma_expected) / sigma_expected
        assert error < 0.001, f"Stress error {error:.4%} exceeds tolerance"
    
    def test_tip_deflection_formula(self, simple_beam):
        """δ_tip = FL³/(3EI) — validate against analytical formula."""
        F = 10000
        simple_beam.apply_tip_load(F)
        
        L = simple_beam.L
        E = simple_beam.material.E
        I = simple_beam.cs.second_moment_of_area
        
        delta_expected = (F * L**3) / (3 * E * I)
        delta_computed = simple_beam.tip_deflection()
        
        error = abs(delta_computed - delta_expected) / delta_expected
        assert error < 0.001, f"Deflection error {error:.4%} exceeds tolerance"
    
    def test_stress_zero_at_neutral_axis(self, simple_beam):
        """Stress must be zero exactly at neutral axis (y=0)."""
        simple_beam.apply_tip_load(10000)
        stress_at_na = simple_beam.get_stress_at_position(x_pos=1.0, y_pos=0.0)
        assert abs(stress_at_na) < 1e-6
    
    def test_stress_linear_in_y(self, simple_beam):
        """Stress must increase linearly from neutral axis to surface."""
        simple_beam.apply_tip_load(10000)
        x_pos = 0.5  # 0.5m from root
        h = simple_beam.cs.h
        
        s1 = abs(simple_beam.get_stress_at_position(x_pos, h/4))   # quarter height
        s2 = abs(simple_beam.get_stress_at_position(x_pos, h/2))   # surface
        
        # Stress at surface should be exactly 2× stress at quarter-height
        ratio = s2 / s1
        assert abs(ratio - 2.0) < 0.01, f"Linearity violated: ratio = {ratio:.4f}"
    
    def test_safety_factor_safe_material(self, simple_beam):
        """Small load should give high safety factor."""
        simple_beam.apply_tip_load(1000)  # Very small load
        sf = simple_beam.material.safety_factor(simple_beam.max_stress())
        assert sf > 3.0, "Small load should give large safety factor"
    
    def test_higher_load_reduces_safety_factor(self, simple_beam):
        """Higher load must reduce the safety factor."""
        simple_beam.apply_tip_load(10000)
        sf1 = simple_beam.material.safety_factor(simple_beam.max_stress())
        
        simple_beam.apply_tip_load(50000)
        sf2 = simple_beam.material.safety_factor(simple_beam.max_stress())
        
        assert sf2 < sf1, "Higher load should reduce safety factor"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])