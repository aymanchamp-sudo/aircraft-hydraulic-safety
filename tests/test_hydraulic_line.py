# tests/test_hydraulic_line.py
"""
Unit tests for the hydraulic tube stress model.

Reference values are hand calculations for the standard tube:
12 mm outer diameter, 1 mm wall (ri = 5 mm, ro = 6 mm), 3000 PSI = 20.7 MPa.
  thin-wall hoop  = P*ri/t                        = 103.5 MPa
  Lame hoop       = P*(ro^2+ri^2)/(ro^2-ri^2)     = 114.8 MPa
  Lame axial      = P*ri^2/(ro^2-ri^2)            =  47.05 MPa
"""

import pytest

from core.material import ALUMINUM_2024_T3
from core.beam import Beam, BeamCrossSection
from core.hydraulic_line import (HydraulicLine, HydraulicLineGeometry,
                                  TITANIUM_TUBE)

P_3000_PSI = 20.7e6
P_5000_PSI = 34.5e6


def make_line(y=0.0, pressure=P_3000_PSI, wall=0.001, model="lame"):
    geom = HydraulicLineGeometry(outer_diameter=0.012, wall_thickness=wall)
    return HydraulicLine("T", geom, TITANIUM_TUBE, 0.2, 3.6, y, pressure,
                         "System_A", stress_model=model)


@pytest.fixture
def loaded_beam():
    """4 m x 0.15 m x 0.20 m Al 2024-T3 fin, 50 kN tip + 3 kN/m."""
    beam = Beam(length=4.0, cross_section=BeamCrossSection(0.15, 0.20),
                material=ALUMINUM_2024_T3, n_stations=100)
    beam.apply_tip_load(50e3)
    beam.apply_distributed_load(3e3)
    return beam


class TestPressureStress:
    def test_lame_hoop_stress(self):
        assert make_line().compute_hoop_stress() / 1e6 == pytest.approx(114.8, abs=0.1)

    def test_thin_wall_hoop_stress(self):
        line = make_line(model="thin_wall")
        assert line.compute_hoop_stress() / 1e6 == pytest.approx(103.5, abs=0.1)

    def test_lame_exceeds_thin_wall(self):
        assert make_line().compute_hoop_stress() > make_line(model="thin_wall").compute_hoop_stress()

    def test_lame_axial_and_radial(self):
        line = make_line()
        assert line.compute_axial_pressure_stress() / 1e6 == pytest.approx(47.05, abs=0.05)
        assert line.compute_radial_stress() == pytest.approx(-P_3000_PSI)

    def test_thin_wall_radial_is_zero(self):
        assert make_line(model="thin_wall").compute_radial_stress() == 0.0

    def test_models_converge_for_thin_walls(self):
        # wall/radius ~ 0.02: both models should agree within 2 %.
        a = make_line(wall=0.0001).compute_hoop_stress()
        b = make_line(wall=0.0001, model="thin_wall").compute_hoop_stress()
        assert a == pytest.approx(b, rel=0.02)

    def test_invalid_model_rejected(self):
        with pytest.raises(ValueError):
            make_line(model="nonsense")


class TestVonMisesAndSafety:
    def test_neutral_axis_von_mises_above_old_biaxial_value(self, loaded_beam):
        old = make_line(model="thin_wall").compute_von_mises_stress(loaded_beam) / 1e6
        new = make_line().compute_von_mises_stress(loaded_beam) / 1e6
        assert old == pytest.approx(89.6, abs=0.2)   # value in the original README run
        assert new > old

    def test_thin_wall_matches_biaxial_formula(self, loaded_beam):
        line = make_line(y=0.05, model="thin_wall")
        s1 = line.compute_bending_stress(loaded_beam) + line.compute_hoop_stress() / 2
        s2 = line.compute_hoop_stress()
        expected = (s1**2 + s2**2 - s1 * s2) ** 0.5
        assert line.compute_von_mises_stress(loaded_beam) == pytest.approx(expected, rel=1e-9)

    def test_safety_factor_falls_with_pressure(self, loaded_beam):
        low = make_line(pressure=P_3000_PSI).compute_safety_factor(loaded_beam)
        high = make_line(pressure=P_5000_PSI).compute_safety_factor(loaded_beam)
        assert high < low

    def test_surface_tube_less_safe_than_neutral_axis_tube(self, loaded_beam):
        centre = make_line(y=0.0).compute_safety_factor(loaded_beam)
        surface = make_line(y=0.09).compute_safety_factor(loaded_beam)
        assert surface < centre

    def test_thinner_wall_less_safe(self, loaded_beam):
        thick = make_line(wall=0.0013).compute_safety_factor(loaded_beam)
        thin = make_line(wall=0.0007).compute_safety_factor(loaded_beam)
        assert thin < thick

    def test_full_analysis_keys_unchanged(self, loaded_beam):
        r = make_line().full_safety_analysis(loaded_beam)
        for key in ("line_id", "y_position", "sigma_von_mises_mpa", "sf_overall",
                    "safety_score", "status", "fatigue_flights"):
            assert key in r
