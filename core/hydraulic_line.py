# Hydraulic line model
# core/hydraulic_line.py

"""
Hydraulic Line Model
=====================
Models a single hydraulic line routed inside an aircraft fin structure.

A hydraulic line is defined by:
1. Geometry: length, inner radius, wall thickness
2. Material: the tube material (often titanium or stainless steel)
3. Position: where it sits inside the fin cross-section and span
4. Operating conditions: pressure, temperature
5. Safety: combined stress analysis

Engineering Reference:
- MIL-H-5440: Military Specification for Aircraft Hydraulic Systems
- AS4395: Aerospace Standard for hydraulic fittings
- FAA AC 25.1435: Hydraulic systems certification
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class LineStatus(Enum):
    """
    Discrete failure/safety status for a hydraulic line.
    Based on safety factor thresholds used in aerospace certification.
    """
    OPTIMAL   = "OPTIMAL"    # SF >= 3.0  — well within safe limits
    SAFE      = "SAFE"       # SF >= 2.0  — acceptable margin
    MARGINAL  = "MARGINAL"   # SF >= 1.5  — minimum acceptable (aerospace std)
    UNSAFE    = "UNSAFE"     # SF >= 1.0  — plastic deformation imminent
    CRITICAL  = "CRITICAL"   # SF <  1.0  — FAILURE (stress exceeds yield)


class FailureMode(Enum):
    """
    Primary failure mode driving the safety assessment.
    """
    NONE          = "No dominant failure mode"
    HOOP_BURST    = "Internal pressure burst (hoop stress dominant)"
    BENDING_FATIGUE = "Cyclic bending fatigue"
    COMBINED_YIELD  = "Combined stress yielding (Von Mises)"
    RESONANCE       = "Structural resonance amplification"


@dataclass
class HydraulicLineGeometry:
    """
    Geometric properties of a hydraulic line (tube).
    
    All dimensions in SI units (meters).
    
    Standard aircraft hydraulic tubing (per AS4395):
    - Outer diameter: 6mm to 50mm
    - Wall thickness: 0.5mm to 3mm
    - Common: 12mm OD, 1mm wall → 10mm inner diameter
    """
    outer_diameter: float      # m — total tube outer diameter
    wall_thickness: float      # m — tube wall thickness
    
    def __post_init__(self):
        """Validate geometry after initialization."""
        if self.wall_thickness >= self.outer_diameter / 2:
            raise ValueError(
                f"Wall thickness ({self.wall_thickness*1000:.1f}mm) must be "
                f"less than outer radius ({self.outer_diameter*500:.1f}mm)"
            )
        if self.outer_diameter <= 0 or self.wall_thickness <= 0:
            raise ValueError("Dimensions must be positive")
    
    @property
    def inner_diameter(self):
        """Inner diameter = outer diameter - 2 × wall thickness."""
        return self.outer_diameter - 2 * self.wall_thickness
    
    @property
    def inner_radius(self):
        """Inner radius (used in pressure calculations)."""
        return self.inner_diameter / 2
    
    @property
    def outer_radius(self):
        """Outer radius."""
        return self.outer_diameter / 2
    
    @property
    def mean_radius(self):
        """Mean radius = average of inner and outer radius."""
        return (self.inner_radius + self.outer_radius) / 2
    
    @property
    def flow_area(self):
        """
        Internal flow cross-sectional area (m²).
        Determines how much fluid volume flows per unit time.
        Q = v × A_flow  (flow rate = velocity × area)
        """
        return np.pi * self.inner_radius**2
    
    @property
    def wall_area(self):
        """
        Cross-sectional area of the tube wall (m²).
        Used for axial stress calculations.
        """
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)
    
    @property
    def thin_wall_ratio(self):
        """
        t/r ratio — checks if thin-wall approximation is valid.
        Thin-wall formula (σ = Pr/t) valid when t/r < 0.1
        """
        return self.wall_thickness / self.inner_radius
    
    def __repr__(self):
        return (
            f"HydraulicLineGeometry:\n"
            f"  Outer diameter:  {self.outer_diameter*1000:.1f} mm\n"
            f"  Inner diameter:  {self.inner_diameter*1000:.1f} mm\n"
            f"  Wall thickness:  {self.wall_thickness*1000:.1f} mm\n"
            f"  Flow area:       {self.flow_area*1e6:.2f} mm²\n"
            f"  Thin-wall valid: {'✅' if self.thin_wall_ratio < 0.1 else '⚠️ Check'}"
        )


class HydraulicLineMaterial:
    """
    Material properties specific to hydraulic tubing.
    
    Hydraulic lines use different materials than the surrounding structure:
    - Titanium Ti-3Al-2.5V: most common in modern aircraft (corrosion resistant)
    - 21-6-9 Stainless Steel: older aircraft, high-temperature zones
    - CRES 304L: lower-pressure hydraulic systems
    
    Note: These are TUBE materials, not the same as structural beam material.
    """
    
    def __init__(self, name, youngs_modulus, yield_strength,
                 ultimate_strength, density,
                 fatigue_strength_coefficient,
                 fatigue_strength_exponent):
        self.name = name
        self.E = youngs_modulus
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength
        self.density = density
        
        # Basquin fatigue parameters (σ_f', b)
        # σ_f' = fatigue strength coefficient (Pa)
        # b    = fatigue strength exponent (dimensionless, typically -0.05 to -0.12)
        # Basquin's Law: σ_a = σ_f' × (2N_f)^b
        self.sigma_f_prime = fatigue_strength_coefficient
        self.b_fatigue = fatigue_strength_exponent
    
    def fatigue_life(self, stress_amplitude):
        """
        Estimate fatigue life in cycles using Basquin's Law.
        
        σ_a = σ_f' × (2N_f)^b
        Solving for N_f:
        N_f = 0.5 × (σ_a / σ_f')^(1/b)
        
        Parameters:
        -----------
        stress_amplitude : float
            Cyclic stress amplitude (Pa) — half the peak-to-peak range
            
        Returns:
        --------
        float : estimated cycles to crack initiation
                Returns inf if stress below endurance limit
        """
        if stress_amplitude <= 0:
            return float('inf')
        
        # Endurance limit ≈ 0.5 × ultimate strength (for steels)
        # For titanium: endurance limit ≈ 0.45 × ultimate
        endurance_limit = 0.45 * self.ultimate_strength
        if stress_amplitude < endurance_limit:
            return float('inf')  # Below endurance limit → infinite life
        
        # Basquin's Law
        two_N_f = (stress_amplitude / self.sigma_f_prime) ** (1 / self.b_fatigue)
        N_f = two_N_f / 2
        return max(N_f, 0)
    
    def cycles_per_flight(self, flight_hours=8.0, frequency=1.0):
        """
        Convert flight hours to stress cycles.
        
        Parameters:
        -----------
        flight_hours : float
            Hours of flight per mission
        frequency : float
            Stress cycling frequency (Hz)
            ~1 Hz for structural gusts, ~10 Hz for engine vibration
            
        Returns:
        --------
        int : stress cycles per flight
        """
        return int(flight_hours * 3600 * frequency)
    
    def __repr__(self):
        return (
            f"HydraulicLineMaterial: {self.name}\n"
            f"  Yield Strength: {self.yield_strength/1e6:.0f} MPa\n"
            f"  Fatigue Coeff:  {self.sigma_f_prime/1e6:.0f} MPa"
        )


# ─── Pre-defined hydraulic line materials ─────────────────────────────────────

# Titanium Ti-3Al-2.5V — most common aircraft hydraulic tubing
# Grade 9 titanium, half the density of steel, excellent fatigue life
TITANIUM_TUBE = HydraulicLineMaterial(
    name="Ti-3Al-2.5V (Grade 9)",
    youngs_modulus=100e9,          # 100 GPa
    yield_strength=520e6,           # 520 MPa
    ultimate_strength=620e6,        # 620 MPa
    density=4480,                   # kg/m³
    fatigue_strength_coefficient=900e6,   # σ_f' = 900 MPa
    fatigue_strength_exponent=-0.08       # b = -0.08
)

# 21-6-9 Stainless Steel — older aircraft standard
STAINLESS_STEEL_TUBE = HydraulicLineMaterial(
    name="21-6-9 Stainless Steel",
    youngs_modulus=193e9,           # 193 GPa
    yield_strength=690e6,           # 690 MPa  
    ultimate_strength=860e6,        # 860 MPa
    density=7900,                   # kg/m³
    fatigue_strength_coefficient=1100e6,  # σ_f'
    fatigue_strength_exponent=-0.09       # b
)


# Weights of the composite safety score the GA maximises (must sum to 1).
# Module-level so the ablation study can change them.
SCORE_WEIGHTS = {'sf': 0.50, 'fatigue': 0.30, 'position': 0.20}


class HydraulicLine:
    """
    Complete model of a hydraulic line inside an aircraft fin.
    
    Combines:
    - Geometry (tube dimensions)
    - Material (tube material properties)
    - Position (where the line sits in the fin)
    - Loading (operating pressure + structural bending stress from fin)
    - Safety analysis (combined stress, fatigue, safety factors)
    
    Position coordinate system:
    ┌─────────────────────────────────────┐
    │  x_start ───────────── x_end       │  ← spanwise extent of line
    │                                     │
    │  y_position (through thickness)     │
    │  0 = neutral axis (center)          │
    │  +h/2 = top surface                 │
    │  -h/2 = bottom surface              │
    └─────────────────────────────────────┘
    """
    
    def __init__(
        self,
        line_id: str,
        geometry: HydraulicLineGeometry,
        material: HydraulicLineMaterial,
        x_start: float,
        x_end: float,
        y_position: float,
        operating_pressure: float,
        system_name: str = "Unknown",
        stress_model: str = "lame"
    ):
        """
        Parameters:
        -----------
        line_id : str
            Unique identifier (e.g., 'HYD-A-1', 'HYD-B-MAIN')
        geometry : HydraulicLineGeometry
            Tube dimensional properties
        material : HydraulicLineMaterial
            Tube material properties
        x_start : float
            Spanwise start position (meters from root)
        x_end : float
            Spanwise end position (meters from root)
        y_position : float
            Through-thickness position (meters from neutral axis)
            0 = safest (neutral axis), ±h/2 = most dangerous (surface)
        operating_pressure : float
            Internal gauge pressure (Pa)
            Commercial: ~20.7 MPa (3000 PSI)
            Military: ~34.5 MPa (5000 PSI)
        system_name : str
            Which hydraulic system (A, B, or C for redundancy tracking)
        stress_model : str
            "lame" (default) = exact thick-wall cylinder stresses at the inner
            surface, with radial stress = -P.
            "thin_wall" = the original approximation (sigma = P*r/t, radial = 0),
            only valid when wall/inner-radius < 0.1.
            Our standard tube (12 mm OD, 1 mm wall) has wall/radius = 0.2, so
            the thin-wall formula under-predicts hoop stress by about 11%.
        """
        if stress_model not in ("lame", "thin_wall"):
            raise ValueError("stress_model must be 'lame' or 'thin_wall'")
        self.stress_model = stress_model
        self.id = line_id
        self.geometry = geometry
        self.material = material
        self.x_start = x_start
        self.x_end = x_end
        self.y_position = y_position
        self.pressure = operating_pressure
        self.system_name = system_name
        
        # Cache for analysis results
        self._last_analysis = None
    
    @property
    def x_center(self):
        """Spanwise center of the line."""
        return (self.x_start + self.x_end) / 2
    
    @property
    def length(self):
        """Length of the hydraulic line segment."""
        return self.x_end - self.x_start
    
    def compute_hoop_stress(self):
        """
        Hoop (circumferential) stress at the inner wall from internal pressure.

        Lame (default, exact for a thick-walled cylinder):
            sigma_hoop = P * (ro^2 + ri^2) / (ro^2 - ri^2)
        Thin-wall approximation (stress_model="thin_wall"):
            sigma_hoop = P * ri / t

        Returns:
        --------
        float : hoop stress (Pa), always positive (tensile)
        """
        P = self.pressure
        ri = self.geometry.inner_radius
        if self.stress_model == "thin_wall":
            return P * ri / self.geometry.wall_thickness
        ro = self.geometry.outer_radius
        return P * (ro**2 + ri**2) / (ro**2 - ri**2)

    def compute_axial_pressure_stress(self):
        """
        Axial stress from the pressure acting on the closed tube ends.

        Lame (closed ends): sigma_axial = P * ri^2 / (ro^2 - ri^2)
        Thin-wall:          sigma_axial = P * ri / (2 * t)

        Returns:
        --------
        float : axial pressure stress (Pa), tensile
        """
        P = self.pressure
        ri = self.geometry.inner_radius
        if self.stress_model == "thin_wall":
            return P * ri / (2 * self.geometry.wall_thickness)
        ro = self.geometry.outer_radius
        return P * ri**2 / (ro**2 - ri**2)

    def compute_radial_stress(self):
        """
        Radial stress at the inner wall: -P (compressive) in the Lame model,
        0 in the thin-wall approximation.
        """
        return 0.0 if self.stress_model == "thin_wall" else -self.pressure

    def compute_bending_stress(self, beam, use_x_center=True):
        """
        Bending stress experienced by the line from fin structural bending.
        
        The line is physically attached to the fin structure.
        When the fin bends, the line bends with it.
        Stress depends on: spanwise position x (determines moment M)
                           through-thickness position y (determines σ = My/I)
        
        Parameters:
        -----------
        beam : Beam object with loads applied
        use_x_center : bool
            If True, evaluate at line's spanwise center
            If False, evaluate at the most stressed point (x_start, i.e., root-nearest)
            
        Returns:
        --------
        float : bending stress experienced by the line (Pa)
        """
        if use_x_center:
            x_eval = self.x_center
        else:
            # More conservative: evaluate at the root-nearest end
            # (higher bending moment closer to root)
            x_eval = self.x_start
        
        stress = beam.get_stress_at_position(x_eval, self.y_position)
        return abs(stress)  # Return magnitude for safety calculations
    
    def compute_von_mises_stress(self, beam):
        """
        Von Mises equivalent stress at the inner wall, full 3-D form:

            s_t = hoop stress
            s_z = bending stress + axial pressure stress  (along the tube)
            s_r = radial stress (-P in the Lame model, 0 in thin-wall)

            sigma_VM = sqrt(0.5 * ((s_t-s_z)^2 + (s_z-s_r)^2 + (s_r-s_t)^2))

        With s_r = 0 this reduces to the old biaxial form
        sqrt(s_z^2 + s_t^2 - s_z*s_t).

        Parameters:
        -----------
        beam : Beam with loads applied

        Returns:
        --------
        float : Von Mises stress (Pa)
        """
        s_t = self.compute_hoop_stress()
        s_z = self.compute_bending_stress(beam) + self.compute_axial_pressure_stress()
        s_r = self.compute_radial_stress()
        return float(np.sqrt(0.5 * ((s_t - s_z)**2 + (s_z - s_r)**2 + (s_r - s_t)**2)))

    def compute_safety_factor(self, beam):
        """
        Compute safety factor based on Von Mises stress.
        
        SF = Yield Strength / Von Mises Stress
        
        Aerospace requirement: SF >= 1.5
        
        Parameters:
        -----------
        beam : Beam with loads applied
        
        Returns:
        --------
        float : safety factor (dimensionless)
        """
        sigma_vm = self.compute_von_mises_stress(beam)
        
        if sigma_vm < 1e-6:
            return float('inf')
        
        return self.material.yield_strength / sigma_vm
    
    def compute_fatigue_life(self, beam):
        """
        Estimate fatigue life in flight cycles.
        
        Assumptions:
        - Bending stress cycles between 0 and max each flight
        - Stress amplitude = max_bending / 2 (half the peak-to-peak)
        - Mean stress correction (Goodman) not applied (conservative)
        - 1 stress cycle per gust encounter (≈ 1 Hz during cruise)
        - Average flight: 8 hours at cruise
        
        Returns:
        --------
        dict with 'cycles_to_crack', 'flights_to_crack', 'years_to_crack'
        """
        sigma_bending = self.compute_bending_stress(beam)
        
        # Stress amplitude (half range for fully reversed cycling)
        sigma_amplitude = sigma_bending / 2
        
        # Cycles to crack initiation
        N_crack = self.material.fatigue_life(sigma_amplitude)
        
        # Convert to flights (assuming 1 Hz gust frequency, 8 hr flights)
        cycles_per_flight = self.material.cycles_per_flight(
            flight_hours=8.0, frequency=1.0
        )
        
        flights_to_crack = N_crack / cycles_per_flight if cycles_per_flight > 0 else float('inf')
        
        # Typical commercial aircraft: 500 flights/year
        years_to_crack = flights_to_crack / 500
        
        return {
            'stress_amplitude_mpa': sigma_amplitude / 1e6,
            'cycles_to_crack': N_crack,
            'flights_to_crack': flights_to_crack,
            'years_to_crack': years_to_crack,
            'endurance_limited': N_crack == float('inf'),
        }
    
    def determine_status(self, safety_factor):
        """
        Convert numerical safety factor to LineStatus enum.
        
        Parameters:
        -----------
        safety_factor : float
        
        Returns:
        --------
        LineStatus enum value
        """
        if safety_factor >= 3.0:
            return LineStatus.OPTIMAL
        elif safety_factor >= 2.0:
            return LineStatus.SAFE
        elif safety_factor >= 1.5:
            return LineStatus.MARGINAL
        elif safety_factor >= 1.0:
            return LineStatus.UNSAFE
        else:
            return LineStatus.CRITICAL
    
    def determine_primary_failure_mode(self, beam):
        """
        Identify which stress is dominant so engineers know what to fix.
        
        Returns:
        --------
        FailureMode enum value
        """
        sigma_bending = self.compute_bending_stress(beam)
        sigma_hoop = self.compute_hoop_stress()
        
        if sigma_hoop > sigma_bending * 2:
            return FailureMode.HOOP_BURST
        elif sigma_bending > sigma_hoop * 2:
            return FailureMode.BENDING_FATIGUE
        else:
            return FailureMode.COMBINED_YIELD
    
    def full_safety_analysis(self, beam):
        """
        Comprehensive safety analysis of this hydraulic line.
        
        This is the primary method used by the Monte Carlo engine
        and optimization algorithm. Returns a complete dict of
        all safety metrics.
        
        Parameters:
        -----------
        beam : Beam with loads applied
        
        Returns:
        --------
        dict : Complete safety assessment with all metrics
        """
        # ── Stress components ──────────────────────────────────────────
        sigma_bending = self.compute_bending_stress(beam)
        sigma_hoop = self.compute_hoop_stress()
        sigma_axial_p = self.compute_axial_pressure_stress()
        sigma_vm = self.compute_von_mises_stress(beam)
        
        # ── Safety factors ─────────────────────────────────────────────
        sf_overall = self.compute_safety_factor(beam)
        sf_hoop = self.material.yield_strength / sigma_hoop if sigma_hoop > 0 else float('inf')
        sf_bending = self.material.yield_strength / sigma_bending if sigma_bending > 0 else float('inf')
        
        # ── Fatigue ────────────────────────────────────────────────────
        fatigue = self.compute_fatigue_life(beam)
        
        # ── Status classification ──────────────────────────────────────
        status = self.determine_status(sf_overall)
        failure_mode = self.determine_primary_failure_mode(beam)
        
        # ── Position metrics ───────────────────────────────────────────
        # How close to neutral axis? (0=exactly on NA, 1=at surface)
        h_half = beam.cs.h / 2
        position_fraction = abs(self.y_position) / h_half
        
        # ── Composite safety score (0 to 100) ─────────────────────────
        # Higher = safer. Used by genetic algorithm for optimization.
        safety_score = self._compute_safety_score(
            sf_overall, fatigue['flights_to_crack'], position_fraction
        )
        
        result = {
            # Identity
            'line_id': self.id,
            'system_name': self.system_name,
            
            # Position
            'x_start': self.x_start,
            'x_end': self.x_end,
            'x_center': self.x_center,
            'y_position': self.y_position,
            'position_fraction': position_fraction,  # 0=NA, 1=surface
            
            # Operating conditions
            'pressure_mpa': self.pressure / 1e6,
            
            # Stress components (MPa)
            'sigma_bending_mpa': sigma_bending / 1e6,
            'sigma_hoop_mpa': sigma_hoop / 1e6,
            'sigma_axial_pressure_mpa': sigma_axial_p / 1e6,
            'sigma_von_mises_mpa': sigma_vm / 1e6,
            
            # Safety factors
            'sf_overall': sf_overall,
            'sf_hoop': sf_hoop,
            'sf_bending': sf_bending,
            
            # Fatigue
            'fatigue_cycles': fatigue['cycles_to_crack'],
            'fatigue_flights': fatigue['flights_to_crack'],
            'fatigue_years': fatigue['years_to_crack'],
            'endurance_limited': fatigue['endurance_limited'],
            
            # Classification
            'status': status.value,
            'primary_failure_mode': failure_mode.value,
            'is_safe': sf_overall >= 1.5,
            
            # Composite score
            'safety_score': safety_score,
        }
        
        self._last_analysis = result
        return result
    
    def _compute_safety_score(self, sf, fatigue_flights, position_fraction):
        """
        Compute composite safety score from 0 (worst) to 100 (best).
        
        Combines:
        - Safety factor (weighted 50%)
        - Fatigue life (weighted 30%)
        - Distance from neutral axis (weighted 20%)
        
        This score is what the genetic algorithm maximizes.
        
        Parameters:
        -----------
        sf : float — safety factor
        fatigue_flights : float — estimated flights to fatigue crack
        position_fraction : float — 0=NA, 1=surface (lower = safer)
        
        Returns:
        --------
        float : safety score 0-100
        """
        # SF component: 0 at SF=1.0, 100 at SF>=3.0
        sf_score = np.clip((sf - 1.0) / 2.0 * 100, 0, 100)
        
        # Fatigue component: 0 below 1000 flights, 100 above 50000 flights
        if fatigue_flights == float('inf'):
            fatigue_score = 100.0
        else:
            fatigue_score = np.clip(
                (np.log10(max(fatigue_flights, 1)) - 3) / 2 * 100, 0, 100
            )
        
        # Position component: 100 at neutral axis, 0 at surface
        position_score = (1 - position_fraction) * 100
        
        # Weighted combination
        w = SCORE_WEIGHTS
        composite = (w['sf'] * sf_score + w['fatigue'] * fatigue_score
                     + w['position'] * position_score)
        
        return round(composite, 2)
    
    def __repr__(self):
        return (
            f"HydraulicLine '{self.id}' [{self.system_name}]\n"
            f"  Position: x=[{self.x_start:.1f}m, {self.x_end:.1f}m], "
            f"y={self.y_position*1000:.1f}mm from NA\n"
            f"  Geometry: OD={self.geometry.outer_diameter*1000:.0f}mm, "
            f"t={self.geometry.wall_thickness*1000:.1f}mm\n"
            f"  Pressure: {self.pressure/1e6:.1f} MPa\n"
            f"  Material: {self.material.name}"
        )