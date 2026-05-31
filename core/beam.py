# Aircraft beam/fin structural model
# core/beam.py

"""
Aircraft Fin Beam Model
========================
Models the aircraft vertical fin as an Euler-Bernoulli cantilever beam.

Engineering Setup:
    - Beam fixed at root (x = 0), free at tip (x = L)
    - Lateral aerodynamic force F applied at tip
    - Rectangular cross-section (simplified fin spar)
    - Uniform material and cross-section along length

Coordinate System:
    x: span direction (0 = root, L = tip)
    y: through-thickness direction (0 = neutral axis, ±h/2 = surfaces)
    z: chord direction (not used in 1D beam theory)

        z (chord)
        │
        │     _____________________ TIP (x = L)
        │    │                     │
        └────│─────────────────────│────→ x (span)
             │      FIN SPAR       │
             │_____________________│
            ROOT (x = 0)
            ↑ Fixed here

"""

import numpy as np
from core.material import Material, ALUMINUM_2024_T3


class BeamCrossSection:
    """
    Rectangular cross-section geometry.
    
    In reality, aircraft spars have I-beam or box-beam cross-sections.
    Rectangular is our simplification — same physics, simpler math.
    
    Cross-section view:
    
        ┌───── b (width) ─────┐
        ┌─────────────────────┐ ─ ─
        │                     │    │
        │         +h/2        │    │
        │    - - - - - - -    │    │ h (height)
        │    NEUTRAL AXIS     │    │
        │    - - - - - - -    │    │
        │         -h/2        │    │
        │                     │    │
        └─────────────────────┘ ─ ─
    """
    
    def __init__(self, width, height):
        """
        Parameters:
        -----------
        width : float
            b — cross-section width in meters (chord direction)
        height : float
            h — cross-section height in meters (thickness direction)
            This is the dimension that matters most for bending stiffness
        """
        self.b = width    # m
        self.h = height   # m
    
    @property
    def area(self):
        """
        Cross-sectional area A = b × h (m²)
        Used for axial stress and weight calculations.
        """
        return self.b * self.h
    
    @property
    def second_moment_of_area(self):
        """
        Second Moment of Area (also called Moment of Inertia) I = bh³/12 (m⁴)
        
        WHY h³? Because:
        - Fibers farther from neutral axis carry more stress
        - The contribution to bending resistance grows as y² (from stress distribution)
        - Integrating y² · dA across the cross-section gives bh³/12
        
        Physical meaning: doubling height → 8× more bending stiffness!
        This is why engineers make beams TALL, not just THICK.
        """
        return (self.b * self.h**3) / 12
    
    @property
    def section_modulus(self):
        """
        Section Modulus Z = I / (h/2) = bh²/6 (m³)
        
        Useful shortcut: σ_max = M / Z
        The higher Z, the more bending moment a section can handle
        before reaching yield stress.
        """
        return self.second_moment_of_area / (self.h / 2)
    
    def __repr__(self):
        I = self.second_moment_of_area
        return (
            f"Cross-Section: {self.b*100:.1f}cm × {self.h*100:.1f}cm\n"
            f"  Area:                 {self.area*1e4:.2f} cm²\n"
            f"  Second Moment (I):    {I*1e8:.4f} × 10⁻⁸ m⁴\n"
            f"  Section Modulus (Z):  {self.section_modulus*1e6:.4f} cm³"
        )


class Beam:
    """
    Cantilever beam model for aircraft fin structural analysis.
    
    This class:
    1. Stores geometry, material, and loading
    2. Computes bending moment distribution M(x)
    3. Computes stress distribution σ(x, y)
    4. Computes deflection distribution δ(x)
    5. Computes natural frequency (vibration)
    6. Evaluates structural safety
    """
    
    def __init__(self, length, cross_section, material, n_stations=100):
        """
        Parameters:
        -----------
        length : float
            L — span of the beam/fin (meters)
            Typical aircraft fin: 3 to 8 meters
        cross_section : BeamCrossSection
            Geometry of the cross-section
        material : Material
            Material properties (E, yield strength, etc.)
        n_stations : int
            Number of points along span for calculations
            More stations → smoother plots (100 is usually enough)
        """
        self.L = length
        self.cs = cross_section
        self.material = material
        self.n = n_stations
        
        # Create array of x positions along the beam span
        # np.linspace(0, L, n) creates n evenly-spaced points from 0 to L
        # x[0] = root (0.0), x[-1] = tip (L)
        self.x = np.linspace(0, length, n_stations)
        
        # Load storage
        # We'll accumulate loads and compute combined effects
        self._tip_load = 0.0          # Point load at tip (N)
        self._distributed_load = 0.0  # Uniform load per unit length (N/m)
        
        # Results storage (filled after apply_loads is called)
        self.moment = None       # Bending moment M(x) in N·m
        self.shear = None        # Shear force V(x) in N
        self.deflection = None   # Deflection δ(x) in meters
        self.stress_top = None   # Stress at top surface σ(x, +h/2) in Pa
        self.stress_bot = None   # Stress at bottom surface σ(x, -h/2) in Pa
    
    def apply_tip_load(self, force):
        """
        Apply a lateral point load at the tip of the beam.
        
        This represents:
        - Aerodynamic side force during yaw maneuver
        - Engine side-wash on vertical fin
        - Gust lateral load
        
        Parameters:
        -----------
        force : float
            Force in Newtons (positive = in +z direction)
            Typical aircraft fin load: 10,000 to 100,000 N
        """
        self._tip_load = force
        self._calculate_all()  # Recalculate everything when load changes
        return self
    
    def apply_distributed_load(self, load_per_unit_length):
        """
        Apply a uniform distributed load along the entire span.
        
        This represents:
        - Aerodynamic pressure distribution on fin surface
        - More realistic than a single tip load
        
        Parameters:
        -----------
        load_per_unit_length : float
            Load intensity in N/m
            e.g., 5000 N/m means 5000 Newtons per meter of span
        """
        self._distributed_load = load_per_unit_length
        self._calculate_all()
        return self
    
    def _calculate_moment_distribution(self):
        """
        Calculate bending moment M at every position x along the span.
        
        PHYSICS:
        For a cantilever with:
        (a) Tip point load F:
            M(x) = F × (L - x)
            Maximum at root: M_root = F × L
            Zero at tip:     M_tip  = 0
        
        (b) Uniform distributed load w (N/m):
            M(x) = (w/2) × (L - x)²
            Maximum at root: M_root = w × L² / 2
        
        (c) Combined:
            M(x) = F(L-x) + (w/2)(L-x)²
        
        This is basic statics — sum of moments about any cross-section.
        
        Returns:
        --------
        np.ndarray : bending moment at each x station (N·m)
        """
        F = self._tip_load
        w = self._distributed_load
        
        # (L - x) appears frequently — compute once
        remaining_length = self.L - self.x  # shape: (n,)
        
        # Bending moment from tip load: M = F(L-x)
        M_tip_load = F * remaining_length
        
        # Bending moment from distributed load: M = w(L-x)²/2
        M_distributed = (w / 2) * remaining_length**2
        
        # Superposition: total moment = sum of individual moments
        # (Valid for linear elastic materials — our assumption)
        M_total = M_tip_load + M_distributed
        
        return M_total
    
    def _calculate_shear_distribution(self):
        """
        Calculate shear force V at every position x.
        
        Shear force is the derivative of bending moment:
        V(x) = dM/dx = -F - w(L-x)
        
        (The negative sign appears because of sign convention)
        
        For tip load only: V = F everywhere (constant shear)
        For distributed load: V varies linearly
        """
        F = self._tip_load
        w = self._distributed_load
        remaining_length = self.L - self.x
        
        V = F + w * remaining_length  # magnitude of shear
        return V
    
    def _calculate_stress_distribution(self):
        """
        Calculate bending stress at top and bottom surfaces.
        
        FORMULA: σ(x, y) = M(x) × y / I
        
        At top surface:    y = +h/2  →  σ_top = +M × (h/2) / I
        At bottom surface: y = -h/2  →  σ_bot = -M × (h/2) / I
        
        Sign convention:
        - Positive M (sagging) → top in compression, bottom in tension
        - We'll track magnitude for safety analysis
        
        Note: M(x) is our array of bending moments at each station
              I and h/2 are constants (uniform cross-section)
        
        Returns:
        --------
        stress_top, stress_bot : np.ndarray
            Stress at top and bottom surfaces at each x station (Pa)
        """
        I = self.cs.second_moment_of_area
        y_max = self.h / 2  # h/2 is shorthand for cs.h/2
        
        # σ = My/I applied to each x station
        stress_top = self.moment * y_max / I   # compression (negative)
        stress_bot = self.moment * (-y_max) / I  # tension (positive)
        
        return stress_top, stress_bot
    
    def _calculate_deflection(self):
        """
        Calculate lateral deflection δ at every position x.
        
        For cantilever with tip load F:
            δ(x) = F·x² (3L - x) / (6EI)
            
        Physical meaning:
        - δ increases from 0 at root (fixed) to maximum at tip
        - Stiffer material (higher E) → less deflection
        - Stiffer cross-section (higher I) → less deflection
        - Much longer beam → massively more deflection (cubic relationship!)
        
        For distributed load w (N/m):
            δ(x) = w·x² (6L² - 4Lx + x²) / (24EI)
        
        Returns:
        --------
        np.ndarray : deflection at each x station (meters)
        """
        E = self.material.E
        I = self.cs.second_moment_of_area
        F = self._tip_load
        w = self._distributed_load
        EI = E * I  # Combined bending stiffness (N·m²)
        
        x = self.x  # positions array
        L = self.L
        
        # Deflection from tip load
        delta_tip_load = (F * x**2 * (3*L - x)) / (6 * EI)
        
        # Deflection from distributed load
        delta_distributed = (w * x**2 * (6*L**2 - 4*L*x + x**2)) / (24 * EI)
        
        return delta_tip_load + delta_distributed
    
    def _calculate_all(self):
        """
        Master calculation function — called whenever loads change.
        Runs all calculations in the correct order.
        
        Order matters:
        1. Moments first (needed for stress calculation)
        2. Shear forces
        3. Stress from moments
        4. Deflection (independent of stress)
        """
        self.moment = self._calculate_moment_distribution()
        self.shear = self._calculate_shear_distribution()
        self.stress_top, self.stress_bot = self._calculate_stress_distribution()
        self.deflection = self._calculate_deflection()
    
    def get_stress_at_position(self, x_pos, y_pos):
        """
        Get stress at any specific (x, y) location in the beam.
        
        This is what we'll use for hydraulic line safety analysis:
        "What stress does the hydraulic line experience at its location?"
        
        Parameters:
        -----------
        x_pos : float
            Position along span (meters, 0=root, L=tip)
        y_pos : float
            Position through thickness (meters, 0=neutral axis, ±h/2=surface)
            
        Returns:
        --------
        float : stress at that point (Pa)
        """
        if self.moment is None:
            raise RuntimeError("No loads applied yet. Call apply_tip_load() first.")
        
        # Find index of nearest x station
        idx = np.argmin(np.abs(self.x - x_pos))
        M_at_x = self.moment[idx]
        
        # σ = My/I
        I = self.cs.second_moment_of_area
        sigma = M_at_x * y_pos / I
        
        return sigma
    
    def get_stress_profile_at_x(self, x_pos, n_points=50):
        """
        Get complete stress distribution across the cross-section at position x.
        
        Useful for:
        - Plotting stress profiles
        - Finding where hydraulic lines would be safest
        - Understanding which zones are most dangerous
        
        Parameters:
        -----------
        x_pos : float
            Spanwise position (meters)
        n_points : int
            Number of y-points in the profile
            
        Returns:
        --------
        y_array : np.ndarray — y positions (from -h/2 to +h/2)
        sigma_array : np.ndarray — stress at each y position (Pa)
        """
        y_array = np.linspace(-self.cs.h/2, self.cs.h/2, n_points)
        idx = np.argmin(np.abs(self.x - x_pos))
        M_at_x = self.moment[idx]
        I = self.cs.second_moment_of_area
        
        sigma_array = M_at_x * y_array / I
        return y_array, sigma_array
    
    def max_stress(self):
        """
        Find the maximum absolute stress anywhere in the beam.
        This occurs at the root, at the outermost fiber.
        
        Returns:
        --------
        float : maximum stress magnitude (Pa)
        """
        if self.stress_top is None:
            raise RuntimeError("No loads applied yet.")
        return np.max(np.abs(self.stress_top))
    
    def tip_deflection(self):
        """
        Return the deflection at the beam tip (maximum deflection).
        
        Returns:
        --------
        float : tip deflection (meters)
        """
        if self.deflection is None:
            raise RuntimeError("No loads applied yet.")
        return self.deflection[-1]  # Last element = tip position
    
    def natural_frequency(self):
        """
        Estimate fundamental natural frequency of the cantilever beam.
        
        Formula: f = (1.875)² / (2π) × √(EI / ρAL⁴)
        
        WHY THIS MATTERS:
        If wind gust frequency ≈ natural frequency → RESONANCE → catastrophic failure
        Boeing 747 fin natural frequency ≈ 1-3 Hz
        Turbulence typically 0.1-10 Hz → must ensure no overlap
        
        Returns:
        --------
        float : natural frequency (Hz)
        """
        E = self.material.E
        I = self.cs.second_moment_of_area
        rho = self.material.density
        A = self.cs.area
        L = self.L
        
        # 1.875 is the first mode shape constant for cantilever beams
        # This comes from solving the beam vibration differential equation
        beta = 1.875104  # First eigenvalue
        
        omega = (beta**2) * np.sqrt(E * I / (rho * A * L**4))
        frequency = omega / (2 * np.pi)
        
        return frequency
    
    def structural_summary(self):
        """
        Print a complete structural analysis summary.
        Used for quick engineering review.
        """
        if self.moment is None:
            print("No loads applied yet.")
            return
        
        max_s = self.max_stress()
        tip_d = self.tip_deflection()
        sf = self.material.safety_factor(max_s)
        fn = self.natural_frequency()
        
        print("=" * 60)
        print(f"STRUCTURAL ANALYSIS SUMMARY — {self.material.name}")
        print("=" * 60)
        print(f"Beam Length:           {self.L:.2f} m")
        print(f"Cross-Section:         {self.cs.b*100:.1f}cm × {self.cs.h*100:.1f}cm")
        print(f"Tip Load Applied:      {self._tip_load/1000:.2f} kN")
        print(f"Distributed Load:      {self._distributed_load:.1f} N/m")
        print("-" * 60)
        print(f"Max Bending Moment:    {np.max(self.moment)/1000:.2f} kN·m (at root)")
        print(f"Max Stress:            {max_s/1e6:.2f} MPa")
        print(f"Yield Strength:        {self.material.yield_strength/1e6:.2f} MPa")
        print(f"Safety Factor:         {sf:.2f} {'✅' if sf >= 1.5 else '❌ UNSAFE!'}")
        print(f"Tip Deflection:        {tip_d*1000:.2f} mm ({tip_d/self.L*100:.3f}% of span)")
        print(f"Natural Frequency:     {fn:.2f} Hz")
        print("=" * 60)
    
    # Shorthand property for convenience
    @property
    def h(self):
        return self.cs.h


if __name__ == "__main__":
    from core.material import ALUMINUM_2024_T3
    
    print("Building aircraft fin beam model...")
    print()
    
    # Define cross-section: 15cm wide × 20cm tall (typical fin spar)
    cs = BeamCrossSection(width=0.15, height=0.20)
    print(cs)
    print()
    
    # Create beam: 4m tall fin (Airbus A320-size vertical fin ≈ 5.87m)
    beam = Beam(
        length=4.0,
        cross_section=cs,
        material=ALUMINUM_2024_T3,
        n_stations=200
    )
    
    # Apply realistic lateral aerodynamic load
    # Typical fin design load: ~50 kN (50,000 N)
    # This comes from:
    # - Aircraft weight × lateral g-load factor
    # - Aerodynamic side force coefficient × dynamic pressure × fin area
    beam.apply_tip_load(50000)  # 50 kN tip load
    beam.apply_distributed_load(3000)   # 3 kN/m distributed
    
    # Print complete structural summary
    beam.structural_summary()