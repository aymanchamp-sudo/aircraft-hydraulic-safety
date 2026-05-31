# Material property definitions
# core/material.py

"""
Material Property Definitions
==============================
This module defines material classes for structural simulation.

Engineering context:
- Aircraft structures are primarily aluminum alloy (Al 2024-T3 or Al 7075-T6)
- Modern aircraft use carbon fiber composites, but we start with aluminum
- Material properties come from aerospace material standards (MIL-HDBK-5)

Why a class instead of just variables?
- Groups related properties together
- Can create multiple materials and compare them
- Can add methods like safety_factor_check() that use the properties
"""

import numpy as np


class Material:
    """
    Represents a structural material with its mechanical properties.
    
    All units follow SI (International System):
    - Stress/Modulus: Pascals (Pa) = N/m²
    - Density: kg/m³
    - Lengths: meters (m)
    """
    
    def __init__(self, name, youngs_modulus, yield_strength, 
                 ultimate_strength, density, poissons_ratio=0.33):
        """
        Initialize a material with its mechanical properties.
        
        Parameters:
        -----------
        name : str
            Material name for labeling plots/reports
        youngs_modulus : float
            E — stiffness (Pa). Aluminum ≈ 70 GPa = 70e9 Pa
        yield_strength : float  
            σ_y — elastic limit (Pa). Al 2024-T3 ≈ 345 MPa = 345e6 Pa
        ultimate_strength : float
            σ_u — fracture stress (Pa). Al 2024-T3 ≈ 483 MPa = 483e6 Pa
        density : float
            ρ — mass per volume (kg/m³). Aluminum ≈ 2700 kg/m³
        poissons_ratio : float
            ν — lateral contraction ratio (dimensionless). Aluminum ≈ 0.33
            (When you stretch a material lengthwise, it contracts sideways)
        """
        self.name = name
        self.E = youngs_modulus          # Pa
        self.yield_strength = yield_strength      # Pa
        self.ultimate_strength = ultimate_strength  # Pa
        self.density = density           # kg/m³
        self.nu = poissons_ratio         # dimensionless
        
        # Derived property: Shear Modulus
        # G = E / (2 * (1 + ν))
        # Governs how material resists twisting (torsion)
        self.G = self.E / (2 * (1 + self.nu))
    
    def safety_factor(self, applied_stress):
        """
        Calculate the safety factor for a given applied stress.
        
        Safety Factor = Yield Strength / Applied Stress
        
        Aerospace standard: SF must be >= 1.5
        SF < 1.0 → material has already yielded (failure!)
        SF = 1.5 → standard aerospace design margin
        SF > 3.0 → over-designed (wasteful weight)
        
        Parameters:
        -----------
        applied_stress : float or np.ndarray
            Stress to evaluate (Pa)
            
        Returns:
        --------
        float : safety factor (dimensionless)
        """
        # Avoid division by zero
        if np.any(np.abs(applied_stress) < 1e-10):
            return float('inf')
        return self.yield_strength / np.max(np.abs(applied_stress))
    
    def is_safe(self, applied_stress, required_sf=1.5):
        """
        Check whether stress is within safe limits.
        
        Parameters:
        -----------
        applied_stress : float or np.ndarray
            Applied stress (Pa)
        required_sf : float
            Required safety factor (default 1.5 for aerospace)
            
        Returns:
        --------
        bool : True if safe, False if unsafe
        """
        return self.safety_factor(applied_stress) >= required_sf
    
    def stress_ratio(self, applied_stress):
        """
        Stress Ratio = Applied Stress / Yield Strength
        
        Used in fatigue analysis and damage tolerance.
        Ratio > 1.0 → yielded (bad)
        Ratio < 0.67 → under design limit (good with SF=1.5)
        
        Parameters:
        -----------
        applied_stress : float
            Applied stress in Pa
            
        Returns:
        --------
        float : stress ratio (dimensionless)
        """
        return np.abs(applied_stress) / self.yield_strength
    
    def __repr__(self):
        """
        String representation — what prints when you type material in terminal.
        """
        return (
            f"Material: {self.name}\n"
            f"  Young's Modulus:    {self.E/1e9:.1f} GPa\n"
            f"  Yield Strength:     {self.yield_strength/1e6:.1f} MPa\n"
            f"  Ultimate Strength:  {self.ultimate_strength/1e6:.1f} MPa\n"
            f"  Density:            {self.density:.0f} kg/m³\n"
            f"  Shear Modulus:      {self.G/1e9:.1f} GPa\n"
            f"  Poisson's Ratio:    {self.nu}"
        )


# =============================================================================
# PRE-DEFINED AEROSPACE MATERIALS
# Properties from MIL-HDBK-5 (Metallic Materials for Aerospace Structures)
# =============================================================================

# Aluminum 2024-T3 — most common aircraft structural aluminum
# Used in: fuselage skins, wing lower surfaces (tension-dominated)
# "T3" means solution heat-treated and cold-worked
ALUMINUM_2024_T3 = Material(
    name="Aluminum 2024-T3",
    youngs_modulus=73.1e9,     # 73.1 GPa
    yield_strength=345e6,       # 345 MPa (tensile)
    ultimate_strength=483e6,    # 483 MPa
    density=2780,               # kg/m³
    poissons_ratio=0.33
)

# Aluminum 7075-T6 — high-strength aircraft aluminum  
# Used in: wing upper surfaces, spars (compression-dominated)
# Stronger but less corrosion-resistant than 2024
ALUMINUM_7075_T6 = Material(
    name="Aluminum 7075-T6",
    youngs_modulus=71.7e9,      # 71.7 GPa
    yield_strength=503e6,        # 503 MPa
    ultimate_strength=572e6,     # 572 MPa
    density=2810,                # kg/m³
    poissons_ratio=0.33
)

# Titanium Ti-6Al-4V — used for hydraulic fittings, high-temp zones
# Much stronger and lighter than steel, but expensive
TITANIUM_6AL_4V = Material(
    name="Titanium Ti-6Al-4V",
    youngs_modulus=113.8e9,     # 113.8 GPa
    yield_strength=880e6,        # 880 MPa
    ultimate_strength=950e6,     # 950 MPa
    density=4430,                # kg/m³
    poissons_ratio=0.342
)

# Mild Steel — for comparison/educational purposes
MILD_STEEL = Material(
    name="Mild Steel",
    youngs_modulus=200e9,       # 200 GPa (very stiff)
    yield_strength=250e6,        # 250 MPa (but weaker than aerospace alloys)
    ultimate_strength=400e6,     # 400 MPa
    density=7850,                # kg/m³ (nearly 3× heavier than aluminum!)
    poissons_ratio=0.29
)


if __name__ == "__main__":
    # Test the material module
    print("=" * 60)
    print("AEROSPACE MATERIALS DATABASE")
    print("=" * 60)
    
    materials = [ALUMINUM_2024_T3, ALUMINUM_7075_T6, TITANIUM_6AL_4V, MILD_STEEL]
    
    for mat in materials:
        print(f"\n{mat}")
        
        # Test safety factor at 200 MPa
        test_stress = 200e6  # 200 MPa
        sf = mat.safety_factor(test_stress)
        safe = mat.is_safe(test_stress)
        ratio = mat.stress_ratio(test_stress)
        
        print(f"  At 200 MPa applied stress:")
        print(f"    Safety Factor:  {sf:.2f} ({'✅ SAFE' if safe else '❌ UNSAFE'})")
        print(f"    Stress Ratio:   {ratio:.3f}")
    
    # Engineering insight: compare strength-to-weight ratios
    print("\n" + "=" * 60)
    print("SPECIFIC STRENGTH COMPARISON (Yield Strength / Density)")
    print("Higher = stronger for less weight — critical for aerospace")
    print("=" * 60)
    
    for mat in materials:
        specific_strength = mat.yield_strength / mat.density
        print(f"  {mat.name:30s}: {specific_strength/1000:.1f} kN·m/kg")
