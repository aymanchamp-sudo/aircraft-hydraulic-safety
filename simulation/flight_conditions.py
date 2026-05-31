# Flight condition definitions
# simulation/flight_conditions.py

"""
Flight Condition Library
=========================
Defines all flight scenarios used in the Monte Carlo simulation.

Each flight condition represents a different phase or scenario
the aircraft may encounter. Monte Carlo samples from these
conditions randomly (or according to flight frequency weights).

Real aircraft encounter different conditions with different frequencies:
- Cruise: 80% of flight time
- Climb/descent: 15%
- Approach: 4%
- Severe turbulence: <1%

We weight our Monte Carlo sampling to reflect this.
"""

import numpy as np
import warnings

from dataclasses import dataclass
from typing import List, Tuple, Optional

from simulation.wind_model import (
    AtmosphericConditions,
    AerodynamicModel,
    TurbulenceModel,
    GustEventModel,
    FlightSimulation
)


@dataclass
class FlightScenario:
    """
    Complete definition of a flight scenario for simulation.
    
    A scenario = atmospheric conditions + turbulence parameters
                 + gust parameters + sampling weight
    """
    name: str
    description: str
    conditions: AtmosphericConditions
    
    # Turbulence parameters
    turbulence_intensity: float    # m/s RMS
    turbulence_correlation: float  # temporal correlation
    
    # Gust parameters
    mean_gust_velocity: float      # m/s
    std_gust_velocity: float       # m/s
    gusts_per_hour: float
    
    # Sampling weight (proportional to frequency in real operations)
    weight: float
    
    # Structural parameters for beam loading
    mean_sideslip_deg: float = 0.0
    std_sideslip_deg: float = 1.0


def build_flight_scenarios():
    """
    Create the complete library of flight scenarios.
    
    Based on aircraft operational specifications and FAA advisory circulars.
    Each scenario represents a distinct operational phase or condition.
    
    Returns:
    --------
    list of FlightScenario objects
    """
    scenarios = [
        
        FlightScenario(
            name="CRUISE_LIGHT_TURB",
            description="Normal cruise, light turbulence (most common)",
            conditions=AtmosphericConditions(
                altitude_m=10668, airspeed_ms=248.1,
                sideslip_angle_deg=0.0
            ),
            turbulence_intensity=1.5,
            turbulence_correlation=0.6,
            mean_gust_velocity=4.0,
            std_gust_velocity=2.0,
            gusts_per_hour=2.0,
            weight=0.45,    # 45% of all flights
            std_sideslip_deg=0.8,
        ),
        
        FlightScenario(
            name="CRUISE_MOD_TURB",
            description="Cruise, moderate turbulence (common in winter/mountains)",
            conditions=AtmosphericConditions(
                altitude_m=10668, airspeed_ms=248.1,
                sideslip_angle_deg=0.0
            ),
            turbulence_intensity=4.0,
            turbulence_correlation=0.7,
            mean_gust_velocity=8.0,
            std_gust_velocity=4.0,
            gusts_per_hour=8.0,
            weight=0.20,    # 20% of flights
            std_sideslip_deg=1.5,
        ),
        
        FlightScenario(
            name="CRUISE_SEVERE_TURB",
            description="Cruise, severe turbulence (rare — CAT/storm encounter)",
            conditions=AtmosphericConditions(
                altitude_m=10668, airspeed_ms=231.0,  # Reduced speed for turbulence
                sideslip_angle_deg=0.0
            ),
            turbulence_intensity=9.0,
            turbulence_correlation=0.8,
            mean_gust_velocity=14.0,
            std_gust_velocity=6.0,
            gusts_per_hour=20.0,
            weight=0.04,    # 4% of flights
            std_sideslip_deg=3.0,
        ),
        
        FlightScenario(
            name="APPROACH_CROSSWIND",
            description="Approach with crosswind (requires sideslip correction)",
            conditions=AtmosphericConditions(
                altitude_m=300, airspeed_ms=72.0,
                sideslip_angle_deg=5.0   # Crabbing into crosswind
            ),
            turbulence_intensity=2.5,
            turbulence_correlation=0.4,
            mean_gust_velocity=7.0,
            std_gust_velocity=4.0,
            gusts_per_hour=15.0,   # Wind shear near ground
            weight=0.12,
            std_sideslip_deg=3.0,
        ),
        
        FlightScenario(
            name="CLIMB_SHEAR",
            description="Climb through wind shear layer",
            conditions=AtmosphericConditions(
                altitude_m=3000, airspeed_ms=154.3,
                sideslip_angle_deg=0.0
            ),
            turbulence_intensity=3.0,
            turbulence_correlation=0.5,
            mean_gust_velocity=10.0,
            std_gust_velocity=5.0,
            gusts_per_hour=10.0,
            weight=0.10,
            std_sideslip_deg=1.5,
        ),
        
        FlightScenario(
            name="MANEUVER_YAW",
            description="Deliberate yaw maneuver (ATC avoidance, approach correction)",
            conditions=AtmosphericConditions(
                altitude_m=5000, airspeed_ms=180.0,
                sideslip_angle_deg=3.0
            ),
            turbulence_intensity=1.0,
            turbulence_correlation=0.3,
            mean_gust_velocity=3.0,
            std_gust_velocity=1.5,
            gusts_per_hour=1.0,
            weight=0.05,
            std_sideslip_deg=4.0,   # High sideslip variability — the maneuver
        ),
        
        FlightScenario(
            name="EXTREME_GUST",
            description="Extreme discrete gust (design case, very rare)",
            conditions=AtmosphericConditions(
                altitude_m=7000, airspeed_ms=205.0,
                sideslip_angle_deg=0.0
            ),
            turbulence_intensity=2.0,
            turbulence_correlation=0.5,
            mean_gust_velocity=17.1,   # FAR 25 design gust velocity
            std_gust_velocity=3.0,
            gusts_per_hour=3.0,
            weight=0.04,
            std_sideslip_deg=2.0,
        ),
    ]
    
    # Verify weights approximately sum to 1
    total_weight = sum(s.weight for s in scenarios)
    if not 0.95 <= total_weight <= 1.05:
        warnings.warn(f"Scenario weights sum to {total_weight:.2f}, expected ~1.0")
    
    return scenarios


class FlightConditionSampler:
    """
    Samples flight scenarios and loads for Monte Carlo simulation.
    
    For each Monte Carlo iteration:
    1. Randomly select a flight scenario (weighted by frequency)
    2. Sample specific conditions within that scenario (sideslip, etc.)
    3. Run the flight simulation to get peak structural load
    4. Return the load for structural/hydraulic analysis
    
    This is the "outer loop" of the Monte Carlo engine.
    """
    
    def __init__(
        self,
        aero_model: AerodynamicModel,
        scenarios: List[FlightScenario],
        seed: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        aero_model : AerodynamicModel — fin aerodynamic parameters
        scenarios : list of FlightScenario
        seed : int or None
        """
        self.aero = aero_model
        self.scenarios = scenarios
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute normalized weights for sampling
        weights = np.array([s.weight for s in scenarios])
        self.weights_normalized = weights / weights.sum()
    
    def sample_scenario(self):
        """
        Randomly select one flight scenario weighted by frequency.
        
        Returns:
        --------
        FlightScenario : selected scenario
        """
        idx = self.rng.choice(len(self.scenarios), p=self.weights_normalized)
        return self.scenarios[idx]
    
    def sample_flight_load(self, flight_hours: float = 4.0):
        """
        Sample a single flight's peak structural load.
        
        Process:
        1. Select scenario (weighted random)
        2. Sample sideslip angle (Gaussian around scenario mean)
        3. Build simulation objects
        4. Run simulation → get peak load
        
        Parameters:
        -----------
        flight_hours : float
        
        Returns:
        --------
        dict:
          'scenario_name': which scenario was selected
          'peak_load': maximum lateral force during this flight (N)
          'conditions': AtmosphericConditions used
          'sideslip_deg': sampled sideslip angle
        """
        scenario = self.sample_scenario()
        
        # Sample sideslip from Gaussian centered on scenario mean
        sideslip = self.rng.normal(
            scenario.mean_sideslip_deg,
            scenario.std_sideslip_deg
        )
        
        # Create conditions with sampled sideslip
        conditions = AtmosphericConditions(
            altitude_m=scenario.conditions.altitude_m,
            airspeed_ms=scenario.conditions.airspeed_ms,
            sideslip_angle_deg=sideslip
        )
        
        # Build models with scenario parameters
        turb_model = TurbulenceModel(
            turbulence_intensity=scenario.turbulence_intensity,
            correlation_length=scenario.turbulence_correlation,
            seed=int(self.rng.integers(0, 2**31))
        )
        
        gust_model = GustEventModel(
            mean_gust_velocity=scenario.mean_gust_velocity,
            std_gust_velocity=scenario.std_gust_velocity,
            mean_gusts_per_hour=scenario.gusts_per_hour,
            seed=int(self.rng.integers(0, 2**31))
        )
        
        # Create and run simulation
        sim = FlightSimulation(
            aero_model=self.aero,
            turbulence_model=turb_model,
            gust_model=gust_model,
            conditions=conditions,
            flight_duration_hours=flight_hours,
            dt=2.0,   # 2-second time steps for speed
            seed=int(self.rng.integers(0, 2**31))
        )
        
        peak_load = sim.peak_load_for_beam()
        
        return {
            'scenario_name': scenario.name,
            'peak_load': peak_load,
            'conditions': conditions,
            'sideslip_deg': sideslip,
            'turbulence_intensity': scenario.turbulence_intensity,
        }
    
    def sample_many_flights(self, n_flights: int, flight_hours: float = 4.0):
        """
        Sample peak loads for many flights efficiently.
        Used for quick distribution estimation.
        
        Parameters:
        -----------
        n_flights : int
        flight_hours : float
        
        Returns:
        --------
        np.ndarray : peak loads for each flight (N)
        """
        return np.array([
            self.sample_flight_load(flight_hours)['peak_load']
            for _ in range(n_flights)
        ])
if __name__ == "__main__":
    scenarios = build_flight_scenarios()

    print("\nFLIGHT SCENARIOS")
    print("=" * 60)

    for s in scenarios:
        print(f"\n{s.name}")
        print(f"  Description: {s.description}")
        print(f"  Weight: {s.weight:.2f}")
        print(f"  Altitude: {s.conditions.altitude_m:.0f} m")
        print(f"  Airspeed: {s.conditions.airspeed_ms:.1f} m/s")
        print(f"  Turbulence Intensity: {s.turbulence_intensity:.1f} m/s")
        print(f"  Gust Mean Velocity: {s.mean_gust_velocity:.1f} m/s")    