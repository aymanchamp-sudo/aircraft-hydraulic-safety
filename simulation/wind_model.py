# Wind and aerodynamic force simulation
# simulation/wind_model.py

"""
Wind and Aerodynamic Force Simulation
=======================================
Generates realistic randomized aerodynamic loads on the aircraft fin.

Models:
1. Base aerodynamic load (function of speed, altitude, angle)
2. Gaussian turbulence (continuous random fluctuations)
3. Discrete gust events (sudden high-intensity loads)
4. Combined time-series of fin lateral force

Physics references:
- FAR 25.341: Gust and turbulence loads
- MIL-A-8861: Military aircraft load requirements
- Hoblit (1988): Gust Loads on Aircraft (definitive textbook)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import warnings


@dataclass
class AtmosphericConditions:
    """
    Defines the atmospheric state at a given flight condition.
    
    Atmosphere model: ISA (International Standard Atmosphere)
    This is the model all aircraft are designed against.
    
    ISA sea level: T=288.15K, P=101325Pa, ρ=1.225 kg/m³
    Every 1000m altitude: T drops ~6.5°C, ρ drops ~8%
    """
    altitude_m: float        # meters above sea level
    airspeed_ms: float       # True Airspeed (TAS) in m/s
    sideslip_angle_deg: float = 0.0  # Aircraft yaw angle (degrees)
    
    # ISA constants
    _T0: float = field(default=288.15, init=False, repr=False)   # K
    _P0: float = field(default=101325.0, init=False, repr=False)  # Pa
    _rho0: float = field(default=1.225, init=False, repr=False)   # kg/m³
    _lapse_rate: float = field(default=0.0065, init=False, repr=False)  # K/m
    
    @property
    def temperature(self):
        """
        ISA temperature at altitude (K).
        T = T0 - L×h  (valid up to ~11km/tropopause)
        """
        if self.altitude_m <= 11000:
            return self._T0 - self._lapse_rate * self.altitude_m
        else:
            return 216.65  # Constant in stratosphere
    
    @property
    def pressure(self):
        """ISA pressure at altitude (Pa)."""
        if self.altitude_m <= 11000:
            return self._P0 * (self.temperature / self._T0) ** 5.2561
        else:
            # Stratosphere: exponential decay
            return 22632 * np.exp(-0.0001577 * (self.altitude_m - 11000))
    
    @property
    def density(self):
        """
        Air density at altitude (kg/m³).
        ρ = P / (R × T)  where R = 287 J/(kg·K) for air
        
        This matters hugely:
        - Sea level: ρ = 1.225 kg/m³
        - Cruise (10km): ρ ≈ 0.414 kg/m³ (34% of sea level)
        → Aerodynamic forces at cruise are only 34% of sea-level values
          BUT aircraft flies much faster, so q = ½ρV² stays significant
        """
        R_air = 287.058  # J/(kg·K)
        return self.pressure / (R_air * self.temperature)
    
    @property
    def dynamic_pressure(self):
        """
        Dynamic pressure q = ½ρV² (Pa).
        
        This is the "intensity" of the aerodynamic environment.
        All aerodynamic forces scale with q.
        """
        return 0.5 * self.density * self.airspeed_ms**2
    
    @property
    def mach_number(self):
        """
        Mach number M = V / a  where a = speed of sound.
        a = √(γRT), γ = 1.4 for air
        """
        gamma = 1.4
        R_air = 287.058
        speed_of_sound = np.sqrt(gamma * R_air * self.temperature)
        return self.airspeed_ms / speed_of_sound
    
    def __repr__(self):
        return (
            f"AtmosphericConditions:\n"
            f"  Altitude:         {self.altitude_m/1000:.1f} km\n"
            f"  Airspeed:         {self.airspeed_ms:.0f} m/s "
            f"(Mach {self.mach_number:.2f})\n"
            f"  Temperature:      {self.temperature-273.15:.1f} °C\n"
            f"  Air density:      {self.density:.4f} kg/m³\n"
            f"  Dynamic pressure: {self.dynamic_pressure/1000:.2f} kPa"
        )


# ─── Standard Flight Conditions ───────────────────────────────────────────────

# Commercial cruise (A320-class)
CRUISE_CONDITIONS = AtmosphericConditions(
    altitude_m=10668,      # 35,000 ft
    airspeed_ms=248.1,     # 485 knots TAS ≈ Mach 0.82
    sideslip_angle_deg=0.0
)

# Approach and landing (high drag configuration)
APPROACH_CONDITIONS = AtmosphericConditions(
    altitude_m=300,
    airspeed_ms=72.0,      # ~140 knots
    sideslip_angle_deg=0.0
)

# Low-altitude turbulence penetration
TURBULENCE_CONDITIONS = AtmosphericConditions(
    altitude_m=3000,
    airspeed_ms=154.3,     # 300 knots — turbulence penetration speed
    sideslip_angle_deg=2.0
)

# Maximum structural speed (design dive speed, VD)
DIVE_CONDITIONS = AtmosphericConditions(
    altitude_m=5000,
    airspeed_ms=205.0,     # ~400 knots
    sideslip_angle_deg=0.0
)


class AerodynamicModel:
    """
    Computes lateral aerodynamic force on the aircraft vertical fin.
    
    The fin generates a lateral force when there is any angle between
    the aircraft's nose and the local airflow direction (sideslip).
    
    In normal flight: sideslip ≈ 0 → fin force ≈ 0
    In crosswind landing: sideslip > 0 → fin force significant
    In yaw maneuver: sideslip transient → large fin force
    In turbulence: random sideslip → random fin force
    """
    
    def __init__(
        self,
        fin_area: float,
        fin_span: float,
        lift_curve_slope: float = 3.5,
        fin_efficiency: float = 0.95
    ):
        """
        Parameters:
        -----------
        fin_area : float
            Fin planform area S (m²)
            Typical: Boeing 737 ≈ 16 m², A320 ≈ 12.5 m²
        fin_span : float
            Fin height b (m) — same as beam length L
        lift_curve_slope : float
            dC_L/dβ — how strongly the fin generates force with sideslip angle
            Typical: 3.5 to 4.5 per radian
        fin_efficiency : float
            η — accounts for tip losses, fuselage interference
            Typical: 0.90 to 0.97
        """
        self.S = fin_area
        self.b = fin_span
        self.CL_alpha = lift_curve_slope   # per radian
        self.eta = fin_efficiency
    
    def base_lateral_force(self, conditions: AtmosphericConditions):
        """
        Compute baseline lateral aerodynamic force on fin.
        
        F = q × S × C_L
        C_L = C_L_α × β × η
        
        Where β = sideslip angle in radians.
        
        Parameters:
        -----------
        conditions : AtmosphericConditions
            
        Returns:
        --------
        float : lateral force (N), positive = starboard (right)
        """
        q = conditions.dynamic_pressure
        beta_rad = np.radians(conditions.sideslip_angle_deg)
        CL = self.CL_alpha * beta_rad * self.eta
        
        return q * self.S * CL
    
    def distributed_load(self, conditions: AtmosphericConditions):
        """
        Compute equivalent distributed load (N/m) along fin span.
        
        Assumes triangular load distribution — larger at root
        (due to chord variation and downwash effects).
        
        Converts total force F to distributed load w such that:
        Total force = integral of w(x) dx = w_root × L / 2
        → w_root = 2F / L
        
        Returns:
        --------
        float : equivalent uniform load per unit span (N/m)
                (we use equivalent uniform load for simplicity)
        """
        F_total = self.base_lateral_force(conditions)
        # Equivalent uniform distributed load
        w_equivalent = F_total / self.b
        return w_equivalent
    
    def gust_force(
        self,
        conditions: AtmosphericConditions,
        gust_velocity_ms: float
    ):
        """
        Compute additional lateral force from a discrete wind gust.
        
        FAR 25.341 Discrete Gust Model:
        The gust is modeled as a "1-cosine" profile in time.
        The equivalent load factor increment is:
        
        ΔF_gust = (ρ × V × U_de × C_L_α × S) / 2 × K_g
        
        Where:
          U_de = derived gust velocity (m/s)
          K_g  = gust alleviation factor (≈ 0.85 for most aircraft)
        
        Simplified for our model:
          ΔF_gust ≈ q × S × (U_de / V) × C_L_α
        
        Parameters:
        -----------
        gust_velocity_ms : float
            Lateral gust velocity component (m/s)
            Positive = from port side (pushes fin to starboard)
            
        Returns:
        --------
        float : incremental gust force (N)
        """
        q = conditions.dynamic_pressure
        rho = conditions.density
        V = conditions.airspeed_ms
        K_g = 0.85  # Gust alleviation factor (aircraft flexibility effect)
        
        # Gust load factor from FAR 25.341
        delta_F = (rho * V * gust_velocity_ms * self.CL_alpha * 
                   self.S * self.eta * K_g) / 2
        
        return delta_F


class TurbulenceModel:
    """
    Generates continuous random turbulence load time series.
    
    Turbulence model: Von Karman (used in FAR 25.341b continuous turbulence)
    
    Simplified implementation using power spectral density (PSD) approach:
    - White noise filtered by a shaping filter
    - Colored noise with realistic frequency content
    
    Von Karman PSD for lateral turbulence:
    Φ_v(ω) = σ²L/π × (1 + 8/3(1.339 L ω/V)²) / (1 + (1.339 L ω/V)²)^(11/6)
    
    We approximate this with a simpler Gaussian model that captures
    the key statistical properties (correct σ, roughly correct spectrum).
    """
    
    def __init__(
        self,
        turbulence_intensity: float = 3.0,
        scale_length: float = 762.0,
        correlation_length: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        turbulence_intensity : float
            σ_w — RMS turbulence velocity (m/s)
            Light:    σ < 1.5 m/s
            Moderate: 1.5 < σ < 6 m/s
            Severe:   σ > 6 m/s  (FAA: avoid if possible)
        scale_length : float
            L_w — turbulence scale length (meters)
            FAR 25: 762m (2500 ft) for altitudes above 762m
        correlation_length : float
            Controls temporal correlation (smoothness) of turbulence
            Larger → smoother (slower-varying) turbulence
        seed : int or None
            Random seed for reproducibility
            None = new random sequence each call
        """
        self.sigma = turbulence_intensity
        self.L_w = scale_length
        self.correlation = correlation_length
        self.rng = np.random.default_rng(seed)
    
    def generate_turbulence_velocities(
        self,
        n_samples: int,
        dt: float = 0.1
    ):
        """
        Generate a time series of random lateral gust velocities.
        
        Method: First-order Gauss-Markov process
        
        v(t+dt) = φ × v(t) + σ_w × √(1-φ²) × N(0,1)
        
        Where φ = exp(-V×dt/L_w) — correlation coefficient
        This produces temporally-correlated turbulence that:
        - Has correct RMS value (= σ_w)
        - Has realistic frequency content (not white noise)
        - Is stationary (statistics don't change with time)
        
        Parameters:
        -----------
        n_samples : int
            Number of time steps to generate
        dt : float
            Time step (seconds)
            
        Returns:
        --------
        np.ndarray : turbulence velocity time series (m/s)
        """
        phi = np.exp(-self.correlation)
        
        velocities = np.zeros(n_samples)
        velocities[0] = self.rng.normal(0, self.sigma)
        
        innovation_std = self.sigma * np.sqrt(1 - phi**2)
        
        for i in range(1, n_samples):
            innovation = self.rng.normal(0, innovation_std)
            velocities[i] = phi * velocities[i-1] + innovation
        
        return velocities
    
    def generate_turbulence_forces(
        self,
        aero_model: AerodynamicModel,
        conditions: AtmosphericConditions,
        n_samples: int,
        dt: float = 0.1
    ):
        """
        Convert turbulence velocities to aerodynamic forces on the fin.
        
        Each velocity sample is passed through the gust_force model
        to get a force time series.
        
        Parameters:
        -----------
        aero_model : AerodynamicModel
        conditions : AtmosphericConditions
        n_samples : int
        dt : float — time step (s)
        
        Returns:
        --------
        tuple: (time_array, force_array)
            time_array  : time stamps (s)
            force_array : lateral force time series (N)
        """
        velocities = self.generate_turbulence_velocities(n_samples, dt)
        
        forces = np.array([
            aero_model.gust_force(conditions, v) for v in velocities
        ])
        
        time = np.arange(n_samples) * dt
        
        return time, forces


class GustEventModel:
    """
    Models discrete gust events — sudden high-intensity lateral loads.
    
    Discrete gusts differ from continuous turbulence:
    - Higher amplitude (can be 2-3× turbulence RMS)
    - Shorter duration (seconds, not minutes)
    - Rarer occurrence (1 per flight, not continuous)
    
    FAR 25.341a: Aircraft must withstand discrete gusts of:
    - U_de = 17.1 m/s at cruise (56 ft/s)
    - U_de = 20.1 m/s at sea level (66 ft/s)
    - 1-cosine shape: U(s) = U_de/2 × (1 - cos(πs/H))
    
    We model random gust events with Gaussian intensity distribution.
    """
    
    def __init__(
        self,
        mean_gust_velocity: float = 8.0,
        std_gust_velocity: float = 4.0,
        mean_gust_duration: float = 3.0,
        mean_gusts_per_hour: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        mean_gust_velocity : float
            Mean lateral gust peak velocity (m/s)
        std_gust_velocity : float
            Standard deviation of gust peak velocity (m/s)
            Higher std → more extreme gusts occasionally
        mean_gust_duration : float
            Average gust duration (seconds)
        mean_gusts_per_hour : float
            Expected number of gust events per hour
        seed : int or None
        """
        self.mu_v = mean_gust_velocity
        self.sigma_v = std_gust_velocity
        self.mu_duration = mean_gust_duration
        self.mean_rate = mean_gusts_per_hour
        self.rng = np.random.default_rng(seed)
    
    def sample_gust_event(self):
        """
        Sample a single random gust event.
        
        Returns:
        --------
        dict with:
          'peak_velocity': gust peak speed (m/s)
          'duration':      gust duration (s)
          'direction':     +1 or -1 (port or starboard)
        """
        # Truncated Gaussian — gusts must be positive velocity
        while True:
            v = self.rng.normal(self.mu_v, self.sigma_v)
            if v > 0:
                break
        
        duration = max(0.5, self.rng.exponential(self.mu_duration))
        direction = self.rng.choice([-1, 1])
        
        return {
            'peak_velocity': v,
            'duration': duration,
            'direction': direction,
        }
    
    def sample_n_gusts(self, n_gusts: int):
        """
        Sample multiple independent gust events.
        
        Parameters:
        -----------
        n_gusts : int
        
        Returns:
        --------
        list of gust event dicts
        """
        return [self.sample_gust_event() for _ in range(n_gusts)]
    
    def expected_gusts_per_flight(self, flight_hours: float):
        """
        Expected number of gust events in a flight using Poisson process.
        
        Gust arrivals are a Poisson process:
        N ~ Poisson(λ × T)
        Where λ = mean_gusts_per_hour, T = flight_hours
        
        Returns:
        --------
        int : sampled number of gusts for this flight
        """
        expected = self.mean_rate * flight_hours
        return self.rng.poisson(expected)


class FlightSimulation:
    """
    Complete flight simulation combining all load sources.
    
    Simulates a single flight from takeoff to landing, generating:
    - Time-varying aerodynamic base load
    - Continuous turbulence overlay
    - Discrete gust events
    - Combined total load time series
    
    This is what gets called thousands of times in Monte Carlo.
    """
    
    def __init__(
        self,
        aero_model: AerodynamicModel,
        turbulence_model: TurbulenceModel,
        gust_model: GustEventModel,
        conditions: AtmosphericConditions,
        flight_duration_hours: float = 4.0,
        dt: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        aero_model : AerodynamicModel
        turbulence_model : TurbulenceModel
        gust_model : GustEventModel
        conditions : AtmosphericConditions
        flight_duration_hours : float
            How long this simulated flight lasts
        dt : float
            Time step (seconds). 1.0s is adequate for structural analysis.
        seed : int or None
            Random seed for reproducibility
        """
        self.aero = aero_model
        self.turb = turbulence_model
        self.gusts = gust_model
        self.conditions = conditions
        self.T = flight_duration_hours * 3600  # Convert to seconds
        self.dt = dt
        self.n_steps = int(self.T / dt)
        self.rng = np.random.default_rng(seed)
    
    def run(self):
        """
        Execute one complete flight simulation.
        
        Returns:
        --------
        dict with complete simulation output:
          'time'          : time array (s)
          'base_load'     : steady aerodynamic force (N)
          'turbulence'    : turbulence force time series (N)
          'gust_load'     : discrete gust force time series (N)
          'total_load'    : combined total load (N)
          'peak_load'     : maximum load during flight (N)
          'rms_load'      : RMS (root mean square) load (N)
          'n_gusts'       : number of discrete gust events
          'conditions'    : atmospheric conditions used
        """
        time = np.arange(self.n_steps) * self.dt
        
        # ── Base aerodynamic load ──────────────────────────────────────
        base_load = np.full(self.n_steps, 
                            self.aero.base_lateral_force(self.conditions))
        
        # ── Continuous turbulence ──────────────────────────────────────
        _, turb_forces = self.turb.generate_turbulence_forces(
            self.aero, self.conditions, self.n_steps, self.dt
        )
        
        # ── Discrete gust events ───────────────────────────────────────
        n_gusts = self.gusts.expected_gusts_per_flight(
            self.T / 3600  # convert back to hours
        )
        gust_load = np.zeros(self.n_steps)
        
        if n_gusts > 0:
            gust_times = self.rng.uniform(0, self.T, n_gusts)
            
            for gust_time in gust_times:
                gust_event = self.gusts.sample_gust_event()
                gust_force = (self.aero.gust_force(
                    self.conditions, gust_event['peak_velocity']
                ) * gust_event['direction'])
                
                # 1-cosine gust shape
                gust_duration_steps = max(
                    1, int(gust_event['duration'] / self.dt)
                )
                gust_start_idx = int(gust_time / self.dt)
                gust_end_idx = min(
                    self.n_steps, gust_start_idx + gust_duration_steps
                )
                
                if gust_start_idx < self.n_steps:
                    n = gust_end_idx - gust_start_idx
                    t_gust = np.linspace(0, np.pi, n)
                    # 1-cosine shape: starts at 0, peaks at middle, returns to 0
                    gust_shape = gust_force * (1 - np.cos(t_gust)) / 2
                    gust_load[gust_start_idx:gust_end_idx] += gust_shape
        
        # ── Combine all load sources ───────────────────────────────────
        total_load = base_load + turb_forces + gust_load
        
        return {
            'time': time,
            'base_load': base_load,
            'turbulence': turb_forces,
            'gust_load': gust_load,
            'total_load': total_load,
            'peak_load': np.max(np.abs(total_load)),
            'rms_load': np.sqrt(np.mean(total_load**2)),
            'mean_load': np.mean(total_load),
            'std_load': np.std(total_load),
            'n_gusts': n_gusts,
            'conditions': self.conditions,
            'flight_hours': self.T / 3600,
        }
    
    def peak_load_for_beam(self):
        """
        Run simulation and return just the peak load.
        Used by Monte Carlo for fast repeated sampling.
        
        Returns:
        --------
        float : peak lateral force (N) experienced during this flight
        """
        result = self.run()
        return result['peak_load']