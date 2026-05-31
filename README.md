# AI-Assisted Aircraft Hydraulic Safety Optimization System

A complete AI-assisted engineering simulation system for optimizing
hydraulic line placement in aircraft fin structures. Combines classical
aerospace engineering physics with modern AI techniques including
Monte Carlo simulation, genetic algorithms, and machine learning.

---

## Project Overview

Aircraft hydraulic lines carry flight-critical fluid under high pressure
(3000-5000 PSI). Their placement inside the fin structure critically
affects both their structural safety (bending stress from aerodynamic
loads) and system reliability (redundancy, separation, overlap).

This project builds a complete simulation and optimization pipeline:
Flight Loads → Structural Analysis → Hydraulic Safety Evaluation
↓               ↓                        ↓
Monte Carlo    Genetic Algorithm         ML Failure Prediction
Simulation      Optimization              (Real-time Dashboard)

---

## Key Results

| Phase | Achievement |
|-------|-------------|
| Structural | 4m fin, Al 2024-T3, 50kN load → SF=1.54, tip deflection=159mm |
| Hydraulic | Standard config score=90.3/100, dangerous config=0.0/100 |
| Monte Carlo | 2000 flights simulated, P(failure)=0.001 for standard config |
| Data Science | Load-SF Pearson r=-0.9829, failures above 96.2 kN threshold |
| Genetic Algorithm | Improved config from 90.3 → 94.1/100 (+4.1 points) in 0.7s |
| Machine Learning | Random Forest AUC=1.0, XGBoost AUC=1.0 on 4000-sample dataset |
| Dashboard | Live Streamlit app with real-time physics simulation |

---

## System Architecture
core/
├── material.py          Material database (Al 2024-T3, Ti-6Al-4V...)
├── beam.py              Euler-Bernoulli cantilever beam physics
├── stress_calculator.py Bending stress, safe zone analysis
├── hydraulic_line.py    Hoop stress, Von Mises, Basquin fatigue
└── hydraulic_configuration.py  Multi-line safety scoring
simulation/
├── wind_model.py        ISA atmosphere, Gauss-Markov turbulence
├── flight_conditions.py 7 flight scenarios (cruise, maneuver, storm...)
└── monte_carlo.py       Full Monte Carlo engine, CSV export
optimization/
└── genetic_optimizer.py Population=50, Generations=80, Tournament selection
ml/
├── dataset_builder.py   Feature engineering, correlation analysis
└── train_model.py       Random Forest + XGBoost, class imbalance handling
visualization/
├── stress_plots.py      Engineering dashboard plots
├── beam_animation.py    60-frame GIF animations
├── interactive_plots.py Plotly 3D HTML visualizations
├── ga_plots.py          Convergence and diversity plots
└── ml_plots.py          ROC curves, confusion matrices, landscapes
dashboard/
└── app.py               Streamlit web application

---

## How to Run

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/aircraft-hydraulic-safety
cd aircraft-hydraulic-safety
pip install -r requirements.txt
```

### Run individual phases
```bash
python main.py 2    # Structural simulation
python main.py 3    # Hydraulic modeling
python main.py 4    # Wind & flight simulation
python main.py 5    # Monte Carlo (takes ~45 seconds)
python main.py 6    # Data science analysis
python main.py 7    # Animations + interactive plots (~90 seconds)
python main.py 8    # Genetic algorithm optimization
python main.py 9    # Machine learning training
python main.py all  # Run all phases in sequence
```

### Launch dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

---

## Physics Models

### Structural (Euler-Bernoulli Beam Theory)
σ = My/I          Bending stress
δ = FL³/3EI       Tip deflection (tip load)
δ = qL⁴/8EI      Tip deflection (distributed load)
SF = σ_yield/σ    Safety factor

### Hydraulic Line Safety
σ_hoop = Pr/t           Hoop stress from internal pressure
σ_VM = √(σ₁²-σ₁σ₂+σ₂²) Von Mises combined stress
N_f = (σ_ref/σ_VM)^b × N_ref   Basquin fatigue life

### Aerodynamic Model
L = ½ρV²·S·CL_α·α      Fin lift force
ρ = ρ₀·(1-L·h/T₀)^(g/LR)  ISA atmosphere
σ_w = σ_w0·exp(-Δt/τ) + ε  Gauss-Markov turbulence

### Genetic Algorithm
Fitness = config_score - SF_penalty - overlap_penalty
- separation_penalty - redundancy_penalty
- surface_penalty

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Physics | NumPy, SciPy |
| Data | Pandas |
| ML | scikit-learn, XGBoost |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Persistence | Joblib |

---

## Research Extensions (Phase 11)

- **FEM**: Replace beam theory with isoparametric finite elements
- **Fatigue**: Miner's rule with rainflow-counted flight spectra
- **CFD**: RANS aerodynamics for high-alpha and compressible flow
- **Aeroelasticity**: Flutter speed verification per FAR 25.629
- **Digital Twin**: Connect dashboard to live flight data recorder

---

## Author

**Ayman**  
S.Y. B.Tech — Computer Science and Business Systems  
K.J. Somaiya School of Engineering  
Somaiya Vidyavihar University, Mumbai
