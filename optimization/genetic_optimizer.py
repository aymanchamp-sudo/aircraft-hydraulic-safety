# Genetic algorithm optimizer
# optimization/genetic_optimizer.py

"""
Genetic Algorithm for Hydraulic Line Placement Optimization
============================================================
Automatically evolves the safest hydraulic configuration.

Chromosome encoding:
  Each individual = [y1, y2, y3, ...] — y-positions of hydraulic lines
  y values are fractions of h/2: -1.0 = bottom surface, +1.0 = top surface

Fitness function:
  Uses the HydraulicConfiguration safety score (0-100)
  Plus penalties for physical violations (overlap, redundancy)

Algorithm:
  Population size: 50
  Generations:     100
  Selection:       Tournament selection (size 3)
  Crossover:       Uniform crossover (50% gene swap)
  Mutation:        Gaussian perturbation (σ = 0.1)
  Elitism:         Top 2 individuals always survive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time
import os


@dataclass
class GAConfig:
    """
    Genetic Algorithm hyperparameters.

    These control how the evolutionary search proceeds.
    Tuning these is itself an art — we use well-tested defaults.
    """
    population_size: int   = 50
    n_generations:  int    = 80
    n_lines:        int    = 3       # Number of hydraulic lines to optimize
    crossover_rate: float  = 0.8     # Probability of crossover between parents
    mutation_rate:  float  = 0.15    # Probability of mutating each gene
    mutation_sigma: float  = 0.08    # Standard deviation of Gaussian mutation
    tournament_size: int   = 3       # Tournament selection size
    elitism_count:   int   = 2       # How many top individuals always survive
    seed:            int   = 42      # For reproducibility

    # Search space bounds (fraction of h/2)
    y_min: float = -0.90   # Cannot be right at the surface (need clearance)
    y_max: float =  0.90


@dataclass
class Individual:
    """
    One individual in the genetic algorithm population.

    Represents a complete hydraulic routing configuration.
    Genes = y-positions of all hydraulic lines (normalized to [-1, 1]).
    """
    genes: np.ndarray        # Shape: (n_lines,) — normalized y-positions
    fitness: float = 0.0     # Safety score (0-100), higher = better
    safety_factors: List[float] = field(default_factory=list)
    config_score: float = 0.0
    valid: bool = True        # False if physical violations exist

    def __lt__(self, other):
        """Enable sorting by fitness."""
        return self.fitness < other.fitness

    def copy(self):
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            safety_factors=self.safety_factors.copy(),
            config_score=self.config_score,
            valid=self.valid,
        )


class FitnessEvaluator:
    """
    Evaluates the fitness (safety score) of a hydraulic configuration.

    Converts normalized gene values to actual HydraulicLine objects,
    builds a HydraulicConfiguration, runs full safety analysis,
    and returns a scalar fitness score.

    This is called once per individual per generation.
    Making it fast is critical for performance.
    """

    def __init__(self, beam, pressure=20.7e6):
        """
        Parameters:
        -----------
        beam : Beam object with loads applied
        pressure : float — hydraulic operating pressure (Pa)
        """
        self.beam     = beam
        self.pressure = pressure
        self.h        = beam.cs.h          # Full cross-section height
        self.h_half   = beam.cs.h / 2     # Half height (max y deviation)
        self.L        = beam.L

        # Import here to avoid circular imports at module level
        from core.hydraulic_line import (HydraulicLine, HydraulicLineGeometry,
                                          TITANIUM_TUBE)
        from core.hydraulic_configuration import HydraulicConfiguration

        self._HydraulicLine  = HydraulicLine
        self._HydGeo         = HydraulicLineGeometry
        self._TitaniumTube   = TITANIUM_TUBE
        self._HydConfig      = HydraulicConfiguration

        # Standard tube geometry (fixed — only positions are optimized)
        self._geometry = HydraulicLineGeometry(
            outer_diameter=0.012,
            wall_thickness=0.001
        )

    def genes_to_y_positions(self, genes: np.ndarray) -> np.ndarray:
        """
        Convert normalized genes [-1, 1] to actual y positions (meters).

        Gene = 0.0  →  y = 0.0m  (neutral axis — safest)
        Gene = 1.0  →  y = +h/2  (top surface — most dangerous)
        Gene = -1.0 →  y = -h/2  (bottom surface — most dangerous)
        """
        return genes * self.h_half

    def build_configuration(self, genes: np.ndarray,
                             config_id: str = "GA_INDIVIDUAL"):
        """
        Build a HydraulicConfiguration from gene values.

        Parameters:
        -----------
        genes : np.ndarray — normalized y positions
        config_id : str

        Returns:
        --------
        HydraulicConfiguration
        """
        y_positions = self.genes_to_y_positions(genes)
        system_names = ['System_A', 'System_B', 'System_C',
                          'System_D', 'System_E']

        lines = []
        for i, y_pos in enumerate(y_positions):
            line = self._HydraulicLine(
                line_id=f"HYD-{i+1}",
                geometry=self._geometry,
                material=self._TitaniumTube,
                x_start=0.2,
                x_end=self.L * 0.9,
                y_position=float(y_pos),
                operating_pressure=self.pressure,
                system_name=system_names[i % len(system_names)],
            )
            lines.append(line)

        return self._HydConfig(config_id, lines)

    def evaluate(self, individual: Individual) -> Individual:
        """
        Compute fitness for one individual.

        Steps:
        1. Convert genes to y-positions
        2. Build hydraulic configuration
        3. Run safety analysis
        4. Compute fitness score with penalties

        Parameters:
        -----------
        individual : Individual

        Returns:
        --------
        Individual (with fitness filled in)
        """
        try:
            config   = self.build_configuration(individual.genes)
            analysis = config.analyze(self.beam)

            # Base score from configuration analysis
            base_score = analysis['config_score']

            # Extract safety factors
            sfs = [r['sf_overall'] for r in analysis['line_results']]

            # Penalty for minimum SF below aerospace standard
            min_sf = min(sfs)
            sf_penalty = 0.0
            if min_sf < 1.5:
                sf_penalty = (1.5 - min_sf) * 30   # Heavy penalty
            elif min_sf < 2.0:
                sf_penalty = (2.0 - min_sf) * 10   # Moderate penalty

            # Penalty for physical violations
            overlap_penalty     = 0  if analysis['no_overlap']    else 40
            separation_penalty  = 0  if analysis['separation_ok'] else 20
            redundancy_penalty  = 0  if analysis['redundancy_ok'] else 15

            # Penalty for lines too close to surface (fragility penalty)
            surface_penalty = 0.0
            y_positions = self.genes_to_y_positions(individual.genes)
            for y in y_positions:
                frac = abs(y) / self.h_half  # 0=NA, 1=surface
                if frac > 0.7:
                    surface_penalty += (frac - 0.7) * 20

            # Final fitness
            fitness = (base_score
                       - sf_penalty
                       - overlap_penalty
                       - separation_penalty
                       - redundancy_penalty
                       - surface_penalty)
            fitness = max(0.0, min(100.0, fitness))

            individual.fitness        = fitness
            individual.safety_factors = sfs
            individual.config_score   = base_score
            individual.valid          = (analysis['no_overlap'] and
                                          analysis['separation_ok'])

        except Exception as e:
            # If evaluation fails, assign zero fitness
            individual.fitness      = 0.0
            individual.valid        = False
            individual.config_score = 0.0

        return individual


class GeneticOptimizer:
    """
    Main Genetic Algorithm engine for hydraulic line placement.

    Evolves a population of hydraulic configurations over many
    generations to find the highest-safety routing.

    Usage:
        optimizer = GeneticOptimizer(ga_config, fitness_evaluator)
        result    = optimizer.run()
        print(result.best_individual.genes)
    """

    def __init__(self, config: GAConfig, evaluator: FitnessEvaluator):
        self.config    = config
        self.evaluator = evaluator
        self.rng       = np.random.default_rng(config.seed)

        # History tracking
        self.generation_stats: List[Dict] = []
        self.best_individual: Optional[Individual] = None

    def _initialize_population(self) -> List[Individual]:
        """
        Create initial random population.

        Strategy: mix of random + smart initialization
        - 80% pure random within bounds
        - 20% seeded near neutral axis (known good region)

        This "warm start" speeds up convergence significantly.
        """
        population = []
        cfg = self.config
        n_smart = max(1, cfg.population_size // 5)  # 20% smart starts

        for i in range(cfg.population_size):
            if i < n_smart:
                # Smart initialization: near neutral axis
                genes = self.rng.uniform(-0.2, 0.2, size=cfg.n_lines)
            else:
                # Random initialization across full search space
                genes = self.rng.uniform(cfg.y_min, cfg.y_max,
                                          size=cfg.n_lines)

            population.append(Individual(genes=genes))

        return population

    def _tournament_selection(self,
                               population: List[Individual]) -> Individual:
        """
        Tournament selection: randomly pick k individuals, return the best.

        WHY TOURNAMENT SELECTION?
        - Maintains diversity (losers still have a chance to be selected)
        - Selection pressure controllable via tournament_size
        - More robust than pure rank selection
        - Doesn't require sorting the entire population

        Parameters:
        -----------
        population : list of evaluated individuals

        Returns:
        --------
        Individual : winner of tournament (copy)
        """
        k = self.config.tournament_size
        contestants = self.rng.choice(len(population), size=k, replace=False)
        winner_idx  = max(contestants, key=lambda i: population[i].fitness)
        return population[winner_idx].copy()

    def _uniform_crossover(self,
                            parent1: Individual,
                            parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover: each gene independently swapped with 50% probability.

        WHY UNIFORM CROSSOVER?
        - Each line position is treated independently
        - No bias toward preserving gene order (unlike single-point crossover)
        - Good for problems where gene position doesn't encode ordering

        Example:
          Parent 1: [0.00, 0.05, -0.04]
          Parent 2: [0.02, 0.03, -0.06]
          Mask:     [True, False, True ]
          Child 1:  [0.00, 0.03, -0.04]  ← P1 where mask=True, P2 otherwise
          Child 2:  [0.02, 0.05, -0.06]  ← opposite
        """
        n = self.config.n_lines
        child1 = parent1.copy()
        child2 = parent2.copy()

        if self.rng.random() < self.config.crossover_rate:
            mask = self.rng.random(n) < 0.5
            child1.genes = np.where(mask, parent1.genes, parent2.genes)
            child2.genes = np.where(mask, parent2.genes, parent1.genes)

        return child1, child2

    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """
        Gaussian mutation: add small random noise to each gene.

        WHY GAUSSIAN MUTATION?
        - Continuous search space → continuous perturbations make sense
        - Small changes explore nearby solutions (exploitation)
        - Occasionally larger changes escape local optima (exploration)
        - σ controls the trade-off

        Mutation is applied gene-by-gene with probability mutation_rate.
        Mutated values are clipped to valid bounds.
        """
        mutant = individual.copy()
        cfg    = self.config

        for i in range(cfg.n_lines):
            if self.rng.random() < cfg.mutation_rate:
                noise = self.rng.normal(0, cfg.mutation_sigma)
                mutant.genes[i] = np.clip(
                    mutant.genes[i] + noise,
                    cfg.y_min, cfg.y_max
                )

        # Reset fitness (needs re-evaluation after mutation)
        mutant.fitness = 0.0
        return mutant

    def _evaluate_population(self,
                               population: List[Individual]) -> List[Individual]:
        """
        Evaluate fitness for all individuals in population.

        In a production system this would be parallelized using
        multiprocessing.Pool.map(). For our purposes, sequential
        evaluation is sufficient.
        """
        return [self.evaluator.evaluate(ind) for ind in population]

    def _record_generation_stats(self,
                                   generation: int,
                                   population: List[Individual]):
        """Record statistics for this generation (used for convergence plots)."""
        fitnesses  = [ind.fitness for ind in population]
        sfs        = [min(ind.safety_factors) if ind.safety_factors else 0
                       for ind in population]

        stats = {
            'generation':    generation,
            'best_fitness':  max(fitnesses),
            'mean_fitness':  np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness':   np.std(fitnesses),
            'best_min_sf':   max(sfs),
            'mean_min_sf':   np.mean(sfs),
            'n_valid':       sum(1 for ind in population if ind.valid),
            'best_genes':    population[np.argmax(fitnesses)].genes.copy(),
        }
        self.generation_stats.append(stats)
        return stats

    def run(self, verbose=True):
        """
        Execute the complete genetic algorithm.

        Returns:
        --------
        GAResult object with best individual and full history
        """
        cfg = self.config
        start_time = time.time()

        if verbose:
            print(f"\n  {'─'*55}")
            print(f"  Genetic Algorithm Starting")
            print(f"  Population: {cfg.population_size}  |  "
                  f"Generations: {cfg.n_generations}  |  "
                  f"Lines: {cfg.n_lines}")
            print(f"  {'─'*55}")

        # ── Initialize ─────────────────────────────────────────────────
        population = self._initialize_population()
        population = self._evaluate_population(population)
        population.sort(key=lambda x: x.fitness, reverse=True)

        self.best_individual = population[0].copy()
        stats = self._record_generation_stats(0, population)

        if verbose:
            print(f"  Gen   0/{cfg.n_generations}: "
                  f"Best={stats['best_fitness']:.1f}  "
                  f"Mean={stats['mean_fitness']:.1f}  "
                  f"MinSF={stats['best_min_sf']:.2f}")

        # ── Main evolution loop ────────────────────────────────────────
        for gen in range(1, cfg.n_generations + 1):

            new_population = []

            # Elitism: top individuals survive unchanged
            for i in range(cfg.elitism_count):
                new_population.append(population[i].copy())

            # Fill rest of population via selection + crossover + mutation
            while len(new_population) < cfg.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                child1, child2 = self._uniform_crossover(parent1, parent2)
                child1 = self._gaussian_mutation(child1)
                child2 = self._gaussian_mutation(child2)

                new_population.append(child1)
                if len(new_population) < cfg.population_size:
                    new_population.append(child2)

            # Evaluate new population
            # (Elites already evaluated — skip them for speed)
            for i in range(cfg.elitism_count, len(new_population)):
                new_population[i] = self.evaluator.evaluate(new_population[i])

            # Sort by fitness
            new_population.sort(key=lambda x: x.fitness, reverse=True)
            population = new_population

            # Update best
            if population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0].copy()

            stats = self._record_generation_stats(gen, population)

            # Progress reporting every 10 generations
            if verbose and (gen % 10 == 0 or gen == cfg.n_generations):
                elapsed = time.time() - start_time
                print(f"  Gen {gen:3d}/{cfg.n_generations}: "
                      f"Best={stats['best_fitness']:.1f}  "
                      f"Mean={stats['mean_fitness']:.1f}  "
                      f"MinSF={stats['best_min_sf']:.2f}  "
                      f"Valid={stats['n_valid']}/{cfg.population_size}  "
                      f"({elapsed:.0f}s)")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n  Evolution complete in {elapsed:.1f}s")
            print(f"  Best fitness: {self.best_individual.fitness:.2f}")
            print(f"  Best genes:   "
                  f"{[f'{g:.3f}' for g in self.best_individual.genes]}")

        return GAResult(
            best_individual=self.best_individual,
            generation_stats=self.generation_stats,
            config=cfg,
            elapsed_seconds=elapsed,
            evaluator=self.evaluator,
        )


class GAResult:
    """
    Complete results from a genetic algorithm run.

    Contains best individual, full evolutionary history,
    and methods for analysis and visualization.
    """

    def __init__(self, best_individual, generation_stats,
                  config, elapsed_seconds, evaluator):
        self.best     = best_individual
        self.history  = generation_stats
        self.config   = config
        self.elapsed  = elapsed_seconds
        self.evaluator = evaluator
        self.history_df = pd.DataFrame(generation_stats)

    def print_summary(self):
        """Print optimized configuration summary."""
        y_positions = self.evaluator.genes_to_y_positions(self.best.genes)
        h_half = self.evaluator.h_half

        print("\n" + "═"*60)
        print("  GENETIC ALGORITHM OPTIMIZATION RESULTS")
        print("═"*60)
        print(f"  Generations:      {self.config.n_generations}")
        print(f"  Population size:  {self.config.population_size}")
        print(f"  Time elapsed:     {self.elapsed:.1f}s")
        print()
        print(f"  OPTIMAL CONFIGURATION FOUND:")
        print(f"  {'Line':<10} {'Gene':>8} {'y-pos (mm)':>12} "
              f"{'% of h/2':>10} {'Zone':>12}")
        print("  " + "─"*55)

        zones = {0.0: 'NEUTRAL AXIS', 0.3: 'INNER ZONE',
                  0.6: 'MIDDLE ZONE', 0.8: 'OUTER ZONE', 1.0: 'SURFACE'}

        for i, (gene, y) in enumerate(zip(self.best.genes, y_positions)):
            frac = abs(y) / h_half
            zone = 'SURFACE' if frac > 0.8 else \
                   'OUTER'   if frac > 0.6 else \
                   'MIDDLE'  if frac > 0.3 else \
                   'INNER'   if frac > 0.05 else 'NEUTRAL AXIS'
            print(f"  Line {i+1:<5}  {gene:>+8.4f}  "
                  f"{y*1000:>10.2f}mm  {frac*100:>8.1f}%  {zone:>12}")

        print()
        print(f"  Best fitness score:  {self.best.fitness:.2f} / 100")
        print(f"  Config score:        {self.best.config_score:.2f} / 100")
        if self.best.safety_factors:
            print(f"  Min safety factor:   {min(self.best.safety_factors):.3f}")
            print(f"  Mean safety factor:  {np.mean(self.best.safety_factors):.3f}")
        print(f"  Physical validity:   {'✅ VALID' if self.best.valid else '❌ INVALID'}")
        print("═"*60)

    def convergence_data(self):
        """Return convergence curve data."""
        return (
            self.history_df['generation'].values,
            self.history_df['best_fitness'].values,
            self.history_df['mean_fitness'].values,
        )