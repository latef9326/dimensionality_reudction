"""
================================================================================
EVOLVING CONVOLUTIONAL FILTERS WITH DIFFERENTIAL EVOLUTION
================================================================================
Assignment: Learning image filters using population-based metaheuristics - Lab Report
Authors:    [Lateef Hanus] & [Michal Chojnacki]
Date:       June 2026
Algorithm:  Differential Evolution (DE/rand/1/bin)
Dataset:    MNIST
================================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
N_FILTERS = 5              # Number of convolutional filters to evolve
FILTER_SIZE = 7            # Spatial size of each filter (7x7)
GENOME_LENGTH = N_FILTERS * FILTER_SIZE * FILTER_SIZE  # 245 real-valued genes

POP_SIZE = 30              # Population size (μ)
MAX_GENERATIONS = 150      # Number of generations
MUT_SCALE = 0.8            # Differential weight (F) – renamed to avoid conflict with torch.nn.functional
CR = 0.9                   # DE crossover probability
BOUNDS = (-1.0, 1.0)       # Gene value bounds
ELITE = True               # Preserve best individual each generation

BATCH_SIZE = 500           # MNIST images for fitness evaluation
SNAPSHOT_GENS = [0, 10, 30, 60, 100, 149]  # Generations to visualize

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Output paths
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUTPUT_DIR, 'DE_Filter_Evolution_Report.pdf')

print("=" * 70)
print("DIFFERENTIAL EVOLUTION: EVOLVING MNIST CONVOLUTIONAL FILTERS")
print("=" * 70)


# ==============================================================================
# 1. LOAD MNIST DATASET
# ==============================================================================
def load_mnist(batch_size=BATCH_SIZE):
    """
    Load a fixed subset of MNIST training images for fitness evaluation.
    Returns tensor of shape (batch_size, 1, 28, 28).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Select first `batch_size` images deterministically
    images = torch.stack([mnist[i][0] for i in range(batch_size)])
    print(f"[DATA] Loaded {batch_size} MNIST images: {images.shape}")
    return images

images = load_mnist()


# ==============================================================================
# 2. FITNESS FUNCTION
# ==============================================================================
def genome_to_filters(genome):
    """
    Convert flat real-valued genome into 5 convolutional filters.
    Output shape: (5, 1, 7, 7) for PyTorch conv2d.
    """
    filters = genome.reshape(N_FILTERS, 1, FILTER_SIZE, FILTER_SIZE)
    return torch.tensor(filters, dtype=torch.float32)


def evaluate_fitness(genome):
    """
    Fitness Function:
    1. Reshape genome into 5 filters
    2. Convolve across MNIST batch (valid convolution, no padding)
    3. Take absolute value of responses
    4. Return MEAN activation over (samples × filters × spatial positions)

    Higher fitness = filters that produce stronger average responses.
    """
    filters = genome_to_filters(genome)

    with torch.no_grad():
        # Convolve: (B, 1, 28, 28) * (5, 1, 7, 7) -> (B, 5, 22, 22)
        responses = F.conv2d(images, filters, padding=0)

        # Absolute response (feature map activations)
        abs_responses = torch.abs(responses)

        # Average over all dimensions: samples, filters, height, width
        fitness = abs_responses.mean().item()

    return fitness


# ==============================================================================
# 3. DIFFERENTIAL EVOLUTION (DE/rand/1/bin)
# ==============================================================================
class DifferentialEvolution:
    """
    Differential Evolution is a population-based metaheuristic that uses
    vector differences for mutation and binomial crossover for recombination.

    Variant: DE/rand/1/bin
    - rand: base vector chosen randomly (not the target)
    - 1: one difference vector used
    - bin: binomial crossover
    """

    def __init__(self, pop_size, genome_length, bounds=BOUNDS):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.bounds = bounds

        # Initialize population uniformly in [bounds[0], bounds[1]]
        self.population = np.random.uniform(
            bounds[0], bounds[1], (pop_size, genome_length)
        )
        self.fitness = np.zeros(pop_size)
        self.best_idx = 0

        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.filter_snapshots = {}  # gen -> best genome

    def initialize(self):
        """Evaluate initial population fitness."""
        print("[INIT] Evaluating initial population...")
        for i in range(self.pop_size):
            self.fitness[i] = evaluate_fitness(self.population[i])

        self.best_idx = np.argmax(self.fitness)
        print(f"[INIT] Best initial fitness: {self.fitness[self.best_idx]:.6f}")

    def mutate(self, target_idx):
        """
        DE/rand/1 Mutation:
        v_i = x_r1 + F * (x_r2 - x_r3)
        where r1, r2, r3 are distinct random indices ≠ target_idx
        """
        candidates = [i for i in range(self.pop_size) if i != target_idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

        mutant = self.population[r1] + MUT_SCALE * (self.population[r2] - self.population[r3])

        # Clamp to bounds
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        """
        Binomial Crossover:
        For each gene j:
            u_j = v_j  if rand() < CR or j == j_rand
            u_j = x_j  otherwise
        j_rand ensures at least one gene comes from mutant.
        """
        trial = np.copy(target)
        j_rand = np.random.randint(self.genome_length)

        for j in range(self.genome_length):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def evolve(self, max_generations):
        """
        Main evolution loop. For each generation:
        1. For each individual, create mutant via DE mutation
        2. Recombine via binomial crossover -> trial vector
        3. Evaluate trial fitness
        4. Greedy selection: keep better of target vs trial
        5. Elitism: preserve best individual
        """
        self.initialize()

        for gen in range(max_generations):
            for i in range(self.pop_size):
                # Mutation
                mutant = self.mutate(i)

                # Crossover
                trial = self.crossover(self.population[i], mutant)

                # Evaluation
                trial_fitness = evaluate_fitness(trial)

                # Selection (greedy - maximize fitness)
                if trial_fitness > self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness > self.fitness[self.best_idx]:
                        self.best_idx = i

            # Elitism: copy best to position 0
            if ELITE:
                self.population[0] = np.copy(self.population[self.best_idx])
                self.fitness[0] = self.fitness[self.best_idx]
                self.best_idx = 0

            # Record statistics
            best_fit = self.fitness[self.best_idx]
            mean_fit = self.fitness.mean()
            self.best_fitness_history.append(best_fit)
            self.mean_fitness_history.append(mean_fit)

            # Save filter snapshots for visualization
            if gen in SNAPSHOT_GENS:
                self.filter_snapshots[gen] = np.copy(self.population[self.best_idx])

            if gen % 10 == 0:
                print(f"[GEN {gen:3d}] Best: {best_fit:.6f} | Mean: {mean_fit:.6f}")

        # Ensure final snapshot
        if (max_generations - 1) not in self.filter_snapshots:
            self.filter_snapshots[max_generations - 1] = np.copy(self.population[self.best_idx])

        print(f"\n[DONE] Final best fitness: {self.fitness[self.best_idx]:.6f}")
        return self.population[self.best_idx]


# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
def plot_filter_grid(filters, title, ax=None):
    """
    Plot a grid of 5 filters as grayscale images.
    filters: numpy array of shape (5, 1, 7, 7) or (245,)
    """
    if filters.ndim == 1:
        filters = filters.reshape(N_FILTERS, 1, FILTER_SIZE, FILTER_SIZE)

    if ax is None:
        fig, ax = plt.subplots(1, N_FILTERS, figsize=(12, 2.5))

    for i in range(N_FILTERS):
        filt = filters[i, 0]  # (7, 7)
        im = ax[i].imshow(filt, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax[i].set_title(f'Filter {i+1}', fontsize=9)
        ax[i].axis('off')

    return ax


def create_evolution_grid(de_algo):
    """
    Create a figure showing how filters evolve across generations.
    Rows = generations, Columns = 5 filters.
    """
    sorted_gens = sorted(de_algo.filter_snapshots.keys())
    n_rows = len(sorted_gens)

    fig, axes = plt.subplots(n_rows, N_FILTERS, figsize=(14, 2.8 * n_rows))

    for row, gen in enumerate(sorted_gens):
        genome = de_algo.filter_snapshots[gen]
        filters = genome.reshape(N_FILTERS, 1, FILTER_SIZE, FILTER_SIZE)

        for col in range(N_FILTERS):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            filt = filters[col, 0]
            im = ax.imshow(filt, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(f'Gen {gen}', fontsize=11, rotation=0,
                             labelpad=40, va='center')
            if row == 0:
                ax.set_title(f'Filter {col+1}', fontsize=10)

    fig.suptitle('Evolution of 5 Convolutional Filters (7×7) Over Generations\n'
                 'Differential Evolution on MNIST', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_fitness_progression(de_algo):
    """
    Line plot showing best and mean fitness over generations.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    gens = range(len(de_algo.best_fitness_history))
    ax.plot(gens, de_algo.best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(gens, de_algo.mean_fitness_history, 'r--', linewidth=1.5, alpha=0.7, label='Mean Fitness')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Average Absolute Response (Fitness)', fontsize=12)
    ax.set_title('Fitness Progression: Differential Evolution on MNIST Filters', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate final value
    final_best = de_algo.best_fitness_history[-1]
    ax.annotate(f'Final: {final_best:.4f}',
                xy=(len(gens)-1, final_best),
                xytext=(len(gens)-30, final_best + 0.01),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')

    plt.tight_layout()
    return fig


def plot_final_activations(best_genome):
    """
    Show the best filters and their responses on a sample MNIST image.
    """
    filters = genome_to_filters(best_genome)
    sample_img = images[0:1]  # (1, 1, 28, 28)

    with torch.no_grad():
        activations = F.conv2d(sample_img, filters)  # (1, 5, 22, 22)

    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 6, figure=fig, wspace=0.3)

    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(sample_img[0, 0], cmap='gray')
    ax0.set_title('Input (MNIST Sample)', fontsize=10)
    ax0.axis('off')

    # Activations
    for i in range(5):
        ax = fig.add_subplot(gs[0, i+1])
        act = torch.abs(activations[0, i]).numpy()
        ax.imshow(act, cmap='hot', interpolation='nearest')
        ax.set_title(f'|Filter {i+1} * Image|', fontsize=9)
        ax.axis('off')

    fig.suptitle('Learned Filters and Their Absolute Responses', fontsize=13, fontweight='bold')
    return fig


# ==============================================================================
# 5. GENERATE PDF REPORT
# ==============================================================================
def generate_pdf_report(de_algo, best_genome):
    """
    Compile all figures and text into a single PDF report.
    """
    with PdfPages(PDF_PATH) as pdf:

        # ===== PAGE 1: Title & Metadata =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.85, 'EVOLVING CONVOLUTIONAL FILTERS',
                ha='center', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.80, 'with Differential Evolution',
                ha='center', fontsize=16, style='italic')
        fig.text(0.5, 0.72, 'Artificial Inteligence — ex Report',
                ha='center', fontsize=12)
        fig.text(0.5, 0.68, f'Authors: [Lateef Hanus] & [Michal chojnacki]',
                ha='center', fontsize=11)
        fig.text(0.5, 0.64, f'Date: {datetime.now().strftime("%B %d, %Y")}',
                ha='center', fontsize=11)

        # Code repository link placeholder
        fig.text(0.5, 0.55, 'Code Repository:', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 0.51, 'https://github.com/latef9326/dimensionality_reudction',
                ha='center', fontsize=10, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))


        # Algorithm description
        desc = """
ALGORITHM: Differential Evolution (DE/rand/1/bin)

Differential Evolution is a population-based metaheuristic introduced by 
Storn & Price (1997). Unlike traditional Genetic Algorithms, DE does NOT use:
  • Crossover operators like one-point or uniform crossover
  • Bit-string representations or binary encoding
  • Fitness-proportionate selection

Instead, DE operates on real-valued vectors and uses:

1. MUTATION (DE/rand/1):
   v_i = x_r1 + F · (x_r2 − x_r3)
   Three distinct random individuals (r1, r2, r3) are selected. 
   The difference vector (x_r2 − x_r3) is scaled by F and added to x_r1.

2. CROSSOVER (Binomial):
   Each gene in the trial vector is inherited from either the mutant 
   or the target, with probability CR. One gene is forced from the 
   mutant to ensure diversity.

3. SELECTION (Greedy):
   The trial vector replaces the target only if it has BETTER fitness.
   This is deterministic, not probabilistic.

HYPERPARAMETERS:
  • Population size (μ):        30
  • Differential weight (F):    0.8
  • Crossover rate (CR):        0.9
  • Gene bounds:                [-1.0, 1.0]
  • Elitism:                    Enabled (best preserved)
  • Generations:                150
  • Genome size:                245 (5 filters × 7×7 weights)
        """
        fig.text(0.1, 0.42, desc, ha='left', fontsize=9, family='monospace',
                verticalalignment='top', linespacing=1.3)

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ===== PAGE 2: Filter Evolution Grid =====
        fig = create_evolution_grid(de_algo)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ===== PAGE 3: Fitness Progression =====
        fig = plot_fitness_progression(de_algo)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ===== PAGE 4: Final Activations =====
        fig = plot_final_activations(best_genome)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ===== PAGE 5: Author Contributions =====
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.92, 'AUTHOR CONTRIBUTIONS', ha='center',
                fontsize=16, fontweight='bold')

        contrib = """
This assignment was completed collaboratively by a team of two students.

┌─────────────────────────────────────────────────────────────────────┐
│  AUTHOR 1: [Lateef Hanus]                                           │
│  ─────────────────────────                                          │
│  • Designed the Differential Evolution algorithm structure          │
│  • Implemented mutation (DE/rand/1) and binomial crossover          │
│  • Set up the fitness function using PyTorch conv2d operations      │
│  • Tuned hyperparameters (F, CR, population size)                   │
│  • Ran experiments and collected fitness trajectories               │
│                                                                     │
│  AUTHOR 2: [Michal Chojnacki]                                       │
│  ─────────────────────────                                          │
│  • Implemented MNIST data loading and preprocessing pipeline        │
│  • Designed and coded all visualization functions                   │
│  • Created the filter evolution grid and fitness progression plots  │
│  • Generated the PDF report with matplotlib.backends.backend_pdf    │
│  • Wrote the algorithm description and contribution documentation   │
└─────────────────────────────────────────────────────────────────────┘

Both authors contributed equally to debugging, result analysis, and 
report writing. All code was reviewed by both team members.

NOTE ON TEAMWORK POLICY:
This submission is from a team of two as required. If working alone 
due to exceptional circumstances, pre-approval from the instructor 
must be obtained per the syllabus.
        """
        fig.text(0.1, 0.85, contrib, ha='left', fontsize=10,
                verticalalignment='top', linespacing=1.4,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"\n[REPORT] PDF saved to: {PDF_PATH}")


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("RUNNING DIFFERENTIAL EVOLUTION")
    print("=" * 70)

    # Initialize and run DE
    de = DifferentialEvolution(POP_SIZE, GENOME_LENGTH)
    best_genome = de.evolve(MAX_GENERATIONS)

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING PDF REPORT")
    print("=" * 70)
    generate_pdf_report(de, best_genome)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print(f"Best fitness achieved: {de.best_fitness_history[-1]:.6f}")
    print(f"Report saved to: {PDF_PATH}")
    print("=" * 70)