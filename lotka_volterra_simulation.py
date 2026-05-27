"""
Stochastic Lattice Lotka-Volterra Model — Monte Carlo Simulation
================================================================
Case study: Effect of predation rate (lambda) on predator-prey dynamics.

Course: Modelling of Biological Systems (Lab 12)
Parameters:
    mu     = 0.025  (predator death rate)
    sigma  = 1.0    (prey reproduction rate)
    D      = 0.2    (diffusion rate)
    lambda = variable (predation rate)

Lattice: 124 x 124, periodic boundary conditions.
One MCS = N_sites random selections.

Cell states: 0 = empty, 1 = predator (A), 2 = prey (B)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

# ========================== PARAMETERS ==========================
LATTICE_SIZE = 124          # Grid dimension (124 x 124)
MU = 0.025                  # Predator spontaneous death rate
SIGMA = 1.0                 # Prey reproduction rate
D = 0.2                     # Diffusion (hopping) rate
MC_STEPS = 1000             # Total Monte Carlo steps
N_RUNS = 3                  # Number of independent runs per parameter

# Cell state constants
EMPTY = 0
PREDATOR = 1
PREY = 2

# ========================== SIMULATION ==========================

def get_neighbors_flat(idx, n):
    """
    Return flat indices of 4 nearest neighbors with periodic boundary conditions.
    idx = i*n + j
    """
    i = idx // n
    j = idx % n
    return np.array([
        ((i - 1) % n) * n + j,   # up
        ((i + 1) % n) * n + j,   # down
        i * n + ((j - 1) % n),   # left
        i * n + ((j + 1) % n)    # right
    ], dtype=np.int32)


def run_simulation(lattice_size, mu, sigma, lam, D, mc_steps, seed=None):
    """
    Run one Monte Carlo simulation of the stochastic lattice Lotka-Volterra model.

    Returns:
        pop_history : ndarray shape (mc_steps, 3)  -> [step, n_predators, n_prey]
        snapshots   : dict {step: grid_2d}         -> lattice snapshots at selected steps
    """
    if seed is not None:
        np.random.seed(seed)

    n = lattice_size
    n_sites = n * n

    # Flat grid: random initial conditions (equal probability for 0, 1, 2)
    grid = np.random.randint(0, 3, size=n_sites).astype(np.int8)

    # Precompute neighbor indices for all sites (periodic BC)
    neighbors = np.zeros((n_sites, 4), dtype=np.int32)
    for idx in range(n_sites):
        neighbors[idx] = get_neighbors_flat(idx, n)

    # Population history
    pop_history = np.zeros((mc_steps, 3))

    # Snapshots at selected steps
    snapshots = {}
    snapshot_steps = {0, 100, 250, 500, 750, 1000}

    for step in range(1, mc_steps + 1):
        # --- One MCS: N_sites random selections ---
        # Batch-generate random numbers for efficiency
        sites = np.random.randint(0, n_sites, size=n_sites)
        rand_vals = np.random.random(size=n_sites)
        dir_vals = np.random.randint(0, 4, size=n_sites)
        prob_vals = np.random.random(size=n_sites)

        for k in range(n_sites):
            idx = sites[k]
            occupant = grid[idx]
            if occupant == EMPTY:
                continue

            r = rand_vals[k]
            nbr = neighbors[idx, dir_vals[k]]

            # 1. Hopping (probability D) — move to empty neighbor
            if r < 0.25:
                if occupant != EMPTY and grid[nbr] == EMPTY:
                    if prob_vals[k] < D:
                        grid[nbr] = occupant
                        grid[idx] = EMPTY

            # 2. Predator death (probability mu)
            elif r < 0.5:
                if occupant == PREDATOR and prob_vals[k] < mu:
                    grid[idx] = EMPTY

            # 3. Predation: A + B -> A + A (probability lambda)
            elif r < 0.75:
                if occupant == PREDATOR and grid[nbr] == PREY:
                    if prob_vals[k] < lam:
                        grid[nbr] = PREDATOR

            # 4. Prey reproduction: B -> B + B on empty neighbors (probability sigma)
            else:
                if occupant == PREY:
                    for d in range(4):
                        nn = neighbors[idx, d]
                        if grid[nn] == EMPTY:
                            if np.random.random() < sigma:
                                grid[nn] = PREY

        # Record populations
        n_pred = np.sum(grid == PREDATOR)
        n_prey = np.sum(grid == PREY)
        pop_history[step - 1] = [step, n_pred, n_prey]

        # Save snapshot
        if step in snapshot_steps:
            snapshots[step] = grid.copy().reshape(n, n)

    return pop_history, snapshots


# ========================== ANALYSIS ==========================

def analyze_lambda(lam_values, mu, sigma, D, lattice_size, mc_steps, n_runs):
    """
    Run simulations for multiple lambda values and compute statistics.
    """
    results = {}

    for lam in lam_values:
        print(f"\n=== Running simulations for lambda = {lam} ===")
        runs = []
        for run in range(n_runs):
            seed = 1000 + int(lam * 100) + run
            hist, snaps = run_simulation(lattice_size, mu, sigma, lam, D, mc_steps, seed=seed)
            runs.append((hist, snaps))
            print(f"  Run {run + 1} (seed={seed}): "
                  f"Final P={hist[-1, 1]:.0f}, B={hist[-1, 2]:.0f}, "
                  f"ratio={hist[-1, 1] / max(hist[-1, 2], 1):.3f}")
        results[lam] = runs

    return results


def plot_dynamics(results, lam_values, output_dir):
    """Plot population dynamics for all lambda values."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, lam in enumerate(lam_values):
        ax = axes[idx]
        runs = results[lam]

        # Plot individual runs (light)
        for hist, _ in runs:
            ax.plot(hist[:, 0], hist[:, 1], color='red', alpha=0.3, linewidth=0.8)
            ax.plot(hist[:, 0], hist[:, 2], color='green', alpha=0.3, linewidth=0.8)

        # Plot mean (bold)
        all_pred = np.array([r[0][:, 1] for r in runs])
        all_prey = np.array([r[0][:, 2] for r in runs])
        mean_pred = all_pred.mean(axis=0)
        mean_prey = all_prey.mean(axis=0)
        steps = runs[0][0][:, 0]

        ax.plot(steps, mean_pred, 'r-', linewidth=2, label='Predators (mean)')
        ax.plot(steps, mean_prey, 'g-', linewidth=2, label='Prey (mean)')

        ax.set_title(f'$\lambda$ = {lam}', fontsize=14)
        ax.set_xlabel('MC Step')
        ax.set_ylabel('Population')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_lambda_dynamics.png', dpi=150)
    plt.show()


def plot_lattice_snapshots(results, lam_values, output_dir):
    """Plot spatial lattice snapshots for selected lambda values."""
    snapshot_steps = [0, 100, 250, 500, 1000]
    cmap = colors.ListedColormap(['white', 'red', 'green'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for lam in lam_values[:3]:  # Plot first 3 lambda values
        hist, snaps = results[lam][0]
        n_snaps = len(snapshot_steps)
        fig, axes = plt.subplots(1, n_snaps, figsize=(18, 4))

        for idx, step in enumerate(snapshot_steps):
            grid = snaps.get(step, snaps[max(snaps.keys())])
            ax = axes[idx]
            ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
            n_pred = np.sum(grid == PREDATOR)
            n_prey = np.sum(grid == PREY)
            ax.set_title(f'Step {step}\nP={n_pred}, B={n_prey}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'Spatial Distribution ($\lambda$ = {lam}) — '
                     f'White=Empty, Red=Predator, Green=Prey',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'lattice_lambda_{lam}.png', dpi=150, bbox_inches='tight')
        plt.show()


def plot_statistics(results, lam_values, output_dir):
    """Plot bar charts comparing final statistics across lambda values."""
    avg_pred, std_pred = [], []
    avg_prey, std_prey = [], []
    ratio_mean, ratio_std = [], []
    final_pred, final_prey = [], []

    for lam in lam_values:
        runs = results[lam]
        preds = [r[0][-100:, 1].mean() for r in runs]
        preys = [r[0][-100:, 2].mean() for r in runs]
        ratios = [r[0][-1, 1] / max(r[0][-1, 2], 1) for r in runs]
        fp = [r[0][-1, 1] for r in runs]
        fb = [r[0][-1, 2] for r in runs]

        avg_pred.append(np.mean(preds))
        std_pred.append(np.std(preds))
        avg_prey.append(np.mean(preys))
        std_prey.append(np.std(preys))
        ratio_mean.append(np.mean(ratios))
        ratio_std.append(np.std(ratios))
        final_pred.append(np.mean(fp))
        final_prey.append(np.mean(fb))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(lam_vals))
    width = 0.35

    # Average populations (last 100 steps)
    axes[0].bar(x - width/2, avg_pred, width, yerr=std_pred,
                label='Predators', color='red', alpha=0.7, capsize=4)
    axes[0].bar(x + width/2, avg_prey, width, yerr=std_prey,
                label='Prey', color='green', alpha=0.7, capsize=4)
    axes[0].set_xlabel('Predation Rate $\lambda$')
    axes[0].set_ylabel('Avg Population (last 100 MCS)')
    axes[0].set_title('Average Populations')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(l) for l in lam_values])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Final ratio
    axes[1].bar(x, ratio_mean, yerr=ratio_std, color='blue', alpha=0.7, capsize=4)
    axes[1].set_xlabel('Predation Rate $\lambda$')
    axes[1].set_ylabel('Predator/Prey Ratio')
    axes[1].set_title('Final Predator/Prey Ratio')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(l) for l in lam_values])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Final populations
    axes[2].bar(x - width/2, final_pred, width, label='Predators', color='red', alpha=0.7)
    axes[2].bar(x + width/2, final_prey, width, label='Prey', color='green', alpha=0.7)
    axes[2].set_xlabel('Predation Rate $\lambda$')
    axes[2].set_ylabel('Final Population')
    axes[2].set_title('Final Populations (step 1000)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(l) for l in lam_values])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'statistics_comparison.png', dpi=150)
    plt.show()


# ========================== MAIN ==========================

if __name__ == '__main__':
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Lambda values to investigate
    lam_values = [0.1, 0.25, 0.5, 0.75, 1.0]

    print("=" * 60)
    print("STOCHASTIC LATTICE LOTKA-VOLTERRA MODEL")
    print("Case Study: Effect of Predation Rate (lambda)")
    print("=" * 60)
    print(f"Lattice size: {LATTICE_SIZE} x {LATTICE_SIZE}")
    print(f"Parameters: mu={MU}, sigma={SIGMA}, D={D}")
    print(f"MC steps: {MC_STEPS}, Runs per lambda: {N_RUNS}")
    print("=" * 60)

    # Run simulations
    results = analyze_lambda(lam_values, MU, SIGMA, D, LATTICE_SIZE, MC_STEPS, N_RUNS)

    # Generate plots
    plot_dynamics(results, lam_values, output_dir)
    plot_lattice_snapshots(results, lam_values, output_dir)
    plot_statistics(results, lam_values, output_dir)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE — Check the 'output' folder for figures.")
    print("=" * 60)
