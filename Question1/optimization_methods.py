"""
Optimization methods for Question 1.

This script implements two sampling-based optimization methods for
n-dimensional ball volume estimation:

1. Quasi-Monte Carlo using Sobol low-discrepancy sequence.
2. Markov Chain Monte Carlo using a random-walk proposal.

It produces:
- optimization_results.csv
- figures/qmc_sampling.png
- figures/markov_mc_sampling.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import gammaln
from scipy.stats import qmc


# ============================================================
# Basic geometry and exact volume
# ============================================================

def exact_nball_volume(n, r):
    """
    Exact volume of an n-dimensional ball with radius r.

    Formula:
        V_n(r) = pi^(n/2) / Gamma(n/2 + 1) * r^n

    The computation is done in log-space for numerical stability.
    """
    n = int(n)

    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if r <= 0:
        raise ValueError("r must be positive.")

    log_v = (n / 2.0) * np.log(np.pi) - gammaln(n / 2.0 + 1.0) + n * np.log(r)

    return float(np.exp(log_v))


def get_shape_config(shape, n):
    """
    Returns the bounding box, center, radius, and box volume.

    A_n:
        ball of radius 0.5 centered at (0.5, ..., 0.5),
        inside [0, 1]^n.

    B_n:
        unit ball centered at the origin,
        inside [-1, 1]^n.
    """
    shape = str(shape).upper()
    n = int(n)

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if shape == "A":
        lower = np.zeros(n)
        upper = np.ones(n)
        center = np.full(n, 0.5)
        radius = 0.5

    elif shape == "B":
        lower = -np.ones(n)
        upper = np.ones(n)
        center = np.zeros(n)
        radius = 1.0

    else:
        raise ValueError("shape must be either 'A' or 'B'.")

    box_volume = float(np.prod(upper - lower))

    return lower, upper, center, radius, box_volume


def is_inside_ball(points, center, radius):
    """
    Checks whether points lie inside the n-dimensional ball.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    squared_distance = np.sum((points - center) ** 2, axis=1)

    return squared_distance <= radius ** 2


def check_power_of_two(N):
    """
    Sobol.random_base2 requires N = 2^m.
    """
    N = int(N)

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a positive power of 2, for example 2**10, 2**12, or 2**14.")

    return int(np.log2(N))


# ============================================================
# Quasi-Monte Carlo
# ============================================================

def quasi_mc_volume(shape, n, N=2**14, seed=42):
    """
    Quasi-Monte Carlo volume estimation using Sobol sequence.

    Estimator:
        V_hat = V_box * mean(I(x_i is inside the ball))

    where x_i are Sobol points mapped to the bounding box.
    """
    m = check_power_of_two(N)

    lower, upper, center, radius, box_volume = get_shape_config(shape, n)

    sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
    unit_points = sampler.random_base2(m=m)

    points = lower + (upper - lower) * unit_points
    inside = is_inside_ball(points, center, radius)

    estimate = box_volume * np.mean(inside)

    return float(estimate)


def get_qmc_points(shape, n=2, N=1024, seed=42):
    """
    Generates QMC points for 2D visualization.
    """
    if n != 2:
        raise ValueError("This visualization function is only for n = 2.")

    m = check_power_of_two(N)

    lower, upper, center, radius, _ = get_shape_config(shape, n)

    sampler = qmc.Sobol(d=n, scramble=True, seed=seed)
    unit_points = sampler.random_base2(m=m)

    points = lower + (upper - lower) * unit_points
    inside = is_inside_ball(points, center, radius)

    return points, inside, center, radius, lower, upper


# ============================================================
# Markov Chain Monte Carlo
# ============================================================

def markov_mc_volume(shape, n, N=2**14, burn_in=1000, step_scale=0.25, seed=42):
    """
    Markov MC volume estimation using a random-walk proposal.

    The target distribution is uniform over the bounding box.
    A proposal is accepted if it remains inside the bounding box.
    Otherwise, the chain stays at the current position.

    Estimator:
        V_hat = V_box * mean(I(X_t is inside the ball))

    Note:
        Markov MC samples are correlated, so this method is used as an
        alternative sampling strategy rather than being assumed to always
        outperform independent Monte Carlo.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive.")

    rng = np.random.default_rng(seed)

    lower, upper, center, radius, box_volume = get_shape_config(shape, n)

    x = rng.uniform(lower, upper)
    step_size = step_scale * (upper - lower)

    hits = np.empty(N, dtype=bool)

    accepted = 0
    kept = 0
    total_steps = N + burn_in

    for t in range(total_steps):
        proposal = x + rng.normal(loc=0.0, scale=step_size, size=n)

        if np.all((proposal >= lower) & (proposal <= upper)):
            x = proposal
            accepted += 1

        if t >= burn_in:
            hits[kept] = np.sum((x - center) ** 2) <= radius ** 2
            kept += 1

    estimate = box_volume * np.mean(hits)
    acceptance_rate = accepted / total_steps

    return float(estimate), float(acceptance_rate)


def get_markov_mc_points(shape, n=2, N=1024, burn_in=200, step_scale=0.25, seed=42):
    """
    Generates Markov MC points for 2D visualization.
    """
    if n != 2:
        raise ValueError("This visualization function is only for n = 2.")

    rng = np.random.default_rng(seed)

    lower, upper, center, radius, _ = get_shape_config(shape, n)

    x = rng.uniform(lower, upper)
    step_size = step_scale * (upper - lower)

    points = []
    total_steps = N + burn_in

    for t in range(total_steps):
        proposal = x + rng.normal(loc=0.0, scale=step_size, size=n)

        if np.all((proposal >= lower) & (proposal <= upper)):
            x = proposal

        if t >= burn_in:
            points.append(x.copy())

    points = np.array(points)
    inside = is_inside_ball(points, center, radius)

    return points, inside, center, radius, lower, upper


# ============================================================
# Experiment table
# ============================================================

def run_optimization_table(N=2**14, seeds=(11, 22, 33, 44, 55), step_scale=0.25, burn_in=1000):
    """
    Runs QMC and Markov MC for A_2, B_2, A_3, and B_3.

    Multiple seeds are used to reduce the dependence on one random run.
    """
    cases = [
        ("A", 2),
        ("B", 2),
        ("A", 3),
        ("B", 3),
    ]

    rows = []

    for shape, n in cases:
        _, _, _, radius, _ = get_shape_config(shape, n)
        exact = exact_nball_volume(n, radius)

        qmc_estimates = np.array([
            quasi_mc_volume(shape, n, N=N, seed=seed)
            for seed in seeds
        ])

        mcmc_outputs = [
            markov_mc_volume(
                shape,
                n,
                N=N,
                burn_in=burn_in,
                step_scale=step_scale,
                seed=seed
            )
            for seed in seeds
        ]

        mcmc_estimates = np.array([output[0] for output in mcmc_outputs])
        mcmc_acceptance_rates = np.array([output[1] for output in mcmc_outputs])

        method_results = [
            ("Quasi-Monte Carlo (Sobol)", qmc_estimates, None),
            ("Markov MC (Random Walk)", mcmc_estimates, np.mean(mcmc_acceptance_rates)),
        ]

        for method_name, estimates, acceptance_rate in method_results:
            mean_estimate = float(np.mean(estimates))
            std_estimate = float(np.std(estimates, ddof=1))

            rows.append({
                "Shape": f"{shape}_{n}",
                "Dimension": n,
                "Method": method_name,
                "N per run": N,
                "Runs": len(seeds),
                "Mean Estimated Volume": mean_estimate,
                "Std. Dev.": std_estimate,
                "Exact Volume": exact,
                "Absolute Error": abs(mean_estimate - exact),
                "Relative Error (%)": 100.0 * abs(mean_estimate - exact) / exact,
                "Mean Acceptance Rate": np.nan if acceptance_rate is None else float(acceptance_rate),
            })

    return pd.DataFrame(rows)


# ============================================================
# Figures
# ============================================================

def plot_qmc_sampling_2d(output_path=None, N=1024, seed=42):
    """
    Saves a 2D visualization of QMC sampling for A_2 and B_2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, shape in zip(axes, ["A", "B"]):
        points, inside, center, radius, lower, upper = get_qmc_points(
            shape,
            n=2,
            N=N,
            seed=seed
        )

        ax.scatter(points[~inside, 0], points[~inside, 1], s=8, alpha=0.35, label="Outside")
        ax.scatter(points[inside, 0], points[inside, 1], s=8, alpha=0.70, label="Inside")

        circle = plt.Circle(center, radius, fill=False, linewidth=2)
        ax.add_patch(circle)

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal")

        ax.set_title(f"QMC Sampling for {shape}_2")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_markov_mc_sampling_2d(output_path=None, N=1024, burn_in=200, step_scale=0.25, seed=42):
    """
    Saves a 2D visualization of Markov MC sampling for A_2 and B_2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, shape in zip(axes, ["A", "B"]):
        points, inside, center, radius, lower, upper = get_markov_mc_points(
            shape,
            n=2,
            N=N,
            burn_in=burn_in,
            step_scale=step_scale,
            seed=seed
        )

        ax.scatter(points[~inside, 0], points[~inside, 1], s=8, alpha=0.35, label="Outside")
        ax.scatter(points[inside, 0], points[inside, 1], s=8, alpha=0.70, label="Inside")

        circle = plt.Circle(center, radius, fill=False, linewidth=2)
        ax.add_patch(circle)

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal")

        ax.set_title(f"Markov MC Sampling for {shape}_2")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    results = run_optimization_table(
        N=2**14,
        seeds=(11, 22, 33, 44, 55),
        step_scale=0.25,
        burn_in=1000
    )

    results_path = base_dir / "optimization_results.csv"
    results.to_csv(results_path, index=False)

    plot_qmc_sampling_2d(
        output_path=figures_dir / "qmc_sampling.png",
        N=1024,
        seed=42
    )

    plot_markov_mc_sampling_2d(
        output_path=figures_dir / "markov_mc_sampling.png",
        N=1024,
        burn_in=200,
        step_scale=0.25,
        seed=42
    )

    print("Optimization experiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Figures saved to: {figures_dir}")
    print()
    print(results)
