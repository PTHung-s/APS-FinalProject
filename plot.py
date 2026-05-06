import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(samples, acceptance_rate, bins=30):

    # Convert samples from list to numpy array
    samples = np.array(samples)

    # Create x values for target distribution
    x_grid = np.linspace(-4, 4, 500)

    # Normalized target density: N(0,1)
    target_density = (
        1 / np.sqrt(2 * np.pi)
    ) * np.exp(-x_grid**2 / 2)

    # Plot
    plt.figure(figsize=(7, 6))

    plt.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Metropolis Samples"
    )

    plt.plot(
        x_grid,
        target_density,
        linewidth=2,
        label="Target Distribution = f(x)/Z"
    )

    plt.title("Sampling with Metropolis Algorithm")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.figtext(
        0.5,
        -0.02,
        f"Acceptance rate: {acceptance_rate:.4f}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

    plt.show()