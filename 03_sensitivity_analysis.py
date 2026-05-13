"""
03_sensitivity_analysis.py

Purpose:
- Repeat entropy analysis using different bin widths.
- Delta values:
    0.05, 0.1, 0.2, 0.5
- Check whether the conclusion about Gaussian closeness is stable.

Input folders:
- wav9/music
- wav9/noise

Output files:
- results/sensitivity_per_clip.csv
- results/sensitivity_summary.csv
- results/sensitivity_gap_plot.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

MUSIC_DIR = Path("wav9/music")
NOISE_DIR = Path("wav9/noise")
RESULTS_DIR = Path("results")

DELTAS = [0.05, 0.1, 0.2, 0.5]

BIN_MIN = -1.0
BIN_MAX = 1.0


def compute_entropy_for_clip(filepath, label, delta):
    """
    Compute entropy values for one clip using a given bin width delta.
    """

    x, sr = sf.read(filepath)

    # Convert stereo to mono if needed.
    if x.ndim > 1:
        x = x.mean(axis=1)

    x = x.astype(float)
    x = np.clip(x, BIN_MIN, BIN_MAX)

    # Create bins using current delta.
    bin_edges = np.arange(BIN_MIN, BIN_MAX + delta, delta)

    # Histogram count.
    counts, _ = np.histogram(x, bins=bin_edges)

    # Convert to probability distribution.
    probs = counts / counts.sum()
    probs_nonzero = probs[probs > 0]

    K = len(counts)

    # Discrete entropy.
    H_Q = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    # Differential entropy estimate.
    h_hat = H_Q + np.log2(delta)

    # Maximum discrete entropy.
    H_max = np.log2(K)

    # Gaussian reference entropy.
    variance = np.var(x)
    eps = 1e-12
    h_Gauss = 0.5 * np.log2(2 * np.pi * np.e * max(variance, eps))

    # Gap.
    gap = h_Gauss - h_hat

    return {
        "file": filepath.name,
        "class": label,
        "sampling_rate": sr,
        "num_samples": len(x),
        "delta": delta,
        "K": K,
        "variance": variance,
        "H_Q": H_Q,
        "h_hat": h_hat,
        "H_max": H_max,
        "h_Gauss": h_Gauss,
        "gap_h_Gauss_minus_h_hat": gap,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    all_files = []

    # Collect all music files.
    for wav_file in sorted(MUSIC_DIR.glob("*.wav")):
        all_files.append((wav_file, "music"))

    # Collect all noise files.
    for wav_file in sorted(NOISE_DIR.glob("*.wav")):
        all_files.append((wav_file, "noise"))

    # Run analysis for each delta.
    for delta in DELTAS:
        print(f"Analyzing Delta = {delta}")

        for filepath, label in all_files:
            rows.append(compute_entropy_for_clip(filepath, label, delta))

    df = pd.DataFrame(rows)

    # Save per-clip sensitivity results.
    per_clip_path = RESULTS_DIR / "sensitivity_per_clip.csv"
    df.to_csv(per_clip_path, index=False)

    # Save summary.
    summary = df.groupby(["delta", "class"])[
        ["H_Q", "h_hat", "H_max", "h_Gauss", "gap_h_Gauss_minus_h_hat"]
    ].agg(["mean", "std"])

    summary_path = RESULTS_DIR / "sensitivity_summary.csv"
    summary.to_csv(summary_path)

    # Plot mean Gaussian gap versus delta.
    gap_mean = (
        df.groupby(["delta", "class"])["gap_h_Gauss_minus_h_hat"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    for label in sorted(gap_mean["class"].unique()):
        sub = gap_mean[gap_mean["class"] == label]
        plt.plot(
            sub["delta"],
            sub["gap_h_Gauss_minus_h_hat"],
            marker="o",
            label=label
        )

    plt.xlabel("Bin width Delta")
    plt.ylabel("Mean gap: h_Gauss - h_hat")
    plt.title("Sensitivity Analysis of Gaussian Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "sensitivity_gap_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\nSensitivity summary:")
    print(summary)

    print("\nSaved files:")
    print(per_clip_path)
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()