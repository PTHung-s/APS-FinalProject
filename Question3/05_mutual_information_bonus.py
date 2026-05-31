"""
05_mutual_information_bonus.py

Purpose:
- Estimate mutual information between X_t and X_{t+lag}.
- X_t is the discretized amplitude at time t.
- X_{t+lag} is the discretized amplitude after a delay.

Interpretation:
- Higher mutual information means stronger temporal dependence.
- Music may have higher temporal dependence because it has rhythm and structure.
- Noise may have lower temporal dependence if it is more random.

Input folders:
- wav9/music
- wav9/noise

Output files:
- results/mutual_information_per_clip.csv
- results/mutual_information_summary.csv
- results/mutual_information_plot.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent

MUSIC_DIR = BASE_DIR / "wav9/music"
NOISE_DIR = BASE_DIR / "wav9/noise"
RESULTS_DIR = BASE_DIR / "results"

DELTA = 0.1

BIN_MIN = -1.0
BIN_MAX = 1.0

# Lags are measured in samples.
# At 48 kHz:
# lag = 1 means 1/48000 seconds.
# lag = 1000 means about 0.0208 seconds.
LAGS = [1, 10, 100, 1000]


def discretize_signal(x):
    """
    Convert continuous amplitude samples into discrete bin indices.
    """

    x = np.clip(x, BIN_MIN, BIN_MAX)

    bin_edges = np.arange(BIN_MIN, BIN_MAX + DELTA, DELTA)
    K = len(bin_edges) - 1

    # Digitize returns indices from 1 to K.
    # Subtract 1 to make them from 0 to K-1.
    indices = np.digitize(x, bin_edges) - 1

    # Fix edge case where x equals BIN_MAX.
    indices = np.clip(indices, 0, K - 1)

    return indices, K


def mutual_information_discrete(x_bins, y_bins, K):
    """
    Estimate discrete mutual information:

    I(X;Y) = sum p(x,y) log2( p(x,y) / (p(x)p(y)) )
    """

    joint_counts = np.zeros((K, K), dtype=float)

    # Count joint occurrences of pairs (X_t, X_{t+lag}).
    for xb, yb in zip(x_bins, y_bins):
        joint_counts[xb, yb] += 1

    joint_probs = joint_counts / joint_counts.sum()

    px = joint_probs.sum(axis=1)
    py = joint_probs.sum(axis=0)

    mi = 0.0

    for i in range(K):
        for j in range(K):
            pxy = joint_probs[i, j]

            if pxy > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy * np.log2(pxy / (px[i] * py[j]))

    return mi


def compute_mi_for_clip(filepath, label, lag):
    """
    Compute mutual information for one clip and one lag.
    """

    x, sr = sf.read(filepath)

    # Convert stereo to mono if needed.
    if x.ndim > 1:
        x = x.mean(axis=1)

    x = x.astype(float)

    x_bins, K = discretize_signal(x)

    # X is signal at time t.
    X = x_bins[:-lag]

    # Y is signal at time t + lag.
    Y = x_bins[lag:]

    mi = mutual_information_discrete(X, Y, K)

    return {
        "file": filepath.name,
        "class": label,
        "sampling_rate": sr,
        "delta": DELTA,
        "lag_samples": lag,
        "lag_seconds": lag / sr,
        "K": K,
        "mutual_information_bits": mi,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    all_files = []

    for wav_file in sorted(MUSIC_DIR.glob("*.wav")):
        all_files.append((wav_file, "music"))

    for wav_file in sorted(NOISE_DIR.glob("*.wav")):
        all_files.append((wav_file, "noise"))

    for lag in LAGS:
        print(f"Computing MI for lag = {lag} samples")

        for filepath, label in all_files:
            rows.append(compute_mi_for_clip(filepath, label, lag))

    df = pd.DataFrame(rows)

    per_clip_path = RESULTS_DIR / "mutual_information_per_clip.csv"
    df.to_csv(per_clip_path, index=False)

    summary = df.groupby(["lag_samples", "class"])[
        "mutual_information_bits"
    ].agg(["mean", "std"])

    summary_path = RESULTS_DIR / "mutual_information_summary.csv"
    summary.to_csv(summary_path)

    # Plot mean MI versus lag.
    mi_mean = (
        df.groupby(["lag_samples", "class"])["mutual_information_bits"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    for label in sorted(mi_mean["class"].unique()):
        sub = mi_mean[mi_mean["class"] == label]
        plt.plot(
            sub["lag_samples"],
            sub["mutual_information_bits"],
            marker="o",
            label=label
        )

    plt.xscale("log")
    plt.xlabel("Lag in samples")
    plt.ylabel("Mean mutual information [bits]")
    plt.title("Mutual Information Between X_t and X_{t+lag}")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "mutual_information_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\nMutual information summary:")
    print(summary)

    print("\nSaved files:")
    print(per_clip_path)
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
