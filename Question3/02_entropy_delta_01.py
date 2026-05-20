"""
02_entropy_delta_01.py

Purpose:
- Analyze 10 music clips and 10 noise clips.
- Use fixed bin width Delta = 0.1.
- Compute:
    1. Discrete entropy H_Q
    2. Differential entropy estimate h_hat
    3. Maximum discrete entropy H_max
    4. Gaussian reference entropy h_Gauss
    5. Gap = h_Gauss - h_hat

Input folders:
- wav9/music
- wav9/noise

Output files:
- results/entropy_per_clip_delta_0.1.csv
- results/entropy_summary_delta_0.1.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf


# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent

MUSIC_DIR = BASE_DIR / "wav9/music"
NOISE_DIR = BASE_DIR / "wav9/noise"
RESULTS_DIR = BASE_DIR / "results"

DELTA = 0.1

# Audio samples are normalized to the range [-1, 1] when read from WAV.
BIN_MIN = -1.0
BIN_MAX = 1.0


def compute_entropy_for_clip(filepath, label):
    """
    Compute entropy-related values for one audio clip.

    Parameters:
    - filepath: path to the WAV file
    - label: class label, either "music" or "noise"

    Returns:
    - A dictionary containing entropy values for this clip.
    """

    # Read audio samples and sampling rate.
    x, sr = sf.read(filepath)

    # If the audio is stereo, convert to mono by averaging left and right channels.
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Convert to floating-point numbers.
    x = x.astype(float)

    # Clip very small numerical overflow to [-1, 1].
    x = np.clip(x, BIN_MIN, BIN_MAX)

    # Create equal-width bins from -1 to 1.
    # With Delta = 0.1, bins are:
    # [-1.0, -0.9), [-0.9, -0.8), ..., [0.9, 1.0]
    bin_edges = np.arange(BIN_MIN, BIN_MAX + DELTA, DELTA)

    # Count how many samples fall into each bin.
    counts, _ = np.histogram(x, bins=bin_edges)

    # Convert counts into probabilities.
    probs = counts / counts.sum()

    # Remove zero probabilities because log2(0) is undefined.
    probs_nonzero = probs[probs > 0]

    # Number of bins.
    K = len(counts)

    # 1. Discrete entropy:
    # H_Q = - sum p_i log2(p_i)
    H_Q = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    # 2. Differential entropy estimate:
    # h_hat = H_Q + log2(Delta)
    h_hat = H_Q + np.log2(DELTA)

    # 3. Maximum possible discrete entropy:
    # H_max = log2(K)
    H_max = np.log2(K)

    # 4. Gaussian reference entropy:
    # h_Gauss = 0.5 log2(2*pi*e*sigma^2)
    variance = np.var(x)

    # Avoid log(0) if a clip is completely silent.
    eps = 1e-12
    h_Gauss = 0.5 * np.log2(2 * np.pi * np.e * max(variance, eps))

    # 5. Gap between Gaussian entropy and estimated entropy.
    # Smaller gap means closer to Gaussian.
    gap = h_Gauss - h_hat

    return {
        "file": filepath.name,
        "class": label,
        "sampling_rate": sr,
        "num_samples": len(x),
        "delta": DELTA,
        "K": K,
        "variance": variance,
        "H_Q": H_Q,
        "h_hat": h_hat,
        "H_max": H_max,
        "h_Gauss": h_Gauss,
        "gap_h_Gauss_minus_h_hat": gap,
    }


def main():
    """
    Main function:
    - Analyze all music WAV files.
    - Analyze all noise WAV files.
    - Save per-clip results and class-level summary.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # Analyze music clips.
    for wav_file in sorted(MUSIC_DIR.glob("*.wav")):
        rows.append(compute_entropy_for_clip(wav_file, "music"))

    # Analyze noise clips.
    for wav_file in sorted(NOISE_DIR.glob("*.wav")):
        rows.append(compute_entropy_for_clip(wav_file, "noise"))

    df = pd.DataFrame(rows)

    # Save result for each individual clip.
    per_clip_path = RESULTS_DIR / "entropy_per_clip_delta_0.1.csv"
    df.to_csv(per_clip_path, index=False)

    # Compute mean and standard deviation for each class.
    summary = df.groupby("class")[
        ["H_Q", "h_hat", "H_max", "h_Gauss", "gap_h_Gauss_minus_h_hat"]
    ].agg(["mean", "std"])

    summary_path = RESULTS_DIR / "entropy_summary_delta_0.1.csv"
    summary.to_csv(summary_path)

    print("\nPer-clip results:")
    print(df)

    print("\nSummary by class:")
    print(summary)

    print("\nSaved files:")
    print(per_clip_path)
    print(summary_path)


if __name__ == "__main__":
    main()
