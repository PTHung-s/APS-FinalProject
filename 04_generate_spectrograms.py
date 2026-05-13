"""
04_generate_spectrograms.py

Purpose:
- Generate spectrogram images for selected music and noise clips.
- A spectrogram shows how frequency content changes over time.

Input folders:
- wav9/music
- wav9/noise

Output folder:
- results/spectrograms
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# =========================
# Configuration
# =========================

MUSIC_DIR = Path("wav9/music")
NOISE_DIR = Path("wav9/noise")
OUT_DIR = Path("results/spectrograms")

# Number of clips to plot for each class.
N_PER_CLASS = 10


def plot_spectrogram(filepath, label):
    """
    Create one spectrogram image for one audio file.
    """

    x, sr = sf.read(filepath)

    # Convert stereo to mono if needed.
    if x.ndim > 1:
        x = x.mean(axis=1)

    x = x.astype(float)

    # Compute spectrogram.
    # freqs: frequency axis
    # times: time axis
    # Sxx: power at each time-frequency cell
    freqs, times, Sxx = spectrogram(
        x,
        fs=sr,
        nperseg=1024,
        noverlap=512,
        scaling="density"
    )

    # Convert power to decibels for easier visualization.
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(9, 5))
    plt.pcolormesh(times, freqs, Sxx_db, shading="gouraud")

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"Spectrogram - {label} - {filepath.name}")
    plt.colorbar(label="Power/Frequency [dB/Hz]")

    # Focus on the lower frequency range because it is easier to interpret.
    plt.ylim(0, 10000)

    plt.tight_layout()

    output_path = OUT_DIR / f"{label}_{filepath.stem}_spectrogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    music_files = sorted(MUSIC_DIR.glob("*.wav"))[:N_PER_CLASS]
    noise_files = sorted(NOISE_DIR.glob("*.wav"))[:N_PER_CLASS]

    print("Generating music spectrograms...")
    for f in music_files:
        plot_spectrogram(f, "music")

    print("\nGenerating noise spectrograms...")
    for f in noise_files:
        plot_spectrogram(f, "noise")

    print("\nDone.")


if __name__ == "__main__":
    main()