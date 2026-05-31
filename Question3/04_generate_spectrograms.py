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

BASE_DIR = Path(__file__).resolve().parent

MUSIC_DIR = BASE_DIR / "wav9/music"
NOISE_DIR = BASE_DIR / "wav9/noise"
OUT_DIR = BASE_DIR / "results/spectrograms"

# Number of clips to plot for each class.
N_PER_CLASS = 10

# Optional frequency cap for the plot (Hz).
# Set to None to show the full band up to Nyquist.
FREQ_MAX_HZ = None

# Font sizes for plots (tune for report readability).
TITLE_FONT_SIZE = 24.0
LABEL_FONT_SIZE = 24.0
TICK_FONT_SIZE = 20.0

# Combined spectrogram figure (shared axes and colorbar).
COMBINED_ENABLED = True
COMBINED_MUSIC_FILENAME = "2026-05-11 17-18-20.wav"
COMBINED_NOISE_FILENAME = "2026-05-11 17-14-19.wav"
COMBINED_OUT_NAME = "combined_music_noise_spectrogram.png"
COMBINED_FONT_SCALE = 0.95

EPS = 1e-12


def compute_spectrogram_db(filepath):
    """
    Compute spectrogram in dB for one audio file.
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
    Sxx_db = 10 * np.log10(Sxx + EPS)

    return freqs, times, Sxx_db


def plot_spectrogram(filepath, label, freqs, times, Sxx_db, vmin, vmax):
    """
    Create one spectrogram image for one audio file.
    """

    plt.figure(figsize=(9, 5))
    plt.pcolormesh(
        times,
        freqs,
        Sxx_db,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel("Time [s]", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Frequency [Hz]", fontsize=LABEL_FONT_SIZE)
    # No title for individual spectrograms (keeps figures clean for reports).
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)

    cbar = plt.colorbar(label="Power/Frequency [dB/Hz]")
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    cbar.set_label("Power/Frequency [dB/Hz]", fontsize=LABEL_FONT_SIZE)

    # Optional frequency cap for easier visual comparison.
    if FREQ_MAX_HZ is not None:
        plt.ylim(0, FREQ_MAX_HZ)

    plt.tight_layout()

    output_path = OUT_DIR / f"{label}_{filepath.stem}_spectrogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_combined_spectrograms(music_path, noise_path):
    """
    Create a combined spectrogram figure with shared axes and colorbar.
    """

    freqs_m, times_m, Sxx_db_m = compute_spectrogram_db(music_path)
    freqs_n, times_n, Sxx_db_n = compute_spectrogram_db(noise_path)

    vmin = float(min(np.min(Sxx_db_m), np.min(Sxx_db_n)))
    vmax = float(max(np.max(Sxx_db_m), np.max(Sxx_db_n)))

    title_size = TITLE_FONT_SIZE * COMBINED_FONT_SCALE
    label_size = LABEL_FONT_SIZE * COMBINED_FONT_SCALE
    tick_size = TICK_FONT_SIZE * COMBINED_FONT_SCALE
    tick_length = 6 * COMBINED_FONT_SCALE
    tick_width = 1.2 * COMBINED_FONT_SCALE

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.25},
    )

    mesh_m = axes[0].pcolormesh(
        times_m,
        freqs_m,
        Sxx_db_m,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )
    # No titles for combined plots (cleaner for reports).

    mesh_n = axes[1].pcolormesh(
        times_n,
        freqs_n,
        Sxx_db_n,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    for ax in axes:
        ax.tick_params(
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        if FREQ_MAX_HZ is not None:
            ax.set_ylim(0, FREQ_MAX_HZ)

    fig.supxlabel("Time [s]", fontsize=label_size)
    fig.supylabel("Frequency [Hz]", fontsize=label_size, x=0.02)

    fig.subplots_adjust(left=0.14, right=0.82, bottom=0.12, top=0.9)
    cbar_ax = fig.add_axes([0.84, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(
        mesh_n,
        cax=cbar_ax,
        label="Power/Frequency [dB/Hz]",
    )
    cbar.ax.tick_params(
        labelsize=tick_size,
        length=tick_length,
        width=tick_width,
    )
    cbar.set_label("Power/Frequency [dB/Hz]", fontsize=label_size, labelpad=10)

    # Layout handled by subplots_adjust + fixed colorbar axis.

    output_path = OUT_DIR / COMBINED_OUT_NAME
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined: {output_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    music_files = sorted(MUSIC_DIR.glob("*.wav"))[:N_PER_CLASS]
    noise_files = sorted(NOISE_DIR.glob("*.wav"))[:N_PER_CLASS]

    files_with_labels = [(f, "music") for f in music_files] + [
        (f, "noise") for f in noise_files
    ]

    # Precompute all spectrograms to get a global dB range.
    spec_data = []
    global_min = None
    global_max = None

    for filepath, label in files_with_labels:
        freqs, times, Sxx_db = compute_spectrogram_db(filepath)
        spec_data.append((filepath, label, freqs, times, Sxx_db))

        local_min = float(np.min(Sxx_db))
        local_max = float(np.max(Sxx_db))

        if global_min is None or local_min < global_min:
            global_min = local_min
        if global_max is None or local_max > global_max:
            global_max = local_max

    print("Generating music spectrograms...")
    for filepath, label, freqs, times, Sxx_db in spec_data:
        if label == "music":
            plot_spectrogram(
                filepath,
                label,
                freqs,
                times,
                Sxx_db,
                global_min,
                global_max,
            )

    print("\nGenerating noise spectrograms...")
    for filepath, label, freqs, times, Sxx_db in spec_data:
        if label == "noise":
            plot_spectrogram(
                filepath,
                label,
                freqs,
                times,
                Sxx_db,
                global_min,
                global_max,
            )

    if COMBINED_ENABLED:
        music_path = MUSIC_DIR / COMBINED_MUSIC_FILENAME
        noise_path = NOISE_DIR / COMBINED_NOISE_FILENAME

        if music_path.exists() and noise_path.exists():
            print("\nGenerating combined spectrogram...")
            plot_combined_spectrograms(music_path, noise_path)
        else:
            print("\nSkipping combined spectrogram (missing file).")

    print("\nDone.")


if __name__ == "__main__":
    main()
