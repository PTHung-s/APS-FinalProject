"""
00_trim_to_9s.py

Purpose:
- Trim all .wav files to 9 seconds (first 9 seconds).
- Source: wav/music/ and wav/noise/
- Output: wav9/music/ and wav9/noise/

Requirements:
- ffmpeg must be installed and available on PATH.
"""

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SRC = BASE_DIR / "wav"
DST = BASE_DIR / "wav9"

DURATION = 9


def trim_wav(src_path, dst_path):
    """
    Trim a WAV file to the first DURATION seconds using ffmpeg.

    Parameters:
    - src_path: path to source .wav file
    - dst_path: path to write trimmed .wav
    """
    cmd = [
        "ffmpeg",
        "-i", str(src_path),
        "-t", str(DURATION),
        "-acodec", "pcm_s16le",
        "-y",
        str(dst_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ERROR trimming {src_path.name}:")
        print(f"  {proc.stderr}")
        return False

    return True


def process_class(label):
    """
    Trim all .wav files for one class (music or noise).
    """
    src_dir = SRC / label
    dst_dir = DST / label

    if not src_dir.exists():
        print(f"WARNING: Source folder not found: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(src_dir.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {src_dir}")
        return

    print(f"\n[{label.upper()}] {len(wav_files)} file(s) in {src_dir}")

    for i, wav in enumerate(wav_files, 1):
        dst_path = dst_dir / wav.name

        print(f"  [{i}/{len(wav_files)}] {wav.name}")

        if dst_path.exists():
            print(f"    Skipped (already exists)")
            continue

        trim_wav(wav, dst_path)


def main():
    for label in ("music", "noise"):
        process_class(label)

    print("\nDone.")


if __name__ == "__main__":
    main()
