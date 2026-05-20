"""
00_convert_mkv_to_wav.py

Purpose:
- Convert .mkv files to .wav (PCM 16-bit, mono) by extracting audio via FFmpeg.
- Music folder: Question3/Music/  ->  Question3/wav/music/
- Noise folder: Question3/noise/  ->  Question3/wav/noise/

Requirements:
- ffmpeg must be installed and available on PATH.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SRC = {
    "music": BASE_DIR / "Music",
    "noise": BASE_DIR / "noise",
}

DST = BASE_DIR / "wav"


def convert_mkv_to_wav(src_path, dst_path):
    """
    Extract audio from an MKV file and save as mono 16-bit WAV.

    Parameters:
    - src_path: path to the .mkv source file
    - dst_path: path to write the .wav output
    """
    cmd = [
        "ffmpeg",
        "-i", str(src_path),
        "-vn",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-y",
        str(dst_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ERROR converting {src_path.name}:")
        print(f"  {proc.stderr}")
        return False

    return True


def process_class(label):
    """
    Convert all .mkv files for one class (music or noise).
    """
    src_dir = SRC[label]
    dst_dir = DST / label

    if not src_dir.exists():
        print(f"WARNING: Source folder not found: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    mkv_files = sorted(src_dir.glob("*.mkv"))

    if not mkv_files:
        print(f"No .mkv files found in {src_dir}")
        return

    print(f"\n[{label.upper()}] {len(mkv_files)} file(s) in {src_dir}")

    for i, mkv in enumerate(mkv_files, 1):
        wav_name = mkv.stem + ".wav"
        dst_path = dst_dir / wav_name

        print(f"  [{i}/{len(mkv_files)}] {mkv.name}  ->  {wav_name}")

        if dst_path.exists():
            print(f"    Skipped (already exists)")
            continue

        convert_mkv_to_wav(mkv, dst_path)


def main():
    for label in ("music", "noise"):
        process_class(label)

    print("\nDone.")


if __name__ == "__main__":
    main()
