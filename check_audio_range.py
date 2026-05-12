from pathlib import Path
import soundfile as sf
import numpy as np

folders = [Path("wav10/music"), Path("wav10/noise")]

for folder in folders:
    print(f"\nFolder: {folder}")

    for wav_file in sorted(folder.glob("*.wav")):
        x, sr = sf.read(wav_file)

        if x.ndim > 1:
            x = x.mean(axis=1)

        print(
            wav_file.name,
            "sr =", sr,
            "min =", np.min(x),
            "max =", np.max(x),
            "dtype =", x.dtype
        )