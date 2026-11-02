
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "audio_energy_features.csv"
ERROR_FILE = OUTPUT_DIR / "audio_energy_errors.csv"

SAMPLE_RATE = 22_050
FRAME_LENGTH = 2_048
HOP_LENGTH = 512

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Avoid log10(0) when converting RMS to dBFS.
RMS_EPSILON = 1e-10


def extract_energy_features(audio_path):
    # Load audio using the same configuration as the rhythm pipeline.
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains non-finite samples.")

    if np.max(np.abs(y)) < 1e-8:
        raise ValueError("Audio is silent or nearly silent.")

    # Calculate RMS for each frame.
    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]

    # Exclude invalid values before calculating statistics.
    rms = rms[np.isfinite(rms)]

    if len(rms) == 0:
        raise ValueError("No valid RMS frames were produced.")

    # Feature 1: mean RMS across all frames.
    rms_energy = float(np.mean(rms))

    # Feature 2: standard deviation of frame-level RMS.
    rms_variability = float(np.std(rms, ddof=0))

    # Feature 3: percentile-based dynamic range in dBFS.
    rms_dbfs = 20 * np.log10(np.maximum(rms, RMS_EPSILON))

    p90 = float(np.percentile(rms_dbfs, 90))
    p10 = float(np.percentile(rms_dbfs, 10))
    dynamic_range = p90 - p10

    return {
        "rms_energy": rms_energy,
        "rms_variability": rms_variability,
        "dynamic_range": float(dynamic_range),
    }


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory does not exist: {AUDIO_DIR}"
        )

    audio_files = sorted(
        path
        for path in AUDIO_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No supported audio files found in: {AUDIO_DIR}"
        )

    results = []
    errors = []

    for index, audio_path in enumerate(audio_files, start=1):
        # Assumes each filename stem is the Spotify track ID.
        spotify_track_id = audio_path.stem

        try:
            features = extract_energy_features(audio_path)

            results.append({
                "spotify_track_id": spotify_track_id,
                **features,
            })

            print(
                f"[{index}/{len(audio_files)}] "
                f"OK: {spotify_track_id}"
            )

        except Exception as exc:
            errors.append({
                "spotify_track_id": spotify_track_id,
                "file": audio_path.name,
                "error": str(exc),
            })

            print(
                f"[{index}/{len(audio_files)}] "
                f"ERROR: {audio_path.name}: {exc}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "spotify_track_id",
        "rms_energy",
        "rms_variability",
        "dynamic_range",
    ]

    result_df = pd.DataFrame(results, columns=columns)

    if result_df["spotify_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate Spotify IDs detected in energy results."
        )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            ERROR_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Error log saved to: {ERROR_FILE}")
    elif ERROR_FILE.exists():
        # Remove a stale error log from an earlier run.
        ERROR_FILE.unlink()

    print("\n========== SUMMARY ==========")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Failed extractions: {len(errors)}")
    print(f"Features saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
