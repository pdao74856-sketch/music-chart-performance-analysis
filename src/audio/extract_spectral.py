
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "audio_spectral_features.csv"
ERROR_FILE = OUTPUT_DIR / "audio_spectral_errors.csv"

SAMPLE_RATE = 22_050
FRAME_LENGTH = 2_048
HOP_LENGTH = 512

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a"
}


# ==========================================
# 2. EXTRACT SPECTRAL FEATURES
# ==========================================

def extract_spectral_features(audio_path):
    # Load audio consistently with previous scripts.
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

    # Feature 1: spectral centroid.
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]

    # Feature 2: spectral bandwidth.
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]

    # Keep only frames where both features are valid.
    valid = np.isfinite(centroid) & np.isfinite(bandwidth)

    centroid = centroid[valid]
    bandwidth = bandwidth[valid]

    if len(centroid) == 0:
        raise ValueError("No valid spectral frames were produced.")

    # One aggregate value per song.
    spectral_centroid = float(np.mean(centroid))
    spectral_bandwidth = float(np.mean(bandwidth))

    if not np.isfinite(spectral_centroid):
        raise ValueError("Invalid spectral centroid result.")

    if not np.isfinite(spectral_bandwidth):
        raise ValueError("Invalid spectral bandwidth result.")

    if spectral_centroid < 0 or spectral_bandwidth < 0:
        raise ValueError("Spectral features cannot be negative.")

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
    }


# ==========================================
# 3. PROCESS ALL AUDIO FILES
# ==========================================

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
        # Assumes filename stem is the Spotify track ID.
        spotify_track_id = audio_path.stem

        try:
            features = extract_spectral_features(audio_path)

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

    # ==========================================
    # 4. SAVE RESULTS AND ERROR LOG
    # ==========================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "spotify_track_id",
        "spectral_centroid",
        "spectral_bandwidth",
    ]

    result_df = pd.DataFrame(results, columns=columns)

    if result_df["spotify_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate Spotify IDs detected in spectral results."
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
        # Remove stale errors from a previous run.
        ERROR_FILE.unlink()

    print("\n========== SUMMARY ==========")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Failed extractions: {len(errors)}")
    print(f"Features saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
