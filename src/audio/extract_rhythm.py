
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "audio_rhythm_features.csv"

SAMPLE_RATE = 22_050
HOP_LENGTH = 512

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def scalar(value):
    """Convert a scalar or one-element array into a Python float."""
    arr = np.asarray(value).reshape(-1)

    if arr.size == 0:
        return np.nan

    result = float(arr[0])
    return result if np.isfinite(result) else np.nan


def extract_rhythm_features(audio_path):
    """Extract tempo, onset rate, and beat interval variability."""

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    duration_minutes = len(y) / sr / 60

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains non-finite samples.")

    if np.max(np.abs(y)) < 1e-8:
        raise ValueError("Audio is silent or nearly silent.")

    # 1. Tempo and beat times
    tempo, beat_times = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="time",
    )

    tempo_bpm = scalar(tempo)

    beat_times = np.asarray(beat_times, dtype=float)
    beat_times = beat_times[np.isfinite(beat_times)]
    beat_times = np.sort(beat_times)

    if len(beat_times) >= 3:
        beat_intervals = np.diff(beat_times)
        beat_interval_variability = float(
            np.std(beat_intervals, ddof=0)
        )
    else:
        beat_interval_variability = np.nan

    # 2. Onset rate
    onset_times = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="time",
        backtrack=False,
    )

    onset_rate = len(onset_times) / duration_minutes

    return {
        "tempo_bpm": tempo_bpm,
        "onset_rate": float(onset_rate),
        "beat_interval_variability": beat_interval_variability,
    }


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    audio_files = sorted(
        path
        for path in AUDIO_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No supported audio files found in {AUDIO_DIR}"
        )

    results = []
    errors = []

    for index, audio_path in enumerate(audio_files, start=1):
        spotify_track_id = audio_path.stem

        try:
            features = extract_rhythm_features(audio_path)

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

    result_df = pd.DataFrame(
        results,
        columns=[
            "spotify_track_id",
            "tempo_bpm",
            "onset_rate",
            "beat_interval_variability",
        ],
    )

    if result_df["spotify_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate Spotify IDs detected in audio results."
        )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if errors:
        error_file = OUTPUT_DIR / "audio_rhythm_errors.csv"
        pd.DataFrame(errors).to_csv(
            error_file,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Error log saved to: {error_file}")

    print(f"\nAudio files found: {len(audio_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Failed extractions: {len(errors)}")
    print(f"Features saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
