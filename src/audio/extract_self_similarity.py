
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "audio_self_similarity_features.csv"
ERROR_FILE = OUTPUT_DIR / "audio_self_similarity_errors.csv"

SAMPLE_RATE = 22_050
FRAME_LENGTH = 2_048
HOP_LENGTH = 512
SEGMENT_SECONDS = 10

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a"
}


# ==========================================
# 2. EXTRACT SELF-SIMILARITY
# ==========================================

def extract_self_similarity(audio_path):
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

    samples_per_segment = int(SEGMENT_SECONDS * sr)
    n_complete_segments = len(y) // samples_per_segment

    if n_complete_segments < 2:
        raise ValueError(
            "Audio has fewer than two complete 10-second segments."
        )

    # Keep only complete, non-overlapping segments.
    usable_length = n_complete_segments * samples_per_segment
    y = y[:usable_length]

    segments = y.reshape(
        n_complete_segments,
        samples_per_segment,
    )

    segment_vectors = []

    for segment in segments:
        chroma = librosa.feature.chroma_stft(
            y=segment,
            sr=sr,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            n_chroma=12,
        )

        # One 12-dimensional vector per segment.
        vector = np.mean(chroma, axis=1)

        if not np.all(np.isfinite(vector)):
            raise ValueError("A segment produced invalid chroma values.")

        # A near-zero vector cannot provide a reliable cosine direction.
        if np.linalg.norm(vector) < 1e-10:
            raise ValueError(
                "A segment has insufficient chroma energy."
            )

        segment_vectors.append(vector)

    vectors = np.asarray(segment_vectors, dtype=float)

    # Normalize each segment vector to unit L2 length.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Pairwise cosine similarities.
    similarity_matrix = cosine_similarity(normalized_vectors)

    # Select only unique pairs i < j.
    upper_triangle = np.triu_indices(
        n=n_complete_segments,
        k=1,
    )

    pair_similarities = similarity_matrix[upper_triangle]

    if len(pair_similarities) == 0:
        raise ValueError("No valid non-overlapping segment pairs.")

    self_similarity = float(np.mean(pair_similarities))

    return {
        "self_similarity": self_similarity,
    }


# ==========================================
# 3. PROCESS AUDIO FILES
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
        spotify_track_id = audio_path.stem

        try:
            features = extract_self_similarity(audio_path)

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
        columns=["spotify_track_id", "self_similarity"],
    )

    if result_df["spotify_track_id"].duplicated().any():
        raise ValueError("Duplicate Spotify IDs in self-similarity results.")

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
        ERROR_FILE.unlink()

    print("\n========== SUMMARY ==========")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Failed extractions: {len(errors)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
