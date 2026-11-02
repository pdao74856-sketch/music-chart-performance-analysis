"""
Extract audio repetition and recurrence features using Recurrence Quantification Analysis (RQA).

Features extracted:
- chroma_repetition_determinism: Fraction of recurrence points in diagonal lines (>= 3s)
  derived from pitch-class energy distribution (harmonic repetition).
- chroma_mean_pattern_duration: Mean duration (seconds) of repeating harmonic patterns.
- chroma_longest_pattern_duration: Longest repeating harmonic pattern (seconds).
- mfcc_repetition_determinism: Fraction of recurrence points in diagonal lines (>= 3s)
  derived from standardized MFCCs (timbral/arrangement repetition).
- mfcc_mean_pattern_duration: Mean duration (seconds) of repeating timbral patterns.
- mfcc_longest_pattern_duration: Longest repeating timbral pattern (seconds).
- duration_analyzed_seconds: Total duration analyzed in seconds.

Methodology:
- Feature aggregation: ~1-second average time bins to capture musical phrases while smoothing micro-timing.
- Thresholding: Adaptive 95th percentile per song (Direction A / Constant recurrence density),
  ensuring determinism measures structural clustering rather than absolute loudness/mastering variance.
- Diagonal line detection: Upper triangle with minimum temporal separation of 10s to eliminate local self-similarity.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "audio_repetition_features.csv"
DEFAULT_ERROR_FILE = OUTPUT_DIR / "audio_repetition_errors.csv"

SAMPLE_RATE = 22_050
FRAME_LENGTH = 2_048
HOP_LENGTH = 512

MIN_SEPARATION_SECONDS = 10
MIN_DIAGONAL_SECONDS = 3
SIMILARITY_PERCENTILE = 95.0

N_CHROMA = 12
N_MFCC = 20

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}

COLUMNS = [
    "spotify_track_id",
    "chroma_repetition_determinism",
    "chroma_mean_pattern_duration",
    "chroma_longest_pattern_duration",
    "mfcc_repetition_determinism",
    "mfcc_mean_pattern_duration",
    "mfcc_longest_pattern_duration",
    "duration_analyzed_seconds",
]


def aggregate_to_seconds(features: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Aggregate frame-level features into ~1-second time bins.
    Handles non-integer frames-per-second ratios robustly without reshape errors.
    """
    n_frames = features.shape[1]
    if n_frames == 0:
        return np.empty((features.shape[0], 0), dtype=np.float64)

    frame_times = np.arange(n_frames) * hop_length / sr
    second_bins = np.floor(frame_times).astype(int)
    n_seconds = int(second_bins[-1]) + 1

    aggregated = np.zeros((features.shape[0], n_seconds), dtype=np.float64)
    for second in range(n_seconds):
        mask = second_bins == second
        if np.any(mask):
            aggregated[:, second] = features[:, mask].mean(axis=1)

    return aggregated


def normalize_columns(features: np.ndarray) -> np.ndarray:
    """L2-normalize each time-step feature vector."""
    norms = np.linalg.norm(features, axis=0)
    valid = np.isfinite(norms) & (norms > 1e-8)

    features = features[:, valid]
    norms = norms[valid]

    if features.shape[1] == 0:
        return features

    return features / norms[np.newaxis, :]


def extract_diagonal_runs(upper_matrix: np.ndarray, min_separation: int, min_run: int) -> list[int]:
    """
    Find contiguous True runs on diagonals above the main diagonal (upper triangle).
    Returns a list of integer run lengths >= min_run.
    """
    n = upper_matrix.shape[0]
    run_lengths = []

    for offset in range(min_separation, n):
        diagonal = np.diagonal(upper_matrix, offset=offset)
        if diagonal.size == 0 or not np.any(diagonal):
            continue

        padded = np.concatenate(([False], diagonal, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)

        lengths = ends - starts
        for length in lengths:
            if length >= min_run:
                run_lengths.append(int(length))

    return run_lengths


def compute_rqa_metrics(features: np.ndarray) -> dict[str, float]:
    """
    Calculate RQA metrics (Determinism, Mean Pattern Duration, Longest Pattern Duration)
    for a normalized second-level feature representation using adaptive percentile thresholding.
    """
    n = features.shape[1]
    min_required = MIN_SEPARATION_SECONDS + MIN_DIAGONAL_SECONDS
    if n < min_required:
        raise ValueError(
            f"Insufficient valid frames ({n}s) for recurrence analysis (minimum {min_required}s required)."
        )

    # Cosine similarity via dot product of unit-normalized column vectors
    similarity = features.T @ features
    similarity = np.clip(similarity, -1.0, 1.0)

    # Upper triangle indices excluding local window (|i - j| >= MIN_SEPARATION_SECONDS)
    i_idx, j_idx = np.triu_indices(n, k=MIN_SEPARATION_SECONDS)
    eligible = similarity[i_idx, j_idx]
    eligible = eligible[np.isfinite(eligible)]

    if eligible.size == 0:
        raise ValueError("No eligible time-step pairs in upper triangle.")

    threshold = float(np.percentile(eligible, SIMILARITY_PERCENTILE))

    # Binary upper-triangular recurrence matrix
    upper_recurrence = np.triu(similarity >= threshold, k=MIN_SEPARATION_SECONDS)
    n_recurrence_points = int(upper_recurrence.sum())

    # Extract diagonal runs
    run_lengths = extract_diagonal_runs(
        upper_recurrence,
        min_separation=MIN_SEPARATION_SECONDS,
        min_run=MIN_DIAGONAL_SECONDS,
    )

    points_in_qualifying_lines = int(sum(run_lengths))

    determinism = (
        float(points_in_qualifying_lines / n_recurrence_points)
        if n_recurrence_points > 0
        else 0.0
    )
    mean_duration = float(np.mean(run_lengths)) if run_lengths else 0.0
    longest_duration = float(max(run_lengths)) if run_lengths else 0.0

    return {
        "determinism": determinism,
        "mean_duration": mean_duration,
        "longest_duration": longest_duration,
    }


def extract_repetition_features(audio_path: Path) -> dict[str, float]:
    """Extract Chroma and MFCC RQA repetition features from an audio file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains non-finite samples (NaN or Inf).")

    if np.max(np.abs(y)) < 1e-8:
        raise ValueError("Audio is silent or nearly silent.")

    duration_analyzed = len(y) / sr
    if duration_analyzed < 30.0:
        raise ValueError(
            f"Audio duration ({duration_analyzed:.1f}s) is shorter than minimum 30 seconds."
        )

    # 1. Chroma Representation (Harmonic / Pitch-Class)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA,
    )
    chroma_sec = aggregate_to_seconds(chroma, sr=sr, hop_length=HOP_LENGTH)
    chroma_norm = normalize_columns(chroma_sec)

    chroma_metrics = compute_rqa_metrics(chroma_norm)

    # 2. MFCC Representation (Timbral / Spectral Envelope)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    mfcc_sec = aggregate_to_seconds(mfcc, sr=sr, hop_length=HOP_LENGTH)

    # Standardize each MFCC coefficient across time to prevent dominant lower coefficients
    mean = mfcc_sec.mean(axis=1, keepdims=True)
    std = mfcc_sec.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    mfcc_sec = (mfcc_sec - mean) / std
    mfcc_norm = normalize_columns(mfcc_sec)

    mfcc_metrics = compute_rqa_metrics(mfcc_norm)

    return {
        "chroma_repetition_determinism": chroma_metrics["determinism"],
        "chroma_mean_pattern_duration": chroma_metrics["mean_duration"],
        "chroma_longest_pattern_duration": chroma_metrics["longest_duration"],
        "mfcc_repetition_determinism": mfcc_metrics["determinism"],
        "mfcc_mean_pattern_duration": mfcc_metrics["mean_duration"],
        "mfcc_longest_pattern_duration": mfcc_metrics["longest_duration"],
        "duration_analyzed_seconds": float(duration_analyzed),
    }


def process_single_track(item: tuple[str, Path]) -> tuple[str, dict | None, str | None, str]:
    """Worker function for multiprocessing."""
    song_id, audio_path = item
    try:
        feats = extract_repetition_features(audio_path)
        return song_id, feats, None, audio_path.name
    except Exception as exc:
        return song_id, None, str(exc), audio_path.name


def find_audio_files(audio_dir: Path) -> dict[str, Path]:
    """Index supported audio files mapped by their track ID (filename stem)."""
    index = {}
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def save_checkpoint(output_path: Path, results: list[dict], existing_df: pd.DataFrame | None):
    """Save/update current results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(results, columns=COLUMNS)
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["spotify_track_id"], keep="last")
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Chroma and MFCC RQA repetition features from audio files."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the first N audio files (useful for testing).",
    )
    parser.add_argument(
        "--song-id",
        type=str,
        default=None,
        help="Process a single specific Spotify track ID.",
    )
    parser.add_argument(
        "--pilot-list",
        type=str,
        default=None,
        help="CSV file containing spotify_track_id or song_id column to filter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save output CSV.",
    )
    parser.add_argument(
        "--error-file",
        type=str,
        default=str(DEFAULT_ERROR_FILE),
        help="Path to save error CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed track IDs in output-file if it exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes (default: up to 8).",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    error_path = Path(args.error_file)

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {AUDIO_DIR}")

    audio_index = find_audio_files(AUDIO_DIR)
    if not audio_index:
        raise FileNotFoundError(f"No audio files found in: {AUDIO_DIR}")

    # Determine track list to process
    if args.song_id:
        if args.song_id not in audio_index:
            raise FileNotFoundError(f"Song ID {args.song_id} not found in {AUDIO_DIR}")
        tracks_to_process = [(args.song_id, audio_index[args.song_id])]
    elif args.pilot_list:
        pilot_df = pd.read_csv(args.pilot_list)
        id_col = "spotify_track_id" if "spotify_track_id" in pilot_df.columns else "song_id"
        pilot_ids = pilot_df[id_col].dropna().unique().tolist()
        tracks_to_process = [
            (tid, audio_index[tid]) for tid in pilot_ids if tid in audio_index
        ]
        print(f"Loaded {len(tracks_to_process)} matched tracks from pilot list: {args.pilot_list}")
    else:
        tracks_to_process = sorted(audio_index.items())

    if args.sample:
        tracks_to_process = tracks_to_process[: args.sample]

    # Resume support
    already_done = set()
    existing_df = None
    if args.resume and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if "spotify_track_id" in existing_df.columns:
                already_done = set(existing_df["spotify_track_id"].dropna().astype(str))
                print(f"Resuming: found {len(already_done)} already completed tracks.")
        except Exception as e:
            print(f"Warning: could not read existing output file for resume: {e}")

    # Filter out already done tracks
    remaining_tracks = [item for item in tracks_to_process if item[0] not in already_done]
    total_remaining = len(remaining_tracks)

    print(
        f"Processing repetition features for {total_remaining} tracks "
        f"(Total in target: {len(tracks_to_process)}, Workers: {args.workers})..."
    )

    results = []
    errors = []

    if total_remaining == 0:
        print("All tracks already processed. Nothing to do.")
        return

    checkpoint_freq = 50

    if args.workers <= 1 or total_remaining == 1:
        # Sequential execution
        for idx, item in enumerate(remaining_tracks, start=1):
            song_id, feats, err, fname = process_single_track(item)
            if feats is not None:
                results.append({"spotify_track_id": song_id, **feats})
                print(
                    f"[{idx}/{total_remaining}] OK: {song_id} | "
                    f"Chroma DET: {feats['chroma_repetition_determinism']:.3f} | "
                    f"MFCC DET: {feats['mfcc_repetition_determinism']:.3f}"
                )
            else:
                errors.append({"spotify_track_id": song_id, "file": fname, "error": err})
                print(f"[{idx}/{total_remaining}] ERROR: {song_id} ({fname}): {err}")

            if idx % checkpoint_freq == 0:
                save_checkpoint(output_path, results, existing_df)
    else:
        # Multiprocessing execution
        completed_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_track = {
                executor.submit(process_single_track, item): item for item in remaining_tracks
            }
            for future in as_completed(future_to_track):
                completed_count += 1
                song_id, feats, err, fname = future.result()
                if feats is not None:
                    results.append({"spotify_track_id": song_id, **feats})
                    if completed_count % 10 == 0 or completed_count == total_remaining:
                        print(
                            f"[{completed_count}/{total_remaining}] OK: {song_id} | "
                            f"Chroma DET: {feats['chroma_repetition_determinism']:.3f} | "
                            f"MFCC DET: {feats['mfcc_repetition_determinism']:.3f}"
                        )
                else:
                    errors.append({"spotify_track_id": song_id, "file": fname, "error": err})
                    print(f"[{completed_count}/{total_remaining}] ERROR: {song_id} ({fname}): {err}")

                if completed_count % checkpoint_freq == 0:
                    save_checkpoint(output_path, results, existing_df)
                    print(f"--- Checkpoint saved: {len(results)} new records saved to disk ---")

    # Final save
    save_checkpoint(output_path, results, existing_df)

    if errors:
        error_df = pd.DataFrame(errors)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        if error_path.exists() and args.resume:
            existing_err = pd.read_csv(error_path)
            error_df = pd.concat([existing_err, error_df], ignore_index=True).drop_duplicates(
                subset=["spotify_track_id"], keep="last"
            )
        error_df.to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"Error log saved to: {error_path}")
    elif error_path.exists() and not args.resume:
        error_path.unlink()

    print("\n========== REPETITION EXTRACTION SUMMARY ==========")
    print(f"Total processed in this run: {len(results)}")
    print(f"Errors in this run: {len(errors)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
