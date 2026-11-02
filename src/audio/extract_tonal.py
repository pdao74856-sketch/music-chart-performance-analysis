"""
Extract tonal and harmony audio features: Key, Mode, and Chord Change Rate.

Features extracted:
- key: Estimated musical tonic pitch class (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
- mode: Musical scale modality ('major' or 'minor').
- key_confidence: Pearson correlation coefficient with the best matching K-S key profile.
- chord_change_rate: Number of valid harmonic/chord transitions per minute of analyzed audio.

Methodology:
- Key & Mode: Krumhansl-Schmuckler (K-S) key-finding algorithm.
  Extracts Constant-Q Chroma (chroma_cqt) to capture pitch classes with logarithmic semitone resolution,
  aggregates energy over time, and computes Pearson correlation against 24 standard K-S key profiles.
- Chord Change Rate: Beat-synchronous harmonic progression.
  Synchronizes chroma frames to detected beats, projects each beat onto 24 standard triad chord templates
  (12 Major + 12 Minor triads), applies a median smoothing filter to remove transient passing notes,
  and counts chord transitions normalized by duration in minutes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "audio_tonal_features.csv"
DEFAULT_ERROR_FILE = OUTPUT_DIR / "audio_tonal_errors.csv"

SAMPLE_RATE = 22_050
HOP_LENGTH = 512

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler standard key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Construct 24 reference key profiles (12 Major, 12 Minor)
KEY_PROFILES = []
KEY_METADATA = []
for i, root in enumerate(PITCH_CLASSES):
    KEY_PROFILES.append(np.roll(MAJOR_PROFILE, i))
    KEY_METADATA.append((root, "major"))
    KEY_PROFILES.append(np.roll(MINOR_PROFILE, i))
    KEY_METADATA.append((root, "minor"))
KEY_PROFILES = np.array(KEY_PROFILES)

# Construct 24 triad chord templates (12 Major, 12 Minor)
CHORD_TEMPLATES = []
for root in range(12):
    # Major triad: root, major third (+4), perfect fifth (+7)
    maj = np.zeros(12)
    maj[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
    maj /= np.linalg.norm(maj)
    CHORD_TEMPLATES.append(maj)

    # Minor triad: root, minor third (+3), perfect fifth (+7)
    min_ = np.zeros(12)
    min_[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
    min_ /= np.linalg.norm(min_)
    CHORD_TEMPLATES.append(min_)
CHORD_TEMPLATES = np.array(CHORD_TEMPLATES)

COLUMNS = [
    "spotify_track_id",
    "key",
    "mode",
    "key_confidence",
    "chord_change_rate",
]


def estimate_key_mode(chroma: np.ndarray) -> tuple[str, str, float]:
    """
    Estimate Key and Mode using Krumhansl-Schmuckler Pearson correlation.
    Returns: (key_name, mode_name, confidence_r)
    """
    chroma_mean = np.mean(chroma, axis=1)
    if np.all(chroma_mean == 0):
        raise ValueError("Chroma energy is zero across all pitch classes.")

    corrs = []
    for profile in KEY_PROFILES:
        r, _ = scipy.stats.pearsonr(chroma_mean, profile)
        corrs.append(r if np.isfinite(r) else -1.0)

    best_idx = int(np.argmax(corrs))
    best_key, best_mode = KEY_METADATA[best_idx]
    best_r = float(corrs[best_idx])

    return best_key, best_mode, best_r


def estimate_chord_change_rate(y: np.ndarray, sr: int, chroma: np.ndarray) -> float:
    """
    Estimate chord change rate per minute using beat-synchronous triad template matching.
    """
    duration_min = (len(y) / sr) / 60.0
    if duration_min < 0.2:
        raise ValueError("Audio duration too short for chord progression analysis.")

    # Beat tracking
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)

    if len(beat_frames) < 8:
        # Fallback to uniform 0.5s intervals if beat tracking fails to detect sufficient beats
        hop_sample = int(0.5 * sr / HOP_LENGTH)
        beat_frames = np.arange(0, chroma.shape[1], max(1, hop_sample))

    # Beat-synchronous chroma (median aggregation across beat frames)
    chroma_beat = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # Normalize beat columns
    norms = np.linalg.norm(chroma_beat, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    chroma_beat = chroma_beat / norms

    # Template matching: cosine similarities with 24 triads
    sims = CHORD_TEMPLATES @ chroma_beat  # (24, n_beats)
    chord_sequence = np.argmax(sims, axis=0)

    # Median filtering with kernel size 3 to remove 1-beat glitches / transient passing notes
    if len(chord_sequence) >= 3:
        smoothed_chords = scipy.signal.medfilt(chord_sequence, kernel_size=3)
    else:
        smoothed_chords = chord_sequence

    # Count chord transitions
    transitions = int(np.sum(smoothed_chords[1:] != smoothed_chords[:-1]))
    chord_change_rate = float(transitions / duration_min)

    return chord_change_rate


def extract_tonal_features(audio_path: Path) -> dict:
    """Extract key, mode, key_confidence, and chord_change_rate from audio file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains non-finite samples (NaN or Inf).")

    if np.max(np.abs(y)) < 1e-8:
        raise ValueError("Audio is silent or nearly silent.")

    duration_seconds = len(y) / sr
    if duration_seconds < 15.0:
        raise ValueError(f"Audio duration ({duration_seconds:.1f}s) shorter than 15s.")

    # Extract Constant-Q Chroma for high semitone precision
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    except Exception:
        # Robust fallback to chroma_stft if CQT encounters signal issues
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    key, mode, conf = estimate_key_mode(chroma)
    chord_rate = estimate_chord_change_rate(y, sr, chroma)

    return {
        "key": key,
        "mode": mode,
        "key_confidence": conf,
        "chord_change_rate": chord_rate,
    }


def process_single_track(item: tuple[str, Path]) -> tuple[str, dict | None, str | None, str]:
    """Worker function for multiprocessing."""
    song_id, audio_path = item
    try:
        feats = extract_tonal_features(audio_path)
        return song_id, feats, None, audio_path.name
    except Exception as exc:
        return song_id, None, str(exc), audio_path.name


def find_audio_files(audio_dir: Path) -> dict[str, Path]:
    """Index supported audio files mapped by their track ID."""
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
        description="Extract Key, Mode, and Chord Change Rate from audio files."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N audio files.",
    )
    parser.add_argument(
        "--song-id",
        type=str,
        default=None,
        help="Process single Spotify track ID.",
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

    remaining_tracks = [item for item in tracks_to_process if item[0] not in already_done]
    total_remaining = len(remaining_tracks)

    print(
        f"Processing tonal features for {total_remaining} tracks "
        f"(Total in target: {len(tracks_to_process)}, Workers: {args.workers})..."
    )

    results = []
    errors = []

    if total_remaining == 0:
        print("All tracks already processed. Nothing to do.")
        return

    checkpoint_freq = 50

    if args.workers <= 1 or total_remaining == 1:
        for idx, item in enumerate(remaining_tracks, start=1):
            song_id, feats, err, fname = process_single_track(item)
            if feats is not None:
                results.append({"spotify_track_id": song_id, **feats})
                print(
                    f"[{idx}/{total_remaining}] OK: {song_id} | "
                    f"Key: {feats['key']} {feats['mode']} (r={feats['key_confidence']:.2f}) | "
                    f"Chord Rate: {feats['chord_change_rate']:.1f}/min"
                )
            else:
                errors.append({"spotify_track_id": song_id, "file": fname, "error": err})
                print(f"[{idx}/{total_remaining}] ERROR: {song_id} ({fname}): {err}")

            if idx % checkpoint_freq == 0:
                save_checkpoint(output_path, results, existing_df)
    else:
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
                            f"Key: {feats['key']} {feats['mode']} (r={feats['key_confidence']:.2f}) | "
                            f"Chord Rate: {feats['chord_change_rate']:.1f}/min"
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

    print("\n========== TONAL EXTRACTION SUMMARY ==========")
    print(f"Total processed in this run: {len(results)}")
    print(f"Errors in this run: {len(errors)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
