"""
Audio Feature Extractor for Inference & Web App (Member 2).
Extracts exactly 17 standardized Librosa acoustic features from full-track audio
(up to 6 minutes / 360 seconds) with production-grade safeguards and defensive typing.

Safeguards:
- 15MB file size limit.
- 360-second decoding cap (duration=360 in librosa.load).
- Defensive float casting for binary 'mode' (1.0 for Major, 0.0 for Minor).
- Full audio preservation for RQA recurrence determinism (min 43s continuous signal).
"""

from io import BytesIO
from pathlib import Path
from typing import Union
import warnings

import librosa
import numpy as np


SAMPLE_RATE = 22_050
HOP_LENGTH = 512
FRAME_LENGTH = 2_048
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_DURATION_SECONDS = 360.0  # 6 minutes

MIN_SEPARATION_SECONDS = 10
MIN_DIAGONAL_SECONDS = 3
SIMILARITY_PERCENTILE = 95.0

# 24 Major & Minor Krumhansl-Schmuckler key profiles
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _get_key_templates():
    templates = []
    metadata = []
    for i, key in enumerate(KEY_NAMES):
        maj_prof = np.roll(MAJOR_PROFILE, i)
        templates.append((maj_prof - np.mean(maj_prof)) / (np.std(maj_prof) + 1e-8))
        metadata.append((key, "major"))

        min_prof = np.roll(MINOR_PROFILE, i)
        templates.append((min_prof - np.mean(min_prof)) / (np.std(min_prof) + 1e-8))
        metadata.append((key, "minor"))
    return np.array(templates), metadata


TEMPLATES, KEY_METADATA = _get_key_templates()


def _estimate_key_mode(chroma: np.ndarray) -> tuple[str, str, float]:
    """Estimate Key, Mode, and Confidence using Krumhansl-Schmuckler profiles."""
    chroma_mean = np.mean(chroma, axis=1)
    norm = np.linalg.norm(chroma_mean)
    if norm < 1e-8 or not np.all(np.isfinite(chroma_mean)):
        return "C", "major", 0.5
    chroma_norm = (chroma_mean - np.mean(chroma_mean)) / (np.std(chroma_mean) + 1e-8)
    correlations = np.dot(TEMPLATES, chroma_norm) / 12.0
    best_idx = int(np.argmax(correlations))
    best_r = float(np.clip(correlations[best_idx], 0.0, 1.0))
    best_key, best_mode = KEY_METADATA[best_idx]
    return best_key, best_mode, best_r


def _estimate_chord_change_rate(y: np.ndarray, sr: int, chroma: np.ndarray) -> float:
    """Estimate chord transitions per minute via smoothed chromagram cosine difference."""
    duration_minutes = (len(y) / sr) / 60.0
    if duration_minutes <= 0.1:
        return 0.0
    smooth_len = max(1, int(0.25 * sr / HOP_LENGTH))
    kernel = np.ones(smooth_len) / smooth_len
    smoothed = np.array([np.convolve(chroma[i], kernel, mode="same") for i in range(12)])
    norms = np.linalg.norm(smoothed, axis=0, keepdims=True) + 1e-8
    normalized = smoothed / norms
    diffs = 1.0 - np.sum(normalized[:, :-1] * normalized[:, 1:], axis=0)
    thresh = float(np.percentile(diffs, 75)) if len(diffs) > 0 else 0.5
    changes = np.sum(diffs > max(thresh, 0.15))
    return float(changes / duration_minutes)


def _aggregate_to_seconds(features: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Aggregate frame-level features into ~1-second time bins."""
    n_frames = features.shape[1]
    if n_frames == 0:
        return np.empty((features.shape[0], 0), dtype=np.float64)
    frame_times = np.arange(n_frames) * hop_length / sr
    second_bins = np.floor(frame_times).astype(int)
    n_seconds = int(second_bins[-1]) + 1
    aggregated = np.zeros((features.shape[0], n_seconds), dtype=np.float64)
    for s in range(n_seconds):
        mask = second_bins == s
        if np.any(mask):
            aggregated[:, s] = features[:, mask].mean(axis=1)
    return aggregated


def _extract_diagonal_runs(upper_matrix: np.ndarray, min_separation: int, min_run: int) -> list[int]:
    """Find contiguous True runs on diagonals above main diagonal."""
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


def _compute_rqa_metrics(features: np.ndarray) -> tuple[float, float, float]:
    """Calculate RQA metrics: (determinism, mean_duration, longest_duration)."""
    n = features.shape[1]
    min_required = MIN_SEPARATION_SECONDS + MIN_DIAGONAL_SECONDS
    if n < min_required:
        return 0.0, 0.0, 0.0

    norms = np.linalg.norm(features, axis=0, keepdims=True)
    valid = (norms > 1e-8) & np.isfinite(norms)
    if not np.all(valid):
        features = np.where(valid, features / (norms + 1e-8), 0.0)
    else:
        features = features / norms

    similarity = np.clip(features.T @ features, -1.0, 1.0)
    i_idx, j_idx = np.triu_indices(n, k=MIN_SEPARATION_SECONDS)
    eligible = similarity[i_idx, j_idx]
    eligible = eligible[np.isfinite(eligible)]
    if eligible.size == 0:
        return 0.0, 0.0, 0.0

    threshold = float(np.percentile(eligible, SIMILARITY_PERCENTILE))
    upper_recurrence = np.triu(similarity >= threshold, k=MIN_SEPARATION_SECONDS)
    n_recurrence_points = int(upper_recurrence.sum())

    runs = _extract_diagonal_runs(upper_recurrence, MIN_SEPARATION_SECONDS, MIN_DIAGONAL_SECONDS)
    pts_in_qualifying = int(sum(runs))
    det = float(pts_in_qualifying / n_recurrence_points) if n_recurrence_points > 0 else 0.0
    mean_dur = float(np.mean(runs)) if len(runs) > 0 else 0.0
    max_dur = float(np.max(runs)) if len(runs) > 0 else 0.0
    return det, mean_dur, max_dur


def _load_audio_safely(audio_source, target_sr: int = 22050, max_duration_sec: float = 360.0):
    """Robust audio loader supporting BytesIO, bytes, and file paths with duration cap."""
    import soundfile as sf
    import scipy.signal as signal

    if isinstance(audio_source, (bytes, bytearray)):
        if len(audio_source) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Kích thước tệp ({len(audio_source)/(1024*1024):.1f}MB) vượt quá giới hạn 15MB.")
        file_obj = BytesIO(audio_source)
    elif isinstance(audio_source, BytesIO):
        size = audio_source.getbuffer().nbytes
        if size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Kích thước tệp ({size/(1024*1024):.1f}MB) vượt quá giới hạn 15MB.")
        file_obj = audio_source
    else:
        path = Path(audio_source)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp âm thanh: {path}")
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Kích thước tệp ({path.stat().st_size/(1024*1024):.1f}MB) vượt quá giới hạn 15MB.")
        file_obj = str(path)

    # Read audio signal
    try:
        y, sr = sf.read(file_obj)
    except Exception:
        # Fallback to librosa if soundfile encounters proprietary containers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_obj, sr=None, mono=False)

    # Convert to mono if stereo/multichannel
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample to target_sr (22050 Hz)
    if sr != target_sr:
        num_samples = int(len(y) * target_sr / sr)
        y = signal.resample(y, num_samples)
        sr = target_sr

    # Duration cap (max 360s / 6 minutes)
    max_samples = int(max_duration_sec * sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    return y.astype(np.float32), sr


def extract_features_for_inference(
    audio_source: Union[str, Path, bytes, BytesIO],
    filename: str = "uploaded_audio.mp3"
) -> dict:
    """
    Extract exact 17 standardized Librosa features for inference.
    Processes full track (up to 360s / 6 minutes) to ensure RQA validity.
    
    Returns:
        dict containing 17 features, mode_numeric (float 0.0 or 1.0), and metadata.
    """
    y, sr = _load_audio_safely(audio_source, target_sr=SAMPLE_RATE, max_duration_sec=MAX_DURATION_SECONDS)

    if len(y) == 0:
        raise ValueError("Tệp âm thanh trống hoặc không đọc được dữ liệu sóng âm.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Dữ liệu âm thanh chứa giá trị bất thường (NaN hoặc Inf).")
    if np.max(np.abs(y)) < 1e-8:
        raise ValueError("Tệp âm thanh bị tắt tiếng hoàn toàn (silent audio).")

    duration_analyzed = len(y) / sr
    duration_minutes = duration_analyzed / 60.0

    # 3. Rhythm & Dynamic Intensity
    tempo_val, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo_val)[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    if len(beat_times) >= 3:
        beat_intervals = np.diff(beat_times)
        beat_variability = float(np.std(beat_intervals, ddof=0))
    else:
        beat_variability = 0.017  # Median fallback if beats too sparse

    tempo_stability = float(1.0 / (1.0 + beat_variability))

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=False)
    onset_rate = float(len(onset_frames) / max(duration_minutes, 0.1))

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    rms = rms[np.isfinite(rms)]
    energy = float(np.mean(rms)) if len(rms) > 0 else 0.1
    rms_variability = float(np.std(rms, ddof=0)) if len(rms) > 0 else 0.05
    loudness_proxy = float(20.0 * np.log10(np.maximum(energy, 1e-10)))

    # 4. Spectral Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    spectral_centroid = float(np.mean(centroid[np.isfinite(centroid)]))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    spectral_bandwidth = float(np.mean(bandwidth[np.isfinite(bandwidth)]))

    # 5. Tonal & Harmony
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    key_str, raw_mode, key_conf = _estimate_key_mode(chroma)
    chord_rate = _estimate_chord_change_rate(y, sr, chroma)

    # SAFE CASTING: Ensure mode is strictly float 1.0 (Major) or 0.0 (Minor)
    mode_val = 1.0 if str(raw_mode).lower() in ["1", "major", "1.0"] else 0.0

    # 6. Repetition RQA (Chroma & MFCC)
    chroma_sec = _aggregate_to_seconds(chroma, sr, HOP_LENGTH)
    chroma_det, chroma_mean_p, chroma_longest_p = _compute_rqa_metrics(chroma_sec)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
    mfcc_sec = _aggregate_to_seconds(mfcc, sr, HOP_LENGTH)
    mfcc_det, mfcc_mean_p, mfcc_longest_p = _compute_rqa_metrics(mfcc_sec)

    return {
        # Core 17 Librosa features
        "tempo": round(tempo, 2),
        "onset_rate": round(onset_rate, 2),
        "tempo_stability": round(tempo_stability, 6),
        "energy": round(energy, 6),
        "rms_variability": round(rms_variability, 6),
        "loudness_proxy": round(loudness_proxy, 2),
        "spectral_centroid": round(spectral_centroid, 2),
        "spectral_bandwidth": round(spectral_bandwidth, 2),
        "chord_change_rate": round(chord_rate, 2),
        "key_confidence": round(key_conf, 4),
        "mode": mode_val,  # 1.0 for Major, 0.0 for Minor (defensive float)
        "chroma_repetition_determinism": round(chroma_det, 4),
        "chroma_mean_pattern_duration": round(chroma_mean_p, 2),
        "chroma_longest_pattern_duration": round(chroma_longest_p, 2),
        "mfcc_repetition_determinism": round(mfcc_det, 4),
        "mfcc_mean_pattern_duration": round(mfcc_mean_p, 2),
        "mfcc_longest_pattern_duration": round(mfcc_longest_p, 2),
        # Metadata
        "duration_analyzed_seconds": round(duration_analyzed, 2),
        "key": key_str,
        "raw_mode": raw_mode,
    }
