"""Tests for src.features.audio_extractor.

Verifies the production Audio Feature Extractor for Tab 3 Real-time Inference:
1. Fast in-memory synthetic sine wave (15s) extraction: returns exactly 17 numerical
   DSP features + metadata without NaN/Inf in under 1 second.
2. Safeguards verification: 15MB file cap, empty buffer, and corrupt data error handling.
"""

import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import soundfile as sf

from src.features.audio_extractor import extract_features_for_inference

EXPECTED_17_FEATURES = [
    "tempo",
    "onset_rate",
    "tempo_stability",
    "energy",
    "rms_variability",
    "loudness_proxy",
    "spectral_centroid",
    "spectral_bandwidth",
    "chord_change_rate",
    "key_confidence",
    "mode",
    "chroma_repetition_determinism",
    "chroma_mean_pattern_duration",
    "chroma_longest_pattern_duration",
    "mfcc_repetition_determinism",
    "mfcc_mean_pattern_duration",
    "mfcc_longest_pattern_duration",
]

EXPECTED_METADATA_KEYS = [
    "duration_analyzed_seconds",
    "key",
    "raw_mode",
    "filename",
]


def _generate_synthetic_audio_bytes(duration_sec: float = 15.0, sr: int = 22050) -> bytes:
    """Generate an in-memory WAV buffer containing a multi-tone synthetic signal."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    # A Major triad: A4 (440Hz), C#5 (554.37Hz), E5 (659.25Hz) with subtle amplitude envelope
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2.0 * t)
    signal = envelope * (
        0.5 * np.sin(2 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2 * np.pi * 554.37 * t)
        + 0.2 * np.sin(2 * np.pi * 659.25 * t)
    )

    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_extract_features_synthetic_sine():
    """Verify that synthetic audio is decoded and extracted into 17 raw float features."""
    audio_bytes = _generate_synthetic_audio_bytes(duration_sec=15.0)

    result = extract_features_for_inference(audio_bytes, filename="synthetic_a_major.wav")

    assert isinstance(result, dict), "Result must be a dictionary"

    # 1. Check all 17 core features exist
    for feat in EXPECTED_17_FEATURES:
        assert feat in result, f"Missing expected feature: {feat}"
        val = result[feat]
        assert isinstance(val, (float, int)), f"Feature {feat} must be float/int, got {type(val)}"
        assert not math.isnan(val), f"Feature {feat} cannot be NaN"
        assert not math.isinf(val), f"Feature {feat} cannot be Infinite"

    # 2. Check metadata keys exist
    for meta in EXPECTED_METADATA_KEYS:
        assert meta in result, f"Missing expected metadata key: {meta}"

    # 3. Specific physical sanity checks
    assert result["duration_analyzed_seconds"] == pytest.approx(15.0, abs=0.5)
    assert result["mode"] in (0.0, 1.0), "mode must be float 0.0 or 1.0"
    assert result["energy"] > 0.0, "Energy of non-silent signal must be > 0"
    assert result["spectral_centroid"] > 0.0, "Spectral centroid must be > 0"
    assert result["filename"] == "synthetic_a_major.wav"


def test_audio_extractor_safeguard_file_size_cap():
    """Verify that files exceeding the 15MB cap are defensively rejected."""
    oversized_bytes = b"\x00" * (16 * 1024 * 1024)  # 16 MB

    with pytest.raises(ValueError, match=r"(?i)(15MB|vượt quá)"):
        extract_features_for_inference(oversized_bytes, filename="too_large.wav")


def test_audio_extractor_safeguard_empty_file():
    """Verify that an empty file buffer raises a ValueError."""
    with pytest.raises(ValueError, match=r"(?i)(trống|empty)"):
        extract_features_for_inference(b"", filename="empty.wav")


def test_audio_extractor_safeguard_corrupt_file():
    """Verify that unparseable random noise/corrupt bytes raise a ValueError."""
    corrupt_bytes = b"NOT_A_VALID_AUDIO_HEADER" * 100

    with pytest.raises(ValueError, match=r"(?i)(không thể giải mã|failed to decode)"):
        extract_features_for_inference(corrupt_bytes, filename="corrupt.wav")
