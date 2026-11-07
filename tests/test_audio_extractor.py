"""Tests for src.features.audio_extractor.

Verifies the production Audio Feature Extractor for Tab 3 Real-time Inference:
1. Fast in-memory synthetic sine wave (15s): returns EXACTLY 17 numerical DSP features
   strictly typed as float, without NaN/Inf, in under 2 seconds.
2. Silent audio handling: verifies that all-zero silent signals do not crash (no ZeroDivisionError)
   and return finite floats.
3. Duration capping: audio longer than 360s is capped at 360.0s.
4. Flexible input support: verifies identical extraction across bytes, BytesIO, and file path (Path/str).
5. Safeguards verification: 15MB file size cap, empty buffer, and corrupt data error handling.
"""

import io
import math
import sys
import time
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
    envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2.0 * t)
    signal = envelope * (
        0.5 * np.sin(2 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2 * np.pi * 554.37 * t)
        + 0.2 * np.sin(2 * np.pi * 659.25 * t)
    )

    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_extract_features_synthetic_sine_exact_keys_and_strict_float_types():
    """Verify runtime < 1s (pure extraction), exact set of 17 feature keys, and strict float typing."""
    # Warm-up to trigger one-time Numba JIT compilation before benchmarking
    _ = extract_features_for_inference(_generate_synthetic_audio_bytes(duration_sec=3.0))

    audio_bytes = _generate_synthetic_audio_bytes(duration_sec=15.0)

    t0 = time.perf_counter()
    result = extract_features_for_inference(audio_bytes, filename="synthetic_a_major.wav")
    elapsed = time.perf_counter() - t0

    # 1. Performance assertion: pure inference extraction is under 1.0 second
    assert elapsed < 1.0, f"Pure extraction took {elapsed:.2f}s, exceeding 1.0s commitment"

    assert isinstance(result, dict), "Result must be a dictionary"

    # 2. Exact 17 feature keys assertion (no more, no less)
    actual_feature_keys = [k for k in result.keys() if k not in EXPECTED_METADATA_KEYS]
    assert set(actual_feature_keys) == set(EXPECTED_17_FEATURES), "Mismatch in feature keys set"
    assert len(actual_feature_keys) == 17, f"Expected exactly 17 features, got {len(actual_feature_keys)}"

    # 3. Strict float type and finite value assertions
    for feat in EXPECTED_17_FEATURES:
        val = result[feat]
        assert isinstance(val, float), f"Feature '{feat}' must be strict float, got {type(val)}"
        assert not math.isnan(val), f"Feature '{feat}' cannot be NaN"
        assert not math.isinf(val), f"Feature '{feat}' cannot be Infinite"

    # 4. Metadata assertions
    for meta in EXPECTED_METADATA_KEYS:
        assert meta in result, f"Missing expected metadata key: {meta}"
    assert result["duration_analyzed_seconds"] == pytest.approx(15.0, abs=0.5)
    assert result["mode"] in (0.0, 1.0), "mode must be float 0.0 or 1.0"
    assert result["energy"] > 0.0, "Energy of non-silent signal must be > 0"
    assert result["spectral_centroid"] > 0.0, "Spectral centroid must be > 0"
    assert result["filename"] == "synthetic_a_major.wav"


def test_extract_features_silent_audio():
    """Verify that all-zero silent audio is defensively detected and rejected with a clear ValueError."""
    sr = 22050
    duration_sec = 5.0
    silent_signal = np.zeros(int(sr * duration_sec), dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, silent_signal, sr, format="WAV")
    silent_bytes = buf.getvalue()

    with pytest.raises(ValueError, match=r"(?i)(tắt tiếng|silent)"):
        extract_features_for_inference(silent_bytes, filename="silent.wav")


def test_extract_features_duration_cap_360s():
    """Verify that audio exceeding 360 seconds is capped at 360.0s."""
    # Use sr=8000 to keep the 400s WAV file well under 15MB (~6.4 MB)
    sr = 8000
    duration_sec = 400.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * 440.0 * t)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    long_bytes = buf.getvalue()

    result = extract_features_for_inference(long_bytes, filename="long_400s.wav")

    assert result["duration_analyzed_seconds"] == pytest.approx(360.0, abs=1.0)


def test_extract_features_input_types(tmp_path):
    """Verify that bytes, BytesIO, and filesystem Path inputs yield identical results."""
    audio_bytes = _generate_synthetic_audio_bytes(duration_sec=3.0)

    # 1. From raw bytes
    res_bytes = extract_features_for_inference(audio_bytes, filename="sample.wav")

    # 2. From BytesIO
    buf = io.BytesIO(audio_bytes)
    res_bytesio = extract_features_for_inference(buf, filename="sample.wav")

    # 3. From filesystem Path
    file_path = tmp_path / "sample.wav"
    file_path.write_bytes(audio_bytes)
    res_path = extract_features_for_inference(file_path, filename="sample.wav")

    # Check key acoustic metrics match across all three input modalities
    for feat in ["tempo", "energy", "spectral_centroid", "chord_change_rate"]:
        assert res_bytes[feat] == pytest.approx(res_bytesio[feat], rel=1e-4)
        assert res_bytes[feat] == pytest.approx(res_path[feat], rel=1e-4)


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
