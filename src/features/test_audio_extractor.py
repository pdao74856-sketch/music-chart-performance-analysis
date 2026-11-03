"""
Unit test for audio_extractor module (Member 2).
Verifies:
1. Feature extraction returns all 17 core Librosa features.
2. Mode is strictly typed as float (1.0 or 0.0).
3. Loudness proxy and tempo stability bounds.
4. 15MB file size safeguard.
"""

from io import BytesIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_extractor import (
    extract_features_for_inference,
    MAX_FILE_SIZE_BYTES,
)


def _generate_synthetic_wav(duration: float = 15.0, sr: int = 22050) -> BytesIO:
    """Generate in-memory WAV audio buffer."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    buf = BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf


def test_extract_features_schema_and_types():
    buf = _generate_synthetic_wav(duration=15.0)
    feats = extract_features_for_inference(buf)

    # Required 17 Librosa features
    required_features = [
        "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
        "loudness_proxy", "spectral_centroid", "spectral_bandwidth",
        "chord_change_rate", "key_confidence", "mode",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration"
    ]
    for feat in required_features:
        assert feat in feats, f"Missing feature: {feat}"

    # Verify mode is strictly float (1.0 or 0.0)
    assert isinstance(feats["mode"], float), f"Expected mode to be float, got {type(feats['mode'])}"
    assert feats["mode"] in [0.0, 1.0], f"Invalid mode value: {feats['mode']}"

    # Verify bounds
    assert feats["loudness_proxy"] <= 0.0, "loudness_proxy must be <= 0 dBFS"
    assert 0.0 < feats["tempo_stability"] <= 1.0, "tempo_stability must be in (0, 1]"
    assert 0.0 <= feats["key_confidence"] <= 1.0, "key_confidence must be in [0, 1]"


def test_file_size_limit_safeguard():
    # 16MB oversized dummy buffer
    oversized = bytes(MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValueError, match="vượt quá giới hạn"):
        extract_features_for_inference(oversized)


if __name__ == "__main__":
    print("Running audio_extractor unit tests...")
    test_extract_features_schema_and_types()
    test_file_size_limit_safeguard()
    print("All unit tests passed successfully!")
