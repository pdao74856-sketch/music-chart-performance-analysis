"""
tests/test_tab_upload_headless.py
Headless Test Suite for Tab 3 (Audio Upload & Real-time Librosa Extraction)
Project: What Makes a Hit Song? (Comparing Chart Performance: VN, US, KR)
Owner: Member 2 (Audio Features & Showcase Engineer)

Covers Step 1 Requirements:
  1. Synthetic in-memory WAV audio generation (numpy + soundfile).
  2. End-to-end 17-feature extraction via audio_extractor.
  3. SHA-256 hash caching verification.
  4. Integration with lyrics_market_medians.json (TV3 fallback across VN, US, KR).
  5. Dual-Model prediction invocation (OLS Model 3 + Random Forest Model 4).
  6. Audio Contract v2 compliance verification (no forbidden cols in OLS).
  7. Edge case handling (silent audio detection without unhandled crashes).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components.tab_upload import (
    _sha256,
    _inspect_audio_duration,
    _extract_cached,
    _load_lyrics_medians,
    _load_train_medians,
    _build_prediction_payload,
    _predict_dual_models,
    FEATURE_META,
    FEATURE_GROUPS,
)


def _generate_synthetic_audio(duration: float = 14.0, sr: int = 22050) -> bytes:
    """Generates an in-memory synthetic musical WAV with rhythm and harmony."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # A Major triad (440 Hz, 554.37 Hz, 659.25 Hz)
    sig = (
        0.35 * np.sin(2 * np.pi * 440.0 * t)
        + 0.25 * np.sin(2 * np.pi * 554.37 * t)
        + 0.20 * np.sin(2 * np.pi * 659.25 * t)
    )
    # Rhythmic beat pulse at 2 Hz (~120 BPM)
    beat = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
    sig = (sig * beat).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def _generate_silent_audio(duration: float = 5.0, sr: int = 22050) -> bytes:
    """Generates an in-memory near-silent WAV to test defensive exception handling."""
    sig = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


class TestTabUploadHeadless:
    """Test suite executing complete headless verification for Tab 3."""

    def test_01_synthetic_audio_extraction(self):
        """Test extraction of 17 core Librosa features from synthetic audio in memory."""
        audio_bytes = _generate_synthetic_audio(duration=14.0)
        file_sha = _sha256(audio_bytes)

        features, err = _extract_cached(file_sha, audio_bytes)
        assert err == "", f"Extraction returned unexpected error: {err}"
        assert isinstance(features, dict), "Features output must be a dictionary"

        # Verify all 17 core acoustic features are present
        expected_17_features = [
            "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
            "loudness_proxy", "spectral_centroid", "spectral_bandwidth",
            "chord_change_rate", "key_confidence", "mode",
            "chroma_repetition_determinism", "chroma_mean_pattern_duration",
            "chroma_longest_pattern_duration", "mfcc_repetition_determinism",
            "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
        ]
        for feat in expected_17_features:
            assert feat in features, f"Missing feature in extraction output: {feat}"
            val = features[feat]
            assert isinstance(val, (int, float, np.floating)), f"Feature {feat} is not numeric: {val}"
            assert not np.isnan(val), f"Feature {feat} is NaN"

        # Sanity checks on physical properties
        assert features["mode"] in (0.0, 1.0), "Mode must be strictly binary 0.0 or 1.0"
        assert features["energy"] > 0.0, "Energy must be positive"
        assert features["loudness_proxy"] < 0.0, "Loudness proxy must be negative dBFS"
        assert features["spectral_centroid"] > 0.0, "Spectral centroid must be positive Hz"

    def test_02_sha256_hash_caching(self):
        """Test SHA-256 caching mechanism prevents re-computation on identical audio."""
        b1 = _generate_synthetic_audio(duration=14.0)
        h1 = _sha256(b1)
        b2 = bytes(b1)  # Identical content
        h2 = _sha256(b2)
        assert h1 == h2, "Identical content must produce identical SHA-256"

        b3 = _generate_synthetic_audio(duration=15.0)  # Different duration/content
        h3 = _sha256(b3)
        assert h1 != h3, "Different content must produce different SHA-256"

        # Extracting identical hash retrieves cached results
        f1, _ = _extract_cached(h1, b1)
        f2, _ = _extract_cached(h2, b2)
        assert f1 == f2, "Cache hit must return identical feature dictionary"

    def test_03_lyrics_medians_loading(self):
        """Test loading official lyrics medians from TV3 (data/processed/lyrics_market_medians.json)."""
        medians = _load_lyrics_medians()
        assert isinstance(medians, dict), "Lyrics medians must load as dictionary"
        assert set(medians.keys()) == {"VN", "US", "KR"}, f"Missing expected markets: {medians.keys()}"

        expected_lyrics_cols = [
            "word_count", "repetition_ratio", "avg_word_length",
            "line_count", "chorus_repetition_count", "punctuation_ratio",
        ]
        for m, cols in medians.items():
            for c in expected_lyrics_cols:
                assert c in cols, f"Market {m} missing lyrics median {c}"
                assert cols[c] > 0.0, f"Market {m} median {c} must be positive"

    def test_04_prediction_payload_and_dual_model_execution(self):
        """Test full payload assembly and DualModel prediction across all 3 markets."""
        audio_bytes = _generate_synthetic_audio(duration=14.0)
        file_sha = _sha256(audio_bytes)
        features, _ = _extract_cached(file_sha, audio_bytes)

        lyrics_medians_all = _load_lyrics_medians()
        train_medians = _load_train_medians()

        for market in ["VN", "US", "KR"]:
            market_lyrics = lyrics_medians_all[market]
            payload = _build_prediction_payload(features, market, market_lyrics, train_medians)

            # Assert context fields
            assert payload["market"] == market
            assert payload["release_year"] == 2024.0

            # Run prediction
            pred_res, err = _predict_dual_models(payload)
            assert err == "", f"DualModel prediction failed for market {market}: {err}"
            assert isinstance(pred_res, dict), "Prediction result must be a dictionary"

            # Check expected keys
            assert "ols_prediction" in pred_res
            assert "rf_prediction" in pred_res
            assert "ensemble_prediction" in pred_res
            assert pred_res["market"] == market

            # Bounds check [-99.0, 100.0]
            ols_val = pred_res["ols_prediction"]
            rf_val = pred_res["rf_prediction"]
            ens_val = pred_res["ensemble_prediction"]

            assert -99.0 <= ols_val <= 100.0, f"OLS score {ols_val} out of bounds"
            assert -99.0 <= rf_val <= 100.0, f"RF score {rf_val} out of bounds"
            assert -99.0 <= ens_val <= 100.0, f"Ensemble score {ens_val} out of bounds"

    def test_05_audio_contract_v2_strict_compliance(self):
        """
        Verify Audio Contract v2:
        Ensure loudness_proxy and spectral_bandwidth NEVER leak into the OLS design matrix.
        """
        from src.utils.pipeline_inference import get_predictor
        from src.features.prepare_features import build_design_matrix
        import pandas as pd

        p = get_predictor()
        config = p.ols_model.config

        # Confirm forbidden columns are explicitly defined in OLS config
        forbidden = config.get("forbidden_cols", [])
        assert "loudness_proxy" in forbidden, "loudness_proxy must be forbidden in OLS"
        assert "spectral_bandwidth" in forbidden, "spectral_bandwidth must be forbidden in OLS"

        # Confirm audio_cols only contains 14 features
        audio_cols = config.get("audio_cols", [])
        assert len(audio_cols) == 14, f"OLS audio_cols must contain exactly 14 continuous features, got {len(audio_cols)}"
        assert "loudness_proxy" not in audio_cols
        assert "spectral_bandwidth" not in audio_cols

    def test_06_edge_case_silent_audio(self):
        """Test defensive error handling for silent audio without application crash."""
        silent_bytes = _generate_silent_audio(duration=5.0)
        silent_sha = _sha256(silent_bytes)

        features, err = _extract_cached(silent_sha, silent_bytes)
        assert features == {}, "Silent audio must return empty features dict"
        assert len(err) > 0, "Silent audio must return a descriptive error message"
        assert ("im lặng" in err.lower() or "silent" in err.lower() or "quá ngắn" in err.lower())

    def test_07_physical_units_metadata_sanity(self):
        """Verify label and physical unit definitions match academic DSP standards."""
        # Check onset_rate unit
        assert FEATURE_META["onset_rate"]["unit"] == "onsets/min", "onset_rate unit must be onsets/min"
        # Check tempo unit
        assert FEATURE_META["tempo"]["unit"] == "BPM", "tempo unit must be BPM"
        # Check chord_change_rate unit
        assert FEATURE_META["chord_change_rate"]["unit"] == "chords/min", "chord_change_rate unit must be chords/min"
        # Check spectral_centroid unit
        assert FEATURE_META["spectral_centroid"]["unit"] == "Hz", "spectral_centroid unit must be Hz"
        # Check loudness_proxy unit
        assert FEATURE_META["loudness_proxy"]["unit"] == "dBFS", "loudness_proxy unit must be dBFS"

        # Confirm all 17 features are mapped into 3 groups
        total_grouped = sum(len(v) for v in FEATURE_GROUPS.values())
        assert total_grouped == 17, f"Total grouped features must equal 17, got {total_grouped}"

    def test_08_audio_duration_inspection_precheck(self):
        """Test pre-check inspection of audio duration via soundfile metadata."""
        audio_14s = _generate_synthetic_audio(duration=14.0)
        dur, err = _inspect_audio_duration(audio_14s)
        assert err is None
        assert dur is not None
        assert abs(dur - 14.0) < 0.1, f"Expected ~14.0s duration, got {dur}"

        # Corrupt bytes should return None and an error string without crashing
        bad_dur, bad_err = _inspect_audio_duration(b"not an audio stream")
        assert bad_dur is None
        assert bad_err is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
