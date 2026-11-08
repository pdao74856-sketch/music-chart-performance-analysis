"""TV3 what-if simulator using the project's trained inference pipelines."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.pipeline_inference import DualModelPredictor


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
MARKETS = ("US", "VN", "KR")


@lru_cache(maxsize=1)
def _load_predictor() -> DualModelPredictor:
    """Load the repository's serialized models once per app process."""
    return DualModelPredictor(models_dir=MODEL_DIR)


def _defaults(predictor: DualModelPredictor) -> dict[str, Any]:
    medians = predictor.metadata.get("train_medians", {})
    return {
        "tempo": float(medians.get("tempo", 121.1)),
        "tempo_stability": float(medians.get("tempo_stability", 0.98)),
        "spectral_centroid": float(medians.get("spectral_centroid", 2225.0)),
        "repetition_ratio": float(medians.get("repetition_ratio", 0.60)),
        "word_count": float(medians.get("word_count", 350.0)),
        "energy": float(medians.get("energy", 0.238)),
        "loudness_proxy": float(
            medians.get(
                "loudness_proxy",
                20.0 * math.log10(float(medians.get("energy", 0.238))),
            )
        ),
    }


def _predict(predictor: DualModelPredictor, raw_row: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Return finite raw model outputs and errors without invoking heuristic fallback."""
    try:
        complete_row = predictor._fill_defaults(raw_row)
    except Exception as exc:
        return {}, [f"Prediction input preparation failed: {exc}"]

    predictions: dict[str, float] = {}
    errors: list[str] = []
    if predictor.ols_model is None:
        errors.append("Required model artifact unavailable: OLS Model 3 (models/mlr_pipeline.pkl)")
    if predictor.rf_model is None:
        errors.append("Required model artifact unavailable: Random Forest Model 4 (models/rf_pipeline.pkl)")
    for label, model, method in (
        ("OLS Model 3", predictor.ols_model, predictor.predict_ols),
        ("Random Forest Model 4", predictor.rf_model, predictor.predict_rf),
    ):
        if model is None:
            continue
        try:
            value = method(complete_row)
            if value is None or not math.isfinite(float(value)):
                errors.append(f"{label} returned no valid finite prediction.")
            else:
                predictions[label] = float(value)
        except Exception as exc:
            errors.append(f"{label} prediction failed: {exc}")
    return predictions, errors


def render() -> None:
    """Render TV3. TV4 can integrate with ``from app.components import tab_simulator; tab_simulator.render()``."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is required to render the TV3 simulator.") from exc
    st.header("What-if Simulator")
    st.caption("Adjust the track features to compare trained OLS Model 3 and Random Forest Model 4 predictions.")

    try:
        predictor = _load_predictor()
    except Exception as exc:
        st.error(f"Could not initialize the model inference pipeline: {exc}")
        return

    defaults = _defaults(predictor)
    with st.form("what_if_simulator"):
        market = st.selectbox("Market", MARKETS, index=0)
        first, second = st.columns(2)
        with first:
            loudness = st.slider("Loudness (dBFS)", -25.0, -6.0, min(-6.0, max(-25.0, defaults["loudness_proxy"])), 0.01)
            tempo = st.slider("Tempo (BPM)", 60.0, 180.0, min(180.0, max(60.0, defaults["tempo"])), 1.0)
            centroid = st.slider("Spectral Centroid (Hz)", 1000.0, 4000.0, min(4000.0, max(1000.0, defaults["spectral_centroid"])), 10.0)
        with second:
            stability = st.slider("Tempo Stability", 0.80, 1.00, min(1.0, max(0.80, defaults["tempo_stability"])), 0.001)
            repetition = st.slider("Repetition Ratio", 0.2, 0.8, min(0.8, max(0.2, defaults["repetition_ratio"])), 0.01)
            word_count = st.slider("Word Count", 100, 600, int(min(600, max(100, defaults["word_count"]))), 1)
        submitted = st.form_submit_button("Simulate")

    st.caption(
        "Ghi chú: Điểm số mô phỏng cho bài hát đương đại (năm phát hành mặc định 2024). "
        "Các đặc trưng âm học và ca từ khác không nằm trên thanh trượt được cố định ở mức trung vị tập huấn luyện."
    )

    # Initial page load always predicts from the metadata defaults. Submitted
    # inputs replace those values; loudness and energy stay physically linked.
    if submitted:
        raw_row = {
            "market": market,
            "loudness_proxy": float(loudness),
            "energy": float(10.0 ** (loudness / 20.0)),
            "tempo": float(tempo),
            "tempo_stability": float(stability),
            "spectral_centroid": float(centroid),
            "repetition_ratio": float(repetition),
            "word_count": float(word_count),
            "release_year": 2024,
        }
    else:
        raw_row = {
            "market": "US",
            "loudness_proxy": defaults["loudness_proxy"],
            "energy": defaults["energy"],
            "tempo": defaults["tempo"],
            "tempo_stability": defaults["tempo_stability"],
            "spectral_centroid": defaults["spectral_centroid"],
            "repetition_ratio": defaults["repetition_ratio"],
            "word_count": defaults["word_count"],
            "release_year": 2024,
        }

    predictions, errors = _predict(predictor, raw_row)
    for error in errors:
        st.error(error)

    cols = st.columns(3)
    # Preserve unmodified model outputs for inspection while displaying values
    # within the chart-score domain used by the application.
    st.session_state["simulator_raw_predictions"] = predictions.copy()
    display_predictions = {
        name: max(-99.0, min(100.0, float(value)))
        for name, value in predictions.items()
    }
    cols[0].metric("OLS Model 3", f"{display_predictions['OLS Model 3']:.2f}" if "OLS Model 3" in display_predictions else "Unavailable")
    cols[1].metric("Random Forest Model 4", f"{display_predictions['Random Forest Model 4']:.2f}" if "Random Forest Model 4" in display_predictions else "Unavailable")
    if len(predictions) == 2:
        ensemble = (display_predictions["OLS Model 3"] + display_predictions["Random Forest Model 4"]) / 2.0
        cols[2].metric("Ensemble (mean)", f"{ensemble:.2f}")
    else:
        cols[2].metric("Ensemble (mean)", "Unavailable")


__all__ = ["render"]
