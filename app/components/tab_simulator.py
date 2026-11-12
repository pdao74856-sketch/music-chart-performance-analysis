"""TV3 what-if simulator using the project's trained inference pipelines."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.pipeline_inference import DualModelPredictor
from app.components.market_position import render_market_position, render_prediction_context
from app.components.prediction_feedback import render_prediction_feedback


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
MARKETS = ("VN", "US", "KR")


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
    st.header("So sánh 3 thị trường")
    st.caption("Thử cùng một cấu hình bài hát tại Việt Nam, Hoa Kỳ và Hàn Quốc.")

    try:
        predictor = _load_predictor()
    except Exception as exc:
        st.error(f"Could not initialize the model inference pipeline: {exc}")
        return

    defaults = _defaults(predictor)
    with st.form("what_if_simulator"):
        controls, action = st.columns([3, 1], vertical_alignment="bottom")
        with controls:
            loudness = st.slider("Loudness (dBFS)", -25.0, -6.0, -15.0, 0.01)
        with action:
            submitted = st.form_submit_button("So sánh 3 thị trường", type="primary", use_container_width=True)
        with st.expander("Tinh chỉnh các đặc trưng khác"):
            first, second = st.columns(2)
            with first:
                tempo = st.slider("Tempo (BPM)", 60.0, 180.0, min(180.0, max(60.0, defaults["tempo"])), 1.0)
                centroid = st.slider("Spectral Centroid (Hz)", 1000.0, 4000.0, min(4000.0, max(1000.0, defaults["spectral_centroid"])), 10.0)
                stability = st.slider("Tempo Stability", 0.80, 1.00, min(1.0, max(0.80, defaults["tempo_stability"])), 0.001)
            with second:
                repetition = st.slider("Repetition Ratio", 0.2, 0.8, min(0.8, max(0.2, defaults["repetition_ratio"])), 0.01)
                word_count = st.slider("Word Count", 100, 600, int(min(600, max(100, defaults["word_count"]))), 1)

    if submitted or "simulator_inputs" not in st.session_state:
        inputs = {
            "loudness_proxy": float(loudness),
            "energy": 10.0 ** (loudness / 20.0),
            "tempo": float(tempo),
            "tempo_stability": float(stability),
            "spectral_centroid": float(centroid),
            "repetition_ratio": float(repetition),
            "word_count": float(word_count),
            "release_year": 2024,
        }

        runs = {market: _predict(predictor, {**inputs, "market": market}) for market in MARKETS}
        st.session_state["simulator_inputs"] = inputs
        st.session_state["simulator_runs"] = runs

    inputs = st.session_state["simulator_inputs"]
    runs = st.session_state["simulator_runs"]
    results = {market: predictions for market, (predictions, _) in runs.items()}
    has_rf_results = any("Random Forest Model 4" in scores for scores in results.values())
    if has_rf_results:
        st.info(
            "**Cách đọc kết quả**  \n"
            "Ví dụ, **Top ≈20%** nghĩa là điểm dự đoán nằm trong khoảng 20% cao nhất của mẫu tham chiếu. "
            "Mỗi thị trường có mẫu tham chiếu riêng. Vạch xanh trên biểu đồ đánh dấu điểm mô phỏng."
        )
    for market, column in zip(MARKETS, st.columns(3)):
        predictions, errors = runs[market]
        with column:
            for error in errors:
                st.error(f"{market} · {error}")
            if "Random Forest Model 4" in predictions:
                render_market_position(st, predictions["Random Forest Model 4"], market, f"simulator_{market}")

    st.session_state["simulator_raw_predictions"] = results
    if has_rf_results:
        render_prediction_context(st)
    with st.expander("Xem điểm OLS, RF và đầu vào mô phỏng"):
        st.caption("Năm phát hành mô phỏng là 2024. Các đặc trưng ngoài phần điều chỉnh dùng trung vị tập huấn luyện.")
        st.table({market: {name: round(score, 2) for name, score in scores.items()} for market, scores in results.items()})
        st.caption(
            f"Loudness: {inputs['loudness_proxy']:.2f} dBFS · "
            f"Tempo: {inputs['tempo']:.1f} BPM · "
            f"Trọng tâm phổ: {inputs['spectral_centroid']:.0f} Hz"
        )
        st.caption(
            f"Độ ổn định nhịp: {inputs['tempo_stability']:.3f} · "
            f"Tỷ lệ lặp: {inputs['repetition_ratio']:.1%} · "
            f"Số từ: {inputs['word_count']:.0f}"
        )
    if has_rf_results:
        render_prediction_feedback(st, "simulator")


__all__ = ["render"]
