"""
app/components/tab_simulator.py
Tab 2: What-If Hit Song Simulator Sliders
Assigned to: Member 3 & Member 4
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from src.utils.pipeline_inference import get_predictor

ROOT_DIR = Path(__file__).resolve().parents[2]
META_PATH = ROOT_DIR / "models" / "model_metadata.json"


def render_tab_simulator():
    st.header("🎛️ Giả Lập Sáng Tác Hit (What-If Simulator)")
    st.markdown(
        """
        Thay đổi các thanh trượt đặc trưng âm học Librosa và ca từ NLP để xem **Điểm thành tích dự báo** 
        phản hồi tức thì thời gian thực từ cả hai mô hình (OLS vs Random Forest) trên thang đo chuẩn `[-99.0, 100.0] pts`.
        """
    )

    # Load medians as safe defaults
    medians = {}
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                medians = meta.get("train_medians", {})
        except Exception:
            pass

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        market = st.selectbox("Thị trường mục tiêu (Market):", ["US", "VN", "KR"], index=1)
    with col_meta2:
        release_year = st.slider("Năm phát hành (Release Year):", 2021, 2026, 2024)

    st.subheader("1. Đặc trưng Tín hiệu Âm học (Librosa)")
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        tempo = st.slider("Tempo (BPM):", 60.0, 190.0, float(medians.get("tempo", 121.1)), 1.0)
        # Giữ 1 slider chính Loudness để chống Out-of-Distribution, tự động tính Energy = 10^(Loudness/20)
        loudness_proxy = st.slider("Độ lớn Âm lượng (Loudness dBFS):", -25.0, -5.0, float(medians.get("loudness_proxy", -12.9)), 0.5)
        energy = float(10.0 ** (loudness_proxy / 20.0))
        st.caption(f"⚡ Energy (RMS suy luận tự động): **{energy:.3f}**")

    with col_a2:
        tempo_stability = st.slider("Tempo Stability:", 0.70, 1.00, float(medians.get("tempo_stability", 0.98)), 0.01)
        spectral_centroid = st.slider("Spectral Centroid (Hz):", 1000.0, 3800.0, float(medians.get("spectral_centroid", 2225.0)), 50.0)
        spectral_bandwidth = st.slider("Spectral Bandwidth (Hz):", 1200.0, 3600.0, float(medians.get("spectral_bandwidth", 2441.0)), 50.0)

    with col_a3:
        onset_rate = st.slider("Onset Rate (events/s):", 80.0, 260.0, float(medians.get("onset_rate", 193.5)), 5.0)
        chord_change_rate = st.slider("Chord Change Rate (/min):", 10.0, 70.0, float(medians.get("chord_change_rate", 44.2)), 1.0)
        key_confidence = st.slider("Key Confidence:", 0.30, 1.00, float(medians.get("key_confidence", 0.72)), 0.05)
        mode = st.radio("Điệu thức (Mode):", [1, 0], format_func=lambda x: "Trưởng (Major)" if x == 1 else "Thứ (Minor)", horizontal=True)

    st.subheader("2. Đặc trưng Cấu trúc Ca từ (NLP)")
    col_l1, col_l2, col_l3 = st.columns(3)

    with col_l1:
        word_count = st.slider("Số từ (Word Count):", 80, 800, int(medians.get("word_count", 350)), 10)
    with col_l2:
        repetition_ratio = st.slider("Tỷ lệ lặp lại (Repetition Ratio):", 0.20, 0.85, float(medians.get("repetition_ratio", 0.60)), 0.01)
    with col_l3:
        chorus_repetition_count = st.slider("Số lần lặp điệp khúc:", 1, 8, int(medians.get("chorus_repetition_count", 3)), 1)

    # Build input payload
    raw_input = {
        "market": market,
        "release_year": release_year,
        "tempo": tempo,
        "energy": energy,
        "tempo_stability": tempo_stability,
        "loudness_proxy": loudness_proxy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "onset_rate": onset_rate,
        "chord_change_rate": chord_change_rate,
        "key_confidence": key_confidence,
        "mode": mode,
        "audio_mode": "major" if mode == 1 else "minor",
        "word_count": word_count,
        "repetition_ratio": repetition_ratio,
        "chorus_repetition_count": chorus_repetition_count,
        "avg_word_length": medians.get("avg_word_length", 3.8),
        "line_count": medians.get("line_count", 48.0),
        "punctuation_ratio": medians.get("punctuation_ratio", 0.05),
        "rms_variability": medians.get("rms_variability", 0.11),
        "chroma_repetition_determinism": medians.get("chroma_repetition_determinism", 0.45),
        "chroma_mean_pattern_duration": medians.get("chroma_mean_pattern_duration", 5.0),
        "chroma_longest_pattern_duration": medians.get("chroma_longest_pattern_duration", 15.0),
        "mfcc_repetition_determinism": medians.get("mfcc_repetition_determinism", 0.48),
        "mfcc_mean_pattern_duration": medians.get("mfcc_mean_pattern_duration", 5.5),
        "mfcc_longest_pattern_duration": medians.get("mfcc_longest_pattern_duration", 18.0)
    }

    # Predict
    predictor = get_predictor()
    res = predictor.predict(raw_input)

    st.markdown("---")
    st.subheader("🎯 Kết Quả Dự Báo Thành Tích (ChartScore)")

    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Dự đoán OLS (Model 3)", f"{res['ols_prediction']:.1f} pts")
    c2.metric("🌲 Dự đoán Random Forest (Model 4)", f"{res['rf_prediction']:.1f} pts")
    c3.metric("🤝 Điểm Trung Bình Ensemble", f"{res['ensemble_prediction']:.1f} pts")

    if res['rf_prediction'] >= 60.0:
        st.success(f"🌟 Tiềm năng Hit cao tại thị trường {market}! Bài hát sở hữu các thông số cân bằng tối ưu.")
    elif res['rf_prediction'] >= 10.0:
        st.info(f"🎶 Mức độ thành tích trung bình khá tại thị trường {market}.")
    else:
        st.warning(f"⚠️ Điểm số khiêm tốn hoặc thứ hạng thấp tại thị trường {market}. Cân nhắc tinh chỉnh lại năng lượng hoặc cấu trúc hook ca từ.")
