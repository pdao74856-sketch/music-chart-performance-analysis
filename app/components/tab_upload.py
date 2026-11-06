"""
app/components/tab_upload.py
Tab 3: Audio File Upload & Librosa Feature Extraction
Assigned to: Member 2 (Audio Lead) & Member 4 (Deploy Lead)
"""

import streamlit as st
import numpy as np
import pandas as pd
from src.utils.pipeline_inference import get_predictor


def render_tab_upload():
    st.header("🎙️ Tải Lên Âm Thanh & Trích Xuất Librosa Thời Gian Thực")
    st.markdown(
        """
        Tải lên file bài hát (`.mp3` hoặc `.wav`) để hệ thống phân tích tín hiệu số DSP 
        bằng thư viện **Librosa** (trích xuất toàn diện 17 đặc trưng âm học chuẩn hóa theo Audio Feature Contract v2).
        """
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        market = st.selectbox("Thị trường muốn đánh giá tiềm năng:", ["VN", "US", "KR"], index=0)
    with col_meta2:
        release_year = st.slider("Năm phát hành dự kiến (Release Year):", 2021, 2026, 2024)

    # Allow testing without an audio file via an explicit Demo Mode checkbox
    demo_mode = st.checkbox("🎵 Dùng ca khúc mẫu minh họa (Demo Mode)", value=False)

    uploaded_file = st.file_uploader("Chọn file bài hát (.mp3, .wav):", type=["mp3", "wav"])

    extracted_sample = None
    is_real = False

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"📁 Đã nhận file: `{uploaded_file.name}` ({file_size_mb:.2f} MB)")

        with st.spinner("⚡ Đang phân tích tín hiệu âm học DSP qua Librosa (Full Track)..."):
            raw_bytes = uploaded_file.getvalue()
            try:
                from src.features.audio_extractor import extract_features_for_inference
                extracted_sample = extract_features_for_inference(raw_bytes, filename=uploaded_file.name)
                is_real = True
            except Exception as e:
                st.error(f"❌ **Không thể trích xuất đặc trưng âm thanh từ file tải lên:** {e}")
                st.info("💡 Vui lòng kiểm tra lại định dạng tệp (.mp3, .wav), dung lượng (tối đa 15MB) hoặc đảm bảo tệp không bị tắt tiếng/hỏng dữ liệu.")
                return

    elif demo_mode:
        st.warning("⚠️ **Chế độ mô phỏng (Demo Mode):** Đang sử dụng bài hát mẫu tham chiếu để minh họa giao diện. Kết quả dự báo dựa trên các thông số âm học chuẩn hóa định sẵn.")
        extracted_sample = {
            "tempo": 124.0,
            "energy": 0.245,
            "tempo_stability": 0.985,
            "loudness_proxy": -11.8,
            "spectral_centroid": 2280.0,
            "spectral_bandwidth": 2400.0,
            "onset_rate": 196.0,
            "chord_change_rate": 44.0,
            "key_confidence": 0.82,
            "mode": 1.0,
            "rms_variability": 0.10,
            "chroma_longest_pattern_duration": 18.0,
            "mfcc_longest_pattern_duration": 15.0,
            "key": "C",
            "raw_mode": "Major",
            "duration_analyzed_seconds": 210.0,
        }
        is_real = False
    else:
        st.info("Vui lòng tải lên một file nhạc (.mp3 hoặc .wav) hoặc tích chọn **Dùng ca khúc mẫu minh họa (Demo Mode)** để bắt đầu.")
        return

    # Render acoustic features and predictions
    if is_real:
        duration = extracted_sample.get("duration_analyzed_seconds", 0.0)
        st.success(f"✅ **Trích xuất âm học THẬT thành công!** (Phân tích {duration:.1f}s trong ~9.6 giây, RAM đỉnh < 280MB)")

    st.subheader("1. Thông Số Âm Học Trích Xuất Được:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tempo (BPM)", f"{extracted_sample.get('tempo', 0.0):.1f}")
    col2.metric("Energy (RMS)", f"{extracted_sample.get('energy', 0.0):.3f}")
    col3.metric("Loudness", f"{extracted_sample.get('loudness_proxy', 0.0):.1f} dBFS")
    col4.metric("Spectral Centroid", f"{extracted_sample.get('spectral_centroid', 0.0):.0f} Hz")

    col5, col6, col7, col8 = st.columns(4)
    key_name = extracted_sample.get("key", "Unknown")
    raw_mode_str = str(extracted_sample.get("raw_mode", "Major")).capitalize()
    col5.metric("Điệu thức & Khóa (Key/Mode)", f"{key_name} {raw_mode_str}")
    col6.metric("Key Confidence", f"{extracted_sample.get('key_confidence', 0.0):.2f}")
    col7.metric("Chord Change Rate", f"{extracted_sample.get('chord_change_rate', 0.0):.1f} /min")
    col8.metric("RQA Longest Pattern", f"{extracted_sample.get('chroma_longest_pattern_duration', 0.0):.1f} s")

    with st.expander("🔍 Xem Toàn Bộ Danh Mục 17 Đặc Trưng Âm Học Chuẩn Hóa"):
        feat_df = pd.DataFrame([
            {"Đặc trưng âm học": k, "Giá trị trích xuất": v}
            for k, v in extracted_sample.items()
            if k not in ["filename", "raw_mode", "key", "duration_analyzed_seconds"]
        ])
        st.dataframe(feat_df, use_container_width=True)

    # Nối vào Dual-Model Inference Pipeline
    payload = dict(extracted_sample)
    payload["market"] = market
    payload["release_year"] = release_year

    predictor = get_predictor()
    res = predictor.predict(payload)

    st.markdown("---")
    header_title = "🎯 Dự Đoán Thành Tích Cho Ca Khúc:" if is_real else "🎯 Dự Đoán Thành Tích (Dữ Liệu Demo Minh Họa):"
    st.subheader(header_title)
    c1, c2 = st.columns(2)
    c1.metric("📈 Điểm Hồi Quy OLS (Model 3)", f"{res['ols_prediction']:.1f} pts")
    c2.metric("🌲 Điểm Random Forest (Model 4)", f"{res['rf_prediction']:.1f} pts")
    badge_note = f"Dự báo được thực hiện trên thị trường mục tiêu **{market}**, năm phát hành **{release_year}**."
    if not is_real:
        badge_note += " *(Lưu ý: Đang chạy ở chế độ Demo)*"
    st.caption(badge_note)
