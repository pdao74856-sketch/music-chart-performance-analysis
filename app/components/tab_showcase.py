"""
app/components/tab_showcase.py
Tab 1: Showcase 15 Test Tracks & Acoustic Radar Chart
Assigned to: Member 2 (with support from Member 4)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils.pipeline_inference import get_predictor
from src.utils.supabase_client import load_clean_modeling_data


def render_tab_showcase():
    st.header("🎵 Thư Viện Bài Hát Tiêu Biểu & Radar Chart Âm Học")
    st.markdown(
        """
        Khám phá 15 bài hát mẫu đại diện được trích xuất từ tập kiểm thử (*Test Set*). 
        Đối chiếu trực quan giữa **Điểm số thực tế** trên bảng xếp hạng với **Dự đoán từ Mô hình OLS và Random Forest**.
        """
    )

    try:
        df_sample = load_clean_modeling_data(source="auto", limit=50)
        # Select 15 representative tracks across VN, US, KR
        tracks_sample = df_sample.drop_duplicates(subset=["title", "market"]).head(15)
    except Exception:
        tracks_sample = pd.DataFrame([
            {"title": "See Tình", "artist": "Hoàng Thùy Linh", "market": "VN", "chart_score": 88.5, "tempo": 120.0, "energy": 0.23, "loudness_proxy": -12.5, "onset_rate": 185.0, "spectral_centroid": 2150.0, "chord_change_rate": 42.0},
            {"title": "Anti-Hero", "artist": "Taylor Swift", "market": "US", "chart_score": 94.2, "tempo": 97.0, "energy": 0.24, "loudness_proxy": -10.8, "onset_rate": 195.0, "spectral_centroid": 2240.0, "chord_change_rate": 44.0},
            {"title": "Super Shy", "artist": "NewJeans", "market": "KR", "chart_score": 92.0, "tempo": 150.0, "energy": 0.28, "loudness_proxy": -8.5, "onset_rate": 205.0, "spectral_centroid": 2520.0, "chord_change_rate": 46.0},
        ])

    track_options = [f"{row['title']} - {row['artist']} ({row['market']})" for _, row in tracks_sample.iterrows()]
    selected_option = st.selectbox("Chọn bài hát để phân tích:", track_options)

    selected_idx = track_options.index(selected_option)
    selected_row = tracks_sample.iloc[selected_idx].to_dict()

    predictor = get_predictor()
    pred_res = predictor.predict(selected_row)

    # 3 Score Comparison Columns
    col1, col2, col3 = st.columns(3)
    actual_score = selected_row.get("chart_score", 50.0)
    col1.metric("🏆 Điểm Thực Tế", f"{actual_score:.1f}")
    col2.metric("📈 Dự Đoán OLS (Model 3)", f"{pred_res['ols_prediction']:.1f}", delta=f"{pred_res['ols_prediction'] - actual_score:+.1f}")
    col3.metric("🌲 Dự Đoán RF (Model 4)", f"{pred_res['rf_prediction']:.1f}", delta=f"{pred_res['rf_prediction'] - actual_score:+.1f}")

    # Radar Chart
    st.subheader("Biểu đồ Mạng Nhện (Acoustic Radar Chart)")
    categories = ['Tempo', 'Energy', 'Loudness', 'Onset Rate', 'Spectral Centroid', 'Chord Change']
    
    # Normalized [0, 1] for radar chart
    values = [
        float(np.clip(selected_row.get("tempo", 120.0) / 180.0, 0.1, 1.0)),
        float(np.clip(selected_row.get("energy", 0.24) / 0.50, 0.1, 1.0)),
        float(np.clip((selected_row.get("loudness_proxy", -12.0) + 25.0) / 19.0, 0.1, 1.0)),
        float(np.clip(selected_row.get("onset_rate", 190.0) / 250.0, 0.1, 1.0)),
        float(np.clip(selected_row.get("spectral_centroid", 2200.0) / 3500.0, 0.1, 1.0)),
        float(np.clip(selected_row.get("chord_change_rate", 42.0) / 60.0, 0.1, 1.0)),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(43, 92, 143, 0.4)',
        line=dict(color='#2b5c8f', width=2),
        name=selected_row.get("title", "Bài hát")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
