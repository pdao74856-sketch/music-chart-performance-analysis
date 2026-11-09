"""
app/components/tab_showcase.py
Tab 1: Thư Viện Bài Mẫu Showcase & Radar Chart Âm Học
Project: What Makes a Hit Song? (Music Chart Performance Analysis 2021-2026)
Owner: Member 2 (Audio DSP & Librosa Feature Engineering Lead)

Key Features:
- Curates 15 representative showcase tracks from the out-of-sample test set (models/test_track_ids.json).
- Represents 3 key markets: VN (5 tracks), US (5 tracks), KR (5 tracks) across all 4 empirical performance tiers.
- Interactive Radar Chart (Spider Chart) via Plotly Graph Objects with closed polygons.
- Min-Max Scaling [0, 1] across the 15 showcase tracks for 6 acoustic dimensions:
  tempo (BPM), energy [0, 1], loudness_proxy (dBFS), onset_rate (onsets/min),
  spectral_centroid (Hz), and chord_change_rate (changes/min).
  Gracefully handles negative dBFS values for loudness_proxy (-25 to -6 dBFS).
- Baseline Median reference polygon for comparative profiling.
- Side-by-side 3-metric comparison: Actual Chart Score vs OLS Model 3 (TV1) vs RF Model 4 (TV4).
- Offline-first resilience with @st.cache_data: Queries Supabase via load_clean_modeling_data(source="auto")
  with seamless fallback to local cache (modeling_dataset_clean_cache.csv).
- Academic econometric commentary on hit song unpredictability and residual interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup Streamlit caching decorator safely for both Streamlit runtime and headless test environments
try:
    import streamlit as st
    _cache_data = st.cache_data(show_spinner=False)
except Exception:
    def _cache_data(func):
        return func

# Root directory resolution (depth 2 from app/components/tab_showcase.py -> repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_PREDICTIONS_PATH = ROOT_DIR / "models" / "test_predictions.csv"
TEST_TRACK_IDS_PATH = ROOT_DIR / "models" / "test_track_ids.json"

# ==============================================================================
# 1. EMPIRICAL CONFIGURATION & CONSTANTS
# ==============================================================================

# Empirical Chart Score Quantiles from Clean Dataset (n=2,098)
# Q1 = -11.93, Median = 15.16, Q3 = 37.58
TIER_THRESHOLDS = {
    "Q1": -11.93,
    "Median": 15.16,
    "Q3": 37.58,
}

# 6 Key Acoustic Dimensions for Radar Chart
RADAR_FEATURE_DEFS = [
    {
        "key": "tempo",
        "label": "Nhịp độ (Tempo)",
        "unit": "BPM",
        "fmt": "{:.1f}",
        "desc": "Tốc độ nhịp đập trung bình của ca khúc",
    },
    {
        "key": "energy",
        "label": "Năng lượng (Energy)",
        "unit": "[0, 1]",
        "fmt": "{:.3f}",
        "desc": "Mức năng lượng âm thanh cảm nhận được (RMS)",
    },
    {
        "key": "loudness_proxy",
        "label": "Độ lớn (Loudness)",
        "unit": "dBFS",
        "fmt": "{:.2f}",
        "desc": "Cường độ âm thanh tương đối (-25 đến -6 dBFS)",
    },
    {
        "key": "onset_rate",
        "label": "Mật độ nốt (Onset Rate)",
        "unit": "onsets/min",
        "fmt": "{:.1f}",
        "desc": "Tần suất bắt đầu của các nốt nhạc mỗi phút",
    },
    {
        "key": "spectral_centroid",
        "label": "Độ sáng phổ (Centroid)",
        "unit": "Hz",
        "fmt": "{:.0f}",
        "desc": "Trọng tâm tần số - phản ánh độ sáng/trong của âm sắc",
    },
    {
        "key": "chord_change_rate",
        "label": "Chuyển hợp âm (Chord Rate)",
        "unit": "changes/min",
        "fmt": "{:.1f}",
        "desc": "Tần suất thay đổi hòa âm/hợp âm mỗi phút",
    },
]

RADAR_KEYS = [f["key"] for f in RADAR_FEATURE_DEFS]
RADAR_LABELS = [f["label"] for f in RADAR_FEATURE_DEFS]

# 15 Representative Tracks (5 VN, 5 US, 5 KR) selected from models/test_track_ids.json
# Balanced across all 4 empirical performance tiers: exactly 1 Tier 1, 2 Tier 2, 1 Tier 3, 1 Tier 4 per market
SHOWCASE_SPECIFICATIONS = [
    # --- VIETNAM (VN) ---
    {
        "spotify_track_id": "2HRgqmZQC0MC7GeNuDIXHN",
        "market": "VN",
        "highlight": "Điểm BXH cao nhất nhóm VN (99.2 điểm, Tier 1); Độ lớn vượt trội (-12.79 dBFS), Độ sáng phổ cao (2,686 Hz)",
    },
    {
        "spotify_track_id": "5ydjxBSUIDn26MFzU3asP4",
        "market": "VN",
        "highlight": "Điểm trên trung vị (29.7 điểm, Tier 2); Tốc độ nhịp 143.6 BPM, Âm lượng êm dịu (-14.03 dBFS)",
    },
    {
        "spotify_track_id": "3Nn9QjxOEj1pleW9xTG6cq",
        "market": "VN",
        "highlight": "Điểm trên trung vị (17.7 điểm, Tier 2); Mật độ nốt dày (242.3 onsets/min), Năng lượng cao (0.293)",
    },
    {
        "spotify_track_id": "3yWacnoBnM0lrzLmXAk44d",
        "market": "VN",
        "highlight": "Điểm dưới trung vị (-11.0 điểm, Tier 3); Tempo chậm rãi (86.1 BPM), Mật độ nốt thấp nhất mẫu VN (110.1 onsets/min)",
    },
    {
        "spotify_track_id": "5Ss7wCp22FOAbfeg75KllJ",
        "market": "VN",
        "highlight": "Điểm đáy BXH (-88.0 điểm, Tier 4); Độ sáng phổ thấp nhất toàn bộ 15 bài mẫu (797 Hz)",
    },

    # --- UNITED STATES (US) ---
    {
        "spotify_track_id": "2tzgSUZOUKopYbRqQOVT8X",
        "market": "US",
        "highlight": "Điểm đỉnh BXH US (82.0 điểm, Tier 1); Mật độ nốt (293.7 onsets/min) và Năng lượng (0.434) cao nhất toàn bộ 15 bài mẫu",
    },
    {
        "spotify_track_id": "3aQem4jVGdhtg116TmJnHz",
        "market": "US",
        "highlight": "Điểm trên trung vị (25.0 điểm, Tier 2); Tần suất đổi hợp âm cao nhất toàn bộ 15 bài mẫu (85.1 changes/min)",
    },
    {
        "spotify_track_id": "7KA4W4McWYRpgf0fWsJZWB",
        "market": "US",
        "highlight": "Điểm trên trung vị (22.1 điểm, Tier 2); Âm lượng nhỏ nhất mẫu (-15.84 dBFS), Năng lượng tối giản (0.161)",
    },
    {
        "spotify_track_id": "200x8ONbPfFzByUr3fVz46",
        "market": "US",
        "highlight": "Điểm dưới trung vị (2.4 điểm, Tier 3); Độ sáng phổ hài hòa (2,522 Hz), Tempo 136.0 BPM",
    },
    {
        "spotify_track_id": "2J051fjLklkoPbzOoTAACZ",
        "market": "US",
        "highlight": "Điểm đáy BXH (-94.0 điểm, Tier 4); Mật độ nốt cao (232.6 onsets/min) nhưng không bám trụ được BXH",
    },

    # --- SOUTH KOREA (KR) ---
    {
        "spotify_track_id": "0AQEShePVSsxMTM6a8ik6t",
        "market": "KR",
        "highlight": "Điểm đỉnh BXH KR (96.0 điểm, Tier 1); Năng lượng cao (0.323), Độ lớn nổi bật (-9.83 dBFS)",
    },
    {
        "spotify_track_id": "1jRYyE9MEg7nxUgWEtsqFp",
        "market": "KR",
        "highlight": "Điểm trên trung vị (36.7 điểm, Tier 2); Tempo thư thái (112.3 BPM), Chuyển hợp âm ổn định (45.2 changes/min)",
    },
    {
        "spotify_track_id": "3khEEPRyBeOUabbmOPJzAG",
        "market": "KR",
        "highlight": "Điểm trên trung vị (19.6 điểm, Tier 2); Năng lượng âm học cao (0.289), Trọng tâm phổ 2,425 Hz",
    },
    {
        "spotify_track_id": "6G3myAs4HemIdtLYhhjGLP",
        "market": "KR",
        "highlight": "Điểm tiệm cận 0 (0.0 điểm, Tier 3); Tempo nhanh nhất 15 bài (161.5 BPM), Chuyển hợp âm dồn dập (72.4 changes/min)",
    },
    {
        "spotify_track_id": "3LEgGAl9zLcQWGTp7ptqtQ",
        "market": "KR",
        "highlight": "Điểm đáy BXH (-94.0 điểm, Tier 4); Độ sáng phổ cao nhất 15 bài (2,835 Hz) nhưng điểm BXH thấp",
    },
]

def classify_chart_tier(score: float) -> Tuple[str, str]:
    """Phân loại thứ hạng thành tích bài hát dựa trên phân vị thực nghiệm."""
    if score >= TIER_THRESHOLDS["Q3"]:
        return "Tier 1", "Viral Hit (Top 25% Đỉnh BXH)"
    elif score >= TIER_THRESHOLDS["Median"]:
        return "Tier 2", "Solid Performer (Trên Trung vị)"
    elif score >= TIER_THRESHOLDS["Q1"]:
        return "Tier 3", "Moderate / Lower (Dưới Trung vị)"
    else:
        return "Tier 4", "Deep Chart (Nhóm Đáy BXH)"


# ==============================================================================
# 2. DATA LOADERS & MIN-MAX SCALING
# ==============================================================================

def compute_min_max_scaling(
    df: pd.DataFrame, feature_keys: List[str]
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """
    Áp dụng Min-Max Scaling đưa cả 6 đặc trưng về thang đo chuẩn [0, 1] trên tập 15 bài mẫu.
    Xử lý an toàn đặc trưng âm học loudness_proxy mang giá trị âm (-25 đến -6 dBFS).
    
    Formula:
        scaled = (x - min) / (max - min)
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    scaled_df = df.copy()

    for key in feature_keys:
        if key not in df.columns:
            continue
        c_min = float(df[key].min())
        c_max = float(df[key].max())
        bounds[key] = (c_min, c_max)

        span = c_max - c_min
        if span > 1e-9:
            scaled_df[f"{key}_scaled"] = (df[key] - c_min) / span
        else:
            scaled_df[f"{key}_scaled"] = 0.5

    return bounds, scaled_df


@_cache_data
def load_showcase_dataset() -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Nạp dữ liệu 15 bài mẫu showcase (được cache qua @st.cache_data):
    1. Truy vấn đặc trưng âm học từ Supabase / cache nội bộ qua load_clean_modeling_data.
    2. Nạp kết quả dự đoán OLS Model 3 (TV1) và RF Model 4 (TV4) từ models/test_predictions.csv.
    3. Thực hiện Min-Max scaling trên 15 bài mẫu cho 6 trục Radar.
    """
    from src.utils.supabase_client import load_clean_modeling_data

    # 1. Load clean features (Supabase first, fallback local cache)
    clean_df = load_clean_modeling_data(source="auto")

    # 2. Load out-of-sample predictions and validate test IDs
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing required predictions file: {TEST_PREDICTIONS_PATH}")
    preds_df = pd.read_csv(TEST_PREDICTIONS_PATH)

    if TEST_TRACK_IDS_PATH.exists():
        with open(TEST_TRACK_IDS_PATH, "r", encoding="utf-8") as f_ids:
            test_ids_data = json.load(f_ids)
            valid_test_set = set(test_ids_data.get("track_ids", []))
            for s in SHOWCASE_SPECIFICATIONS:
                if s["spotify_track_id"] not in valid_test_set:
                    raise ValueError(f"Showcase track {s['spotify_track_id']} not in test split!")

    # 3. Merge predictions with clean acoustic features
    feature_cols = [
        "spotify_track_id",
        "market",
        "tempo",
        "energy",
        "loudness_proxy",
        "onset_rate",
        "spectral_centroid",
        "chord_change_rate",
        "release_year",
    ]
    available_cols = [c for c in feature_cols if c in clean_df.columns]
    merged = pd.merge(
        preds_df,
        clean_df[available_cols],
        on=["spotify_track_id", "market"],
        how="inner",
    )

    # 4. Filter strictly the 15 curated showcase tracks & dynamically derive tiers
    spec_rows: List[pd.DataFrame] = []
    for spec in SHOWCASE_SPECIFICATIONS:
        tid = spec["spotify_track_id"]
        mkt = spec["market"]
        match = merged[(merged["spotify_track_id"] == tid) & (merged["market"] == mkt)]
        if not match.empty:
            row_copy = match.iloc[0:1].copy()
            act_score = float(row_copy["chart_score_actual"].iloc[0])
            tier_code, tier_desc = classify_chart_tier(act_score)
            row_copy["tier_code"] = tier_code
            row_copy["tier_name"] = f"{tier_code}: {tier_desc}"
            row_copy["highlight"] = spec["highlight"]
            spec_rows.append(row_copy)

    if not spec_rows:
        raise ValueError("Could not find any of the 15 showcase tracks in the dataset.")

    showcase_df = pd.concat(spec_rows, ignore_index=True)

    # 5. Add error metrics
    showcase_df["error_ols"] = showcase_df["pred_ols_m3"] - showcase_df["chart_score_actual"]
    showcase_df["error_rf"] = showcase_df["pred_rf"] - showcase_df["chart_score_actual"]
    showcase_df["abs_error_ols"] = showcase_df["error_ols"].abs()
    showcase_df["abs_error_rf"] = showcase_df["error_rf"].abs()

    # 6. Compute Min-Max scaling across the 15 showcase tracks
    bounds, showcase_df = compute_min_max_scaling(showcase_df, RADAR_KEYS)

    return showcase_df, bounds


# ==============================================================================
# 3. RADAR CHART GENERATION (PLOTLY GRAPH OBJECTS)
# ==============================================================================

def create_radar_chart(
    selected_row: pd.Series,
    showcase_df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Any:
    """
    Vẽ biểu đồ mạng nhện Radar Chart bằng plotly.graph_objects:
    - 6 trục âm học chuẩn hóa [0, 1].
    - Lớp 1 (Selected Track): Đa giác khép kín tô màu gradient, viền đậm.
    - Lớp 2 (Showcase Median): Đa giác nét đứt màu xám chuẩn đối chuẩn.
    - Hover Tooltip hiển thị giá trị thực tế kèm đơn vị đo (BPM, dBFS, Hz, v.v.).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    # Compute median values across the 15 showcase tracks for both raw and scaled
    median_scaled = [float(showcase_df[f"{k}_scaled"].median()) for k in RADAR_KEYS]
    track_scaled = [float(selected_row[f"{k}_scaled"]) for k in RADAR_KEYS]

    # Raw values for hover tooltip
    track_raw = [float(selected_row[k]) for k in RADAR_KEYS]
    median_raw = [float(showcase_df[k].median()) for k in RADAR_KEYS]

    # Format hover text with physical units and Min-Max context
    track_hover_text = []
    median_hover_text = []
    for f_def, t_val, m_val, t_sc, m_sc in zip(
        RADAR_FEATURE_DEFS, track_raw, median_raw, track_scaled, median_scaled
    ):
        k = f_def["key"]
        unit = f_def["unit"]
        fmt = f_def["fmt"]
        label = f_def["label"]
        min_v, max_v = bounds.get(k, (0.0, 1.0))

        t_txt = (
            f"<b>{label}</b><br>"
            f"Giá trị thực: {fmt.format(t_val)} {unit}<br>"
            f"Chuẩn hóa: {t_sc:.2f} / 1.00<br>"
            f"Thang mẫu (15 bài): [{fmt.format(min_v)}, {fmt.format(max_v)}]"
        )
        track_hover_text.append(t_txt)

        m_txt = (
            f"<b>{label} (Trung vị 15 bài)</b><br>"
            f"Trung vị thực: {fmt.format(m_val)} {unit}<br>"
            f"Chuẩn hóa: {m_sc:.2f} / 1.00"
        )
        median_hover_text.append(m_txt)

    # To close polygons in Plotly Scatterpolar, repeat the first element at the end
    categories_closed = RADAR_LABELS + [RADAR_LABELS[0]]
    track_scaled_closed = track_scaled + [track_scaled[0]]
    track_hover_closed = track_hover_text + [track_hover_text[0]]
    median_scaled_closed = median_scaled + [median_scaled[0]]
    median_hover_closed = median_hover_text + [median_hover_text[0]]

    track_title = str(selected_row.get("title", "Selected Track"))
    track_artist = str(selected_row.get("artist", ""))

    fig = go.Figure()

    # Trace 1: Median Reference Polygon (dashed gray)
    fig.add_trace(
        go.Scatterpolar(
            r=median_scaled_closed,
            theta=categories_closed,
            name="Trung vị 15 bài mẫu",
            line=dict(color="#94A3B8", width=1.8, dash="dash"),
            marker=dict(size=4, color="#94A3B8"),
            hoverinfo="text",
            text=median_hover_closed,
        )
    )

    # Trace 2: Selected Track Polygon (Indigo fill)
    fig.add_trace(
        go.Scatterpolar(
            r=track_scaled_closed,
            theta=categories_closed,
            name=f"{track_title} ({selected_row.get('market', '')})",
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.25)",
            line=dict(color="#6366F1", width=2.8),
            marker=dict(size=7, color="#6366F1"),
            hoverinfo="text",
            text=track_hover_closed,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.2 (Thấp)", "0.4", "0.6", "0.8", "1.0 (Cao)"],
                tickfont=dict(size=9, color="#64748B"),
                gridcolor="rgba(148, 163, 184, 0.25)",
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickfont=dict(size=11, color="#64748B"),
                gridcolor="rgba(148, 163, 184, 0.25)",
            ),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=40, t=30, b=70),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ==============================================================================
# 4. STREAMLIT RENDER FUNCTION
# ==============================================================================

def render() -> None:
    """
    Render giao diện Tab 1 Showcase & Radar Chart.
    Thiết kế để Member 4 có thể tích hợp an toàn qua import độc lập.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is required to render Tab 1 Showcase.") from exc

    st.header("🎯 Thư Viện Bài Mẫu Showcase & Radar Chart Âm Học")
    st.caption(
        "Khám phá hồ sơ âm học và đối chuẩn dự đoán trên 15 ca khúc tiêu biểu từ tập kiểm thử độc lập (OOS Test Set $n=382$ ca khúc / $410$ quan sát), "
        "đại diện đầy đủ cho 3 thị trường Việt Nam (VN), Hoa Kỳ (US) và Hàn Quốc (KR)."
    )

    # 1. Nạp dữ liệu với cơ chế cache của Streamlit
    try:
        showcase_df, bounds = load_showcase_dataset()
    except Exception as exc:
        st.error(f"⚠️ Không thể khởi tạo dữ liệu Showcase: {exc}")
        st.info("Vui lòng đảm bảo các tệp `models/test_predictions.csv` và dữ liệu clean cache khả dụng.")
        return

    # 2. Bộ lọc Thị trường & Lựa chọn ca khúc
    c_mkt, c_trk = st.columns([1, 2])
    with c_mkt:
        market_options = ["Tất cả (15 bài)", "Việt Nam (VN)", "Hoa Kỳ (US)", "Hàn Quốc (KR)"]
        market_choice = st.selectbox(
            "Lọc theo thị trường:",
            market_options,
            index=0,
            key="showcase_market_filter",
        )

    # Lọc dataframe theo lựa chọn
    filtered_df = showcase_df.copy()
    if market_choice == "Việt Nam (VN)":
        filtered_df = filtered_df[filtered_df["market"] == "VN"]
    elif market_choice == "Hoa Kỳ (US)":
        filtered_df = filtered_df[filtered_df["market"] == "US"]
    elif market_choice == "Hàn Quốc (KR)":
        filtered_df = filtered_df[filtered_df["market"] == "KR"]

    track_options = []
    track_id_map = {}
    for _, r in filtered_df.iterrows():
        label = f"[{r['market']}] {r['title']} - {r['artist']} (Thực tế: {r['chart_score_actual']:.1f} điểm)"
        track_options.append(label)
        track_id_map[label] = (r["spotify_track_id"], r["market"])

    if not track_options:
        st.warning("Không có ca khúc nào phù hợp với bộ lọc đã chọn.")
        return

    with c_trk:
        selected_label = st.selectbox(
            "Chọn ca khúc phân tích:",
            track_options,
            index=0,
            key=f"showcase_track_picker_{market_choice}",
        )

    # 3. Lấy thông tin bài hát được chọn (phòng thủ an toàn kép chống KeyError)
    if selected_label not in track_id_map:
        selected_label = track_options[0]
    sel_tid, sel_mkt = track_id_map[selected_label]
    sel_match = filtered_df[
        (filtered_df["spotify_track_id"] == sel_tid) & (filtered_df["market"] == sel_mkt)
    ]
    if sel_match.empty:
        st.warning("Không tìm thấy thông tin ca khúc đã chọn.")
        return
    sel_row = sel_match.iloc[0]

    actual_score = float(sel_row["chart_score_actual"])
    pred_ols = float(sel_row["pred_ols_m3"])
    pred_rf = float(sel_row["pred_rf"])
    err_ols = float(sel_row["error_ols"])
    err_rf = float(sel_row["error_rf"])
    abs_ols = float(sel_row["abs_error_ols"])
    abs_rf = float(sel_row["abs_error_rf"])

    tier_code, tier_desc = classify_chart_tier(actual_score)

    st.markdown("---")

    # 4. Banner Thông Tin Ca Khúc & Phân Tầng Thứ Hạng
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        st.subheader(f"🎵 {sel_row['title']} – {sel_row['artist']}")
        st.markdown(
            f"**Thị trường:** `{sel_row['market']}` | "
            f"**Năm phát hành:** `{int(sel_row.get('release_year', 2022))}` | "
            f"**Spotify ID:** `{sel_tid}` | "
            f"[🔗 Mở trên Spotify](https://open.spotify.com/track/{sel_tid})"
        )
        st.caption(f"💡 *Đặc điểm âm học nổi bật:* {sel_row.get('highlight', 'Không có mô tả')}")

    with b_col2:
        badge_color = {
            "Tier 1": "🔥",
            "Tier 2": "⭐",
            "Tier 3": "📈",
            "Tier 4": "📉",
        }.get(tier_code, "📊")
        st.metric(
            label="Phân Tầng Thành Tích",
            value=f"{badge_color} {tier_code}",
            help=f"Dựa trên phân vị thực nghiệm: Q1={TIER_THRESHOLDS['Q1']}, Med={TIER_THRESHOLDS['Median']}, Q3={TIER_THRESHOLDS['Q3']}",
        )
        st.caption(tier_desc)

    # 5. Giao Diện Song Song: Radar Chart (Trái) & 3 Cột Kết Quả Đối Chiếu (Phải)
    col_chart, col_metrics = st.columns([3, 2])

    with col_chart:
        st.markdown("#### 🕸️ Biểu Đồ Mạng Nhện Hồ Sơ Âm Học (Radar Chart)")
        st.caption(
            "Biểu đồ chuẩn hóa Min-Max Scaling $[0, 1]$ trên 15 ca khúc đại diện. "
            "Đường xám đứt đoạn biểu thị **Trung vị 15 bài mẫu** làm mốc tham chiếu đối chuẩn."
        )

        fig = create_radar_chart(sel_row, showcase_df, bounds)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không thể hiển thị biểu đồ tương tác (thiếu thư viện Plotly).")

    with col_metrics:
        st.markdown("#### ⚖️ Đối Chiếu Song Song 3 Chỉ Số")
        st.caption("So sánh giữa điểm thực tế trên BXH Spotify và dự đoán từ 2 kiến trúc mô hình:")

        # Ô 1: Điểm thực tế
        st.metric(
            label="1. Điểm BXH Thực Tế (Actual Score)",
            value=f"{actual_score:.1f} điểm",
            help="Điểm chuẩn hóa phản ánh thời gian bám trụ và thứ hạng đỉnh trên BXH Spotify [-99.0, 100.0].",
        )

        # Ô 2: Dự đoán OLS Model 3
        st.metric(
            label="2. Dự Đoán OLS Model 3 (TV1)",
            value=f"{pred_ols:.1f} điểm",
            delta=f"{err_ols:+.1f} pts (Sai số OLS)",
            delta_color="off",
            help="Mô hình Hồi quy Tuyến tính Đa biến có tương tác thị trường (R²_OOS = 0.0340 / 3.40%, R²_Train = 0.0656 / 6.56%, Wald kiểm định tương tác: χ²(10) = 17.91, p = 0.057).",
        )

        # Ô 3: Dự đoán Random Forest Model 4
        st.metric(
            label="3. Dự Đoán Random Forest Model 4 (TV4)",
            value=f"{pred_rf:.1f} điểm",
            delta=f"{err_rf:+.1f} pts (Sai số RF)",
            delta_color="off",
            help="Mô hình Rừng ngẫu nhiên phi tuyến tính (R²_OOS = 0.0686 / 6.86%, MAE_OOS = 29.13, n_estimators=150, max_depth=6, min_samples_leaf=5).",
        )

        # Nhận định so sánh sai số
        st.markdown("---")
        diff_err = abs_ols - abs_rf
        if abs(diff_err) < 0.5:
            st.info("⚖️ **Đánh giá:** Cả OLS Model 3 và Random Forest Model 4 đều cho độ chính xác tương đương trên ca khúc này.")
        elif diff_err < 0:
            st.success(f"🎯 **Đánh giá:** OLS Model 3 dự đoán sát thực tế hơn (sai số tuyệt đối {abs_ols:.1f} pts so với {abs_rf:.1f} pts của RF, chênh lệch {abs(diff_err):.1f} điểm).")
        else:
            st.success(f"🌲 **Đánh giá:** Random Forest Model 4 dự đoán sát thực tế hơn (sai số tuyệt đối {abs_rf:.1f} pts so với {abs_ols:.1f} pts của OLS, chênh lệch {abs(diff_err):.1f} điểm).")

    # 6. Bảng Chi Tiết 6 Thông Số Âm Học Đo Lường
    with st.expander("🔍 Xem Bảng Đo Lường Chi Tiết 6 Chiều Âm Học & Thang Đo Chuẩn Hóa", expanded=False):
        table_rows = []
        for f_def in RADAR_FEATURE_DEFS:
            k = f_def["key"]
            unit = f_def["unit"]
            fmt = f_def["fmt"]
            raw_val = float(sel_row[k])
            scaled_val = float(sel_row[f"{k}_scaled"])
            c_min, c_max = bounds.get(k, (0.0, 1.0))
            med_val = float(showcase_df[k].median())

            table_rows.append({
                "Đặc Trưng Âm Học": f_def["label"],
                "Giá Trị Ca Khúc": f"{fmt.format(raw_val)} {unit}",
                "Trung Vị Mẫu (15 bài)": f"{fmt.format(med_val)} {unit}",
                "Biên Độ Mẫu [Min - Max]": f"[{fmt.format(c_min)} - {fmt.format(c_max)}] {unit}",
                "Điểm Chuẩn Hóa [0, 1]": f"{scaled_val:.3f}",
                "Mô Tả Vật Lý": f_def["desc"],
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # 7. Nhận Định Kinh Tế Lượng Học Thuật (Academic & Econometric Note)
    st.markdown("---")
    st.markdown("### 🎓 Nhận Định Kinh Tế Lượng & Tính Bất Định Của Ca Khúc Hit")
    ols_min = float(showcase_df["pred_ols_m3"].min())
    ols_max = float(showcase_df["pred_ols_m3"].max())
    rf_min = float(showcase_df["pred_rf"].min())
    rf_max = float(showcase_df["pred_rf"].max())

    econometric_markdown = (
        r"📌 **Bối cảnh phương sai giải thích ($R^2_{\text{OOS}} \approx 3.4\% - 6.9\%$):**"
        "\n\n"
        f"1. **Hiện tượng Co cụm về Trung vị (Shrinkage to Mean):** Điểm dự đoán của cả OLS (khoảng ${ols_min:+.1f}$ đến ${ols_max:+.1f}$) "
        f"và Random Forest (khoảng ${rf_min:+.1f}$ đến ${rf_max:+.1f}$) trên 15 bài mẫu đều hội tụ quanh trung vị mẫu ($15.16$). Điều này phản ánh bản chất thống kê lành mạnh: "
        r"khi một tập đặc trưng chỉ giải thích một phần phương sai ($R^2_{\text{OOS, OLS}}=0.0340$, $R^2_{\text{OOS, RF}}=0.0686$), các mô hình tối ưu không được phép 'dự đoán cực đoan' "
        "để tránh bùng nổ sai số bình phương.\n\n"
        r"2. **Khoảng cách tại nhóm ca khúc đỉnh cao (Superstar Effect):** Đối với các ca khúc Tier 1 đạt điểm cực đại ($\ge 80$), "
        r"khoảng cách sai số lớn giữa dự đoán (~$10-20$) và thực tế ($80-99$) phản ánh giới hạn của các đặc trưng nội tại bài hát: "
        r"*các yếu tố ngoài mô hình (như danh tiếng tích lũy của nghệ sĩ, chiến dịch truyền thông, mức độ viral mạng xã hội, vị trí trên các playlist đề xuất...) "
        r"không được đo lường trực tiếp trong dữ liệu và có thể liên quan mật thiết đến phần lớn phương sai chưa được giải thích ($1 - R^2 \approx 93\% - 96\%).*\n\n"
        r"3. **Kiểm định Paired Cluster Bootstrap ($\Delta\text{MAE} = +0.6505$, $95\%$ CI $[-0.8057, +2.0268]$):** "
        r"Do khoảng tin cậy 95% của chênh lệch sai số tuyệt đối chứa giá trị 0, sự vượt trội của Random Forest so với OLS Model 3 "
        r"chưa đạt ý nghĩa thống kê ở mức $\alpha = 0.05$ trên toàn bộ 3 thị trường, khẳng định mối quan hệ giữa đặc trưng âm học và thành tích BXH mang tính chất tương quan biên (marginal association) hơn là các mẫu hình phi tuyến tính phức tạp."
    )
    st.info(econometric_markdown)

    # 8. Bảng Tổng Hợp Toàn Bộ 15 Bài Mẫu
    with st.expander("📋 Xem Danh Sách Tổng Hợp Toàn Bộ 15 Ca Khúc Showcase", expanded=False):
        summary_cols = [
            "market",
            "title",
            "artist",
            "chart_score_actual",
            "pred_ols_m3",
            "pred_rf",
            "abs_error_ols",
            "abs_error_rf",
            "tier_name",
        ]
        display_summary = showcase_df[summary_cols].copy()
        display_summary.columns = [
            "Thị Trường",
            "Tên Bài Hát",
            "Nghệ Sĩ",
            "Điểm Thực Tế",
            "Dự Đoán OLS",
            "Dự Đoán RF",
            "|Sai Số OLS|",
            "|Sai Số RF|",
            "Phân Tầng",
        ]
        display_summary["Điểm Thực Tế"] = display_summary["Điểm Thực Tế"].round(1)
        display_summary["Dự Đoán OLS"] = display_summary["Dự Đoán OLS"].round(1)
        display_summary["Dự Đoán RF"] = display_summary["Dự Đoán RF"].round(1)
        display_summary["|Sai Số OLS|"] = display_summary["|Sai Số OLS|"].round(1)
        display_summary["|Sai Số RF|"] = display_summary["|Sai Số RF|"].round(1)

        st.dataframe(display_summary, use_container_width=True, hide_index=True)
