"""
tests/test_tab_showcase_headless.py
Automated Headless Test Suite for Tab 1 (Showcase Library & Radar Chart)
Project: What Makes a Hit Song? (Music Chart Performance Analysis 2021-2026)
Owner: Member 2 (Audio DSP & Librosa Feature Engineering Lead)

Test Coverage:
1. Dataset loading & integrity: Exactly 15 curated tracks across 3 markets (5 VN, 5 US, 5 KR).
2. Out-of-sample test split compliance: All 15 track IDs exist in models/test_track_ids.json.
3. Multi-tier representation: Spanning all 4 empirical tiers (Tier 1 to Tier 4).
4. Min-Max Scaling precision: Scaled values strictly within [0.0, 1.0], with special emphasis on negative loudness_proxy (dBFS).
5. Radar Chart figure topology: Plotly Graph Objects Figure with 2 traces, closed 6-dimension polygons (7 vertices).
6. Dual-model metrics consistency: Finite predictions for OLS Model 3 and RF Model 4, accurate error computations.
7. Graceful degradation: Handles headless environments without UI display errors.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import pytest

from app.components.tab_showcase import (
    RADAR_FEATURE_DEFS,
    RADAR_KEYS,
    RADAR_LABELS,
    SHOWCASE_SPECIFICATIONS,
    TIER_THRESHOLDS,
    classify_chart_tier,
    compute_min_max_scaling,
    create_radar_chart,
    load_showcase_dataset,
    render,
)


# ==============================================================================
# TEST 1: SHOWCASE DATASET INTEGRITY & 15 TRACKS CURATION
# ==============================================================================

def test_showcase_dataset_loads_exactly_15_tracks():
    """Kiểm tra tập bài mẫu showcase nạp đủ 15 ca khúc và phân bố đều 5 bài/thị trường."""
    df, bounds = load_showcase_dataset()

    assert len(df) == 15, f"Kỳ vọng đúng 15 bài mẫu nhưng thực tế nạp {len(df)} bài."

    market_counts = df["market"].value_counts().to_dict()
    assert market_counts.get("VN") == 5, f"Kỳ vọng 5 bài VN, thực tế: {market_counts.get('VN')}"
    assert market_counts.get("US") == 5, f"Kỳ vọng 5 bài US, thực tế: {market_counts.get('US')}"
    assert market_counts.get("KR") == 5, f"Kỳ vọng 5 bài KR, thực tế: {market_counts.get('KR')}"


def test_showcase_tracks_exist_in_test_track_ids():
    """Kiểm tra 100% ID của 15 bài mẫu đều thuộc tập kiểm định ngoài mẫu models/test_track_ids.json."""
    test_ids_path = ROOT_DIR / "models" / "test_track_ids.json"
    assert test_ids_path.exists(), f"Không tìm thấy file: {test_ids_path}"

    with open(test_ids_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_ids_set = set(data["track_ids"])

    df, _ = load_showcase_dataset()
    for tid in df["spotify_track_id"]:
        assert tid in test_ids_set, f"Track ID {tid} không nằm trong out-of-sample test split!"


# ==============================================================================
# TEST 2: EMPIRICAL TIER DISTRIBUTION
# ==============================================================================

def test_showcase_tier_distribution():
    """Kiểm tra tập 15 bài bao quát cả 4 phân tầng thành tích từ siêu hit đến đáy BXH."""
    df, _ = load_showcase_dataset()

    tiers_found = set()
    for score in df["chart_score_actual"]:
        code, _ = classify_chart_tier(score)
        tiers_found.add(code)

    assert "Tier 1" in tiers_found, "Thiếu đại diện Tier 1 (Viral Hit)!"
    assert "Tier 2" in tiers_found, "Thiếu đại diện Tier 2 (Solid Performer)!"
    assert "Tier 3" in tiers_found, "Thiếu đại diện Tier 3 (Moderate/Lower)!"
    assert "Tier 4" in tiers_found, "Thiếu đại diện Tier 4 (Deep Chart)!"


def test_classify_chart_tier_boundaries():
    """Kiểm tra logic phân tầng theo các ngưỡng phân vị Q1, Median, Q3."""
    q1 = TIER_THRESHOLDS["Q1"]
    med = TIER_THRESHOLDS["Median"]
    q3 = TIER_THRESHOLDS["Q3"]

    assert classify_chart_tier(q3 + 10)[0] == "Tier 1"
    assert classify_chart_tier(q3)[0] == "Tier 1"
    assert classify_chart_tier(med + 1)[0] == "Tier 2"
    assert classify_chart_tier(med)[0] == "Tier 2"
    assert classify_chart_tier(q1 + 1)[0] == "Tier 3"
    assert classify_chart_tier(q1)[0] == "Tier 3"
    assert classify_chart_tier(q1 - 10)[0] == "Tier 4"


# ==============================================================================
# TEST 3: MIN-MAX SCALING ON 6 ACOUSTIC FEATURES (INCL. LOUDNESS)
# ==============================================================================

def test_min_max_scaling_bounds_and_loudness_handling():
    """
    Kiểm tra Min-Max Scaling trên 6 trục:
    - Mọi giá trị sau chuẩn hóa đều nằm trong [0.0, 1.0].
    - Giá trị nhỏ nhất được gán 0.0, giá trị lớn nhất được gán 1.0.
    - loudness_proxy mang giá trị âm (-25 đến -6 dBFS) co giãn chuẩn xác, không bị lỗi toán học.
    """
    df, bounds = load_showcase_dataset()

    for key in RADAR_KEYS:
        scaled_col = f"{key}_scaled"
        assert scaled_col in df.columns, f"Thiếu cột scaled {scaled_col}"

        vals = df[scaled_col].values
        assert np.all(vals >= -1e-6), f"Tồn tại giá trị âm trong {scaled_col}: {vals.min()}"
        assert np.all(vals <= 1.0 + 1e-6), f"Tồn tại giá trị vượt 1.0 trong {scaled_col}: {vals.max()}"
        assert np.isclose(vals.min(), 0.0, atol=1e-5), f"Min của {scaled_col} phải bằng 0.0"
        assert np.isclose(vals.max(), 1.0, atol=1e-5), f"Max của {scaled_col} phải bằng 1.0"

    # Chi tiết kiểm tra loudness_proxy
    c_min, c_max = bounds["loudness_proxy"]
    assert c_min < c_max, "Min loudness phải nhỏ hơn Max loudness"
    assert c_min < 0.0 and c_max < 0.0, "Loudness proxy phải mang giá trị âm dBFS"

    loud_raw = df["loudness_proxy"].values
    loud_scaled = df["loudness_proxy_scaled"].values
    expected_scaled = (loud_raw - c_min) / (c_max - c_min)
    np.testing.assert_allclose(loud_scaled, expected_scaled, atol=1e-6)


# ==============================================================================
# TEST 4: RADAR CHART TOPOLOGY VIA PLOTLY
# ==============================================================================

def test_radar_chart_figure_generation():
    """Kiểm tra hàm create_radar_chart tạo biểu đồ Radar Chart hợp lệ với 2 traces khép kín."""
    df, bounds = load_showcase_dataset()
    selected_row = df.iloc[0]

    fig = create_radar_chart(selected_row, df, bounds)
    assert fig is not None, "Figure không được là None khi plotly khả dụng."

    # Kiểm tra 2 traces: Trace 0 là Median, Trace 1 là Selected Track
    assert len(fig.data) == 2, f"Kỳ vọng 2 traces nhưng thực tế có {len(fig.data)}"

    median_trace = fig.data[0]
    track_trace = fig.data[1]

    assert "Trung vị" in median_trace.name
    assert selected_row["title"] in track_trace.name or selected_row["market"] in track_trace.name

    # Đa giác khép kín: 6 chiều -> 7 điểm (điểm đầu lặp lại ở cuối)
    assert len(median_trace.r) == 7, f"Đa giác median phải có 7 điểm khép kín, thực tế: {len(median_trace.r)}"
    assert len(track_trace.r) == 7, f"Đa giác track phải có 7 điểm khép kín, thực tế: {len(track_trace.r)}"
    assert median_trace.r[0] == median_trace.r[-1], "Điểm đầu và cuối trace median phải trùng nhau"
    assert track_trace.r[0] == track_trace.r[-1], "Điểm đầu và cuối trace track phải trùng nhau"

    # Kiểm tra cấu hình radial axis
    radial = fig.layout.polar.radialaxis
    assert radial.range == (0, 1.05) or list(radial.range) == [0, 1.05]


# ==============================================================================
# TEST 5: DUAL-MODEL PREDICTIONS & ERROR METRICS
# ==============================================================================

def test_dual_model_predictions_consistency():
    """Kiểm tra tính hợp lệ và hữu hạn của các dự đoán OLS Model 3 và RF Model 4."""
    df, _ = load_showcase_dataset()

    for _, r in df.iterrows():
        act = r["chart_score_actual"]
        ols = r["pred_ols_m3"]
        rf = r["pred_rf"]

        assert np.isfinite(act), f"Điểm thực tế không hợp lệ: {act}"
        assert np.isfinite(ols), f"Dự đoán OLS không hợp lệ: {ols}"
        assert np.isfinite(rf), f"Dự đoán RF không hợp lệ: {rf}"

        # Kiểm tra tính chính xác của sai số
        assert np.isclose(r["error_ols"], ols - act, atol=1e-5)
        assert np.isclose(r["error_rf"], rf - act, atol=1e-5)
        assert np.isclose(r["abs_error_ols"], abs(ols - act), atol=1e-5)
        assert np.isclose(r["abs_error_rf"], abs(rf - act), atol=1e-5)


# ==============================================================================
# TEST 6: STREAMLIT RENDER EXECUTION HEADLESS
# ==============================================================================

def test_render_executes_without_exceptions():
    """Kiểm tra hàm render() có thể chạy trơn tru trong môi trường headless mock Streamlit."""
    mock_st = MagicMock()

    # Mock columns to return a list of dummy column contexts
    def dummy_columns(spec):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [MagicMock() for _ in range(n)]

    mock_st.columns.side_effect = dummy_columns
    mock_st.expander.return_value.__enter__.return_value = MagicMock()
    mock_st.expander.return_value.__exit__.return_value = None

    # Mock selectbox to return the first option from options list
    mock_st.selectbox.side_effect = lambda label, options, **kwargs: options[0]

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        # Call render
        render()

    # Assert key functions were invoked
    assert mock_st.header.called, "st.header phải được gọi"
    assert mock_st.selectbox.called, "st.selectbox phải được gọi"
    assert mock_st.metric.called, "st.metric phải được gọi cho 3 chỉ số song song"
