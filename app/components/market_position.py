"""Show the position of a Random Forest score within a market."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.utils.tier_classifier import MARKET_NAMES, get_acoustic_fit_profile, load_acoustic_thresholds


@lru_cache(maxsize=1)
def load_reference_scores() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "models" / "test_predictions.csv"
    return pd.read_csv(path, usecols=["market", "pred_rf", "chart_score_actual", "title"]).dropna(subset=["market", "pred_rf"])


def render_prediction_context(st) -> None:
    """Keep test-set context available without competing with the result."""
    with st.expander("Giới hạn dự đoán và kết quả kiểm thử"):
        st.markdown(
            "Sai số tuyệt đối trung bình trên tập kiểm thử khoảng **29–30 điểm**. "
            "Top % phản ánh vị trí của điểm dự đoán trong mẫu tham chiếu. "
            "Tập dữ liệu gồm các ca khúc từng lọt bảng xếp hạng Spotify."
        )
        thresholds = load_acoustic_thresholds()
        metadata = thresholds["metadata"]
        rf = thresholds.get("empirical_hit_screener", {}).get("rf", {})
        if "n_top22" in rf and "tier1_count_top22" in rf and "n_total" in metadata:
            selected_rate = f"{rf['tier1_rate_top22']:.1f}".replace(".", ",")
            other_rate = f"{rf['tier1_rate_rest']:.1f}".replace(".", ",")
            cutoff = f"{metadata['empirical_hit_threshold']:.2f}".replace(".", ",")
            st.markdown(
                f"Trong phân tích thăm dò gộp ba thị trường ({metadata['n_total']} cặp bài hát–thị trường), "
                f"**{rf['tier1_count_top22']}/{rf['n_top22']} cặp ({selected_rate}%)** "
                f"trong nhóm điểm Random Forest cao có ChartScore từ {cutoff} trở lên. "
                f"Tỷ lệ tương ứng ở nhóm còn lại là {other_rate}%."
            )
            st.caption("Các tỷ lệ kiểm thử mô tả cả nhóm được chọn.")


def render_market_position(st, score: float, market: str, key: str, *, explain: bool = False) -> None:
    profile = get_acoustic_fit_profile(score, market=market, model_type="rf")
    top_percent = f"{profile['top_percent']:.0f}"
    st.metric(MARKET_NAMES[market], f"Top ≈{top_percent}%", border=True)
    if explain:
        st.info(
            f"**Cách đọc kết quả**  \n"
            f"Điểm Random Forest của bài này nằm trong khoảng **{top_percent}% cao nhất** "
            f"so với các điểm dự đoán trong mẫu tham chiếu tại **{MARKET_NAMES[market]}**."
        )
    st.caption(f"Điểm Random Forest: {score:+.1f}")
    try:
        reference = load_reference_scores()
    except (OSError, ValueError):
        st.caption("Biểu đồ điểm tham chiếu chưa khả dụng.")
        return

    scores = reference.loc[reference["market"] == market, "pred_rf"]
    if scores.empty:
        return

    figure = go.Figure(go.Histogram(
        x=scores,
        histnorm="percent",
        xbins={"start": -100, "end": 100, "size": 5},
        marker_color="#8dbaff",
        hovertemplate="Điểm RF: %{x}<br>Tỷ lệ: %{y:.1f}%<extra></extra>",
    ))
    figure.add_vline(x=score, line_color="#174db7", line_width=3)
    lower = min(-35, float(reference["pred_rf"].min()) - 3, score - 3)
    upper = max(45, float(reference["pred_rf"].max()) + 3, score + 3)
    figure.update_layout(
        template="plotly_white", height=175, bargap=0.08,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        xaxis={"range": [lower, upper], "title": "Điểm RF", "fixedrange": True},
        yaxis={"visible": False, "fixedrange": True},
    )
    st.plotly_chart(figure, key=key, config={"displayModeBar": False})
