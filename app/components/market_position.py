"""Show the position of a Random Forest score within a market."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.utils.tier_classifier import MARKET_NAMES, get_acoustic_fit_profile


@lru_cache(maxsize=1)
def load_reference_scores() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "models" / "test_predictions.csv"
    return pd.read_csv(path, usecols=["market", "pred_rf", "chart_score_actual", "title"]).dropna(subset=["market", "pred_rf"])


def render_market_position(st, score: float, market: str, key: str) -> None:
    profile = get_acoustic_fit_profile(score, market=market, model_type="rf")
    st.metric(MARKET_NAMES[market], f"Top ≈{profile['top_percent']:.0f}%", border=True)
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
