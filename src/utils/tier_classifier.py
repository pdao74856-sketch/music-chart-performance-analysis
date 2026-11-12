#!/usr/bin/env python3
"""
src/utils/tier_classifier.py
Unified Acoustic Pattern & Chart Tier Classification Engine
Project: What Makes a Hit Song? (Music Chart Performance Analysis)

Responsibilities:
  - Classify actual chart scores against empirical thresholds (Q1=-11.93, Med=15.16, Q3=37.58).
  - Benchmark predicted scores against market-specific frozen quantiles (models/acoustic_fit_thresholds.json).
  - Calculate acoustic pattern percentiles and tiers (Tier 1 to Tier 4).
  - Provide empirical hit screener statistics and citations (Herremans 2019, Salganik 2006, Pachet 2008).
  - Deliver acoustic vs. commercial gap analysis (Superstar effect vs. Marketing deficit).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = ROOT_DIR / "models" / "acoustic_fit_thresholds.json"

# Empirical Chart Score Quantiles from Clean Dataset (n=2,098)
EMPIRICAL_CHART_THRESHOLDS = {
    "Q1": -11.93,
    "Median": 15.16,
    "Q3": 37.58,
}

MARKET_NAMES = {
    "VN": "Việt Nam",
    "US": "Hoa Kỳ",
    "KR": "Hàn Quốc",
    "ALL": "Toàn bộ thị trường",
}


def load_acoustic_thresholds() -> Dict[str, Any]:
    """Loads frozen acoustic fit thresholds and screener stats from JSON."""
    if THRESHOLDS_PATH.exists():
        try:
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback structure if JSON is temporarily unreachable
    return {
        "metadata": {
            "baseline_hit_rate": 25.12,
            "empirical_hit_threshold": 37.58,
            "top22_quantile": 0.7732,
        },
        "empirical_hit_screener": {
            "ols": {
                "tier1_rate_top22": 44.09,
                "tier1_rate_rest": 19.56,
                "odds_ratio": 3.24,
                "diff_mean": 13.34,
                "ci_95": [4.06, 22.63],
            },
            "rf": {
                "tier1_rate_top22": 35.48,
                "tier1_rate_rest": 22.08,
                "odds_ratio": 1.94,
                "diff_mean": 17.78,
                "ci_95": [8.93, 26.63],
            },
        },
        "markets": {
            "VN": {
                "ols": {"min": -11.2, "q1": 4.98, "median": 8.54, "q3": 12.49, "p78": 12.97, "p90": 15.37, "max": 19.69},
                "rf": {"min": -30.09, "q1": 1.08, "median": 12.69, "q3": 15.39, "p78": 15.76, "p90": 18.85, "max": 38.01},
            },
            "US": {
                "ols": {"min": -32.28, "q1": 9.16, "median": 14.71, "q3": 22.2, "p78": 22.5, "p90": 27.57, "max": 65.52},
                "rf": {"min": -32.25, "q1": 10.76, "median": 14.48, "q3": 17.29, "p78": 18.49, "p90": 26.67, "max": 42.32},
            },
            "KR": {
                "ols": {"min": -15.03, "q1": 3.88, "median": 10.1, "q3": 14.51, "p78": 15.49, "p90": 19.64, "max": 32.73},
                "rf": {"min": -33.43, "q1": 9.74, "median": 13.13, "q3": 16.71, "p78": 16.95, "p90": 23.45, "max": 36.25},
            },
            "ALL": {
                "ols": {"min": -32.28, "q1": 5.75, "median": 11.23, "q3": 16.73, "p78": 17.67, "p90": 23.72, "max": 65.52},
                "rf": {"min": -33.43, "q1": 9.27, "median": 13.37, "q3": 16.68, "p78": 16.99, "p90": 24.48, "max": 42.32},
            },
        },
    }


def classify_chart_tier(score: float) -> Tuple[str, str]:
    """
    Phân loại thứ hạng thành tích thực tế trên BXH dựa trên phân vị thực nghiệm.

    Returns:
        (tier_code, tier_desc) e.g. ("Tier 1", "Viral Hit (Top 25% Đỉnh BXH)")
    """
    if score >= EMPIRICAL_CHART_THRESHOLDS["Q3"]:
        return "Tier 1", "Viral Hit (Top 25% Đỉnh BXH)"
    elif score >= EMPIRICAL_CHART_THRESHOLDS["Median"]:
        return "Tier 2", "Solid Performer (Trên Trung vị)"
    elif score >= EMPIRICAL_CHART_THRESHOLDS["Q1"]:
        return "Tier 3", "Moderate / Lower (Dưới Trung vị)"
    else:
        return "Tier 4", "Deep Chart (Nhóm Đáy BXH)"


def estimate_percentile_in_distribution(score: float, quantiles: Dict[str, float]) -> float:
    """
    Ước lượng bách phân vị [0.0, 100.0] của điểm dự đoán thông qua nội suy tuyến tính từng đoạn
    trên các mốc phân vị thực nghiệm: min (0%), q1 (25%), median (50%), q3 (75%), p90 (90%), max (100%).
    """
    c_min = quantiles.get("min", -30.0)
    q1 = quantiles.get("q1", 5.0)
    med = quantiles.get("median", 12.0)
    q3 = quantiles.get("q3", 16.0)
    p90 = quantiles.get("p90", 22.0)
    c_max = quantiles.get("max", 40.0)

    if score <= c_min:
        return 1.0
    if score >= c_max:
        return 99.0

    # Piecewise interpolation
    if score < q1:
        span = q1 - c_min
        frac = (score - c_min) / span if span > 1e-6 else 0.5
        return round(1.0 + frac * 24.0, 1)
    elif score < med:
        span = med - q1
        frac = (score - q1) / span if span > 1e-6 else 0.5
        return round(25.0 + frac * 25.0, 1)
    elif score < q3:
        span = q3 - med
        frac = (score - med) / span if span > 1e-6 else 0.5
        return round(50.0 + frac * 25.0, 1)
    elif score < p90:
        span = p90 - q3
        frac = (score - q3) / span if span > 1e-6 else 0.5
        return round(75.0 + frac * 15.0, 1)
    else:
        span = c_max - p90
        frac = (score - p90) / span if span > 1e-6 else 0.5
        return round(90.0 + frac * 9.0, 1)


def get_acoustic_fit_profile(
    score: float,
    market: str = "VN",
    model_type: str = "rf",
) -> Dict[str, Any]:
    """
    Tính toán hồ sơ đối chuẩn mẫu hình âm học theo thị trường.

    Args:
        score: Điểm dự đoán từ mô hình (OLS Model 3 hoặc RF Model 4).
        market: Mã thị trường ('VN', 'US', 'KR', 'ALL').
        model_type: Loại mô hình dùng tham chiếu ('rf' hoặc 'ols').

    Returns:
        Dict chứa:
          - percentile: Bách phân vị ước lượng [0.0 - 100.0]
          - top_percent: Top X% dẫn đầu (100 - percentile)
          - tier_code: 'Tier 1' đến 'Tier 4'
          - tier_title: Tên chuẩn hóa của phân tầng âm học
          - badge_color: Icon biểu trưng
          - is_top22: True nếu ca khúc lọt vào nhóm Top 22% điểm âm học
          - empirical_screener: Thống kê kiểm chứng thực nghiệm tương ứng
          - description: Nhận định mô tả mẫu hình âm học
    """
    thresholds = load_acoustic_thresholds()
    mkt_key = market if market in thresholds.get("markets", {}) else "ALL"
    mkt_data = thresholds.get("markets", {}).get(mkt_key, {})
    mkt_name = MARKET_NAMES.get(mkt_key, mkt_key)

    model_key = "rf" if model_type.lower() in ("rf", "random_forest") else "ols"
    q_dict = mkt_data.get(model_key, thresholds.get("markets", {}).get("ALL", {}).get(model_key, {}))

    percentile = estimate_percentile_in_distribution(score, q_dict)
    top_percent = max(1.0, round(100.0 - percentile, 1))

    # Top 22% screener check
    p78_val = q_dict.get("p78", q_dict.get("q3", 16.0))
    is_top22 = bool(score >= p78_val)

    # 4 Acoustic Pattern Tiers
    q3_val = q_dict.get("q3", 16.0)
    med_val = q_dict.get("median", 12.0)
    q1_val = q_dict.get("q1", 5.0)

    screener_stats = thresholds.get("empirical_hit_screener", {}).get(model_key, {})

    if score >= q3_val:
        tier_code = "Tier 1"
        tier_title = "Mẫu hình Âm học Dẫn đầu (High Acoustic Alignment)"
        badge_color = "🏆"
        description = (
            f"Cấu trúc âm học của ca khúc nằm trong nhóm dẫn đầu ({top_percent:.0f}% điểm cao nhất) "
            f"so với các ca khúc tại thị trường {mkt_name}."
        )
    elif score >= med_val:
        tier_code = "Tier 2"
        tier_title = "Mẫu hình Âm học Chuẩn Đại chúng (Mainstream Fit)"
        badge_color = "⭐"
        description = (
            f"Cấu trúc âm học thuộc nhóm phổ thông, an toàn, trên trung vị thị trường {mkt_name} "
            f"(vị trí {top_percent:.0f}%). Mẫu hình âm thanh dễ tiếp cận với đa số người nghe."
        )
    elif score >= q1_val:
        tier_code = "Tier 3"
        tier_title = "Mẫu hình Âm học Phân hóa / Thể nghiệm (Niche / Distinctive)"
        badge_color = "📈"
        description = (
            f"Cấu trúc âm học mang tính thể nghiệm hoặc kén người nghe trên BXH đại chúng {mkt_name} "
            f"(dưới trung vị, vị trí {top_percent:.0f}%)."
        )
    else:
        tier_code = "Tier 4"
        tier_title = "Mẫu hình Âm học Lệch chuẩn Thị trường (Divergent Pattern)"
        badge_color = "📉"
        description = (
            f"Cấu trúc âm học lệch xa khỏi các khuôn mẫu âm nhạc thịnh hành trên BXH {mkt_name} "
            f"(nhóm đáy {top_percent:.0f}%)."
        )

    return {
        "score": score,
        "market": mkt_key,
        "market_name": mkt_name,
        "model_type": model_key,
        "percentile": percentile,
        "top_percent": top_percent,
        "tier_code": tier_code,
        "tier_title": tier_title,
        "badge_color": badge_color,
        "is_top22": is_top22,
        "screener_stats": screener_stats,
        "description": description,
    }


def get_gap_analysis(
    actual_tier: str,
    acoustic_tier: str,
    actual_score: float,
    pred_score: float,
    market: str = "VN",
) -> Dict[str, str]:
    """
    Phân tích độ lệch giữa thành tích thương mại thực tế và dự báo âm học (Acoustic-Market Gap Analysis).
    Giải thích hiện tượng Siêu sao (Superstar Effect) hoặc Thiếu hụt Truyền thông (Marketing Deficit).
    """
    mkt_name = MARKET_NAMES.get(market, market)

    # Extract tier numbers
    act_num = int(actual_tier[-1]) if actual_tier and actual_tier[-1].isdigit() else 2
    aco_num = int(acoustic_tier[-1]) if acoustic_tier and acoustic_tier[-1].isdigit() else 2

    if act_num == 1 and aco_num == 1:
        category = "Đồng thuận Âm học & Thương mại (Consensus Hit)"
        icon = "🔥"
        analysis = (
            f"Ca khúc sở hữu cấu trúc âm học thuộc nhóm dẫn đầu tại {mkt_name}, "
            "cộng hưởng hoàn hảo cùng độ nhận diện nghệ sĩ và chiến dịch phát hành để trở thành Mega-hit."
        )
    elif act_num == 1 and aco_num > 1:
        category = "Hiệu ứng Siêu sao / Đột phá Truyền thông (Superstar / Marketing-Driven)"
        icon = "🚀"
        analysis = (
            f"Mặc dù cấu trúc âm học chỉ nằm ở dải trung bình ({acoustic_tier}), ca khúc vẫn bứt phá lên đỉnh BXH ({actual_tier}). "
            "Điều này minh chứng cho ảnh hưởng xã hội áp đảo (Salganik et al., 2006): danh tiếng tích lũy của nghệ sĩ, "
            "vị trí trên playlist biên tập và chiến dịch truyền thông đã bù đắp hoàn toàn cho đặc trưng âm học thông thường."
        )
    elif act_num >= 3 and aco_num == 1:
        category = "Âm học Chuẩn nhưng Thiếu hụt Bệ phóng (Marketing-Deficit / Sleeper Track)"
        icon = "💡"
        analysis = (
            f"Ca khúc sở hữu cấu trúc âm học rất tối ưu (lọt {acoustic_tier} dẫn đầu), nhưng thành tích BXH lại rơi vào nhóm thấp ({actual_tier}). "
            "Đây là minh chứng kinh điển cho thấy âm học chỉ là điều kiện cần (Pachet & Roy, 2008): nếu thiếu vắng bệ phóng "
            "truyền thông và mức độ nhận diện thương mại, một tác phẩm dù chuẩn thị hiếu vẫn có thể bị chìm lấp."
        )
    elif act_num == 4 and aco_num == 4:
        category = "Âm học Lệch chuẩn & Không Thâm nhập Đại chúng (Atypical / Niche)"
        icon = "🔍"
        analysis = (
            f"Cả thành tích BXH và mẫu hình âm học đều thuộc nhóm đáy ({actual_tier}). "
            "Ca khúc mang phong cách độc lập hoặc kén người nghe, không được tối ưu cho các thuật toán đề xuất đại chúng."
        )
    else:
        category = "Tương thích Trung vị Thị trường (Mainstream Stability)"
        icon = "⚖️"
        analysis = (
            f"Ca khúc có thành tích thực tế ({actual_tier}) và mẫu hình âm học ({acoustic_tier}) tương đối đồng nhất quanh trung vị thị trường {mkt_name}. "
            "Đặc trưng âm học phản ánh đúng vị thế bám trụ ổn định trên BXH."
        )

    return {
        "category": category,
        "icon": icon,
        "analysis": analysis,
    }
