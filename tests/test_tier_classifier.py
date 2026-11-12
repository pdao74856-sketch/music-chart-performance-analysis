"""
tests/test_tier_classifier.py
Unit tests for the unified acoustic pattern and chart tier classifier engine.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from src.utils.tier_classifier import (
    classify_chart_tier,
    estimate_percentile_in_distribution,
    get_acoustic_fit_profile,
    get_gap_analysis,
    load_acoustic_thresholds,
    EMPIRICAL_CHART_THRESHOLDS,
)


def test_classify_chart_tier_boundaries():
    # Tier 1: >= Q3 (37.58)
    t1_code, _ = classify_chart_tier(37.58)
    assert t1_code == "Tier 1"
    t1_high, _ = classify_chart_tier(99.0)
    assert t1_high == "Tier 1"

    # Tier 2: >= Median (15.16) and < Q3
    t2_code, _ = classify_chart_tier(15.16)
    assert t2_code == "Tier 2"
    t2_mid, _ = classify_chart_tier(30.0)
    assert t2_mid == "Tier 2"

    # Tier 3: >= Q1 (-11.93) and < Median
    t3_code, _ = classify_chart_tier(-11.93)
    assert t3_code == "Tier 3"
    t3_mid, _ = classify_chart_tier(0.0)
    assert t3_code == "Tier 3"

    # Tier 4: < Q1 (-11.93)
    t4_code, _ = classify_chart_tier(-12.0)
    assert t4_code == "Tier 4"
    t4_low, _ = classify_chart_tier(-94.0)
    assert t4_low == "Tier 4"


def test_load_acoustic_thresholds():
    thresholds = load_acoustic_thresholds()
    assert "metadata" in thresholds
    assert "empirical_hit_screener" in thresholds
    assert "markets" in thresholds
    assert "VN" in thresholds["markets"]
    assert "US" in thresholds["markets"]
    assert "KR" in thresholds["markets"]


def test_get_acoustic_fit_profile():
    # Test high score in VN market
    profile_vn = get_acoustic_fit_profile(18.0, market="VN", model_type="rf")
    assert profile_vn["tier_code"] in ["Tier 1", "Tier 2"]
    assert profile_vn["is_top22"] is True
    assert 0.0 <= profile_vn["percentile"] <= 100.0
    assert 0.0 <= profile_vn["top_percent"] <= 100.0
    assert "screener_stats" in profile_vn

    # Test low score in US market
    profile_us_low = get_acoustic_fit_profile(-20.0, market="US", model_type="ols")
    assert profile_us_low["tier_code"] in ["Tier 3", "Tier 4"]
    assert profile_us_low["is_top22"] is False


def test_get_gap_analysis_scenarios():
    # Scenario 1: Consensus Hit
    gap1 = get_gap_analysis("Tier 1", "Tier 1", 90.0, 18.0, market="VN")
    assert "Đồng thuận" in gap1["category"]

    # Scenario 2: Superstar Effect
    gap2 = get_gap_analysis("Tier 1", "Tier 3", 82.0, 14.0, market="US")
    assert "Siêu sao" in gap2["category"]
    assert "Salganik" in gap2["analysis"]

    # Scenario 3: Marketing Deficit
    gap3 = get_gap_analysis("Tier 4", "Tier 1", -94.0, 19.0, market="US")
    assert "Thiếu hụt" in gap3["category"]
    assert "Pachet" in gap3["analysis"]
