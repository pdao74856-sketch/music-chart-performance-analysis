#!/usr/bin/env python3
"""
scripts/compute_acoustic_fit_thresholds.py
Calculate and freeze acoustic pattern quantiles & hit screener statistics
across markets (VN, US, KR, ALL) from the out-of-sample test set (n=410).

Outputs:
  - models/acoustic_fit_thresholds.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_PRED_PATH = ROOT_DIR / "models" / "test_predictions.csv"
OUTPUT_JSON_PATH = ROOT_DIR / "models" / "acoustic_fit_thresholds.json"
EMPIRICAL_Q3_THRESHOLD = 37.58  # Hit tier 1 cutoff


def compute_thresholds():
    if not TEST_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_PRED_PATH}")

    df = pd.read_csv(TEST_PRED_PATH)
    n_total = len(df)
    df["is_tier1_actual"] = df["chart_score_actual"] >= EMPIRICAL_Q3_THRESHOLD

    # Top 22% quantile definition (93 tracks / 410 = ~22.68%)
    top22_quantile = 1.0 - (93.0 / n_total)

    result = {
        "metadata": {
            "source": "models/test_predictions.csv (Out-of-sample Test Set, n=410)",
            "n_total": n_total,
            "empirical_hit_threshold": EMPIRICAL_Q3_THRESHOLD,
            "top22_quantile": round(top22_quantile, 4),
            "baseline_hit_rate": round(float(df["is_tier1_actual"].mean()) * 100, 2),
        },
        "empirical_hit_screener": {},
        "markets": {},
    }

    # 1. Compute empirical screener performance for OLS & RF
    for model_key, pred_col in [("ols", "pred_ols_m3"), ("rf", "pred_rf")]:
        top_mask = df.groupby("market")[pred_col].transform(
            lambda s: s >= s.quantile(top22_quantile)
        )
        top = df[top_mask]
        rest = df[~top_mask]

        n_top = len(top)
        n_rest = len(rest)
        tier1_top = int(top["is_tier1_actual"].sum())
        tier1_rest = int(rest["is_tier1_actual"].sum())

        tier1_top_rate = float(top["is_tier1_actual"].mean()) * 100
        tier1_rest_rate = float(rest["is_tier1_actual"].mean()) * 100

        mean_top = float(top["chart_score_actual"].mean())
        mean_rest = float(rest["chart_score_actual"].mean())
        diff_mean = mean_top - mean_rest

        s1, s2 = float(top["chart_score_actual"].var()), float(rest["chart_score_actual"].var())
        se_diff = float(np.sqrt(s1 / n_top + s2 / n_rest))
        df_welch = (s1 / n_top + s2 / n_rest) ** 2 / (
            (s1 / n_top) ** 2 / (n_top - 1) + (s2 / n_rest) ** 2 / (n_rest - 1)
        )
        t_crit = float(stats.t.ppf(0.975, df_welch))
        ci_low = round(diff_mean - t_crit * se_diff, 2)
        ci_high = round(diff_mean + t_crit * se_diff, 2)

        _, p_fisher = stats.fisher_exact(
            [[tier1_top, n_top - tier1_top], [tier1_rest, n_rest - tier1_rest]]
        )

        p_pool = (tier1_top + tier1_rest) / (n_top + n_rest)
        se_prop = np.sqrt(p_pool * (1 - p_pool) * (1 / n_top + 1 / n_rest))
        z_stat = (tier1_top_rate / 100 - tier1_rest_rate / 100) / se_prop
        p_z = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        odds_ratio = (
            (tier1_top / (n_top - tier1_top)) / (tier1_rest / (n_rest - tier1_rest))
            if (n_top - tier1_top) > 0 and tier1_rest > 0
            else 0.0
        )

        result["empirical_hit_screener"][model_key] = {
            "n_top22": n_top,
            "n_rest": n_rest,
            "tier1_count_top22": tier1_top,
            "tier1_rate_top22": round(tier1_top_rate, 2),
            "tier1_rate_rest": round(tier1_rest_rate, 2),
            "odds_ratio": round(odds_ratio, 2),
            "p_value_fisher": float(p_fisher),
            "p_value_z_test": float(p_z),
            "mean_actual_top22": round(mean_top, 2),
            "mean_actual_rest": round(mean_rest, 2),
            "diff_mean": round(diff_mean, 2),
            "ci_95": [ci_low, ci_high],
        }

    # 2. Compute market-specific distribution percentiles
    for mkt in ["VN", "US", "KR", "ALL"]:
        sub = df if mkt == "ALL" else df[df["market"] == mkt]
        mkt_dict = {"n_tracks": len(sub)}

        for model_key, pred_col in [("ols", "pred_ols_m3"), ("rf", "pred_rf")]:
            vals = sub[pred_col]
            mkt_dict[model_key] = {
                "min": round(float(vals.min()), 2),
                "q1": round(float(vals.quantile(0.25)), 2),
                "median": round(float(vals.quantile(0.50)), 2),
                "q3": round(float(vals.quantile(0.75)), 2),
                "p78": round(float(vals.quantile(top22_quantile)), 2),
                "p90": round(float(vals.quantile(0.90)), 2),
                "max": round(float(vals.max()), 2),
                "mean": round(float(vals.mean()), 2),
                "std": round(float(vals.std()), 2),
            }

        result["markets"][mkt] = mkt_dict

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {OUTPUT_JSON_PATH}")
    return result


if __name__ == "__main__":
    compute_thresholds()
