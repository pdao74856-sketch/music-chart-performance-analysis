"""
Script: src/modeling/generate_forest_plots.py
Author: Member 1 (Modeling & Analysis Lead)
Purpose:
  1. Update reports/model3_coefficients.csv with holm, q_bh, label_type, evidence_tier.
  2. Perform strict mathematical assertions:
     - Holm >= p_raw, q_bh <= Holm, values <= 1.0, Holm monotonic.
     - Cross-check market slopes with statsmodels res.t_test contrast.
     - Dynamic US-slope loading from model_metadata.json (no hardcoding).
  3. Export reports/forest_main_data.csv and reports/forest_market_slopes_data.csv with unit, n_obs, n_tracks.
  4. Render 2 publication-grade Forest Plots (300 DPI) to reports/figures/ without text collisions:
     - reports/figures/forest_main.png
     - reports/figures/forest_market_slopes.png
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Force UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import multipletests

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.prepare_features import get_split
from src.modeling.ols_utils import fit_ols

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. MATHEMATICAL AUDIT & COEFFICIENT PROCESSING
# -----------------------------------------------------------------------------
def process_coefficients_and_wald():
    # 1. Load metadata dynamically
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Dynamic interaction bases from metadata
    interaction_bases = [f"{c}_std" for c in meta["interaction_cols"]]
    print(f"[Dynamic Load] Interaction features from model_metadata: {interaction_bases}")

    coef_path = REPORTS_DIR / "model3_coefficients.csv"
    wald_path = REPORTS_DIR / "model3_wald_features.csv"
    slopes_path = REPORTS_DIR / "model_market_slopes.csv"

    df_coef = pd.read_csv(coef_path)
    df_wald = pd.read_csv(wald_path)
    df_slopes = pd.read_csv(slopes_path)

    # Dynamic label_type
    df_coef["label_type"] = df_coef["feature"].apply(
        lambda f: "US-slope" if f in interaction_bases else "shared"
    )

    # Main Effects Family (24 predictors in M2b: non-const, non-interaction)
    main_mask = ~df_coef["feature"].str.contains("__x__") & (df_coef["feature"] != "const")
    p_main = df_coef.loc[main_mask, "p_value"].values

    _, holm_main, _, _ = multipletests(p_main, method="holm")
    _, bh_main, _, _ = multipletests(p_main, method="fdr_bh")

    # Strict mathematical assertions for Main Effects
    assert (holm_main >= p_main - 1e-9).all(), "Assertion Failed: Holm must be >= p_raw"
    assert (bh_main <= holm_main + 1e-9).all(), "Assertion Failed: q_bh must be <= Holm"
    assert (holm_main <= 1.0 + 1e-9).all(), "Assertion Failed: Holm must be <= 1.0"
    assert (bh_main <= 1.0 + 1e-9).all(), "Assertion Failed: q_bh must be <= 1.0"
    # Monotonicity: sorted p_main must yield sorted holm_main
    sort_idx = np.argsort(p_main)
    sorted_holm = holm_main[sort_idx]
    assert np.all(np.diff(sorted_holm) >= -1e-9), "Assertion Failed: Holm must be monotonic with p_raw"
    print("[Assert Passed] Main Effects: Holm >= p_raw, q_bh <= Holm, values <= 1.0, Holm monotonic.")

    df_coef.loc[main_mask, "holm"] = holm_main
    df_coef.loc[main_mask, "q_bh"] = bh_main

    # Interaction Family (Wald tests by feature from model3_wald_features.csv)
    p_wald = df_wald["p_raw"].values
    _, wald_bh, _, _ = multipletests(p_wald, method="fdr_bh")
    df_wald["q_bh"] = wald_bh

    # Assertions for Interaction Wald
    assert (df_wald["p_holm"].values >= p_wald - 1e-9).all(), "Assertion Failed: Wald Holm >= p_raw"
    assert (wald_bh <= df_wald["p_holm"].values + 1e-9).all(), "Assertion Failed: Wald q_bh <= Holm"
    assert (df_wald["p_holm"].values <= 1.0 + 1e-9).all(), "Assertion Failed: Wald Holm <= 1.0"
    assert (wald_bh <= 1.0 + 1e-9).all(), "Assertion Failed: Wald q_bh <= 1.0"
    sort_w_idx = np.argsort(p_wald)
    sorted_w_holm = df_wald["p_holm"].values[sort_w_idx]
    assert np.all(np.diff(sorted_w_holm) >= -1e-9), "Assertion Failed: Wald Holm monotonic"
    print("[Assert Passed] Wald Interactions: Holm >= p_raw, q_bh <= Holm, values <= 1.0, Holm monotonic.")

    # Tier helper function
    def get_tier(p_raw, holm, q_bh):
        if pd.isna(holm):
            return "không có bằng chứng"
        if holm < 0.05:
            return "vững"
        elif (p_raw < 0.05 and holm >= 0.05) or (q_bh < 0.10):
            return "gợi ý"
        else:
            return "không có bằng chứng"

    df_wald["evidence_tier"] = [
        get_tier(r["p_raw"], r["p_holm"], r["q_bh"]) for _, r in df_wald.iterrows()
    ]

    # Map Wald results to interaction terms in df_coef
    wald_map = {}
    for _, row in df_wald.iterrows():
        feat = row["feature"]
        wald_map[feat] = {
            "chi2": row["chi2"],
            "p_raw": row["p_raw"],
            "holm": row["p_holm"],
            "q_bh": row["q_bh"],
            "evidence_tier": row["evidence_tier"],
        }

    inter_mask = df_coef["feature"].str.contains("__x__")
    for idx in df_coef[inter_mask].index:
        feat_name = df_coef.loc[idx, "feature"].split("__x__")[0].replace("_std", "")
        if feat_name in wald_map:
            df_coef.loc[idx, "holm"] = wald_map[feat_name]["holm"]
            df_coef.loc[idx, "q_bh"] = wald_map[feat_name]["q_bh"]

    # Intercept const
    df_coef.loc[df_coef["feature"] == "const", "holm"] = np.nan
    df_coef.loc[df_coef["feature"] == "const", "q_bh"] = np.nan

    # Assign evidence_tier to df_coef
    tiers = []
    for _, row in df_coef.iterrows():
        if row["feature"] == "const":
            tiers.append("không có bằng chứng")
        elif "__x__" in row["feature"]:
            feat_name = row["feature"].split("__x__")[0].replace("_std", "")
            tiers.append(wald_map[feat_name]["evidence_tier"])
        else:
            tiers.append(get_tier(row["p_value"], row["holm"], row["q_bh"]))

    df_coef["evidence_tier"] = tiers

    df_coef.to_csv(coef_path, index=False)
    df_wald.to_csv(wald_path, index=False)

    return df_coef, df_wald, df_slopes, wald_map, meta


# -----------------------------------------------------------------------------
# 2. CROSS CHECK MARKET SLOPES WITH T-TEST CONTRAST
# -----------------------------------------------------------------------------
def cross_check_market_slopes(df_slopes, meta):
    print("\n[Cross-Check] Verifying market slopes via statsmodels t_test contrast on M3...")
    split = get_split()
    cfg = split.config
    continuous_cols = [c + "_std" for c in cfg["audio_cols"]] + ["mode"] + [c + "_std" for c in cfg["lyrics_cols"]] + [cfg["year_col"] + "_std", "Market_VN", "Market_KR"]
    inter_cols = [base + "_std__x__" + m for base in cfg["interaction_cols"] for m in ["VN", "KR"]]
    m3_cols = continuous_cols + inter_cols
    res3 = fit_ols(split.X_train[m3_cols], split.y_train, split.ids_train)

    param_names = list(res3.params.index)
    p_len = len(param_names)

    records = []
    for base in meta["interaction_cols"]:
        clean_base = base
        base_col = f"{base}_std"

        # 1. US (contrast = 1 on base_col)
        c_us = np.zeros(p_len)
        c_us[param_names.index(base_col)] = 1.0
        tt_us = res3.t_test(c_us)
        slope_us = float(tt_us.effect[0])
        se_us = float(tt_us.sd[0, 0])
        p_us = float(tt_us.pvalue)

        row_us = df_slopes[(df_slopes["feature"] == clean_base) & (df_slopes["market"] == "US (Baseline)")].iloc[0]
        records.append({
            "feature": clean_base,
            "market": "US",
            "manual_slope": row_us["slope"],
            "ttest_slope": slope_us,
            "manual_se": row_us["se_cluster"],
            "ttest_se": se_us,
            "diff_slope": abs(row_us["slope"] - slope_us),
            "diff_se": abs(row_us["se_cluster"] - se_us),
        })

        for m in ["VN", "KR"]:
            c_m = np.zeros(p_len)
            c_m[param_names.index(base_col)] = 1.0
            c_m[param_names.index(f"{base_col}__x__{m}")] = 1.0
            tt_m = res3.t_test(c_m)
            slope_m = float(tt_m.effect[0])
            se_m = float(tt_m.sd[0, 0])
            p_m = float(tt_m.pvalue)

            row_m = df_slopes[(df_slopes["feature"] == clean_base) & (df_slopes["market"] == m)].iloc[0]
            records.append({
                "feature": clean_base,
                "market": m,
                "manual_slope": row_m["slope"],
                "ttest_slope": slope_m,
                "manual_se": row_m["se_cluster"],
                "ttest_se": se_m,
                "diff_slope": abs(row_m["slope"] - slope_m),
                "diff_se": abs(row_m["se_cluster"] - se_m),
            })

    df_check = pd.DataFrame(records)
    assert (df_check["diff_slope"] < 1e-4).all(), "Assertion Failed: Slopes mismatch between manual and t_test"
    assert (df_check["diff_se"] < 1e-4).all(), "Assertion Failed: SE mismatch between manual and t_test"
    print("[Assert Passed] Market Slopes: 100% agreement between model_market_slopes.csv and t_test contrast (diff < 1e-4).")
    print("\n--- BẢNG KIỂM CHÉO ĐỘ DỐC THEO THỊ TRƯỜNG ---")
    print(df_check.to_string(index=False))

    return split


# -----------------------------------------------------------------------------
# 3. COMPUTE AND PRINT SD BY MARKET FOR 5 INTERACTION FEATURES
# -----------------------------------------------------------------------------
def print_market_sds(split, meta):
    raw_train = split.raw_train
    market_train = split.market_train
    inter_feats = meta["interaction_cols"]

    records = []
    for f in inter_feats:
        s_us = raw_train[market_train == "US"][f]
        s_vn = raw_train[market_train == "VN"][f]
        s_kr = raw_train[market_train == "KR"][f]

        records.append({
            "feature": f,
            "unit": "raw units",
            "SD_Overall": raw_train[f].std(),
            "SD_US": s_us.std(),
            "SD_VN": s_vn.std(),
            "SD_KR": s_kr.std(),
        })
        if f == "word_count":
            s_us_log = np.log1p(s_us.astype(float))
            s_vn_log = np.log1p(s_vn.astype(float))
            s_kr_log = np.log1p(s_kr.astype(float))
            s_all_log = np.log1p(raw_train[f].astype(float))
            records.append({
                "feature": "log1p(word_count)",
                "unit": "log-scale",
                "SD_Overall": s_all_log.std(),
                "SD_US": s_us_log.std(),
                "SD_VN": s_vn_log.std(),
                "SD_KR": s_kr_log.std(),
            })

    df_sd = pd.DataFrame(records)
    print("\n--- SD CỦA 5 BIẾN TƯƠNG TÁC (TẬP TRAIN N=1,685) THEO TỪNG THỊ TRƯỜNG ---")
    print(df_sd.to_string(index=False))


# -----------------------------------------------------------------------------
# 4. REDESIGNED FOREST PLOT (A): MAIN ASSOCIATIONS
# -----------------------------------------------------------------------------
def plot_forest_main(df_coef, split):
    main_mask = ~df_coef["feature"].str.contains("__x__") & (df_coef["feature"] != "const")
    df_plot = df_coef[main_mask].copy()

    feature_meta = {
        # Audio Continuous (14)
        "tempo_std": ("Tempo", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "onset_rate_std": ("Onset Rate", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "tempo_stability_std": ("Tempo Stability", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "energy_std": ("Energy (RMS)", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "rms_variability_std": ("RMS Variability", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "spectral_centroid_std": ("Spectral Centroid", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "chord_change_rate_std": ("Chord Change Rate", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "key_confidence_std": ("Key Confidence", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "chroma_repetition_determinism_std": ("Chroma Repetition Determinism", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "chroma_mean_pattern_duration_std": ("Chroma Mean Pattern Duration", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "chroma_longest_pattern_duration_std": ("Chroma Longest Pattern Duration", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "mfcc_repetition_determinism_std": ("MFCC Repetition Determinism", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "mfcc_mean_pattern_duration_std": ("MFCC Mean Pattern Duration", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "mfcc_longest_pattern_duration_std": ("MFCC Longest Pattern Duration", "Đặc tính âm thanh (Audio - 15 biến)", "SD"),
        "mode": ("Mode (Major=1, Minor=0)", "Đặc tính âm thanh (Audio - 15 biến)", "unit"),
        # Lyrics (6)
        "word_count_std": ("Word Count*", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        "repetition_ratio_std": ("Repetition Ratio", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        "avg_word_length_std": ("Avg Word Length", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        "line_count_std": ("Line Count", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        "chorus_repetition_count_std": ("Chorus Repetition Count", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        "punctuation_ratio_std": ("Punctuation Ratio", "Lời bài hát (Lyrics - 6 biến)", "SD"),
        # Context (3)
        "release_year_std": ("Release Year", "Kiểm soát & Bối cảnh (Context - 3 biến)", "SD"),
        "Market_VN": ("Market: Vietnam vs US", "Kiểm soát & Bối cảnh (Context - 3 biến)", "unit"),
        "Market_KR": ("Market: South Korea vs US", "Kiểm soát & Bối cảnh (Context - 3 biến)", "unit"),
    }

    df_plot["display_name"] = df_plot["feature"].map(lambda f: feature_meta.get(f, (f, "Other", "SD"))[0])
    df_plot["group"] = df_plot["feature"].map(lambda f: feature_meta.get(f, (f, "Other", "SD"))[1])
    df_plot["unit"] = df_plot["feature"].map(lambda f: feature_meta.get(f, (f, "Other", "SD"))[2])
    df_plot["n_obs"] = len(split.X_train)
    df_plot["n_tracks"] = split.ids_train.nunique()

    # Save CSV for TV4
    tab4_main_csv = REPORTS_DIR / "forest_main_data.csv"
    cols_tab4 = [
        "feature", "display_name", "group", "label_type", "unit",
        "coef", "bse_cluster", "ci_95_lower", "ci_95_upper",
        "t_stat", "p_value", "holm", "q_bh", "evidence_tier",
        "n_obs", "n_tracks", "vif"
    ]
    df_plot[cols_tab4].to_csv(tab4_main_csv, index=False)
    print(f"[OK] Saved Tab 4 source data: {tab4_main_csv}")

    color_map = {
        "vững": "#1B7837",              # Deep Forest Green
        "gợi ý": "#D95F02",             # Vivid Amber/Orange
        "không có bằng chứng": "#7F7F7F" # Muted Gray
    }

    group_order = {
        "Kiểm soát & Bối cảnh (Context - 3 biến)": 3,
        "Lời bài hát (Lyrics - 6 biến)": 2,
        "Đặc tính âm thanh (Audio - 15 biến)": 1,
    }
    df_plot["group_rank"] = df_plot["group"].map(group_order)
    df_plot = df_plot.sort_values(by=["group_rank", "coef"], ascending=[True, True]).reset_index(drop=True)

    # Multi-panel figure: Left = Forest Plot, Right = Dedicated Text Table (Zero Collision)
    fig, (ax_plot, ax_text) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(15.5, 11),
        dpi=300,
        gridspec_kw={"width_ratios": [3.2, 1.8], "wspace": 0.05}
    )

    y_positions = np.arange(len(df_plot))

    # Reference null line at x=0
    ax_plot.axvline(0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.85, zorder=1)

    for y in y_positions:
        ax_plot.axhline(y, color="#EBEBEB", linestyle="-", linewidth=0.8, zorder=0)

    # Shaded band
    ax_plot.axvspan(-16, 0, color="#F8F9FA", alpha=0.5, zorder=0)
    ax_plot.axvspan(0, 16, color="#FCFDFD", alpha=0.5, zorder=0)

    for i, row in df_plot.iterrows():
        c = color_map[row["evidence_tier"]]
        ax_plot.plot([row["ci_95_lower"], row["ci_95_upper"]], [i, i], color=c, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax_plot.plot([row["ci_95_lower"], row["ci_95_lower"]], [i - 0.2, i + 0.2], color=c, linewidth=1.8, zorder=3)
        ax_plot.plot([row["ci_95_upper"], row["ci_95_upper"]], [i - 0.2, i + 0.2], color=c, linewidth=1.8, zorder=3)
        ax_plot.plot(
            row["coef"], i,
            marker="o" if row["label_type"] == "shared" else "s",
            markersize=7.5,
            color=c,
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.2,
            zorder=4
        )

    # Left Y-Axis: Clean names without redundant [US-slope] text
    ax_plot.set_yticks(y_positions)
    ax_plot.set_yticklabels(df_plot["display_name"], fontsize=10.5, fontweight="medium")
    ax_plot.set_xlim(-16.0, 16.0)
    ax_plot.set_xlabel(
        "Estimated Association on ChartScore (Points, 95% Cluster-Robust CI)\n"
        "[per 1 SD for continuous, per 1 unit for binary indicators]",
        fontsize=11.0, fontweight="bold", labelpad=8
    )

    # Section divider lines
    group_changes = []
    current_grp = None
    for idx, row in df_plot.iterrows():
        if row["group"] != current_grp:
            if current_grp is not None:
                group_changes.append(idx - 0.5)
            current_grp = row["group"]

    for y_div in group_changes:
        ax_plot.axhline(y_div, color="#888888", linestyle="-", linewidth=1.2, zorder=2)
        ax_text.axhline(y_div, color="#888888", linestyle="-", linewidth=1.2, zorder=2)

    # Right Axis: Text Table (No collision possible)
    ax_text.set_ylim(ax_plot.get_ylim())
    ax_text.axis("off")

    # Column Titles
    ax_text.text(0.05, len(df_plot) - 0.1, "Coeff [95% CI]", fontsize=10, fontweight="bold", color="#111111", va="bottom", ha="left")
    ax_text.text(0.55, len(df_plot) - 0.1, "p_raw (Holm)", fontsize=10, fontweight="bold", color="#111111", va="bottom", ha="left")
    ax_text.text(0.85, len(df_plot) - 0.1, "Tier", fontsize=10, fontweight="bold", color="#111111", va="bottom", ha="left")

    for i, row in df_plot.iterrows():
        ci_str = f"{row['coef']:+5.2f} [{row['ci_95_lower']:+5.2f}, {row['ci_95_upper']:+5.2f}]"
        if row["holm"] < 0.001:
            p_str = f"{row['p_value']:.1e} (<0.001)"
        else:
            p_str = f"{row['p_value']:.3f} ({row['holm']:.3f})"

        c = color_map[row["evidence_tier"]]
        ax_text.text(0.05, i, ci_str, fontsize=9.2, fontfamily="monospace", va="center", ha="left", color="#222222")
        ax_text.text(0.55, i, p_str, fontsize=9.0, fontfamily="monospace", va="center", ha="left", color=c if row["evidence_tier"] != "không có bằng chứng" else "#555555")
        ax_text.text(0.85, i, row["evidence_tier"], fontsize=8.8, fontweight="bold" if row["evidence_tier"] != "không có bằng chứng" else "normal", va="center", ha="left", color=c)

    # Footnote
    ax_plot.text(
        -15.8, -1.8,
        "* Word Count được chuẩn hóa trên thang đo log(1 + word_count). Sai số chuẩn cụm tính theo spotify_track_id.",
        fontsize=8.5, fontstyle="italic", color="#555555"
    )

    # Legend at bottom below plot
    patch_vung = mpatches.Patch(color=color_map["vững"], label="Vững (Holm < 0.05)")
    patch_goiy = mpatches.Patch(color=color_map["gợi ý"], label="Gợi ý (p_raw < 0.05 hoặc q_BH < 0.10)")
    patch_khong = mpatches.Patch(color=color_map["không có bằng chứng"], label="Không có bằng chứng")
    marker_circle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", markersize=7, label="Hệ số chung")
    marker_square = plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#444444", markersize=7, label="Độ dốc cơ sở US (US-slope)")

    fig.legend(
        handles=[patch_vung, patch_goiy, patch_khong, marker_circle, marker_square],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        fontsize=9.5,
        ncol=5,
        title="Mức bằng chứng thực nghiệm & Bản chất hệ số",
        title_fontproperties={"weight": "bold", "size": 10}
    )

    # Main Figure Title
    fig.suptitle(
        "Forest Plot: Model 3 Predictor Associations with ChartScore\n"
        "N = 1,685 observations (1,527 tracks)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98
    )

    out_png = FIGURES_DIR / "forest_main.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved 300 DPI figure: {out_png}")


# -----------------------------------------------------------------------------
# 5. REDESIGNED FOREST PLOT (B): MARKET SLOPES
# -----------------------------------------------------------------------------
def plot_forest_market_slopes(df_slopes, wald_map, split):
    market_palette = {
        "US (Baseline)": "#0072B2",  # Deep Blue
        "VN": "#D55E00",             # Vermilion Orange
        "KR": "#009E73",             # Bluish Green
    }
    market_labels = {
        "US (Baseline)": "US (Reference Market)",
        "VN": "Vietnam (VN)",
        "KR": "South Korea (KR)",
    }

    feature_display = {
        "energy": "Energy (RMS)",
        "spectral_centroid": "Spectral Centroid (Hz)",
        "avg_word_length": "Average Word Length (Characters)",
        "word_count": "Word Count (log1p scale)",
        "repetition_ratio": "Lyrics Repetition Ratio",
    }

    df_slopes = df_slopes.copy()
    df_slopes["display_name"] = df_slopes["feature"].map(feature_display)
    df_slopes["unit"] = "SD"
    df_slopes["n_obs"] = len(split.X_train)
    df_slopes["n_tracks"] = split.ids_train.nunique()

    for f in df_slopes["feature"].unique():
        w = wald_map[f]
        mask = (df_slopes["feature"] == f)
        df_slopes.loc[mask, "wald_chi2"] = w["chi2"]
        df_slopes.loc[mask, "wald_p_raw"] = w["p_raw"]
        df_slopes.loc[mask, "wald_p_holm"] = w["holm"]
        df_slopes.loc[mask, "wald_q_bh"] = w["q_bh"]
        df_slopes.loc[mask, "wald_evidence_tier"] = w["evidence_tier"]

    # Export Tab 4 source data
    tab4_slopes_csv = REPORTS_DIR / "forest_market_slopes_data.csv"
    df_slopes.to_csv(tab4_slopes_csv, index=False)
    print(f"[OK] Saved Tab 4 source data: {tab4_slopes_csv}")

    features_ordered = [
        "spectral_centroid",
        "energy",
        "avg_word_length",
        "word_count",
        "repetition_ratio",
    ]

    fig, axes = plt.subplots(
        nrows=5, ncols=2,
        figsize=(15.5, 12.5),
        dpi=300,
        gridspec_kw={"width_ratios": [3.2, 1.8], "wspace": 0.05, "hspace": 0.45}
    )

    for i, feat in enumerate(features_ordered):
        ax_plot = axes[i, 0]
        ax_text = axes[i, 1]

        sub = df_slopes[df_slopes["feature"] == feat].copy()
        market_order = ["KR", "VN", "US (Baseline)"]
        sub["order"] = sub["market"].map(lambda m: market_order.index(m))
        sub = sub.sort_values("order").reset_index(drop=True)

        w_info = wald_map[feat]
        tier_tag = w_info["evidence_tier"].upper()
        header_text = (
            f"{feature_display[feat]}   |   "
            f"Joint Wald (2 DOF): χ² = {w_info['chi2']:.2f} (p = {w_info['p_raw']:.3f}, "
            f"p_Holm = {w_info['holm']:.3f}, q_BH = {w_info['q_bh']:.3f})   ->   [{tier_tag}]"
        )
        ax_plot.set_title(header_text, fontsize=10.0, fontweight="bold", loc="left", pad=6, color="#111111")

        # Zero reference line
        ax_plot.axvline(0, color="#333333", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
        ax_plot.axvspan(-14, 0, color="#F8F9FA", alpha=0.6, zorder=0)
        ax_plot.axvspan(0, 16, color="#FCFDFD", alpha=0.6, zorder=0)

        y_positions = np.arange(len(sub))

        for y_idx, row in sub.iterrows():
            mkt = row["market"]
            c = market_palette[mkt]
            ax_plot.plot([row["ci_95_lower"], row["ci_95_upper"]], [y_idx, y_idx], color=c, linewidth=2.4, zorder=3)
            ax_plot.plot([row["ci_95_lower"], row["ci_95_lower"]], [y_idx - 0.2, y_idx + 0.2], color=c, linewidth=1.8, zorder=3)
            ax_plot.plot([row["ci_95_upper"], row["ci_95_upper"]], [y_idx - 0.2, y_idx + 0.2], color=c, linewidth=1.8, zorder=3)
            ax_plot.plot(row["slope"], y_idx, marker="o", markersize=8, color=c, markeredgecolor="#FFFFFF", markeredgewidth=1.3, zorder=4)

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([market_labels[m] for m in sub["market"]], fontsize=9.5, fontweight="medium")
        ax_plot.set_xlim(-13.5, 16.0)
        ax_plot.grid(axis="x", color="#EBEBEB", linestyle=":", linewidth=0.8)

        # Right side dedicated text column (Zero collision)
        ax_text.set_ylim(ax_plot.get_ylim())
        ax_text.axis("off")

        for y_idx, row in sub.iterrows():
            slope_str = f"Slope = {row['slope']:+5.2f} [{row['ci_95_lower']:+5.2f}, {row['ci_95_upper']:+5.2f}]"
            p_str = f"p = {row['p_value']:.3f}"
            c = market_palette[row["market"]]
            ax_text.text(0.05, y_idx, slope_str, fontsize=9.2, fontfamily="monospace", va="center", ha="left", color="#222222")
            ax_text.text(0.72, y_idx, p_str, fontsize=9.0, fontfamily="monospace", va="center", ha="left", color=c)

    # Bottom axis label on the last plot
    axes[-1, 0].set_xlabel("Market-Specific Slope on ChartScore per 1 SD (Points, 95% Cluster-Robust CI)", fontsize=11.0, fontweight="bold", labelpad=8)
    axes[-1, 1].set_xlabel("", fontsize=11.0)

    # Main Figure Title
    fig.suptitle(
        "Market-specific slopes: Acoustic & Lyric Slopes Across US, VN, and KR\n"
        "N = 1,685 observations (1,527 tracks)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985
    )

    # Legend at bottom below figure
    legend_patches = [
        mpatches.Patch(color=market_palette["US (Baseline)"], label="US Market (Baseline β)"),
        mpatches.Patch(color=market_palette["VN"], label="Vietnam Market (β + λ_VN)"),
        mpatches.Patch(color=market_palette["KR"], label="South Korea Market (β + λ_KR)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        fontsize=10.0,
        ncol=3
    )

    out_png = FIGURES_DIR / "forest_market_slopes.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved 300 DPI figure: {out_png}")


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 75)
    print("AUDITING MODEL 3 MULTIPLE TESTING & RENDERING REDESIGNED FOREST PLOTS")
    print("=" * 75)
    df_coef, df_wald, df_slopes, wald_map, meta = process_coefficients_and_wald()
    split = cross_check_market_slopes(df_slopes, meta)
    print_market_sds(split, meta)
    plot_forest_main(df_coef, split)
    plot_forest_market_slopes(df_slopes, wald_map, split)
    print("\n" + "=" * 75)
    print("ALL AUDITS, ASSERTIONS, CSVS, AND REDESIGNED PLOTS COMPLETED.")
    print("=" * 75)
