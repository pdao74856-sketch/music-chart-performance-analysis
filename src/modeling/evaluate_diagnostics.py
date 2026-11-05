"""
Evaluation and Diagnostics Script for Model 3 (Full OLS + Cluster SE)
and Robustness / Sensitivity Analysis.

Covers:
- Residual Diagnostics: Q-Q plots, Residuals vs Fitted, Breusch-Pagan test, Cook's distance, Residual distribution by market.
- Sensitivity Analysis: Exclude catalog tracks released before 1980 (Train only), re-fit M3 with newly fitted scaler & q01, compare metrics and coefficients.
- Chapter 4 synthesis: Top 20 test prediction errors joined with weeks_on_chart, chapter 4 numbers collation.

Author: Member 1 (Modeling & Analysis Lead)
Project: What Makes a Hit Song? (2021-2026)
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.prepare_features import (
    load_clean_modeling_data,
    get_split,
    build_design_matrix,
    DEFAULT_CONFIG,
)
from src.modeling.ols_utils import fit_ols, wald_block, metrics


def safe_to_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely writes a DataFrame to CSV via temporary file and atomic replace,
    with retries on PermissionError.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    df.to_csv(tmp_path, **kwargs)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if filepath.exists():
                os.replace(tmp_path, filepath)
            else:
                tmp_path.rename(filepath)
            break
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)


def generate_diagnostics():
    print("=" * 80)
    print("MODEL 3 EVALUATION & DIAGNOSTICS SUITE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Step 0: Load Clean Modeling Data & Setup Split
    # --------------------------------------------------------------------------
    print("\n[Step 0] Loading clean modeling data directly from Supabase...")
    df = load_clean_modeling_data(source="supabase")
    data_source_name = getattr(df, "data_source", "supabase")
    print(f"Data source: {data_source_name}")
    print(f"Total dataset rows: {len(df)}")
    if len(df) != 2098:
        raise ValueError(f"Expected 2,098 rows from Supabase, but got {len(df)}!")

    split = get_split(df=df, seed=43)
    n_train = len(split.raw_train)
    n_test = len(split.raw_test)
    n_test_tracks = split.raw_test["spotify_track_id"].nunique()
    print(f"Train split rows: {n_train} (expected 1,685)")
    print(f"Test split rows:  {n_test} (expected 410)")
    print(f"Test unique tracks: {n_test_tracks} (expected 382)")

    if n_train != 1685 or n_test != 410 or n_test_tracks != 382:
        raise ValueError("Data split counts do not match project specification (1685 train, 410 test, 382 test tracks)!")

    cfg = split.config
    audio_cols_std = [f"{c}_std" for c in cfg["audio_cols"]] + ["mode"]
    lyrics_cols_std = [f"{c}_std" for c in cfg["lyrics_cols"]]
    context_cols_std = [f"{cfg['year_col']}_std", "Market_VN", "Market_KR"]
    interaction_bases = cfg["interaction_cols"]  # 5 features
    inter_cols_std = [
        f"{base}_std__x__{m}"
        for base in interaction_bases
        for m in ["VN", "KR"]
    ]
    m3_cols = audio_cols_std + lyrics_cols_std + context_cols_std + inter_cols_std
    print(f"Model 3 total predictors: {len(m3_cols)}")

    # Create reports/diagnostics directory
    out_dir = repo_root / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # VIỆC 1 – Chẩn đoán phần dư của M3 (đã fit trên Train)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VIỆC 1: CHẨN ĐOÁN PHẦN DƯ CỦA M3 (FIT TRÊN TRAIN)")
    print("=" * 60)

    # Fit M3 on Train with Cluster-Robust SE
    res3 = fit_ols(split.X_train[m3_cols], split.y_train, split.raw_train["spotify_track_id"])
    
    # Train predictions & residuals
    X_train_const = sm.add_constant(split.X_train[m3_cols], has_constant="skip")
    y_pred_train = res3.predict(X_train_const)
    resid_train = split.y_train.values - y_pred_train.values

    # Test predictions & residuals
    X_test_const = sm.add_constant(split.X_test[m3_cols], has_constant="skip")
    y_pred_test = res3.predict(X_test_const)
    resid_test = split.y_test.values - y_pred_test.values

    # a) Q-Q plot phần dư Train và phần dư Test (2 panel)
    print("\n--- [1a] Q-Q Plot of Residuals (Train & Test) ---")
    qq_path = out_dir / "qq_residuals.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    
    # Train Q-Q
    stats.probplot(resid_train, dist="norm", plot=axes[0])
    axes[0].set_title(f"Train Residuals Q-Q Plot\n(N = {len(resid_train):,} rows, {split.raw_train['spotify_track_id'].nunique():,} tracks)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Theoretical Quantiles", fontsize=10)
    axes[0].set_ylabel("Sample Quantiles (Residuals)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].get_lines()[0].set_markersize(3.5)
    axes[0].get_lines()[0].set_color("#1f77b4")
    axes[0].get_lines()[0].set_alpha(0.6)
    axes[0].get_lines()[1].set_color("#d62728")
    axes[0].get_lines()[1].set_linewidth(1.5)

    # Test Q-Q
    stats.probplot(resid_test, dist="norm", plot=axes[1])
    axes[1].set_title(f"Test Residuals Q-Q Plot\n(N = {len(resid_test):,} rows, {n_test_tracks:,} tracks)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=10)
    axes[1].set_ylabel("Sample Quantiles (Residuals)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].get_lines()[0].set_markersize(4.0)
    axes[1].get_lines()[0].set_color("#2ca02c")
    axes[1].get_lines()[0].set_alpha(0.7)
    axes[1].get_lines()[1].set_color("#d62728")
    axes[1].get_lines()[1].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(qq_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Q-Q plot to: {qq_path}")

    # b) Phần dư theo giá trị dự đoán (Train và Test, tô màu theo market)
    print("\n--- [1b] Residuals vs Fitted Values (Train & Test by Market) ---")
    rvf_path = out_dir / "residuals_vs_fitted.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    market_colors = {"US": "#1f77b4", "KR": "#2ca02c", "VN": "#ff7f0e"}

    # Train Panel
    train_m = split.raw_train["market"].values
    for m in ["US", "KR", "VN"]:
        mask = (train_m == m)
        axes[0].scatter(
            y_pred_train.values[mask],
            resid_train[mask],
            c=market_colors.get(m, "gray"),
            label=f"Market {m} ({mask.sum():,} rows)",
            alpha=0.5,
            edgecolors="none",
            s=22,
        )
    axes[0].axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[0].set_title(f"Train: Residuals vs Fitted\n(N = {len(resid_train):,} rows, {split.raw_train['spotify_track_id'].nunique():,} tracks)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Fitted Values (chart_score)", fontsize=10)
    axes[0].set_ylabel("Residuals (chart_score - y_hat)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", framealpha=0.8, fontsize=9)

    # Test Panel
    test_m = split.raw_test["market"].values
    for m in ["US", "KR", "VN"]:
        mask = (test_m == m)
        axes[1].scatter(
            y_pred_test.values[mask],
            resid_test[mask],
            c=market_colors.get(m, "gray"),
            label=f"Market {m} ({mask.sum():,} rows)",
            alpha=0.6,
            edgecolors="none",
            s=28,
        )
    axes[1].axhline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].set_title(f"Test: Residuals vs Fitted\n(N = {len(resid_test):,} rows, {n_test_tracks:,} tracks)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Fitted Values (chart_score)", fontsize=10)
    axes[1].set_ylabel("Residuals (chart_score - y_hat)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left", framealpha=0.8, fontsize=9)

    plt.tight_layout()
    plt.savefig(rvf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Residuals vs Fitted plot to: {rvf_path}")

    # c) Breusch–Pagan trên Train
    print("\n--- [1c] Breusch–Pagan Heteroskedasticity Test (Train) ---")
    bp_lm_stat, bp_p_val, bp_f_stat, bp_fp_val = het_breuschpagan(
        res3.resid,
        X_train_const,
    )
    print(f"Breusch-Pagan LM Statistic: {bp_lm_stat:.4f}")
    print(f"Breusch-Pagan LM p-value:   {bp_p_val:.4e}")
    print(f"Breusch-Pagan F Statistic:  {bp_f_stat:.4f}")
    print(f"Breusch-Pagan F p-value:   {bp_fp_val:.4e}")
    print("NOTE: SE cụm vẫn dùng bất kể kết quả kiểm định phương sai sai số thay đổi.")

    # d) Cook's distance trên Train (lấy từ OLS thường, unclustered)
    print("\n--- [1d] Cook's Distance on Train (Unclustered OLS Influence) ---")
    print("Ghi chú phương pháp: Cook's distance được tính từ mô hình OLS thường trên Train (unclustered OLS),")
    print("do statsmodels không hỗ trợ get_influence() trực tiếp cho đối tượng fit có cov_type='cluster'.")
    res3_unclustered = getattr(res3, "standard_res", None)
    if res3_unclustered is None:
        res3_unclustered = sm.OLS(split.y_train, X_train_const).fit(cov_type="nonrobust")
    
    infl = res3_unclustered.get_influence()
    cooks_d, _ = infl.cooks_distance
    threshold_4_n = 4.0 / n_train
    print(f"Cook's distance reference threshold (4/n): {threshold_4_n:.6f} (n = {n_train})")

    df_cooks = pd.DataFrame({
        "spotify_track_id": split.raw_train["spotify_track_id"].values,
        "market": split.raw_train["market"].values,
        "title": split.raw_train["title"].values,
        "artist": split.raw_train["artist"].values,
        "chart_score": split.raw_train["chart_score"].values,
        "cooks_d": cooks_d,
    })
    df_cooks_sorted = df_cooks.sort_values(by="cooks_d", ascending=False).reset_index(drop=True)
    df_cooks_top10 = df_cooks_sorted.head(10).copy()
    
    print("\nTop 10 observations with highest Cook's distance on Train:")
    print(df_cooks_top10[["spotify_track_id", "market", "title", "artist", "chart_score", "cooks_d"]].to_string(index=False))

    cooks_csv_path = out_dir / "cooks_top10.csv"
    safe_to_csv(df_cooks_top10, cooks_csv_path, index=False)
    print(f"Saved Cook's distance top 10 to: {cooks_csv_path}")

    # e) Phân phối phần dư theo market trên Test
    print("\n--- [1e] Test Residual Distribution by Market ---")
    df_resid_test = pd.DataFrame({
        "market": split.raw_test["market"].values,
        "residual": resid_test,
    })
    
    market_stats_rows = []
    # Overall
    all_res = df_resid_test["residual"].values
    market_stats_rows.append({
        "market": "Overall",
        "n_rows": len(all_res),
        "mean": float(np.mean(all_res)),
        "sd": float(np.std(all_res, ddof=1)),
        "p05": float(np.percentile(all_res, 5)),
        "p50_median": float(np.percentile(all_res, 50)),
        "p95": float(np.percentile(all_res, 95)),
    })
    
    # By Market
    for m in ["US", "KR", "VN"]:
        sub_res = df_resid_test[df_resid_test["market"] == m]["residual"].values
        market_stats_rows.append({
            "market": f"Market_{m}",
            "n_rows": len(sub_res),
            "mean": float(np.mean(sub_res)),
            "sd": float(np.std(sub_res, ddof=1)),
            "p05": float(np.percentile(sub_res, 5)),
            "p50_median": float(np.percentile(sub_res, 50)),
            "p95": float(np.percentile(sub_res, 95)),
        })

    df_resid_by_market = pd.DataFrame(market_stats_rows)
    print(df_resid_by_market.to_string(index=False))

    resid_market_csv_path = out_dir / "residual_by_market.csv"
    safe_to_csv(df_resid_by_market, resid_market_csv_path, index=False)
    print(f"Saved Test residual distribution to: {resid_market_csv_path}")


    # --------------------------------------------------------------------------
    # VIỆC 2 – Độ nhạy: bỏ bài phát hành trước 1980
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VIỆC 2: ĐỘ NHẠY – BỎ BÀI PHÁT HÀNH TRƯỚC 1980")
    print("=" * 60)

    # Filter Train: release_year >= 1980
    raw_train_filtered = split.raw_train[split.raw_train["release_year"] >= 1980].copy().reset_index(drop=True)
    raw_test_intact = split.raw_test.copy().reset_index(drop=True)

    dropped_train_rows = len(split.raw_train) - len(raw_train_filtered)
    # Test is kept intact for valid comparison
    dropped_test_rows = 0
    test_pre1980_count = (raw_test_intact["release_year"] < 1980).sum()

    print(f"Số dòng bị loại ở Train (release_year < 1980): {dropped_train_rows} dòng (Train còn {len(raw_train_filtered)} dòng)")
    print(f"Số dòng bị loại ở Test: {dropped_test_rows} dòng (giữ nguyên toàn bộ {len(raw_test_intact)} dòng; số dòng < 1980 trong Test là {test_pre1980_count})")

    # Fit new StandardScaler and compute new q01 on filtered Train
    cfg_filtered = dict(cfg)
    q01_filtered = float(raw_train_filtered[cfg_filtered["year_col"]].quantile(0.01))
    cfg_filtered["winsorize_year_q01"] = q01_filtered
    print(f"q01 release_year mới trên Train đã lọc: {q01_filtered:.4f} (gốc: {cfg.get('winsorize_year_q01', 'None'):.4f})")

    raw_train_scaler_input = raw_train_filtered.copy()
    if "word_count" in raw_train_scaler_input.columns:
        raw_train_scaler_input["word_count"] = np.log1p(raw_train_scaler_input["word_count"].astype(float))
    raw_train_scaler_input[cfg_filtered["year_col"]] = raw_train_scaler_input[cfg_filtered["year_col"]].clip(lower=q01_filtered)

    continuous_cols = cfg_filtered["audio_cols"] + cfg_filtered["lyrics_cols"] + [cfg_filtered["year_col"]]
    scaler_filtered = StandardScaler()
    scaler_filtered.fit(raw_train_scaler_input[continuous_cols])

    # Build design matrices with new scaler
    X_train_filtered, _ = build_design_matrix(raw_train_filtered, scaler_filtered, cfg_filtered)
    X_test_filtered, _ = build_design_matrix(raw_test_intact, scaler_filtered, cfg_filtered)

    # Fit M3 on filtered Train
    res3_filtered = fit_ols(
        X_train_filtered[m3_cols],
        raw_train_filtered["chart_score"],
        raw_train_filtered["spotify_track_id"],
    )

    # Evaluate on intact Test set
    X_test_filtered_const = sm.add_constant(X_test_filtered[m3_cols], has_constant="skip")
    y_pred_filtered_test = res3_filtered.predict(X_test_filtered_const)
    
    test_metrics_filtered = metrics(
        split.y_test,
        y_pred_filtered_test,
        market=split.raw_test["market"],
        track_ids=split.raw_test["spotify_track_id"],
        n_boot=1000,
        seed=43,
    )
    ov_filt = test_metrics_filtered[test_metrics_filtered["Market"] == "Overall"].iloc[0]
    
    r2_filt = float(ov_filt["R2"])
    mae_filt = float(ov_filt["MAE"])
    rmse_filt = float(ov_filt["RMSE"])

    print(f"\n--- So sánh hiệu năng Test giữa M3 gốc và M3 lọc (loại release_year < 1980) ---")
    print(f"M3 Gốc (baseline):  R² = 0.0340 | MAE = 29.78 | RMSE = 38.30")
    print(f"M3 Lọc (>= 1980):   R² = {r2_filt:.4f} | MAE = {mae_filt:.2f} | RMSE = {rmse_filt:.2f}")
    print(f"Chênh lệch (Lọc - Gốc): ΔR² = {r2_filt - 0.0340:+.4f} | ΔMAE = {mae_filt - 29.78:+.2f} | ΔRMSE = {rmse_filt - 38.30:+.2f}")

    # Wald tests for Blocks on filtered model
    w_inter_filt = wald_block(res3_filtered, inter_cols_std)
    w_lyrics_filt = wald_block(res3_filtered, lyrics_cols_std)

    print(f"\n--- Wald Tests on Filtered Model ---")
    print(f"Interactions Block (10 DOF): p = {w_inter_filt['p_value']:.4f} (Chi2 = {w_inter_filt['statistic']:.4f}, gốc p = 0.0566 / ~0.057)")
    print(f"Lyrics Block (6 DOF):        p = {w_lyrics_filt['p_value']:.4f} (Chi2 = {w_lyrics_filt['statistic']:.4f}, gốc p = 0.0679 / ~0.068)")

    # Read baseline coefficients
    df_base_coefs = pd.read_csv(repo_root / "reports" / "model3_coefficients.csv")
    base_coef_map = dict(zip(df_base_coefs["feature"], df_base_coefs["coef"]))

    # Compare coefficients: find sign changes or magnitude changes > 50%
    comparison_records = []
    filtered_params = res3_filtered.params

    for feat in m3_cols:
        b_base = base_coef_map.get(feat, np.nan)
        b_filt = filtered_params.get(feat, np.nan)
        
        abs_diff = abs(b_filt - b_base)
        sign_changed = (np.sign(b_base) != np.sign(b_filt)) and (b_base != 0) and (b_filt != 0)
        
        pct_change = abs(b_filt - b_base) / abs(b_base) * 100.0 if abs(b_base) > 1e-9 else np.nan
        is_large_change = (pct_change > 50.0) or sign_changed
        
        comparison_records.append({
            "feature": feat,
            "coef_baseline": b_base,
            "coef_filtered": b_filt,
            "abs_diff": abs_diff,
            "pct_change": pct_change,
            "sign_changed": sign_changed,
            "flagged_sensitivity": is_large_change,
        })

    df_sensitivity = pd.DataFrame(comparison_records)
    df_flagged = df_sensitivity[df_sensitivity["flagged_sensitivity"]].copy().reset_index(drop=True)

    print(f"\n--- Bảng các hệ số đổi dấu hoặc thay đổi > 50% so với M3 gốc ({len(df_flagged)} biến) ---")
    print(df_flagged[["feature", "coef_baseline", "coef_filtered", "pct_change", "sign_changed"]].to_string(index=False))

    sens_csv_path = out_dir / "sensitivity_pre1980.csv"
    safe_to_csv(df_sensitivity, sens_csv_path, index=False)
    print(f"Saved full sensitivity comparison to: {sens_csv_path}")


    # --------------------------------------------------------------------------
    # VIỆC 3 – Bảng tổng hợp cho Chương 4 và kiểm tra bài sai số lớn
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VIỆC 3: BẢNG TỔNG HỢP CHO CHƯƠNG 4 & KIỂM TRA BÀI SAI SỐ LỚN")
    print("=" * 60)

    # 3a) Top 20 tracks có |phần dư| lớn nhất trên Test
    print("\n--- [3a] Top 20 tracks with largest |residual| on Test ---")
    df_test_errors = pd.DataFrame({
        "spotify_track_id": split.raw_test["spotify_track_id"].values,
        "market": split.raw_test["market"].values,
        "title": split.raw_test["title"].values,
        "chart_score": split.raw_test["chart_score"].values,
        "y_pred": y_pred_test.values,
        "residual": resid_test,
        "abs_residual": np.abs(resid_test),
    })

    df_top20_errors = df_test_errors.sort_values(by="abs_residual", ascending=False).head(20).copy()

    # Read weeks_on_chart from data/processed/modeling_dataset.csv (chỉ đọc)
    modeling_dataset_path = repo_root / "data" / "processed" / "modeling_dataset.csv"
    print(f"Reading weeks_on_chart from: {modeling_dataset_path}")
    df_modeling_meta = pd.read_csv(
        modeling_dataset_path,
        usecols=["spotify_track_id", "market", "weeks_on_chart"],
    )
    
    # Left join onto top 20
    df_top20_merged = df_top20_errors.merge(
        df_modeling_meta,
        on=["spotify_track_id", "market"],
        how="left",
    )

    missing_weeks_count = df_top20_merged["weeks_on_chart"].isnull().sum()
    if missing_weeks_count > 0:
        print(f"[CẢNH BÁO] Có {missing_weeks_count} dòng trong top 20 bị thiếu weeks_on_chart sau khi join. Để trống theo yêu cầu.")
    else:
        print(f"Khớp nối 100% cột weeks_on_chart cho cả {len(df_top20_merged)} dòng (0 dòng thiếu).")

    top20_final_cols = ["spotify_track_id", "market", "title", "chart_score", "y_pred", "residual", "weeks_on_chart"]
    df_top20_out = df_top20_merged[top20_final_cols].copy()
    
    print("\nTop 20 Test Errors:")
    print(df_top20_out.to_string(index=False))

    top20_csv_path = out_dir / "top_errors_test.csv"
    safe_to_csv(df_top20_out, top20_csv_path, index=False)
    print(f"Saved top 20 errors to: {top20_csv_path}")

    # 3b) Tạo reports/diagnostics/chapter4_numbers.md: gom các số có sẵn
    print("\n--- [3b] Generating chapter4_numbers.md (verbatim copy with citations) ---")
    md_path = out_dir / "chapter4_numbers.md"

    def format_table_as_md(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join([":---"] * len(headers)) + " |"
        rows = []
        for _, r in df.iterrows():
            row_str = "| " + " | ".join(str(val) for val in r) + " |"
            rows.append(row_str)
        return header_line + "\n" + sep_line + "\n" + "\n".join(rows)

    f_comp = repo_root / "reports" / "model_comparison_table.csv"
    f_test = repo_root / "reports" / "model_test_comparison.csv"
    f_wald_b = repo_root / "reports" / "model_wald_blocks.csv"
    f_wald_f = repo_root / "reports" / "model3_wald_features.csv"
    f_boot = repo_root / "reports" / "paired_bootstrap.csv"

    df_comp_table = pd.read_csv(f_comp)
    df_test_table = pd.read_csv(f_test)
    df_wald_f_table = pd.read_csv(f_wald_f)
    df_boot_table = pd.read_csv(f_boot)

    with open(f_wald_b, "r", encoding="utf-8") as fp:
        raw_wald_b_text = fp.read()
    import io
    parts_wald_b = raw_wald_b_text.split("# HIERARCHICAL MODEL INFORMATION CRITERIA (TRAIN SET N=1685)")
    t1_text = parts_wald_b[0].replace("# JOINT WALD TESTS ON MODEL 3 (TRAIN SET N=1685, CLUSTER SE BY TRACK)", "").strip()
    df_wald_b_blocks = pd.read_csv(io.StringIO(t1_text)).dropna(how="all")
    df_wald_b_info = pd.read_csv(io.StringIO(parts_wald_b[1].strip())).dropna(how="all")

    md_blocks = [
        "# Bảng Tổng Hợp Số Liệu Chính Chương 4 (Kết Quả Mô Hình Hóa)",
        "",
        "> **Ghi chú tính toàn vẹn dữ liệu**: Mọi con số trong tài liệu này được trích xuất nguyên văn (verbatim copy) trực tiếp từ các tệp báo cáo gốc tương ứng trong thư mục `reports/`. Tuyệt đối không tự tính toán lại hay làm tròn thủ công.",
        "",
        "---",
        "",
        "## 1. So Sánh Phân Cấp Các Mô Hình Trên Tập Huấn Luyện (Train Set N = 1,685)",
        f"**Nguồn tệp**: `{f_comp.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_comp_table),
        "",
        "---",
        "",
        "## 2. Đánh Giá Khả Năng Tổng Quát Hóa Trên Tập Kiểm Tra Độc Lập (Test Set N = 410, 382 Track)",
        f"**Nguồn tệp**: `{f_test.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_test_table),
        "",
        "---",
        "",
        "## 3. Kiểm Định Wald Đồng Thời Theo Khối Biến Trên Mô Hình 3 (Train Cluster SE)",
        f"**Nguồn tệp**: `{f_wald_b.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_wald_b_blocks),
        "",
        "### Tiêu Chuẩn Thông Tin Phân Cấp Mô Hình (Train N = 1,685)",
        f"**Nguồn tệp**: `{f_wald_b.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_wald_b_info),
        "",
        "---",
        "",
        "## 4. Kiểm Định Wald Cho Từng Biến Tương Tác Của Mô Hình 3 (2 DOF)",
        f"**Nguồn tệp**: `{f_wald_f.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_wald_f_table),
        "",
        "---",
        "",
        "## 5. So Sánh Từng Cặp Bằng Bootstrap Theo Cụm (Paired Cluster Bootstrap Test)",
        f"**Nguồn tệp**: `{f_boot.relative_to(repo_root).as_posix()}`",
        "",
        format_table_as_md(df_boot_table),
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_blocks) + "\n")
    print(f"Saved Chapter 4 numbers collation to: {md_path}")

    print("\n" + "=" * 80)
    print("ALL DIAGNOSTICS & EVALUATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    generate_diagnostics()
