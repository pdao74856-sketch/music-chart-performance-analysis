"""
Script: src/modeling/paired_bootstrap_nested.py
Author: Member 1 (Modeling & Analysis Lead)
Purpose:
  Paired Cluster Bootstrap for Nested Hierarchical Models (M1 -> M2 -> M2b -> M3)
  on Out-of-Sample Test Set (N = 410 rows, 382 tracks).

  Comparisons:
    1. Model 2 (Audio + Lyrics) vs Model 1 (Audio Baseline)
    2. Model 2b (+ Context) vs Model 2 (Audio + Lyrics)
    3. Model 3 (Full OLS) vs Model 2b (+ Context)
    4. Model 3 (Full OLS) vs Model 1 (Audio Baseline)

  Samples:
    - Overall (N=410)
    - Market_US (N=165)
    - Market_KR (N=140)
    - Market_VN (N=105)

  Method:
    Cluster bootstrap by spotify_track_id (sampling tracks with replacement).
    Identical resample indices across all models.
    Seed: 43
    n_boot: 1,000 (matched to evaluate_step5.py)
    Sign convention:
      delta_mae = MAE(baseline) - MAE(model) (positive = model has lower error, better)
      delta_r2  = R2(model) - R2(baseline)  (positive = model has higher R2, better)
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Force UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def safe_to_csv(df: pd.DataFrame, path: Path, max_retries: int = 5, retry_delay: float = 0.5, **kwargs):
    """Write DataFrame to CSV atomically using temporary file and replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}")
    df.to_csv(tmp_path, **kwargs)
    
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(retry_delay)
    
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    raise last_err


def calc_metrics(yt_sub: np.ndarray, yp_sub: np.ndarray) -> Tuple[float, float, float]:
    """Calculate R2, MAE, and RMSE exactly as in evaluate_step5.py."""
    ss_res = np.sum((yt_sub - yp_sub) ** 2)
    ss_tot = np.sum((yt_sub - np.mean(yt_sub)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(yt_sub - yp_sub)))
    rmse = float(np.sqrt(np.mean((yt_sub - yp_sub) ** 2)))
    return r2, mae, rmse


def main():
    print("=" * 80)
    print("PAIRED CLUSTER BOOTSTRAP FOR NESTED MODEL COMPARISONS")
    print("=" * 80)

    # 1. Inputs check
    f_preds = REPO_ROOT / "models" / "test_predictions.csv"
    f_orig_paired = REPO_ROOT / "reports" / "paired_bootstrap.csv"
    f_out = REPO_ROOT / "reports" / "diagnostics" / "paired_bootstrap_nested.csv"

    if not f_preds.exists():
        raise FileNotFoundError(f"Missing required file: {f_preds}")
    if not f_orig_paired.exists():
        raise FileNotFoundError(f"Missing required file: {f_orig_paired}")

    print(f"Reading predictions from : {f_preds}")
    print(f"Reading original ref from: {f_orig_paired}")

    # Read test predictions
    df_preds = pd.read_csv(f_preds)
    required_cols = [
        "spotify_track_id", "market", "chart_score_actual",
        "pred_m1", "pred_m2", "pred_m2b", "pred_ols_m3"
    ]
    for c in required_cols:
        if c not in df_preds.columns:
            raise KeyError(f"Column '{c}' not found in {f_preds}")

    print(f"Loaded test predictions  : {len(df_preds)} rows, {df_preds['spotify_track_id'].nunique()} tracks")

    # Inspect n_boot from evaluate_step5.py
    n_boot = 1000
    seed = 43
    print(f"Parameters from evaluate_step5.py: n_boot = {n_boot}, seed = {seed}")

    # Define candidate models and their prediction vectors
    models_dict = {
        "Model 1 (Audio Baseline)": df_preds["pred_m1"].values,
        "Model 2 (Audio + Lyrics)": df_preds["pred_m2"].values,
        "Model 2b (+ Context)":     df_preds["pred_m2b"].values,
        "Model 3 (Full OLS)":       df_preds["pred_ols_m3"].values,
    }

    # Samples definition
    samples_dict = {
        "Overall": np.ones(len(df_preds), dtype=bool),
        "Market_US": (df_preds["market"].values == "US"),
        "Market_KR": (df_preds["market"].values == "KR"),
        "Market_VN": (df_preds["market"].values == "VN"),
    }

    # Define model pairs: (candidate, baseline/reference)
    comparison_pairs = [
        ("Model 2 (Audio + Lyrics)", "Model 1 (Audio Baseline)"),
        ("Model 2b (+ Context)",     "Model 2 (Audio + Lyrics)"),
        ("Model 3 (Full OLS)",       "Model 2b (+ Context)"),
        ("Model 3 (Full OLS)",       "Model 1 (Audio Baseline)"),
    ]

    # Pre-generate cluster bootstrap indices
    print("\nRunning cluster bootstrap resamples by track...")
    unique_tracks = df_preds["spotify_track_id"].unique()
    n_tracks = len(unique_tracks)
    rng = np.random.default_rng(seed)

    track_to_indices = {}
    for idx, tid in enumerate(df_preds["spotify_track_id"]):
        track_to_indices.setdefault(tid, []).append(idx)

    boot_indices_list = []
    for _ in range(n_boot):
        sample_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        idx_pool = []
        for st in sample_tracks:
            idx_pool.extend(track_to_indices[st])
        boot_indices_list.append(np.array(idx_pool))

    # Pre-calculate bootstrap metrics for each model and sample
    boot_metrics = {m: {s: {"r2": [], "mae": []} for s in samples_dict} for m in models_dict}
    y_test_arr = df_preds["chart_score_actual"].values

    for b_idx in boot_indices_list:
        yt_b = y_test_arr[b_idx]
        for s_name, s_mask in samples_dict.items():
            s_mask_b = s_mask[b_idx]
            if not np.any(s_mask_b):
                continue
            yt_sub = yt_b[s_mask_b]
            for m_name, yp_arr in models_dict.items():
                yp_sub = yp_arr[b_idx][s_mask_b]
                r2, mae, _ = calc_metrics(yt_sub, yp_sub)
                boot_metrics[m_name][s_name]["r2"].append(r2)
                boot_metrics[m_name][s_name]["mae"].append(mae)

    # Compute paired statistics
    records = []
    diff_checks = []

    for cand, base in comparison_pairs:
        for s_name, s_mask in samples_dict.items():
            yt_sub = y_test_arr[s_mask]
            yp_cand = models_dict[cand][s_mask]
            yp_base = models_dict[base][s_mask]

            r2_cand, mae_cand, _ = calc_metrics(yt_sub, yp_cand)
            r2_base, mae_base, _ = calc_metrics(yt_sub, yp_base)

            # Point estimates
            delta_mae_pt = mae_base - mae_cand  # positive = candidate model has lower MAE (better)
            delta_r2_pt = r2_cand - r2_base     # positive = candidate model has higher R2 (better)

            # Direct check for Overall
            if s_name == "Overall":
                direct_diff_r2 = r2_cand - r2_base
                direct_diff_mae = mae_base - mae_cand
                diff_checks.append({
                    "pair": f"{cand} vs {base}",
                    "diff_r2_diff": abs(delta_r2_pt - direct_diff_r2),
                    "diff_mae_diff": abs(delta_mae_pt - direct_diff_mae),
                })

            # Bootstrap distributions
            mae_base_boots = np.array(boot_metrics[base][s_name]["mae"])
            mae_cand_boots = np.array(boot_metrics[cand][s_name]["mae"])
            delta_mae_boots = mae_base_boots - mae_cand_boots

            r2_base_boots = np.array(boot_metrics[base][s_name]["r2"])
            r2_cand_boots = np.array(boot_metrics[cand][s_name]["r2"])
            delta_r2_boots = r2_cand_boots - r2_base_boots

            d_mae_ci = [float(np.percentile(delta_mae_boots, 2.5)), float(np.percentile(delta_mae_boots, 97.5))]
            d_r2_ci = [float(np.percentile(delta_r2_boots, 2.5)), float(np.percentile(delta_r2_boots, 97.5))]

            pct_mae_better = float(np.mean(delta_mae_boots > 0) * 100.0)
            pct_r2_better = float(np.mean(delta_r2_boots > 0) * 100.0)

            records.append({
                "model": cand,
                "baseline": base,
                "sample": s_name,
                "delta_mae": delta_mae_pt,
                "delta_mae_ci_95": f"[{d_mae_ci[0]:+.2f}, {d_mae_ci[1]:+.2f}]",
                "pct_mae_better": pct_mae_better,
                "delta_r2": delta_r2_pt,
                "delta_r2_ci_95": f"[{d_r2_ci[0]:+.4f}, {d_r2_ci[1]:+.4f}]",
                "pct_r2_better": pct_r2_better,
            })

    df_out = pd.DataFrame(records)
    safe_to_csv(df_out, f_out, index=False)
    print(f"\n[OK] Saved results ({len(df_out)} rows) to: {f_out}")

    # Validation check
    print("\n" + "=" * 80)
    print("KIỂM TRA ĐỐI CHIẾU ƯỚC LƯỢNG ĐIỂM VỚI TEST_PREDICTIONS TRỰC TIẾP (OVERALL):")
    print("=" * 80)
    max_diff_r2 = max(c["diff_r2_diff"] for c in diff_checks)
    max_diff_mae = max(c["diff_mae_diff"] for c in diff_checks)
    for c in diff_checks:
        print(f"  {c['pair']:<55} | diff|R2|: {c['diff_r2_diff']:.2e} | diff|MAE|: {c['diff_mae_diff']:.2e}")
    print(f"\n  Max |diff| delta_r2 : {max_diff_r2:.2e}")
    print(f"  Max |diff| delta_mae: {max_diff_mae:.2e}")
    print("=" * 80)

    # Print full table
    print("\nTOÀN BỘ BẢNG KẾT QUẢ (paired_bootstrap_nested.csv):")
    print("=" * 115)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_out.to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
