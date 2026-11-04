"""
Script: src/modeling/evaluate_step5.py
Author: Member 1 (Modeling & Analysis Lead)
Purpose:
  Step 5 Implementation for Capstone Modeling Suite:
  1. Fit M1, M2, M2b on Train and predict Test set.
     Update models/test_predictions.csv (pred_m1, pred_m2, pred_m2b).
     Verify pred_ols_m3 unchanged (R2=0.034459, MAE=29.709509, RMSE=38.295332).
  2. Generate reports/model_test_comparison.csv:
     M1, M2, M2b, M3 + 2 baselines (Global Train Mean, Market Train Mean).
     Overall and by market (US, KR, VN) with 95% cluster bootstrap CIs (1,000 resamples by track, seed=43).
  3. Generate reports/paired_bootstrap.csv:
     Paired bootstrap comparison (identical resample indices) against baselines:
     Delta MAE (baseline - model, positive = better) and Delta R2 (model - baseline).
     Point estimates, 95% CIs, and pct iterations > 0.
  4. Generate reports/model_wald_blocks.csv:
     Joint Wald tests on Train (clustered SE by track) for 4 blocks:
     Audio (15), Lyrics (6), Context (3), Interactions (10), with Holm correction over the 4 blocks.
     Also report AIC and BIC for M1, M2, M2b, M3.
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
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from statsmodels.stats.multitest import multipletests

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.prepare_features import get_split
from src.modeling.ols_utils import fit_ols, wald_block

REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("STEP 5: HIERARCHICAL TEST PREDICTIONS, PAIRED BOOTSTRAP & WALD BLOCKS")
    print("=" * 75)

    # 1. Load Split
    print("\n[1/5] Loading data split via get_split()...")
    split = get_split()
    X_train = split.X_train
    X_test = split.X_test
    y_train = split.y_train
    y_test = split.y_test
    ids_train = split.ids_train
    ids_test = split.ids_test
    market_train = split.market_train
    market_test = split.market_test
    cfg = split.config

    print(f"  Train: {len(X_train)} rows ({ids_train.nunique()} tracks)")
    print(f"  Test:  {len(X_test)} rows ({ids_test.nunique()} tracks)")

    # Define model subsets
    audio_cols_std = [f"{c}_std" for c in cfg["audio_cols"]] + ["mode"] # 15
    lyrics_cols_std = [f"{c}_std" for c in cfg["lyrics_cols"]]           # 6
    context_cols_std = [f"{cfg['year_col']}_std", "Market_VN", "Market_KR"] # 3
    interaction_bases = cfg["interaction_cols"]
    inter_cols_std = [
        f"{base}_std__x__{m}"
        for base in interaction_bases
        for m in ["VN", "KR"]
    ] # 10

    m1_cols = audio_cols_std
    m2_cols = m1_cols + lyrics_cols_std
    m2b_cols = m2_cols + context_cols_std
    m3_cols = m2b_cols + inter_cols_std

    # 2. Fit Models on Train
    print("\n[2/5] Fitting M1, M2, M2b, M3 on Train (Clustered SE by spotify_track_id)...")
    res1 = fit_ols(X_train[m1_cols], y_train, ids_train)
    res2 = fit_ols(X_train[m2_cols], y_train, ids_train)
    res2b = fit_ols(X_train[m2b_cols], y_train, ids_train)
    res3 = fit_ols(X_train[m3_cols], y_train, ids_train)

    # Predictions on Test
    X1_test = sm.add_constant(X_test[m1_cols], has_constant="add")
    X2_test = sm.add_constant(X_test[m2_cols], has_constant="add")
    X2b_test = sm.add_constant(X_test[m2b_cols], has_constant="add")
    X3_test = sm.add_constant(X_test[m3_cols], has_constant="add")

    pred_m1 = res1.predict(X1_test)
    pred_m2 = res2.predict(X2_test)
    pred_m2b = res2b.predict(X2b_test)
    pred_m3 = res3.predict(X3_test)

    # Baseline predictions
    global_mean = float(y_train.mean())
    pred_base_global = np.full(len(y_test), fill_value=global_mean)

    market_means = y_train.groupby(market_train).mean().to_dict()
    pred_base_market = market_test.map(market_means).values

    # 3. Update models/test_predictions.csv
    print("\n[3/5] Updating models/test_predictions.csv...")
    test_pred_path = MODELS_DIR / "test_predictions.csv"
    df_preds = pd.read_csv(test_pred_path)

    # Update predictions for M1, M2, M2b, M3, residual
    r2_m3 = r2_score(y_test, pred_m3)
    mae_m3 = mean_absolute_error(y_test, pred_m3)
    rmse_m3 = root_mean_squared_error(y_test, pred_m3)
    print(f"  M3 Check (Energy Model): R2={r2_m3:.6f}, MAE={mae_m3:.6f}, RMSE={rmse_m3:.6f}")

    df_preds["pred_m1"] = pred_m1.values
    df_preds["pred_m2"] = pred_m2.values
    df_preds["pred_m2b"] = pred_m2b.values
    df_preds["pred_ols_m3"] = pred_m3.values
    df_preds["residual"] = y_test.values - pred_m3.values
    df_preds.to_csv(test_pred_path, index=False)
    print(f"  [OK] Successfully filled pred_m1, pred_m2, pred_m2b, pred_ols_m3, residual in {test_pred_path}")

    # 4. Joint Cluster Bootstrap (1,000 iterations, seed=43)
    print("\n[4/5] Running Paired Cluster Bootstrap (1,000 iterations by track, seed=43)...")
    unique_tracks = ids_test.unique()
    n_tracks = len(unique_tracks)
    n_boot = 1000
    rng = np.random.default_rng(43)

    # Track to index mapping for fast resampling
    track_to_indices = {}
    for idx, tid in enumerate(ids_test):
        track_to_indices.setdefault(tid, []).append(idx)

    # Pre-generate resampled index arrays
    boot_indices_list = []
    for _ in range(n_boot):
        sample_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        idx_pool = []
        for st in sample_tracks:
            idx_pool.extend(track_to_indices[st])
        boot_indices_list.append(np.array(idx_pool))

    # All models dict
    models_dict = {
        "Baseline: Global Train Mean": {"preds": pred_base_global, "num_params": 1},
        "Baseline: Market Train Mean": {"preds": pred_base_market, "num_params": 3},
        "Model 1 (Audio Baseline)":    {"preds": pred_m1.values,  "num_params": len(m1_cols) + 1},
        "Model 2 (Audio + Lyrics)":    {"preds": pred_m2.values,  "num_params": len(m2_cols) + 1},
        "Model 2b (+ Context)":        {"preds": pred_m2b.values, "num_params": len(m2b_cols) + 1},
        "Model 3 (Full OLS)":          {"preds": pred_m3.values,  "num_params": len(m3_cols) + 1},
    }

    # Samples to evaluate
    samples_dict = {
        "Overall": np.ones(len(y_test), dtype=bool),
        "Market_US": (market_test.values == "US"),
        "Market_KR": (market_test.values == "KR"),
        "Market_VN": (market_test.values == "VN"),
    }

    # Helper functions
    def calc_metrics(yt_sub, yp_sub):
        ss_res = np.sum((yt_sub - yp_sub) ** 2)
        ss_tot = np.sum((yt_sub - np.mean(yt_sub)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        mae = float(np.mean(np.abs(yt_sub - yp_sub)))
        rmse = float(np.sqrt(np.mean((yt_sub - yp_sub) ** 2)))
        return r2, mae, rmse

    # A. Model Test Comparison Table
    test_comp_records = []
    # Pre-calculate bootstrap metric distributions for paired comparisons
    # structure: boot_metrics[model_name][sample_name] = {'r2': [...], 'mae': [...], 'rmse': [...]}
    boot_metrics = {m: {s: {"r2": [], "mae": [], "rmse": []} for s in samples_dict} for m in models_dict}

    y_test_arr = y_test.values

    for b, b_idx in enumerate(boot_indices_list):
        yt_b = y_test_arr[b_idx]
        for s_name, s_mask in samples_dict.items():
            s_mask_b = s_mask[b_idx]
            if not np.any(s_mask_b):
                continue
            yt_sub = yt_b[s_mask_b]
            for m_name, m_info in models_dict.items():
                yp_sub = m_info["preds"][b_idx][s_mask_b]
                r2, mae, rmse = calc_metrics(yt_sub, yp_sub)
                boot_metrics[m_name][s_name]["r2"].append(r2)
                boot_metrics[m_name][s_name]["mae"].append(mae)
                boot_metrics[m_name][s_name]["rmse"].append(rmse)

    # Build reports/model_test_comparison.csv
    for m_name, m_info in models_dict.items():
        for s_name, s_mask in samples_dict.items():
            yt_sub = y_test_arr[s_mask]
            yp_sub = m_info["preds"][s_mask]
            r2_pt, mae_pt, rmse_pt = calc_metrics(yt_sub, yp_sub)

            r2_boots = boot_metrics[m_name][s_name]["r2"]
            mae_boots = boot_metrics[m_name][s_name]["mae"]
            rmse_boots = boot_metrics[m_name][s_name]["rmse"]

            r2_ci = [float(np.percentile(r2_boots, 2.5)), float(np.percentile(r2_boots, 97.5))]
            mae_ci = [float(np.percentile(mae_boots, 2.5)), float(np.percentile(mae_boots, 97.5))]
            rmse_ci = [float(np.percentile(rmse_boots, 2.5)), float(np.percentile(rmse_boots, 97.5))]

            test_comp_records.append({
                "model": m_name,
                "sample": s_name,
                "num_params": m_info["num_params"],
                "n": int(np.sum(s_mask)),
                "r2": r2_pt,
                "r2_ci_95": f"[{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]",
                "mae": mae_pt,
                "mae_ci_95": f"[{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]",
                "rmse": rmse_pt,
                "rmse_ci_95": f"[{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]",
            })

    df_test_comp = pd.DataFrame(test_comp_records)
    comp_csv_path = REPORTS_DIR / "model_test_comparison.csv"
    df_test_comp.to_csv(comp_csv_path, index=False)
    print(f"  [OK] Saved {comp_csv_path}")

    # B. Paired Bootstrap Comparison Table
    paired_records = []
    baselines = ["Baseline: Global Train Mean", "Baseline: Market Train Mean"]
    candidate_models = [
        "Model 1 (Audio Baseline)",
        "Model 2 (Audio + Lyrics)",
        "Model 2b (+ Context)",
        "Model 3 (Full OLS)",
    ]

    for cand in candidate_models:
        for base in baselines:
            for s_name, s_mask in samples_dict.items():
                yt_sub = y_test_arr[s_mask]
                yp_cand = models_dict[cand]["preds"][s_mask]
                yp_base = models_dict[base]["preds"][s_mask]

                r2_cand, mae_cand, _ = calc_metrics(yt_sub, yp_cand)
                r2_base, mae_base, _ = calc_metrics(yt_sub, yp_base)

                delta_mae_pt = mae_base - mae_cand  # positive = candidate model has lower MAE (better)
                delta_r2_pt = r2_cand - r2_base     # positive = candidate model has higher R2 (better)

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

                paired_records.append({
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

    df_paired = pd.DataFrame(paired_records)
    paired_csv_path = REPORTS_DIR / "paired_bootstrap.csv"
    df_paired.to_csv(paired_csv_path, index=False)
    print(f"  [OK] Saved {paired_csv_path}")

    # 5. Wald Blocks & Model AIC/BIC Table
    print("\n[5/5] Computing Joint Wald Tests on Train & AIC/BIC...")
    # Core 4 blocks of M3
    blocks_m3 = [
        ("Audio Block (Main, M3)", audio_cols_std),
        ("Lyrics Block (Main, M3)", lyrics_cols_std),
        ("Context Block (M3)", context_cols_std),
        ("Interactions Block (M3)", inter_cols_std),
    ]

    wald_records = []
    p_raw_list = []
    for b_name, b_cols in blocks_m3:
        w = wald_block(res3, b_cols)
        wald_records.append({
            "test_scope": "Model 3 Core Blocks",
            "block": b_name,
            "model": "M3 (Full OLS)",
            "num_features": len(b_cols),
            "chi2": float(w["statistic"]),
            "df": int(w["df"]),
            "p_value": float(w["p_value"]),
        })
        p_raw_list.append(float(w["p_value"]))

    _, p_holm, _, _ = multipletests(p_raw_list, method="holm")
    for i, rec in enumerate(wald_records):
        rec["p_holm"] = float(p_holm[i])

    # Additional Joint Wald Tests (PHA 4)
    # Audio full in M3 (15 main + 4 interactions: energy & spectral_centroid with VN, KR = 19 DOF)
    audio_inter_cols = [f"{b}_std__x__{m}" for b in ["energy", "spectral_centroid"] for m in ["VN", "KR"]]
    audio_full_cols = audio_cols_std + audio_inter_cols
    w_audio_full = wald_block(res3, audio_full_cols)
    wald_records.append({
        "test_scope": "Full Feature Family (with Interactions)",
        "block": "Audio Full Block (Main + Interactions, 19 DOF)",
        "model": "M3 (Full OLS)",
        "num_features": len(audio_full_cols),
        "chi2": float(w_audio_full["statistic"]),
        "df": int(w_audio_full["df"]),
        "p_value": float(w_audio_full["p_value"]),
        "p_holm": np.nan,
    })

    # Lyrics full in M3 (6 main + 6 interactions: avg_word_length, word_count, repetition_ratio with VN, KR = 12 DOF)
    lyrics_inter_cols = [f"{b}_std__x__{m}" for b in ["avg_word_length", "word_count", "repetition_ratio"] for m in ["VN", "KR"]]
    lyrics_full_cols = lyrics_cols_std + lyrics_inter_cols
    w_lyrics_full = wald_block(res3, lyrics_full_cols)
    wald_records.append({
        "test_scope": "Full Feature Family (with Interactions)",
        "block": "Lyrics Full Block (Main + Interactions, 12 DOF)",
        "model": "M3 (Full OLS)",
        "num_features": len(lyrics_full_cols),
        "chi2": float(w_lyrics_full["statistic"]),
        "df": int(w_lyrics_full["df"]),
        "p_value": float(w_lyrics_full["p_value"]),
        "p_holm": np.nan,
    })

    # Wald Audio in M1 (15 DOF)
    w_audio_m1 = wald_block(res1, audio_cols_std)
    wald_records.append({
        "test_scope": "Model Specific Joint Significance",
        "block": "Audio Block in M1 (15 DOF)",
        "model": "M1 (Audio Baseline)",
        "num_features": len(audio_cols_std),
        "chi2": float(w_audio_m1["statistic"]),
        "df": int(w_audio_m1["df"]),
        "p_value": float(w_audio_m1["p_value"]),
        "p_holm": np.nan,
    })

    # Wald Lyrics in M2 (6 DOF)
    w_lyrics_m2 = wald_block(res2, lyrics_cols_std)
    wald_records.append({
        "test_scope": "Model Specific Joint Significance",
        "block": "Lyrics Block in M2 (6 DOF)",
        "model": "Model 2 (Audio + Lyrics)",
        "num_features": len(lyrics_cols_std),
        "chi2": float(w_lyrics_m2["statistic"]),
        "df": int(w_lyrics_m2["df"]),
        "p_value": float(w_lyrics_m2["p_value"]),
        "p_holm": np.nan,
    })

    df_wald_blocks = pd.DataFrame(wald_records)

    # Model AIC/BIC Table
    aic_bic_records = [
        {"model": "M1 (Audio Baseline)", "num_params": len(m1_cols) + 1, "r2_train": res1.rsquared, "adj_r2_train": res1.rsquared_adj, "aic": res1.aic, "bic": res1.bic},
        {"model": "Model 2 (Audio + Lyrics)", "num_params": len(m2_cols) + 1, "r2_train": res2.rsquared, "adj_r2_train": res2.rsquared_adj, "aic": res2.aic, "bic": res2.bic},
        {"model": "Model 2b (+ Context)",     "num_params": len(m2b_cols) + 1, "r2_train": res2b.rsquared, "adj_r2_train": res2b.rsquared_adj, "aic": res2b.aic, "bic": res2b.bic},
        {"model": "M3 (Full OLS)",       "num_params": len(m3_cols) + 1, "r2_train": res3.rsquared, "adj_r2_train": res3.rsquared_adj, "aic": res3.aic, "bic": res3.bic},
    ]
    df_aic_bic = pd.DataFrame(aic_bic_records)

    # Combine into model_wald_blocks.csv
    wald_csv_path = REPORTS_DIR / "model_wald_blocks.csv"
    with open(wald_csv_path, "w", encoding="utf-8") as f:
        f.write("# JOINT WALD TESTS ON MODEL 3 (TRAIN SET N=1685, CLUSTER SE BY TRACK)\n")
        df_wald_blocks.to_csv(f, index=False)
        f.write("\n# HIERARCHICAL MODEL INFORMATION CRITERIA (TRAIN SET N=1685)\n")
        df_aic_bic.to_csv(f, index=False)
    print(f"  [OK] Saved {wald_csv_path}")

    # Print Summary Tables
    print("\n" + "=" * 75)
    print("TABLE 1: MODEL TEST PERFORMANCE COMPARISON (N=410, 1,000 BOOTSTRAPS)")
    print("=" * 75)
    print(df_test_comp[df_test_comp["sample"] == "Overall"][["model", "num_params", "n", "r2", "r2_ci_95", "mae", "mae_ci_95", "rmse", "rmse_ci_95"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("TABLE 2: PAIRED BOOTSTRAP OVERALL (BASELINE - MODEL, POSITIVE = BETTER)")
    print("=" * 75)
    print(df_paired[df_paired["sample"] == "Overall"][["model", "baseline", "delta_mae", "delta_mae_ci_95", "pct_mae_better", "delta_r2", "delta_r2_ci_95", "pct_r2_better"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("TABLE 3A: JOINT WALD TESTS ACROSS 4 BLOCKS (MODEL 3 TRAIN)")
    print("=" * 75)
    print(df_wald_blocks.to_string(index=False))

    print("\n" + "=" * 75)
    print("TABLE 3B: HIERARCHICAL MODEL AIC / BIC (TRAIN SET N=1,685)")
    print("=" * 75)
    print(df_aic_bic.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
