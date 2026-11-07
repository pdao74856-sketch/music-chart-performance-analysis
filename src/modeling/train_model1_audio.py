"""
Official Training & Diagnostic Script for Model 1 (Audio-only OLS Baseline)
Project: "What Makes a Hit Song?" (2021-2026)
Author: Member 2 (Audio Features & Showcase Engineer)

Description:
Fits the pure acoustic baseline model using 15 standardized Librosa features
(14 continuous features + 1 unscaled binary mode).
Applies Cluster-Robust Standard Errors grouped by spotify_track_id to account
for repeated track observations across markets (VN, US, KR).
Evaluates baseline R^2, Adjusted R^2, AIC, BIC, and Test set MAE/RMSE.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.prepare_features import get_split
from src.utils.ols_cluster_fit import fit_ols_cluster, metrics


def train_model1(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Train Model 1 (Audio Baseline) with Cluster-Robust SE and evaluate performance.
    
    Parameters:
        df: Optional pre-loaded clean modeling DataFrame. If None, loaded via load_clean_modeling_data().

    Returns:
        Dict containing model results, coefficient table, and test evaluation metrics.
    """
    print("=" * 70)
    print("TRAINING MODEL 1: AUDIO-ONLY OLS BASELINE (CLUSTER-ROBUST SE)")
    print("=" * 70)

    # 1. Load data split via canonical get_split() (Frozen seed 43)
    print("\n[Step 1] Loading frozen Train/Test split via get_split()...")
    split = get_split(df=df)

    X_train = split.X_train
    y_train = split.y_train
    ids_train = split.ids_train

    X_test = split.X_test
    y_test = split.y_test
    ids_test = split.ids_test
    market_test = split.market_test

    cfg = split.config
    print(f"  Train set: {len(X_train)} observations ({ids_train.nunique()} unique tracks)")
    print(f"  Test set:  {len(X_test)} observations ({ids_test.nunique()} unique tracks)")

    # 2. Extract exactly 15 Audio-only features from model_metadata.json (TV1 deliverable)
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m1_cols = meta.get("model_architecture", {}).get("model_1_audio", [])
    else:
        m1_cols = []

    # Fallback to config derivation if metadata missing or empty
    if not m1_cols:
        m1_cols = [f"{c}_std" for c in cfg["audio_cols"]] + ["mode"]

    print(f"\n[Step 2] Verifying Model 1 feature contract ({len(m1_cols)} features)...")
    for idx, col in enumerate(m1_cols, 1):
        print(f"   {idx:2d}. {col}")

    # Ensure no forbidden/context/leakage features exist in Model 1
    forbidden = ["Market_VN", "Market_KR", "release_year", "release_year_std", "word_count", "rank"]
    leaked = [col for col in forbidden if col in m1_cols]
    if leaked:
        raise ValueError(f"[CRITICAL ERROR] Forbidden features detected in Model 1: {leaked}")

    # 3. Fit OLS with Cluster-Robust Standard Errors grouped by spotify_track_id
    print("\n[Step 3] Fitting OLS with Cluster-Robust SE (by spotify_track_id)...")
    res1 = fit_ols_cluster(y=y_train, X=X_train[m1_cols], groups=ids_train)

    train_mae = float(np.mean(np.abs(y_train - res1.fittedvalues)))

    print(f"\n--- Model 1 Estimation Summary ---")
    print(f"  Observations (N):    {int(res1.nobs)}")
    print(f"  Cluster Groups (G):  {res1.n_clusters} unique tracks")
    print(f"  R-squared:           {res1.rsquared:.6f}")
    print(f"  Adjusted R-squared:  {res1.rsquared_adj:.6f}")
    print(f"  AIC:                 {res1.aic:.2f}")
    print(f"  BIC:                 {res1.bic:.2f}")
    print(f"  Train MAE:           {train_mae:.4f}")

    # 4. Extract coefficient table with cluster-robust diagnostics
    print("\n[Step 4] Extracting coefficient table with cluster vs unclustered SE...")
    params = res1.params
    cluster_bse = res1.bse
    standard_bse = res1.standard_bse
    t_stats = res1.tvalues
    p_values = res1.pvalues
    conf_int = res1.conf_int()

    coef_records = []
    for var_name in params.index:
        coef_records.append({
            "feature": var_name,
            "coef": float(params[var_name]),
            "se_cluster": float(cluster_bse[var_name]),
            "se_standard": float(standard_bse[var_name]),
            "se_ratio": float(cluster_bse[var_name] / standard_bse[var_name]),
            "t_stat": float(t_stats[var_name]),
            "p_value": float(p_values[var_name]),
            "ci_95_lower": float(conf_int.loc[var_name, 0]),
            "ci_95_upper": float(conf_int.loc[var_name, 1]),
        })

    df_coef = pd.DataFrame(coef_records)

    # 5. Out-of-sample prediction and evaluation on Test set
    print("\n[Step 5] Evaluating out-of-sample predictions on Test set...")
    X_test_const = sm.add_constant(X_test[m1_cols], has_constant="skip")
    y_pred_test = res1.predict(X_test_const)

    df_test_metrics = metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        market=market_test,
        track_ids=ids_test,
        seed=43
    )
    df_test_metrics["model"] = "Model 1 (Audio Baseline)"

    print("\n--- Test Set Metrics (Overall & by Market) ---")
    print(df_test_metrics.to_string(index=False))

    # 6. Export deliverables
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    coef_path = reports_dir / "model1_coefficients.csv"
    metrics_path = reports_dir / "model1_test_metrics.csv"

    df_coef.to_csv(coef_path, index=False)
    df_test_metrics.to_csv(metrics_path, index=False)

    print(f"\n[Artifacts Exported]")
    print(f"  1. Coefficients: {coef_path}")
    print(f"  2. Test Metrics: {metrics_path}")
    print("\n[SUCCESS] Model 1 training & evaluation completed successfully.")

    test_mae_overall = float(df_test_metrics.loc[df_test_metrics["Market"] == "Overall", "MAE"].iloc[0])

    return {
        "model_result": res1,
        "coefficients": df_coef,
        "test_metrics": df_test_metrics,
        "feature_cols": m1_cols,
        "train_mae": train_mae,
        "test_mae": test_mae_overall,
    }


if __name__ == "__main__":
    train_model1()
