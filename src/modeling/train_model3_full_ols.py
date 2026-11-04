"""
Official Training & Diagnostic Script for Model 3 (Full OLS + Cluster SE + Interactions)
and Hierarchical Model Comparison (M1 -> M2 -> M2b -> M3).

Author: Member 1 (Modeling & Analysis Lead)
Project: What Makes a Hit Song? (2021-2026)
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import joblib

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.prepare_features import get_split, build_design_matrix, DEFAULT_CONFIG
from src.modeling.ols_utils import fit_ols, wald_block, metrics
from src.modeling.mlr_pipeline import MLRPipeline


def compute_market_slopes(
    res: Any,
    interaction_bases: List[str],
) -> pd.DataFrame:
    """
    Computes market-specific slopes for interaction features:
    - US slope = beta_base
    - VN slope = beta_base + lambda_VN
    - KR slope = beta_base + lambda_KR
    with cluster-robust standard errors: Var(beta + lambda) = Var(beta) + Var(lambda) + 2*Cov(beta, lambda).
    """
    params = res.params
    cov = res.cov_params()
    records = []
    
    for base in interaction_bases:
        feat_clean = base.replace("_std", "")
        var_base = cov.loc[base, base]
        se_us = np.sqrt(var_base)
        b_us = params[base]
        t_us = b_us / se_us
        p_us = float(2 * (1 - norm.cdf(np.abs(t_us))))
        
        records.append({
            "feature": feat_clean,
            "market": "US (Baseline)",
            "slope": b_us,
            "se_cluster": se_us,
            "t_stat": t_us,
            "p_value": p_us,
            "ci_95_lower": b_us - 1.96 * se_us,
            "ci_95_upper": b_us + 1.96 * se_us,
        })
        
        for m in ["VN", "KR"]:
            inter_col = f"{base}__x__{m}"
            b_inter = params[inter_col]
            slope_m = b_us + b_inter
            var_inter = cov.loc[inter_col, inter_col]
            cov_base_inter = cov.loc[base, inter_col]
            var_m = var_base + var_inter + 2.0 * cov_base_inter
            se_m = np.sqrt(max(var_m, 1e-12))
            t_m = slope_m / se_m
            p_m = float(2 * (1 - norm.cdf(np.abs(t_m))))
            
            records.append({
                "feature": feat_clean,
                "market": m,
                "slope": slope_m,
                "se_cluster": se_m,
                "t_stat": t_m,
                "p_value": p_m,
                "ci_95_lower": slope_m - 1.96 * se_m,
                "ci_95_upper": slope_m + 1.96 * se_m,
            })
            
    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("TRAINING MODEL 3 & EVALUATING HIERARCHICAL NESTED SUITE (M1 -> M2 -> M2b -> M3)")
    print("=" * 70)
    
    # 1. Load frozen split
    print("\n[Step 1] Loading split data via get_split()...")
    split = get_split()
    
    X_train = split.X_train
    X_test = split.X_test
    y_train = split.y_train
    y_test = split.y_test
    ids_train = split.ids_train
    ids_test = split.ids_test
    market_train = split.market_train
    market_test = split.market_test
    raw_train = split.raw_train
    raw_test = split.raw_test
    cfg = split.config
    scaler = split.scaler
    
    print(f"Train set: {len(X_train)} rows ({ids_train.nunique()} tracks)")
    print(f"Test set:  {len(X_test)} rows ({ids_test.nunique()} tracks)")
    print(f"Design matrix columns count: {len(split.feature_names)}")

    # 2. Define Model Subsets (Strict Hierarchical Nested Architecture)
    audio_cols_std = [f"{c}_std" for c in cfg["audio_cols"]] + ["mode"]
    lyrics_cols_std = [f"{c}_std" for c in cfg["lyrics_cols"]]
    context_cols_std = [f"{cfg['year_col']}_std", "Market_VN", "Market_KR"]
    interaction_bases = cfg["interaction_cols"]  # 5 features
    inter_cols_std = [
        f"{base}_std__x__{m}"
        for base in interaction_bases
        for m in ["VN", "KR"]
    ]
    
    m1_cols = audio_cols_std
    m2_cols = m1_cols + lyrics_cols_std
    m2b_cols = m2_cols + context_cols_std
    m3_cols = m2b_cols + inter_cols_std
    
    print(f"\nFeature counts per model:")
    print(f"  M1 (Audio Baseline): {len(m1_cols)} features")
    print(f"  M2 (Audio + Lyrics): {len(m2_cols)} features")
    print(f"  M2b (+ Context):     {len(m2b_cols)} features")
    print(f"  M3 (Full OLS):       {len(m3_cols)} features")
    
    # 3. Fit Models on Train Set
    print("\n[Step 2] Fitting Hierarchical Models with Cluster-Robust SE (by spotify_track_id)...")
    res1 = fit_ols(X_train[m1_cols], y_train, ids_train)
    res2 = fit_ols(X_train[m2_cols], y_train, ids_train)
    res2b = fit_ols(X_train[m2b_cols], y_train, ids_train)
    res3 = fit_ols(X_train[m3_cols], y_train, ids_train)
    
    # 4. Wald Tests for Nested Progression
    w_lyrics = wald_block(res2, lyrics_cols_std)
    w_context = wald_block(res2b, context_cols_std)
    w_inter = wald_block(res3, inter_cols_std)
    
    comp_records = [
        {
            "model": "M1 (Audio Baseline)",
            "num_features": len(m1_cols),
            "r_squared": res1.rsquared,
            "adj_r_squared": res1.rsquared_adj,
            "aic": res1.aic,
            "bic": res1.bic,
            "delta_r2": 0.0,
            "nested_wald_test": "N/A (Baseline)",
            "wald_chi2": np.nan,
            "wald_df": np.nan,
            "wald_p_value": np.nan,
        },
        {
            "model": "M2 (Audio + Lyrics)",
            "num_features": len(m2_cols),
            "r_squared": res2.rsquared,
            "adj_r_squared": res2.rsquared_adj,
            "aic": res2.aic,
            "bic": res2.bic,
            "delta_r2": res2.rsquared - res1.rsquared,
            "nested_wald_test": "M1 -> M2 (Lyrics 6 DOF)",
            "wald_chi2": w_lyrics["statistic"],
            "wald_df": w_lyrics["df"],
            "wald_p_value": w_lyrics["p_value"],
        },
        {
            "model": "M2b (+ Context)",
            "num_features": len(m2b_cols),
            "r_squared": res2b.rsquared,
            "adj_r_squared": res2b.rsquared_adj,
            "aic": res2b.aic,
            "bic": res2b.bic,
            "delta_r2": res2b.rsquared - res2.rsquared,
            "nested_wald_test": "M2 -> M2b (Context 3 DOF)",
            "wald_chi2": w_context["statistic"],
            "wald_df": w_context["df"],
            "wald_p_value": w_context["p_value"],
        },
        {
            "model": "M3 (Full + Interactions)",
            "num_features": len(m3_cols),
            "r_squared": res3.rsquared,
            "adj_r_squared": res3.rsquared_adj,
            "aic": res3.aic,
            "bic": res3.bic,
            "delta_r2": res3.rsquared - res2b.rsquared,
            "nested_wald_test": "M2b -> M3 (Interactions 10 DOF)",
            "wald_chi2": w_inter["statistic"],
            "wald_df": w_inter["df"],
            "wald_p_value": w_inter["p_value"],
        },
    ]
    df_comp = pd.DataFrame(comp_records)
    print("\n=== HIERARCHICAL MODEL COMPARISON (TRAIN SET N=1,685) ===")
    print(df_comp.to_string(index=False))
    
    # 5. Stepwise Feature-level Wald Tests on Model 3 Interactions (Holm-adjusted)
    print("\n[Step 3] Interaction Tests by Feature (2 DOF each) with Holm-Bonferroni Correction...")
    raw_pvals = []
    feature_tests = []
    for base in interaction_bases:
        feat_clean = base.replace("_std", "")
        pair = [f"{base}_std__x__VN", f"{base}_std__x__KR"]
        w = wald_block(res3, pair)
        feature_tests.append({
            "feature": feat_clean,
            "chi2": w["statistic"],
            "df": w["df"],
            "p_raw": w["p_value"],
        })
        raw_pvals.append(w["p_value"])
        
    _, p_holm, _, _ = multipletests(raw_pvals, method="holm")
    for i, t in enumerate(feature_tests):
        t["p_holm"] = p_holm[i]
        t["significant_05"] = (p_holm[i] < 0.05)
        
    df_feat_wald = pd.DataFrame(feature_tests).sort_values("p_raw")
    print(df_feat_wald.to_string(index=False))

    # 6. Market Slopes
    print("\n[Step 4] Computing Market-specific Slopes for Interaction Features...")
    df_slopes = compute_market_slopes(res3, [f"{b}_std" for b in interaction_bases])
    print(df_slopes.to_string(index=False))
    
    # 7. Model 3 Coefficients Table with VIF, Multiple Testing & Test Levels
    vif_path = repo_root / "reports" / "vif_after.csv"
    vif_map = {}
    if vif_path.exists():
        vif_ref = pd.read_csv(vif_path)
        vif_map = dict(zip(vif_ref["feature"], vif_ref["vif_final"]))
        
    coef_records = []
    conf_int = res3.conf_int()
    for col in res3.params.index:
        coef = res3.params[col]
        bse_cluster = res3.bse[col]
        bse_standard = res3.standard_bse[col]
        se_ratio = bse_cluster / bse_standard if bse_standard > 0 else 1.0
        t_stat = res3.tvalues[col]
        p_val = res3.pvalues[col]
        ci_low = conf_int.loc[col, 0]
        ci_high = conf_int.loc[col, 1]
        vif_val = vif_map.get(col, np.nan)
        
        is_interaction = ("_x_" in col) or ("×" in col)
        test_level = "coef_interaction" if is_interaction else "coef"
        is_us_slope = (col.replace("_std", "") in interaction_bases) or (col in interaction_bases)
        label_type = "US-slope" if (is_us_slope and not is_interaction and col != "const") else "shared"

        coef_records.append({
            "feature": col,
            "coef": coef,
            "bse_cluster": bse_cluster,
            "bse_standard": bse_standard,
            "se_inflation_ratio": se_ratio,
            "t_stat": t_stat,
            "p_value": p_val,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "vif": vif_val,
            "test_level": test_level,
            "label_type": label_type,
        })
    df_coefs = pd.DataFrame(coef_records)

    # Multiple Testing on Main Effects family (non-interaction, non-const)
    main_mask = (df_coefs["test_level"] == "coef") & (df_coefs["feature"] != "const")
    p_main = df_coefs.loc[main_mask, "p_value"].values
    _, holm_main, _, _ = multipletests(p_main, method="holm")
    _, bh_main, _, _ = multipletests(p_main, method="fdr_bh")

    df_coefs["holm"] = np.nan
    df_coefs["q_bh"] = np.nan
    df_coefs["evidence_tier"] = pd.Series(None, index=df_coefs.index, dtype=object)

    df_coefs.loc[main_mask, "holm"] = holm_main
    df_coefs.loc[main_mask, "q_bh"] = bh_main

    def get_tier(p_raw, holm, q_bh):
        if pd.isna(holm):
            return np.nan
        if holm < 0.05:
            return "vững"
        elif (p_raw < 0.05 and holm >= 0.05) or (q_bh < 0.10):
            return "gợi ý"
        else:
            return "không có bằng chứng"

    for idx in df_coefs[main_mask].index:
        df_coefs.loc[idx, "evidence_tier"] = get_tier(
            df_coefs.loc[idx, "p_value"],
            df_coefs.loc[idx, "holm"],
            df_coefs.loc[idx, "q_bh"]
        )

    # For const:
    const_idx = df_coefs[df_coefs["feature"] == "const"].index
    if len(const_idx) > 0:
        df_coefs.loc[const_idx, "evidence_tier"] = "không có bằng chứng"

    # For interaction terms (test_level == "coef_interaction"):
    # holm, q_bh, evidence_tier remain np.nan (TRỐNG) as required
    
    # 8. Single Test Evaluation (Strictly Run Once)
    print("\n[Step 5] Evaluating Model 3 on Frozen Test Set (N=410, 1,000 Cluster Bootstraps)...")
    X_test_const = sm.add_constant(X_test[m3_cols], has_constant="add")
    y_pred_test = res3.predict(X_test_const)
    
    # Baseline 1: Global Train Mean
    global_train_mean = float(y_train.mean())
    y_pred_base_global = np.full_like(y_test, fill_value=global_train_mean)
    
    # Baseline 2: Market-specific Train Mean
    market_train_means = y_train.groupby(market_train).mean().to_dict()
    y_pred_base_market = market_test.map(market_train_means).values
    
    # Metrics
    df_test_m3 = metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        market=market_test,
        track_ids=ids_test,
        n_boot=1000,
        seed=43
    )
    df_test_m3["model"] = "Model 3 (Full OLS)"
    
    df_test_base_global = metrics(
        y_true=y_test,
        y_pred=y_pred_base_global,
        market=market_test,
        track_ids=ids_test,
        n_boot=1000,
        seed=43
    )
    df_test_base_global["model"] = "Baseline: Global Train Mean"
    
    df_test_base_mkt = metrics(
        y_true=y_test,
        y_pred=y_pred_base_market,
        market=market_test,
        track_ids=ids_test,
        n_boot=1000,
        seed=43
    )
    df_test_base_mkt["model"] = "Baseline: Market Train Mean"
    
    df_test_all = pd.concat([df_test_m3, df_test_base_mkt, df_test_base_global], ignore_index=True)
    print("\n=== TEST SET PERFORMANCE & BENCHMARK COMPARISON ===")
    cols_to_print = [c for c in ["model", "Market", "N", "R2", "MAE", "RMSE", "OOB_Predictions_Pct"] if c in df_test_all.columns]
    if cols_to_print:
        print(df_test_all[cols_to_print].to_string(index=False))
    else:
        print(df_test_all.to_string(index=False))

    # 9. Save Outputs
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    df_comp.to_csv(reports_dir / "model_comparison_table.csv", index=False)
    df_coefs.to_csv(reports_dir / "model3_coefficients.csv", index=False)
    df_slopes.to_csv(reports_dir / "model_market_slopes.csv", index=False)
    df_feat_wald.to_csv(reports_dir / "model3_wald_features.csv", index=False)
    df_test_all.to_csv(reports_dir / "model_test_metrics.csv", index=False)
    
    # 10. Package Lightweight MLRPipeline Object
    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        meta_loaded = json.load(f)
        
    pipeline_obj = MLRPipeline(
        config=cfg,
        scaler=scaler,
        feature_names=split.feature_names,
        ols_params=res3.params,
        ols_summary_str=str(res3.summary()),
        train_medians=meta_loaded["train_medians"],
    )
    
    pipeline_path = models_dir / "mlr_pipeline.pkl"
    joblib.dump(pipeline_obj, pipeline_path)
    print(f"\n[OK] Model 3 MLRPipeline serialized successfully: {pipeline_path}")
    
    # Verification of predict_raw from raw test dataframe
    y_check = pipeline_obj.predict_raw(raw_test)
    assert np.allclose(y_check, y_pred_test, atol=1e-7), "MLRPipeline predict_raw mismatch!"
    print("[OK] Verified MLRPipeline.predict_raw(raw_test) matches model output exactly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
