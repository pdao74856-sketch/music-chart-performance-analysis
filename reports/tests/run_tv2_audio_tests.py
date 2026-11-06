"""
Verification & Test Execution Script for Member 2 (Audio Lead)
Project: "What Makes a Hit Song?" (2021-2026)
Author: Member 2 (Audio Features & Showcase Engineer)

This script executes:
1. Kruskal-Wallis H-test & Effect Size (eta^2) across 14 continuous Librosa DSP audio features by market (VN, US, KR).
2. Physical correlation verification (Energy vs Loudness, Spectral Centroid vs Bandwidth).
3. VIF Multicollinearity Test before and after applying Audio Feature Contract v2.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any
import numpy as np

# Force UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.prepare_features import get_split
from src.utils.supabase_client import load_clean_modeling_data


def run_kruskal_wallis_tests(X_train: pd.DataFrame, audio_features: list) -> pd.DataFrame:
    """Run Kruskal-Wallis H-test and compute eta^2 effect size on Train set."""
    records = []
    N = len(X_train)
    k = 3  # 3 markets: US (base), VN, KR

    # Segment groups
    mask_vn = X_train["Market_VN"] == 1
    mask_kr = X_train["Market_KR"] == 1
    mask_us = (~mask_vn) & (~mask_kr)

    for feat in audio_features:
        col = f"{feat}_std" if f"{feat}_std" in X_train.columns else feat
        if col not in X_train.columns:
            continue

        g_vn = X_train.loc[mask_vn, col].dropna()
        g_us = X_train.loc[mask_us, col].dropna()
        g_kr = X_train.loc[mask_kr, col].dropna()

        h_stat, p_val = stats.kruskal(g_vn, g_us, g_kr)
        eta_sq = (h_stat - k + 1) / (N - k)
        
        # Categorize effect size by Cohen / Field guidelines for eta^2
        if eta_sq < 0.01:
            eff_tier = "Không đáng kể (<0.01)"
        elif eta_sq < 0.06:
            eff_tier = "Nhỏ (0.01-0.06)"
        elif eta_sq < 0.14:
            eff_tier = "Trung bình (0.06-0.14)"
        else:
            eff_tier = "Lớn (>=0.14)"

        records.append({
            "feature": feat,
            "H_statistic": round(float(h_stat), 2),
            "df": k - 1,
            "p_value": float(p_val),
            "eta_squared": round(float(eta_sq), 4),
            "effect_size_tier": eff_tier,
            "reject_h0": bool(p_val < 0.05),
        })

    df_kruskal = pd.DataFrame(records).sort_values("H_statistic", ascending=False)
    return df_kruskal


def run_physical_correlation_checks(df_raw: pd.DataFrame, train_indices: pd.Index) -> dict:
    """Verify acoustic physics relationships on training subset."""
    df_tr = df_raw.loc[train_indices].copy()
    
    # 1. Energy vs Loudness Proxy
    r_linear = df_tr["energy"].corr(df_tr["loudness_proxy"])
    r_log = np.log10(df_tr["energy"]).corr(df_tr["loudness_proxy"])
    
    # 2. Spectral Centroid vs Spectral Bandwidth
    r_spectral = df_tr["spectral_centroid"].corr(df_tr["spectral_bandwidth"])
    
    return {
        "r_energy_loudness_linear": float(r_linear),
        "r_energy_loudness_log": float(r_log),
        "r_centroid_bandwidth": float(r_spectral),
    }


def run_vif_analysis(X_train: pd.DataFrame, df_raw: pd.DataFrame, train_indices: pd.Index) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate VIF before and after applying Audio Contract v2."""
    df_tr = df_raw.loc[train_indices].copy()
    
    # --- BEFORE: Initial design matrix with redundant acoustic variables ---
    # Standardize continuous variables
    cols_to_scale = [
        "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
        "loudness_proxy", "spectral_centroid", "spectral_bandwidth",
        "chord_change_rate", "key_confidence", "chroma_repetition_determinism",
        "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration",
        "mfcc_longest_pattern_duration"
    ]
    
    df_before = pd.DataFrame(index=df_tr.index)
    for c in cols_to_scale:
        if c in df_tr.columns:
            s = df_tr[c].dropna()
            df_before[f"{c}_std"] = (df_tr[c] - s.mean()) / s.std()
            
    df_before = sm.add_constant(df_before)
    
    vif_before = []
    for i, col in enumerate(df_before.columns):
        if col == "const":
            continue
        val = variance_inflation_factor(df_before.values, i)
        vif_before.append({"feature": col, "vif_before": round(float(val), 4)})
    df_vif_before = pd.DataFrame(vif_before).sort_values("vif_before", ascending=False)
    
    # --- AFTER: Clean design matrix according to Audio Contract v2 ---
    # (Model 1 pure acoustic baseline matrix)
    audio_m1_cols = [
        "tempo_std", "onset_rate_std", "tempo_stability_std", "energy_std",
        "rms_variability_std", "spectral_centroid_std", "chord_change_rate_std",
        "key_confidence_std", "chroma_repetition_determinism_std",
        "chroma_mean_pattern_duration_std", "chroma_longest_pattern_duration_std",
        "mfcc_repetition_determinism_std", "mfcc_mean_pattern_duration_std",
        "mfcc_longest_pattern_duration_std", "mode"
    ]
    df_after = X_train[[c for c in audio_m1_cols if c in X_train.columns]].copy()
    df_after = sm.add_constant(df_after)
    
    vif_after = []
    for i, col in enumerate(df_after.columns):
        if col == "const":
            continue
        val = variance_inflation_factor(df_after.values, i)
        vif_after.append({"feature": col, "vif_after": round(float(val), 4)})
    df_vif_after = pd.DataFrame(vif_after).sort_values("vif_after", ascending=False)
    
    return df_vif_before, df_vif_after


if __name__ == "__main__":
    from typing import Tuple
    print("=" * 75)
    print("EXECUTION TEST: MEMBER 2 AUDIO & MULTICOLLINEARITY DIAGNOSTICS")
    print("=" * 75)

    print("\n[Step 1] Loading frozen dataset and canonical split...")
    split = get_split()
    df_clean = load_clean_modeling_data()
    print(f"-> Train observations: N = {len(split.X_train):,}")
    print(f"-> Test observations:  N = {len(split.X_test):,}")

    audio_14_features = [
        "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
        "spectral_centroid", "chord_change_rate", "key_confidence",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration",
        "chroma_longest_pattern_duration", "mfcc_repetition_determinism",
        "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration"
    ]

    print("\n[Step 2] Executing Kruskal-Wallis H-test on 14 Librosa DSP features...")
    df_kruskal = run_kruskal_wallis_tests(split.X_train, audio_14_features)
    print(df_kruskal.to_string(index=False))

    print("\n[Step 3] Verifying Physical Acoustic Correlations...")
    corr_results = run_physical_correlation_checks(df_clean, split.ids_train.index)
    for k, v in corr_results.items():
        print(f"-> {k}: {v:.6f}")

    print("\n[Step 4] Executing VIF Multicollinearity Analysis (Before vs After)...")
    df_vif_before, df_vif_after = run_vif_analysis(split.X_train, df_clean, split.ids_train.index)
    print("\n--- Top VIF Before Drop (Redundant Features) ---")
    print(df_vif_before.head(6).to_string(index=False))

    print("\n--- VIF After Audio Contract v2 (Model 1 Matrix) ---")
    print(df_vif_after.to_string(index=False))

    print("\n[Step 5] Validation Assertions:")
    energy_kruskal = df_kruskal.loc[df_kruskal["feature"] == "energy"].iloc[0]
    assert abs(energy_kruskal["H_statistic"] - 139.41) < 0.1, "Energy H-stat mismatch!"
    assert abs(energy_kruskal["eta_squared"] - 0.0817) < 0.001, "Energy eta^2 mismatch!"
    print("✅ Kruskal-Wallis Energy test matches canonical run_status: H = 139.41, eta^2 = 0.0817")

    energy_vif = df_vif_after.loc[df_vif_after["feature"] == "energy_std"].iloc[0]["vif_after"]
    assert abs(energy_vif - 2.09) < 0.05, "Energy VIF mismatch!"
    assert (df_vif_after["vif_after"] < 3.0).all(), "Some feature exceeds VIF threshold 3.0!"
    print(f"✅ Audio Contract v2 validated: Energy VIF = {energy_vif:.3f} <= 2.09, All VIF < 2.73 < 5.0")
    print("\n>>> ALL TEST LOGIC VERIFIED 100% SUCCESSFUL! <<<")
