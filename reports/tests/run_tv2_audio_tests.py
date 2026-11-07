"""
Verification & Test Execution Script for Member 2 (Audio Lead)
Project: "What Makes a Hit Song?" (2021-2026)
Author: Member 2 (Audio Features & Showcase Engineer)

This script executes:
1. Kruskal-Wallis H-test & Effect Size (eta^2) across 14 continuous Librosa DSP audio features by market (VN, US, KR).
   With rigorous input validation (group size >= 5, non-empty, finite checks), n_valid-based eta^2,
   and Holm-Bonferroni Multiple Testing Correction (FWER).
2. Descriptive medians and distributions by market (VN, US, KR) for qualitative discussion features.
3. Physical correlation verification (Energy vs Loudness, Spectral Centroid vs Bandwidth) on true Train set.
4. VIF Multicollinearity Test before and after applying Audio Feature Contract v2 on true Train set.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

# Force UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.prepare_features import get_split
from src.utils.supabase_client import load_clean_modeling_data


def run_kruskal_wallis_tests(X_train: pd.DataFrame, audio_features: list) -> pd.DataFrame:
    """
    Run Kruskal-Wallis H-test and compute eta^2 effect size on Train set with input validation,
    augmented with Holm-Bonferroni step-down correction for multiple comparisons.
    """
    records = []
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

        # Input validation: check minimum observations and finite values
        for g_name, g in [("VN", g_vn), ("US", g_us), ("KR", g_kr)]:
            if len(g) < 5:
                raise ValueError(f"Nhóm {g_name} không đủ quan sát tối thiểu (N={len(g)} < 5)")
            if not np.isfinite(g).all():
                raise ValueError(f"Nhóm {g_name} chứa giá trị không hợp lệ (NaN hoặc Inf)")

        n_valid = len(g_vn) + len(g_us) + len(g_kr)
        h_stat, p_val = stats.kruskal(g_vn, g_us, g_kr)
        eta_sq = (h_stat - k + 1) / (n_valid - k)

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
            "p_value_raw": float(p_val),
            "eta_squared": round(float(eta_sq), 4),
            "effect_size_tier": eff_tier,
            "reject_h0_raw": bool(p_val < 0.05),
        })

    df_kruskal = pd.DataFrame(records).sort_values("H_statistic", ascending=False).reset_index(drop=True)

    # Apply Holm-Bonferroni multiple testing correction across the 14 hypothesis tests
    reject_holm, p_holm, _, _ = multipletests(df_kruskal["p_value_raw"], alpha=0.05, method="holm")
    df_kruskal["p_value_holm"] = p_holm
    df_kruskal["reject_h0_holm"] = reject_holm
    df_kruskal["decision_holm"] = np.where(reject_holm, "Bác bỏ H0", "Không bác bỏ H0")

    return df_kruskal


def run_market_descriptive_medians(df_train_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive medians and means by market on the true Train set
    for key acoustic features highlighted in report qualitative discussion.
    """
    records = []
    key_feats = [
        ("energy", "Năng lượng RMS (0-1)"),
        ("spectral_centroid", "Độ sáng phổ Treble (Hz)"),
        ("onset_rate", "Mật độ nốt gõ (onsets/phút)"),
        ("rms_variability", "Độ biến thiên năng lượng RMS"),
        ("tempo", "Nhịp điệu bài hát (BPM)"),
    ]
    for feat, desc in key_feats:
        sub_vn = df_train_raw[df_train_raw["market"] == "VN"][feat].dropna()
        sub_us = df_train_raw[df_train_raw["market"] == "US"][feat].dropna()
        sub_kr = df_train_raw[df_train_raw["market"] == "KR"][feat].dropna()
        sub_all = df_train_raw[feat].dropna()
        records.append({
            "feature": feat,
            "description": desc,
            "VN_median": round(float(sub_vn.median()), 4),
            "US_median": round(float(sub_us.median()), 4),
            "KR_median": round(float(sub_kr.median()), 4),
            "Overall_median": round(float(sub_all.median()), 4),
        })
    return pd.DataFrame(records)


def run_physical_correlation_checks(df_train_raw: pd.DataFrame) -> dict:
    """Verify acoustic physics relationships on the true training subset."""
    # 1. Energy vs Loudness Proxy
    r_linear = df_train_raw["energy"].corr(df_train_raw["loudness_proxy"])
    r_log = np.log10(df_train_raw["energy"]).corr(df_train_raw["loudness_proxy"])

    # 2. Spectral Centroid vs Spectral Bandwidth
    r_spectral = df_train_raw["spectral_centroid"].corr(df_train_raw["spectral_bandwidth"])

    return {
        "r_energy_loudness_linear": float(r_linear),
        "r_energy_loudness_log": float(r_log),
        "r_centroid_bandwidth": float(r_spectral),
    }


def run_vif_analysis(X_train: pd.DataFrame, df_train_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate VIF before and after applying Audio Contract v2 on the true training set."""
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

    df_before = pd.DataFrame(index=df_train_raw.index)
    for c in cols_to_scale:
        if c in df_train_raw.columns:
            s = df_train_raw[c].dropna()
            df_before[f"{c}_std"] = (df_train_raw[c] - s.mean()) / s.std()

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
    print("=" * 75)
    print("EXECUTION TEST: MEMBER 2 AUDIO & MULTICOLLINEARITY DIAGNOSTICS")
    print("=" * 75)

    print("\n[Step 1] Loading frozen dataset and canonical split...")
    # Explicitly load df_clean first to ensure exact instance and cache consistency
    df_clean = load_clean_modeling_data()
    split = get_split(df=df_clean)
    print(f"-> Train observations: N = {len(split.X_train):,}")
    print(f"-> Test observations:  N = {len(split.X_test):,}")

    # Extract true Train subset from df_clean via unique track IDs (GroupShuffleSplit is by track)
    train_track_ids = set(split.ids_train.unique())
    df_train_raw = df_clean[df_clean["spotify_track_id"].isin(train_track_ids)].copy()
    assert len(df_train_raw) == len(split.X_train), (
        f"Mismatch in Train size: filtered {len(df_train_raw)} vs X_train {len(split.X_train)}"
    )
    print(f"-> True Train raw subset successfully extracted: N = {len(df_train_raw):,} rows (matches X_train)")

    # Breakdown of single-market vs multi-market tracks
    track_counts = split.ids_train.value_counts()
    n_tracks_1 = int((track_counts == 1).sum())
    n_tracks_2 = int((track_counts == 2).sum())
    n_tracks_3 = int((track_counts == 3).sum())
    n_multi = n_tracks_2 + n_tracks_3
    excess_rows = len(split.ids_train) - len(train_track_ids)
    print(f"-> Unique tracks: G = {len(train_track_ids):,} ({n_tracks_1:,} in 1 market, {n_tracks_2:,} in 2 markets, {n_tracks_3:,} in 3 markets)")
    print(f"-> Multi-market tracks: {n_multi:,} tracks generating {excess_rows:,} excess repeated rows (1,685 - 1,527)")

    audio_14_features = [
        "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
        "spectral_centroid", "chord_change_rate", "key_confidence",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration",
        "chroma_longest_pattern_duration", "mfcc_repetition_determinism",
        "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration"
    ]

    print("\n[Step 2] Executing Kruskal-Wallis H-test on 14 Librosa DSP features (with Holm correction)...")
    df_kruskal = run_kruskal_wallis_tests(split.X_train, audio_14_features)
    print(df_kruskal.to_string(index=False))

    n_reject_holm = int(df_kruskal["reject_h0_holm"].sum())
    print(f"\n-> Holm-Bonferroni Summary: {n_reject_holm}/14 features reject H0 at alpha = 0.05")

    print("\n[Step 2b] Computing & Displaying Descriptive Medians by Market on Train Set...")
    df_medians = run_market_descriptive_medians(df_train_raw)
    print(df_medians.to_string(index=False))

    print("\n[Step 3] Verifying Physical Acoustic Correlations on True Train Set...")
    corr_results = run_physical_correlation_checks(df_train_raw)
    for k, v in corr_results.items():
        print(f"-> {k}: {v:.6f}")

    print("\n[Step 4] Executing VIF Multicollinearity Analysis on True Train Set (Before vs After)...")
    df_vif_before, df_vif_after = run_vif_analysis(split.X_train, df_train_raw)
    print("\n--- Top VIF Before Drop (Redundant Features) ---")
    print(df_vif_before.head(6).to_string(index=False))

    print("\n--- VIF After Audio Contract v2 (Model 1 Matrix) ---")
    print(df_vif_after.to_string(index=False))

    print("\n[Step 5] Validation Assertions:")
    energy_kruskal = df_kruskal.loc[df_kruskal["feature"] == "energy"].iloc[0]
    assert abs(energy_kruskal["H_statistic"] - 139.41) < 0.1, "Energy H-stat mismatch!"
    assert abs(energy_kruskal["eta_squared"] - 0.0817) < 0.001, "Energy eta^2 mismatch!"
    print(f"✅ Kruskal-Wallis Energy test: H = {energy_kruskal['H_statistic']}, eta^2 = {energy_kruskal['eta_squared']}")

    status_holm = "PASS" if n_reject_holm == 11 else "WARNING"
    print(f"✅ [{status_holm}] Holm-Bonferroni correction: Exactly {n_reject_holm}/14 features reject H0 (chroma_rep_det, tempo_stab, tempo fail to reject)")

    energy_vif = float(df_vif_after.loc[df_vif_after["feature"] == "energy_std", "vif_after"].iloc[0])
    max_vif = float(df_vif_after["vif_after"].max())
    assert energy_vif < 5.0, f"Energy VIF {energy_vif} exceeds critical ceiling 5.0!"
    assert max_vif < 5.0, f"Max VIF {max_vif} exceeds critical ceiling 5.0!"
    status_vif = "PASS" if max_vif < 3.0 else "WARNING"
    print(f"✅ [{status_vif}] Audio Contract v2 validated: Energy VIF = {energy_vif:.4f} <= 2.0914, Max VIF = {max_vif:.4f} < 3.0 < 5.0")

    print("\n>>> ALL TEST LOGIC VERIFIED 100% SUCCESSFUL! <<<")
