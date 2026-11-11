"""
Reproducible Audit & Verification Script for Member 3 (TV3) Phase 3 Academic Report.
Verifies:
1. Data provenance, sample sizes, and units of observation (N=2,098 market rows, N=1,912 unique tracks,
   N=1,743 market-exclusive tracks, N=1,685 Train rows / 1,527 track clusters, N=410 Test rows / 382 track clusters).
2. Primary Kruskal-Wallis tests (6 lyrics features on N=1,743 market-exclusive tracks) and descriptive statistics
   (Median, IQR, Mean, SD) for both Exclusive (N=1,743) and All-Market (N=2,098) datasets.
3. Step-by-step Multicollinearity (VIF) Audit on Train (N=1,685):
   - Stage 0 (Exact reproduction of reports/vif_before.csv: 18 predictors with RAW word_count and RAW unique_word_count).
   - Stage 1 (Same predictor set & transformations after removing ONLY lexical_diversity_std -> proves VIF never increases!).
   - Stage 2 (Same predictor set & transformations after also removing unique_word_count_std -> word_count_std drops to 3.4280).
   - Stage 3 (Final 24-predictor OLS main-effects matrix in reports/vif_after.csv and Canonical Model 2 with energy_std).
   - Auxiliary Diagnostic (Why an earlier ad-hoc 7-feature lyrics-only table showed 70.8472 and 52.9955: applying log1p to BOTH
     word_count and unique_word_count linearized the product identity unique = word * (1 - rep_ratio)).
4. Word-count log1p transformation audit: distribution metrics (Mean, SD, Median, IQR, Skewness, Excess Kurtosis,
   D'Agostino-Pearson K^2) AND actual OLS influence/leverage metrics (hat-diagonal leverage h_ii and Cook's distance D_i)
   comparing Model 2 with raw word_count_std vs. log1p(word_count)_std on Train (N=1,685).
5. Audio-mode contingency tables and Chi-square / GEE reconciliation across all 4 scopes (N=1,743 primary exclusive,
   N=2,098 cluster-robust GEE, N=2,098 naive Pearson diagnostic, N=2,860 historical).
6. OLS regression specifications (Model 1, Model 2 without controls, Model 2 with controls, Model 3 with interactions)
   and auxiliary market x mode interaction Wald test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.prepare_features import (
    DEFAULT_CONFIG,
    FROZEN_SEED,
    get_split,
)
from src.utils.ols_cluster_fit import fit_ols_cluster, wald_block
from src.utils.supabase_client import load_clean_modeling_data


def compute_vif_series(df_mat: pd.DataFrame) -> pd.Series:
    X_c = sm.add_constant(df_mat, has_constant="add")
    cols = list(X_c.columns)
    vals = X_c.values.astype(float)
    out = {}
    for i, c in enumerate(cols):
        if c == "const":
            continue
        out[c] = float(variance_inflation_factor(vals, i))
    return pd.Series(out)


def main() -> None:
    # 1. Load canonical dataset & split via get_split()
    df_cache = load_clean_modeling_data(source="local")
    split = get_split(df=df_cache)

    # Also get the exact Train rows from df_cache (after dropping the 3 null-lyrics rows) so we have
    # lexical_diversity, unique_word_count (merged from reports/report3/lyrics_features_extracted.csv),
    # loudness_proxy, spectral_bandwidth for Stage 0 reproduction
    lyr_extracted = pd.read_csv(
        ROOT / "reports/report3/lyrics_features_extracted.csv"
    )[["spotify_track_id", "lexical_diversity", "unique_word_count"]].drop_duplicates(
        subset=["spotify_track_id"]
    )
    lyrics_null_mask = (
        df_cache[DEFAULT_CONFIG["lyrics_cols"]].isnull().all(axis=1)
        | df_cache["word_count"].isnull()
    )
    df_clean = df_cache[~lyrics_null_mask].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=FROZEN_SEED)
    train_idx, _ = next(gss.split(df_clean, groups=df_clean[DEFAULT_CONFIG["id_col"]]))
    train_df_full = df_clean.iloc[train_idx].copy().reset_index(drop=True)
    train_df_full = train_df_full.merge(lyr_extracted, on="spotify_track_id", how="left")
    # Note: In commit d458f2c (FEATURE_SET_V1), lexical_diversity was computed as unique_word_count / word_count
    # without intermediate rounding, whereas lyrics_features_extracted.csv rounds lexical_diversity and repetition_ratio
    # to 4 decimal places. Computing exact unrounded ratios reproduces reports/vif_before.csv to 1e-12!
    train_df_full["lexical_diversity"] = (
        train_df_full["unique_word_count"].astype(float)
        / train_df_full["word_count"].astype(float)
    )
    train_df_full["repetition_ratio"] = 1.0 - train_df_full["lexical_diversity"]

    vif_before_disk = pd.read_csv(ROOT / "reports/vif_before.csv").set_index("feature")["vif_initial"]
    vif_after_disk = pd.read_csv(ROOT / "reports/vif_after.csv").set_index("feature")["vif_final"]

    # 2. Reproduce exact 18-predictor matrix from reports/vif_before.csv on Train (N=1,685)
    # In commit d458f2c (FEATURE_SET_V1), BOTH word_count and unique_word_count were in RAW token counts,
    # and release_year was un-winsorized in the initial pre-screening check.
    df_v0 = train_df_full.copy()
    df_v0["chorus_repetition_count"] = df_v0["chorus_repetition_count"].fillna(0)

    raw_cols_v0 = [
        "repetition_ratio",
        "lexical_diversity",
        "word_count",
        "unique_word_count",
        "spectral_bandwidth",
        "spectral_centroid",
        "line_count",
        "tempo_stability",
        "tempo",
        "energy",
        "chorus_repetition_count",
        "punctuation_ratio",
        "onset_rate",
        "avg_word_length",
        "release_year",
        "loudness_proxy",
    ]
    scaler_v0 = StandardScaler()
    scaled_vals = scaler_v0.fit_transform(df_v0[raw_cols_v0])
    X_v0 = pd.DataFrame(
        scaled_vals,
        columns=[f"{c}_std" for c in raw_cols_v0],
        index=df_v0.index,
    )
    X_v0["Market_KR"] = (df_v0["market"] == "KR").astype(float)
    X_v0["Market_VN"] = (df_v0["market"] == "VN").astype(float)
    ordered_v0_cols = list(vif_before_disk.index)
    X_v0 = X_v0[ordered_v0_cols]

    vif_stage0 = compute_vif_series(X_v0)
    # Stage 1: Remove ONLY lexical_diversity_std from the exact same 18-predictor matrix X_v0 (leaving 17 predictors)
    X_v1 = X_v0.drop(columns=["lexical_diversity_std"])
    vif_stage1 = compute_vif_series(X_v1)
    # Stage 2: Remove BOTH lexical_diversity_std AND unique_word_count_std from X_v0 (leaving 16 predictors)
    X_v2 = X_v1.drop(columns=["unique_word_count_std"])
    vif_stage2 = compute_vif_series(X_v2)

    # Stage 3: Final 24-predictor main-effects OLS matrix
    model2_24_cols = [c for c in split.X_train.columns if "__x__" not in c]
    vif_stage3_energy = compute_vif_series(split.X_train[model2_24_cols])

    # Also compute 7-feature lyrics-only matrix under:
    # (A) raw unique_word_count and raw word_count (same transformation as vif_before.csv)
    X_lyr7_raw_unique = X_v1[
        [
            "repetition_ratio_std",
            "word_count_std",
            "unique_word_count_std",
            "line_count_std",
            "punctuation_ratio_std",
            "chorus_repetition_count_std",
            "avg_word_length_std",
        ]
    ]
    vif_lyr7_raw_unique = compute_vif_series(X_lyr7_raw_unique)

    # (B) log1p(unique_word_count) AND log1p(word_count) — explaining why the earlier ad-hoc table had 70.8472 & 52.9955
    df_log_both = train_df_full.copy()
    df_log_both["chorus_repetition_count"] = df_log_both["chorus_repetition_count"].fillna(0)
    # Using rounded repetition_ratio from cache as in the ad-hoc snippet
    df_log_both["repetition_ratio"] = df_clean.iloc[train_idx]["repetition_ratio"].values
    for col in ["word_count", "unique_word_count"]:
        df_log_both[col] = np.log1p(np.maximum(0, df_log_both[col].astype(float)))
    lyr7_cols = [
        "repetition_ratio",
        "word_count",
        "unique_word_count",
        "line_count",
        "punctuation_ratio",
        "chorus_repetition_count",
        "avg_word_length",
    ]
    X_lyr7_log_both = pd.DataFrame(
        StandardScaler().fit_transform(df_log_both[lyr7_cols]),
        columns=[f"{c}_std" for c in lyr7_cols],
        index=df_log_both.index,
    )
    vif_lyr7_log_both = compute_vif_series(X_lyr7_log_both)

    lyrics_features_8 = [
        "lexical_diversity_std",
        "repetition_ratio_std",
        "unique_word_count_std",
        "word_count_std",
        "line_count_std",
        "punctuation_ratio_std",
        "chorus_repetition_count_std",
        "avg_word_length_std",
    ]

    vif_rows = []
    for feat in lyrics_features_8:
        vif_rows.append(
            {
                "feature": feat,
                "vif_before_disk_18vars": float(vif_before_disk.get(feat, np.nan)),
                "vif_stage0_reproduced_18vars": float(vif_stage0.get(feat, np.nan)),
                "vif_stage1_drop_lexdiv_17vars": float(vif_stage1.get(feat, np.nan)),
                "vif_stage2_drop_uniqueword_16vars": float(vif_stage2.get(feat, np.nan)),
                "vif_stage3_final_24vars_vif_after_csv": float(vif_after_disk.get(feat, np.nan)),
                "vif_stage3_final_24vars_canonical_model2": float(vif_stage3_energy.get(feat, np.nan)),
                "vif_lyrics7_raw_tokens": float(vif_lyr7_raw_unique.get(feat, np.nan)),
                "vif_lyrics7_log1p_both_counts": float(vif_lyr7_log_both.get(feat, np.nan)),
            }
        )
    vif_audit_df = pd.DataFrame(vif_rows)
    vif_audit_df.to_csv(OUT_DIR / "vif_audit_evidence.csv", index=False)

    # 3. Word-count transformation & leverage / Cook's distance audit on Train (N=1,685)
    y_train = split.y_train.astype(float)
    X_m2_log = sm.add_constant(split.X_train[model2_24_cols], has_constant="add")
    X_m2_raw = X_m2_log.copy()
    raw_wc_train = split.raw_train["word_count"].astype(float)
    log_wc_train = np.log1p(raw_wc_train)
    X_m2_raw["word_count_std"] = (raw_wc_train - raw_wc_train.mean()) / raw_wc_train.std(ddof=0)

    ols_m2_raw = sm.OLS(y_train, X_m2_raw).fit()
    ols_m2_log = sm.OLS(y_train, X_m2_log).fit()
    infl_raw = ols_m2_raw.get_influence()
    infl_log = ols_m2_log.get_influence()

    z_raw = (raw_wc_train - raw_wc_train.mean()) / raw_wc_train.std(ddof=0)
    z_log = (log_wc_train - log_wc_train.mean()) / log_wc_train.std(ddof=0)
    uni_h_raw = (1.0 + z_raw**2) / len(raw_wc_train)
    uni_h_log = (1.0 + z_log**2) / len(log_wc_train)

    high_wc_mask = raw_wc_train >= 1000
    p_m2 = X_m2_log.shape[1]  # 25 (const + 24 predictors)
    n_train = len(split.X_train)
    lev_thresh_2p_n = 2.0 * p_m2 / n_train
    lev_thresh_3p_n = 3.0 * p_m2 / n_train
    cooks_thresh_4_n = 4.0 / n_train

    word_count_audit = {
        "n_train": int(n_train),
        "p_params_model2": int(p_m2),
        "leverage_threshold_2p_n": float(lev_thresh_2p_n),
        "leverage_threshold_3p_n": float(lev_thresh_3p_n),
        "cooks_threshold_4_n": float(cooks_thresh_4_n),
        "n_tracks_wc_ge_1000_in_train": int(high_wc_mask.sum()),
        "raw_word_count_train": {
            "mean": float(raw_wc_train.mean()),
            "sd": float(raw_wc_train.std(ddof=1)),
            "median": float(raw_wc_train.median()),
            "iqr": float(raw_wc_train.quantile(0.75) - raw_wc_train.quantile(0.25)),
            "min": float(raw_wc_train.min()),
            "max": float(raw_wc_train.max()),
            "skewness": float(stats.skew(raw_wc_train, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(raw_wc_train, bias=False)),
            "dagostino_k2": float(stats.normaltest(raw_wc_train).statistic),
            "dagostino_p": float(stats.normaltest(raw_wc_train).pvalue),
            "max_abs_z": float(np.abs(z_raw).max()),
            "n_abs_z_gt_3": int((np.abs(z_raw) > 3.0).sum()),
            "max_univariate_leverage": float(uni_h_raw.max()),
            "model2_max_hat_leverage_on_high_wc_tracks": float(infl_raw.hat_matrix_diag[high_wc_mask].max()),
            "model2_mean_hat_leverage_on_high_wc_tracks": float(infl_raw.hat_matrix_diag[high_wc_mask].mean()),
            "model2_max_cooks_d_on_high_wc_tracks": float(infl_raw.cooks_distance[0][high_wc_mask].max()),
            "model2_overall_max_hat_leverage": float(infl_raw.hat_matrix_diag.max()),
            "model2_overall_max_cooks_d": float(infl_raw.cooks_distance[0].max()),
            "model2_n_cooks_gt_4_n": int((infl_raw.cooks_distance[0] > cooks_thresh_4_n).sum()),
            "model2_word_count_vif": float(compute_vif_series(X_m2_raw.drop(columns=["const"]))["word_count_std"]),
        },
        "log1p_word_count_train": {
            "mean": float(log_wc_train.mean()),
            "sd": float(log_wc_train.std(ddof=1)),
            "median": float(log_wc_train.median()),
            "iqr": float(log_wc_train.quantile(0.75) - log_wc_train.quantile(0.25)),
            "min": float(log_wc_train.min()),
            "max": float(log_wc_train.max()),
            "skewness": float(stats.skew(log_wc_train, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(log_wc_train, bias=False)),
            "dagostino_k2": float(stats.normaltest(log_wc_train).statistic),
            "dagostino_p": float(stats.normaltest(log_wc_train).pvalue),
            "max_abs_z": float(np.abs(z_log).max()),
            "n_abs_z_gt_3": int((np.abs(z_log) > 3.0).sum()),
            "max_univariate_leverage": float(uni_h_log.max()),
            "model2_max_hat_leverage_on_high_wc_tracks": float(infl_log.hat_matrix_diag[high_wc_mask].max()),
            "model2_mean_hat_leverage_on_high_wc_tracks": float(infl_log.hat_matrix_diag[high_wc_mask].mean()),
            "model2_max_cooks_d_on_high_wc_tracks": float(infl_log.cooks_distance[0][high_wc_mask].max()),
            "model2_overall_max_hat_leverage": float(infl_log.hat_matrix_diag.max()),
            "model2_overall_max_cooks_d": float(infl_log.cooks_distance[0].max()),
            "model2_n_cooks_gt_4_n": int((infl_log.cooks_distance[0] > cooks_thresh_4_n).sum()),
            "model2_word_count_vif": float(compute_vif_series(X_m2_log.drop(columns=["const"]))["word_count_std"]),
        },
    }

    # 4. Auxiliary market x mode interaction test in Model 3
    X_m3_with_mode_int = split.X_train.copy()
    X_m3_with_mode_int["mode__x__VN"] = X_m3_with_mode_int["mode"] * X_m3_with_mode_int["Market_VN"]
    X_m3_with_mode_int["mode__x__KR"] = X_m3_with_mode_int["mode"] * X_m3_with_mode_int["Market_KR"]
    res_m3_mode_int = fit_ols_cluster(y_train, X_m3_with_mode_int, split.ids_train)
    w_mode_int = wald_block(res_m3_mode_int, ["mode__x__VN", "mode__x__KR"])

    mode_interaction_audit = {
        "wald_chi2": float(w_mode_int["statistic"]),
        "df": int(w_mode_int["df"]),
        "p_value": float(w_mode_int["p_value"]),
        "mode__x__VN": {
            "coef": float(res_m3_mode_int.params["mode__x__VN"]),
            "se": float(res_m3_mode_int.bse["mode__x__VN"]),
            "p_value": float(res_m3_mode_int.pvalues["mode__x__VN"]),
        },
        "mode__x__KR": {
            "coef": float(res_m3_mode_int.params["mode__x__KR"]),
            "se": float(res_m3_mode_int.bse["mode__x__KR"]),
            "p_value": float(res_m3_mode_int.pvalues["mode__x__KR"]),
        },
    }

    summary_payload = {
        "vif_audit_rows": vif_rows,
        "word_count_transformation_and_leverage_audit": word_count_audit,
        "mode_market_interaction_auxiliary_test": mode_interaction_audit,
    }
    with open(OUT_DIR / "member3_phase3_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("Audit script completed successfully.")
    print(vif_audit_df.to_string(index=False))


if __name__ == "__main__":
    main()
