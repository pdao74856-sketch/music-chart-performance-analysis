#!/usr/bin/env python3
"""
src/modeling/train_model4_rf_benchmark.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Pipeline, Supabase, Benchmark & Deploy Lead)
Specification: Plan v2 (Dual-Model Methodology: Random Forest Benchmark + GroupKFold)

Responsibilities:
  - Trains Model 4: Random Forest Benchmark using exact frozen Train/Test split (N=2,098).
  - Uses all 17 Librosa audio features (including energy, loudness_proxy, spectral_bandwidth)
    + 6 NLP lyrics features (log1p word_count) + context controls (Market dummies, release_year).
  - 5-fold Cross-Validation using GroupKFold(groups=spotify_track_id) on the Train set.
  - Hyperparameter tuning across max_depth: [6, 8, 10] and min_samples_leaf: [2, 5].
  - Evaluates performance metrics (MAE, RMSE, R²) overall and by market (VN, US, KR).
  - Computes Permutation Feature Importance on the held-out Test set.
  - Updates models/test_predictions.csv with pred_rf.
  - Exports:
      1. models/rf_pipeline.pkl
      2. reports/figures/rf_permutation_importance.png
      3. reports/report5/rf_benchmark_metrics.json
      4. Updates models/model_metadata.json with rf_metrics and rf_best_params
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.supabase_client import load_clean_modeling_data

# Directories setup
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORT5_DIR = ROOT_DIR / "reports" / "report5"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT5_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# FEATURE CONTRACT FOR RANDOM FOREST (MODEL 4)
# Note: Random Forest is non-linear and tree-based, so it safely includes:
#   - Both energy and loudness_proxy (no multicollinearity sensitivity)
#   - Both spectral_centroid and spectral_bandwidth
#   - Raw non-standardized values (easier tree splits and visual interpretability)
# ==============================================================================
RF_AUDIO_FEATURES = [
    "tempo",
    "onset_rate",
    "tempo_stability",
    "energy",
    "loudness_proxy",
    "rms_variability",
    "spectral_centroid",
    "spectral_bandwidth",
    "chord_change_rate",
    "key_confidence",
    "mode",
    "chroma_repetition_determinism",
    "chroma_mean_pattern_duration",
    "chroma_longest_pattern_duration",
    "mfcc_repetition_determinism",
    "mfcc_mean_pattern_duration",
    "mfcc_longest_pattern_duration",
]

RF_LYRICS_FEATURES = [
    "log_word_count",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "punctuation_ratio",
]

RF_CONTROL_FEATURES = [
    "release_year",
    "Market_VN",
    "Market_KR",
]

RF_ALL_FEATURES = RF_AUDIO_FEATURES + RF_LYRICS_FEATURES + RF_CONTROL_FEATURES


def load_dataset() -> pd.DataFrame:
    """Loads clean modeling data from Supabase (or local fallback)."""
    print("[TrainModel4] Fetching clean modeling dataset (view_clean_modeling_dataset)...")
    df = load_clean_modeling_data(source="auto")
    print(f"[TrainModel4] Loaded {len(df):,} total records.")
    return df


def preprocess_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses features for Random Forest without standardization or target-data leakage."""
    df_proc = df.copy()

    # 1. Mode column harmonization (Major=1.0, Minor=0.0)
    if "mode" in df_proc.columns:
        df_proc["mode"] = df_proc["mode"].apply(lambda x: 1.0 if str(x).lower() in ("1", "major", "1.0") else 0.0)
    elif "audio_mode" in df_proc.columns:
        df_proc["mode"] = df_proc["audio_mode"].apply(lambda x: 1.0 if str(x).lower() in ("1", "major", "1.0") else 0.0)

    # 2. Log-transform word_count (log1p)
    if "word_count" in df_proc.columns:
        df_proc["log_word_count"] = np.log1p(df_proc["word_count"].astype(float).fillna(0.0))

    # 3. Market dummies (Base: US -> Market_VN, Market_KR)
    if "market" in df_proc.columns:
        mkt_upper = df_proc["market"].astype(str).str.upper()
        df_proc["Market_VN"] = (mkt_upper == "VN").astype(float)
        df_proc["Market_KR"] = (mkt_upper == "KR").astype(float)

    # 4. Domain-specific structural fill (chorus_repetition_count with 0 for songs without detected chorus)
    if "chorus_repetition_count" in df_proc.columns:
        df_proc["chorus_repetition_count"] = df_proc["chorus_repetition_count"].fillna(0.0)

    return df_proc


def train_and_evaluate_rf():
    """Main training, cross-validation, evaluation, and artifact export pipeline."""
    # 1. Load data
    df_raw = load_dataset()
    df_proc = preprocess_rf_features(df_raw)

    # 2. Align with frozen train/test split from models/test_track_ids.json
    test_json_path = MODELS_DIR / "test_track_ids.json"
    if not test_json_path.exists():
        raise FileNotFoundError(f"[ERROR] Required {test_json_path} not found! Run prepare_features.py first.")

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_meta = json.load(f)
    test_ids = set(test_meta["track_ids"] if isinstance(test_meta, dict) else test_meta)

    # Partition into Train and Test
    test_mask = df_proc["spotify_track_id"].isin(test_ids)
    df_train = df_proc[~test_mask].copy().reset_index(drop=True)
    df_test = df_proc[test_mask].copy().reset_index(drop=True)

    print(f"\n[TrainModel4] Frozen Split: Train={len(df_train):,} rows ({df_train['spotify_track_id'].nunique():,} songs), "
          f"Test={len(df_test):,} rows ({df_test['spotify_track_id'].nunique():,} songs)")

    X_train = df_train[RF_ALL_FEATURES]
    y_train = df_train["chart_score"].astype(float)
    groups_train = df_train["spotify_track_id"]

    X_test = df_test[RF_ALL_FEATURES]
    y_test = df_test["chart_score"].astype(float)

    # 3. 5-fold GroupKFold Cross-Validation on Train set using Leak-Free Pipeline
    print("\n[TrainModel4] Running GridSearchCV with 5-fold GroupKFold and Leak-free Pipeline...")
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
    param_grid = {
        "rf__max_depth": [6, 8, 10],
        "rf__min_samples_leaf": [2, 5],
        "rf__n_estimators": [150],
    }

    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline: Pipeline = grid_search.best_estimator_
    best_rf: RandomForestRegressor = best_pipeline.named_steps["rf"]
    best_params = {k.replace("rf__", ""): v for k, v in grid_search.best_params_.items()}
    best_cv_mae = -grid_search.best_score_
    print(f"\n[TrainModel4] Best Hyperparameters: {best_params}")
    print(f"[TrainModel4] Best CV MAE: {best_cv_mae:.2f} points")

    # 4. Evaluation on Test Set (Overall & Broken Down by Market)
    y_pred_test = best_pipeline.predict(X_test)

    mae_overall = float(mean_absolute_error(y_test, y_pred_test))
    rmse_overall = float(root_mean_squared_error(y_test, y_pred_test))
    r2_overall = float(r2_score(y_test, y_pred_test))

    print("\n================== TEST SET PERFORMANCE (MODEL 4 RF) ==================")
    print(f"Overall Test Set (N={len(y_test)}):")
    print(f"  -> R² Score: {r2_overall:.4f}")
    print(f"  -> MAE:      {mae_overall:.2f} points")
    print(f"  -> RMSE:     {rmse_overall:.2f} points")

    market_metrics = {}
    for m in ["VN", "US", "KR"]:
        mask_m = (df_test["market"] == m)
        if mask_m.sum() > 0:
            y_true_m = y_test[mask_m]
            y_pred_m = y_pred_test[mask_m]
            mae_m = float(mean_absolute_error(y_true_m, y_pred_m))
            rmse_m = float(root_mean_squared_error(y_true_m, y_pred_m))
            r2_m = float(r2_score(y_true_m, y_pred_m))
            market_metrics[m] = {
                "n_samples": int(mask_m.sum()),
                "mae": round(mae_m, 2),
                "rmse": round(rmse_m, 2),
                "r2": round(r2_m, 4),
            }
            print(f"Market {m} (N={mask_m.sum()}): MAE={mae_m:.2f}, RMSE={rmse_m:.2f}, R²={r2_m:.4f}")

    # 5. Permutation Feature Importance on Test Set
    # Note: Evaluated on best_pipeline (row-wise test permutation importance)
    print("\n[TrainModel4] Computing Permutation Importance on Test Set...")
    perm_result = permutation_importance(
        best_pipeline, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    df_importance = pd.DataFrame({
        "feature": RF_ALL_FEATURES,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std,
    }).sort_values(by="importance", ascending=True)

    # Export Permutation Plot
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid")
    df_top_plot = df_importance.tail(15)
    plt.barh(
        df_top_plot["feature"],
        df_top_plot["importance"],
        xerr=df_top_plot["std"],
        color="#1f77b4",
        edgecolor="#0b4069",
        capsize=3,
        alpha=0.88
    )
    plt.title("Model 4: Random Forest Permutation Feature Importance (Test Set)\nMetric: Increase in MAE upon Permutation", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Mean Importance (MAE degradation when permuted)", fontsize=11)
    plt.ylabel("Features", fontsize=11)
    plt.tight_layout()

    plot_path = FIGURES_DIR / "rf_permutation_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[TrainModel4] Saved Permutation Importance plot to {plot_path.relative_to(ROOT_DIR)}")

    # 6. Save Model Pipeline Binary (includes imputer + RF)
    model_path = MODELS_DIR / "rf_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"[TrainModel4] Serialized Random Forest Pipeline to {model_path.relative_to(ROOT_DIR)}")

    # 7. Update models/test_predictions.csv with pred_rf
    pred_csv_path = MODELS_DIR / "test_predictions.csv"
    if pred_csv_path.exists():
        df_pred = pd.read_csv(pred_csv_path)
        # Create map from (spotify_track_id, market) -> pred_rf
        pred_map = {}
        for _, row in df_test.iterrows():
            key = (row["spotify_track_id"], row["market"])
            idx_in_test = _
            pred_map[key] = y_pred_test[idx_in_test]

        df_pred["pred_rf"] = df_pred.apply(
            lambda r: pred_map.get((r["spotify_track_id"], r["market"]), np.nan), axis=1
        )
        df_pred.to_csv(pred_csv_path, index=False)
        print(f"[TrainModel4] Updated pred_rf column in {pred_csv_path.relative_to(ROOT_DIR)} ({df_pred['pred_rf'].notnull().sum()} predictions).")

    # 8. Export Benchmark Metrics JSON
    benchmark_payload = {
        "model_name": "Model 4: Random Forest Benchmark (GroupKFold)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_count": len(RF_ALL_FEATURES),
        "features": RF_ALL_FEATURES,
        "optimal_hyperparameters": best_params,
        "cv_mae": round(best_cv_mae, 2),
        "overall_metrics": {
            "mae": round(mae_overall, 2),
            "rmse": round(rmse_overall, 2),
            "r2": round(r2_overall, 4),
        },
        "by_market": market_metrics,
        "top_features_importance": df_importance.sort_values(by="importance", ascending=False).head(10).to_dict(orient="records"),
    }

    metrics_path = REPORT5_DIR / "rf_benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"[TrainModel4] Saved Benchmark metrics to {metrics_path.relative_to(ROOT_DIR)}")

    # 9. Update models/model_metadata.json
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            meta_json["rf_metrics"] = benchmark_payload["overall_metrics"]
            meta_json["rf_by_market"] = benchmark_payload["by_market"]
            meta_json["rf_best_params"] = best_params
            meta_json["rf_features"] = RF_ALL_FEATURES
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_json, f, indent=2)
            print(f"[TrainModel4] Synchronized {meta_path.name} with Random Forest metrics.")
        except Exception as e:
            print(f"[TrainModel4] Warning: Could not update {meta_path.name}: {e}")

    print("\n======================================================================")
    print("[SUCCESS] MODEL 4 RANDOM FOREST BENCHMARK TRAINING & EXPORT COMPLETED!")
    print("======================================================================")
    return benchmark_payload


if __name__ == "__main__":
    train_and_evaluate_rf()
