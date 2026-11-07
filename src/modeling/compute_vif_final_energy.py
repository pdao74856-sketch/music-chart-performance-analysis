"""
Script: src/modeling/compute_vif_final_energy.py
Computes VIF for the 24 main effects of Model 3 on Train (n=1685),
using energy_std (excluding loudness_proxy_std and spectral_bandwidth_std),
and outputs reports/vif_final_energy.csv.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.prepare_features import get_split, load_clean_modeling_data


def safe_to_csv(df: pd.DataFrame, filepath: Path, **kwargs) -> None:
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


def main():
    # 1. Load data & get split (seed=43, Train n=1685)
    df_raw = load_clean_modeling_data()
    split = get_split(df=df_raw, seed=43)
    print("Train/Test:", len(split.X_train), len(split.X_test))
    X_train = split.X_train

    n_train = len(X_train)
    if n_train != 1685:
        raise ValueError(f"Kỳ vọng Train n=1685 nhưng thực tế là {n_train}")

    # 2. Extract the 24 main feature columns of Model 3 (audio + mode + lyrics + release_year_std + Market_VN + Market_KR)
    # Strictly exclude interaction terms (__x__)
    cfg = split.config
    audio_cols_std = [f"{c}_std" for c in cfg["audio_cols"]]  # 14 continuous audio
    mode_col = ["mode"]                                       # 1 binary mode
    lyrics_cols_std = [f"{c}_std" for c in cfg["lyrics_cols"]]# 6 lyrics
    year_col_std = [f"{cfg['year_col']}_std"]                 # 1 release year
    market_cols = ["Market_VN", "Market_KR"]                  # 2 market dummies

    cols_24 = audio_cols_std + mode_col + lyrics_cols_std + year_col_std + market_cols

    if len(cols_24) != 24:
        raise ValueError(f"Kỳ vọng 24 biến nhưng thực tế có {len(cols_24)} biến: {cols_24}")

    # Assert energy_std is present, and loudness_proxy_std / spectral_bandwidth_std are absent
    assert "energy_std" in cols_24, "Thiếu energy_std trong danh sách 24 biến!"
    assert "loudness_proxy_std" not in cols_24, "loudness_proxy_std không được phép có mặt!"
    assert "spectral_bandwidth_std" not in cols_24, "spectral_bandwidth_std không được phép có mặt!"

    # 3. Add constant and compute VIF using statsmodels variance_inflation_factor
    X_sub = X_train[cols_24].copy()
    X_with_const = sm.add_constant(X_sub)

    category_map = {
        "tempo_std": "Audio Rhythm",
        "onset_rate_std": "Audio Rhythm",
        "tempo_stability_std": "Audio Rhythm",
        "energy_std": "Audio Energy",
        "rms_variability_std": "Audio Energy",
        "spectral_centroid_std": "Audio Timbre",
        "chord_change_rate_std": "Audio Harmony",
        "key_confidence_std": "Audio Harmony",
        "chroma_repetition_determinism_std": "Audio RQA",
        "chroma_mean_pattern_duration_std": "Audio RQA",
        "chroma_longest_pattern_duration_std": "Audio RQA",
        "mfcc_repetition_determinism_std": "Audio RQA",
        "mfcc_mean_pattern_duration_std": "Audio RQA",
        "mfcc_longest_pattern_duration_std": "Audio RQA",
        "mode": "Audio Binary",
        "word_count_std": "Lyrics",
        "repetition_ratio_std": "Lyrics",
        "avg_word_length_std": "Lyrics",
        "line_count_std": "Lyrics",
        "chorus_repetition_count_std": "Lyrics",
        "punctuation_ratio_std": "Lyrics",
        "release_year_std": "Control",
        "Market_VN": "Dummy",
        "Market_KR": "Dummy",
    }

    records = []
    for i, col in enumerate(X_with_const.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X_with_const.values, i)
        records.append({
            "feature": col,
            "vif_final": round(float(vif_val), 4),
            "category": category_map[col],
            "_raw_vif": float(vif_val),
        })

    df_vif = pd.DataFrame(records).sort_values("vif_final", ascending=False).reset_index(drop=True)

    # 4. Save to reports/vif_final_energy.csv
    out_csv = repo_root / "reports" / "vif_final_energy.csv"
    export_df = df_vif[["feature", "vif_final", "category"]].copy()
    safe_to_csv(export_df, out_csv, index=False)

    # 5. Print results exactly as requested
    max_row = df_vif.iloc[0]
    energy_row = df_vif[df_vif["feature"] == "energy_std"].iloc[0]

    print(f"Số biến: {len(df_vif)}")
    print(f"Max VIF: {max_row['vif_final']:.4f} ({max_row['feature']})")
    print(f"VIF của energy_std: {energy_row['vif_final']:.4f}")


if __name__ == "__main__":
    main()
