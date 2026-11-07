"""
Script: src/modeling/evaluate_diagnostics_reset.py
Ramsey RESET Test for Model 3 (Full OLS, 35 parameters) on Train set.
Outputs: reports/diagnostics/reset_test.json
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.prepare_features import load_clean_modeling_data, get_split
from src.utils.ols_cluster_fit import fit_ols_cluster, wald_block


def safe_to_json(data: dict, filepath: Path, **kwargs) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)
    
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


def run_reset_test():
    print("=" * 80)
    print("RAMSEY RESET TEST - MODEL 3 (FULL OLS)")
    print("=" * 80)

    # 1. Load data & get split (seed=43)
    df = load_clean_modeling_data(source="supabase")
    split = get_split(df=df, seed=43)

    cfg = split.config
    year_col = cfg["year_col"]
    audio_cols_std = [f"{c}_std" for c in cfg["audio_cols"]] + ["mode"]
    lyrics_cols_std = [f"{c}_std" for c in cfg["lyrics_cols"]]
    context_cols_std = [f"{year_col}_std", "Market_VN", "Market_KR"]
    interaction_bases = cfg["interaction_cols"]
    inter_cols_std = [
        f"{base}_std__x__{m}"
        for base in interaction_bases
        for m in ["VN", "KR"]
    ]
    m3_cols = audio_cols_std + lyrics_cols_std + context_cols_std + inter_cols_std

    X_train = split.X_train[m3_cols]
    y_train = split.y_train
    track_ids = split.raw_train["spotify_track_id"]
    assert (split.raw_train["spotify_track_id"].values == split.ids_train.values).all()
    n_obs = len(y_train)
    n_params = len(m3_cols) + 1  # 34 predictors + const = 35

    print(f"Số quan sát Train (n): {n_obs}")
    print(f"Số tham số M3 (k):     {n_params}")

    # Fit M3 via standard project function
    res3 = fit_ols_cluster(y=y_train, X=X_train, groups=track_ids)

    # 2. Statsmodels linear_reset with power=[2, 3], test_type='fitted', use_f=True
    # Uses res3.standard_res (unclustered OLS) for standard F test
    reset_res = linear_reset(res3.standard_res, power=[2, 3], test_type="fitted", use_f=True)

    stat_f = float(reset_res.statistic)
    p_f = float(reset_res.pvalue)
    df_num = int(reset_res.df_num)
    df_denom = int(reset_res.df_denom)

    # 3. Cluster-robust version: add y_hat^2 and y_hat^3 into X, refit with cluster SE, joint Wald test
    X_train_const = sm.add_constant(X_train, has_constant="skip")
    y_hat = res3.predict(X_train_const)

    X_aug = X_train.copy()
    X_aug["y_hat_sq"] = y_hat ** 2
    X_aug["y_hat_cu"] = y_hat ** 3

    res_aug = fit_ols_cluster(y=y_train, X=X_aug, groups=track_ids)
    wald_test_res = wald_block(res_aug, ["y_hat_sq", "y_hat_cu"])

    wald_chi2 = float(wald_test_res["statistic"])
    wald_df = int(wald_test_res["df"])
    wald_p = float(wald_test_res["p_value"])

    # 4. Check df_denom
    expected_df_denom = n_obs - n_params - 2  # 1685 - 35 - 2 = 1648
    if df_denom != expected_df_denom:
        print(f"CẢNH BÁO: df_denom thực tế ({df_denom}) KHÁC kỳ vọng ({expected_df_denom})!")
    else:
        print(f"Kiểm tra df_denom: {df_denom} == {expected_df_denom} (Khớp kỳ vọng)")

    # 5. Save to reports/diagnostics/reset_test.json
    out_dir = repo_root / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reset_test.json"

    run_date_str = datetime.now().strftime("%Y-%m-%d")
    results_json = {
        "n": n_obs,
        "n_params": n_params,
        "powers": [2, 3],
        "F": stat_f,
        "p_F": p_f,
        "df_num": df_num,
        "df_denom": df_denom,
        "wald_chi2": wald_chi2,
        "wald_df": wald_df,
        "wald_p": wald_p,
        "ngay_chay": run_date_str,
    }

    safe_to_json(results_json, out_file, indent=2)
    print(f"Đã lưu kết quả vào: {out_file}")

    # 6. Print 5 main numbers and conclusion
    print("\n" + "=" * 80)
    print("5 CON SỐ CHÍNH CỦA KIỂM ĐỊNH RESET:")
    print("=" * 80)
    print(f"1. F-statistic:        {stat_f:.4f}")
    print(f"2. p_F:                {p_f:.4e}")
    print(f"3. df (num, denom):    ({df_num}, {df_denom})")
    print(f"4. Wald chi2 (cụm):    {wald_chi2:.4f} (df = {wald_df})")
    print(f"5. p_Wald (cụm):       {wald_p:.4f}")
    print("=" * 80)

    # Conclusion according to rule
    if p_f < 0.05 or wald_p < 0.05:
        print("Kết luận: p nhỏ = dấu hiệu sai dạng hàm hoặc thiếu biến.")
    else:
        print("Kết luận: p lớn = không phát hiện dấu hiệu.")
    print("=" * 80)


if __name__ == "__main__":
    run_reset_test()
