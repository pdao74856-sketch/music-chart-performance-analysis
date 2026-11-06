#!/usr/bin/env python3
"""
src/modeling/member4_paired_bootstrap.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Integration, Benchmark & Deploy Lead)

Purpose:
  Executes Paired Cluster Bootstrap (B=1,000, seed=43, clustered by spotify_track_id)
  comparing Model 4 (Random Forest) vs Model 3 (OLS Full Model) on held-out Test Set (N=410).
  Applies the exact same global sampling method as Member 1:
  generates boot_indices_list once on the full Test set, then filters for Overall, VN, US, KR.
  Exports reports/rf_vs_ols_paired_bootstrap.csv and updates reports/figures/rf_vs_ols_market_comparison.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Đọc dữ liệu tập Test
    test_csv = MODELS_DIR / "test_predictions.csv"
    df_test = pd.read_csv(test_csv)
    print(f"Loaded Test Set: N = {len(df_test)} observations across {df_test['market'].nunique()} markets.")

    # 2. Chuẩn bị các phạm vi đánh giá
    n_boot = 1000
    seed = 43
    samples_dict = {
        "Overall": np.ones(len(df_test), dtype=bool),
        "VN": df_test["market"].values == "VN",
        "US": df_test["market"].values == "US",
        "KR": df_test["market"].values == "KR",
    }

    # 3. Tạo bootstrap indices MỘT LẦN theo phương pháp Member 1
    unique_tracks = df_test["spotify_track_id"].unique()
    n_tracks = len(unique_tracks)
    rng = np.random.default_rng(seed)

    track_to_indices = {}
    for idx, tid in enumerate(df_test["spotify_track_id"]):
        track_to_indices.setdefault(tid, []).append(idx)

    boot_indices_list = []
    for _ in range(n_boot):
        sample_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        idx_pool = []
        for track_id in sample_tracks:
            idx_pool.extend(track_to_indices[track_id])
        boot_indices_list.append(np.array(idx_pool))

    # 4. Thực thi Paired Bootstrap cho từng phạm vi
    y_test = df_test["chart_score_actual"].values
    pred_m3 = df_test["pred_ols_m3"].values
    pred_rf = df_test["pred_rf"].values

    results = []
    for scope, mask in samples_dict.items():
        # Điểm ước lượng thực tế trên mẫu ban đầu (Observed Point Estimates)
        yt_obs = y_test[mask]
        yo_obs = pred_m3[mask]
        yr_obs = pred_rf[mask]

        obs_delta_mae = mean_absolute_error(yt_obs, yo_obs) - mean_absolute_error(yt_obs, yr_obs)
        obs_delta_r2 = r2_score(yt_obs, yr_obs) - r2_score(yt_obs, yo_obs)

        delta_r2_list = []
        delta_mae_list = []

        for b_idx in boot_indices_list:
            # Lọc các dòng thuộc phạm vi tương ứng từ mẫu bootstrap toàn cục
            sub_mask = mask[b_idx]
            if not np.any(sub_mask):
                continue
            idx_sub = b_idx[sub_mask]

            yt = y_test[idx_sub]
            yo = pred_m3[idx_sub]
            yr = pred_rf[idx_sub]

            delta_r2_list.append(r2_score(yt, yr) - r2_score(yt, yo))
            delta_mae_list.append(mean_absolute_error(yt, yo) - mean_absolute_error(yt, yr))

        ci_r2 = np.percentile(delta_r2_list, [2.5, 97.5])
        ci_mae = np.percentile(delta_mae_list, [2.5, 97.5])

        res = {
            "scope": scope,
            "n_obs": int(np.sum(mask)),
            "n_unique_tracks": int(df_test.loc[mask, "spotify_track_id"].nunique()),
            "obs_delta_r2": obs_delta_r2,
            "delta_r2_bootstrap_mean": np.mean(delta_r2_list),
            "delta_r2_ci_lower": ci_r2[0],
            "delta_r2_ci_upper": ci_r2[1],
            "obs_delta_mae": obs_delta_mae,
            "delta_mae_bootstrap_mean": np.mean(delta_mae_list),
            "delta_mae_ci_lower": ci_mae[0],
            "delta_mae_ci_upper": ci_mae[1],
        }
        results.append(res)
        print(f"Scope: {scope:7s} | Observed Delta MAE = {res['obs_delta_mae']:+.4f} | Bootstrap Mean = {res['delta_mae_bootstrap_mean']:+.4f} | 95% CI = [{res['delta_mae_ci_lower']:+.4f}, {res['delta_mae_ci_upper']:+.4f}]")
        print(f"        | Observed Delta R2  = {res['obs_delta_r2']*100:+.2f}% | Bootstrap Mean = {res['delta_r2_bootstrap_mean']*100:+.2f}% | 95% CI = [{res['delta_r2_ci_lower']:+.4f}, {res['delta_r2_ci_upper']:+.4f}]")

    # 5. Lưu file CSV
    out_csv = REPORTS_DIR / "rf_vs_ols_paired_bootstrap.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved verified bootstrap results to {out_csv.relative_to(ROOT_DIR)}")

    # 6. Cập nhật biểu đồ trực quan hóa
    market_labels = ["Vietnam (VN)", "United States (US)", "South Korea (KR)", "Overall (Test)"]
    market_keys = ["VN", "US", "KR", "Overall"]

    metrics = []
    for key in market_keys:
        sub = df_test if key == "Overall" else df_test[df_test["market"] == key]
        yt = sub["chart_score_actual"].values
        yo = sub["pred_ols_m3"].values
        yr = sub["pred_rf"].values
        metrics.append({
            "n": len(sub),
            "r2_ols": r2_score(yt, yo) * 100,
            "r2_rf": r2_score(yt, yr) * 100,
            "mae_ols": mean_absolute_error(yt, yo),
            "mae_rf": mean_absolute_error(yt, yr),
        })
    m_df = pd.DataFrame(metrics)

    x = np.arange(len(market_labels))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    rects1 = ax1.bar(x - width/2, m_df["r2_ols"], width, label="Model 3 (OLS)", color="#4A90E2", edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, m_df["r2_rf"], width, label="Model 4 (Random Forest)", color="#50E3C2", edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Out-of-Sample $R^2$ (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Out-of-Sample Variance Explained ($R^2$)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m}\n(N={n})" for m, n in zip(market_labels, m_df["n"])], fontsize=10)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0E6251")

    rects3 = ax2.bar(x - width/2, m_df["mae_ols"], width, label="Model 3 (OLS)", color="#4A90E2", edgecolor="black", linewidth=0.8)
    rects4 = ax2.bar(x + width/2, m_df["mae_rf"], width, label="Model 4 (Random Forest)", color="#E67E22", edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Mean Absolute Error (MAE - lower is better)", fontsize=12, fontweight="bold")
    ax2.set_title("Prediction Error Comparison (MAE)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m}\n(N={n})" for m, n in zip(market_labels, m_df["n"])], fontsize=10)
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#78281F")

    plt.suptitle("Out-of-Sample Benchmark: Random Forest vs. OLS across Markets (Test Set N=410)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "rf_vs_ols_market_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure to {fig_path.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    main()
