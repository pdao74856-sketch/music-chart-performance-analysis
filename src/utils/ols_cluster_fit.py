"""
Module: src.utils.ols_cluster_fit
Shared utility to fit OLS regression with Cluster-Robust Standard Errors grouped by track ID.
Standardized across Model 1 (TV2: Audio-only), Model 2 (TV3: Audio + Lyrics), and Model 3 (TV1: Full OLS).

Key components:
- fit_ols_cluster(): OLS with Cluster-Robust SE (by spotify_track_id) & unclustered comparison.
- fit_ols(): Wrapper supporting (X, y, groups) parameter ordering.
- wald_block(): Joint Wald test (Chi-squared test on cluster-robust covariance matrix).
- metrics(): R^2, MAE, RMSE overall and by market with cluster bootstrap CIs.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


# ==============================================================================
# 1. FIT OLS WITH CLUSTER-ROBUST SE
# ==============================================================================
def fit_ols_cluster(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    groups: Union[pd.Series, np.ndarray],
) -> RegressionResultsWrapper:
    """
    Fits an OLS regression with Cluster-Robust Standard Errors (cov_type='cluster')
    grouped by spotify_track_id to account for intra-cluster correlation across markets.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Continuous target vector (chart_score).
    X : pd.DataFrame
        Design matrix of explanatory features (constant will be added if missing).
    groups : pd.Series or np.ndarray
        Cluster identifier (spotify_track_id).

    Returns
    -------
    RegressionResultsWrapper
        statsmodels RegressionResults with cov_type='cluster'.
        Enriched with diagnostic attributes:
        - .standard_res: Unclustered OLS results for comparison.
        - .standard_bse: pd.Series of unclustered standard errors.
        - .se_ratio: pd.Series of cluster_se / standard_se inflation ratios.
        - .cluster_groups: pd.Series of integer factorized group codes.
        - .n_clusters: Total number of unique cluster groups.
    """
    # 1. Kiểm tra tính hợp lệ của dữ liệu đầu vào và các chiều
    if len(groups) != len(X) or len(groups) != len(y):
        raise ValueError(
            f"Kích thước không khớp: len(groups)={len(groups)}, len(X)={len(X)}, len(y)={len(y)}"
        )
    if pd.isna(groups).any():
        raise ValueError("Mảng định danh cụm (groups) chứa giá trị NaN / null.")

    # 2. Thêm cột hằng số 'const' (bỏ qua nếu X đã có sẵn cột hằng số, thêm nếu chưa có)
    X_const = sm.add_constant(X, has_constant="skip")

    # 3. Integer factorize cluster groups
    group_codes, unique_groups = pd.factorize(groups)

    # 4. Fit Clustered OLS
    model = sm.OLS(y, X_const)
    res_cluster = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": group_codes}
    )

    # 5. Fit Standard (unclustered) OLS để tính hệ số lạm phát sai số chuẩn (SE ratio)
    res_standard = model.fit(cov_type="nonrobust")

    # 6. Đính kèm các thuộc tính chẩn đoán bổ sung
    res_cluster.standard_res = res_standard
    res_cluster.standard_bse = res_standard.bse
    res_cluster.se_ratio = res_cluster.bse / res_standard.bse
    res_cluster.cluster_groups = pd.Series(group_codes, index=X.index, name="cluster_group")
    res_cluster.n_clusters = len(unique_groups)

    return res_cluster


def fit_ols(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    groups: Union[pd.Series, np.ndarray],
) -> RegressionResultsWrapper:
    """Wrapper nhận thứ tự tham số (X, y, groups) tương thích ngược."""
    return fit_ols_cluster(y=y, X=X, groups=groups)


# ==============================================================================
# 2. JOINT WALD BLOCK TEST (WALD CHI-SQUARED TEST TRÊN COVARIANCE CỤM)
# ==============================================================================
def wald_block(
    result: RegressionResultsWrapper,
    col_names: List[str],
) -> Dict[str, Any]:
    """
    Thực hiện kiểm định Wald khối (Joint Wald Test) H0: beta_j = 0 cho mọi biến trong col_names,
    sử dụng ma trận hiệp phương sai cluster-robust (Wald chi2 test, không dùng F cổ điển).

    Parameters
    ----------
    result : RegressionResultsWrapper
        Kết quả mô hình OLS với cluster SE (từ fit_ols_cluster).
    col_names : List[str]
        Danh sách tên các biến cần kiểm định đồng thời.

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa:
        - "statistic": float (thống kê Wald Chi-squared)
        - "p_value": float (p-value từ phân phối Chi-squared)
        - "df": int (bậc tự do = số ràng buộc = len(col_names))
        - "tested_features": List[str]
    """
    missing_cols = [c for c in col_names if c not in result.params.index]
    if missing_cols:
        raise ValueError(f"Các biến không tồn tại trong tham số mô hình: {missing_cols}")

    # Xây dựng chuỗi ràng buộc: "col1 = 0, col2 = 0, ..."
    r_matrix = ", ".join([f"{col} = 0" for col in col_names])
    
    # use_f=False để tính thống kê Wald Chi-squared trên ma trận hiệp phương sai cụm
    wald_test_res = result.wald_test(r_matrix, use_f=False, scalar=True)

    stat = float(np.squeeze(wald_test_res.statistic))
    pval = float(np.squeeze(wald_test_res.pvalue))
    df = int(len(col_names))

    return {
        "statistic": stat,
        "p_value": pval,
        "df": df,
        "tested_features": col_names,
    }


# ==============================================================================
# 3. METRICS EVALUATION WITH BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================
def metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    market: Optional[Union[pd.Series, np.ndarray]] = None,
    track_ids: Optional[Union[pd.Series, np.ndarray]] = None,
    n_boot: int = 1000,
    seed: int = 43,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Tính toán các chỉ số đánh giá hồi quy (R^2, MAE, RMSE) cho toàn bộ và theo từng thị trường,
    kèm theo khoảng tin cậy 95% bootstrap theo cụm (cluster bootstrap theo track_id).
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    n = len(y_t)

    def _calc_raw_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
        err = yt - yp
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        oob_count = int(np.sum((yp < 1.0) | (yp > 100.0)))
        oob_pct = float(oob_count / len(yp) * 100.0)
        return {"r2": r2, "mae": mae, "rmse": rmse, "oob_count": oob_count, "oob_pct": oob_pct}

    subsets: Dict[str, np.ndarray] = {"Overall": np.arange(n)}
    if market is not None:
        m_arr = np.asarray(market).ravel()
        for m_val in np.unique(m_arr):
            subsets[str(m_val)] = np.where(m_arr == m_val)[0]

    rows = []
    rng = np.random.default_rng(seed)

    for subset_name, idxs in subsets.items():
        sub_yt = y_t[idxs]
        sub_yp = y_p[idxs]
        sub_n = len(idxs)
        base = _calc_raw_metrics(sub_yt, sub_yp)

        boot_r2, boot_mae, boot_rmse = [], [], []
        if track_ids is not None:
            sub_tracks = np.asarray(track_ids).ravel()[idxs]
            u_tracks = np.unique(sub_tracks)
            n_tracks = len(u_tracks)
            track_to_idx = {t: np.where(sub_tracks == t)[0] for t in u_tracks}

            for _ in range(n_boot):
                sampled_tracks = rng.choice(u_tracks, size=n_tracks, replace=True)
                b_idx_list = [track_to_idx[t] for t in sampled_tracks]
                b_idxs = np.concatenate(b_idx_list)
                b_res = _calc_raw_metrics(sub_yt[b_idxs], sub_yp[b_idxs])
                boot_r2.append(b_res["r2"])
                boot_mae.append(b_res["mae"])
                boot_rmse.append(b_res["rmse"])
        else:
            for _ in range(n_boot):
                b_idxs = rng.choice(sub_n, size=sub_n, replace=True)
                b_res = _calc_raw_metrics(sub_yt[b_idxs], sub_yp[b_idxs])
                boot_r2.append(b_res["r2"])
                boot_mae.append(b_res["mae"])
                boot_rmse.append(b_res["rmse"])

        low_q = alpha / 2.0 * 100.0
        high_q = (1.0 - alpha / 2.0) * 100.0

        rows.append({
            "Market": subset_name,
            "N": sub_n,
            "R2": base["r2"],
            "R2_ci_lower": float(np.percentile(boot_r2, low_q)),
            "R2_ci_upper": float(np.percentile(boot_r2, high_q)),
            "MAE": base["mae"],
            "MAE_ci_lower": float(np.percentile(boot_mae, low_q)),
            "MAE_ci_upper": float(np.percentile(boot_mae, high_q)),
            "RMSE": base["rmse"],
            "RMSE_ci_lower": float(np.percentile(boot_rmse, low_q)),
            "RMSE_ci_upper": float(np.percentile(boot_rmse, high_q)),
            "OOB_Predictions_Count": base["oob_count"],
            "OOB_Predictions_Pct": base["oob_pct"],
        })

    return pd.DataFrame(rows)


__all__ = ["fit_ols_cluster", "fit_ols", "wald_block", "metrics"]
