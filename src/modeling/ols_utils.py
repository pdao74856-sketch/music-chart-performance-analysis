"""
Shared OLS & Diagnostic Utilities for Capstone Modeling Suite.
Standardized across Model 1 (TV2), Model 2 (TV3), and Model 3 (TV1).

Key components:
- fit_ols(): OLS with Cluster-Robust SE (by spotify_track_id) & comparison against standard SE.
- wald_block(): Joint Wald hypothesis testing using clustered covariance matrix.
- metrics(): R^2, MAE, RMSE overall and decomposed by market, with 95% bootstrap CIs.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


# ==============================================================================
# 1. FIT OLS WITH CLUSTER-ROBUST SE
# ==============================================================================
def fit_ols(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    groups: Union[pd.Series, np.ndarray],
) -> RegressionResultsWrapper:
    """
    Fits an OLS regression with Cluster-Robust Standard Errors grouped by track ID.
    
    Parameters
    ----------
    X : pd.DataFrame
        Design matrix (without constant; constant will be automatically added).
    y : pd.Series or np.ndarray
        Continuous target vector (chart_score).
    groups : pd.Series or np.ndarray
        Cluster identifier (spotify_track_id) to account for multi-market correlation.
        
    Returns
    -------
    RegressionResultsWrapper
        statsmodels RegressionResults with cov_type='cluster'.
        Enriched with additional attributes:
        - .standard_res: Unclustered standard OLS results for comparison.
        - .standard_bse: pd.Series of unclustered standard errors.
        - .se_ratio: pd.Series of cluster_se / standard_se inflation ratios.
        - .cluster_groups: pd.Series of integer factorized group codes.
    """
    # 1. Add intercept column 'const'
    X_const = sm.add_constant(X, has_constant="add")
    
    # 2. Integer factorize cluster groups
    group_codes, unique_groups = pd.factorize(groups)
    
    # 3. Fit Clustered OLS
    model = sm.OLS(y, X_const)
    res_cluster = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": group_codes}
    )
    
    # 4. Fit Standard (unclustered) OLS for SE ratio benchmarking
    res_standard = model.fit(cov_type="nonrobust")
    
    # 5. Enrich result object with diagnostic metadata
    res_cluster.standard_res = res_standard
    res_cluster.standard_bse = res_standard.bse
    res_cluster.se_ratio = res_cluster.bse / res_standard.bse
    res_cluster.cluster_groups = pd.Series(group_codes, index=X.index, name="cluster_group")
    res_cluster.n_clusters = len(unique_groups)
    
    return res_cluster


# ==============================================================================
# 2. JOINT WALD BLOCK TEST
# ==============================================================================
def wald_block(
    result: RegressionResultsWrapper,
    col_names: List[str],
) -> Dict[str, Any]:
    """
    Performs a joint Wald hypothesis test H0: beta_j = 0 for all j in col_names,
    using the cluster-robust parameter covariance matrix.
    
    Parameters
    ----------
    result : RegressionResultsWrapper
        Fitted model result returned from fit_ols().
    col_names : List[str]
        List of feature column names to test jointly.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "statistic": float (Chi-squared or F statistic)
        - "p_value": float
        - "df": int / float (degrees of freedom / restrictions)
        - "tested_features": List[str]
    """
    missing_cols = [c for c in col_names if c not in result.params.index]
    if missing_cols:
        raise ValueError(f"Features not found in model parameters: {missing_cols}")
        
    # Build hypothesis restrictions: "col1 = 0, col2 = 0, ..."
    r_matrix = ", ".join([f"{col} = 0" for col in col_names])
    
    wald_test_res = result.wald_test(r_matrix, scalar=True)
    
    # Extract scalar statistic
    stat = float(np.squeeze(wald_test_res.statistic))
    pval = float(np.squeeze(wald_test_res.pvalue))
    df = int(wald_test_res.df_denom) if wald_test_res.df_denom is not None else len(col_names)
    
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
    Computes regression evaluation metrics (R^2, MAE, RMSE) overall and by market,
    accompanied by 95% bootstrap confidence intervals and out-of-bounds tracking.
    
    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True target values (chart_score).
    y_pred : pd.Series or np.ndarray
        Predicted target values.
    market : pd.Series or np.ndarray, optional
        Categorical market labels ("US", "VN", "KR").
    track_ids : pd.Series or np.ndarray, optional
        Track IDs for cluster-aware bootstrap resampling (resampling tracks rather than rows).
    n_boot : int, default 1000
        Number of bootstrap iterations.
    seed : int, default 43
        Random seed for bootstrap reproducibility.
    alpha : float, default 0.05
        Significance level for (1 - alpha) CI (default 95% CI).
        
    Returns
    -------
    pd.DataFrame
        Table summarizing Sample, N, R2, R2_CI, MAE, MAE_CI, RMSE, RMSE_CI, Pct_OutOfBounds.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)})")
        
    records = []
    
    # Helper to calculate core metrics
    def _calc_core_metrics(yt_sub: np.ndarray, yp_sub: np.ndarray) -> Tuple[float, float, float, float]:
        ss_res = np.sum((yt_sub - yp_sub) ** 2)
        ss_tot = np.sum((yt_sub - np.mean(yt_sub)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        mae = float(np.mean(np.abs(yt_sub - yp_sub)))
        rmse = float(np.sqrt(np.mean((yt_sub - yp_sub) ** 2)))
        oob_pct = float(np.mean((yp_sub < -99.0) | (yp_sub > 100.0)) * 100.0)
        return r2, mae, rmse, oob_pct

    # Helper for bootstrap CIs
    rng = np.random.default_rng(seed)
    
    def _bootstrap(
        yt_sub: np.ndarray,
        yp_sub: np.ndarray,
        tids_sub: Optional[np.ndarray],
    ) -> Dict[str, Tuple[float, float]]:
        n = len(yt_sub)
        r2_boots, mae_boots, rmse_boots = [], [], []
        
        if tids_sub is not None:
            # Cluster bootstrap by track ID
            unique_tids = np.unique(tids_sub)
            tid_to_indices = {tid: np.where(tids_sub == tid)[0] for tid in unique_tids}
            for _ in range(n_boot):
                sampled_tids = rng.choice(unique_tids, size=len(unique_tids), replace=True)
                sample_idx = np.concatenate([tid_to_indices[tid] for tid in sampled_tids])
                r2_b, mae_b, rmse_b, _ = _calc_core_metrics(yt_sub[sample_idx], yp_sub[sample_idx])
                r2_boots.append(r2_b)
                mae_boots.append(mae_b)
                rmse_boots.append(rmse_b)
        else:
            # Row-level bootstrap
            for _ in range(n_boot):
                sample_idx = rng.choice(n, size=n, replace=True)
                r2_b, mae_b, rmse_b, _ = _calc_core_metrics(yt_sub[sample_idx], yp_sub[sample_idx])
                r2_boots.append(r2_b)
                mae_boots.append(mae_b)
                rmse_boots.append(rmse_b)
                
        def _get_ci(boots):
            valid = [b for b in boots if not np.isnan(b)]
            if len(valid) == 0:
                return (np.nan, np.nan)
            return (
                float(np.percentile(valid, 100 * (alpha / 2))),
                float(np.percentile(valid, 100 * (1 - alpha / 2)))
            )
            
        return {
            "r2_ci": _get_ci(r2_boots),
            "mae_ci": _get_ci(mae_boots),
            "rmse_ci": _get_ci(rmse_boots),
        }

    # 1. Overall evaluation
    tids_arr = np.asarray(track_ids) if track_ids is not None else None
    r2_all, mae_all, rmse_all, oob_all = _calc_core_metrics(yt, yp)
    cis_all = _bootstrap(yt, yp, tids_arr)
    
    records.append({
        "sample": "Overall",
        "n": len(yt),
        "r2": r2_all,
        "r2_ci_95": f"[{cis_all['r2_ci'][0]:.4f}, {cis_all['r2_ci'][1]:.4f}]",
        "mae": mae_all,
        "mae_ci_95": f"[{cis_all['mae_ci'][0]:.2f}, {cis_all['mae_ci'][1]:.2f}]",
        "rmse": rmse_all,
        "rmse_ci_95": f"[{cis_all['rmse_ci'][0]:.2f}, {cis_all['rmse_ci'][1]:.2f}]",
        "pct_out_of_bounds": oob_all,
    })
    
    # 2. Decompose by market if provided
    if market is not None:
        mkt_arr = np.asarray(market)
        for m in sorted(np.unique(mkt_arr)):
            mask = (mkt_arr == m)
            yt_m, yp_m = yt[mask], yp[mask]
            tids_m = tids_arr[mask] if tids_arr is not None else None
            r2_m, mae_m, rmse_m, oob_m = _calc_core_metrics(yt_m, yp_m)
            cis_m = _bootstrap(yt_m, yp_m, tids_m)
            records.append({
                "sample": f"Market_{m}",
                "n": int(np.sum(mask)),
                "r2": r2_m,
                "r2_ci_95": f"[{cis_m['r2_ci'][0]:.4f}, {cis_m['r2_ci'][1]:.4f}]",
                "mae": mae_m,
                "mae_ci_95": f"[{cis_m['mae_ci'][0]:.2f}, {cis_m['mae_ci'][1]:.2f}]",
                "rmse": rmse_m,
                "rmse_ci_95": f"[{cis_m['rmse_ci'][0]:.2f}, {cis_m['rmse_ci'][1]:.2f}]",
                "pct_out_of_bounds": oob_m,
            })
            
    df_metrics = pd.DataFrame(records)
    return df_metrics
