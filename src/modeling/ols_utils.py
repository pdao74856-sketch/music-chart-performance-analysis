"""
Shared OLS & Diagnostic Utilities for Capstone Modeling Suite.
Re-exported from src.utils.ols_cluster_fit.
"""

from src.utils.ols_cluster_fit import (
    fit_ols,
    fit_ols_cluster,
    wald_block,
    metrics,
)

__all__ = ["fit_ols", "fit_ols_cluster", "wald_block", "metrics"]
