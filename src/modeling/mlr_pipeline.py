"""
Module: src.modeling.mlr_pipeline
Stand-alone, self-contained MLRPipeline inference class for Model 3 OLS.
Bundles feature engineering configuration, scaler, feature ordering, and fitted parameters
to enable zero-overhead real-time prediction for Streamlit Web App (Tab 1, Tab 2, Tab 3).
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.features.prepare_features import build_design_matrix


class MLRPipeline:
    """
    Lightweight, production-ready inference wrapper bundling configuration,
    StandardScaler, feature schema, parameter weights, and training medians.
    Enables single-call prediction directly from raw DataFrames.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        scaler: StandardScaler,
        feature_names: List[str],
        ols_params: pd.Series,
        ols_summary_str: str,
        train_medians: Dict[str, float],
    ):
        self.config = config
        self.scaler = scaler
        self.feature_names = feature_names
        self.ols_params = ols_params
        self.ols_summary_str = ols_summary_str
        self.train_medians = train_medians

    def predict_raw(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw input DataFrame through build_design_matrix and computes predicted chart_score.
        
        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw song data (containing unscaled audio and lyrics columns).
            
        Returns
        -------
        np.ndarray
            Predicted continuous chart_score values.
        """
        X, _ = build_design_matrix(df_raw, self.scaler, self.config)
        X_const = sm.add_constant(X, has_constant="skip")
        
        # Intercept + 34 features = 35 columns
        expected_cols = ["const"] + list(self.feature_names)
        return np.dot(X_const[expected_cols].values, self.ols_params[expected_cols].values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts chart_score directly from an already transformed design matrix X.
        """
        X_const = sm.add_constant(X, has_constant="skip")
        expected_cols = ["const"] + list(self.feature_names)
        return np.dot(X_const[expected_cols].values, self.ols_params[expected_cols].values)


__all__ = ["MLRPipeline"]
