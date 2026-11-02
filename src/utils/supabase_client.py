#!/usr/bin/env python3
"""
src/utils/supabase_client.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Pipeline, Supabase, Benchmark & Deploy Lead)

Responsibilities:
  - Manages Supabase client authentication (via .env locally or st.secrets on Cloud).
  - Fetches cleaned data from `view_clean_modeling_dataset` or `modeling_dataset`.
  - Provides robust offline fallback to local CSV when network or credentials are unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load local environment if present
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def get_supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    """Retrieves Supabase URL and Key from environment or Streamlit secrets."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    # If running inside Streamlit Cloud, fallback to st.secrets
    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                url = st.secrets.get("SUPABASE_URL", url)
                key = st.secrets.get("SUPABASE_KEY", key)
        except Exception:
            pass

    return url, key


def get_supabase_client():
    """Initializes and returns an authenticated Supabase client."""
    url, key = get_supabase_credentials()
    if not url or not key:
        raise ValueError(
            "Supabase credentials not configured. Please set SUPABASE_URL and SUPABASE_KEY in .env or st.secrets."
        )
    
    from supabase import create_client
    return create_client(url, key)


def load_clean_modeling_data(
    source: str = "auto",
    select: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads the clean modeling dataset.

    Parameters:
    -----------
    source : str
        'auto' (try Supabase first, fallback to local CSV),
        'supabase' (force Supabase query),
        'local' (force local CSV loading).
    select : list of str, optional
        Specific column names to query.
    limit : int, optional
        Maximum number of rows to retrieve.

    Returns:
    --------
    pd.DataFrame containing clean modeling records.
    """
    df = None

    if source in ("auto", "supabase"):
        url, key = get_supabase_credentials()
        if url and key:
            try:
                print(f"[SupabaseClient] Connecting to {url[:25]}...")
                client = get_supabase_client()
                
                # Try query from view_clean_modeling_dataset
                col_str = ",".join(select) if select else "*"
                query = client.table("view_clean_modeling_dataset").select(col_str)
                if limit:
                    query = query.limit(limit)
                
                response = query.execute()
                if response.data and len(response.data) > 0:
                    df = pd.DataFrame(response.data)
                    print(f"[SupabaseClient] Successfully fetched {len(df):,} rows from view_clean_modeling_dataset.")
            except Exception as e:
                print(f"[SupabaseClient] Warning: Failed to query Supabase ({e}).")
                if source == "supabase":
                    raise

    # Fallback to local processed dataset
    if df is None:
        local_path = ROOT_DIR / "data" / "processed" / "modeling_dataset.csv"
        if not local_path.exists():
            raise FileNotFoundError(f"Local modeling dataset not found at {local_path}")
        
        print(f"[SupabaseClient] Loading from local CSV fallback: {local_path.name}")
        df_raw = pd.read_csv(local_path)
        
        # Apply standard clean filter: is_sampled = True and has_both_rows = True
        mask = (df_raw.get("is_sampled") == True) & (df_raw.get("has_both_rows") == True)
        df = df_raw[mask].copy()

        # Harmonize column names and transformations if needed
        if "tempo" not in df.columns and "tempo_bpm" in df.columns:
            df["tempo"] = df["tempo_bpm"]
        if "energy" not in df.columns and "rms_energy" in df.columns:
            df["energy"] = df["rms_energy"]
        if "mode" not in df.columns and "audio_mode" in df.columns:
            df["mode"] = df["audio_mode"].apply(lambda x: 1 if str(x).lower() in ("1", "major") else 0)

        # Standardize dBFS loudness_proxy if not already in dBFS
        # 20 * log10(max(energy, 1e-10))
        if "energy" in df.columns:
            calculated_loudness = 20.0 * np.log10(np.maximum(df["energy"].values, 1e-10))
            # If current loudness_proxy is outside [-30, 0] dB range, replace with dBFS formula
            if df["loudness_proxy"].mean() > 0 or df["loudness_proxy"].min() < -100:
                df["loudness_proxy"] = calculated_loudness

        # Standardize tempo_stability: 1 / (1 + beat_interval_variability)
        if "beat_interval_variability" in df.columns:
            df["tempo_stability"] = 1.0 / (1.0 + df["beat_interval_variability"].values)

        if select:
            avail_cols = [c for c in select if c in df.columns]
            df = df[avail_cols]
        
        if limit:
            df = df.head(limit)
            
        print(f"[SupabaseClient] Local clean dataset loaded: {len(df):,} rows.")

    return df


if __name__ == "__main__":
    df = load_clean_modeling_data(source="auto")
    print("\nDataset Summary:")
    print("Shape:", df.shape)
    print("Markets distribution:", df["market"].value_counts().to_dict() if "market" in df else "N/A")
