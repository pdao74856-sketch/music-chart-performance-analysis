
#!/usr/bin/env python3
"""
src/utils/supabase_client.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Pipeline, Supabase, Benchmark & Deploy Lead)

Responsibilities:
  - Manages Supabase client authentication (via .env locally or st.secrets on Cloud).
  - Fetches cleaned data from `view_clean_modeling_dataset` (2,098 rows) with automatic pagination.
  - Provides robust offline fallback to local CSV when network or credentials are unavailable.
"""

from __future__ import annotations
import hashlib

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
    """Retrieves Supabase URL and Key from environment (.env) or Streamlit secrets."""
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






# Default 29 clean modeling columns of view_clean_modeling_dataset
DEFAULT_SELECT_COLUMNS = [
    "spotify_track_id",
    "market",
    "title",
    "artist",
    "chart_score",
    "release_year",
    "tempo",
    "onset_rate",
    "tempo_stability",
    "energy",
    "rms_variability",
    "loudness_proxy",
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
    "word_count",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "punctuation_ratio",
]


def load_clean_modeling_data(
    source: str = "auto",
    select: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    if select is None:
        select = list(DEFAULT_SELECT_COLUMNS)
    """
    Loads the clean modeling dataset (2,098 rows from view_clean_modeling_dataset).
    Includes automatic pagination (.range) to bypass the PostgREST 1,000-row limit.

    Parameters:
    -----------
    source : str
        'auto' (try Supabase first, fallback to local CSV),
        'supabase' (force Supabase query, no fallback),
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
    data_source = None

    if source in ("auto", "supabase"):
        url, key = get_supabase_credentials()
        if url and key:
            try:
                print(f"[SupabaseClient] Connecting to {url[:25]}...")
                client = get_supabase_client()
                
                col_str = ",".join(select) if select else "*"
                all_records = []
                batch_size = 1000
                start_idx = 0
                
                while True:
                    end_idx = start_idx + batch_size - 1
                    if limit and (start_idx + batch_size) > limit:
                        end_idx = limit - 1
                        
                    res = (
                        client.table("view_clean_modeling_dataset")
                        .select(col_str)
                        .range(start_idx, end_idx)
                        .execute()
                    )
                    
                    batch_data = res.data or []
                    all_records.extend(batch_data)
                    
                    # If returned less than 1,000 rows or reached limit, all rows collected
                    if len(batch_data) < batch_size or (limit and len(all_records) >= limit):
                        break
                        
                    start_idx += batch_size
                
                if all_records:
                    df = pd.DataFrame(all_records)
                    data_source = "supabase"
                    print(f"[SupabaseClient] Successfully fetched all {len(df):,} clean rows from view_clean_modeling_dataset.")
            except Exception as e:
                print(f"[SupabaseClient] Warning: Failed to query Supabase ({e}).")
                if source == "supabase":
                    raise RuntimeError(f"[SupabaseClient] Supabase live query failed and fallback disabled for source='supabase': {e}") from e

    # Fallback to local processed dataset if offline or not configured
    if df is None:
        if source == "supabase":
            raise RuntimeError("[SupabaseClient] Supabase live query returned no data and fallback disabled for source='supabase'.")
        clean_cache_path = ROOT_DIR / "data" / "processed" / "modeling_dataset_clean_cache.csv"
        local_path = clean_cache_path if clean_cache_path.exists() else (ROOT_DIR / "data" / "processed" / "modeling_dataset.csv")
        if not local_path.exists():
            raise FileNotFoundError(f"Local modeling dataset not found at {local_path}")
        
        print(f"[SupabaseClient] Loading from local CSV fallback: {local_path.name}")
        df_raw = pd.read_csv(local_path)
        data_source = "cache"
        
        # Apply standard clean filter if loading raw modeling_dataset.csv
        if "is_sampled" in df_raw.columns and "has_both_rows" in df_raw.columns:
            mask = (df_raw.get("is_sampled") == True) & (df_raw.get("has_both_rows") == True)
            df = df_raw[mask].copy()
        else:
            df = df_raw.copy()

    # Harmonize column names and transformations if needed
    if "tempo" not in df.columns and "tempo_bpm" in df.columns:
        df["tempo"] = df["tempo_bpm"]
    if "energy" not in df.columns and "rms_energy" in df.columns:
        df["energy"] = df["rms_energy"]

    # Strict audio_mode validation: NULL or invalid values MUST raise ValueError (never silent 0)
    def parse_audio_mode(val):
        if pd.isna(val):
            raise ValueError(f"[CRITICAL] NULL value encountered in audio_mode: {val}")
        s = str(val).strip().lower()
        if s in ("major", "1", "1.0"):
            return 1
        elif s in ("minor", "0", "0.0"):
            return 0
        else:
            raise ValueError(f"[CRITICAL] Invalid / unexpected audio_mode value: {val!r}")

    if "audio_mode" in df.columns:
        df["mode"] = df["audio_mode"].apply(parse_audio_mode)
    elif "mode" in df.columns:
        # If mode column exists, validate it and synchronize audio_mode
        df["mode"] = df["mode"].apply(parse_audio_mode)
        df["audio_mode"] = df["mode"].apply(lambda x: "major" if x == 1 else "minor")
    else:
        raise KeyError("[CRITICAL] Neither 'audio_mode' nor 'mode' column found in modeling dataset!")

    # Standardize dBFS loudness_proxy if not already in dBFS: 20 * log10(max(energy, 1e-10))
    if "energy" in df.columns and "loudness_proxy" in df.columns:
        df["loudness_proxy"] = pd.to_numeric(df["loudness_proxy"], errors="coerce")
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
        if df["loudness_proxy"].mean() > 0 or df["loudness_proxy"].min() < -100:
            df["loudness_proxy"] = 20.0 * np.log10(np.maximum(df["energy"].values, 1e-10))

    # Standardize tempo_stability: 1 / (1 + beat_interval_variability)
    if "beat_interval_variability" in df.columns and "tempo_stability" not in df.columns:
        df["tempo_stability"] = 1.0 / (1.0 + pd.to_numeric(df["beat_interval_variability"], errors="coerce").values)

    # Cast numeric columns where present
    numeric_candidates = [
        "chart_score", "release_year", "tempo", "onset_rate", "tempo_stability",
        "energy", "rms_variability", "loudness_proxy", "spectral_centroid",
        "spectral_bandwidth", "chord_change_rate", "key_confidence", "mode",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
        "word_count", "repetition_ratio", "avg_word_length", "line_count",
        "chorus_repetition_count", "punctuation_ratio"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if select:
        avail_cols = [c for c in select if c in df.columns]
        for extra in ["audio_mode", "mode"]:
            if extra in df.columns and extra not in avail_cols:
                avail_cols.append(extra)
        df = df[avail_cols]
    
    if limit:
        df = df.head(limit)

    # Fingerprint over all model columns: sorted by (spotify_track_id, market), rounded to 6 decimal places
    sort_cols = [c for c in ["spotify_track_id", "market"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    model_fp_cols = [c for c in DEFAULT_SELECT_COLUMNS if c in df_sorted.columns]
    df_fp = df_sorted[model_fp_cols].copy()
    for col in df_fp.select_dtypes(include=[np.floating, float]).columns:
        df_fp[col] = df_fp[col].round(6)
    fp_bytes = df_fp.to_csv(index=False).encode("utf-8")
    data_fingerprint = hashlib.sha256(fp_bytes).hexdigest()

    df.attrs["data_source"] = data_source
    df.attrs["data_fingerprint"] = data_fingerprint
    print(f"[SupabaseClient] Data Source: {data_source} | Rows: {len(df):,} | Fingerprint: {data_fingerprint}")

    return df

    return df


if __name__ == "__main__":
    df = load_clean_modeling_data(source="auto")
    print("\nDataset Summary:")
    print("Shape:", df.shape)
    print("Markets distribution:", df["market"].value_counts().to_dict() if "market" in df else "N/A")