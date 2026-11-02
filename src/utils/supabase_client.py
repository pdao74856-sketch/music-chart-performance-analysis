"""
Module: src.utils.supabase_client
Shared Supabase client and data loader for project modeling.
Supports online REST fetching with automatic local caching.
"""

import os
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd
from dotenv import dotenv_values

# Default local cache path
CACHE_PATH = Path("data/processed/modeling_dataset_clean_cache.csv")


# Default 32 clean modeling columns to select from Supabase REST API
DEFAULT_SELECT_COLUMNS = [
    # Identifiers & metadata (7)
    "spotify_track_id",
    "market",
    "chart_score",
    "title",
    "artist",
    "album",
    "release_date",
    # Year control & mode (2)
    "release_year",
    "audio_mode",
    # 16 Audio features Librosa (16)
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
    "chroma_repetition_determinism",
    "chroma_mean_pattern_duration",
    "chroma_longest_pattern_duration",
    "mfcc_repetition_determinism",
    "mfcc_mean_pattern_duration",
    "mfcc_longest_pattern_duration",
    # 6 Lyrics features NLP (6)
    "word_count",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "punctuation_ratio",
    # Audit column (1)
    "lyrics_status",
]


def get_supabase_credentials() -> Dict[str, str]:
    """Retrieves Supabase credentials from local .env or environment variables."""
    # Try dotenv first from project root
    env_path = Path(".env")
    cfg = dotenv_values(env_path) if env_path.exists() else {}
    
    url = os.getenv("SUPABASE_URL") or cfg.get("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY") or cfg.get("SUPABASE_KEY", "")
    
    # Fallback to streamlit secrets if running inside streamlit app
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_KEY", "")
        except Exception:
            pass

    return {"url": url.strip(), "key": key.strip()}


def load_clean_modeling_data(
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_path: Path = CACHE_PATH,
    select: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Loads the clean modeling dataset (is_sampled = TRUE AND has_both_rows = TRUE).
    Tries local cache first if available and force_refresh is False.
    Otherwise queries Supabase REST API with pagination.

    Parameters
    ----------
    use_cache : bool
        Whether to load from/save to local CSV cache.
    force_refresh : bool
        If True, ignores local cache and queries Supabase directly.
    cache_path : Path
        Path to local CSV cache.
    select : list of str or str, optional
        Specific columns to query via REST API to minimize RAM and network bandwidth.
        Defaults to DEFAULT_SELECT_COLUMNS.
    """
    if use_cache and not force_refresh and cache_path.exists():
        df_cached = pd.read_csv(cache_path)
        return df_cached

    creds = get_supabase_credentials()
    url = creds["url"]
    key = creds["key"]

    if not url or not key:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise ValueError(
            "Supabase credentials not found in .env, environment, or Streamlit secrets, "
            f"and local cache '{cache_path}' does not exist."
        )

    # Prepare select columns string
    if select is None:
        select_cols = DEFAULT_SELECT_COLUMNS
    elif isinstance(select, str):
        select_cols = [c.strip() for c in select.split(",")]
    else:
        select_cols = list(select)
    select_param = ",".join(select_cols)

    # Query with pagination (1000 items per request)
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    page_size = 1000

    while True:
        endpoint = (
            f"{url}/rest/v1/modeling_dataset?"
            f"is_sampled=eq.true&has_both_rows=eq.true"
            f"&select={urllib.parse.quote(select_param)}"
            f"&offset={offset}&limit={page_size}"
        )
        req = urllib.request.Request(
            endpoint,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            }
        )
        try:
            with urllib.request.urlopen(req) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
                if not batch:
                    break
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as e:
            if cache_path.exists():
                print(f"[WARN] Failed to query Supabase ({e}). Falling back to cached data.")
                return pd.read_csv(cache_path)
            raise RuntimeError(f"Error fetching data from Supabase REST API: {e}")

    df = pd.DataFrame(all_rows)

    # Save to local cache
    if use_cache and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df
