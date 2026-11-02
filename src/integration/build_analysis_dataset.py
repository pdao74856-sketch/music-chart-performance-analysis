#!/usr/bin/env python3
"""
src/integration/build_analysis_dataset.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Integration, Quality Control & Report Coordination)

Purpose:
  Builds the unified master modeling dataset combining:
    - Target ChartScore & chart longevity from Member 1 (chart_performance)
    - 20 acoustic signal features from Member 2 (audio_features)
    - 10 NLP lexical features from Member 3 (lyrics_features)
    - Track metadata and release_year from songs & sampling provenance
    - Preserves primary grain: 1 row = 1 (spotify_track_id, market)
    - Outputs modeling_dataset.csv, missingness_by_market.csv, excluded_records.csv,
      and descriptive_statistics.csv.

Usage:
  python src/integration/build_analysis_dataset.py                 # offline local mode (default)
  python src/integration/build_analysis_dataset.py --source supabase  # online Supabase mode
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]


def load_data_local() -> dict[str, pd.DataFrame]:
    """Loads all raw and processed source CSVs from local repo repository."""
    print("[1/5] Loading source files from local repository...")

    # Member 1: Chart Performance
    cp_path = ROOT / "reports" / "report4" / "chart_data" / "member1" / "chart_performance_source.csv"
    if not cp_path.exists():
        raise FileNotFoundError(f"Missing {cp_path}")
    df_cp = pd.read_csv(cp_path)
    print(f"  -> Chart performance loaded: {len(df_cp):,} rows")

    # Member 2: Audio Features
    af_path = ROOT / "reports" / "report4" / "chart_data" / "member2" / "audio_features_eda_source.csv"
    if not af_path.exists():
        raise FileNotFoundError(f"Missing {af_path}")
    df_af = pd.read_csv(af_path)
    print(f"  -> Audio features loaded: {len(df_af):,} rows ({df_af['spotify_track_id'].nunique():,} unique tracks)")

    # Member 3: Lyrics Features
    lf_path = ROOT / "reports" / "report3" / "lyrics_features_extracted.csv"
    if not lf_path.exists():
        raise FileNotFoundError(f"Missing {lf_path}")
    df_lf = pd.read_csv(lf_path)
    print(f"  -> Lyrics features loaded: {len(df_lf):,} rows ({df_lf['spotify_track_id'].nunique():,} unique tracks)")

    # Sampling provenance
    sample_path = ROOT / "data" / "sampling" / "member2_sample_list.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}")
    df_sample = pd.read_csv(sample_path)
    print(f"  -> Sample list loaded: {len(df_sample):,} rows ({df_sample['spotify_track_id'].nunique():,} unique tracks)")

    # YouTube mapping
    yt_path = ROOT / "data" / "sampling" / "youtube_matching_results.csv"
    df_yt = pd.read_csv(yt_path) if yt_path.exists() else pd.DataFrame()
    print(f"  -> YouTube mapping loaded: {len(df_yt):,} rows")

    return {
        "cp": df_cp,
        "af": df_af,
        "lf": df_lf,
        "sample": df_sample,
        "yt": df_yt,
    }


def load_data_supabase() -> dict[str, pd.DataFrame]:
    """Loads datasets directly from Supabase if credentials are present, else fallbacks to local."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("[!] Supabase credentials not found in .env; falling back to local files.")
        return load_data_local()

    try:
        from supabase import create_client
        sb = create_client(url, key)
        print("[1/5] Querying public.modeling_dataset from Supabase...")
        res = sb.table("modeling_dataset").select("*").eq("is_sampled", True).execute()
        df = pd.DataFrame(res.data)
        print(f"  -> Fetched {len(df)} records from Supabase.")
        # If successfully fetched from view, return as dict
        return {"modeling_view": df}
    except Exception as exc:
        print(f"[!] Warning: Supabase query failed ({exc}). Falling back to local files.")
        return load_data_local()


def integrate_datasets(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes relational integration:
    - Filters chart_performance to sampled tracks (1,912 tracks across 23 quarters).
    - Joins metadata from sample_list (title, artist, release_date).
    - Derives release_year from release_date (strictly independent from sampling year).
    - Joins Member 2 audio features on (spotify_track_id, market).
    - Joins Member 3 lyrics features on spotify_track_id.
    - Adds data quality audit flags.
    """
    print("[2/5] Performing deterministic relational joins...")
    df_cp = data["cp"]
    df_af = data["af"]
    df_lf = data["lf"]
    df_sample = data["sample"]
    df_yt = data["yt"]

    # 1. Unique sampled track IDs
    sampled_track_ids = set(df_sample["spotify_track_id"].unique())
    print(f"  -> Unique sampled tracks in project scope: {len(sampled_track_ids):,}")

    # 2. Filter chart_performance to sampled tracks (grain: spotify_track_id, market)
    cp_sampled = df_cp[df_cp["spotify_track_id"].isin(sampled_track_ids)].copy()
    print(f"  -> Chart performance entries for sampled tracks: {len(cp_sampled):,} (KR: {(cp_sampled['market']=='KR').sum()}, US: {(cp_sampled['market']=='US').sum()}, VN: {(cp_sampled['market']=='VN').sum()})")

    # 3. Metadata extraction from sample list (deduplicated by spotify_track_id)
    meta_cols = ["spotify_track_id", "title", "artist", "release_date"]
    df_meta = df_sample[meta_cols].drop_duplicates("spotify_track_id").copy()
    df_meta["release_date"] = pd.to_datetime(df_meta["release_date"], errors="coerce")
    df_meta["release_year"] = df_meta["release_date"].dt.year.astype("Int64")

    # Join metadata
    m0 = cp_sampled.merge(df_meta, on="spotify_track_id", how="left")

    # 4. Join Audio Features on (spotify_track_id, market)
    # Audio features has 20 columns plus spotify_track_id, market
    audio_feature_cols = [
        "tempo_bpm", "beat_interval_variability", "tempo_stability", "onset_rate",
        "rms_energy", "rms_variability", "dynamic_range", "loudness_proxy",
        "spectral_centroid", "spectral_bandwidth", "key_confidence", "chord_change_rate",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
        "audio_key", "audio_mode"
    ]
    af_subset = df_af[["spotify_track_id", "market"] + [c for c in audio_feature_cols if c in df_af.columns]].copy()
    m1 = m0.merge(af_subset, on=["spotify_track_id", "market"], how="left")

    # 5. Join YouTube mapping audit
    if not df_yt.empty and "spotify_track_id" in df_yt.columns:
        yt_sub = df_yt[["spotify_track_id", "youtube_video_id", "is_official"]].drop_duplicates("spotify_track_id")
        m1 = m1.merge(yt_sub, on="spotify_track_id", how="left")
        m1["audio_mapping_is_selected"] = True
    else:
        m1["audio_mapping_is_selected"] = True

    # 6. Join Lyrics Features on spotify_track_id (track-level NLP features)
    lyrics_cols = [
        "word_count", "unique_word_count", "lexical_diversity", "repetition_ratio",
        "avg_word_length", "line_count", "chorus_repetition_count", "title_word_in_lyrics",
        "punctuation_ratio", "pronoun_ratio", "language", "lyrics_status", "extraction_status"
    ]
    lf_subset = df_lf[["spotify_track_id"] + [c for c in lyrics_cols if c in df_lf.columns]].drop_duplicates("spotify_track_id").copy()
    m2 = m1.merge(lf_subset, on="spotify_track_id", how="left")

    # 7. Audit quality flags
    m2["has_audio"] = m2["tempo_bpm"].notnull()
    m2["has_lyrics_row"] = m2["lyrics_status"].notnull()
    m2["has_both_rows"] = m2["has_audio"] & (m2["lyrics_status"] == "success")
    m2["is_sampled"] = True

    # 8. Check uniqueness of primary grain
    dups = m2.duplicated(subset=["spotify_track_id", "market"]).sum()
    if dups > 0:
        raise ValueError(f"Integration integrity violation: {dups} duplicate primary keys on (spotify_track_id, market)!")
    print(f"  -> Verified zero duplicate keys on (spotify_track_id, market). Total rows: {len(m2):,}")

    # Build longitudinal quarterly observations dataset (N = 2,760)
    print("  -> Constructing longitudinal quarterly trend dataset (2,760 observations)...")
    q_df = df_sample.copy()
    q_df["release_date"] = pd.to_datetime(q_df["release_date"], errors="coerce")
    q_df["release_year"] = q_df["release_date"].dt.year.astype("Int64")
    # Merge audio features at track level
    af_track = df_af.drop_duplicates("spotify_track_id")[["spotify_track_id"] + [c for c in audio_feature_cols if c in df_af.columns]]
    q_df = q_df.merge(af_track, on="spotify_track_id", how="left")
    q_df = q_df.merge(lf_subset, on="spotify_track_id", how="left")

    return m2, q_df


def generate_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generates detailed missingness breakdown by market and overall."""
    print("[3/5] Computing missingness and exclusion metrics...")
    markets = ["VN", "US", "KR", "Overall"]
    records = []

    features = [
        "lifetime_average_rank", "chart_score", "peak_rank", "weeks_on_chart", "weeks_in_top10",
        "tempo_bpm", "beat_interval_variability", "tempo_stability", "onset_rate",
        "rms_energy", "rms_variability", "dynamic_range", "loudness_proxy",
        "spectral_centroid", "spectral_bandwidth", "key_confidence", "chord_change_rate",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
        "audio_key", "audio_mode",
        "word_count", "unique_word_count", "lexical_diversity", "repetition_ratio",
        "avg_word_length", "line_count", "chorus_repetition_count", "title_word_in_lyrics",
        "punctuation_ratio", "pronoun_ratio", "language"
    ]

    for feat in features:
        if feat not in df.columns:
            continue
        row_data = {"feature": feat}
        for m in ["VN", "US", "KR"]:
            sub = df[df["market"] == m]
            total_n = len(sub)
            null_n = sub[feat].isnull().sum()
            row_data[f"{m}_total"] = total_n
            row_data[f"{m}_missing"] = null_n
            row_data[f"{m}_pct"] = round(null_n / total_n * 100, 2)

        # Overall
        total_all = len(df)
        null_all = df[feat].isnull().sum()
        row_data["Overall_total"] = total_all
        row_data["Overall_missing"] = null_all
        row_data["Overall_pct"] = round(null_all / total_all * 100, 2)

        # Classification of reason
        if null_all == 0:
            row_data["missing_reason"] = "Complete (0% missing)"
        elif feat in ["word_count", "unique_word_count", "lexical_diversity", "repetition_ratio", "avg_word_length", "line_count", "punctuation_ratio"]:
            row_data["missing_reason"] = "Instrumental tracks without lyrics (unavailable_instrumental)"
        elif feat == "chorus_repetition_count":
            row_data["missing_reason"] = "Chorus structure not reliably identifiable (team convention: NULL, never 0)"
        elif feat == "pronoun_ratio":
            row_data["missing_reason"] = "Multilingual/unsupported language dictionary or instrumental (NULL)"
        elif feat == "title_word_in_lyrics":
            row_data["missing_reason"] = "Language-limited stopword filtering or instrumental (NULL)"
        else:
            row_data["missing_reason"] = "Data unavailable"

        records.append(row_data)

    return pd.DataFrame(records)


def generate_excluded_records_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies all records excluded under complete-case analysis and classifies reasons.
    """
    excluded = []
    for idx, row in df.iterrows():
        reasons = []
        # Check lyrics status
        if row.get("lyrics_status") == "not_found":
            reasons.append("Instrumental track (no vocal lyrics)")
        # Check unmeasurable NLP
        if pd.isnull(row.get("chorus_repetition_count")):
            reasons.append("Unmeasurable chorus repetition")
        if pd.isnull(row.get("pronoun_ratio")):
            reasons.append(f"Pronoun ratio undefined for language '{row.get('language')}'")
        if pd.isnull(row.get("title_word_in_lyrics")):
            reasons.append("Title word match unmeasurable")

        if reasons:
            excluded.append({
                "spotify_track_id": row["spotify_track_id"],
                "market": row["market"],
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "language": row.get("language", ""),
                "lyrics_status": row.get("lyrics_status", ""),
                "chart_score": row.get("chart_score", np.nan),
                "exclusion_reasons": "; ".join(reasons),
                "complete_case_excluded": True,
                "audio_available": bool(row.get("has_audio", False)),
                "lyrics_available": bool(row.get("lyrics_status") == "success")
            })

    return pd.DataFrame(excluded)


def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes master parametric and non-parametric summary statistics."""
    print("[4/5] Computing parametric and non-parametric descriptive statistics...")
    numeric_cols = [
        "chart_score", "lifetime_average_rank", "peak_rank", "weeks_on_chart", "weeks_in_top10",
        "tempo_bpm", "beat_interval_variability", "tempo_stability", "onset_rate",
        "rms_energy", "rms_variability", "dynamic_range", "loudness_proxy",
        "spectral_centroid", "spectral_bandwidth", "key_confidence", "chord_change_rate",
        "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
        "word_count", "unique_word_count", "lexical_diversity", "repetition_ratio",
        "avg_word_length", "line_count", "chorus_repetition_count", "punctuation_ratio", "pronoun_ratio"
    ]

    records = []
    groups = [("VN", df[df["market"] == "VN"]),
              ("US", df[df["market"] == "US"]),
              ("KR", df[df["market"] == "KR"]),
              ("Overall", df)]

    for market_name, sub in groups:
        for col in numeric_cols:
            if col not in sub.columns:
                continue
            series = sub[col].dropna()
            n_valid = len(series)
            n_missing = len(sub) - n_valid
            if n_valid == 0:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            sk = float(stats.skew(series, bias=False)) if n_valid > 2 else np.nan
            kt = float(stats.kurtosis(series, bias=False)) if n_valid > 3 else np.nan

            records.append({
                "market": market_name,
                "feature": col,
                "n_obs": len(sub),
                "valid_n": n_valid,
                "missing_n": n_missing,
                "missing_pct": round(n_missing / len(sub) * 100, 2),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "median": round(float(series.median()), 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(sk, 4) if not np.isnan(sk) else None,
                "kurtosis": round(kt, 4) if not np.isnan(kt) else None,
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Member 4: Build Analysis Dataset")
    parser.add_argument("--source", choices=["local", "supabase"], default="local",
                        help="Data source mode: 'local' (offline CSVs) or 'supabase' (REST API)")
    args = parser.parse_args()

    print("=" * 70)
    print("MEMBER 4: UNIFIED DATASET INTEGRATION PIPELINE (REPORT 3 & 4)")
    print(f"Data source: {args.source.upper()}")
    print("=" * 70)

    # 1. Load data
    data = load_data_supabase() if args.source == "supabase" else load_data_local()

    # 2. Integrate
    df_modeling, df_quarterly = integrate_datasets(data)

    # 3. Create missingness report
    df_missing = generate_missingness_report(df_modeling)

    # 4. Create excluded records report
    df_excluded = generate_excluded_records_report(df_modeling)

    # 5. Create descriptive statistics
    df_stats = generate_descriptive_statistics(df_modeling)

    # 6. Save outputs
    print("[5/5] Exporting integrated datasets and quality reports...")
    out_dir_proc = ROOT / "data" / "processed"
    out_dir_r3 = ROOT / "reports" / "report3"
    out_dir_proc.mkdir(parents=True, exist_ok=True)
    out_dir_r3.mkdir(parents=True, exist_ok=True)

    # Modeling dataset master
    p_model = out_dir_proc / "modeling_dataset.csv"
    df_modeling.to_csv(p_model, index=False, encoding="utf-8-sig")
    print(f"  -> Saved master modeling dataset: {p_model} ({len(df_modeling):,} rows)")

    # Modeling dataset complete-case (for regression benchmarks)
    df_complete = df_modeling[~df_modeling["spotify_track_id"].isin(df_excluded["spotify_track_id"])]
    p_comp = out_dir_proc / "modeling_dataset_complete_case.csv"
    df_complete.to_csv(p_comp, index=False, encoding="utf-8-sig")
    print(f"  -> Saved complete-case modeling dataset: {p_comp} ({len(df_complete):,} rows)")

    # Quarterly trends
    p_q = out_dir_proc / "quarterly_feature_trends.csv"
    df_quarterly.to_csv(p_q, index=False, encoding="utf-8-sig")
    print(f"  -> Saved quarterly trend dataset: {p_q} ({len(df_quarterly):,} rows)")

    # Missingness by market
    p_miss = out_dir_r3 / "missingness_by_market.csv"
    df_missing.to_csv(p_miss, index=False, encoding="utf-8-sig")
    print(f"  -> Saved missingness report: {p_miss}")

    # Excluded records
    p_ex = out_dir_r3 / "excluded_records.csv"
    df_excluded.to_csv(p_ex, index=False, encoding="utf-8-sig")
    print(f"  -> Saved excluded records audit: {p_ex} ({len(df_excluded):,} excluded entries)")

    # Descriptive statistics
    p_stat = out_dir_r3 / "descriptive_statistics.csv"
    df_stats.to_csv(p_stat, index=False, encoding="utf-8-sig")
    print(f"  -> Saved descriptive statistics table: {p_stat} ({len(df_stats):,} metrics)")

    print("\n[SUCCESS] Integration pipeline completed flawlessly.")


if __name__ == "__main__":
    main()
