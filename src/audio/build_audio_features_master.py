"""
Merge all audio feature extraction outputs into a single master dataset
matching the exact schema of public.audio_features in Supabase.

Input files in data/processed/:
- audio_duration_reference.csv
- audio_rhythm_features.csv
- audio_energy_features.csv
- audio_spectral_features.csv
- audio_tonal_features.csv
- audio_repetition_features.csv

Output file:
- data/processed/audio_features_master.csv
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "audio_features_master.csv"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


def fetch_youtube_mappings() -> dict[str, int]:
    """Fetch all is_selected mapping IDs from Supabase to populate youtube_mapping_id."""
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("Supabase credentials not found in .env; skipping youtube_mapping_id fetch.")
            return {}

        client = create_client(url, key)
        mapping_dict = {}
        page = 0
        page_size = 1000

        print("Fetching youtube_mapping_id from Supabase...")
        while True:
            start = page * page_size
            end = start + page_size - 1
            res = (
                client.table("youtube_mapping")
                .select("id, spotify_track_id")
                .eq("is_selected", True)
                .range(start, end)
                .execute()
            )
            for row in res.data:
                mapping_dict[row["spotify_track_id"]] = row["id"]
            if len(res.data) < page_size:
                break
            page += 1

        print(f"Successfully fetched {len(mapping_dict)} mapping IDs.")
        return mapping_dict
    except Exception as exc:
        print(f"Warning: Could not fetch youtube_mapping IDs from Supabase: {exc}")
        return {}


def main():
    print("Building audio_features_master.csv without self_similarity...")

    # 1. Load duration reference
    df_duration = pd.read_csv(PROCESSED_DIR / "audio_duration_reference.csv")
    print(f"Loaded duration: {len(df_duration)} rows")

    # 2. Load rhythm features
    df_rhythm = pd.read_csv(PROCESSED_DIR / "audio_rhythm_features.csv")
    print(f"Loaded rhythm: {len(df_rhythm)} rows")

    # 3. Load energy features
    df_energy = pd.read_csv(PROCESSED_DIR / "audio_energy_features.csv")
    print(f"Loaded energy: {len(df_energy)} rows")

    # 4. Load spectral features
    df_spectral = pd.read_csv(PROCESSED_DIR / "audio_spectral_features.csv")
    print(f"Loaded spectral: {len(df_spectral)} rows")

    # 5. Load tonal features (key, mode, confidence, chord change rate)
    df_tonal = pd.read_csv(PROCESSED_DIR / "audio_tonal_features.csv")
    print(f"Loaded tonal: {len(df_tonal)} rows")

    # 6. Load repetition features (RQA)
    df_rep = pd.read_csv(PROCESSED_DIR / "audio_repetition_features.csv")
    print(f"Loaded repetition: {len(df_rep)} rows")

    # Merge all on spotify_track_id
    master = df_duration.merge(df_rhythm, on="spotify_track_id", how="inner")
    master = master.merge(df_energy, on="spotify_track_id", how="inner")
    master = master.merge(df_spectral, on="spotify_track_id", how="inner")
    master = master.merge(df_tonal, on="spotify_track_id", how="inner")
    master = master.merge(df_rep, on="spotify_track_id", how="inner")

    print(f"Merged master dataset: {len(master)} rows")

    # Fetch youtube_mapping_id
    yt_mappings = fetch_youtube_mappings()
    if yt_mappings:
        master["youtube_mapping_id"] = master["spotify_track_id"].map(yt_mappings)
    else:
        master["youtube_mapping_id"] = None

    # Populate legacy alias columns
    master["tempo"] = master["tempo_bpm"]
    master["energy"] = master["rms_energy"]
    master["loudness_proxy"] = master["dynamic_range"]
    master["tempo_stability"] = master["beat_interval_variability"]

    # Status columns
    master["audio_status"] = "success"
    master["audio_error"] = None

    # Align columns to Supabase schema order (without self_similarity)
    ordered_columns = [
        "spotify_track_id",
        "youtube_mapping_id",
        # Legacy columns
        "tempo",
        "energy",
        "loudness_proxy",
        "tempo_stability",
        # Rhythm
        "tempo_bpm",
        "onset_rate",
        "beat_interval_variability",
        # Energy
        "rms_energy",
        "rms_variability",
        "dynamic_range",
        # Spectral
        "spectral_centroid",
        "spectral_bandwidth",
        # Duration
        "duration_seconds",
        "duration_analyzed_seconds",
        # Tonal & Harmony
        "key",
        "mode",
        "key_confidence",
        "chord_change_rate",
        # Repetition (RQA standardized)
        "chroma_repetition_determinism",
        "chroma_mean_pattern_duration",
        "chroma_longest_pattern_duration",
        "mfcc_repetition_determinism",
        "mfcc_mean_pattern_duration",
        "mfcc_longest_pattern_duration",
        # Metadata / Status
        "audio_status",
        "audio_error",
    ]

    master = master[ordered_columns]

    # QA checks
    assert len(master) == 1912, f"Expected 1912 rows, got {len(master)}"
    assert master["spotify_track_id"].nunique() == 1912, "Duplicate IDs detected!"
    assert "self_similarity" not in master.columns, "self_similarity should be dropped!"
    
    # Save master file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n========== MASTER AUDIO FEATURES SUMMARY ==========")
    print(f"Saved master file to: {OUTPUT_FILE}")
    print(f"Total songs: {len(master)}")
    print(f"Total columns: {len(master.columns)}")
    print(f"Columns: {master.columns.tolist()}")
    print("Missing values per column:")
    print(master.isnull().sum()[master.isnull().sum() > 0])


if __name__ == "__main__":
    main()
