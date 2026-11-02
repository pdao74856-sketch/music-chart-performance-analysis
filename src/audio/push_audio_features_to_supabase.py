"""
Upload master audio features dataset to Supabase audio_features table.

Prerequisites:
- Supabase migration 20261025041000_update_audio_features_repetition_tonal.sql
  must be applied (either via GitHub Actions CI/CD or Supabase CLI).
- Environment variables SUPABASE_URL and SUPABASE_KEY set in .env.

Usage:
    python src/audio/push_audio_features_to_supabase.py [--dry-run] [--batch-size 500]
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = PROJECT_ROOT / "data" / "processed" / "audio_features_master.csv"

load_dotenv(PROJECT_ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Upload master audio features to Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="Validate records without sending to Supabase.")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for upsert.")
    args = parser.parse_args()

    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master file not found at: {MASTER_CSV}")

    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded master file with {len(df)} rows and {len(df.columns)} columns.")

    # Replace NaN with None for valid JSON serialization
    df = df.replace({np.nan: None})
    records = df.to_dict(orient="records")

    if args.dry_run:
        print(f"[DRY-RUN] Validation passed. {len(records)} records ready for upload.")
        print(f"Sample record keys: {list(records[0].keys())}")
        return

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in .env")

    from supabase import create_client
    client = create_client(url, key)

    total = len(records)
    print(f"Uploading {total} records to 'audio_features' table in batches of {args.batch_size}...")

    for i in range(0, total, args.batch_size):
        batch = records[i : i + args.batch_size]
        client.table("audio_features").upsert(batch, on_conflict="spotify_track_id").execute()
        print(f"  Uploaded batch [{i + 1} - {min(i + len(batch), total)} / {total}] OK")

    print("\n========== UPLOAD COMPLETED SUCCESSFULLY ==========")
    print(f"Total {total} records successfully upserted into Supabase 'audio_features'.")


if __name__ == "__main__":
    main()
