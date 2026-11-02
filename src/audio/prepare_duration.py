
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/sampling/youtube_matching_results.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "audio_duration_reference.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "spotify_track_id",
        "duration_ms",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["spotify_track_id"] = (
        df["spotify_track_id"].astype("string").str.strip()
    )
    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"], errors="coerce"
    )

    # Convert Spotify duration from milliseconds to seconds.
    df["duration_seconds"] = df["duration_ms"] / 1000

    # Validate IDs and durations.
    invalid = (
        df["spotify_track_id"].isna()
        | df["spotify_track_id"].eq("")
        | df["duration_seconds"].isna()
        | df["duration_seconds"].le(0)
    )

    if invalid.any():
        print(
            f"Warning: {invalid.sum()} rows have invalid IDs or durations."
        )

    output = df.loc[
        ~invalid, ["spotify_track_id", "duration_seconds"]
    ].copy()

    # Each Spotify track ID should appear only once.
    duplicates = output["spotify_track_id"].duplicated(keep=False)

    if duplicates.any():
        duplicate_ids = (
            output.loc[duplicates, "spotify_track_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Duplicate Spotify IDs found. Resolve them before "
            f"creating features: {duplicate_ids[:10]}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Rows saved: {len(output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
