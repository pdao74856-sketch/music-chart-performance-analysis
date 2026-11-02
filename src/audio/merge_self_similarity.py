
from pathlib import Path
import pandas as pd

root = Path.cwd()  # Chạy từ thư mục gốc dự án

original_file = (
    root / "data/processed/audio_self_similarity_features.csv"
)
retry_file = (
    root / "data/processed/audio_self_similarity_retry.csv"
)
merged_file = (
    root / "data/processed/audio_self_similarity_features_merged.csv"
)

original = pd.read_csv(original_file)
retry = pd.read_csv(retry_file)

if original["spotify_track_id"].duplicated().any():
    raise ValueError("Original file contains duplicate track IDs.")

if retry["spotify_track_id"].duplicated().any():
    raise ValueError("Retry file contains duplicate track IDs.")

overlap = set(original["spotify_track_id"].astype(str)) & set(
    retry["spotify_track_id"].astype(str)
)

if overlap:
    raise ValueError(
        f"Retry IDs already exist in original output: {len(overlap)}"
    )

# Keep the original feature schema.
retry_for_merge = retry[
    ["spotify_track_id", "self_similarity"]
]

merged = pd.concat(
    [
        original[["spotify_track_id", "self_similarity"]],
        retry_for_merge,
    ],
    ignore_index=True,
)

if merged["spotify_track_id"].duplicated().any():
    raise ValueError("Duplicate IDs detected after merging.")

if merged["self_similarity"].isna().any():
    raise ValueError("Missing self_similarity values detected.")

if not merged["self_similarity"].between(-1, 1).all():
    raise ValueError("Self-similarity outside [-1, 1].")

merged.to_csv(merged_file, index=False)

print(f"Original successful songs: {len(original)}")
print(f"Successfully retried songs: {len(retry)}")
print(f"Total songs after merge: {len(merged)}")
print(f"Saved to: {merged_file}")
