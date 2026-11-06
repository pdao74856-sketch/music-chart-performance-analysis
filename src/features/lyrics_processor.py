"""Stable-ID interface to the project's canonical lyrics feature extractor."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.lyrics.extract_lyrics_features import (
    FEATURE_NAMES, calculate_features, load_yaml,
)

MODEL2_FEATURES = [
    "word_count", "repetition_ratio", "avg_word_length", "line_count",
    "chorus_repetition_count", "punctuation_ratio",
]


def process_lyrics_text(
    spotify_track_id: str, text: str | None, *, title: str = "",
    language: str = "mixed", config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the six Model 2 variables while preserving missing/error state."""
    if not isinstance(spotify_track_id, str) or not spotify_track_id.strip():
        raise ValueError("spotify_track_id is required")
    root = Path(__file__).resolve().parents[2]
    cfg = config if config is not None else load_yaml(root / "src/lyrics/language_rules.yaml")
    status = "success"
    if text is None:
        status = "missing_file"
    elif not text.strip():
        status = "empty_file"
    if status == "success":
        extracted = calculate_features(text, title, language, cfg)
        result = {feature: extracted.get(feature) for feature in MODEL2_FEATURES}
        error_reason = None
        if extracted.get("word_count") == 0:
            status = "failed"
            error_reason = "no_tokens_recognized_after_tokenization"
    else:
        result = {feature: None for feature in MODEL2_FEATURES}
        error_reason = None
    return {"spotify_track_id": str(spotify_track_id), **result,
            "extraction_status": status, "error_reason": error_reason}


def process_lyrics_file(
    spotify_track_id: str, lyrics_dir: str | Path, *, title: str = "",
    language: str = "mixed", config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(lyrics_dir) / f"{spotify_track_id}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = None
    except (OSError, UnicodeError):
        row = process_lyrics_text(spotify_track_id, None, title=title,
                                  language=language, config=config)
        row["extraction_status"] = "read_error"
        return row
    return process_lyrics_text(spotify_track_id, text, title=title,
                               language=language, config=config)


def process_metadata(metadata: pd.DataFrame, lyrics_dir: str | Path,
                     *, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Process one metadata row per track; output remains unique by Spotify ID."""
    if "spotify_track_id" not in metadata:
        raise ValueError("metadata requires spotify_track_id")
    raw_ids = metadata["spotify_track_id"].tolist()
    invalid_rows = [index for index, value in zip(metadata.index, raw_ids)
                    if not isinstance(value, str) or not value.strip()]
    if invalid_rows:
        raise ValueError(f"metadata has missing or invalid spotify_track_id at rows: {invalid_rows}")
    if pd.Series(raw_ids).duplicated().any():
        raise ValueError("metadata contains duplicate spotify_track_id values")
    rows = []
    for row in metadata.to_dict("records"):
        rows.append(process_lyrics_file(
            row["spotify_track_id"], lyrics_dir,
            title=str(row.get("title") or ""),
            language=str(row.get("language") or "mixed"), config=config,
        ))
    return pd.DataFrame(rows)


__all__ = ["FEATURE_NAMES", "MODEL2_FEATURES", "process_lyrics_text",
           "process_lyrics_file", "process_metadata"]
