#!/usr/bin/env python3
"""Build complete raw lyrics inventory for Stage 2 (Member 3).

Outputs:
- reports/report3/lyrics_inventory.csv (primary deliverable)
- data/processed/lyrics_inventory.csv (processed copy)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow importing tokenizer from extract_lyrics_features
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lyrics.extract_lyrics_features import (
    SUPPORTED_LANGUAGES,
    load_yaml,
    normalize_text,
    resolve_lyrics_path,
    tokenize,
)

DEFAULT_STATUS_CSV = "data/processed/lyrics_status.csv"
DEFAULT_RESULTS_JSON = "data/raw/lyrics_collection_results.json"
DEFAULT_LYRICS_DIR = "data/raw/lyrics"
DEFAULT_CONFIG_YAML = "src/lyrics/language_rules.yaml"
DEFAULT_OUTPUT_REPORT = "reports/report3/lyrics_inventory.csv"
DEFAULT_OUTPUT_PROCESSED = "data/processed/lyrics_inventory.csv"

KNOWN_INSTRUMENTAL_IDS = {
    "6WY4wvlmgccWapnIg14Vy0",  # Merry Christmas Mr. Lawrence - Ryuichi Sakamoto
    "1Px44tw80Q4rcCl8FxDGOi",  # Who (Instrumental) - Jimin
    "7M30L99pePlQ3wh8IxVUsD",  # With You - Instrumental - Jimin
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_markets_from_json(json_path: Path) -> dict[str, str]:
    if not json_path.is_file():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    markets_by_id: dict[str, str] = {}
    for tid, item in data.items():
        prov = item.get("sampling_provenance") or []
        markets = sorted({
            str(entry.get("market")).strip()
            for entry in prov
            if isinstance(entry, dict) and entry.get("market")
        })
        markets_by_id[tid] = ";".join(markets)
    return markets_by_id


def is_instrumental_track(track_id: str, title: str) -> bool:
    if track_id in KNOWN_INSTRUMENTAL_IDS:
        return True
    return "instrumental" in (title or "").lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build raw lyrics inventory CSV.")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--input", default=DEFAULT_STATUS_CSV, help="Input lyrics_status.csv")
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON, help="Input lyrics_collection_results.json")
    parser.add_argument("--lyrics-dir", default=DEFAULT_LYRICS_DIR, help="Directory of raw .txt lyrics")
    parser.add_argument("--config", default=DEFAULT_CONFIG_YAML, help="Path to language_rules.yaml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_REPORT, help="Primary output CSV path")
    parser.add_argument("--secondary-output", default=DEFAULT_OUTPUT_PROCESSED, help="Secondary output CSV path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_file = (root / args.input).resolve()
    results_json = (root / args.results_json).resolve()
    lyrics_dir = (root / args.lyrics_dir).resolve()
    config_path = (root / args.config).resolve()
    output_report = (root / args.output).resolve()
    output_processed = (root / args.secondary_output).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {input_file}")
    if not lyrics_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {lyrics_dir}")

    config = load_yaml(config_path) if config_path.is_file() else {}
    markets_map = load_markets_from_json(results_json)
    rows = read_csv(input_file)

    existing_files = {
        path.name
        for path in lyrics_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    # Pre-scan for duplicate track IDs and duplicate file mappings
    track_id_counts = Counter((r.get("spotify_track_id") or "").strip() for r in rows)
    file_mapping_counts: Counter[str] = Counter()
    for r in rows:
        fp = resolve_lyrics_path(root, (r.get("lyrics_file") or "").strip(), lyrics_dir)
        if fp is not None:
            file_mapping_counts[fp.name] += 1

    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    referenced_files: set[str] = set()

    for row in rows:
        track_id = (row.get("spotify_track_id") or "").strip()
        title = (row.get("title") or "").strip()
        artist = (row.get("artist") or "").strip()
        market = (row.get("market") or "").strip() or markets_map.get(track_id, "")
        source = (row.get("source") or "").strip()
        language = (row.get("language") or "").strip().lower()
        lyrics_status = (row.get("lyrics_status") or "").strip().lower()
        lyrics_file = (row.get("lyrics_file") or "").strip()

        if language in ("", "unknown", "nan", "none"):
            language = "unknown"

        language_counts[language] += 1

        resolved_path = resolve_lyrics_path(root, lyrics_file, lyrics_dir)
        file_exists = bool(resolved_path is not None and resolved_path.is_file())
        is_readable: bool | str = False
        is_empty: bool | str = ""
        word_count: int | str = ""
        inventory_status = ""
        validation_status = ""
        error_reasons: list[str] = []

        if not track_id:
            error_reasons.append("missing_spotify_track_id")
        elif track_id_counts[track_id] > 1:
            error_reasons.append("duplicate_spotify_track_id")

        if resolved_path is not None and file_mapping_counts[resolved_path.name] > 1:
            error_reasons.append(f"multiple_tracks_mapped_to_file:{resolved_path.name}")

        if file_exists and resolved_path is not None:
            referenced_files.add(resolved_path.name)
            try:
                text = resolved_path.read_text(encoding="utf-8-sig")
                is_readable = True
                stripped = text.strip()
                is_empty = not bool(stripped)

                if is_empty:
                    inventory_status = "empty_file"
                    validation_status = "invalid_empty_file"
                    error_reasons.append("lyrics_file_is_empty_or_whitespace")
                else:
                    normalized = normalize_text(text, config)
                    tokens = tokenize(normalized, language, config)
                    word_count = len(tokens)
                    if word_count == 0:
                        inventory_status = "no_valid_tokens"
                        validation_status = "invalid_no_tokens"
                        error_reasons.append("no_tokens_after_normalization")
                    elif lyrics_status != "success":
                        inventory_status = "file_exists_status_mismatch"
                        validation_status = "warning_status_mismatch"
                        error_reasons.append(f"file_exists_but_lyrics_status={lyrics_status}")
                    elif language == "unknown":
                        inventory_status = "available_unknown_language"
                        validation_status = "valid_language_limited"
                        error_reasons.append("language_unknown")
                    elif language not in SUPPORTED_LANGUAGES:
                        inventory_status = "available"
                        validation_status = "valid_language_limited"
                        error_reasons.append(f"language_limited:{language}")
                    else:
                        inventory_status = "available"
                        validation_status = "valid"
            except (OSError, UnicodeError) as exc:
                is_readable = False
                is_empty = ""
                inventory_status = "read_error"
                validation_status = "invalid_read_error"
                error_reasons.append(f"{type(exc).__name__}:{exc}")
        else:
            is_readable = False
            is_empty = ""
            if lyrics_status == "success":
                inventory_status = "missing_file"
                validation_status = "invalid_missing_file"
                error_reasons.append("lyrics_status_success_but_file_missing")
            elif is_instrumental_track(track_id, title):
                inventory_status = "lyrics_unavailable_instrumental"
                validation_status = "unavailable_instrumental"
                error_reasons.append("instrumental_track_no_vocal_lyrics")
            else:
                inventory_status = "lyrics_unavailable"
                validation_status = "unavailable_not_found"
                error_reasons.append(f"lyrics_status={lyrics_status or 'missing'}")

        if "duplicate_spotify_track_id" in error_reasons or any(
            r.startswith("multiple_tracks_mapped_to_file") for r in error_reasons
        ):
            validation_status = "invalid_duplicate_mapping"

        status_counts[inventory_status] += 1
        validation_counts[validation_status] += 1

        output_rows.append({
            "spotify_track_id": track_id,
            "title": title,
            "artist": artist,
            "market": market,
            "source": source,
            "lyrics_status": lyrics_status,
            "language": language,
            "lyrics_file": lyrics_file,
            "file_exists": str(file_exists).lower(),
            "is_empty": str(is_empty).lower() if isinstance(is_empty, bool) else "",
            "is_readable": str(is_readable).lower(),
            "word_count": word_count,
            "inventory_status": inventory_status,
            "validation_status": validation_status,
            "error_reason": ";".join(error_reasons),
        })

    orphan_files = sorted(existing_files - referenced_files)

    fields = [
        "spotify_track_id",
        "title",
        "artist",
        "market",
        "source",
        "lyrics_status",
        "language",
        "lyrics_file",
        "file_exists",
        "is_empty",
        "is_readable",
        "word_count",
        "inventory_status",
        "validation_status",
        "error_reason",
    ]

    for out_path in (output_report, output_processed):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    print("===== LYRICS INVENTORY =====")
    print(f"Tổng số dòng metadata: {len(rows)}")
    print(f"Số file lyrics trong thư mục: {len(existing_files)}")
    print(f"Số file không được metadata tham chiếu (orphan): {len(orphan_files)}")
    print("\nTrạng thái kiểm kê (inventory_status):")
    for st, cnt in sorted(status_counts.items()):
        print(f"  {st}: {cnt}")
    print("\nTrạng thái xác thực (validation_status):")
    for vst, cnt in sorted(validation_counts.items()):
        print(f"  {vst}: {cnt}")
    print("\nPhân bố ngôn ngữ (language):")
    for lang, cnt in language_counts.most_common():
        print(f"  {lang}: {cnt}")
    print(f"\nĐã lưu: {output_report}")
    print(f"Đã lưu bản sao processed: {output_processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
