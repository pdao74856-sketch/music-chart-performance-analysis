#!/usr/bin/env python3
"""Create data/processed/lyrics_status.csv from data/raw/lyrics_collection_results.json."""
import csv
import json
import os

RESULT_FILE = os.path.join("data", "raw", "lyrics_collection_results.json")
OUTPUT_FILE = os.path.join("data", "processed", "lyrics_status.csv")


def load_results():
    if not os.path.exists(RESULT_FILE):
        print("Không tìm thấy:")
        print(RESULT_FILE)
        raise SystemExit(1)
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_markets(item: dict) -> str:
    provenance = item.get("sampling_provenance") or []
    markets = sorted({
        str(entry.get("market")).strip()
        for entry in provenance
        if isinstance(entry, dict) and entry.get("market")
    })
    return ";".join(markets)


def create_handoff():
    results = load_results()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    fieldnames = [
        "spotify_track_id",
        "title",
        "artist",
        "market",
        "language",
        "source",
        "lyrics_status",
        "collection_status",
        "lyrics_file",
    ]

    rows = []
    for track_id, item in results.items():
        lang = item.get("language")
        lang_str = str(lang).strip() if lang is not None else ""
        rows.append({
            "spotify_track_id": track_id,
            "title": item.get("title", ""),
            "artist": item.get("artist", ""),
            "market": extract_markets(item),
            "language": lang_str,
            "source": item.get("source") or "",
            "lyrics_status": item.get("lyrics_status") or "",
            "collection_status": item.get("collection_status") or "",
            "lyrics_file": item.get("lyrics_file") or "",
        })

    rows.sort(key=lambda x: x["spotify_track_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(1 for row in rows if row["lyrics_status"] == "success")
    not_found = sum(1 for row in rows if row["lyrics_status"] == "not_found")
    failed = sum(1 for row in rows if row["lyrics_status"] == "failed")

    print("\nLYRICS HANDOFF CREATED")
    print(f"Total tracks: {len(rows)}")
    print(f"Lyrics success: {success}")
    print(f"Not found: {not_found}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_handoff()
