#!/usr/bin/env python3
"""Upsert extracted lyrics features into Supabase public.lyrics_features table.

Safety guarantees:
- Never prints or logs SUPABASE_URL, SUPABASE_KEY, or any secrets.
- Uses spotify_track_id as conflict key (idempotent on re-run, never creates duplicates).
- Never deletes tracks that are not in the current batch.
- Preserves NULL for unmeasurable/missing features (never coerces NULL to 0).
- Verifies data by reading back rows from Supabase after upserting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import create_client

DEFAULT_INPUT = "reports/report3/lyrics_features_extracted.csv"
DEFAULT_LOG_JSON = "logs/supabase_upsert_log.json"

INT_COLUMNS = {
    "word_count",
    "unique_word_count",
    "line_count",
    "chorus_repetition_count",
}
FLOAT_COLUMNS = {
    "lexical_diversity",
    "repetition_ratio",
    "avg_word_length",
    "punctuation_ratio",
    "pronoun_ratio",
}
BOOL_COLUMNS = {"title_word_in_lyrics"}
TEXT_COLUMNS = {"spotify_track_id", "language", "lyrics_status"}

DB_COLUMNS = [
    "spotify_track_id",
    "word_count",
    "unique_word_count",
    "lexical_diversity",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "chorus_repetition_count",
    "title_word_in_lyrics",
    "punctuation_ratio",
    "pronoun_ratio",
    "language",
    "lyrics_status",
]


def parse_db_value(col: str, raw: Any) -> Any:
    if raw is None:
        return None
    val_str = str(raw).strip()
    if not val_str:
        return None
    if col in INT_COLUMNS:
        return int(float(val_str))
    if col in FLOAT_COLUMNS:
        num = float(val_str)
        return num if math.isfinite(num) else None
    if col in BOOL_COLUMNS:
        lower = val_str.lower()
        if lower in ("true", "1"):
            return True
        if lower in ("false", "0"):
            return False
        return None
    return val_str


def fetch_existing_track_ids(supabase_client: Any, page_size: int = 1000) -> set[str]:
    existing: set[str] = set()
    start = 0
    while True:
        resp = (
            supabase_client.table("lyrics_features")
            .select("spotify_track_id")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        for r in rows:
            tid = r.get("spotify_track_id")
            if tid:
                existing.add(tid)
        if len(rows) < page_size:
            break
        start += page_size
    return existing


def fetch_all_db_rows(supabase_client: Any, page_size: int = 1000) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    start = 0
    while True:
        resp = (
            supabase_client.table("lyrics_features")
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert lyrics_features to Supabase.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to lyrics_features_extracted.csv")
    parser.add_argument("--log-json", default=DEFAULT_LOG_JSON, help="Path to upsert summary log JSON")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for Supabase upsert")
    args = parser.parse_args()

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_KEY in environment.", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: Input CSV not found: {input_path}", file=sys.stderr)
        return 2

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))

    sb = create_client(supabase_url, supabase_key)
    existing_before = fetch_existing_track_ids(sb)
    print(f"Existing rows in Supabase lyrics_features before upsert: {len(existing_before)}")

    payloads: list[dict[str, Any]] = []
    seen_in_batch: set[str] = set()
    insert_count = 0
    update_count = 0
    skip_count = 0
    error_count = 0
    errors: list[str] = []

    for row in csv_rows:
        tid = (row.get("spotify_track_id") or "").strip()
        if not tid or tid in seen_in_batch:
            skip_count += 1
            continue
        seen_in_batch.add(tid)

        try:
            record = {col: parse_db_value(col, row.get(col)) for col in DB_COLUMNS}
            payloads.append(record)
            if tid in existing_before:
                update_count += 1
            else:
                insert_count += 1
        except Exception as exc:
            error_count += 1
            errors.append(f"{tid}:{type(exc).__name__}:{exc}")

    # Upsert in batches
    batch_size = max(1, args.batch_size)
    upserted_total = 0
    for i in range(0, len(payloads), batch_size):
        batch = payloads[i : i + batch_size]
        try:
            sb.table("lyrics_features").upsert(batch, on_conflict="spotify_track_id").execute()
            upserted_total += len(batch)
            print(f"Upserted batch {i // batch_size + 1}: {upserted_total}/{len(payloads)} rows")
        except Exception as exc:
            error_count += len(batch)
            errors.append(f"batch_{i}:{type(exc).__name__}:{exc}")
            print(f"ERROR upserting batch starting at {i}: {type(exc).__name__}", file=sys.stderr)
            return 1

    # Read back and verify
    db_rows_after = fetch_all_db_rows(sb)
    db_by_id = {r["spotify_track_id"]: r for r in db_rows_after}

    mismatch_count = 0
    for rec in payloads:
        tid = rec["spotify_track_id"]
        db_rec = db_by_id.get(tid)
        if not db_rec:
            mismatch_count += 1
            errors.append(f"missing_after_upsert:{tid}")
            continue
        for col in ("word_count", "unique_word_count", "line_count", "chorus_repetition_count", "title_word_in_lyrics", "language", "lyrics_status"):
            if db_rec.get(col) != rec.get(col):
                mismatch_count += 1
                errors.append(f"value_mismatch:{tid}:{col}")

    non_null_counts = {
        col: sum(1 for r in db_rows_after if r.get(col) is not None)
        for col in DB_COLUMNS
    }
    lang_dist = Counter(r.get("language") for r in db_rows_after)
    status_dist = Counter(r.get("lyrics_status") for r in db_rows_after)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table": "public.lyrics_features",
        "input_csv_rows": len(csv_rows),
        "existing_rows_before": len(existing_before),
        "inserted_count": insert_count,
        "updated_count": update_count,
        "skipped_count": skip_count,
        "error_count": error_count,
        "verified_rows_after": len(db_rows_after),
        "verification_mismatches": mismatch_count,
        "non_null_column_counts": non_null_counts,
        "language_distribution": dict(lang_dist.most_common()),
        "lyrics_status_distribution": dict(status_dist),
        "errors": errors[:50],
    }

    log_path = Path(args.log_json)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== SUPABASE UPSERT & VERIFICATION SUMMARY =====")
    print(f"Input CSV rows:        {len(csv_rows)}")
    print(f"Inserted (new):        {insert_count}")
    print(f"Updated (existing):    {update_count}")
    print(f"Skipped:               {skip_count}")
    print(f"Errors:                {error_count}")
    print(f"Verified DB row count: {len(db_rows_after)}")
    print(f"Read-back mismatches:  {mismatch_count}")
    print(f"Summary log saved to:  {log_path}")

    return 0 if (error_count == 0 and mismatch_count == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
