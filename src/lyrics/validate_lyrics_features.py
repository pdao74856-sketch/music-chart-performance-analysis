#!/usr/bin/env python3
"""Validate extracted lyrics features and cross-artifact consistency for ADY201m."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT = "reports/report3/lyrics_features_extracted.csv"
DEFAULT_LOG = "logs/lyrics_extraction_log.csv"
DEFAULT_INVENTORY = "reports/report3/lyrics_inventory.csv"
DEFAULT_LYRICS_DIR = "data/raw/lyrics"
DEFAULT_OUTPUT = "reports/report3/lyrics_feature_validation.csv"

FEATURES = [
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
]

REQUIRED_COLUMNS = [
    "spotify_track_id",
    "language",
    "lyrics_status",
    "extraction_status",
    *FEATURES,
]

RATIO_COLUMNS = [
    "lexical_diversity",
    "repetition_ratio",
    "punctuation_ratio",
    "pronoun_ratio",
]

REQUIRED_MEASURABLE_FEATURES = [
    "word_count",
    "unique_word_count",
    "lexical_diversity",
    "repetition_ratio",
    "avg_word_length",
    "line_count",
    "punctuation_ratio",
]

PRONOUN_SUPPORTED_LANGUAGES = {"en", "vi", "ko", "es"}
TITLE_SUPPORTED_LANGUAGES = {"en", "vi", "ko", "mixed", "es"}
UNAVAILABLE_STATUSES = {"not_found", "missing_file", "empty_file", "read_error", "not_attempted"}

CHORUS_TAG_RE = re.compile(
    r"\[\s*(?:chorus|refrain|hook|điệp khúc)\b[^\]]*\]",
    re.IGNORECASE,
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file while preserving empty cells."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def is_missing(value: Any) -> bool:
    return value is None or not str(value).strip()


def parse_number(value: Any) -> float | None:
    """Return a finite float, or None when the value is invalid."""
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def has_explicit_chorus_tag(track_id: str, lyrics_dir: Path | None) -> bool:
    if not track_id or lyrics_dir is None or not lyrics_dir.is_dir():
        return True
    p = lyrics_dir / f"{track_id}.txt"
    if not p.is_file():
        return False
    try:
        text = p.read_text(encoding="utf-8-sig")
    except Exception:
        return False
    return bool(CHORUS_TAG_RE.search(text))


def validate_row(row: dict[str, Any], lyrics_dir: Path | None = None) -> list[str]:
    """Return a list of validation issues for one feature row."""
    issues: list[str] = []

    track_id = (row.get("spotify_track_id") or "").strip()
    status = (row.get("extraction_status") or "").strip()
    lyrics_status = (row.get("lyrics_status") or "").strip().lower()
    language = (row.get("language") or "").strip().lower()

    if not track_id:
        issues.append("missing_spotify_track_id")

    if not status:
        issues.append("missing_extraction_status")

    # 1. Unavailable lyrics must never have features filled with 0 or non-null values
    if status in UNAVAILABLE_STATUSES or lyrics_status != "success":
        for col in FEATURES:
            if not is_missing(row.get(col)):
                issues.append(f"unexpected_non_null_{col}_for_{status or lyrics_status}")

    # 2. Measurable basic features for success and language_limited
    if status in ("success", "language_limited"):
        for column in REQUIRED_MEASURABLE_FEATURES:
            if is_missing(row.get(column)):
                issues.append(f"missing_{column}")

        word_count = parse_number(row.get("word_count"))
        unique_count = parse_number(row.get("unique_word_count"))
        lexical = parse_number(row.get("lexical_diversity"))
        repetition = parse_number(row.get("repetition_ratio"))
        avg_length = parse_number(row.get("avg_word_length"))
        line_count = parse_number(row.get("line_count"))

        if word_count is not None:
            if word_count <= 0 or not word_count.is_integer():
                issues.append("invalid_word_count")

        if unique_count is not None:
            if unique_count <= 0 or not unique_count.is_integer():
                issues.append("invalid_unique_word_count")

        if word_count is not None and unique_count is not None and word_count > 0:
            if unique_count > word_count:
                issues.append("unique_word_count_exceeds_word_count")

        if avg_length is not None and avg_length <= 0:
            issues.append("invalid_avg_word_length")

        if line_count is not None:
            if line_count <= 0 or not line_count.is_integer():
                issues.append("invalid_line_count")

        if (
            word_count is not None
            and unique_count is not None
            and word_count > 0
            and lexical is not None
            and repetition is not None
        ):
            expected_lexical = unique_count / word_count
            expected_repetition = (word_count - unique_count) / word_count

            if not math.isclose(lexical, expected_lexical, rel_tol=1e-6, abs_tol=1e-6):
                issues.append("lexical_diversity_formula_mismatch")

            if not math.isclose(repetition, expected_repetition, rel_tol=1e-6, abs_tol=1e-6):
                issues.append("repetition_ratio_formula_mismatch")

            if not math.isclose(lexical + repetition, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                issues.append("lexical_plus_repetition_not_one")

    # 3. Validate non-negative counts for any row where present
    for count_col in ("word_count", "unique_word_count", "line_count"):
        val = row.get(count_col)
        if not is_missing(val):
            num = parse_number(val)
            if num is None or num < 0 or not num.is_integer():
                if f"invalid_{count_col}" not in issues:
                    issues.append(f"invalid_{count_col}")

    if not is_missing(row.get("avg_word_length")):
        avg_num = parse_number(row.get("avg_word_length"))
        if avg_num is None or avg_num < 0:
            if "invalid_avg_word_length" not in issues:
                issues.append("invalid_avg_word_length")

    # 4. Validate all non-null ratios, regardless of extraction status
    for column in RATIO_COLUMNS:
        value = row.get(column)
        if is_missing(value):
            continue
        number = parse_number(value)
        if number is None:
            issues.append(f"invalid_number_{column}")
        elif not 0.0 <= number <= 1.0:
            issues.append(f"out_of_range_{column}")

    # 5. Validate language-dependent features are not falsely filled with 0
    if language not in PRONOUN_SUPPORTED_LANGUAGES:
        if not is_missing(row.get("pronoun_ratio")):
            issues.append("pronoun_ratio_present_for_unsupported_or_mixed_language")
    elif status == "success" and is_missing(row.get("pronoun_ratio")):
        issues.append("missing_pronoun_ratio_for_supported_language")

    if language not in TITLE_SUPPORTED_LANGUAGES:
        if not is_missing(row.get("title_word_in_lyrics")):
            issues.append("title_word_in_lyrics_present_for_unsupported_language")

    # 6. Validate boolean values when present
    title_value = row.get("title_word_in_lyrics")
    if not is_missing(title_value):
        if str(title_value).strip().lower() not in {"true", "false"}:
            issues.append("invalid_title_word_in_lyrics")

    # 7. Validate chorus_repetition_count: never 0 unless explicit [Chorus] tag exists
    chorus_value = row.get("chorus_repetition_count")
    if not is_missing(chorus_value):
        chorus_count = parse_number(chorus_value)
        if chorus_count is None or chorus_count < 0 or not chorus_count.is_integer():
            issues.append("invalid_chorus_repetition_count")
        elif int(chorus_count) == 0 and not has_explicit_chorus_tag(track_id, lyrics_dir):
            issues.append("chorus_repetition_zero_without_explicit_chorus_tag")

    if status == "language_limited" and not language:
        issues.append("missing_language_for_language_limited_row")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extracted lyrics feature CSV.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to lyrics_features_extracted.csv")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to lyrics_extraction_log.csv")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY, help="Path to lyrics_inventory.csv")
    parser.add_argument("--lyrics-dir", default=DEFAULT_LYRICS_DIR, help="Path to raw lyrics directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to validation report CSV")
    parser.add_argument("--check-db", action="store_true", help="Also verify consistency with Supabase lyrics_features")
    args = parser.parse_args()

    input_path = Path(args.input)
    log_path = Path(args.log)
    inventory_path = Path(args.inventory)
    lyrics_dir = Path(args.lyrics_dir) if args.lyrics_dir else None
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Cannot find feature CSV: {input_path}", file=sys.stderr)
        return 2

    headers, rows = read_csv(input_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(headers))
    if missing_columns:
        print(
            "ERROR: Feature CSV is missing required columns: " + ", ".join(missing_columns),
            file=sys.stderr,
        )
        return 2

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    validation_rows: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    row_by_id: dict[str, dict[str, str]] = {}

    for row in rows:
        track_id = (row.get("spotify_track_id") or "").strip()
        if track_id and track_id in seen_ids:
            duplicate_ids.add(track_id)
        if track_id:
            seen_ids.add(track_id)
            row_by_id[track_id] = row

        issues = validate_row(row, lyrics_dir=lyrics_dir)
        validation_status = "pass" if not issues else "fail"
        status = (row.get("extraction_status") or "").strip()
        status_counts[status or "missing_status"] += 1

        for issue in issues:
            issue_counts[issue] += 1

        validation_rows.append({
            "spotify_track_id": track_id,
            "extraction_status": status,
            "validation_status": validation_status,
            "issues": ";".join(issues),
        })

    # Mark duplicate Track IDs in the validation report
    for result in validation_rows:
        if result["spotify_track_id"] in duplicate_ids:
            result["validation_status"] = "fail"
            existing = result["issues"]
            duplicate_issue = "duplicate_spotify_track_id"
            if duplicate_issue not in existing.split(";"):
                result["issues"] = (existing + ";" + duplicate_issue).strip(";")
            issue_counts[duplicate_issue] += 1

    # Cross-artifact consistency checks: Log & Inventory
    cross_artifact_issues: list[str] = []

    if log_path.exists():
        _, log_rows = read_csv(log_path)
        log_by_id = {(r.get("spotify_track_id") or "").strip(): r for r in log_rows}
        if len(log_rows) != len(rows):
            cross_artifact_issues.append(f"log_row_count_mismatch:{len(log_rows)}!={len(rows)}")
        for tid, f_row in row_by_id.items():
            l_row = log_by_id.get(tid)
            if not l_row:
                cross_artifact_issues.append(f"missing_in_log:{tid}")
            else:
                if (l_row.get("language") or "").strip() != (f_row.get("language") or "").strip():
                    cross_artifact_issues.append(f"log_language_mismatch:{tid}")
                if (l_row.get("extraction_status") or "").strip() != (f_row.get("extraction_status") or "").strip():
                    cross_artifact_issues.append(f"log_status_mismatch:{tid}")
                if (l_row.get("word_count") or "").strip() != (f_row.get("word_count") or "").strip():
                    cross_artifact_issues.append(f"log_word_count_mismatch:{tid}")
    else:
        cross_artifact_issues.append(f"log_file_not_found:{log_path}")

    if inventory_path.exists():
        _, inv_rows = read_csv(inventory_path)
        inv_by_id = {(r.get("spotify_track_id") or "").strip(): r for r in inv_rows}
        if len(inv_rows) != len(rows):
            cross_artifact_issues.append(f"inventory_row_count_mismatch:{len(inv_rows)}!={len(rows)}")
        for tid, f_row in row_by_id.items():
            i_row = inv_by_id.get(tid)
            if not i_row:
                cross_artifact_issues.append(f"missing_in_inventory:{tid}")
            else:
                if (i_row.get("language") or "").strip() != (f_row.get("language") or "").strip():
                    cross_artifact_issues.append(f"inventory_language_mismatch:{tid}")
                if (i_row.get("lyrics_status") or "").strip() != (f_row.get("lyrics_status") or "").strip():
                    cross_artifact_issues.append(f"inventory_lyrics_status_mismatch:{tid}")
                if (i_row.get("word_count") or "").strip() != (f_row.get("word_count") or "").strip():
                    cross_artifact_issues.append(f"inventory_word_count_mismatch:{tid}")
    else:
        cross_artifact_issues.append(f"inventory_file_not_found:{inventory_path}")

    if args.check_db:
        try:
            from dotenv import load_dotenv
            from supabase import create_client

            load_dotenv()
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if not url or not key:
                cross_artifact_issues.append("supabase_credentials_missing")
            else:
                sb = create_client(url, key)
                db_rows = []
                start = 0
                while True:
                    resp = sb.table("lyrics_features").select("*").range(start, start + 999).execute()
                    if not resp.data:
                        break
                    db_rows.extend(resp.data)
                    if len(resp.data) < 1000:
                        break
                    start += 1000
                if len(db_rows) != len(rows):
                    cross_artifact_issues.append(f"supabase_row_count_mismatch:{len(db_rows)}!={len(rows)}")
                db_by_id = {r["spotify_track_id"]: r for r in db_rows}
                for tid, f_row in row_by_id.items():
                    d_row = db_by_id.get(tid)
                    if not d_row:
                        cross_artifact_issues.append(f"missing_in_supabase:{tid}")
                    else:
                        if (d_row.get("language") or "") != (f_row.get("language") or ""):
                            cross_artifact_issues.append(f"supabase_language_mismatch:{tid}")
                        if (d_row.get("lyrics_status") or "") != (f_row.get("lyrics_status") or ""):
                            cross_artifact_issues.append(f"supabase_status_mismatch:{tid}")
                        f_wc = int(f_row["word_count"]) if f_row.get("word_count") else None
                        if d_row.get("word_count") != f_wc:
                            cross_artifact_issues.append(f"supabase_word_count_mismatch:{tid}")
        except Exception as exc:
            cross_artifact_issues.append(f"supabase_check_error:{type(exc).__name__}:{exc}")

    for issue in cross_artifact_issues:
        issue_counts[issue.split(":")[0]] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["spotify_track_id", "extraction_status", "validation_status", "issues"],
        )
        writer.writeheader()
        writer.writerows(validation_rows)

    total_rows = len(rows)
    pass_count = sum(r["validation_status"] == "pass" for r in validation_rows)
    fail_count = total_rows - pass_count
    measurable_count = status_counts["success"] + status_counts["language_limited"]

    print("Lyrics feature validation finished.")
    print(f"Input rows: {total_rows}")
    print(f"Unique non-empty Track IDs: {len(seen_ids)}")
    print(f"Duplicate Track IDs: {len(duplicate_ids)}")
    print(f"Validation pass: {pass_count}/{total_rows} ({pass_count / total_rows * 100:.2f}%)" if total_rows else "Validation pass: 0")
    print(f"Validation failures: {fail_count}/{total_rows} ({fail_count / total_rows * 100:.2f}%)" if total_rows else "Validation failures: 0")
    if total_rows:
        print(
            f"Lyrics feature coverage (success + language_limited): "
            f"{measurable_count}/{total_rows} ({measurable_count / total_rows * 100:.2f}%)"
        )

    print("\nExtraction status counts:")
    for status, count in sorted(status_counts.items()):
        pct = (count / total_rows * 100) if total_rows else 0.0
        print(f"  {status}: {count} ({pct:.2f}%)")

    print("\nCross-artifact consistency:")
    if cross_artifact_issues:
        for item in cross_artifact_issues[:20]:
            print(f"  ISSUE: {item}")
    else:
        print("  PASS: Features CSV, Extraction Log, and Inventory are 100% consistent.")
        if args.check_db:
            print("  PASS: Supabase lyrics_features table is 100% consistent with Features CSV.")

    print("\nValidation issues:")
    if issue_counts:
        for issue, count in sorted(issue_counts.items()):
            print(f"  {issue}: {count}")
    else:
        print("  No issues found.")

    print(f"\nValidation report: {output_path}")
    return 1 if (issue_counts or cross_artifact_issues or fail_count > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
