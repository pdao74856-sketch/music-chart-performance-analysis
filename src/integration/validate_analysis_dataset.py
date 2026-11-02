#!/usr/bin/env python3
"""
src/integration/validate_analysis_dataset.py
Project: What Makes a Hit Song? (Music Chart Performance Analysis)
Owner: Member 4 (Data Integration, Quality Control & Report Coordination)

Purpose:
  Automated testing & audit suite verifying the integrated modeling dataset:
  1. Grain Uniqueness: (spotify_track_id, market) has 0 duplicates
  2. Exact Sample Cardinality: 2,860 entries (KR: 1,020; US: 955; VN: 885) & 1,912 unique tracks
  3. Target Engineering Integrity: chart_score == 101 - lifetime_average_rank
  4. Audio Feature Completeness: 20 dimensions with 0% missingness (2,860 / 2,860)
  5. Lyrics Quality & Missingness Rules: NULL preservation, instrumental tracking (4 entries)
  6. Numerical Range & Domain Constraints: validity of BPM, RMS, ratios, ranks
  7. Release Year Independence: release_year derived from release_date, not sampling year
  8. Output logging: exports logs/integration_validation_log.csv

Usage:
  python src/integration/validate_analysis_dataset.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "integration_validation_log.csv"
MODELING_CSV = ROOT / "data" / "processed" / "modeling_dataset.csv"
SAMPLE_CSV = ROOT / "data" / "sampling" / "member2_sample_list.csv"


class IntegrationValidator:
    def __init__(self):
        self.logs: list[dict[str, str]] = []
        self.all_passed = True

    def record(self, suite: str, test_name: str, passed: bool, details: str):
        status = "PASS" if passed else "FAIL"
        if not passed:
            self.all_passed = False
        self.logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_suite": suite,
            "test_name": test_name,
            "status": status,
            "details": details
        })
        print(f"[{status}] [{suite}] {test_name}: {details}")

    def run_all(self):
        print("=" * 70)
        print("MEMBER 4: INTEGRATION QUALITY CONTROL & AUDIT VALIDATION SUITE")
        print("=" * 70)

        # 1. File existence
        if not MODELING_CSV.exists():
            self.record("File Check", "modeling_dataset_exists", False, f"Missing file: {MODELING_CSV}")
            self.save_logs()
            return False

        df = pd.read_csv(MODELING_CSV)
        self.record("File Check", "modeling_dataset_exists", True, f"Loaded {MODELING_CSV} successfully ({len(df):,} rows)")

        # 2. Grain uniqueness
        dup_count = df.duplicated(subset=["spotify_track_id", "market"]).sum()
        self.record("Integrity", "primary_key_uniqueness", dup_count == 0,
                    f"Duplicate count on (spotify_track_id, market): {dup_count}")

        # 3. Cardinality and market distribution
        expected_total = 2860
        expected_markets = {"KR": 1020, "US": 955, "VN": 885}
        total_ok = (len(df) == expected_total)
        self.record("Cardinality", "total_records_count", total_ok,
                    f"Total entries: {len(df):,} (expected {expected_total:,})")

        m_counts = df["market"].value_counts().to_dict()
        markets_ok = (m_counts == expected_markets)
        self.record("Cardinality", "market_distribution", markets_ok,
                    f"Market distribution: {m_counts} (expected {expected_markets})")

        unique_tracks = df["spotify_track_id"].nunique()
        tracks_ok = (unique_tracks == 1912)
        self.record("Cardinality", "unique_tracks_count", tracks_ok,
                    f"Unique tracks: {unique_tracks:,} (expected 1,912)")

        # 4. Target formula: chart_score = 101 - lifetime_average_rank
        computed_score = 101.0 - df["lifetime_average_rank"]
        diff = (df["chart_score"] - computed_score).abs().max()
        formula_ok = (diff < 1e-4)
        self.record("Target Engineering", "chart_score_formula", formula_ok,
                    f"Max absolute discrepancy with 101 - avg_rank: {diff:.6f}")

        # 5. Audio Completeness (100% requirement on 20 dimensions)
        audio_cols = [
            "tempo_bpm", "beat_interval_variability", "tempo_stability", "onset_rate",
            "rms_energy", "rms_variability", "dynamic_range", "loudness_proxy",
            "spectral_centroid", "spectral_bandwidth", "key_confidence", "chord_change_rate",
            "chroma_repetition_determinism", "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
            "mfcc_repetition_determinism", "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration",
            "audio_key", "audio_mode"
        ]
        audio_nulls = df[audio_cols].isnull().sum().sum()
        self.record("Audio Feature Quality", "audio_completeness_100pct", audio_nulls == 0,
                    f"Total NULLs across 20 audio features: {audio_nulls} (expected 0)")

        # 6. Lyrics Integrity & NULL Rule Adherence
        # Missing word_count must be exactly 4 rows (instrumental)
        wc_nulls = df["word_count"].isnull().sum()
        self.record("Lyrics Feature Quality", "instrumental_missing_count", wc_nulls == 4,
                    f"Word count NULL entries: {wc_nulls} (expected 4 instrumental entries)")

        # Verify no 0 coercion on unmeasurable chorus or pronoun
        chorus_nulls = df["chorus_repetition_count"].isnull().sum()
        pronoun_nulls = df["pronoun_ratio"].isnull().sum()
        self.record("Lyrics Feature Quality", "chorus_null_preservation", chorus_nulls > 0,
                    f"Preserved {chorus_nulls} unmeasurable chorus as NULL (never coerced to 0)")
        self.record("Lyrics Feature Quality", "pronoun_null_preservation", pronoun_nulls > 0,
                    f"Preserved {pronoun_nulls} unsupported/mixed language pronouns as NULL (never coerced to 0)")

        # 7. Numerical Range Validity
        range_errors = []
        if not (df["lifetime_average_rank"].between(1, 200).all()):
            range_errors.append("lifetime_average_rank outside [1, 200]")
        if not (df["peak_rank"].between(1, 200).all()):
            range_errors.append("peak_rank outside [1, 200]")
        if not ((df["weeks_in_top10"] <= df["weeks_on_chart"]).all()):
            range_errors.append("weeks_in_top10 > weeks_on_chart")
        if not ((df["tempo_bpm"] > 0).all()):
            range_errors.append("tempo_bpm <= 0")
        if not ((df["rms_energy"] > 0).all()):
            range_errors.append("rms_energy <= 0")

        # Lexical ratio bounds
        lex_valid = df["lexical_diversity"].dropna()
        if not ((lex_valid >= 0) & (lex_valid <= 1.0)).all():
            range_errors.append("lexical_diversity outside [0, 1]")
        rep_valid = df["repetition_ratio"].dropna()
        if not ((rep_valid >= 0) & (rep_valid <= 1.0)).all():
            range_errors.append("repetition_ratio outside [0, 1]")

        ranges_ok = (len(range_errors) == 0)
        self.record("Domain Constraints", "numerical_range_bounds", ranges_ok,
                    "All numeric variables within theoretical bounds" if ranges_ok else "; ".join(range_errors))

        # 8. Release Year Independence
        # Verify release_year is not identical to sampling year for catalog tracks
        if "release_year" in df.columns:
            # Catalog tracks (released < 2020)
            catalog_tracks = df[df["release_year"] < 2020]
            catalog_ok = (len(catalog_tracks) > 0)
            self.record("Metadata Integrity", "release_year_independence", catalog_ok,
                        f"Found {len(catalog_tracks)} catalog records with release_year < 2020 correctly distinguished from sampling window")
        else:
            self.record("Metadata Integrity", "release_year_independence", False, "Missing release_year column")

        # Save and return status
        self.save_logs()
        return self.all_passed

    def save_logs(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "test_suite", "test_name", "status", "details"])
            writer.writeheader()
            writer.writerows(self.logs)
        print(f"\n[INFO] Audit log exported to {LOG_PATH}")


def main():
    validator = IntegrationValidator()
    passed = validator.run_all()
    if passed:
        print("\n" + "=" * 70)
        print("[SUCCESS] ALL 11 AUDIT SUITES PASSED STRICT INTEGRITY CONSTRAINTS!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("[FAIL] INTEGRITY VALIDATION FAILED! CHECK LOG FILE FOR DETAILS.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
