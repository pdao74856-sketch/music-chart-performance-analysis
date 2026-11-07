"""Fit Member 3's audio baseline and audio-plus-lyrics models on get_split().

Historical results from the pre-canonical pipeline remain in reports/report4.
This runner writes the migrated results to reports/report4/member3_canonical.
Run from the repository root: python -m src.modeling.train_model2_lyrics
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.features.prepare_features import SplitResult, get_split
from src.utils.ols_cluster_fit import fit_ols_cluster, metrics, wald_block
from src.utils.supabase_client import load_clean_modeling_data

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "reports/report4/member3_canonical"
CACHE_PATH = ROOT / "data/processed/modeling_dataset_clean_cache.csv"
EXPECTED_CACHE_ROWS = 2098

AUDIO_FEATURES = [
    "tempo_std", "onset_rate_std", "tempo_stability_std", "energy_std",
    "rms_variability_std", "spectral_centroid_std", "chord_change_rate_std",
    "key_confidence_std", "chroma_repetition_determinism_std",
    "chroma_mean_pattern_duration_std", "chroma_longest_pattern_duration_std",
    "mfcc_repetition_determinism_std", "mfcc_mean_pattern_duration_std",
    "mfcc_longest_pattern_duration_std",
]
LYRICS_FEATURES = [
    "word_count_std", "repetition_ratio_std", "avg_word_length_std",
    "line_count_std", "chorus_repetition_count_std", "punctuation_ratio_std",
]
CONTROL_FEATURES = ["release_year_std", "Market_VN", "Market_KR"]
MODEL1_FEATURES = AUDIO_FEATURES + ["mode"] + CONTROL_FEATURES
MODEL2_FEATURES = MODEL1_FEATURES + LYRICS_FEATURES


def load_member3_split() -> SplitResult:
    """Load the required local clean cache, then apply canonical preprocessing."""
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Required Member 3 cache is missing: {CACHE_PATH}")

    df = load_clean_modeling_data(source="local")
    if len(df) != EXPECTED_CACHE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CACHE_ROWS} rows from the local cache, got {len(df)}"
        )
    if df.attrs.get("data_source") != "cache":
        raise ValueError(
            "Local loader did not report the clean cache as its data source: "
            f"{df.attrs.get('data_source')!r}"
        )

    split = get_split(df=df)
    split.meta["data_source"] = "local cache"
    split.meta["data_path"] = CACHE_PATH.relative_to(ROOT).as_posix()
    split.meta["data_fingerprint"] = df.attrs.get("data_fingerprint")
    split.meta["source_rows"] = len(df)
    return split


def _validate_and_select_features(split: SplitResult) -> tuple[list[str], list[str]]:
    """Validate the agreed model design and return its explicit column lists."""
    expected_lyrics = [
        "word_count_std", "repetition_ratio_std", "avg_word_length_std",
        "line_count_std", "chorus_repetition_count_std", "punctuation_ratio_std",
    ]
    if LYRICS_FEATURES != expected_lyrics:
        raise AssertionError("Model 2 lyrics feature definition changed unexpectedly")

    required = MODEL2_FEATURES
    missing_from_contract = [c for c in required if c not in split.feature_names]
    missing_train = [c for c in required if c not in split.X_train.columns]
    missing_test = [c for c in required if c not in split.X_test.columns]
    if missing_from_contract or missing_train or missing_test:
        raise ValueError(
            "Canonical split is missing required model columns: "
            f"feature_names={missing_from_contract}, "
            f"X_train={missing_train}, X_test={missing_test}"
        )
    interaction_columns = [c for c in MODEL2_FEATURES if "__x__" in c]
    if interaction_columns:
        raise AssertionError(f"Interaction terms must be excluded: {interaction_columns}")
    if MODEL2_FEATURES[:len(MODEL1_FEATURES)] != MODEL1_FEATURES:
        raise AssertionError("Model 2 must equal Model 1 plus exactly six lyrics columns")
    return list(MODEL1_FEATURES), list(MODEL2_FEATURES)


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def run(output_dir: Path | None = None) -> dict[str, Any]:
    """Fit and evaluate the two agreed models using the canonical split only."""
    split = load_member3_split()
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    m1_features, m2_features = _validate_and_select_features(split)

    if len(split.X_train) != len(split.y_train) or len(split.ids_train) != len(split.y_train):
        raise ValueError("Canonical Train rows, targets, and IDs do not align")
    if len(split.X_test) != len(split.y_test) or len(split.ids_test) != len(split.y_test):
        raise ValueError("Canonical Test rows, targets, and IDs do not align")
    if not (split.X_train.index.equals(split.y_train.index) and
            split.X_train.index.equals(split.ids_train.index)):
        raise ValueError("Canonical Train matrix, target, and IDs have different row indexes")
    if not (split.X_test.index.equals(split.y_test.index) and
            split.X_test.index.equals(split.ids_test.index)):
        raise ValueError("Canonical Test matrix, target, and IDs have different row indexes")
    if set(split.ids_train).intersection(set(split.ids_test)):
        raise ValueError("Canonical Train/Test track IDs overlap")

    model_columns = {
        "Model 1 (audio + controls)": m1_features,
        "Model 2 (audio + lyrics + controls)": m2_features,
    }
    fitted, predictions, score_rows, metric_frames = {}, {}, [], []
    for model_name, columns in model_columns.items():
        result = fit_ols_cluster(split.y_train, split.X_train[columns], split.ids_train)
        fitted[model_name] = result
        test_design = sm.add_constant(split.X_test[columns], has_constant="add")
        prediction = np.asarray(result.predict(test_design))
        predictions[model_name] = prediction

        evaluated = metrics(
            split.y_test, prediction,
            market=split.market_test,
            track_ids=split.ids_test,
            n_boot=1000,
            seed=43,
        )
        evaluated.insert(0, "model", model_name)
        metric_frames.append(evaluated)
        overall = evaluated.loc[evaluated["Market"].eq("Overall")].iloc[0]
        score_rows.append({
            "model": model_name,
            "N": int(len(split.y_test)),
            "R2": float(overall["R2"]),
            "adjusted_R2": _adjusted_r2(float(overall["R2"]), len(split.y_test), len(columns)),
            "RMSE": float(overall["RMSE"]),
            "MAE": float(overall["MAE"]),
        })

    lyrics_wald = wald_block(fitted["Model 2 (audio + lyrics + controls)"], LYRICS_FEATURES)
    lyrics_wald.update({
        "method": "cluster-robust Wald chi-square",
        "alpha": 0.05,
        "conclusion": "reject H0" if lyrics_wald["p_value"] < 0.05 else "fail to reject H0",
    })

    pd.DataFrame(score_rows).to_csv(output_dir / "member3_model_comparison.csv", index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output_dir / "member3_model_test_metrics.csv", index=False
    )
    coefficient_rows = []
    for model_name, result in fitted.items():
        confidence = result.conf_int()
        for term in result.params.index:
            coefficient_rows.append({
                "model": model_name,
                "term": term,
                "coefficient": float(result.params[term]),
                "cluster_se": float(result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "ci_lower": float(confidence.loc[term, 0]),
                "ci_upper": float(confidence.loc[term, 1]),
            })
    pd.DataFrame(coefficient_rows).to_csv(
        output_dir / "member3_model_coefficients.csv", index=False
    )
    (output_dir / "member3_lyrics_joint_wald.json").write_text(
        json.dumps(lyrics_wald, indent=2) + "\n", encoding="utf-8"
    )

    excluded_tracks = split.meta.get("excluded_tracks", [])
    run_summary = {
        "source_pipeline": "src.features.prepare_features.get_split()",
        "data_source": split.meta["data_source"],
        "data_path": split.meta["data_path"],
        "data_fingerprint": split.meta["data_fingerprint"],
        "source_rows": int(split.meta["source_rows"]),
        "source_rows_before_canonical_cleaning": int(
            split.meta.get("initial_rows", len(split.X_train) + len(split.X_test))
        ),
        "total_rows_returned_by_get_split": int(len(split.X_train) + len(split.X_test)),
        "rows_after_canonical_cleaning": int(
            split.meta.get("cleaned_rows", len(split.X_train) + len(split.X_test))
        ),
        "train_observations": int(len(split.X_train)),
        "test_observations": int(len(split.X_test)),
        "train_unique_track_ids": int(split.ids_train.nunique()),
        "test_unique_track_ids": int(split.ids_test.nunique()),
        "train_track_market_observations": int(len(split.raw_train)),
        "test_track_market_observations": int(len(split.raw_test)),
        "train_test_track_ids_disjoint": not bool(
            set(split.ids_train).intersection(set(split.ids_test))
        ),
        "model1_features": m1_features,
        "model2_features": m2_features,
        "interaction_terms_excluded": True,
        "model1_model2_observations_identical": True,
        "canonical_preprocessing": {
            "lyrics_rows_dropped_when_all_lyrics_null_or_word_count_null": True,
            "chorus_repetition_count_missing_values_filled_with": 0.0,
            "other_required_nulls": "raise an error after cleaning",
            "word_count_transform": "log1p",
            "release_year_transform": "lower winsorization at Train 1st percentile",
            "continuous_scaling": "StandardScaler fit on Train only, applied to Train and Test",
            "additional_model_preprocessing": "none",
        },
        "excluded_rows_reported_by_get_split": int(len(excluded_tracks)),
        "excluded_tracks": excluded_tracks,
        "drop_provenance_limitations": (
            "get_split reports excluded track IDs, titles, markets, scores, and a reason; "
            "it does not provide source-row identifiers or a full audit of all upstream filtering."
        ),
        "source_row_count_meta_note": (
            "get_split metadata records input and cleaned rows separately; total_rows_returned_by_get_split "
            "is the retained Train plus Test observation count."
        ),
        "previous_canonical_artifacts_note": (
            "Previous committed canonical artifacts used the 2,856-row local fallback "
            "(2,296 Train / 560 Test); this summary records the new 2,098-row cache run."
        ),
        "split_seed": split.meta.get("seed"),
        "test_size": split.meta.get("test_size"),
        "winsorize_year_q01": split.meta.get("winsorize_year_q01"),
        "comparison": score_rows,
        "lyrics_joint_wald": lyrics_wald,
    }
    (output_dir / "member3_model_run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"comparison": score_rows, "wald": lyrics_wald, "run_summary": run_summary}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
