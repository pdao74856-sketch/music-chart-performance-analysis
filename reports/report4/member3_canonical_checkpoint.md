# Member 3 Canonical Pipeline Checkpoint

**Run date:** 2026-11-07 (Asia/Bangkok)
**Pipeline:** `src.features.prepare_features.get_split()`
**Status:** Model 2 rerun completed from the required local clean cache.

## Data source and canonical split

The runner requires `data/processed/modeling_dataset_clean_cache.csv` to exist, calls `load_clean_modeling_data(source="local")`, verifies the loader returns 2,098 rows, and passes that DataFrame to `get_split(df=...)`. This prevents Supabase from taking precedence and prevents fallback to the older CSV when the clean cache is missing. The loader log identified the source as `cache` and reported fingerprint `bb0aea6d42049da728c8339bbf92388991825d9128cb85122df081e71a0a6e81`.

| Count | Observations |
| --- | ---: |
| Local cache input | 2,098 |
| Rows retained after canonical cleaning | 2,095 |
| Train observations | 1,685 |
| Test observations | 410 |
| Unique Train track IDs | 1,527 |
| Unique Test track IDs | 382 |
| Rows excluded by `get_split()` | 3 |

Train and Test track IDs are disjoint. The split uses seed 43 and `test_size=0.2`. The three rows excluded by the canonical lyrics drop rule are listed with IDs and reasons in `member3_model_run_summary.json`.

## Prior canonical run and snapshot provenance

The previously committed canonical artifacts were generated from the older local `data/processed/modeling_dataset.csv` fallback: 2,856 input and retained observations, 2,296 Train, 560 Test, and Wald chi-square(6) = 9.9845, p = 0.1253. Those results were not from the 2,098-row clean cache. This checkpoint records both runs distinctly. The previous root-level historical checkpoint artifacts have been cleaned up to prevent contradiction and maintain a single source of truth; all validated results are consolidated under `reports/report4/member3_canonical/`.

The results below are from the 2,098-row cache identified by the fingerprint above. No result was adjusted to match an expected p-value.

## Models and preprocessing

Both models use the same canonical Train and Test observations. Model 1 has 18 features. Model 2 has 24 features: Model 1 plus exactly six lyrics features. No interaction terms are included.

**Model 1 — Audio baseline:**

```text
tempo_std, onset_rate_std, tempo_stability_std, energy_std,
rms_variability_std, spectral_centroid_std, chord_change_rate_std,
key_confidence_std, chroma_repetition_determinism_std,
chroma_mean_pattern_duration_std, chroma_longest_pattern_duration_std,
mfcc_repetition_determinism_std, mfcc_mean_pattern_duration_std,
mfcc_longest_pattern_duration_std, mode, release_year_std, Market_VN, Market_KR
```

**Model 2 — Audio plus lyrics:** Model 1 plus exactly:

```text
word_count_std, repetition_ratio_std, avg_word_length_std, line_count_std,
chorus_repetition_count_std, punctuation_ratio_std
```

Canonical preprocessing remains in `get_split()`: drop rows when all lyrics features or `word_count` are null, fill missing `chorus_repetition_count` with zero, apply `log1p(word_count)`, lower-winsorize `release_year` at the Train 1st percentile, and fit continuous-feature scaling on Train only. Other required nulls cause an error. The runner adds no preprocessing.

## Results

| Model | Test N | R² | Adjusted R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 (audio + controls) | 410 | 0.04699 | 0.00312 | 38.04605 | 29.73866 |
| Model 2 (audio + lyrics + controls) | 410 | 0.03649 | -0.02357 | 38.25501 | 30.05689 |

The cluster-robust Wald test jointly tests the six Model 2 lyrics coefficients: **χ²(6) = 13.5932, p = 0.03452599**. At α = 0.05, this run rejects the joint null. The p-value is the computed result, not a target imposed by the runner.

Per-market metrics are in `member3_model_test_metrics.csv`; coefficients are in `member3_model_coefficients.csv`.

## Validation

- Focused tests: `python -m unittest tests.test_lyrics_model2 -v` — 12 passed.
- Full suite: `python -m unittest discover -s tests -v` — 23 passed.
- The runner confirms the local cache source, the 2,098-row input, and the Train/Test ID disjointness in `member3_model_run_summary.json`.

## Canonical result files

All model artifacts are under `reports/report4/member3_canonical/`:

- `member3_model_comparison.csv`
- `member3_model_test_metrics.csv`
- `member3_model_coefficients.csv`
- `member3_lyrics_joint_wald.json`
- `member3_model_run_summary.json`

These canonical outputs serve as the authoritative Model 2 results for Checkpoint 2.
