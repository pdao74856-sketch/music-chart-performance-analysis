# Member 3 Canonical Pipeline Checkpoint

**Run date:** 2026-11-06 (Asia/Bangkok)
**Pipeline:** `src.features.prepare_features.get_split()`
**Status:** Migrated results generated in `reports/report4/member3_canonical/`.

## Relationship to the historical checkpoint

The prior Member 3 run and its outputs in the parent `reports/report4/` directory are preserved unchanged. That historical workflow used frozen Train/Test ID artifacts, Train-only market-median lyric imputation, and its own scaler. This checkpoint records a separate run using Member 1's canonical split and preprocessing. The two sets of results should not be compared as though they came from identical preprocessing.

## Data returned by the canonical split

The runner called `get_split()` and used its returned matrices, targets, and track IDs as the sole source for fitting and evaluation. In this run the loader log reported the local fallback `data/processed/modeling_dataset.csv` (`cache`):

| Count | Observations |
| --- | ---: |
| Rows entering canonical cleaning | 2,856 |
| Rows retained by `get_split()` | 2,856 |
| Train observations | 2,296 |
| Test observations | 560 |
| Unique Train track IDs | 1,527 |
| Unique Test track IDs | 382 |

No rows were excluded by the canonical lyrics drop rule in this run. `get_split()` exposes exclusions in its metadata when applicable. The `SplitResult` does not expose the loader source or a complete audit of upstream filtering; this run's local-fallback provenance comes from the loader log.

The historical source snapshot contains **2,098 rows**, so it differs from the **2,856-observation** local CSV fallback used here. The historical 2,098-row snapshot is not currently available in this workspace, and canonical results have **not** been verified on it.

## Models

Both models use the same canonical Train and Test observations. No interaction columns are included.

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

The canonical code drops rows when all lyrics features are null or `word_count` is null, fills missing `chorus_repetition_count` with zero, applies `log1p(word_count)`, lower-winsorizes `release_year` at the Train 1st percentile, then fits continuous-feature scaling on Train only. Other required nulls cause an error. No extra imputation, scaling, or transformation is done by this runner.

## Results

| Model | Test N | R² | Adjusted R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 (audio + controls) | 560 | 0.03208 | -0.00012 | 40.92506 | 32.80475 |
| Model 2 (audio + lyrics + controls) | 560 | 0.02852 | -0.01506 | 41.00031 | 32.94139 |

The cluster-robust Wald test of exactly the six Model 2 lyrics coefficients is **χ²(6) = 9.9845, p = 0.1253**. At α = 0.05, this run fails to reject the joint null. This is the result of the current canonical local-fallback run; it is not an attempt to reproduce the historical Wald statistic.

Per-market metrics and bootstrap confidence intervals are in `member3_model_test_metrics.csv`; coefficients are in `member3_model_coefficients.csv`.

## Canonical result files

All files are under `reports/report4/member3_canonical/`:

- `member3_model_comparison.csv`
- `member3_model_test_metrics.csv`
- `member3_model_coefficients.csv`
- `member3_lyrics_joint_wald.json`
- `member3_model_run_summary.json`

These files are separate from and do not replace the historical checkpoint artifacts.
