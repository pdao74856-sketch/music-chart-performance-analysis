# Member 3 Checkpoint Status — CP1 and CP2

**Run date:** 2026-11-06 (Vietnam time)
**Branch:** `feature/member3-lyrics-model2`

## Status

- **CP1 — Train-only lyrics market medians: Complete.** The existing untracked `data/processed/lyrics_market_medians.json` was preserved. Its six feature medians and valid counts recalculate exactly from official Train rows.
- **CP2 — Lyrics processor and Model 2: Complete for the available integrated dataset.** The processor delegates to the canonical extractor. The modeling runner reuses the six lyrics columns already in the integrated dataset, imputes with frozen Train-only market medians, and compares both models on identical official-split observations.

## Data and split provenance

The source used was the local fallback `data/processed/modeling_dataset.csv`, as configured in `src/utils/supabase_client.py`. No live Supabase query was made or claimed. Official split IDs came from `models/train_track_ids.json` and `models/test_track_ids.json`; they contain 1,527 Train and 382 Test track IDs, with no ID overlap. The source has 1,912 unique tracks; three tracks (four market rows) are not present in either frozen ID list and were excluded from this run. Of listed Test IDs, the source contains 560 observations; four listed Test IDs have no source rows.

| Split | Track IDs in artifact | Observed rows | KR | US | VN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,527 | 2,296 | 814 | 758 | 724 |
| Test | 382 | 560 | 203 | 197 | 160 |

Both models use all 2,296 Train rows and the same 560 Test rows. Test rows are not used to fit medians or feature scaling.

## CP1 median validation

The artifact records `data/processed/modeling_dataset.csv` and `models/train_track_ids.json` as provenance. Medians were recalculated over Train rows grouped by market, excluding missing values feature by feature. All six expected fields exist for VN, US, and KR. The recalculated market blocks match the existing artifact exactly, including `n_rows` and valid counts. `chorus_repetition_count` is nonmissing for 686/724 VN, 681/758 US, and 767/814 KR Train rows. All other listed features are present in every Train row. There are no non-finite medians. The JSON was not rewritten.

## Lyrics processing

`src/features/lyrics_processor.py` wraps `src/lyrics/extract_lyrics_features.py` and exposes exactly `word_count`, `repetition_ratio`, `avg_word_length`, `line_count`, `chorus_repetition_count`, and `punctuation_ratio`, keyed by `spotify_track_id`. It preserves language-aware tokenization and Vietnamese Unicode diacritics from the existing extractor; empty/missing files produce null features and a distinct status, unreadable files produce `read_error`, and nonempty text yielding zero recognized tokens is `failed` with reason `no_tokens_recognized_after_tokenization`. Metadata processing rejects duplicate IDs and missing, blank, whitespace-only, or non-string track IDs before converting them to strings. Tokenization follows `src/lyrics/language_rules.yaml`: English/Spanish Unicode word tokens, Vietnamese whitespace syllable tokens, and Korean whitespace-delimited eojeol. `repetition_ratio = (tokens - unique tokens) / tokens`; `punctuation_ratio = Unicode punctuation code points / trimmed text characters`; `line_count` counts nonblank lines. Chorus count uses explicit section tags or repeated multi-line blocks; it is a repetition proxy, not semantic chorus recognition, and may be null when no block is identifiable.

The modeling dataset already contains the integrated lyrics features, so this run reused those columns rather than re-reading and re-extracting all raw files. Missing features were imputed from the saved Train-only market medians. Pre-imputation missingness is recorded in `member3_model2_missingness.csv`; after imputation there are zero missing Model 2 feature values. Missing collection, empty-file and extraction-error statuses remain in the source dataset metadata; they are not inferred from the numeric features.

## Model specification and results

Model 1 uses the 14 existing audio features plus `audio_mode`, `release_year`, and market indicators. Model 2 adds the six specified lyrics variables. Continuous variables are standardized with a scaler fit on Train only; `word_count` receives the existing `log1p` transform. Both models use OLS with cluster-robust covariance by `spotify_track_id`, through `src/utils/ols_cluster_fit.py`.

| Model | Test N | R² | Adjusted R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1 (audio + controls) | 560 | 0.02508 | -0.00736 | 41.07289 | 32.98239 |
| Model 2 (audio + lyrics + controls) | 560 | 0.02154 | -0.02236 | 41.14741 | 33.14744 |

The six lyrics coefficients were jointly tested with a **cluster-robust Wald chi-square** test (not a classical partial F-test): χ²(6) = 9.3630, p = 0.15417, α = 0.05. We fail to reject the null that all six lyrics coefficients are zero in this specification. On this Test sample, adding lyrics does not improve these metrics; this is an association/model-comparison result, not a causal claim.

Per-market test metrics and cluster-bootstrap intervals are in `member3_model_test_metrics.csv`; coefficient estimates are in `member3_model_coefficients.csv`. The shared input helper's metric bootstrap uses the Test set for evaluation only.

## Tests and limitations

Executed successfully: `python -m unittest discover -s tests -v` — all 17 repository tests passed. Coverage includes English/Vietnamese/Korean/empty/missing/zero-token processor inputs, six-field output, invalid metadata IDs, official split disjointness, exact Train median validation, market imputation, and Train-only scaling, alongside the pre-existing extraction pipeline cases.

The analysis is bounded by the local CSV snapshot and frozen split coverage: four listed Test IDs have no rows to evaluate, and three source tracks are outside the official ID artifacts. The processor is provided for stable-ID raw file processing; this model run reuses the already integrated feature columns to avoid redundant extraction. Model 2 improves no reported test metric in this run.

## Files

Created/modified implementation and tests:

- `src/features/lyrics_processor.py`
- `src/modeling/train_model2_lyrics.py`
- `tests/test_lyrics_model2.py`
- `reports/report4/member3_checkpoint_status.md`

Generated outputs:

- `reports/report4/member3_model_comparison.csv`
- `reports/report4/member3_model_test_metrics.csv`
- `reports/report4/member3_model_coefficients.csv`
- `reports/report4/member3_model2_missingness.csv`
- `reports/report4/member3_lyrics_joint_wald.json`
- `reports/report4/member3_model_run_summary.json`

The validated pre-existing untracked `data/processed/lyrics_market_medians.json` remains unchanged.
