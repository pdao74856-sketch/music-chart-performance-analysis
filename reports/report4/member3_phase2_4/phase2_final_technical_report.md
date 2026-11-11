# Phase 2 Final Statistical Validation and Handover

**Project:** ADY201m — What Makes a Hit Song? / **Role:** Member 3 — Lyrics & Text NLP Lead
**Scope:** Phase 2.4 statistical validation only. This document is a technical handover, not the Phase 3 narrative report or presentation material.

## 1. Objective and scope

This handover independently recalculates the approved Phase 2.2 primary and sensitivity analyses, reconciles the historical Report 4 audio-mode statistic, and packages machine-readable outputs and a run log. No models were retrained and no source data, teammate-owned code, CP3 Simulator code, or historical artifacts were changed.

The primary population is restricted to tracks appearing in exactly one of VN, US, or KR. Results describe this exclusive-track sample only. They may be affected by selection into the exclusive subset and should not be generalized automatically to tracks shared across markets or to all music in those markets.

## 2. Data sources and provenance

The reproducibility script uses:

- `data/processed/modeling_dataset_clean_cache.csv` — canonical 2,098 market observations and 1,912 Spotify track IDs; supplies the validated snapshot and audio mode.
- `reports/report3/lyrics_features_extracted.csv` — one row per track, including all six NLP features and `lyrics_status`.
- `data/raw/lyrics_collection_results.json` — track-to-market provenance. Market-level rows are reconstructed from distinct `(spotify_track_id, market)` pairs.
- `reports/report4/chart_data/member2/fig3_tonal_harmony_market.csv` — historical Report 4 aggregate used for audio-mode reconciliation.
- `reports/report4/chart_data/member2/audio_features_eda_source.csv` — historical row-level source used to check the aggregate table.

The script recomputes the canonical logical fingerprint using the existing loader's column selection, sorting, float rounding to six decimals, CSV serialization, and SHA-256 procedure. It matched **`bb0aea6d42049da728c8339bbf92388991825d9128cb85122df081e71a0a6e81`**. It also verifies 2,098 rows, 1,912 unique IDs, 2,098 unique track-market pairs, exact agreement of provenance pairs with cache pairs, and 1,743 exclusive tracks. NLP joins use `spotify_track_id`; market expansion and cache validation use `(spotify_track_id, market)`. Joins are validated for cardinality and unmatched pairs.

The script cross-checks extracted `word_count`, `repetition_ratio`, `avg_word_length`, and `chorus_repetition_count` against the cache on matching pairs with a `1e-12` floating tolerance. `unique_word_count` and `lexical_diversity` are supplied by the extracted feature table and joined by track ID, since those columns are not present in the modeling cache.

## 3. Approved methodology and sample construction

### Primary NLP analyses

Six Kruskal–Wallis tests compare VN, US, and KR in the market-exclusive sample, using complete cases separately for each feature. Holm correction is applied across exactly these six p-values at α=0.05. Epsilon-squared is calculated as `(H - k + 1) / (N - k)`, with `k=3` groups. Chorus complete cases are primary; missing chorus counts are not zero-filled in the primary test.

`lexical_diversity = unique_word_count / word_count` and `repetition_ratio = 1 - lexical_diversity`. Their identical test results are algebraically dependent and do not constitute independent evidence.

### Primary audio-mode analysis

Pearson's Chi-Square test uses a 3×2 market-by-mode table on the same exclusive subset. Each track contributes once. Cramér's V is reported as an effect-size summary.

### Sensitivity analyses

- Rank-based GEE uses average ranks divided by `N + 1`, market indicators with US as reference, Gaussian family, independence working correlation, robust sandwich covariance, and track clusters.
- Audio-mode logistic GEE codes major as 1 and minor as 0, with the same market reference, working correlation, covariance, and clustering approach.
- Chorus zero-fill sensitivity first excludes the three tracks with `lyrics_status=not_found`, then treats the remaining missing chorus counts as zero. This is an assumption-based sensitivity analysis, not primary evidence.
- Chorus missingness uses a binary `isna()` response in cluster-aware logistic GEE. The all-market Pearson test is retained as a diagnostic only because repeated tracks violate row independence.

GEE and Kruskal–Wallis analyses have different estimands: GEE estimates market associations on market rows while accounting for track clustering; Kruskal–Wallis compares distributions among tracks exclusive to one market.

## 4. Primary NLP results

| Feature | VN N | US N | KR N | H | df | Raw p | Holm p | Epsilon-squared | Holm significant? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| word_count | 487 | 721 | 532 | 269.8847 | 2 | 2.485e-59 | 1.242e-58 | 0.1542 | Yes |
| unique_word_count | 487 | 721 | 532 | 135.7840 | 2 | 3.273e-30 | 6.545e-30 | 0.0770 | Yes |
| lexical_diversity | 487 | 721 | 532 | 147.7139 | 2 | 8.401e-33 | 3.360e-32 | 0.0839 | Yes* |
| repetition_ratio | 487 | 721 | 532 | 147.7139 | 2 | 8.401e-33 | 3.360e-32 | 0.0839 | Yes* |
| avg_word_length | 487 | 721 | 532 | 557.0660 | 2 | 1.083e-121 | 6.498e-121 | 0.3196 | Yes |
| chorus_repetition_count | 460 | 629 | 512 | 2.3681 | 2 | 0.3060 | 0.3060 | 0.0002 | No |

*The two ratio features are deterministic transforms of one another and must not be described as independent evidence. The first five rows meet the statistical significance threshold after Holm correction, but statistical significance alone does not establish a large practical difference. Epsilon-squared is largest for average word length (about 0.320) and word count (about 0.154); chorus repetition has a negligible estimated effect in this complete-case sample.

## 5. Primary audio-mode results

Observed counts and within-market percentages:

| Market | Major | Minor | Total | Major % | Minor % |
|---|---:|---:|---:|---:|---:|
| VN | 336 | 152 | 488 | 68.85% | 31.15% |
| US | 396 | 325 | 721 | 54.92% | 45.08% |
| KR | 311 | 223 | 534 | 58.24% | 41.76% |

Expected major/minor counts are VN 292.016/195.984, US 431.442/289.558, and KR 319.542/214.458. Pearson **χ²(2)=24.3142, p=5.251×10⁻⁶**, with **Cramér’s V=0.1181**. Expected counts are all ample. The association is statistically detectable, while Cramér’s V indicates a modest association; it is not evidence that market causes mode or popularity.

## 6. Sensitivity analyses

### Chorus zero-fill

After dropping the three missing-lyrics tracks, zero-filled counts are VN 27, US 92, and KR 20 (139 total), leaving N=1,740: VN 487, US 721, KR 532. The sensitivity result is **H=7.7039, df=2, p=0.02124, epsilon-squared=0.00328**. This differs from the primary complete-case result (p=0.3060). The assumption that an unidentifiable chorus count means zero is not established, so complete cases remain the primary analysis. The sensitivity effect size is very small despite the p-value.

### All-market rank-based GEE for NLP

| Feature | Rows | Track clusters | Wald χ² | df | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| word_count | 2,095 | 1,909 | 290.0772 | 2 | 1.025e-63 | 5.123e-63 |
| unique_word_count | 2,095 | 1,909 | 128.0916 | 2 | 1.532e-28 | 6.128e-28 |
| lexical_diversity | 2,095 | 1,909 | 108.2418 | 2 | 3.130e-24 | 9.391e-24 |
| repetition_ratio | 2,095 | 1,909 | 108.2418 | 2 | 3.130e-24 | 9.391e-24 |
| avg_word_length | 2,095 | 1,909 | 772.3520 | 2 | 1.932e-168 | 1.159e-167 |
| chorus_repetition_count | 1,943 | 1,764 | 3.3309 | 2 | 0.1891 | 0.1891 |

All models converged in 2 reported iterations, produced finite robust market-contrast covariance matrices of rank 2, and emitted no warnings. The pattern of significance agrees with the exclusive-track primary results, but the estimands differ. The repeated lexical/repetition results remain dependent for the same algebraic reason stated above.

### Audio-mode logistic GEE

With 2,098 rows and 1,912 track clusters, the omnibus Wald test is **χ²(2)=21.7232, p=1.918×10⁻⁵**. It converged without warnings. Relative to US, the odds ratios for major mode are **1.646 for VN** and **1.124 for KR**. This is a cluster-aware sensitivity analysis on market rows, not the primary Pearson test.

### Chorus missingness

The extracted table has 148 unique tracks missing chorus values; provenance expansion gives 155 missing market rows. The 148 tracks include three `not_found` lyrics tracks and 145 tracks with lyrics but no reliably identified chorus repetition. Track-exclusive data contain 142 missing observations (VN 28, US 92, KR 22), including the three missing-lyrics tracks. The all-market count includes those three rows; after excluding them, preprocessing fills 152 remaining all-market chorus nulls with zero. These units are not additive: six missing tracks repeat across markets and contribute seven additional market rows.

On the exclusive subset, Pearson missingness association is **χ²(2)=35.8621, p=1.632×10⁻⁸**; each track contributes once. The all-market naive Pearson diagnostic is **χ²(2)=39.6862, p=2.411×10⁻⁹**, but repeated rows violate independence, so this p-value is not valid as an independent-observation test. Cluster-aware logistic GEE on 2,098 rows/1,912 clusters gives **Wald χ²(2)=36.0346, p=1.497×10⁻⁸**, with VN vs US OR=0.4241 and KR vs US OR=0.2996; it converged without warnings. These results indicate market association in the observed data but do not identify the missingness mechanism (MCAR/MAR/MNAR) or establish causality.

## 7. Historical audio-mode reconciliation

The cited Report 4 aggregation has 72 market×mode×key rows totaling 2,860 observations. Its reconstructed market-by-mode table is VN 567/318, US 539/416, KR 598/422 (major/minor). Pearson recalculation gives **χ²(2)=11.6962, p=0.002885**. This matches a recalculation from the cited 2,860-row audio feature source, but does not reproduce Report 4’s written `p=0.42`.

The v4.1 reference `χ²≈0.51, df=2, p≈0.774` is internally numerically coherent, but its source table, snapshot, and method have not been identified. Neither historical reference is treated as verified. The historical repeated-market-row result also does not use the same observational unit as the approved exclusive-track primary result.

## 8. Effect sizes, limitations, and threats to validity

- The exclusive subset removes within-track market duplication but may introduce selection bias and limits generalization to tracks occurring in exactly one market.
- P-values describe evidence against the test null under the stated sampling and model assumptions; they do not measure practical importance or causality.
- Ratio features are algebraically dependent; they must not be counted as independent supporting findings.
- Zero-filled chorus analysis depends on an unsupported missing-equals-zero assumption. Complete-case chorus is primary.
- Missingness rates differ across markets, but the analyses do not establish why data are missing or whether missingness is random.
- Naive all-market Pearson tests are diagnostic where tracks repeat; cluster-aware GEE is the all-market inferential sensitivity method.
- GEE and Kruskal–Wallis analyses address different observational populations and estimands.
- Historical Report 4 and v4.1 discrepancies remain unresolved as to original calculation provenance.

## 9. Reproducibility and environment

From the repository root, run the script with a **new, non-existing** output directory (the script refuses to overwrite existing outputs):

```powershell
python reports/report4/member3_phase2_4/reproduce_phase2_4.py --output-dir reports/report4/member3_phase2_4/reproduction_run_YYYYMMDD_HHMMSS
```

The successful final handover run is in `reports/report4/member3_phase2_4/results_final/`. It contains nine CSV result/audit tables and `run_log.json`, which records input SHA-256 hashes, canonical logical fingerprint, branch/commit/status at run time, package versions, outputs, limitations, and individual PASS/FAIL checks. The script exits non-zero for a fingerprint/count/schema/checkpoint failure and never overwrites an existing output directory.

Environment used: **Python 3.14.6, pandas 3.0.3, NumPy 2.4.6, SciPy 1.17.1, statsmodels 0.15.0**. The project requirements specify SciPy 1.18.1, so the runtime SciPy version differs from that requirement. No package was installed or upgraded. Reproduction passed the approved numerical checkpoints in the actual environment; this version difference remains an environment limitation.

## 10. Final conclusions and handover to Phase 3

In the market-exclusive sample, five NLP features differ statistically across markets after Holm correction; chorus repetition does not in the primary complete-case analysis. The audio-mode distribution is associated with market in the primary Pearson test, with modest Cramér’s V. Sensitivity analyses support a similar direction of evidence for most NLP features, while retaining their distinct estimands and assumptions. The chorus zero-fill result is not primary. Missingness is market-associated in these data, but its mechanism is unknown. The historical Report 4 `p=0.42` cannot be reproduced from its cited source, and the v4.1 target remains unverified.

For the next phase, carry forward the primary exclusive-track results, their sample sizes and effect sizes, the algebraic dependence of lexical diversity and repetition ratio, the chorus missingness/zero-fill limitation, and the unresolved historical audio-mode discrepancy. Keep sensitivity and diagnostic findings labeled as such. Do not convert these observational associations into causal claims.
