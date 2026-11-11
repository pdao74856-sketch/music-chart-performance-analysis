# Member 3 (TV3) Phase 3 Academic Report — Cross-Market Lyrical NLP, Audio-Mode Distribution, and OLS Regression Integration

**Course:** ADY201m — Applied Data Science Project  
**Project Title:** *What Makes a Hit Song? Comparing Factors Affecting Music Popularity Across Vietnam, the United States, and South Korea (2021–2026)*  
**Role:** Member 3 (TV3 — Lyrics & Text NLP Lead | Audio + Lyrics Regression Model)  
**Branch:** `feature/member3-cp3-simulator`

---

## Executive Summary

This report presents the finalized statistical analysis of natural language processing (NLP) lyrics features, categorical audio-mode distributions, multicollinearity screening, feature transformations, and Ordinary Least Squares (OLS) regression integration across three music markets: **Vietnam (VN)**, **the United States (US)**, and **South Korea (KR)**.

From the canonical modeling dataset of **$N = 2,098$ track-market observations** representing **$1,912$ unique Spotify tracks**, **$169$ tracks** appear in two or more markets. To prevent cross-market track overlap from violating the independence assumption of classical non-parametric tests, the **primary distributional analysis** restricts the sample to **$N = 1,743$ market-exclusive tracks** (`VN`: $488$, `US`: $721$, `KR`: $534$). Across the six candidate lyrics features tested via Kruskal–Wallis with Holm–Bonferroni correction ($\alpha = 0.0500$), five lexical features (`avg_word_length`, `word_count`, `lexical_diversity`, `repetition_ratio`, and `unique_word_count`) exhibit statistically significant cross-market differences ($p_{\text{Holm}} \le 6.55 \times 10^{-30}$), whereas `chorus_repetition_count` does not ($H = 2.3681, \text{df} = 2, p_{\text{Holm}} = 0.3060, \varepsilon^2 = 0.0002$). Effect sizes are largest for `avg_word_length` ($\varepsilon^2 = 0.3196$) and `word_count` ($\varepsilon^2 = 0.1542$), reflecting both genuine lyrical conventions and fundamental morphological/tokenization differences across Vietnamese (whitespace-delimited syllables), English (word tokens), and Korean (whitespace-delimited agglutinative *eojeols* mixed with English).

For regression modeling on the canonical Training split (**$N = 1,685$ observations across $1,527$ track clusters**), Variance Inflation Factor (VIF) screening resolves two distinct forms of collinearity: (1) the exact linear algebraic identity $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$ ($\text{VIF} = 1.0008 \times 10^{15}$), addressed by dropping `lexical_diversity_std`, and (2) severe empirical collinearity between `word_count_std` ($\text{VIF} = 16.4196$) and `unique_word_count_std` ($\text{VIF} = 14.8026$), addressed by dropping `unique_word_count_std`. Introducing `line_count_std` and `punctuation_ratio_std` yields six lyrical OLS predictors whose VIFs in the 24-predictor main-effects matrix all lie safely below the project threshold ($\text{max VIF} = 3.7737 < 5.0$). Applying the natural logarithmic transformation $\ln(1 + \text{word\_count})$ reduces training-set skewness from $+1.5765$ to $-0.7931$, compresses excess kurtosis from $+6.4205$ to $+2.6455$, and lowers the maximum hat-matrix leverage ($h_{ii}$) among extreme rap/hip-hop tracks ($\ge 1,000$ words) from $0.0764$ to $0.0502$.

In the primary exclusive-track Chi-square analysis ($N = 1,743$), categorical `audio_mode` (`Major` vs. `Minor`) differs significantly across markets ($\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}$), driven by a higher prevalence of Major-key tracks in Vietnam ($68.85\%$) than in South Korea ($58.24\%$) and the United States ($54.92\%$), though with a small effect size ($\text{Cramér's V} = 0.1181$). In the cluster-robust OLS hierarchy (`ChartScore = 101 - lifetime_average_rank`), the six-feature lyrics block is jointly statistically significant in **Model 2 (`audio + lyrics + controls`, 24 predictors)** ($\text{Wald }\chi^2(6) = 13.5932, p = 0.0345$) and in the nested **Audio + Lyrics specification without controls (21 predictors)** ($\text{Wald }\chi^2(6) = 15.5722, p = 0.0162$), even though no single lyrical predictor reaches $p < 0.05$ individually after controlling for 15 audio features, `release_year_std`, and market fixed effects. All findings represent observational associations within charted tracks and do not establish causal effects on commercial popularity.

---

## 1. Research Context & Scope of Member 3

Within the ADY201m research project comparing music popularity across Vietnam (`VN`), the United States (`US`), and South Korea (`KR`), Member 3 (TV3) is responsible for five core analytical components:

1. **Cross-Market Lyrical NLP Distributional Analysis:** Testing whether the distributions of six extracted lyrics features (`word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, and `chorus_repetition_count`) differ across `VN`, `US`, and `KR` using non-parametric Kruskal–Wallis tests on market-exclusive tracks ($N = 1,743$) and cluster-robust rank-based Generalized Estimating Equations (GEE) sensitivity models on all market observations ($N = 2,095$).
2. **Multicollinearity (VIF) Screening & Feature Selection for OLS:** Diagnosing exact and near-linear dependencies among candidate lyrical variables on the Training split ($N = 1,685$) and establishing the final six-predictor lyrical block (`word_count_std`, `repetition_ratio_std`, `avg_word_length_std`, `line_count_std`, `chorus_repetition_count_std`, and `punctuation_ratio_std`) for regression modeling.
3. **Distributional Transformation & Leverage Diagnostics for `word_count`:** Evaluating the effect of the $\ln(1 + \text{word\_count})$ transformation on skewness, kurtosis, normality tests, hat-matrix leverage ($h_{ii}$), and Cook's distance ($D_i$).
4. **Categorical `audio_mode` Cross-Market Testing & Multi-Sample Reconciliation:** Conducting the primary Pearson Chi-square test of independence between `market` and `audio_mode` (`Major` vs. `Minor`) on market-exclusive tracks ($N = 1,743$), cluster-robust logistic GEE sensitivity analysis on all market observations ($N = 2,098$), and reconciling historical/draft statistics.
5. **OLS Regression Integration (Model 1, Model 2, and Model 3):** Evaluating how `mode` (`Major = 1.0`, `Minor = 0.0`) and the six-feature lyrical block enter the cluster-robust OLS regression models predicting `chart_score`, why `market × mode` interactions are omitted from the locked Model 3 specification, and why joint Wald significance differs from individual coefficient $p$-values.

*Formatting Convention:* Across all tables in this report, integer counts (`N`, `n`, `clusters`, `df`, `iterations`) are formatted as exact integers with comma separators; test statistics (`H`, $\chi^2$, $\text{Wald }\chi^2$), effect sizes ($\varepsilon^2$, $\text{Cramér's V}$), regression coefficients ($\hat{\beta}$), standard errors ($SE$), odds ratios ($\text{OR}$), VIF values, and $p$-values $\ge 0.0001$ are rounded to **4 decimal places**; $p$-values $< 0.0001$ are reported in scientific notation with **2 decimal places** in the mantissa (e.g., `2.48e-59`); expected cell counts and percentages (`%`) are rounded to **2 decimal places**.

---

## 2. Data Provenance, Units of Analysis & Feature Operationalization

### 2.1. Data Provenance & Explicit Units of Observation

The extracted track-level lyrics table (`reports/report3/lyrics_features_extracted.csv`) contains exactly **$1,912$ unique tracks** (`spotify_track_id`). When joined with the quarterly sampling provenance record (`data/raw/lyrics_collection_results.json`), these $1,912$ tracks expand to **$2,098$ unique `(spotify_track_id, market)` observations** in the canonical modeling cache (`data/processed/modeling_dataset_clean_cache.csv`). Because **$169$ tracks** charted in two or more markets (`KR & VN`: $93$, `KR & US`: $49$, `KR & US & VN`: $17$, `US & VN`: $10$), analyses across the project operate on five distinct, explicitly defined sample scopes:

- **Scope A — All-Market Track-Market Observations ($N = 2,098$ rows; $1,912$ unique track clusters):** `VN = 608`, `US = 797`, `KR = 693`. Used for market-level descriptive visualization and cluster-robust GEE sensitivity models (`N = 2,095` valid lyrics rows across `1,909` clusters after excluding 3 instrumental/missing-lyrics rows).
- **Scope B — Unique Track Table ($N = 1,912$ unique `spotify_track_id` rows):** Used for track-level extraction provenance (`LRCLIB`: $1,836$; `Manual`: $44$; `Genius`: $29$; `not_found`: $3$ instrumental/unavailable tracks, leaving $1,909$ tracks with valid lyrics).
- **Scope C — Market-Exclusive Tracks ($N = 1,743$ rows; $1,743$ unique tracks):** `VN = 488`, `US = 721`, `KR = 534`. Excludes the $169$ multi-market tracks so that each track contributes at most one observation. Used for all **primary non-parametric Kruskal–Wallis and Pearson Chi-square inference** ($N = 1,740$ complete cases for lexical features; $N = 1,601$ complete cases for `chorus_repetition_count`).
- **Scope D — Canonical OLS Modeling Split ($N = 2,095$ valid rows $\to$ Train $N = 1,685$ rows / $1,527$ track clusters; Test $N = 410$ rows / $382$ track clusters):** Created by `src/features/prepare_features.py` (`get_split()`, `FROZEN_SEED = 43`) using `GroupShuffleSplit` grouped by `spotify_track_id`, guaranteeing zero track leakage between Train and Test.
- **Scope E — Historical Report 4 EDA Dataset ($N = 2,860$ rows):** `VN = 885`, `US = 955`, `KR = 1,020` from `reports/report4/chart_data/member2/fig3_tonal_harmony_market.csv` and `audio_features_eda_source.csv`, used strictly for historical audio-mode reconciliation.

**Table 1. Data Provenance Audit, Input SHA-256 Hashes, and Canonical Sample Sizes.**

| Record Type | Audited Component / Metric | Relative Repository Path / Scope | Verified Value / SHA-256 Fingerprint |
|---|---|---|---|
| Input File | Canonical modeling dataset (`cache`) | `data/processed/modeling_dataset_clean_cache.csv` | SHA-256: `16e8119c679373e45626d3282dcac673bae4579ba7f8b64330295d33a4f2837c` |
| Input File | Extracted track-level lyrics features (`features`) | `reports/report3/lyrics_features_extracted.csv` | SHA-256: `727a4b8fb4a0030b173ce094eb005bb80efed1a145ad7fbd1e35f2c882dbace7` |
| Input File | Lyrics collection & market provenance (`provenance`) | `data/raw/lyrics_collection_results.json` | SHA-256: `878c43c8baf81734dae644c77daf455061f4d6e0414c3aa148edb7cf381ceb7b` |
| Input File | Historical aggregated tonal harmony table (`historical_mode`) | `reports/report4/chart_data/member2/fig3_tonal_harmony_market.csv` | SHA-256: `68f24dbf5a9562446cae8fa90f212935aa16b0c395ea1eb334816b43637281a4` |
| Input File | Historical row-level audio EDA source (`historical_rows`) | `reports/report4/chart_data/member2/audio_features_eda_source.csv` | SHA-256: `6a88350e0c530514930e56f9dbd115f23e9491ecf8fb0afb649cc1d42c3727ee` |
| Audit Metric | Canonical logical fingerprint (`canonical_logical_fingerprint`) | `data/processed/modeling_dataset_clean_cache.csv` | `bb0aea6d42049da728c8339bbf92388991825d9128cb85122df081e71a0a6e81` |
| Audit Metric | Canonical track-market rows (`canonical_market_rows`) | All-market sample (VN: 608, US: 797, KR: 693) | `2,098` |
| Audit Metric | Canonical unique Spotify tracks (`canonical_unique_tracks`) | Track-level sample (`spotify_track_id`) | `1,912` |
| Audit Metric | Market-exclusive tracks (`exclusive_tracks`) | Primary inference sample (VN: 488, US: 721, KR: 534) | `1,743` |
| Audit Metric | Multi-market overlapping tracks (`multi_market_tracks`) | Excluded from primary independent tests (`1,912 - 1,743`) | `169` |
| Audit Metric | Canonical OLS Training split (`train_rows` / `train_tracks`) | `get_split()` (`seed = 43`, 80% grouped by track) | `1,685` rows / `1,527` track clusters |
| Audit Metric | Canonical OLS Test split (`test_rows` / `test_tracks`) | `get_split()` (`seed = 43`, 20% grouped by track) | `410` rows / `382` track clusters |
| Audit Metric | Duplicate `(spotify_track_id, market)` pairs | Provenance & canonical cache alignment check | `0` |

*Source: `reports/report4/member3_phase2_4/results_final_audit/09_data_provenance_audit.csv` and `reports/report4/member3_canonical/member3_model_run_summary.json`.*

---

### 2.2. Operationalization of Candidate Lyrics Features, OLS Predictors & Audio Mode

A critical methodological distinction must be maintained between the **six lyrics features tested in the Phase 2.4 Kruskal–Wallis distributional study** and the **six lyrical predictors retained in the OLS regression models** after VIF screening:

- **Tested in Primary Kruskal–Wallis (6 features):** `word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, and `chorus_repetition_count`.
- **Retained in OLS Models 2 and 3 (6 predictors):** `word_count_std`, `repetition_ratio_std`, `avg_word_length_std`, `line_count_std`, `chorus_repetition_count_std`, and `punctuation_ratio_std` (where `lexical_diversity` and `unique_word_count` are excluded due to multicollinearity, and structural features `line_count` and `punctuation_ratio` from `lyrics_features_extracted.csv` are included).

**Table 2. Operational Definitions, Physical Units, and Role of Lyrical and Audio-Mode Variables.**

| Feature Name | Measurement Unit (Raw Scale) | Operational Definition in Pipeline (`src/lyrics/extract_lyrics_features.py` & `prepare_features.py`) | Role in Kruskal–Wallis (Phase 2.4) | Role in OLS Regression (`Model 2` / `Model 3`) |
|---|---|---|---|---|
| `word_count` | Tokens / track | Total number of word tokens after structural tag removal (`[Verse]`, `[Chorus]`) and language-aware tokenization. | **Primary Tested** ($N = 1,740$) | **Retained** as $\text{Z-score}(\ln(1 + \text{word\_count}))$ (`word_count_std`) |
| `unique_word_count` | Unique tokens / track | Count of distinct case-normalized word tokens in the track's lyrics. | **Primary Tested** ($N = 1,740$) | **Excluded via VIF** ($\text{VIF} = 14.8026$; product identity with `word_count`) |
| `lexical_diversity` | Ratio $[0, 1]$ | Type–Token Ratio ($\text{TTR} = \text{unique\_word\_count} / \text{word\_count}$). | **Primary Tested** ($N = 1,740$) | **Excluded via VIF** ($\text{VIF} = 1.0008 \times 10^{15}$; exact identity $1 - \text{repetition\_ratio}$) |
| `repetition_ratio` | Ratio $[0, 1]$ | Proportion of repeated tokens: $(\text{word\_count} - \text{unique\_word\_count}) / \text{word\_count} = 1 - \text{lexical\_diversity}$. | **Primary Tested** ($N = 1,740$) | **Retained** as Z-score (`repetition_ratio_std`, $\text{VIF} = 1.7669$) |
| `avg_word_length` | Characters / token | Mean Unicode character length across all tokens in the track. | **Primary Tested** ($N = 1,740$) | **Retained** as Z-score (`avg_word_length_std`, $\text{VIF} = 1.2150$) |
| `chorus_repetition_count` | Additional chorus occurrences | Count of repeated chorus blocks identified by repeated multi-line block matching (`NULL` if no reliable chorus block is detected). | **Primary Tested** ($N = 1,601$ complete cases; `NULL` preserved) | **Retained** as Z-score after technical zero-fill (`NULL -> 0.0`, $\text{VIF} = 1.2983$) |
| `line_count` | Non-empty lines / track | Total number of non-empty lyrical lines after stripping section headers. | Descriptive EDA (`member3.md`) | **Retained** as Z-score (`line_count_std`, $\text{VIF} = 2.7631$) |
| `punctuation_ratio` | Punctuation chars / total chars | Ratio of expressive punctuation marks to total character count. | Descriptive EDA (`member3.md`) | **Retained** as Z-score (`punctuation_ratio_std`, $\text{VIF} = 1.3776$) |
| `audio_mode` (`mode`) | Binary indicator $\{0, 1\}$ | Categorical Librosa tonal modality (`Major` vs. `Minor`), encoded in OLS as unscaled binary `mode` (`Major = 1.0`, `Minor = 0.0` reference). | **Primary Audio Chi-Square** ($N = 1,743$) | **Retained** as unscaled binary main effect (`mode`, $\text{VIF} = 1.2014$ in Model 2) |

*Source: `src/lyrics/extract_lyrics_features.py`, `src/features/prepare_features.py`, and `reports/report3/lyrics_features_extracted.csv`.*

---

## 3. Cross-Market Lyrics Analysis (Kruskal–Wallis, Descriptives & Sensitivity)

### 3.1. Primary Kruskal–Wallis Analysis on Market-Exclusive Tracks ($N = 1,743$)

Because the six lyrics features exhibit pronounced skewness, heavy tails, or discrete count steps across markets, the primary cross-market comparison applies the non-parametric **Kruskal–Wallis $H$-test** on the **$N = 1,743$ market-exclusive tracks**. Each feature is evaluated on its complete-case observations (`VN = 487`, `US = 721`, `KR = 532`, total $N = 1,740$ for the five lexical features after excluding 3 instrumental/missing-lyrics tracks; `VN = 460`, `US = 629`, `KR = 512`, total $N = 1,601$ for `chorus_repetition_count`). Family-wise error rate across the six tests is controlled via the **Holm–Bonferroni step-down procedure** at $\alpha = 0.0500$, and omnibus effect magnitude is quantified by epsilon-squared:
$$\varepsilon^2 = \frac{H - k + 1}{N - k}, \quad k = 3.$$

**Table 3. Complete-Case Sample Sizes by Market in the Primary Market-Exclusive Analysis ($N = 1,743$ Exclusive Tracks).**

| Lyrics Feature (`feature`) | Measurement Unit | VN Complete Cases ($n_{\text{VN}}$) | US Complete Cases ($n_{\text{US}}$) | KR Complete Cases ($n_{\text{KR}}$) | Total Valid Exclusive Tracks ($N$) | Excluded Exclusive Tracks (Reason) |
|---|---|---:|---:|---:|---:|---|
| `word_count` | Tokens / track | 487 | 721 | 532 | 1,740 | 3 tracks (`lyrics_status = not_found`) |
| `unique_word_count` | Unique tokens / track | 487 | 721 | 532 | 1,740 | 3 tracks (`lyrics_status = not_found`) |
| `lexical_diversity` | Ratio $[0, 1]$ | 487 | 721 | 532 | 1,740 | 3 tracks (`lyrics_status = not_found`) |
| `repetition_ratio` | Ratio $[0, 1]$ | 487 | 721 | 532 | 1,740 | 3 tracks (`lyrics_status = not_found`) |
| `avg_word_length` | Characters / token | 487 | 721 | 532 | 1,740 | 3 tracks (`lyrics_status = not_found`) |
| `chorus_repetition_count` | Additional chorus blocks | 460 | 629 | 512 | 1,601 | 142 tracks (3 `not_found` + 139 undetected chorus `NULL`) |

*Source: `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv`.*

**Table 4. Primary Kruskal–Wallis Test Results, Holm–Bonferroni Adjusted $p$-Values, and $\varepsilon^2$ Effect Sizes on Market-Exclusive Tracks.**

| Lyrics Feature (`feature`) | Valid $N$ | Kruskal–Wallis $H$ | $\text{df}$ | Unadjusted $p$ (`p_raw`) | Holm-Adjusted $p$ (`p_holm`) | Effect Size ($\varepsilon^2$) | Significance at $\alpha = 0.0500$ | Substantive Magnitude & Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `avg_word_length` | 1,740 | 557.0660 | 2 | 1.08e-121 | 6.50e-121 | 0.3196 | Significant (`True`) | **Large effect** ($\approx 31.96\%$ rank variance); driven by English word tokens vs. Vietnamese syllables and Korean Hangul blocks |
| `word_count` | 1,740 | 269.8847 | 2 | 2.48e-59 | 1.24e-58 | 0.1542 | Significant (`True`) | **Moderate-to-large effect** ($\approx 15.42\%$ rank variance); highest token count in VN syllables, lowest in KR *eojeols* |
| `lexical_diversity`* | 1,740 | 147.7139 | 2 | 8.40e-33 | 3.36e-32 | 0.0839 | Significant (`True`) | **Moderate effect** ($\approx 8.39\%$ rank variance); highest surface TTR in KR agglutinative *eojeols* |
| `repetition_ratio`* | 1,740 | 147.7139 | 2 | 8.40e-33 | 3.36e-32 | 0.0839 | Significant (`True`) | **Moderate effect**; exact algebraic complement of `lexical_diversity` ($1 - \text{TTR}$, identical $H$ and $\varepsilon^2$) |
| `unique_word_count` | 1,740 | 135.7840 | 2 | 3.27e-30 | 6.55e-30 | 0.0770 | Significant (`True`) | **Moderate effect** ($\approx 7.70\%$ rank variance); VN and US exceed KR in distinct surface tokens |
| `chorus_repetition_count` | 1,601 | 2.3681 | 2 | 0.3060 | 0.3060 | 0.0002 | Not Significant (`False`) | **Negligible effect**; complete-case distribution is invariant across markets (median $= 1.0000$ in all three markets) |

*Source: `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv`.*  
*Note: The asterisk (*) denotes that `lexical_diversity` and `repetition_ratio` satisfy $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$; because one is a strictly decreasing linear function of the other, their rank orderings are exact reversals and their Kruskal–Wallis statistics are algebraically identical ($H = 147.7139$). They represent one statistical phenomenon, not two independent discoveries.*

---

### 3.2. Cross-Market Descriptive Statistics & Post-Hoc Pairwise Status

The locked Phase 2.4 verification pipeline (`reports/report4/member3_phase2_4/reproduce_phase2_4.py`) tests the omnibus 3-market hypothesis and **does not execute separate non-parametric pairwise post-hoc tests** (such as Dunn's test or pairwise Mann–Whitney $U$). Rather than making unsupported claims about unexecuted post-hoc tests, directional interpretation is grounded in two complementary sources of project evidence: (1) the exact market-level medians, IQRs, means, and standard deviations across both the market-exclusive sample ($N = 1,743$) and the all-market sample ($N = 2,098$) in **Table 5**, and (2) the explicit market contrast estimates (`VN vs. US` and `KR vs. US`) from the cluster-robust rank-based GEE sensitivity models in **Table 6**.

**Table 5. Cross-Market Descriptive Statistics (Median, IQR, Mean, SD) for the Six Lyrics Features Across Exclusive ($N = 1,743$) and All-Market ($N = 2,098$) Scopes.**

| Lyrics Feature (`feature`) | Sample Scope | VN: Median (IQR) | VN: Mean (SD) | US: Median (IQR) | US: Mean (SD) | KR: Median (IQR) | KR: Mean (SD) | Empirical Ordering of Medians |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `word_count` | Exclusive ($N = 1,740$) | 421.00 (217.50) | 456.41 (177.93) | 356.00 (211.00) | 395.44 (191.88) | 286.00 (158.75) | 290.40 (140.28) | **VN $>$ US $>$ KR** |
| `word_count` | All-Market ($N = 2,095$) | 396.00 (217.50) | 432.66 (174.67) | 358.00 (211.00) | 396.92 (189.08) | 298.00 (161.50) | 307.10 (142.82) | **VN $>$ US $>$ KR** |
| `unique_word_count` | Exclusive ($N = 1,740$) | 160.00 (92.00) | 173.08 (70.89) | 127.00 (74.00) | 147.12 (74.52) | 118.00 (59.00) | 127.62 (62.60) | **VN $>$ US $>$ KR** |
| `unique_word_count` | All-Market ($N = 2,095$) | 149.00 (88.00) | 162.41 (69.37) | 127.00 (73.00) | 146.17 (73.09) | 118.00 (56.00) | 127.53 (59.44) | **VN $>$ US $>$ KR** |
| `lexical_diversity` | Exclusive ($N = 1,740$) | 0.3779 (0.1121) | 0.3866 (0.0868) | 0.3746 (0.1102) | 0.3837 (0.0944) | 0.4524 (0.1861) | 0.4677 (0.1411) | **KR $>$ VN $\approx$ US** |
| `lexical_diversity` | All-Market ($N = 2,095$) | 0.3737 (0.1150) | 0.3833 (0.0901) | 0.3713 (0.1133) | 0.3797 (0.0938) | 0.4244 (0.1817) | 0.4437 (0.1388) | **KR $>$ VN $\approx$ US** |
| `repetition_ratio` | Exclusive ($N = 1,740$) | 0.6221 (0.1121) | 0.6134 (0.0868) | 0.6254 (0.1102) | 0.6163 (0.0944) | 0.5476 (0.1861) | 0.5323 (0.1411) | **US $\approx$ VN $>$ KR** |
| `repetition_ratio` | All-Market ($N = 2,095$) | 0.6263 (0.1150) | 0.6167 (0.0901) | 0.6287 (0.1133) | 0.6203 (0.0938) | 0.5756 (0.1817) | 0.5563 (0.1388) | **US $\approx$ VN $>$ KR** |
| `avg_word_length` | Exclusive ($N = 1,740$) | 3.3194 (0.2751) | 3.4000 (0.4069) | 3.8530 (0.3040) | 3.8920 (0.3468) | 3.2927 (0.9911) | 3.4739 (1.1659) | **US $>$ VN $\approx$ KR** (Mean: **US $>$ KR $>$ VN**) |
| `avg_word_length` | All-Market ($N = 2,095$) | 3.3498 (0.4259) | 3.4621 (0.4472) | 3.8462 (0.3109) | 3.8786 (0.3438) | 3.4737 (0.9212) | 3.5313 (1.0527) | **US $>$ KR $>$ VN** |
| `chorus_repetition_count` | Exclusive ($N = 1,601$) | 1.0000 (1.0000) | 1.6652 (0.9008) | 1.0000 (1.0000) | 1.6518 (0.9942) | 1.0000 (1.0000) | 1.6191 (0.9224) | **VN $=$ US $=$ KR** (Median $= 1.0000$) |
| `chorus_repetition_count` | All-Market ($N = 1,943$) | 1.0000 (1.0000) | 1.6574 (0.9216) | 1.0000 (1.0000) | 1.6581 (0.9820) | 1.0000 (1.0000) | 1.6171 (0.9318) | **VN $=$ US $=$ KR** (Median $= 1.0000$) |

*Source: Computed from `data/processed/modeling_dataset_clean_cache.csv`, `reports/report3/lyrics_features_extracted.csv`, `reports/report3/lyrics_statistics.csv`, `reports/report4/chart_data/member3/01_word_count_by_market.csv`, and `reports/report4/chart_data/member3/03_repetition_ratio_by_market.csv`.*

---

### 3.3. Descriptive Visualizations & Language/Tokenization Context

The three figures below illustrate the cross-market distributions on the **all-market track-market observations ($N = 2,098$ total rows; $N = 2,095$ valid lyrics rows)** audited in Phase 2.3. As noted in their captions, these plots include the $169$ multi-market tracks and serve as descriptive market-level context alongside the primary exclusive-track tests.

![Distribution of Word Count by Market — All-Market Descriptive Boxplot](reports/report4/figures/member3/01_boxplot_word_count_by_market.png)

*Figure 1. Distribution of total word tokens (`word_count`) across markets on all valid track-market observations (VN: $n = 607 / N = 608$, Median $= 396.0$, Mean $= 432.7$; US: $n = 797 / N = 797$, Median $= 358.0$, Mean $= 396.9$; KR: $n = 691 / N = 693$, Median $= 298.0$, Mean $= 307.1$; total $N = 2,095$ valid rows). All three distributions are right-skewed with upper-tail hip-hop/rap tracks exceeding $1,000$ to $1,985$ tokens.*  
*Source: Image `reports/report4/figures/member3/01_boxplot_word_count_by_market.png`; chart data `reports/report4/chart_data/member3/01_word_count_by_market.csv`.*

![Distribution of Repetition Ratio by Market — All-Market Descriptive Boxplot](reports/report4/figures/member3/03_boxplot_repetition_ratio_by_market.png)

*Figure 2. Distribution of lyrical repetition ratio (`repetition_ratio = 1 - lexical_diversity`) across markets on all valid track-market observations (VN: $n = 607$, Median $= 0.626$, Mean $= 0.617$; US: $n = 797$, Median $= 0.629$, Mean $= 0.620$; KR: $n = 691$, Median $= 0.576$, Mean $= 0.556$). Because Korean whitespace tokenization groups stems with grammatical particles into distinct surface *eojeols*, KR exhibits lower token-level repetition and higher surface lexical diversity than VN and US.*  
*Source: Image `reports/report4/figures/member3/03_boxplot_repetition_ratio_by_market.png`; chart data `reports/report4/chart_data/member3/03_repetition_ratio_by_market.csv`.*

![Distribution of Detected Lyrics Language by Market](reports/report4/figures/member3/09_stacked_language_by_market.png)

*Figure 3. 100% stacked bar chart of detected lyrics language (`language`) within each sampled chart market ($N = 2,098$ rows: VN $N = 608$, US $N = 797$, KR $N = 693$). In VN: Vietnamese (`vi`) $= 60.36\%$ ($n = 367$), English (`en`) $= 28.62\%$ ($n = 174$), Mixed (`mixed`) $= 6.41\%$ ($n = 39$), Korean (`ko`) $= 3.12\%$ ($n = 19$), Other/Unknown $= 1.48\%$ ($n = 9$). In US: English (`en`) $= 93.10\%$ ($n = 742$), Spanish (`es`) $= 5.27\%$ ($n = 42$), Mixed $= 1.63\%$ ($n = 13$). In KR: English (`en`) $= 48.92\%$ ($n = 339$), Korean (`ko`) $= 28.72\%$ ($n = 199$), Mixed (`mixed`) $= 17.17\%$ ($n = 119$), Spanish (`es`) $= 0.14\%$ ($n = 1$), Other/Unknown $= 5.05\%$ ($n = 35$).*  
*Source: Image `reports/report4/figures/member3/09_stacked_language_by_market.png`; chart data `reports/report4/chart_data/member3/09_language_distribution_by_market.csv`.*

---

### 3.4. Sensitivity Analyses: Cluster-Robust Rank-GEE, Chorus Zero-Fill & Missingness

To verify that excluding the $169$ multi-market tracks does not distort our conclusions, **Table 6** and **Table 7** report the all-market **Rank-based Gaussian GEE** sensitivity analysis ($N = 2,095$ rows across $1,909$ track clusters for lexical features; $N = 1,943$ rows across $1,764$ clusters for `chorus_repetition_count`), where the response is the fractional rank $\text{average rank} / (N + 1)$, `US` is the reference market, working correlation is `independence`, and standard errors use the track-clustered sandwich estimator.

**Table 6. All-Market Rank-Based GEE Sensitivity Analysis: Sample Sizes, Market Contrast Estimates, and Omnibus Wald Tests.**

| Lyrics Feature (`feature`) | Market Rows ($N$) | Track Clusters (`clusters`) | Contrast Estimate: VN vs. US | Contrast Estimate: KR vs. US | Omnibus Wald $\chi^2$ | $\text{df}$ | Unadjusted $p$ (`p_raw`) | Holm-Adjusted $p$ (`p_holm`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `word_count` | 2,095 | 1,909 | +0.0753 | -0.1520 | 290.0772 | 2 | 1.02e-63 | 5.12e-63 |
| `unique_word_count` | 2,095 | 1,909 | +0.0903 | -0.0707 | 128.0916 | 2 | 1.53e-28 | 6.13e-28 |
| `lexical_diversity` | 2,095 | 1,909 | +0.0103 | +0.1359 | 108.2418 | 2 | 3.13e-24 | 9.39e-24 |
| `repetition_ratio` | 2,095 | 1,909 | -0.0103 | -0.1359 | 108.2418 | 2 | 3.13e-24 | 9.39e-24 |
| `avg_word_length` | 2,095 | 1,909 | -0.3018 | -0.2633 | 772.3520 | 2 | 1.93e-168 | 1.16e-167 |
| `chorus_repetition_count` | 1,943 | 1,764 | -0.0012 | -0.0212 | 3.3309 | 2 | 0.1891 | 0.1891 |

*Source: `reports/report4/member3_phase2_4/results_final_audit/04_all_market_rank_gee_sensitivity.csv`.*  
*Note: Contrast coefficients measure the expected shift on the normalized fractional rank scale $[0, 1]$ relative to the US reference market. The source CSV does not export confidence intervals for individual rank-contrast coefficients. Rank-GEE targets the marginal all-market population-averaged estimand rather than the market-exclusive track estimand.*

**Table 7. All-Market Rank-Based GEE Sensitivity Analysis: Model Configuration and Convergence Diagnostics.**

| Lyrics Feature (`feature`) | Rank Transformation | Working Correlation | Covariance Estimator | Converged? | Fit Iterations | Finite Robust Covariance? | Market Contrast Covariance Rank |
|---|---|---|---|---|---:|---|---:|
| `word_count` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |
| `unique_word_count` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |
| `lexical_diversity` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |
| `repetition_ratio` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |
| `avg_word_length` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |
| `chorus_repetition_count` | `average rank / (N + 1)` | `independence` | `robust sandwich` | Yes (`True`) | 2 | Yes (`True`) | 2 |

*Source: `reports/report4/member3_phase2_4/results_final_audit/04_all_market_rank_gee_sensitivity.csv`.*

#### Chorus Zero-Fill Sensitivity & Missingness Audit (`03_chorus_zero_fill_sensitivity.csv`, `06_`, `07_`)
Whereas the five lexical features are missing only for the $3$ instrumental/not-found tracks, `chorus_repetition_count` is `NULL` for **$148$ unique tracks** ($155$ all-market rows; $142$ market-exclusive tracks, comprising $3$ `not_found` tracks and $139$ lyrical tracks where no repeated multi-line chorus block was algorithmically detected). Under a **Zero-Fill Sensitivity Scenario** (`03_chorus_zero_fill_sensitivity.csv`, Tables 8–9) that imputes `0` for those $139$ lyrical tracks (`VN`: $27$, `US`: $92$, `KR`: $20$; total $N = 1,740$), the Kruskal–Wallis statistic shifts from $H = 2.3681$ ($p = 0.3060$) to $H = 7.7039$ ($p = 0.0212$, $\varepsilon^2 = 0.0033$).

Why does zero-filling change the $p$-value? As proven by the missingness audit (`06_chorus_missingness_reconciliation.csv` and `07_missingness_association.csv`, Tables 10–13), missing chorus detection is strongly associated with market: in the exclusive sample ($N = 1,743$), `US` has a missingness rate of **$12.76\%$ ($92/721$)** compared with **$5.74\%$ ($28/488$)** in `VN` and **$4.12\%$ ($22/534$)** in `KR` (primary Pearson $\chi^2(2) = 35.8621, p = 1.63 \times 10^{-8}$; all-market cluster-robust logistic GEE $\text{Wald }\chi^2(2) = 36.0346, p = 1.50 \times 10^{-8}$, with $\text{OR}_{\text{VN vs US}} = 0.4241, p = 3.23 \times 10^{-5}$ and $\text{OR}_{\text{KR vs US}} = 0.2996, p = 4.64 \times 10^{-8}$). Because through-composed US rap/hip-hop tracks often lack a verbatim repeated chorus block, replacing `NULL` with `0` disproportionate pulls down the US distribution. Consequently, **complete-case analysis ($p = 0.3060$) is retained as the primary distributional result**, while zero-fill (`NULL -> 0.0`) is used strictly as an engineering imputation rule in OLS preprocessing (`prepare_features.py`) to avoid discarding $139$ valid songs from multivariable regression.

**Table 8. Chorus Zero-Fill Sensitivity Analysis: Sample Sizes and Zero-Filled Counts by Market ($N = 1,740$).**

| Market / Scope | Valid Lyrics Exclusive Tracks ($n$) | Complete-Case Chorus Tracks | Zero-Filled Chorus Tracks (`zero_filled_n`) | Zero-Filled Percentage (%) |
|---|---:|---:|---:|---:|
| VN | 487 | 460 | 27 | 5.54 |
| US | 721 | 629 | 92 | 12.76 |
| KR | 532 | 512 | 20 | 3.76 |
| **All Exclusive Lyrics (`ALL`)** | **1,740** | **1,601** | **139** | **7.99** |

*Source: `reports/report4/member3_phase2_4/results_final_audit/03_chorus_zero_fill_sensitivity.csv`.*

**Table 9. Comparison of Primary Complete-Case vs. Sensitivity Zero-Fill Kruskal–Wallis Results for `chorus_repetition_count`.**

| Analysis Specification | Sample Size ($N$) | Zero-Filled Count | Kruskal–Wallis $H$ | $\text{df}$ | Unadjusted $p$ (`p_raw`) | Effect Size ($\varepsilon^2$) | Methodological Status & Assumption |
|---|---:|---:|---:|---:|---:|---:|---|
| **Primary Complete-Case** | 1,601 | 0 | 2.3681 | 2 | 0.3060 | 0.0002 | **Primary Result:** `NULL` preserved as missing when chorus structure is undetected |
| **Sensitivity Zero-Fill** | 1,740 | 139 | 7.7039 | 2 | 0.0212 | 0.0033 | **Sensitivity Only:** Assumes undetected chorus (`NULL`) equals exactly `0` repetitions |

*Source: `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv` and `03_chorus_zero_fill_sensitivity.csv`.*

**Table 10. Reconciliation of `chorus_repetition_count` Missingness Across Observation Units.**

| Observation Unit Scope | Total Units ($N$) | Total Missing Chorus (`missing_chorus_n`) | VN Missing | US Missing | KR Missing | Includes 3 `not_found` Lyrics Tracks? | Technical Note on Counting Unit |
|---|---:|---:|---:|---:|---:|---|---|
| Unique Tracks (`unique track`) | 1,912 | 148 | 33* | 95* | 27* | Yes (`True`) | Market counts sum to $155$ because $6$ missing-chorus tracks appear in multiple markets |
| All-Market Rows (`all-market track-market row`) | 2,098 | 155 | 33 | 95 | 27 | Yes (`True`) | Includes 3 rows with `lyrics_status = not_found` (1 in VN, 2 in KR) |
| Market-Exclusive Tracks (`market-exclusive track`) | 1,743 | 142 | 28 | 92 | 22 | Yes (`True`) | Each track contributes exactly one independent row (142 = 3 `not_found` + 139 `zero_filled_n`) |

*Source: `reports/report4/member3_phase2_4/results_final_audit/06_chorus_missingness_reconciliation.csv`.*

**Table 11. Primary Missingness Association: Pearson Chi-Square Test of Chorus Missingness by Market on Exclusive Tracks ($N = 1,743$).**

| Market | Total Exclusive Tracks ($N$) | Observed Present ($n$) | Observed Missing ($n$) | Expected Present | Expected Missing | Missing Rate (%) | Pearson $\chi^2$ | $\text{df}$ | Unadjusted $p$ (`p_raw`) | Minimum Expected Cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN | 488 | 460 | 28 | 448.24 | 39.76 | 5.74 | 35.8621 | 2 | 1.63e-08 | 39.76 |
| US | 721 | 629 | 92 | 662.26 | 58.74 | 12.76 | 35.8621 | 2 | 1.63e-08 | 39.76 |
| KR | 534 | 512 | 22 | 490.50 | 43.50 | 4.12 | 35.8621 | 2 | 1.63e-08 | 39.76 |

*Source: `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv`.*

**Table 12. Diagnostic All-Market Missingness Contingency Table ($N = 2,098$ Repeated-Track Market Rows — Diagnostic Only).**

| Market | Total Market Rows ($N$) | Observed Present ($n$) | Observed Missing ($n$) | Expected Present | Expected Missing | Missing Rate (%) | Diagnostic Pearson $\chi^2$ | $\text{df}$ | Diagnostic $p$ | Minimum Expected Cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN | 608 | 575 | 33 | 563.08 | 44.92 | 5.43 | 39.6862 | 2 | 2.41e-09 | 44.92 |
| US | 797 | 702 | 95 | 738.12 | 58.88 | 11.92 | 39.6862 | 2 | 2.41e-09 | 44.92 |
| KR | 693 | 666 | 27 | 641.80 | 51.20 | 3.90 | 39.6862 | 2 | 2.41e-09 | 44.92 |

*Source: `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv`.*  
*Note: Because 169 tracks repeat across markets in the $N = 2,098$ table, classical Pearson $\chi^2 = 39.6862$ violates row independence and is reported strictly for descriptive cell reconciliation; valid all-market inference uses the cluster-robust logistic GEE in Table 13.*

**Table 13. Cluster-Robust Logistic GEE for Chorus Missingness Across All Market Observations ($N = 2,098$ Rows, $1,912$ Clusters).**

| Market Contrast (vs. US Reference) | Market Rows ($N$) | Track Clusters (`clusters`) | Log-Odds Coefficient ($\hat{\beta}$) | Odds Ratio ($\text{OR}$) | Coefficient $p$ | Omnibus Wald $\chi^2$ | $\text{df}$ | Omnibus $p$ | Convergence & Covariance Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VN vs. US | 2,098 | 1,912 | -0.8578 | 0.4241 | 3.23e-05 | 36.0346 | 2 | 1.50e-08 | Converged (2 iter); finite sandwich cov (rank 2) |
| KR vs. US | 2,098 | 1,912 | -1.2054 | 0.2996 | 4.64e-08 | 36.0346 | 2 | 1.50e-08 | Converged (2 iter); finite sandwich cov (rank 2) |

*Source: `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv`.*

---

## 4. Multicollinearity (VIF) Screening & Log-Transformation Diagnostics

### 4.1. Multicollinearity (VIF) Audit & Resolution of the Transformation Anomaly

Before estimating the OLS regression models on the canonical Training split (**$N = 1,685$ observations across $1,527$ track clusters**), the eight candidate lyrics features (`lexical_diversity`, `repetition_ratio`, `unique_word_count`, `word_count`, `line_count`, `punctuation_ratio`, `chorus_repetition_count`, and `avg_word_length`) were screened for multicollinearity using the Variance Inflation Factor ($\text{VIF}_j = 1 / (1 - R_j^2)$). Two distinct mathematical mechanisms of collinearity were identified and resolved:

1. **Exact Linear Algebraic Identity (`lexical_diversity` vs. `repetition_ratio`):**  
   By definition, $\text{lexical\_diversity} = \text{unique\_word\_count} / \text{word\_count}$ and $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$. After Z-score standardization, $\text{repetition\_ratio\_std} = -\text{lexical\_diversity\_std}$ identically. Including both variables alongside an intercept (`const`) makes the design matrix rank-deficient (`SingularMatrixWarning`, condition number $2.74 \times 10^{15}$), yielding floating-point infinity in `reports/vif_before.csv`:
   $$\text{VIF}(\text{lexical\_diversity\_std}) = \text{VIF}(\text{repetition\_ratio\_std}) = 1,000,799,917,193,443.5 \approx 1.0008 \times 10^{15}.$$
   **Decision 1:** Exclude `lexical_diversity_std` (`forbidden_cols` in `prepare_features.py`) and retain `repetition_ratio_std` to measure hook/lyrical repetition directly.

2. **Multiplicative / Near-Linear Collinearity (`unique_word_count` vs. `word_count` and `repetition_ratio`):**  
   By definition, $\text{unique\_word\_count} = \text{word\_count} \times (1 - \text{repetition\_ratio})$. In the initial 18-predictor screening matrix (`reports/vif_before.csv`, where both `word_count` and `unique_word_count` were on raw token scales), removing `lexical_diversity_std` immediately drops $\text{VIF}(\text{repetition\_ratio\_std})$ from $1.0008 \times 10^{15}$ to **$4.3372$**, while **`word_count_std` remains at $16.4196$** and **`unique_word_count_std` remains at $14.8026$** — both far exceeding the project threshold of $\text{VIF} < 5.0$. Removing `unique_word_count_std` at Step 2 reduces $\text{VIF}(\text{word\_count\_std})$ from $16.4196$ to **$3.4030$** (and **$3.7637$ / $3.7737$** in the final 24-predictor main-effects OLS matrices after log-transforming `word_count`).

> [!IMPORTANT]
> **Methodological Audit Note — Why Removing a Predictor Never Increases VIF on the Same Matrix:**  
> By the projection theorem of Ordinary Least Squares, removing a regressor from a fixed design matrix $X$ can only decrease or leave unchanged the auxiliary $R_j^2$ (and therefore $\text{VIF}_j = 1/(1 - R_j^2)$) of every remaining regressor. In an earlier draft comparison, a column labeled "7-feature lyrics block after dropping `lexical_diversity`" showed $\text{VIF}(\text{word\_count\_std}) = 70.8472$ and $\text{VIF}(\text{unique\_word\_count\_std}) = 52.9955$ next to `reports/vif_before.csv` ($16.4383$ and $14.8224$). Our reproducible audit (`reports/report4/member3_phase3/verify_member3_phase3_audit.py`) proves that those higher numbers were **not** caused by dropping `lexical_diversity_std`; rather, that auxiliary calculation applied $\ln(1 + x)$ to **both** `word_count` and `unique_word_count`. Because $\text{unique\_word\_count} = \text{word\_count} \times (1 - \text{repetition\_ratio})$, taking logarithms converts a multiplicative product into an almost exact additive linear combination:
> $$\ln(1 + \text{unique\_word\_count}) \approx \ln(1 + \text{word\_count}) + \ln(1 - \text{repetition\_ratio}),$$
> which inflates the linear $R^2$ among $(\ln\text{word\_count}, \ln\text{unique\_word\_count}, \text{repetition\_ratio})$ and raises their VIFs to $70.8472$, $52.9955$, and $21.6134$. When evaluated on the **same predictor set and identical feature transformations** (Table 14, Columns 3 $\to$ 4 $\to$ 5), VIF is strictly non-increasing at every step.

**Table 14. Step-by-Step Multicollinearity (VIF) Audit Across Predictor Sets and Transformations on the Canonical Training Split ($N = 1,685$).**

| Candidate Lyrics Feature | `reports/vif_before.csv` (18 Predictors, Raw Counts, Rounded TTR) | Stage 0: 18 Predictors, Raw Counts (With `lexical_diversity_std`) | Stage 1: Same 17 Predictors After Dropping Only `lexical_diversity_std` | Stage 2: Same 16 Predictors After Also Dropping `unique_word_count_std` | Stage 3A: Final 24 OLS Main Effects (`reports/vif_after.csv`, `log1p(word_count)`) | Stage 3B: Final 24 OLS Main Effects (Canonical Model 2, `log1p(word_count)`) | Auxiliary Diagnostic: 7-Feature Lyrics Block if `log1p` Applied to Both Counts | Final Pipeline Decision (`prepare_features.py`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lexical_diversity_std` | 1.0008e+15 | 1.0008e+15 | *Dropped (Step 1)* | *Dropped (Step 1)* | — | — | — | **Excluded (`forbidden_cols`):** Exact linear identity $\text{TTR} = 1 - \text{repetition\_ratio}$ |
| `repetition_ratio_std` | 1.0008e+15 | 1.0008e+15 | 4.3372 | 1.4740 | 1.7670 | 1.7669 | 21.6134 | **Retained for OLS:** Captures lyrical hook repetition ($\text{VIF} = 1.7669 < 5.0$) |
| `unique_word_count_std` | 14.8224 | 14.8026 | 14.8026 | *Dropped (Step 2)* | — | — | 52.9955 | **Excluded (`forbidden_cols`):** Multiplicative collinearity with `word_count` ($\text{VIF} > 14.8 > 5.0$) |
| `word_count_std` | 16.4383 | 16.4196 | 16.4196 | 3.4030 | 3.7637 | 3.7737 | 70.8472 | **Retained for OLS:** Log-transformed $\ln(1 + \text{word\_count})$ ($\text{VIF} = 3.7737 < 5.0$) |
| `line_count_std` | 3.1107 | 3.1336 | 3.1336 | 3.0489 | 2.7497 | 2.7631 | 2.4503 | **Retained for OLS:** Structural verse/line length indicator ($\text{VIF} = 2.7631 < 5.0$) |
| `punctuation_ratio_std` | 1.3420 | 1.3482 | 1.3482 | 1.3464 | 1.3680 | 1.3776 | 1.0714 | **Retained for OLS:** Expressive punctuation density ($\text{VIF} = 1.3776 < 5.0$) |
| `chorus_repetition_count_std` | 1.3455 | 1.3429 | 1.3429 | 1.2993 | 1.2991 | 1.2983 | 1.2828 | **Retained for OLS:** Chorus repetition frequency ($\text{VIF} = 1.2983 < 5.0$) |
| `avg_word_length_std` | 1.1894 | 1.1883 | 1.1883 | 1.1368 | 1.2137 | 1.2150 | 1.0802 | **Retained for OLS:** Morphological character-per-token index ($\text{VIF} = 1.2150 < 5.0$) |

*Source: `reports/vif_before.csv`, `reports/vif_after.csv`, `reports/report4/member3_phase3/vif_audit_evidence.csv`, and `reports/report4/member3_phase3/verify_member3_phase3_audit.py`.*  
*Note: The tiny $0.02$ difference between `reports/vif_before.csv` (`16.4383` / `14.8224`) and Stage 0 (`16.4196` / `14.8026`) arises solely because `lyrics_features_extracted.csv` rounds `lexical_diversity` and `repetition_ratio` to 4 decimal places (`1e-4`), slightly perturbing the floating-point pseudoinverse of the rank-deficient Stage 0 matrix. In Stage 0 $\to$ Stage 1 (exact unrounded identity), dropping `lexical_diversity_std` leaves `word_count_std` (`16.4196 -> 16.4196`) and `unique_word_count_std` (`14.8026 -> 14.8026`) completely unchanged to 8 decimal places.*

---

### 4.2. Word-Count Transformation ($\ln(1 + \text{word\_count})$) & Leverage/Influence Diagnostics

In `src/features/prepare_features.py` (`build_design_matrix()`, lines 179–180), `word_count` undergoes the natural log transformation $\text{word\_count}_{\text{log}} = \ln(1 + \text{word\_count})$ prior to Z-score standardization on the Training split ($N = 1,685$). **Table 15** documents both the distributional shape metrics across all three sample scopes and the exact OLS influence diagnostics (hat-matrix leverage $h_{ii} = x_i^\top (X^\top X)^{-1} x_i$ and Cook's distance $D_i$) in the 24-predictor Model 2 specification.

**Table 15. Distributional Shape and OLS Leverage/Influence Diagnostics for Raw `word_count` vs. $\ln(1 + \text{word\_count})$. **

| Sample Scope & Metric Category | Variable Scale | Sample Size ($N$) | Mean (SD) | Median (IQR) | Min – Max | Skewness | Excess Kurtosis | D'Agostino–Pearson $K^2$ ($p$-value) | Extreme Tail & Model 2 OLS Leverage / Influence Diagnostics ($N = 1,685, p = 25$) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Train OLS Split** | Raw `word_count` (tokens) | 1,685 | 378.8510 (178.8282) | 348.0000 (186.0000) | 11.00 – 1,985.00 | +1.5765 | +6.4205 | 608.6408 (6.84e-133) | Max $\lvert z\rvert = 8.9842$ ($23$ rows $\lvert z\rvert>3$); Max univariate $h_{ii} = 0.0485$; on $9$ tracks with $\ge 1,000$ words: **Max Model 2 $h_{ii} = 0.0764$** ($> 3p/N = 0.0445$), Mean $h_{ii} = 0.0366$, Max Cook's $D = 0.0032$ ($> 4/N = 0.0024$); $103$ rows exceed $D_i > 4/N$ |
| **Train OLS Split** | $\ln(1 + \text{word\_count})$ (log-tokens) | 1,685 | 5.8303 (0.4906) | 5.8551 (0.5211) | 2.4849 – 7.5939 | -0.7931 | +2.6455 | 249.1376 (7.95e-55) | Max $\lvert z\rvert = 6.8206$ ($18$ rows $\lvert z\rvert>3$); Max univariate $h_{ii} = 0.0282$ ($-41.8\%$); on $9$ tracks with $\ge 1,000$ words: **Max Model 2 $h_{ii} = 0.0502$ ($-34.3\%$), Mean $h_{ii} = 0.0253$ ($-30.8\%$)** ($< 2p/N = 0.0297$), Max Cook's $D = 0.0018$ ($< 4/N = 0.0024$); $99$ rows exceed $D_i > 4/N$ |
| **Market-Exclusive Valid** | Raw `word_count` (tokens) | 1,740 | 380.3902 (185.1735) | 348.0000 (206.0000) | 11.00 – 1,985.00 | +1.4304 | +5.2929 | 551.8841 (1.34e-120) | Right-skewed with heavy upper tail from US/VN rap tracks |
| **Market-Exclusive Valid** | $\ln(1 + \text{word\_count})$ (log-tokens) | 1,740 | 5.8240 (0.5160) | 5.8551 (0.5784) | 2.4849 – 7.5939 | -0.7891 | +2.0872 | 231.7107 (4.97e-51) | Substantially compressed upper tail; mean (`5.8240`) aligns closely with median (`5.8551`) |
| **All-Market Valid** | Raw `word_count` (tokens) | 2,095 | 377.6482 (178.3304) | 347.0000 (191.0000) | 11.00 – 1,985.00 | +1.4606 | +5.4457 | 678.9669 (3.84e-148) | Matches `01_word_count_by_market.csv` descriptive boxplot distribution |
| **All-Market Valid** | $\ln(1 + \text{word\_count})$ (log-tokens) | 2,095 | 5.8259 (0.4930) | 5.8522 (0.5396) | 2.4849 – 7.5939 | -0.7498 | +2.2355 | 272.2471 (8.01e-60) | Stabilizes variance across markets while remaining non-normal ($p < 0.0001$) |

*Source: `reports/report4/member3_phase3/member3_phase3_audit_summary.json`, `reports/diagnostics/diagnostics_summary.json`, and `reports/diagnostics/cooks_top10.csv`.*

**Statistical Interpretation of the Transformation:**
1. **Reduction of Right-Skewness and Tail Kurtosis:** On the Training split ($N = 1,685$), raw `word_count` spans from $11$ to $1,985$ tokens with severe positive skewness ($+1.5765$, $|z_{\max}| = 8.9842$) and heavy tails ($\text{excess kurtosis} = +6.4205$). Applying $\ln(1 + \text{word\_count})$ halves the absolute skewness (to $-0.7931$) and reduces excess kurtosis by $58.8\%$ (to $+2.6455$).
2. **Verified Reduction of High-Word-Count Leverage and Influence:** For the $9$ extreme rap/hip-hop tracks in Train with $\text{word\_count} \ge 1,000$ tokens, the $\ln(1 + \text{word\_count})$ transformation reduces maximum univariate leverage from $0.0485$ to $0.0282$ ($-41.8\%$), reduces mean 24-predictor hat-matrix leverage $h_{ii}$ from $0.0366$ (above the $2p/N = 0.0297$ warning threshold) to $0.0253$ (below $2p/N$), and reduces their maximum Cook's distance $D_i$ in Model 2 from $0.0032$ (above $4/N = 0.0024$) to $0.0018$ (below $4/N$). Across the full Model 3 specification (`reports/diagnostics/diagnostics_summary.json`), maximum Cook's distance is $0.0152$ (`1XZE0InCx3SdCnLpVsPpZl`), well below the classical $D_i < 0.50$ severe-influence cutoff.
3. **Non-Normality Caveat:** Because the D'Agostino–Pearson omnibus test on $\ln(1 + \text{word\_count})$ still rejects exact normality ($K^2 = 249.1376, p = 7.95 \times 10^{-55}$) due to a small cluster of very short skits/interludes in the lower tail ($11$–$35$ words), **we do not claim that the log transformation achieves normality**; rather, it compresses right-tail extremes and reduces high-word-count leverage in OLS estimation.

---

## 5. Audio Mode Cross-Market Analysis & Multi-Sample Reconciliation

### 5.1. Primary Audio-Mode Contingency Table & Chi-Square Test ($N = 1,743$)

Although `audio_mode` (`Major` vs. `Minor`) belongs to the acoustic feature domain (Member 2), Member 3 audited its cross-market distribution and its inclusion in the joint Audio + Lyrics OLS models. On the primary sample of **$N = 1,743$ market-exclusive tracks**, all six cells of the $3 \times 2$ contingency table have expected frequencies $\ge 195.98 \gg 5$, satisfying all classical conditions for Pearson's Chi-square test of independence.

**Table 16. Primary $3 \times 2$ Contingency Table and Pearson Chi-Square Test of `audio_mode` by Market on Market-Exclusive Tracks ($N = 1,743$).**

| Market (`market`) | Audio Mode (`audio_mode`) | Observed Count ($n$) | Expected Count | Market Total ($N$) | Within-Market Share (%) | Pearson $\chi^2$ | $\text{df}$ | Unadjusted $p$ (`p_raw`) | Cramér's V | Significance ($\alpha = 0.0500$) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **VN** | Major (`major`) | 336 | 292.02 | 488 | 68.85 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **VN** | Minor (`minor`) | 152 | 195.98 | 488 | 31.15 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **US** | Major (`major`) | 396 | 431.44 | 721 | 54.92 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **US** | Minor (`minor`) | 325 | 289.56 | 721 | 45.08 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **KR** | Major (`major`) | 311 | 319.54 | 534 | 58.24 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **KR** | Minor (`minor`) | 223 | 214.46 | 534 | 41.76 | 24.3142 | 2 | 5.25e-06 | 0.1181 | Significant (`True`) |
| **Total (`ALL`)** | **Major: 1,043 / Minor: 700** | **1,743** | **1,743.00** | **1,743** | **Major: 59.84 / Minor: 40.16** | **24.3142** | **2** | **5.25e-06** | **0.1181** | **Significant (Small Effect)** |

*Source: `reports/report4/member3_phase2_4/results_final_audit/02_primary_audio_mode_chi_square.csv`.*

---

### 5.2. Reconciliation Across All Observation Scopes ($N = 1,743$, $N = 2,098$, and $N = 2,860$)

Earlier project drafts and task descriptions referenced conflicting statistics for `audio_mode` — specifically marginal totals of `Major = 1,275/2,098 = 60.77%, Minor = 823/2,098 = 39.23%`, a draft Report 4 claim of `p = 0.42`, and a planning target of `χ² = 0.51, df = 2, p = 0.774`. **Table 17** and **Table 18** provide a complete, transparent reconciliation across all four analytical scopes in the repository:

1. **Primary Market-Exclusive Sample ($N = 1,743$, `02_primary_audio_mode_chi_square.csv`):**  
   `Major = 1,043 (59.84%)`, `Minor = 700 (40.16%)`; Pearson $\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}, \text{Cramér's V} = 0.1181$. This is the **official primary inference result** because each track is counted once.
2. **All-Market Canonical Sample ($N = 2,098$ rows across $1,912$ unique tracks, `data/processed/modeling_dataset_clean_cache.csv` & `05_audio_mode_logistic_gee.csv`):**  
   Here, `VN` has $414\text{ Major } (68.09\%) / 194\text{ Minor } (31.91\%)$, `US` has $450\text{ Major } (56.46\%) / 347\text{ Minor } (43.54\%)$, and `KR` has $411\text{ Major } (59.31\%) / 282\text{ Minor } (40.69\%)$. Summing across all $2,098$ rows yields **`Major = 1,275 / 2,098 (60.77%)` and `Minor = 823 / 2,098 (39.23%)`** — explaining the exact origin of the $1,275 / 823$ counts! However, on this exact $2,098$-row table, the **naive Pearson Chi-square is $\chi^2(2) = 20.5006, p = 3.54 \times 10^{-5}$** ($\text{Cramér's V} = 0.0989$, diagnostic only due to 169 repeated tracks), and the **cluster-robust Binomial Logistic GEE (`05_audio_mode_logistic_gee.csv`) yields $\text{Wald }\chi^2(2) = 21.7232, p = 1.92 \times 10^{-5}$** ($\text{OR}_{\text{VN vs US}} = 1.6456, p = 7.25 \times 10^{-6}$; $\text{OR}_{\text{KR vs US}} = 1.1239, p = 0.2471$).
3. **Historical Report 4 EDA Dataset ($N = 2,860$ rows, `08_historical_audio_mode_reconciliation.csv`):**  
   Reconstructing the $3 \times 2$ table from both `reports/report4/chart_data/member2/fig3_tonal_harmony_market.csv` (72 aggregated rows) and `audio_features_eda_source.csv` (2,860 track-market rows) yields `VN: 567 Major / 318 Minor (885)`, `US: 539 Major / 416 Minor (955)`, and `KR: 598 Major / 422 Minor (1,020)` (`Total: 1,704 Major [59.58%] / 1,156 Minor [40.42%]`). Exact recomputation gives **Pearson $\chi^2(2) = 11.6962, p = 0.0029$** (`0.002885`).
4. **Unverified Legacy Claims (`p = 0.42` and `χ² = 0.51, p = 0.774`):**  
   Neither `p = 0.42` (from an early Report 4 text draft) nor `χ² = 0.51, df = 2, p = 0.774` (from an early v4.1 task prompt) can be reproduced from any dataset in the repository; per `08_historical_audio_mode_reconciliation.csv`, both are formally classified as **Not Reproduced / Source Unavailable (`NOT VERIFIED`)**.

**Table 17. Cluster-Robust Logistic GEE Sensitivity Analysis for `audio_mode` (`Major = 1`, `Minor = 0`) on All Market Observations ($N = 2,098$, $1,912$ Clusters).**

| Market Contrast (vs. US Reference) | Market Rows ($N$) | Track Clusters (`clusters`) | Log-Odds Coefficient ($\hat{\beta}$) | Odds Ratio ($\text{OR}$) | Coefficient $p$ | Omnibus Wald $\chi^2$ | $\text{df}$ | Omnibus $p$ | Convergence & Covariance Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VN vs. US | 2,098 | 1,912 | +0.4981 | 1.6456 | 7.25e-06 | 21.7232 | 2 | 1.92e-05 | Converged (2 iter); finite sandwich cov (rank 2) |
| KR vs. US | 2,098 | 1,912 | +0.1168 | 1.1239 | 0.2471 | 21.7232 | 2 | 1.92e-05 | Converged (2 iter); finite sandwich cov (rank 2) |

*Source: `reports/report4/member3_phase2_4/results_final_audit/05_audio_mode_logistic_gee.csv`.*

**Table 18. Comprehensive Reconciliation of `audio_mode` Contingency Counts and Hypothesis Tests Across All Four Analytical Scopes.**

| Analytical Scope & Method | Observation Unit | Sample Size ($N$) | Major / Minor Breakdown by Market | Overall Major / Minor Totals (%) | Test Statistic ($\text{df} = 2$) | $p$-Value | Effect Size / Contrast Summary | Evidence File & Audit Status |
|---|---|---:|---|---|---:|---:|---|---|
| **1. Primary Audio Analysis (Pearson $\chi^2$)** | Market-exclusive tracks (1 row per track) | 1,743 | VN: 336 / 152 (68.85% / 31.15%)<br>US: 396 / 325 (54.92% / 45.08%)<br>KR: 311 / 223 (58.24% / 41.76%) | Major: **1,043** (59.84%)<br>Minor: **700** (40.16%) | $\chi^2 = 24.3142$ | 5.25e-06 | $\text{Cramér's V} = 0.1181$ (Small effect) | `02_primary_audio_mode_chi_square.csv` — **OFFICIAL PRIMARY RESULT** |
| **2. All-Market Cluster-Robust Logistic GEE** | Track-market rows clustered by $1,912$ `spotify_track_id` | 2,098 | VN: 414 / 194 (68.09% / 31.91%)<br>US: 450 / 347 (56.46% / 43.54%)<br>KR: 411 / 282 (59.31% / 40.69%) | Major: **1,275** (60.77%)<br>Minor: **823** (39.23%) | $\text{Wald }\chi^2 = 21.7232$ | 1.92e-05 | $\text{OR}_{\text{VN vs US}} = 1.6456$ ($p = 7.25\text{e-}06$)<br>$\text{OR}_{\text{KR vs US}} = 1.1239$ ($p = 0.2471$) | `05_audio_mode_logistic_gee.csv` — **CLUSTER-AWARE SENSITIVITY** |
| **3. All-Market Naive Pearson $\chi^2$** | Track-market rows (ignores 169 multi-market tracks) | 2,098 | VN: 414 / 194<br>US: 450 / 347<br>KR: 411 / 282 | Major: **1,275** (60.77%)<br>Minor: **823** (39.23%) | $\chi^2 = 20.5006$ | 3.54e-05 | $\text{Cramér's V} = 0.0989$ (Violates row independence) | `modeling_dataset_clean_cache.csv` — **DIAGNOSTIC ONLY (Origin of 1,275 / 823)** |
| **4. Historical Report 4 Reconciliation** | Historical EDA track-market rows (`fig3_tonal_harmony_market.csv`) | 2,860 | VN: 567 / 318 (64.07% / 35.93%)<br>US: 539 / 416 (56.44% / 43.56%)<br>KR: 598 / 422 (58.63% / 41.37%) | Major: **1,704** (59.58%)<br>Minor: **1,156** (40.42%) | $\chi^2 = 11.6962$ | 0.0029 | Expected cells: VN $527.29/357.71$, US $568.99/386.01$, KR $607.72/412.28$ | `08_historical_audio_mode_reconciliation.csv` — **HISTORICAL AUDIT (`p=0.42` & `χ²=0.51, p=0.774` NOT VERIFIED)** |

*Source: `02_primary_audio_mode_chi_square.csv`, `05_audio_mode_logistic_gee.csv`, `08_historical_audio_mode_reconciliation.csv`, and `data/processed/modeling_dataset_clean_cache.csv`.*

---

## 6. Regression Modeling Decision & OLS Results (`mode` and Lyrics Block)

### 6.1. Encoding of `mode` and Rationale for Omitting `market × mode` Interactions

In `src/features/prepare_features.py` (lines 195–248) and `models/model_metadata.json`, categorical `audio_mode` is encoded as an **unscaled binary indicator `mode`** where **`Major = 1.0`** and **`Minor = 0.0` (reference category)**, while market fixed effects use **`US` as the reference market (`base_market = "US"`)**, generating binary indicators `Market_VN` and `Market_KR`.

Why is `mode` included as an additive main effect across Models 1, 2, and 3, while `market × mode` interaction terms are omitted from Model 3?
1. **Substantive Effect-Size Threshold & Parsimony Design:** Although `audio_mode` differs statistically across markets in the primary Chi-square test ($\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}$), its association with market is weak ($\text{Cramér's V} = 0.1181$). Under the pre-specified econometric architecture (`DEFAULT_CONFIG["interaction_cols"]`), to preserve degrees of freedom and prevent variance inflation on $N = 1,685$ training observations, market interactions (`__x__VN`, `__x__KR`) were restricted to the **five continuous features showing the strongest cross-market distributional divergence**: two audio features (`energy`, `spectral_centroid`) and three lyrics features (`avg_word_length`, `word_count`, `repetition_ratio`), creating exactly $5 \times 2 = 10$ interaction terms in Model 3 ($34$ total predictors).
2. **Direct Empirical Verification (Auxiliary Cluster-Robust Wald Test):** When `mode__x__VN` and `mode__x__KR` are added to Model 3 on the canonical Training split ($N = 1,685$ rows, $1,527$ track clusters), their **joint cluster-robust Wald test is non-significant: $\text{Wald }\chi^2(2) = 2.4278, p = 0.2970$** (`mode__x__VN`: $\hat{\beta} = -7.3935, SE_{\text{cluster}} = 4.9270, p = 0.1335$; `mode__x__KR`: $\hat{\beta} = -1.7713, SE_{\text{cluster}} = 4.6697, p = 0.7045$). Thus, both model-design parsimony and empirical hypothesis testing confirm that an additive main effect for `mode` is sufficient.

---

### 6.2. Cluster-Robust OLS Regression Results Across Model Specifications

All OLS models predict continuous `chart_score = 101 - lifetime_average_rank` (empirical domain $[-99.0, 100.0]$) on the canonical Training split (**$N = 1,685$ rows across $1,527$ unique track clusters**) using cluster-robust standard errors grouped by `spotify_track_id` (`src/utils/ols_cluster_fit.py`) and are evaluated out-of-sample on the disjoint Test split (**$N = 410$ rows across $382$ track clusters**).

To ensure complete transparency across the repository's modeling outputs, **Table 19** and **Table 20** distinguish four exact model specifications:
- **Model 1 Canonical (`audio + controls`, 18 predictors, `reports/report4/member3_canonical/member3_model_coefficients.csv`):** 14 continuous audio features + binary `mode` + 3 control variables (`release_year_std`, `Market_VN`, `Market_KR`).
- **Model 2 Canonical (`audio + lyrics + controls`, 24 predictors, `reports/report4/member3_canonical/member3_model_coefficients.csv`):** Model 1 (18 predictors) + the 6 standardized lyrics predictors (`word_count_std`, `repetition_ratio_std`, `avg_word_length_std`, `line_count_std`, `chorus_repetition_count_std`, `punctuation_ratio_std`).
- **Model 2 Nested Hierarchy Without Controls (`Audio + Lyrics`, 21 predictors, `reports/model_wald_blocks.csv`):** 15 audio features (14 continuous + `mode`) + 6 lyrics predictors, estimated prior to adding the 3 context controls (`Model 2b [+ Context]` in `reports/model_wald_blocks.csv` equals the 24-predictor Model 2).
- **Model 3 Full OLS (`Audio + Lyrics + Controls + 10 Market Interactions`, 34 predictors, `reports/model3_coefficients.csv`):** Adds 10 market interaction terms (`energy`, `spectral_centroid`, `avg_word_length`, `word_count`, and `repetition_ratio` crossed with `Market_VN` and `Market_KR`).

**Table 19. Cluster-Robust OLS Estimates for `mode` and the Six Lyrical Predictors Across Canonical Model 1, Canonical Model 2, and Full Model 3 (Train $N = 1,685$, $1,527$ Track Clusters).**

| Model Specification | Predictor Term (`term`) | Scale / Interpretation | Estimate ($\hat{\beta}$) | Cluster-Robust $SE$ | $t$-Statistic | $p$-Value | 95% Confidence Interval | VIF | Coefficient Scope |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| **Model 1 Canonical** (18 vars) | `mode` (`Major = 1`, `Minor = 0`) | Unscaled binary indicator | +3.2387 | 2.1283 | +1.5217 | 0.1281 | $[-0.9327,\ +7.4101]$ | 1.1875 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `mode` (`Major = 1`, `Minor = 0`) | Unscaled binary indicator | +3.4233 | 2.1304 | +1.6069 | 0.1081 | $[-0.7521,\ +7.5987]$ | 1.2014 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `word_count_std` | 1 SD in $\ln(1 + \text{word\_count})$ | -2.0718 | 1.9776 | -1.0476 | 0.2948 | $[-5.9477,\ +1.8042]$ | 3.7737 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `repetition_ratio_std` | 1 SD in `repetition_ratio` | +1.9631 | 1.3462 | +1.4582 | 0.1448 | $[-0.6755,\ +4.6017]$ | 1.7669 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `avg_word_length_std` | 1 SD in `avg_word_length` | -0.8968 | 1.0262 | -0.8739 | 0.3822 | $[-2.9082,\ +1.1146]$ | 1.2150 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `line_count_std` | 1 SD in `line_count` | +2.5327 | 1.5103 | +1.6770 | 0.0935 | $[-0.4274,\ +5.4927]$ | 2.7631 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `chorus_repetition_count_std` | 1 SD in `chorus_repetition_count` | +0.9848 | 1.0448 | +0.9425 | 0.3459 | $[-1.0630,\ +3.0325]$ | 1.2983 | Pooled main effect |
| **Model 2 Canonical** (24 vars) | `punctuation_ratio_std` | 1 SD in `punctuation_ratio` | +2.0943 | 1.1159 | +1.8767 | 0.0606 | $[-0.0929,\ +4.2815]$ | 1.3776 | Pooled main effect |
| **Model 3 Full OLS** (34 vars) | `mode` (`Major = 1`, `Minor = 0`) | Unscaled binary indicator | +3.2146 | 2.1350 | +1.5057 | 0.1322 | $[-0.9700,\ +7.3993]$ | 1.2011 | Pooled main effect |
| **Model 3 Full OLS** (34 vars) | `word_count_std` | 1 SD in $\ln(1 + \text{word\_count})$ | -3.8447 | 2.9355 | -1.3097 | 0.1903 | $[-9.5981,\ +1.9087]$ | 3.7637 | **US baseline slope** |
| **Model 3 Full OLS** (34 vars) | `repetition_ratio_std` | 1 SD in `repetition_ratio` | +1.8813 | 2.5292 | +0.7439 | 0.4570 | $[-3.0757,\ +6.8384]$ | 1.7670 | **US baseline slope** |
| **Model 3 Full OLS** (34 vars) | `avg_word_length_std` | 1 SD in `avg_word_length` | +4.2917 | 3.5269 | +1.2169 | 0.2237 | $[-2.6208,\ +11.2043]$ | 1.2137 | **US baseline slope** |
| **Model 3 Full OLS** (34 vars) | `line_count_std` | 1 SD in `line_count` | +2.6231 | 1.5871 | +1.6527 | 0.0984 | $[-0.4877,\ +5.7338]$ | 2.7497 | Pooled main effect |
| **Model 3 Full OLS** (34 vars) | `chorus_repetition_count_std` | 1 SD in `chorus_repetition_count` | +0.8928 | 1.0705 | +0.8340 | 0.4043 | $[-1.2054,\ +2.9910]$ | 1.2991 | Pooled main effect |
| **Model 3 Full OLS** (34 vars) | `punctuation_ratio_std` | 1 SD in `punctuation_ratio` | +2.2653 | 1.1543 | +1.9625 | 0.0497 | $[+0.0029,\ +4.5277]$ | 1.3680 | Pooled main effect ($p_{\text{Holm}} = 0.7953$) |
| **Model 3 Full OLS** (34 vars) | `avg_word_length_std__x__VN` | VN slope differential vs. US | -6.9729 | 4.2444 | -1.6428 | 0.1004 | $[-15.2917,\ +1.3460]$ | — | Market interaction |
| **Model 3 Full OLS** (34 vars) | `avg_word_length_std__x__KR` | KR slope differential vs. US | -5.2801 | 3.6754 | -1.4366 | 0.1508 | $[-12.4839,\ +1.9236]$ | — | Market interaction |
| **Model 3 Full OLS** (34 vars) | `word_count_std__x__VN` | VN slope differential vs. US | +4.3095 | 3.2726 | +1.3169 | 0.1879 | $[-2.1046,\ +10.7237]$ | — | Market interaction |
| **Model 3 Full OLS** (34 vars) | `word_count_std__x__KR` | KR slope differential vs. US | +0.5812 | 2.9733 | +0.1955 | 0.8450 | $[-5.2465,\ +6.4088]$ | — | Market interaction |
| **Model 3 Full OLS** (34 vars) | `repetition_ratio_std__x__VN` | VN slope differential vs. US | +1.4760 | 3.3146 | +0.4453 | 0.6561 | $[-5.0205,\ +7.9725]$ | — | Market interaction |
| **Model 3 Full OLS** (34 vars) | `repetition_ratio_std__x__KR` | KR slope differential vs. US | +0.3989 | 2.9114 | +0.1370 | 0.8910 | $[-5.3073,\ +6.1051]$ | — | Market interaction |

*Source: `reports/report4/member3_canonical/member3_model_coefficients.csv` and `reports/model3_coefficients.csv`.*

**Table 20. Joint Wald Block Tests, In-Sample Fit ($N = 1,685$), and Out-of-Sample Test Performance ($N = 410$) Across the OLS Model Hierarchy.**

| Model / Hypothesis Test | Predictor Block Tested | Predictors ($p$) / Parameters ($k$) | Joint Wald $\chi^2$ | $\text{df}$ | Unadjusted $p$ | Holm $p$ | Train $R^2$ (Adj. $R^2$) | AIC / BIC (Train) | Test $R^2$ (Adj. $R^2$, $N=410$) | Test RMSE / MAE ($N=410$) | Source Artifact |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| **M1 Audio Baseline** (No Controls) | 15 Audio features (`14 continuous + mode`) | 15 / 16 | 30.7982 | 15 | 0.0093 | — | 0.0165 (0.0076) | 17,146.72 / 17,233.59 | — | — | `reports/model_wald_blocks.csv` |
| **Model 2 Nested** (`Audio + Lyrics`, No Controls) | **6 Lyrics main effects** (added to 15 Audio) | 21 / 22 | **15.5722** | **6** | **0.0162** | — | 0.0256 (0.0133) | 17,142.95 / 17,262.40 | — | — | `reports/model_wald_blocks.csv` |
| **Model 1 Canonical** (`Audio + Controls`) | 15 Audio + 3 Controls (`year`, `VN`, `KR`) | 18 / 19 | — | — | — | — | 0.0480 (0.0377) | — | 0.0470 (0.0031) | 38.0461 / 29.7387 | `member3_model_comparison.csv` |
| **Model 2 Canonical / M2b** (`Audio + Lyrics + Controls`) | **6 Lyrics main effects** (controlling for Audio + Controls) | 24 / 25 | **13.5932** | **6** | **0.0345** | — | 0.0562 (0.0425) | 17,095.26 / 17,230.99 | 0.0365 (-0.0236) | 38.2550 / 30.0569 | `member3_lyrics_joint_wald.json` & `model_wald_blocks.csv` |
| **Model 3 Full OLS** (`+ 10 Market Interactions`) | **6 Lyrics main effects** (3 US slopes + 3 pooled slopes) | 34 / 35 | **11.7440** | **6** | **0.0679** | 0.1131 | 0.0656 (0.0464) | 17,098.35 / 17,288.38 | 0.0272* | 38.4389 / 29.6631* | `reports/model_wald_blocks.csv` |
| **Model 3 Full OLS** | **12-DOF Full Lyrics Family** (6 Main + 6 Interactions) | 34 / 35 | **19.6847** | **12** | **0.0733** | — | 0.0656 (0.0464) | 17,098.35 / 17,288.38 | — | — | `reports/model_wald_blocks.csv` |
| **Model 3 Auxiliary Check** | **2-DOF `mode × Market` Interactions** (`mode__x__VN`, `mode__x__KR`) | 36 / 37 | **2.4278** | **2** | **0.2970** | — | — | — | — | — | `member3_phase3_audit_summary.json` |

*Source: `reports/report4/member3_canonical/member3_lyrics_joint_wald.json`, `reports/report4/member3_canonical/member3_model_comparison.csv`, `reports/report4/member3_canonical/member3_model_test_metrics.csv`, and `reports/model_wald_blocks.csv`. (*Note: Model 3 pre-1980 sensitivity test metrics from `reports/diagnostics/diagnostics_summary.json`).*

#### Reconciling Joint Wald Significance Across Model 2 and Model 3
A critical statistical insight emerges from **Table 19** and **Table 20**:
1. **Why is the 6-feature lyrics block jointly significant in Model 2 ($p = 0.0345$ with controls; $p = 0.0162$ without controls) even though no individual lyrics predictor has $p < 0.05$ in Model 2?**  
   In Model 2 (`member3_model_coefficients.csv`), individual lyrical predictors show moderate positive or negative partial associations (`punctuation_ratio_std`: $\hat{\beta} = +2.0943, p = 0.0606$; `line_count_std`: $\hat{\beta} = +2.5327, p = 0.0935$; `repetition_ratio_std`: $\hat{\beta} = +1.9631, p = 0.1448$; `word_count_std`: $\hat{\beta} = -2.0718, p = 0.2948$). Because song length (`word_count_std`), verse count (`line_count_std`), and hook repetition (`repetition_ratio_std`) share moderate covariance ($\text{VIF}$ between $1.7669$ and $3.7737$), their individual $t$-tests partition overlapping variance, whereas the 6-DOF cluster-robust **Joint Wald test** evaluates the quadratic form $\hat{\beta}_{\text{lyrics}}^\top \hat{\Sigma}_{\text{lyrics}}^{-1} \hat{\beta}_{\text{lyrics}} = 13.5932$ ($p = 0.0345$), confirming that the lyrical block as a whole contributes statistically significant explanatory information beyond audio features and market/year controls on the Training split.
2. **Why does the 6-DOF lyrics main-effect Wald test change to $\chi^2(6) = 11.7440$ ($p = 0.0679, p_{\text{Holm}} = 0.1131$) in Model 3?**  
   In Model 2, all six lyrics coefficients represent **pooled slopes across all three markets** ($N = 1,685$). In Model 3, because `word_count_std`, `repetition_ratio_std`, and `avg_word_length_std` are interacted with `Market_VN` and `Market_KR`, their "main effect" terms estimate **slopes within the US reference market only** ($n_{\text{US, train}} \approx 640$), increasing their cluster-robust standard errors (e.g., $SE(\text{avg\_word\_length\_std})$ rises from $1.0262$ in Model 2 to $3.5269$ in Model 3) and shifting the 6-DOF test to marginal/suggestive significance ($p = 0.0679$).
3. **Out-of-Sample Predictive Reality:**  
   On the held-out Test split ($N = 410$), both Model 1 ($R^2 = 0.0470, \text{RMSE} = 38.0461, \text{MAE} = 29.7387$) and Model 2 ($R^2 = 0.0365, \text{RMSE} = 38.2550, \text{MAE} = 30.0569$) explain less than $5\%$ of the variance in `chart_score`. Adding 6 lyrical parameters slightly increases test RMSE by $0.2090$ points and reduces adjusted $R^2$ from $+0.0031$ to $-0.0236$. This demonstrates that while lyrical structure is statistically differentiated across markets and jointly associated with chart score in-sample, audio and surface NLP features alone have very low out-of-sample predictive power for commercial chart rank, which is heavily driven by artist stardom, marketing, playlist placement, and social virality.

---

## 7. Methodological Limitations & Threats to Validity

1. **Selection Bias in the Market-Exclusive Primary Sample:**  
   Restricting primary Kruskal–Wallis and Pearson Chi-square tests to the $N = 1,743$ market-exclusive tracks guarantees independent observations (one row per track) but excludes $169$ multi-market tracks — which are predominantly global megahits charting simultaneously in `US`, `VN`, and `KR`. While the all-market Rank-GEE ($N = 2,095$) and Logistic GEE ($N = 2,098$) sensitivity analyses confirm the same qualitative conclusions, primary exclusive-track estimates cannot be blindly generalized to cross-border hits.
2. **Cross-Linguistic Tokenization & Morphological Artifacts:**  
   As shown in Figure 3, `market` is not identical to `language` (e.g., `KR` contains $48.92\%$ English, $28.72\%$ pure Korean, and $17.17\%$ mixed Korean–English tracks; `VN` contains $60.36\%$ Vietnamese and $28.62\%$ English tracks). Furthermore, the NLP pipeline tokenizes Vietnamese by whitespace into monosyllabic *tiếng* (syllables), Korean by whitespace into agglutinative *eojeols* (stem + grammatical particles), and English by word regex. Consequently, the large cross-market differences in `avg_word_length` ($\varepsilon^2 = 0.3196$) and `word_count` ($\varepsilon^2 = 0.1542$) reflect surface orthographic and morphological tokenization units rather than language-invariant semantic length.
3. **Algebraic Redundancy Between `lexical_diversity` and `repetition_ratio`:**  
   Because $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$, their identical Kruskal–Wallis statistics ($H = 147.7139, p_{\text{Holm}} = 3.36 \times 10^{-32}$) and opposite-signed Rank-GEE contrasts ($\pm 0.0103, \pm 0.1359$) represent a single mathematical degree of freedom.
4. **Non-Random Missingness in `chorus_repetition_count`:**  
   Unidentified chorus structure (`NULL` in $148$ unique tracks / $142$ exclusive tracks) occurs significantly more often in `US` ($12.76\%$) than in `VN` ($5.74\%$) or `KR` ($4.12\%$) ($p = 1.63 \times 10^{-8}$). Treating `NULL` as `0` artificially induces a significant Kruskal–Wallis result ($p = 0.0212$ vs. complete-case $p = 0.3060$). In OLS regression, zero-filling is a pragmatic engineering compromise to preserve sample size ($N = 1,685$) and must be interpreted with caution.
5. **Difference in Estimands Across Statistical Frameworks:**  
   Kruskal–Wallis tests equality of rank distributions among market-exclusive tracks; Rank-GEE estimates marginal population-averaged rank contrasts across all market rows clustered by track; and OLS estimates linear conditional expectations of `chart_score`. Results across these three frameworks complement one another but do not estimate the same parameter.
6. **Observational Design & Non-Causal Interpretation:**  
   All analyses are conditioned on songs that already entered the quarterly Spotify Top 200 sample (survivorship/selection conditioning). Neither Kruskal–Wallis differences nor OLS regression coefficients establish causal effects; altering a song's word count, repetition ratio, or musical mode will not causally guarantee higher chart performance.

---

## 8. Final Conclusions for Member 3

1. **Cross-Market Lyrical Differentiation (Primary Kruskal–Wallis, $N = 1,743$):** Five of the six tested lyrics features differ significantly across `VN`, `US`, and `KR` after Holm adjustment ($\alpha = 0.0500$): `avg_word_length` ($H = 557.0660, p_{\text{Holm}} = 6.50 \times 10^{-121}, \varepsilon^2 = 0.3196$), `word_count` ($H = 269.8847, p_{\text{Holm}} = 1.24 \times 10^{-58}, \varepsilon^2 = 0.1542$), `lexical_diversity` and `repetition_ratio` ($H = 147.7139, p_{\text{Holm}} = 3.36 \times 10^{-32}, \varepsilon^2 = 0.0839$), and `unique_word_count` ($H = 135.7840, p_{\text{Holm}} = 6.55 \times 10^{-30}, \varepsilon^2 = 0.0770$). Complete-case `chorus_repetition_count` does not differ significantly across markets ($H = 2.3681, p_{\text{Holm}} = 0.3060, \varepsilon^2 = 0.0002$).
2. **Multicollinearity Screening & Log Transformation (Train $N = 1,685$):** Removing `lexical_diversity_std` ($\text{VIF} = 1.0008 \times 10^{15}$) and `unique_word_count_std` ($\text{VIF} = 14.8026$) eliminates both exact algebraic singularity and multiplicative token-count collinearity, leaving six lyrical OLS predictors with $\text{VIF} \in [1.2150, 3.7737] < 5.0$. Applying $\ln(1 + \text{word\_count})$ reduces training skewness from $+1.5765$ to $-0.7931$ and reduces maximum Model 2 hat-matrix leverage on extreme $\ge 1,000$-word tracks from $0.0764$ to $0.0502$.
3. **Categorical Audio Mode ($N = 1,743$ vs. $N = 2,098$):** In the primary exclusive-track test ($N = 1,743$), `audio_mode` (`Major`: $59.84\%$, `Minor`: $40.16\%$) is significantly associated with market ($\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}, \text{Cramér's V} = 0.1181$), with `VN` exhibiting the highest Major-key share ($68.85\%$). On the all-market dataset ($N = 2,098$, where `Major = 1,275 [60.77%]` and `Minor = 823 [39.23%]`), cluster-robust logistic GEE confirms this association ($\text{Wald }\chi^2(2) = 21.7232, p = 1.92 \times 10^{-5}$).
4. **OLS Regression Integration (Train $N = 1,685$, Test $N = 410$):** Binary `mode` (`Major = 1.0`, `Minor = 0.0`) enters Models 1, 2, and 3 as an additive main effect ($\hat{\beta} = +3.2387$ to $+3.4233, p = 0.1081$ to $0.1322$), and an auxiliary Wald test confirms that `market × mode` interactions are non-significant ($\text{Wald }\chi^2(2) = 2.4278, p = 0.2970$). The six-feature lyrics block is jointly significant in **Model 2 (`audio + lyrics + controls`, $\text{Wald }\chi^2(6) = 13.5932, p = 0.0345$)** and in **Model 2 without controls ($\text{Wald }\chi^2(6) = 15.5722, p = 0.0162$)**, while out-of-sample test metrics ($R^2_{\text{test}} = 0.0365$) underscore that intrinsic audio and lyrical features explain only a modest fraction of commercial chart performance.

---

## 9. Data Sources, Artifacts & Reproducibility Index

All tables, statistical tests, and diagnostics in this report are 100% reproducible from the following repository artifacts (all paths relative to the repository root):

- **Canonical Modeling Dataset & Provenance:** `data/processed/modeling_dataset_clean_cache.csv`, `reports/report3/lyrics_features_extracted.csv`, `data/raw/lyrics_collection_results.json`, `reports/report3/lyrics_statistics.csv`
- **Phase 2.4 Audited Statistical Outputs (9 CSVs + Run Log):** `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv` through `09_data_provenance_audit.csv`, `reports/report4/member3_phase2_4/results_final_audit/run_log.json`, and `reports/report4/member3_phase2_4/reproduce_phase2_4.py`
- **Preprocessing, VIF Screening & Diagnostics:** `src/features/prepare_features.py`, `reports/vif_before.csv`, `reports/vif_after.csv`, `reports/vif_decision_log.csv`, `reports/diagnostics/diagnostics_summary.json`, `reports/diagnostics/cooks_top10.csv`, `reports/diagnostics/reset_test.json`
- **Canonical OLS Modeling & Wald Block Outputs:** `src/modeling/train_model2_lyrics.py`, `src/utils/ols_cluster_fit.py`, `models/model_metadata.json`, `reports/report4/member3_canonical/member3_model_run_summary.json`, `reports/report4/member3_canonical/member3_model_coefficients.csv`, `reports/report4/member3_canonical/member3_lyrics_joint_wald.json`, `reports/report4/member3_canonical/member3_model_comparison.csv`, `reports/report4/member3_canonical/member3_model_test_metrics.csv`, `reports/model3_coefficients.csv`, `reports/model3_wald_features.csv`, `reports/model_wald_blocks.csv`
- **Phase 3 Reproducible Verification Script & Evidence:** `reports/report4/member3_phase3/verify_member3_phase3_audit.py`, `reports/report4/member3_phase3/vif_audit_evidence.csv`, `reports/report4/member3_phase3/member3_phase3_audit_summary.json`, `reports/report4/member3_phase3/member3_phase3_data_and_table_guide.md`, and `reports/report4/member3_phase3/member3_phase3_validation.md`
