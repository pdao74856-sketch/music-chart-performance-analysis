# Data, Table, and Figure Integration Guide — Member 3 (TV3) Phase 3

**Course:** ADY201m — Applied Data Science Project  
**Project Title:** *What Makes a Hit Song? Comparing Factors Affecting Music Popularity Across Vietnam, the United States, and South Korea (2021–2026)*  
**Primary Report:** `reports/report4/member3_phase3/member3_phase3_report.md`  
**Validation Report:** `reports/report4/member3_phase3/member3_phase3_validation.md`

---

## 1. Integration Architecture in the Master Project Report

When integrating Member 3's finalized Phase 3 academic contribution (`reports/report4/member3_phase3/member3_phase3_report.md`) into the project's master report (`reports/report4/report4.md` and the final regression report):

1. **Section 4 (`Lyrical NLP & Cross-Market Text Structure — Member 3`):**
   - Retain the descriptive boxplots and language composition overview in Subsections 4.1–4.2 (Figures 1–3).
   - Insert **Section 3 of `member3_phase3_report.md`** (Tables 1–13) immediately following the descriptive overview to provide the formal non-parametric Kruskal–Wallis primary inference ($N = 1,743$), market-level parametric/non-parametric descriptives ($N = 1,743$ and $N = 2,098$), cluster-robust Rank-GEE sensitivity analysis ($N = 2,095$), chorus zero-fill sensitivity analysis ($N = 1,740$), and chorus missingness reconciliation ($N = 1,912$, $N = 2,098$, and $N = 1,743$).
2. **Acoustic Cross-Reference (`Audio Mode` — Bridge Between Member 2 and Member 3):**
   - Insert **Section 5 of `member3_phase3_report.md`** (Tables 16–18) as a dedicated cross-domain subsection bridging Member 2's tonal harmony analysis and Member 3's regression preparation, clearly distinguishing the primary exclusive-track Chi-square result ($\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}, N = 1,743$) from the all-market marginal counts (`1,275 Major / 823 Minor`, $N = 2,098$) and the historical EDA dataset ($N = 2,860$).
3. **Stage 5 / Phase 3 Regression Modeling Section (`Multicollinearity Screening, Log Transformation & OLS Hierarchy`):**
   - Insert **Section 4** (Tables 14–15: Step-by-step VIF screening and $\ln(1 + \text{word\_count})$ distributional/leverage diagnostics) and **Section 6** (Tables 19–20: Cluster-robust OLS estimates for `mode` and the 6-feature lyrics block across Model 1, Model 2, and Model 3).

---

## 2. Complete Index of All 20 Tables in `member3_phase3_report.md`

All file paths below are **relative to the repository root**. Every value in `member3_phase3_report.md` has been cross-checked against its exact source CSV or JSON artifact.

**Table A1. Master Mapping of Tables 1–20 in `member3_phase3_report.md` to Source Artifacts and Analytical Scopes.**

| Table No. | Table Title in `member3_phase3_report.md` | Exact Relative Path(s) to Source Artifact(s) | Analytical Classification & Unit of Observation | Key Statistical / Methodological Notes |
|---|---|---|---|---|
| **Table 1** | Data Provenance Audit, Input SHA-256 Hashes, and Canonical Sample Sizes | `reports/report4/member3_phase2_4/results_final_audit/09_data_provenance_audit.csv`, `reports/report4/member3_canonical/member3_model_run_summary.json` | Provenance Audit; all 5 project sample scopes | Documents SHA-256 hashes of 5 input files, canonical logical fingerprint `bb0aea6d...`, and exact row/cluster counts ($2,098$; $1,912$; $1,743$; $1,685/1,527$; $410/382$). |
| **Table 2** | Operational Definitions, Physical Units, and Role of Lyrical and Audio-Mode Variables | `src/lyrics/extract_lyrics_features.py`, `src/features/prepare_features.py`, `reports/report3/lyrics_features_extracted.csv` | Methodological Specification; track-level & OLS design matrix | Explicitly distinguishes the 6 features tested in Kruskal–Wallis from the 6 lyrical predictors retained in OLS after VIF screening (`line_count` and `punctuation_ratio` replace `lexical_diversity` and `unique_word_count`). |
| **Table 3** | Complete-Case Sample Sizes by Market in the Primary Market-Exclusive Analysis | `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv` | Primary Analysis; market-exclusive tracks ($N = 1,740$ and $N = 1,601$) | Preserves `NULL` for undetected chorus blocks in primary analysis (excludes 3 `not_found` tracks for lexical features and 142 tracks for chorus). |
| **Table 4** | Primary Kruskal–Wallis Test Results, Holm–Bonferroni Adjusted $p$-Values, and $\varepsilon^2$ Effect Sizes | `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv` | Primary Inference; market-exclusive tracks ($N = 1,740$ / $1,601$) | Core inferential table for Member 3. Flags the exact algebraic identity $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$ ($H = 147.7139$). |
| **Table 5** | Cross-Market Descriptive Statistics (Median, IQR, Mean, SD) for the Six Lyrics Features Across Exclusive ($N = 1,743$) and All-Market ($N = 2,098$) Scopes | `data/processed/modeling_dataset_clean_cache.csv`, `reports/report3/lyrics_features_extracted.csv`, `reports/report3/lyrics_statistics.csv`, `reports/report4/chart_data/member3/01_word_count_by_market.csv`, `03_repetition_ratio_by_market.csv` | Descriptive Statistics; exclusive ($N = 1,740/1,601$) & all-market ($N = 2,095/1,943$) | Provides exact parametric (Mean, SD) and non-parametric (Median, IQR) summaries to establish directional market ordering without fabricating unexecuted post-hoc tests. |
| **Table 6** | All-Market Rank-Based GEE Sensitivity Analysis: Sample Sizes, Market Contrast Estimates, and Omnibus Wald Tests | `reports/report4/member3_phase2_4/results_final_audit/04_all_market_rank_gee_sensitivity.csv` | Sensitivity Analysis; all-market rows clustered by `spotify_track_id` ($N = 2,095/1,943$; $1,909/1,764$ clusters) | Reports VN vs. US and KR vs. US fractional rank contrasts, omnibus Wald $\chi^2(2)$, raw $p$, and Holm $p$. Notes that CIs are not exported by the source CSV. |
| **Table 7** | All-Market Rank-Based GEE Sensitivity Analysis: Model Configuration and Convergence Diagnostics | `reports/report4/member3_phase2_4/results_final_audit/04_all_market_rank_gee_sensitivity.csv` | Sensitivity Diagnostics; all-market Rank-GEE | Confirms convergence in 2 iterations, finite sandwich covariance, and full contrast rank $= 2$ for all 6 models. |
| **Table 8** | Chorus Zero-Fill Sensitivity Analysis: Sample Sizes and Zero-Filled Counts by Market | `reports/report4/member3_phase2_4/results_final_audit/03_chorus_zero_fill_sensitivity.csv` | Sensitivity Analysis; market-exclusive valid lyrics ($N = 1,740$) | Documents the 139 zero-filled chorus observations by market (`VN`: 27, `US`: 92, `KR`: 20). |
| **Table 9** | Comparison of Primary Complete-Case vs. Sensitivity Zero-Fill Kruskal–Wallis Results for `chorus_repetition_count` | `reports/report4/member3_phase2_4/results_final_audit/01_primary_kruskal_wallis.csv`, `03_chorus_zero_fill_sensitivity.csv` | Primary vs. Sensitivity Comparison ($N = 1,601$ vs. $N = 1,740$) | Explains why imputing `NULL -> 0` shifts $p$ from $0.3060$ ($\varepsilon^2 = 0.0002$) to $0.0212$ ($\varepsilon^2 = 0.0033$) due to higher US missingness. |
| **Table 10** | Reconciliation of `chorus_repetition_count` Missingness Across Observation Units | `reports/report4/member3_phase2_4/results_final_audit/06_chorus_missingness_reconciliation.csv` | Descriptive Reconciliation ($N = 1,912$, $N = 2,098$, $N = 1,743$) | Reconciles unique-track missingness ($148$), all-market row missingness ($155$), and exclusive-track missingness ($142 = 3\text{ not\_found} + 139\text{ zero-filled}$). |
| **Table 11** | Primary Missingness Association: Pearson Chi-Square Test of Chorus Missingness by Market on Exclusive Tracks | `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv` | Primary Missingness Inference; exclusive tracks ($N = 1,743$) | Valid independent Pearson $\chi^2(2) = 35.8621, p = 1.63 \times 10^{-8}$ (`US` missing rate $12.76\%$ vs. `VN` $5.74\%$ and `KR` $4.12\%$). |
| **Table 12** | Diagnostic All-Market Missingness Contingency Table ($N = 2,098$ Repeated-Track Market Rows) | `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv` | Diagnostic Only; all-market rows ($N = 2,098$) | Clearly labeled as diagnostic only ($\chi^2(2) = 39.6862, p = 2.41 \times 10^{-9}$) because repeated tracks violate row independence. |
| **Table 13** | Cluster-Robust Logistic GEE for Chorus Missingness Across All Market Observations | `reports/report4/member3_phase2_4/results_final_audit/07_missingness_association.csv` | Sensitivity Missingness Inference; all-market rows ($N = 2,098$, $1,912$ clusters) | Reports log-odds, Odds Ratios ($\text{OR}_{\text{VN vs US}} = 0.4241$, $\text{OR}_{\text{KR vs US}} = 0.2996$), omnibus $\text{Wald }\chi^2(2) = 36.0346, p = 1.50 \times 10^{-8}$, and convergence diagnostics. |
| **Table 14** | Step-by-Step Multicollinearity (VIF) Audit Across Predictor Sets and Transformations on the Canonical Training Split | `reports/vif_before.csv`, `reports/vif_after.csv`, `reports/report4/member3_phase3/vif_audit_evidence.csv` | OLS Multicollinearity Audit; Train split ($N = 1,685$) | Proves that on a fixed predictor set and transformation (Stage 0 $\to$ Stage 1 $\to$ Stage 2), dropping `lexical_diversity_std` leaves `word_count_std` ($16.4196$) and `unique_word_count_std` ($14.8026$) unchanged, and explains why applying `log1p` to both counts in an auxiliary 7-var check produced $70.8472$ and $52.9955$. |
| **Table 15** | Distributional Shape and OLS Leverage/Influence Diagnostics for Raw `word_count` vs. $\ln(1 + \text{word\_count})$ | `reports/report4/member3_phase3/member3_phase3_audit_summary.json`, `reports/diagnostics/diagnostics_summary.json`, `reports/diagnostics/cooks_top10.csv` | Transformation & Influence Audit; Train ($N = 1,685$), Exclusive ($N = 1,740$), All-Market ($N = 2,095$) | Documents Mean, SD, Median, IQR, Skewness, Excess Kurtosis, D'Agostino $K^2$, and exact Model 2 hat-matrix leverage ($h_{ii}$) and Cook's distance ($D_i$) before and after $\ln(1 + \text{word\_count})$. |
| **Table 16** | Primary $3 \times 2$ Contingency Table and Pearson Chi-Square Test of `audio_mode` by Market on Market-Exclusive Tracks | `reports/report4/member3_phase2_4/results_final_audit/02_primary_audio_mode_chi_square.csv` | Primary Audio Analysis; market-exclusive tracks ($N = 1,743$) | Reports observed counts, expected counts ($\ge 195.98$), market shares, $\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}$, and $\text{Cramér's V} = 0.1181$. |
| **Table 17** | Cluster-Robust Logistic GEE Sensitivity Analysis for `audio_mode` on All Market Observations | `reports/report4/member3_phase2_4/results_final_audit/05_audio_mode_logistic_gee.csv` | Sensitivity Audio Analysis; all-market rows ($N = 2,098$, $1,912$ clusters) | Reports log-odds, Odds Ratios ($\text{OR}_{\text{VN vs US}} = 1.6456$, $\text{OR}_{\text{KR vs US}} = 1.1239$), omnibus $\text{Wald }\chi^2(2) = 21.7232, p = 1.92 \times 10^{-5}$, and convergence status. |
| **Table 18** | Comprehensive Reconciliation of `audio_mode` Contingency Counts and Hypothesis Tests Across All Four Analytical Scopes | `02_primary_audio_mode_chi_square.csv`, `05_audio_mode_logistic_gee.csv`, `08_historical_audio_mode_reconciliation.csv`, `data/processed/modeling_dataset_clean_cache.csv` | Multi-Scope Reconciliation ($N = 1,743$; $N = 2,098$; $N = 2,860$) | Reconciles `1,043/700` ($N=1,743$), `1,275/823` ($N=2,098$), and `1,704/1,156` ($N=2,860$, $\chi^2 = 11.6962, p = 0.0029$), while documenting `p = 0.42` and `χ² = 0.51, p = 0.774` as unverified legacy claims. |
| **Table 19** | Cluster-Robust OLS Estimates for `mode` and the Six Lyrical Predictors Across Canonical Model 1, Canonical Model 2, and Full Model 3 | `reports/report4/member3_canonical/member3_model_coefficients.csv`, `reports/model3_coefficients.csv` | OLS Regression Inference; Train split ($N = 1,685$, $1,527$ track clusters) | Reports $\hat{\beta}$, cluster-robust $SE$, $t$-stat, $p$-value, 95% CI, VIF, and coefficient scope (pooled main effect in Model 2 vs. US reference slope for interacted features in Model 3). |
| **Table 20** | Joint Wald Block Tests, In-Sample Fit ($N = 1,685$), and Out-of-Sample Test Performance ($N = 410$) Across the OLS Model Hierarchy | `reports/report4/member3_canonical/member3_lyrics_joint_wald.json`, `member3_model_comparison.csv`, `member3_model_test_metrics.csv`, `reports/model_wald_blocks.csv`, `member3_phase3_audit_summary.json` | OLS Block Inference & Model Evaluation; Train ($N = 1,685$) & Test ($N = 410$) | Reconciles all three lyrics block Wald tests ($\chi^2(6) = 15.5722, p = 0.0162$ without controls; $\chi^2(6) = 13.5932, p = 0.0345$ in canonical Model 2 with controls; $\chi^2(6) = 11.7440, p = 0.0679$ in Model 3) plus the auxiliary `mode × Market` Wald test ($\chi^2(2) = 2.4278, p = 0.2970$). |

---

## 3. Column Mapping & Number Formatting Conventions

**Table A2. Mapping of Technical CSV/JSON Field Names to Academic Report Labels and Rounding Rules.**

| Source CSV / JSON Column Name | Academic Report Column Label | Formatting & Rounding Rule |
|---|---|---|
| `n_vn`, `n_us`, `n_kr`, `n_total`, `n`, `market_total`, `total_n`, `rows`, `N` | Sample Size (`N`, `n`), Market Total (`N`), Market Rows (`N`) | Exact integer with comma thousands separator (e.g., `1,743`, `2,098`) |
| `clusters`, `train_unique_track_ids`, `test_unique_track_ids` | Track Clusters (`clusters`) | Exact integer with comma thousands separator (e.g., `1,912`, `1,527`, `382`) |
| `observed`, `zero_filled_n`, `missing_chorus_n`, `vn_missing`, `us_missing`, `kr_missing` | Observed Count (`n`), Zero-Filled Count, Missing Chorus Count | Exact integer |
| `expected`, `min_expected` | Expected Count, Minimum Expected Cell | Fixed-point decimal to **2 decimal places** (e.g., `292.02`, `39.76`) |
| `market_percent`, `missing_percent` | Within-Market Share (`%`), Missing Rate (`%`) | Fixed-point decimal to **2 decimal places** (e.g., `68.85`, `12.76`) |
| `h_statistic` | Kruskal–Wallis $H$ | Fixed-point decimal to **4 decimal places** (e.g., `269.8847`) |
| `chi_square`, `wald_chi_square`, `omnibus_wald_chi_square`, `chi2`, `statistic` | Pearson $\chi^2$, Omnibus Wald $\chi^2$, Joint Wald $\chi^2$ | Fixed-point decimal to **4 decimal places** (e.g., `24.3142`, `13.5932`) |
| `df`, `omnibus_df`, `num_features` | Degrees of Freedom ($\text{df}$) | Exact integer |
| `p_raw`, `p_holm`, `coefficient_p`, `omnibus_p`, `p_value`, `holm` | Unadjusted $p$, Holm-Adjusted $p$, Coefficient $p$, Omnibus $p$ | If $p \ge 0.0001$: **4 decimal places** (e.g., `0.3060`, `0.0345`); if $p < 0.0001$: scientific notation with **2 decimal places** (e.g., `2.48e-59`, `5.25e-06`) |
| `epsilon_squared` | Effect Size ($\varepsilon^2$) | Fixed-point decimal to **4 decimal places** (e.g., `0.3196`, `0.0002`) |
| `cramers_v` | Cramér's V | Fixed-point decimal to **4 decimal places** (e.g., `0.1181`) |
| `vn_vs_us_coefficient`, `kr_vs_us_coefficient` | Contrast Estimate: VN vs. US, KR vs. US | Signed decimal to **4 decimal places** (e.g., `+0.0753`, `-0.1520`) |
| `coefficient_log_odds` | Log-Odds Coefficient ($\hat{\beta}$) | Signed decimal to **4 decimal places** (e.g., `+0.4981`, `-0.8578`) |
| `odds_ratio` | Odds Ratio ($\text{OR}$) | Fixed-point decimal to **4 decimal places** (e.g., `1.6456`, `0.4241`) |
| `coefficient`, `coef` | OLS Estimate ($\hat{\beta}$) | Signed decimal to **4 decimal places** (e.g., `+3.4233`, `-2.0718`) |
| `cluster_se`, `bse_cluster` | Cluster-Robust $SE$ | Fixed-point decimal to **4 decimal places** (e.g., `2.1304`, `1.9776`) |
| `ci_lower`, `ci_upper`, `ci_95_lower`, `ci_95_upper` | 95% Confidence Interval (`[ci_lower, ci_upper]`) | Signed decimal to **4 decimal places** (e.g., `[-0.7521, +7.5987]`) |
| `vif_initial`, `vif_final`, `vif` | Variance Inflation Factor ($\text{VIF}$) | Fixed-point decimal to **4 decimal places** (`1.0008e+15` for singular matrix) |

---

## 4. Index of Referenced Figures & Underlying Chart Data CSVs

All three figures embedded in `member3_phase3_report.md` were visually inspected and cross-checked against their underlying CSV chart-data files.

**Table A3. Referenced Visual Figures, Underlying Chart Data CSVs, and Visual Audit Notes.**

| Figure No. | Figure Subject | Relative Path to PNG Image | Relative Path to Underlying Chart CSV | Mandatory Methodological & Visual Audit Notes |
|---|---|---|---|---|
| **Figure 1** | Cross-market boxplot of `word_count` | `reports/report4/figures/member3/01_boxplot_word_count_by_market.png` | `reports/report4/chart_data/member3/01_word_count_by_market.csv` | Displays all valid track-market rows ($N = 2,095$: VN $607$, US $797$, KR $691$). In-plot annotation (`Median` and `Mean`: VN `396.0` and `432.7`, US `358.0` and `396.9`, KR `298.0` and `307.1`) matches CSV 100%. |
| **Figure 2** | Cross-market boxplot of `repetition_ratio` | `reports/report4/figures/member3/03_boxplot_repetition_ratio_by_market.png` | `reports/report4/chart_data/member3/03_repetition_ratio_by_market.csv` | Displays all valid track-market rows ($N = 2,095$). In-plot annotation (`Median` and `Mean`: VN `0.626` and `0.617`, US `0.629` and `0.620`, KR `0.576` and `0.556`) matches CSV 100%. Note algebraic identity `1 - lexical_diversity`. |
| **Figure 3** | 100% stacked bar chart of `language` by `market` | `reports/report4/figures/member3/09_stacked_language_by_market.png` | `reports/report4/chart_data/member3/09_language_distribution_by_market.csv` | Displays all $N = 2,098$ market rows (VN $608$, US $797$, KR $693$). All percentage and count labels match CSV 100%. Minor cosmetic note: at the top of the `KR` bar, the label `other/unknown: 5.1% (n=35)` extends slightly beyond the bar width but remains fully legible and does not overlap the legend. |
