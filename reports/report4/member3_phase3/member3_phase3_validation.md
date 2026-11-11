# Statistical Validation & Technical Audit Report — Member 3 (TV3) Phase 3

**Course:** ADY201m — Applied Data Science Project  
**Project Title:** *What Makes a Hit Song? Comparing Factors Affecting Music Popularity Across Vietnam, the United States, and South Korea (2021–2026)*  
**Primary Report Audited:** `reports/report4/member3_phase3/member3_phase3_report.md`  
**Supporting Guide Audited:** `reports/report4/member3_phase3/member3_phase3_data_and_table_guide.md`  
**Reproducible Audit Script:** `reports/report4/member3_phase3/verify_member3_phase3_audit.py`  
**Audit Evidence Outputs:** `reports/report4/member3_phase3/vif_audit_evidence.csv`, `reports/report4/member3_phase3/member3_phase3_audit_summary.json`

---

## 1. Scope & Safety Compliance Check

This validation report documents the comprehensive statistical, methodological, and technical verification of Member 3's finalized Phase 3 deliverables. All work was executed strictly on branch `feature/member3-cp3-simulator` without switching branches, committing, pushing, merging, or modifying Git history. No raw data, canonical caches (`data/processed/modeling_dataset_clean_cache.csv`), Phase 2.4 scripts/outputs, or other members' files were modified.

**Table V1. Provenance, Artifact Integrity, and Reproducibility Checks.**

| Verification Item | Audited Target / File Path | Verification Result | Status |
|---|---|---|---|
| Git branch & working tree safety | `feature/member3-cp3-simulator` (`git diff --check`) | Clean formatting; only `reports/report4/member3_phase3/` updated | **PASS** |
| Phase 2.4 reproduction script | `reports/report4/member3_phase2_4/reproduce_phase2_4.py` | Exists and verified against `run_log.json` | **PASS** |
| Phase 2.4 audit CSVs (9 files) | `reports/report4/member3_phase2_4/results_final_audit/*.csv` | 9/9 SHA-256 byte-hashes match `results_final/` | **PASS** |
| Automated Phase 2.4 validation suite | `reports/report4/member3_phase2_4/results_final_audit/run_log.json` | `119 PASS, 0 FAIL` | **PASS** |
| Canonical dataset rows & unique tracks | `data/processed/modeling_dataset_clean_cache.csv` | `2,098` market rows / `1,912` unique tracks / `1,743` exclusive tracks | **PASS** |
| Canonical logical fingerprint | `bb0aea6d42049da728c8339bbf92388991825d9128cb85122df081e71a0a6e81` | Exact match across cache, Phase 2.4, and Model 2 runner | **PASS** |
| Canonical Train/Test split (`seed = 43`) | `src/features/prepare_features.py` (`get_split()`) | Train: `1,685` rows (`1,527` tracks); Test: `410` rows (`382` tracks); 0 overlap | **PASS** |
| Phase 3 VIF & Leverage Audit Script | `reports/report4/member3_phase3/verify_member3_phase3_audit.py` | Executes cleanly (`exit code 0`), writes `vif_audit_evidence.csv` & JSON | **PASS** |
| Referenced figure paths & CSV chart data | `reports/report4/figures/member3/*.png` & `chart_data/member3/*.csv` | All 3 PNGs and 3 CSVs exist and match 100% | **PASS** |

---

## 2. Mandatory Statistical Audit 2.1 — Multicollinearity (VIF) Investigation

### 2.1.1. The Apparent Anomaly Investigated
In an earlier draft comparison table, the VIF values for the lyrics features were juxtaposed across columns as follows:
- `lexical_diversity_std`: `1.0008 × 10^15 -> Dropped`
- `repetition_ratio_std`: `1.0008 × 10^15 -> 21.6134 -> 1.7670 -> 1.7669`
- `unique_word_count_std`: `14.8224 -> 52.9955 -> Dropped`
- `word_count_std`: `16.4383 -> 70.8472 -> 3.7637 -> 3.7737`

**Statistical Question:** Why did `unique_word_count_std` appear to jump from `14.8224` to `52.9955` and `word_count_std` jump from `16.4383` to `70.8472` after removing `lexical_diversity_std`, given the mathematical theorem that removing a predictor from the same OLS design matrix can never increase the VIF of any remaining predictor?

### 2.1.2. Exact Root Cause Proven by Code and Data Audit
By inspecting commit `d458f2c` (`feat(tv1): finalize FEATURE_SET_V1 after VIF screening`), `reports/vif_before.csv`, `src/features/prepare_features.py`, and running `reports/report4/member3_phase3/verify_member3_phase3_audit.py` on the canonical Training split ($N = 1,685$), we proved that the two columns came from **two different predictor matrices with different functional transformations**:

1. **How `reports/vif_before.csv` (`16.4383` and `14.8224`) was computed:**  
   Computed on an **18-predictor initial matrix** (`8` candidate lyrics features + `7` Librosa audio features + `release_year_std` + `Market_KR` + `Market_VN`, $N = 1,685$) in which **both `word_count` and `unique_word_count` were in raw integer token counts (un-logged)**, and `lexical_diversity` and `repetition_ratio` were merged from `reports/report3/lyrics_features_extracted.csv` (rounded to 4 decimal places). Because $\text{repetition\_ratio} = 1 - \text{lexical\_diversity}$, the matrix was rank-deficient ($\text{VIF} = 1,000,799,917,193,443.5$).
2. **What happens on the exact same 18-predictor matrix (raw counts) when `lexical_diversity_std` is removed (Stage 0 $\to$ Stage 1):**  
   When `lexical_diversity_std` is dropped while keeping the exact same remaining 17 predictors and raw token counts (`verify_member3_phase3_audit.py`), $\text{VIF}(\text{repetition\_ratio\_std})$ drops from $1.0008 \times 10^{15}$ to **`4.3372`**, while **`word_count_std` remains unchanged at `16.4196`** (`16.419642 -> 16.419642`) and **`unique_word_count_std` remains unchanged at `14.8026`** (`14.802630 -> 14.802634`, identical up to floating-point machine precision of the singular Stage 0 pseudoinverse). Removing `unique_word_count_std` next (Stage 1 $\to$ Stage 2) drops $\text{VIF}(\text{word\_count\_std})$ from `16.4196` to **`3.4030`** and $\text{VIF}(\text{repetition\_ratio\_std})$ from `4.3372` to **`1.4740`**. **No VIF ever increases when removing a predictor on the same matrix.**
3. **Why the auxiliary 7-predictor column showed `70.8472`, `52.9955`, and `21.6134`:**  
   In that ad-hoc 7-feature lyrics-only calculation, the script applied `np.log1p` ($\ln(1 + x)$) to **both** `word_count` and `unique_word_count`. Because $\text{unique\_word\_count} = \text{word\_count} \times (1 - \text{repetition\_ratio})$, taking logarithms transforms the nonlinear multiplicative relation into an almost exact linear additive identity:
   $$\ln(1 + \text{unique\_word\_count}) \approx \ln(1 + \text{word\_count}) + \ln(1 - \text{repetition\_ratio}).$$
   Since $\ln(1 - \text{repetition\_ratio})$ is nearly linear over the empirical range of `repetition_ratio` ($\text{IQR} \approx [0.55, 0.68]$), log-transforming both counts dramatically increases their mutual linear $R^2$, raising their VIFs in the 7-variable log-transformed block to `70.8472` (`word_count_std`), `52.9955` (`unique_word_count_std`), and `21.6134` (`repetition_ratio_std`). By contrast, on the 7-variable lyrics-only block with **raw** token counts, the VIFs are `16.1330`, `14.3401`, and `4.1577`.

**Table V2. Complete Reproducible VIF Audit Evidence (`reports/report4/member3_phase3/vif_audit_evidence.csv`, Train $N = 1,685$).**

| Feature (`feature`) | `reports/vif_before.csv` (18 Vars, Raw Counts, Rounded TTR) | Stage 0: 18 Vars, Raw Counts (Exact Unrounded TTR) | Stage 1: Drop Only `lexical_diversity_std` (17 Vars, Raw Counts) | Stage 2: Also Drop `unique_word_count_std` (16 Vars, Raw Counts) | Stage 3A: `reports/vif_after.csv` (24 Vars, `log1p(word_count)`, `loudness_proxy`) | Stage 3B: Canonical Model 2 (24 Vars, `log1p(word_count)`, `energy`) | 7 Lyrics Vars Only (Raw Token Counts) | 7 Lyrics Vars Only (`log1p` on Both `word_count` & `unique_word_count`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical_diversity_std` | 1.0008e+15 | 1.0008e+15 | *Dropped* | *Dropped* | — | — | — | — |
| `repetition_ratio_std` | 1.0008e+15 | 1.0008e+15 | 4.3372 | 1.4740 | 1.7670 | 1.7669 | 4.1577 | 21.6134 |
| `unique_word_count_std` | 14.8224 | 14.8026 | 14.8026 | *Dropped* | — | — | 14.3401 | 52.9955 |
| `word_count_std` | 16.4383 | 16.4196 | 16.4196 | 3.4030 | 3.7637 | 3.7737 | 16.1330 | 70.8472 |
| `line_count_std` | 3.1107 | 3.1336 | 3.1336 | 3.0489 | 2.7497 | 2.7631 | 2.7434 | 2.4503 |
| `punctuation_ratio_std` | 1.3420 | 1.3482 | 1.3482 | 1.3464 | 1.3680 | 1.3776 | 1.0803 | 1.0714 |
| `chorus_repetition_count_std` | 1.3455 | 1.3429 | 1.3429 | 1.2993 | 1.2991 | 1.2983 | 1.3283 | 1.2828 |
| `avg_word_length_std` | 1.1894 | 1.1883 | 1.1883 | 1.1368 | 1.2137 | 1.2150 | 1.0692 | 1.0802 |

---

## 3. Mandatory Statistical Audit 2.2–2.6 — Summary of Verification Results

### 3.1. Word-Count Transformation & Leverage/Cook's Distance Audit (Part 2.2)
- **Verified Distributional Metrics (Train $N = 1,685$):** Raw `word_count` has Mean $= 378.8510$, SD $= 178.8282$, Median $= 348.0000$, IQR $= 186.0000$, Skewness $= +1.5765$, Excess Kurtosis $= +6.4205$, D'Agostino–Pearson $K^2 = 608.6408$ ($p = 6.84 \times 10^{-133}$). Transformed $\ln(1 + \text{word\_count})$ has Mean $= 5.8303$, SD $= 0.4906$, Median $= 5.8551$, IQR $= 0.5211$, Skewness $= -0.7931$, Excess Kurtosis $= +2.6455$, D'Agostino–Pearson $K^2 = 249.1376$ ($p = 7.95 \times 10^{-55}$).
- **Verified Leverage & Cook's Distance Diagnostics (`member3_phase3_audit_summary.json`):** Rather than making unverified claims about leverage, `verify_member3_phase3_audit.py` explicitly computes hat-matrix diagonals $h_{ii}$ and Cook's distances $D_i$ in the 24-predictor Model 2 ($p = 25$ parameters including intercept; $2p/N = 0.0297$, $3p/N = 0.0445$, $4/N = 0.0024$). On the $9$ extreme tracks with $\text{word\_count} \ge 1,000$ tokens, $\ln(1 + \text{word\_count})$ reduces maximum univariate leverage from $0.0485$ to $0.0282$ ($-41.8\%$), reduces maximum Model 2 hat-matrix leverage $h_{ii}$ from $0.0764$ to $0.0502$ ($-34.3\%$), reduces mean $h_{ii}$ on those tracks from $0.0366$ (above $2p/N$) to $0.0253$ (below $2p/N$), and reduces their maximum Cook's distance from $0.0032$ (above $4/N$) to $0.0018$ (below $4/N$).

### 3.2. Kruskal–Wallis & Feature-List Alignment Audit (Part 2.3)
- **Verified Exact Alignment:** Table 2, Table 3, and Table 4 in `member3_phase3_report.md` explicitly distinguish the **6 features tested in Kruskal–Wallis** (`word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, `chorus_repetition_count`) from the **6 predictors retained in OLS** (`word_count_std`, `repetition_ratio_std`, `avg_word_length_std`, `line_count_std`, `chorus_repetition_count_std`, `punctuation_ratio_std`).
- **Post-Hoc Status:** Verified that `reproduce_phase2_4.py` did not run Dunn's or Mann–Whitney pairwise post-hoc tests; `member3_phase3_report.md` explicitly states this and grounds directional comparisons in Table 5 (Medians/IQRs/Means/SDs) and Table 6 (Rank-GEE contrasts).

### 3.3. Audio-Mode Multi-Scope Reconciliation Audit (Part 2.4)
- **Verified Across All 4 Scopes (Table 16, Table 17, Table 18):**
  1. Primary Market-Exclusive ($N = 1,743$): `Major = 1,043 (59.84%)`, `Minor = 700 (40.16%)`; Pearson $\chi^2(2) = 24.3142, p = 5.25 \times 10^{-6}, \text{Cramér's V} = 0.1181$.
  2. All-Market Cluster-Robust Logistic GEE ($N = 2,098$, $1,912$ clusters): `Major = 1,275 (60.77%)`, `Minor = 823 (39.23%)`; $\text{Wald }\chi^2(2) = 21.7232, p = 1.92 \times 10^{-5}$.
  3. All-Market Naive Pearson Diagnostic ($N = 2,098$): $\chi^2(2) = 20.5006, p = 3.54 \times 10^{-5}$ (identifies the exact source of the `1,275 / 823` marginal counts while flagging repeated-track violation).
  4. Historical Report 4 EDA ($N = 2,860$): `Major = 1,704 (59.58%)`, `Minor = 1,156 (40.42%)`; Pearson $\chi^2(2) = 11.6962, p = 0.0029$. Legacy claims `p = 0.42` and `χ² = 0.51, p = 0.774` are documented as `NOT VERIFIED`.

### 3.4. OLS Specifications & `market × mode` Interaction Audit (Parts 2.5 & 2.6)
- **Verified Auxiliary Interaction Test:** Adding `mode__x__VN` and `mode__x__KR` to Model 3 on Train ($N = 1,685$, $1,527$ clusters) yields $\text{Wald }\chi^2(2) = 2.4278, p = 0.2970$ (`mode__x__VN`: $\hat{\beta} = -7.3935, SE = 4.9270, p = 0.1335$; `mode__x__KR`: $\hat{\beta} = -1.7713, SE = 4.6697, p = 0.7045$).
- **Verified Lyrics Block Wald Tests Across Specifications:**
  - Model 2 without controls (`Audio + Lyrics`, 21 predictors): $\text{Wald }\chi^2(6) = 15.5722, p = 0.0162$.
  - Canonical Model 2 with controls (`Audio + Lyrics + Controls`, 24 predictors): $\text{Wald }\chi^2(6) = 13.5932, p = 0.0345$.
  - Full Model 3 (`34 predictors` with 10 market interactions): 6-DOF lyrics main-effect $\text{Wald }\chi^2(6) = 11.7440, p = 0.0679$ ($p_{\text{Holm}} = 0.1131$); 12-DOF full lyrics family $\text{Wald }\chi^2(12) = 19.6847, p = 0.0733$.

---

## 4. Remaining Open Project-Level Notes

1. **External Assignment Specification:** No standalone `ADY201m.docx` or `ADY201m.pdf` syllabus file is stored inside the repository; page/word limits and citation style should be checked against the lecturer's course handout during final team compilation.
2. **Unverified Legacy Draft Numbers:** As documented in `08_historical_audio_mode_reconciliation.csv` and Table 18, the historical Report 4 draft value `p = 0.42` and the early planning target `χ² = 0.51, df = 2, p = 0.774` do not correspond to any dataset in the repository and remain formally classified as `NOT VERIFIED`.
3. **Readiness for Lecturer Review:** All statistical calculations, units of analysis, VIF progressions, log-transformation leverage diagnostics, contingency tables, and OLS specifications in `member3_phase3_report.md` have been independently reproduced and verified. **The Member 3 Phase 3 report is 100% ready for academic submission and lecturer review.**
