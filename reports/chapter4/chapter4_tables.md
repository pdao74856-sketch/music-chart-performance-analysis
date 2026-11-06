# Báo Cáo Thống Kê & Bảng Dữ Liệu Chương 4 (Kết Quả Mô Hình Hóa)

## Bảng T1: Thiết lập Nghiên Cứu và Phân Chia Tập Dữ Liệu
Nguồn: `view_clean_modeling_dataset`, `reports/model_comparison_table.csv`, `models/train_track_ids.json`, `models/test_track_ids.json`

| Hạng mục thiết lập | Tập Train | Tập Test | Kiểm tra / Dữ liệu |
| :--- | :---: | :---: | :--- |
| Số quan sát (rows) | 1,685 | 410 | Tổng 2.095 rows (sau khi loại 3 bài instrumental) |
| Số ca khúc duy nhất (unique tracks) | 1,527 | 382 | Số track giao nhau giữa 2 tập (kiểm bằng code): 0 |
| Random Seed | 43 | 43 | Đọc từ `train_track_ids.json` và `test_track_ids.json` |
| Miền giá trị mục tiêu (`chart_score`) | [-99.000, 100.000] | [-94.000, 99.152] | Điểm chuẩn hóa từ hạng (101 - rank); tỉ lệ điểm <= 0 trên 2.098 dòng của view: 33.4% (701/2.098) |
| Số tham số M1 (Audio Baseline) | 16 | 16 | 15 đặc trưng âm học + hằng số |
| Số tham số M2 (Audio + Lyrics) | 22 | 22 | 15 audio + 6 ca từ + hằng số |
| Số tham số M2b (+ Context) | 25 | 25 | 21 đặc trưng + release_year + 2 dummy thị trường (VN, KR) + hằng số |
| Số tham số M3 (Full OLS + Interactions) | 35 | 35 | 24 đặc trưng + 10 tương tác thị trường + hằng số |

---

## Bảng T2: Độ Phù Hợp của Các Mô Hình Phân Cấp Trên Tập Huấn Luyện (In-Sample Goodness of Fit)
Nguồn: `reports/model_comparison_table.csv`  
*Ghi chú: TRAIN, không phải ngoài mẫu.*  

| Mô hình | Số biến ($p$) | $R^2$ | $R^2$ hiệu chỉnh | AIC | BIC | $\Delta R^2$ | Kiểm định lồng nhau | Wald $\chi^2$ | df | $p$-value Wald |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| M1 (Audio Baseline) | 15 | 0.016 | 0.008 | 17146.721 | 17233.593 | 0.000 | N/A (Baseline) | N/A | N/A | N/A |
| M2 (Audio + Lyrics) | 21 | 0.026 | 0.013 | 17142.946 | 17262.395 | 0.009 | M1 -> M2 (Lyrics 6 DOF) | 15.572 | 6 | 0.016 |
| M2b (+ Context) | 24 | 0.056 | 0.043 | 17095.256 | 17230.994 | 0.031 | M2 -> M2b (Context 3 DOF) | 38.147 | 3 | < 0.001 |
| M3 (Full + Interactions) | 34 | 0.066 | 0.046 | 17098.347 | 17288.381 | 0.009 | M2b -> M3 (Interactions 10 DOF) | 17.906 | 10 | 0.057 |

---

## Bảng T3: Hiệu Năng Dự Báo Ngoài Mẫu Trên Tập Kiểm Tra Độc Lập (Out-of-Sample Test Set)
Nguồn: `reports/model_test_comparison.csv`

| Mô hình | Mẫu đánh giá | $n$ | $R^2$ Test | 95% CI $R^2$ | MAE | 95% CI MAE | RMSE | 95% CI RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: Global Train Mean | Overall | 410 | -0.001 | [-0.0196, -0.0000] | 30.367 | [28.01, 32.94] | 38.992 | [36.33, 41.94] |
| Baseline: Global Train Mean | Market_US | 165 | -0.010 | [-0.0734, -0.0000] | 28.977 | [25.40, 32.49] | 36.706 | [32.45, 40.66] |
| Baseline: Global Train Mean | Market_KR | 140 | -0.001 | [-0.0480, -0.0000] | 28.004 | [24.10, 32.10] | 36.733 | [31.94, 41.39] |
| Baseline: Global Train Mean | Market_VN | 105 | -0.003 | [-0.0593, -0.0000] | 35.703 | [30.77, 40.82] | 44.949 | [39.78, 50.25] |
| Baseline: Market Train Mean | Overall | 410 | 0.002 | [-0.0229, 0.0163] | 30.267 | [27.91, 32.85] | 38.937 | [36.29, 41.87] |
| Baseline: Market Train Mean | Market_US | 165 | -7.218e-06 | [-0.0324, -0.0000] | 28.246 | [24.46, 31.85] | 36.531 | [32.15, 40.71] |
| Baseline: Market Train Mean | Market_KR | 140 | -0.006 | [-0.0657, -0.0000] | 28.379 | [24.56, 32.51] | 36.827 | [32.06, 41.59] |
| Baseline: Market Train Mean | Market_VN | 105 | -6.482e-05 | [-0.0532, -0.0000] | 35.961 | [31.05, 41.13] | 44.886 | [39.72, 49.98] |
| Model 1 (Audio Baseline) | Overall | 410 | -0.001 | [-0.0326, 0.0241] | 30.634 | [28.33, 33.13] | 38.997 | [36.28, 41.92] |
| Model 1 (Audio Baseline) | Market_US | 165 | -0.015 | [-0.1007, 0.0290] | 29.297 | [25.85, 32.78] | 36.804 | [32.65, 40.94] |
| Model 1 (Audio Baseline) | Market_KR | 140 | -0.003 | [-0.0656, 0.0393] | 28.179 | [24.39, 32.27] | 36.770 | [32.03, 41.70] |
| Model 1 (Audio Baseline) | Market_VN | 105 | 0.004 | [-0.0632, 0.0450] | 36.006 | [31.23, 41.12] | 44.799 | [39.56, 50.05] |
| Model 2 (Audio + Lyrics) | Overall | 410 | -0.002 | [-0.0374, 0.0290] | 30.786 | [28.48, 33.23] | 39.004 | [36.28, 41.86] |
| Model 2 (Audio + Lyrics) | Market_US | 165 | -0.020 | [-0.1053, 0.0330] | 29.329 | [25.91, 32.74] | 36.891 | [32.75, 41.05] |
| Model 2 (Audio + Lyrics) | Market_KR | 140 | -0.006 | [-0.0819, 0.0461] | 28.456 | [24.55, 32.50] | 36.829 | [32.21, 41.56] |
| Model 2 (Audio + Lyrics) | Market_VN | 105 | 0.011 | [-0.0592, 0.0493] | 36.184 | [31.50, 41.19] | 44.645 | [39.62, 49.72] |
| Model 2b (+ Context) | Overall | 410 | 0.036 | [-0.0151, 0.0822] | 30.057 | [27.90, 32.52] | 38.255 | [35.55, 41.05] |
| Model 2b (+ Context) | Market_US | 165 | 0.016 | [-0.0986, 0.1112] | 27.906 | [24.42, 31.70] | 36.237 | [31.36, 40.55] |
| Model 2b (+ Context) | Market_KR | 140 | 0.021 | [-0.0668, 0.0812] | 28.367 | [24.63, 32.42] | 36.331 | [31.97, 41.15] |
| Model 2b (+ Context) | Market_VN | 105 | 0.061 | [-0.0192, 0.1025] | 35.689 | [31.21, 40.24] | 43.486 | [38.75, 48.05] |
| Model 3 (Full OLS) | Overall | 410 | 0.034 | [-0.0178, 0.0824] | 29.780 | [27.48, 32.25] | 38.304 | [35.43, 41.08] |
| Model 3 (Full OLS) | Market_US | 165 | 0.028 | [-0.0926, 0.1312] | 27.639 | [24.14, 31.32] | 36.007 | [31.08, 40.47] |
| Model 3 (Full OLS) | Market_KR | 140 | 0.038 | [-0.0644, 0.1102] | 27.772 | [24.10, 31.84] | 36.003 | [31.50, 40.89] |
| Model 3 (Full OLS) | Market_VN | 105 | 0.025 | [-0.0487, 0.0726] | 35.824 | [31.13, 40.51] | 44.311 | [39.46, 49.05] |

*Ghi chú: Nguồn reports/model_test_comparison.csv; quy ước dấu trong paired bootstrap: delta_mae = MAE(tham chiếu) - MAE(model), dương = model tốt hơn; delta_r2 = R2(model) - R2(tham chiếu).*  


---

## Bảng T4: Kiểm Định Wald Đồng Thời Theo Nhóm Biến (Joint Wald Block Tests)
Nguồn: `reports/model_wald_blocks.csv`  
*Ghi chú: Audio Full / Lyrics Full là thăm dò, ngoài họ Holm.*  

| Phân loại kiểm định | Khối biến (Block) | Mô hình | Số biến (df) | Wald $\chi^2$ | $p$-value thô | $p$-value Holm | $p$ thô < 0.05? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Model 3 Core Blocks | Audio Block (Main, M3) | M3 (Full OLS) | 15 | 39.843 | < 0.001 | 0.001 | - |
| Model 3 Core Blocks | Lyrics Block (Main, M3) | M3 (Full OLS) | 6 | 11.744 | 0.068 | 0.113 | - |
| Model 3 Core Blocks | Context Block (M3) | M3 (Full OLS) | 3 | 39.274 | < 0.001 | < 0.001 | - |
| Model 3 Core Blocks | Interactions Block (M3) | M3 (Full OLS) | 10 | 17.906 | 0.057 | 0.113 | - |
| Full Feature Family (with Interactions) | Audio Full Block (Main + Interactions, 19 DOF) | M3 (Full OLS) | 19 | 44.330 | < 0.001 | - | Có |
| Full Feature Family (with Interactions) | Lyrics Full Block (Main + Interactions, 12 DOF) | M3 (Full OLS) | 12 | 19.685 | 0.073 | - | Không |
| Model Specific Joint Significance | Audio Block in M1 (15 DOF) | M1 (Audio Baseline) | 15 | 30.798 | 0.009 | - | Có |
| Model Specific Joint Significance | Lyrics Block in M2 (6 DOF) | Model 2 (Audio + Lyrics) | 6 | 15.572 | 0.016 | - | Có |

---

## Bảng T5: Hệ Số Hồi Quy Các Biến Chính Của Mô Hình 3 (Main Effects)
Nguồn: `reports/model3_coefficients.csv`  
*Ghi chú: per 1 SD (Train); riêng mode, Market_VN, Market_KR: chuyển 0 -> 1 (không chuẩn hóa).*  

| Đặc trưng (`feature`) | Hệ số $\beta$ (`coef`) | SE cụm (`bse_cluster`) | $t$-statistic | $p$-value thô | 95% CI | $p$-value Holm | Mức độ bằng chứng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `const` | 12.840 | 2.485 | 5.167 | < 0.001 | [7.970, 17.710] | - | - |
| `tempo_std` | 3.588 | 1.452 | 2.471 | 0.013 | [0.742, 6.435] | 0.270 | gợi ý |
| `onset_rate_std` | -1.495 | 1.227 | -1.218 | 0.223 | [-3.900, 0.911] | 1.000 | chưa đủ bằng chứng |
| `tempo_stability_std` | -0.023 | 1.253 | -0.018 | 0.985 | [-2.479, 2.433] | 1.000 | chưa đủ bằng chứng |
| `energy_std` | -4.622 | 1.943 | -2.378 | 0.017 | [-8.432, -0.813] | 0.313 | gợi ý |
| `rms_variability_std` | 0.704 | 1.420 | 0.496 | 0.620 | [-2.079, 3.487] | 1.000 | chưa đủ bằng chứng |
| `spectral_centroid_std` | 4.425 | 1.714 | 2.582 | 0.010 | [1.066, 7.784] | 0.206 | gợi ý |
| `chord_change_rate_std` | -3.131 | 1.420 | -2.205 | 0.027 | [-5.914, -0.348] | 0.466 | gợi ý |
| `key_confidence_std` | -2.153 | 1.120 | -1.923 | 0.054 | [-4.348, 0.041] | 0.817 | chưa đủ bằng chứng |
| `chroma_repetition_determinism_std` | -2.333 | 0.961 | -2.427 | 0.015 | [-4.217, -0.449] | 0.289 | gợi ý |
| `chroma_mean_pattern_duration_std` | 1.532 | 1.293 | 1.185 | 0.236 | [-1.002, 4.065] | 1.000 | chưa đủ bằng chứng |
| `chroma_longest_pattern_duration_std` | 0.755 | 1.697 | 0.445 | 0.656 | [-2.571, 4.080] | 1.000 | chưa đủ bằng chứng |
| `mfcc_repetition_determinism_std` | 0.170 | 1.122 | 0.152 | 0.879 | [-2.029, 2.369] | 1.000 | chưa đủ bằng chứng |
| `mfcc_mean_pattern_duration_std` | -1.244 | 1.238 | -1.005 | 0.315 | [-3.671, 1.183] | 1.000 | chưa đủ bằng chứng |
| `mfcc_longest_pattern_duration_std` | -0.254 | 1.569 | -0.162 | 0.871 | [-3.329, 2.820] | 1.000 | chưa đủ bằng chứng |
| `mode` | 3.215 | 2.135 | 1.506 | 0.132 | [-0.970, 7.399] | 1.000 | chưa đủ bằng chứng |
| `word_count_std` | -3.845 | 2.935 | -1.310 | 0.190 | [-9.598, 1.909] | 1.000 | chưa đủ bằng chứng |
| `repetition_ratio_std` | 1.881 | 2.529 | 0.744 | 0.457 | [-3.076, 6.838] | 1.000 | chưa đủ bằng chứng |
| `avg_word_length_std` | 4.292 | 3.527 | 1.217 | 0.224 | [-2.621, 11.204] | 1.000 | chưa đủ bằng chứng |
| `line_count_std` | 2.623 | 1.587 | 1.653 | 0.098 | [-0.488, 5.734] | 1.000 | chưa đủ bằng chứng |
| `chorus_repetition_count_std` | 0.893 | 1.071 | 0.834 | 0.404 | [-1.205, 2.991] | 1.000 | chưa đủ bằng chứng |
| `punctuation_ratio_std` | 2.265 | 1.154 | 1.962 | 0.050 | [0.003, 4.528] | 0.795 | gợi ý |
| `release_year_std` | 7.645 | 1.389 | 5.504 | < 0.001 | [4.922, 10.367] | < 0.001 | vững |
| `Market_VN` | -8.854 | 2.932 | -3.019 | 0.003 | [-14.601, -3.106] | 0.058 | gợi ý |
| `Market_KR` | -8.566 | 2.962 | -2.892 | 0.004 | [-14.371, -2.761] | 0.084 | gợi ý |

---

## Bảng T6: Hệ Số Các Biến Tương Tác Thị Trường Của Mô Hình 3 (Interactions Block)
Nguồn: `reports/model3_coefficients.csv`

| Biến tương tác (`feature`) | Hệ số $\lambda$ (`coef`) | SE cụm (`bse_cluster`) | $t$-statistic | $p$-value thô | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `energy_std__x__VN` | 4.693 | 2.704 | 1.736 | 0.083 | [-0.606, 9.993] |
| `energy_std__x__KR` | 3.691 | 2.443 | 1.511 | 0.131 | [-1.097, 8.478] |
| `spectral_centroid_std__x__VN` | -5.189 | 2.397 | -2.165 | 0.030 | [-9.886, -0.491] |
| `spectral_centroid_std__x__KR` | 0.445 | 2.726 | 0.163 | 0.870 | [-4.898, 5.789] |
| `avg_word_length_std__x__VN` | -6.973 | 4.244 | -1.643 | 0.100 | [-15.292, 1.346] |
| `avg_word_length_std__x__KR` | -5.280 | 3.675 | -1.437 | 0.151 | [-12.484, 1.924] |
| `word_count_std__x__VN` | 4.310 | 3.273 | 1.317 | 0.188 | [-2.105, 10.724] |
| `word_count_std__x__KR` | 0.581 | 2.973 | 0.195 | 0.845 | [-5.246, 6.409] |
| `repetition_ratio_std__x__VN` | 1.476 | 3.315 | 0.445 | 0.656 | [-5.021, 7.972] |
| `repetition_ratio_std__x__KR` | 0.399 | 2.911 | 0.137 | 0.891 | [-5.307, 6.105] |

---

## Bảng T7: Độ Dốc Thực Tế Theo Từng Thị Trường (Market-Specific Slopes)
Nguồn: `reports/model_market_slopes.csv`  
*Ghi chú: p chưa hiệu chỉnh đa kiểm định; độ dốc avg_word_length không so sánh được giữa các thị trường (ngôn ngữ khác nhau).*  

| Đặc trưng (`feature`) | Thị trường (`market`) | Độ dốc thực tế (`slope`) | SE cụm (`se_cluster`) | $t$-statistic | $p$-value | 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `energy` | US (nhóm tham chiếu) | -4.622 | 1.943 | -2.378 | 0.017 | [-8.432, -0.813] |
| `energy` | VN | 0.071 | 2.377 | 0.030 | 0.976 | [-4.587, 4.729] |
| `energy` | KR | -0.932 | 2.120 | -0.439 | 0.660 | [-5.087, 3.224] |
| `spectral_centroid` | US (nhóm tham chiếu) | 4.425 | 1.714 | 2.582 | 0.010 | [1.066, 7.784] |
| `spectral_centroid` | VN | -0.764 | 1.780 | -0.429 | 0.668 | [-4.253, 2.725] |
| `spectral_centroid` | KR | 4.870 | 2.257 | 2.158 | 0.031 | [0.447, 9.293] |
| `avg_word_length` | US (nhóm tham chiếu) | 4.292 | 3.527 | 1.217 | 0.224 | [-2.621, 11.204] |
| `avg_word_length` | VN | -2.681 | 2.506 | -1.070 | 0.285 | [-7.593, 2.231] |
| `avg_word_length` | KR | -0.988 | 1.176 | -0.841 | 0.401 | [-3.293, 1.316] |
| `word_count` | US (nhóm tham chiếu) | -3.845 | 2.935 | -1.310 | 0.190 | [-9.598, 1.909] |
| `word_count` | VN | 0.465 | 2.684 | 0.173 | 0.863 | [-4.796, 5.726] |
| `word_count` | KR | -3.264 | 2.528 | -1.291 | 0.197 | [-8.218, 1.691] |
| `repetition_ratio` | US (nhóm tham chiếu) | 1.881 | 2.529 | 0.744 | 0.457 | [-3.076, 6.838] |
| `repetition_ratio` | VN | 3.357 | 2.454 | 1.368 | 0.171 | [-1.453, 8.167] |
| `repetition_ratio` | KR | 2.280 | 1.748 | 1.305 | 0.192 | [-1.145, 5.706] |

---

## Bảng T8: Chẩn Đoán Phần Dư và Phân Tích Độ Nhạy Mô Hình (Diagnostics & Sensitivity)
Nguồn: `reports/diagnostics/diagnostics_summary.json`, `reports/diagnostics/sensitivity_pre1980.csv`, `reports/model_test_comparison.csv`

| Phép kiểm tra / Thử nghiệm độ nhạy | Thước đo thống kê | Giá trị số | Ghi chú từ file |
| :--- | :--- | :---: | :--- |
| Kiểm định Breusch–Pagan (Train) | Lagrange Multiplier (LM) stat | 84.068 | p = < 0.001 |
| Kiểm định Breusch–Pagan (Train) | F statistic | 2.548 | p = < 0.001 |
| Khoảng cách Cook (Train) | Ngưỡng tham chiếu (4/n) | 0.002 | n = 1,685 |
| Khoảng cách Cook (Train) | Giá trị Cook's d lớn nhất | 0.015 | max Cook's d |
| Khoảng cách Cook (Train) | Số quan sát vượt ngưỡng 4/n | 95 / 1,685 | n > 4/n |
| Phân tích độ nhạy (bỏ bài < 1980) | Số bài bị loại ở Train | 17 dòng | Train còn 1,668 dòng |
| Phân tích độ nhạy (bỏ bài < 1980) | Ngưỡng cắt đuôi q01 năm | 1979.840 | Giữ nguyên từ Train gốc |
| Phân tích độ nhạy (bỏ bài < 1980) | Test R² / MAE / RMSE | 0.027 / 29.663 / 38.439 | M3 gốc: 0.034 / 29.780 / 38.304 (bản lọc pre-1980 được fit lại trên dữ liệu live) |
| Phân tích độ nhạy (bỏ bài < 1980) | Wald p khối Tương tác (10 DOF) | 0.194 | M3 gốc: p = 0.057 |
| Phân tích độ nhạy (bỏ bài < 1980) | Wald p khối Lyrics (6 DOF) | 0.124 | M3 gốc: p = 0.068 |
| Phân tích độ nhạy (bỏ bài < 1980) | Dịch chuyển biến `release_year_std` | 1.481 SE | |delta/se| > 1 |

---

## Bảng T9: So Sánh Cặp Các Mô Hình Lồng Nhau (Test, Bootstrap Theo Track)
Nguồn: `reports/diagnostics/paired_bootstrap_nested.csv`  
*Ghi chú: delta_mae = MAE(tham chiếu) - MAE(model), dương = model tốt hơn; delta_r2 = R2(model) - R2(tham chiếu); CI 95% chưa hiệu chỉnh đa so sánh; pct_*_better không phải p-value.*  

| Mô hình (`model`) | Tham chiếu (`baseline`) | Mẫu đánh giá (`sample`) | $\Delta$ MAE | 95% CI $\Delta$ MAE | % MAE tốt hơn | $\Delta R^2$ | 95% CI $\Delta R^2$ | % $R^2$ tốt hơn |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Model 2 (Audio + Lyrics) | Model 1 (Audio Baseline) | Overall | -0.153 | [-0.49, +0.19] | 19.2% | -3.316e-04 | [-0.0183, +0.0163] | 50.2% |
| Model 2 (Audio + Lyrics) | Model 1 (Audio Baseline) | Market_US | -0.031 | [-0.51, +0.42] | 45.3% | -0.005 | [-0.0325, +0.0193] | 36.7% |
| Model 2 (Audio + Lyrics) | Model 1 (Audio Baseline) | Market_KR | -0.277 | [-0.98, +0.45] | 21.2% | -0.003 | [-0.0488, +0.0338] | 42.5% |
| Model 2 (Audio + Lyrics) | Model 1 (Audio Baseline) | Market_VN | -0.177 | [-0.62, +0.24] | 19.5% | 0.007 | [-0.0146, +0.0265] | 74.2% |
| Model 2b (+ Context) | Model 2 (Audio + Lyrics) | Overall | 0.729 | [+0.13, +1.38] | 99.0% | 0.038 | [+0.0014, +0.0778] | 97.8% |
| Model 2b (+ Context) | Model 2 (Audio + Lyrics) | Market_US | 1.422 | [+0.14, +2.83] | 98.2% | 0.036 | [-0.0550, +0.1343] | 78.5% |
| Model 2b (+ Context) | Model 2 (Audio + Lyrics) | Market_KR | 0.089 | [-0.46, +0.63] | 59.4% | 0.027 | [-0.0035, +0.0555] | 96.1% |
| Model 2b (+ Context) | Model 2 (Audio + Lyrics) | Market_VN | 0.494 | [-0.35, +1.37] | 86.5% | 0.051 | [+0.0005, +0.1035] | 97.6% |
| Model 3 (Full OLS) | Model 2b (+ Context) | Overall | 0.277 | [-0.10, +0.63] | 92.1% | -0.002 | [-0.0253, +0.0188] | 41.9% |
| Model 3 (Full OLS) | Model 2b (+ Context) | Market_US | 0.268 | [-0.33, +0.86] | 82.0% | 0.012 | [-0.0168, +0.0430] | 79.7% |
| Model 3 (Full OLS) | Model 2b (+ Context) | Market_KR | 0.595 | [+0.15, +1.01] | 99.4% | 0.018 | [-0.0059, +0.0402] | 91.8% |
| Model 3 (Full OLS) | Model 2b (+ Context) | Market_VN | -0.135 | [-1.04, +0.74] | 40.3% | -0.036 | [-0.0828, +0.0146] | 7.4% |
| Model 3 (Full OLS) | Model 1 (Audio Baseline) | Overall | 0.853 | [+0.08, +1.68] | 98.3% | 0.035 | [-0.0095, +0.0822] | 92.8% |
| Model 3 (Full OLS) | Model 1 (Audio Baseline) | Market_US | 1.659 | [+0.05, +3.32] | 97.9% | 0.043 | [-0.0598, +0.1525] | 80.2% |
| Model 3 (Full OLS) | Model 1 (Audio Baseline) | Market_KR | 0.407 | [-0.66, +1.34] | 78.8% | 0.041 | [-0.0171, +0.0863] | 93.0% |
| Model 3 (Full OLS) | Model 1 (Audio Baseline) | Market_VN | 0.182 | [-1.00, +1.39] | 61.6% | 0.022 | [-0.0515, +0.0918] | 70.9% |

---

