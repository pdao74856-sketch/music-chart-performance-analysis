# Báo cáo Phân tích Tương quan Đa biến & Khám phá Dữ liệu Tích hợp (Report 4)
**Học phần:** ADY201m - Applied Data Analytics  
**Dự án:** What Makes a Hit Song? (2021–2026)  
**Phân hệ:** Bivariate & Multivariate Integrated EDA  
**Biến mục tiêu Regression:** `chart_score` ($101 - \text{lifetime\_average\_rank}$)  
**Nguồn dữ liệu:** View `public.modeling_dataset` (N = 2,860 entries / 1,912 tracks; VN: 885, US: 955, KR: 1,020)

---

## 1. Tổng quan Nghiên cứu & Mục tiêu
Báo cáo này mở rộng phạm vi phân tích đơn biến sang **Phân tích Tương quan Đa biến (Multivariate Integrated EDA)**, kết nối trực tiếp các đặc trưng Âm học (Audio Features, Member 2) và Ca từ (Lyrics Features, Member 3) với các chỉ số Thành tích Bảng xếp hạng (`chart_score`, `weeks_on_chart`, `peak_rank`, Member 1) và Siêu dữ liệu (`market`, `release_year`).

Mọi nhận định trong báo cáo tuân thủ nguyên tắc **Mối liên hệ tương quan (Association), không khẳng định Quan hệ nhân quả (Causation)**, đối chiếu chuẩn xác 100% với số liệu thực nghiệm để phục vụ bài toán Mô hình hóa Hồi quy (Report 5).

---

## 2. Danh mục Biểu đồ và Dữ liệu Nguồn Tích hợp

| STT | Tên Biểu đồ | File Ảnh (300 DPI) | File Dữ liệu Nguồn (CSV) | Kiểm định Thống kê Chính |
| :---: | :--- | :--- | :--- | :--- |
| **Hình 1** | Ma trận Tương quan Đa biến | `reports/report4/figures/integrated/fig1_multimodal_correlation_heatmap.png` | `fig1_correlation_matrix.csv` | Hệ số tương quan hạng Spearman ($r_s$) |
| **Hình 2** | Phân tầng Thành tích theo ChartScore | `reports/report4/figures/integrated/fig2_chart_score_tiers_profiling.png` | `fig2_chart_score_tiers_summary.csv` | Kruskal-Wallis Test ($H$-test, $\alpha = 0.05$) |
| **Hình 3** | Tương quan Hồi quy & LOWESS | `reports/report4/figures/integrated/fig3_features_vs_chartscore_scatter.png` | `fig3_features_vs_chartscore_scatter.csv` | Spearman $r_s$, Pearson $r$ & Đường LOWESS |
| **Hình 4** | Dịch chuyển theo Năm & Thị trường | `reports/report4/figures/integrated/fig4_temporal_market_evolution.png` | `fig4_temporal_market_evolution.csv` | Bootstrap 95% Confidence Interval (2021–2026) |

---

## 3. Các Phát hiện Thực nghiệm Chính

### 3.1. Phân tích Tương quan Đa biến (Hình 1 - Spearman Correlation)
1. **Mối tương quan với Điểm Thành tích (`chart_score`):**
   - Trên tập mẫu gộp toàn cầu ($N = 2,860$), các biến đơn lẻ có tương quan tuyến tính/hạng rất khiêm tốn ($|r_s| < 0.05$):
     - `spectral_bandwidth`: $r_s = +0.040, p = 0.031 < 0.05$
     - `word_count`: $r_s = +0.044, p = 0.018 < 0.05$
     - `punctuation_ratio`: $r_s = +0.043, p = 0.023 < 0.05$
     - `tempo_bpm`: $r_s = +0.024$ ($p = 0.199$)
     - `spectral_centroid`: $r_s = +0.023$ ($p = 0.209$)
     - `repetition_ratio`: $r_s = +0.017$ ($p = 0.356$)
     - `chord_change_rate`: $r_s = +0.016$ ($p = 0.403$)
     - `rms_energy`: $r_s = +0.007$ ($p = 0.706$)
     - `mfcc_repetition_determinism`: $r_s = -0.020$ ($p = 0.292$)
   - *Nguyên nhân cốt lõi (Market Moderation):* Do mỗi thị trường có xu hướng thẩm mỹ âm học đối nghịch (Hàn Quốc chuộng năng lượng cao $r = +0.095$, Việt Nam lại chuộng năng lượng êm dịu $r = -0.052$). Khi phân tích gộp mà không kiểm soát thị trường, các xu hướng đối lập tự triệt tiêu lẫn nhau.
2. **Yếu tố Quyết định Tuổi thọ Trụ hạng (`weeks_on_chart`):**
   - Đáng chú ý, **`weeks_on_chart` tương quan nghịch rất mạnh với độ gắt và độ chói của âm thanh ($p < 0.001$)**:
     - `spectral_centroid`: $r_s = -0.104, p < 0.001$
     - `spectral_bandwidth`: $r_s = -0.099, p < 0.001$
     - `rms_energy`: $r_s = -0.094, p < 0.001$
     - `onset_rate`: $r_s = -0.084, p < 0.001$
     - `avg_word_length`: $r_s = -0.148, p < 0.001$
     - `pronoun_ratio`: $r_s = -0.136, p < 0.001$
     - `punctuation_ratio`: $r_s = -0.147, p < 0.001$
   - Ngược lại, **`dynamic_range`** ($r_s = +0.056, p = 0.003$) và **`repetition_ratio`** ($r_s = +0.062, p = 0.001$) tương quan thuận có ý nghĩa với độ bền bài hát. Các ca khúc giữ khoảng thở tự nhiên và có hook điệp khúc lặp lại dễ nhớ sẽ ít gây mỏi tai (auditory fatigue) và trụ hạng lâu hơn.

### 3.2. So sánh Phân tầng Thành tích (Hình 2 - Performance Tiers)
- Phân chia $N = 2,860$ quan sát thành 3 Tiers theo phân vị toán học:
  - **Tier 1 (High Performance, Top 25%):** Score $\ge 32.93$ ($N = 715$).
  - **Tier 2 (Moderate Performance, 50% ở giữa):** $-22.51 \le \text{Score} < 32.93$ ($N = 1,430$).
  - **Tier 3 (Low Performance, Bottom 25%):** Score $< -22.51$ ($N = 715$).
- Kết quả kiểm định Kruskal-Wallis trên mẫu gộp cho thấy sự phân bố giữa 3 Tiers tương đối đồng đều trên các trục năng lượng và lặp lại (`rms_energy`: $H = 2.13, p = 0.345$; `mfcc_repetition_determinism`: $H = 1.08, p = 0.582$; `chord_change_rate`: $H = 0.50, p = 0.778$; `repetition_ratio`: $H = 0.73, p = 0.693$).
- Điều này chứng minh: Không có một "công thức âm học đơn nhất" nào đảm bảo lọt vào Tier 1 cho toàn cầu. Mỗi tầng đều chứa các bài hit đại diện cho các phong cách khác biệt của 3 nước.

### 3.3. Bằng chứng về Tính Phi tuyến & Ngưỡng Bão hòa (Hình 3 - LOWESS Regression)
- Đồ thị phân tán cho thấy các đặc trưng âm học và ca từ có tương quan tuyến tính rất thấp với `chart_score` ($|r| < 0.05$), các điểm dữ liệu phân tán rộng.
- Đường làm mịn LOWESS xác nhận sự tồn tại của **ngưỡng bão hòa (Diminishing Returns)**: việc tăng năng lượng hay mức độ lặp lại lên quá cao ($> 0.35$) không tiếp tục làm tăng điểm số thứ hạng.

### 3.4. Xu hướng Tiến hóa Dọc theo Thời gian (Hình 4 - Temporal Trends 2021–2026)
- Phân tầng năng lượng âm học (RMS Energy) duy trì ổn định rõ nét: Hàn Quốc luôn dẫn đầu (~0.25–0.26), Hoa Kỳ ở giữa (~0.23–0.24), và Việt Nam giữ khoảng thở lớn nhất (~0.21–0.22).
- Nhịp điệu bài hát (Tempo) tại Mỹ và Hàn Quốc ổn định quanh 120–123 BPM; Việt Nam ghi nhận sự dịch chuyển nhịp độ tăng dần từ ~118 BPM (2021) lên ~123 BPM (2024–2025).

---

## 4. Khuyến nghị Định hướng cho Mô hình Regression (Member 4 / Report 5)
1. **Bắt buộc Kiểm soát Hiệu ứng Thị trường (Market Fixed Effects & Interactions):**
   - Mô hình hồi quy OLS gộp đơn giản sẽ thất bại do hiện tượng Simpson's Paradox / Market Moderation.
   - Bắt buộc phải áp dụng **Mô hình Hiệu ứng Cố định (Fixed Effects Regression)** hoặc bổ sung biến tương tác $\text{Audio} \times \text{Market}$.
2. **Khử Đa Cộng Tuyến (Multicollinearity):**
   - Giữ `rms_energy` và loại bỏ `loudness_proxy` (Pearson $r > 0.95$).
   - Giữ `beat_interval_variability` và loại bỏ `tempo_stability`.
3. **Ưu tiên Thuật toán Học máy Dạng Cây (Tree-Based Ensembles):**
   - Do mối quan hệ có tính phi tuyến, ngưỡng bão hòa và độ phân tán lớn, các mô hình **Random Forest, LightGBM, và XGBoost Regressor** sẽ cho độ chính xác ($R^2$, RMSE) vượt trội so với Hồi quy Tuyến tính OLS thuần túy.
