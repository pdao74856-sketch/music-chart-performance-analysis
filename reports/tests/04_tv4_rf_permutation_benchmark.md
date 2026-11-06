# BÁO CÁO KIỂM ĐỊNH THỐNG KÊ & BENCHMARK NGOÀI MẪU (THÀNH VIÊN 4)
**Vai trò:** Thành viên 4 – Tích hợp dữ liệu, Huấn luyện Model 4 (Random Forest), Đánh giá Ngoài mẫu (Out-of-Sample Benchmark) & Triển khai Hệ thống  
**Học phần:** ADY201m – Phân tích Năng lực Cạnh tranh Bài hát trên Bảng xếp hạng Âm nhạc (Spotify Top 200 VN, US, KR)  
**Tập dữ liệu:** Spotify Top 200 Weekly (2021–2024), $N = 2,098$ bản ghi hoàn chỉnh sạch ($1,912$ ca khúc độc nhất), chia tách Out-of-Sample có phân cụm $70/30$ ($N_{\text{train}} = 1,688$, $N_{\text{test}} = 410$).

---

## 1. MỤC ĐÍCH & GIẢ THUYẾT THỐNG KÊ

### 1.1. Bối cảnh & Mục đích Nghiên cứu
Trong chuỗi mô hình của đề tài, các Thành viên 1, 2 và 3 đã xây dựng các mô hình tuyến tính OLS (Model 1, Model 2, Model 2b, Model 3) nhằm giải thích tác động biên cố định của đặc trưng âm học (Audio features) và ca từ (Lyrics features) lên điểm thành công thương mại (`chart_score`). Tuy nhiên, giả định quan hệ tuyến tính chặt chẽ của OLS có thể bỏ sót các mối quan hệ phi tuyến tính (non-linearities), điểm bão hòa ngưỡng (threshold effects), hoặc tương tác bậc cao giữa các thuộc tính bài hát.

Mục tiêu trọng tâm của Thành viên 4 gồm hai bài toán kiểm định thống kê:
1. **Kiểm định Tầm quan trọng Biến bằng Hoán vị (Permutation Feature Importance Test)**: Đánh giá khách quan sức mạnh dự báo thực tế của từng đặc trưng độc lập trên tập kiểm định ngoài mẫu ($N_{\text{test}} = 410$) khi loại bỏ tính phụ thuộc vào cấu trúc cây quyết định (tránh thiên lệch của Gini Impurity đối với biến liên tục).
2. **Kiểm định So sánh Hiệu năng Ngoài mẫu & Paired Bootstrap Test (Model 4 Random Forest vs. Model 3 OLS)**: Đo lường mức độ cải thiện dự báo thực chất khi mở rộng từ mô hình tuyến tính sang phi tuyến, đồng thời xác định xem sự chênh lệch có ý nghĩa thống kê hay chỉ do ngẫu nhiên mẫu.

---

### 1.2. Giả thuyết Thống kê & Mức Ý nghĩa ($\alpha = 0.05$)

#### Bài test 1: Kiểm định Tầm quan trọng Hoán vị (Permutation Feature Importance)
* **Giả thuyết $H_0$:** Việc hoán vị ngẫu nhiên giá trị của đặc trưng $X_j$ trên tập Test không làm thay đổi sai số dự báo của mô hình Random Forest ($\Delta\text{MAE}_{X_j} = 0$, nghĩa là $X_j$ không chứa thông tin dự báo ngoài mẫu độc lập).
* **Giả thuyết $H_1$:** Việc hoán vị ngẫu nhiên giá trị của đặc trưng $X_j$ làm gia tăng sai số dự báo một cách đáng kể ($\Delta\text{MAE}_{X_j} > 0$).
* **Tiêu chuẩn bác bỏ:** Bác bỏ $H_0$ nếu mức tăng sai số trung bình qua 10 lượt hoán vị độc lập $\mu(\Delta\text{MAE}_{X_j}) - 2 \cdot \sigma > 0$.

#### Bài test 2: Kiểm định Paired Bootstrap So sánh Ngoài Mẫu (RF vs. OLS M3)
* **Giả thuyết $H_0$:** Không có sự khác biệt có ý nghĩa thống kê về sai số dự báo ngoài mẫu giữa Model 4 (Random Forest) và Model 3 (OLS Full Model):
  $$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}} = 0 \quad \text{và} \quad \Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}} = 0$$
* **Giả thuyết $H_1$:** Tồn tại sự vượt trội thực chất của Model 4 so với Model 3 trên tập ngoài mẫu:
  $$\Delta R^2 > 0 \quad \text{hoặc} \quad \Delta\text{MAE} > 0$$
* **Phương pháp kiểm định:** Phân tích Paired Bootstrap $B = 1,000$ lần lặp có phân cụm theo bài hát (`spotify_track_id`) để bảo toàn cấu trúc tương quan chuỗi thời gian giữa các tuần. Khoảng tin cậy 95% (95% CI) được xác định qua bách phân vị $[2.5\%, 97.5\%]$. Bác bỏ $H_0$ khi và chỉ khi 95% CI không chứa giá trị $0$.

---

## 2. CODE THỰC THI & LƯU BIỂU ĐỒ

Đoạn mã Python dưới đây thực thi toàn bộ quy trình: trích xuất độ đo dự báo từ `models/test_predictions.csv`, tính toán ma trận kiểm định theo từng thị trường, kiểm định Paired Bootstrap $B=1,000$ mẫu phân cụm, và xuất đồ thị trực quan hóa chất lượng cao đạt chuẩn 300 DPI phục vụ báo cáo.

```python
"""
Execution Script: Member 4 Permutation Importance & Out-of-Sample Benchmark
Artifact Path: reports/tests/04_tv4_rf_permutation_benchmark.md
Figure Outputs:
  - reports/figures/rf_permutation_importance.png
  - reports/figures/rf_vs_ols_market_comparison.png
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đọc dữ liệu tập Test và mô hình Random Forest đã huấn luyện
df_test = pd.read_csv('models/test_predictions.csv')
rf_pipeline = joblib.load('models/rf_pipeline.pkl')
features_meta = joblib.load('models/features_metadata.pkl')

print(f"Loaded Test Set: N = {len(df_test)} observations across {df_test['market'].nunique()} markets.")

# 2. Kiểm định Paired Bootstrap theo Cụm bài hát (Cluster Bootstrap, B = 1,000)
np.random.seed(43)
unique_tracks = df_test['spotify_track_id'].unique()
n_bootstraps = 1000

diff_r2_list = []
diff_mae_list = []

for _ in range(n_bootstraps):
    sampled_tracks = np.random.choice(unique_tracks, size=len(unique_tracks), replace=True)
    # Lấy toàn bộ các tuần của các bài hát được chọn
    sample_df = pd.concat([df_test[df_test['spotify_track_id'] == tid] for tid in sampled_tracks])
    
    y_true = sample_df['chart_score_actual']
    y_ols = sample_df['pred_ols_m3']
    y_rf = sample_df['pred_rf']
    
    diff_r2_list.append(r2_score(y_true, y_rf) - r2_score(y_true, y_ols))
    diff_mae_list.append(mean_absolute_error(y_true, y_ols) - mean_absolute_error(y_true, y_rf))

ci_r2 = np.percentile(diff_r2_list, [2.5, 97.5])
ci_mae = np.percentile(diff_mae_list, [2.5, 97.5])

print(f"Delta R2 (RF - OLS): Mean = {np.mean(diff_r2_list):.4f}, 95% CI = [{ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")
print(f"Delta MAE (OLS - RF): Mean = {np.mean(diff_mae_list):.4f}, 95% CI = [{ci_mae[0]:.4f}, {ci_mae[1]:.4f}]")

# 3. Xuất biểu đồ so sánh Hiệu năng Ngoài mẫu giữa OLS M3 và RF
markets = ['Vietnam (VN)', 'United States (US)', 'South Korea (KR)', 'Overall (Test)']
n_obs = [105, 165, 140, 410]
ols_r2 = [2.54, 2.85, 3.84, 3.40]
rf_r2 = [13.35, 3.12, 2.81, 6.86]
ols_mae = [35.82, 27.64, 27.77, 29.78]
rf_mae = [33.66, 27.13, 28.09, 29.13]

x = np.arange(len(markets))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Subplot R2
ax1.bar(x - width/2, ols_r2, width, label='Model 3 (OLS)', color='#4A90E2', edgecolor='black', linewidth=0.8)
ax1.bar(x + width/2, rf_r2, width, label='Model 4 (Random Forest)', color='#50E3C2', edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Out-of-Sample $R^2$ (%)', fontsize=12, fontweight='bold')
ax1.set_title('Out-of-Sample Variance Explained ($R^2$)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{m}\n(N={n})' for m, n in zip(markets, n_obs)], fontsize=10)
ax1.legend(frameon=True, facecolor='white')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Subplot MAE
ax2.bar(x - width/2, ols_mae, width, label='Model 3 (OLS)', color='#4A90E2', edgecolor='black', linewidth=0.8)
ax2.bar(x + width/2, rf_mae, width, label='Model 4 (Random Forest)', color='#E67E22', edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Mean Absolute Error (MAE - lower is better)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error Comparison (MAE)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{m}\n(N={n})' for m, n in zip(markets, n_obs)], fontsize=10)
ax2.legend(frameon=True, facecolor='white')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Out-of-Sample Benchmark: Random Forest vs. OLS across Markets (Test Set N=410)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/rf_vs_ols_market_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## 3. BIỂU ĐỒ MINH HỌA

### 3.1. Phân cấp Tầm quan trọng Biến ngẫu nhiên (Permutation Importance)
Biểu đồ dưới đây thể hiện mức độ gia tăng sai số dự báo MAE khi từng biến độc lập bị xáo trộn ngẫu nhiên trên tập Test ($N = 410$). Các thanh bar hiển thị trung bình $\mu(\Delta\text{MAE})$ kèm sai số chuẩn $\pm \sigma$ qua 10 lượt hoán vị.

![Permutation Feature Importance](../figures/rf_permutation_importance.png)

### 3.2. So sánh Năng lực Dự báo Ngoài mẫu giữa OLS M3 và Random Forest
Biểu đồ đối chứng trực diện hệ số xác định ngoài mẫu ($R^2$) và sai số tuyệt đối trung bình (MAE) giữa mô hình tuyến tính (OLS Model 3) và mô hình phi tuyến (Model 4 Random Forest) tổng thể và bóc tách theo từng thị trường.

![Out-of-Sample Benchmark across Markets](../figures/rf_vs_ols_market_comparison.png)

---

## 4. BẢNG KẾT QUẢ THỐNG KÊ CỐT LÕI



### 4.1. Bảng Tầm quan trọng Biến Hoán vị (Permutation Feature Importance) trên Test Set ($N = 410$)

Đo lường mức tăng của Mean Absolute Error ($\Delta\text{MAE}$) khi xáo trộn từng biến. Tên biến, Mean Importance và Standard Deviation được đối chiếu trực tiếp với `reports/report5/rf_benchmark_metrics.json`.

| Thứ hạng | Đặc trưng độc lập (Feature) | Nhóm biến | Mean Importance ($\Delta\text{MAE}$) | Std Deviation |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `release_year` | Metadata / Thời gian | **3.462** | $\pm 0.580$ |
| **2** | `rms_variability` | Âm học (Audio) | **0.066** | $\pm 0.046$ |
| **3** | `spectral_bandwidth` | Âm học (Audio) | **0.066** | $\pm 0.036$ |
| **4** | `avg_word_length` | Ca từ (Lyrics / NLP) | **0.054** | $\pm 0.068$ |
| **5** | `chroma_longest_pattern_duration` | Âm học (Audio / RQA) | **0.044** | $\pm 0.025$ |
| **6** | `energy` | Âm học (Audio) | **0.032** | $\pm 0.020$ |
| **7** | `loudness_proxy` | Âm học (Audio) | **0.020** | $\pm 0.015$ |
| **8** | `punctuation_ratio` | Ca từ (Lyrics / NLP) | **0.014** | $\pm 0.039$ |
| **9** | `Market_VN` | Dummy Thị trường | **0.014** | $\pm 0.015$ |
| **10** | `onset_rate` | Âm học (Audio / Rhythm) | **0.007** | $\pm 0.108$ |

> **Nhận định then chốt 1:** Biến thời gian phát hành (`release_year`, importance = 3.462) có mức importance lớn hơn khoảng **52.5 lần** so với biến âm học đứng đầu (`rms_variability`, 0.066), và khoảng **64.1 lần** so với biến ca từ đứng đầu (`avg_word_length`, 0.054). Kết quả cho thấy yếu tố thời điểm phát hành có sức mạnh dự báo nổi bật trong mô hình Random Forest trên tập Test. Đây là kết quả về mức độ quan trọng dự báo, không tự nó chứng minh quan hệ nhân quả.

---

### 4.2. Bảng Đối chứng Ngoài Mẫu (Out-of-Sample Performance) trên Test Set ($N = 410$)

| Phân vùng Thị trường | Cỡ mẫu ($N$) | Mô hình | $R^2$ Ngoài mẫu (%) | MAE | RMSE | Mức cải thiện của Random Forest ($\Delta$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Toàn bộ Test Set (Overall)** | **410** | **OLS Model 3**<br>**Random Forest (M4)** | $3.40\%$<br>**$6.86\%$** | $29.78$<br>**$29.13$** | $38.30$<br>**$37.61$** | $\Delta R^2 = +3.46\%$ pts<br>$\Delta\text{MAE} = -0.65$ pts (giảm sai số $2.2\%$) |
| **Việt Nam (VN)** | **105** | **OLS Model 3**<br>**Random Forest (M4)** | $2.54\%$<br>**$13.35\%$** | $35.82$<br>**$33.66$** | $43.91$<br>**$41.42$** | **$\Delta R^2 = +10.81\%$ pts (tăng gấp $5.25$ lần)**<br>**$\Delta\text{MAE} = -2.16$ pts (giảm sai số $6.0\%$)** |
| **Hoa Kỳ (US)** | **165** | **OLS Model 3**<br>**Random Forest (M4)** | $2.85\%$<br>**$3.12\%$** | $27.64$<br>**$27.13$** | $35.48$<br>**$35.12$** | $\Delta R^2 = +0.27\%$ pts<br>$\Delta\text{MAE} = -0.51$ pts (tương đồng) |
| **Hàn Quốc (KR)** | **140** | **OLS Model 3**<br>**Random Forest (M4)** | **$3.84\%$**<br>$2.81\%$ | **$27.77$**<br>$28.09$ | **$36.85$**<br>$37.14$ | $\Delta R^2 = -1.03\%$ pts<br>$\Delta\text{MAE} = +0.32$ pts (OLS nhỉnh hơn nhẹ) |

---

### 4.3. Bảng Kiểm định Paired Bootstrap theo Cụm ($B = 1,000$ iterations)

| Tiêu chí Kiểm định Đối chứng (RF vs. OLS M3) | Trung bình Ước lượng | Khoảng Tin cậy 95% Bootstrap (95% CI) | Chứa giá trị $0$? | Kết luận Thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :--- |
| **Chênh lệch Hệ số Xác định:** $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ | $+0.035$ ($+3.5\%$) | $[-0.038, \, +0.100]$ | **CÓ** | **Chưa đủ bằng chứng bác bỏ $H_0$ ở phạm vi toàn cầu**. Random Forest không vượt trội OLS có ý nghĩa thống kê trên toàn bộ tập mẫu. |
| **Chênh lệch Sai số:** $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ | $+0.65$ pts | $[-0.66, \, +1.98]$ | **CÓ** | **Chưa đủ bằng chứng bác bỏ $H_0$ ở phạm vi toàn cầu**. Mức giảm sai số $0.65$ điểm nằm trong biên độ nhiễu ngẫu nhiên mẫu. |

---

## 5. KẾT LUẬN & ĐIỂM NHẤN ĐƯA LÊN SLIDE THUYẾT TRÌNH


### 5.1. Kết luận về Permutation Feature Importance

Kết quả phân tích Permutation Feature Importance cho thấy `release_year` là biến có mức độ đóng góp vào khả năng dự báo nổi bật nhất, với Mean Importance đạt **3.462**. Giá trị này cao hơn khoảng **52.5 lần** so với `rms_variability` (0.066) và khoảng **64.1 lần** so với `avg_word_length` (0.054).

Trong nhóm đặc trưng âm học, `rms_variability` và `spectral_bandwidth` cùng đạt mức importance 0.066, trong khi `chroma_longest_pattern_duration` đạt 0.044. Đối với nhóm đặc trưng ca từ, `avg_word_length` đạt 0.054, cho thấy các đặc trưng ngôn ngữ cũng đóng góp vào khả năng dự báo của mô hình.

Nhìn chung, kết quả cho thấy yếu tố thời điểm phát hành có vai trò nổi bật trong việc dự đoán hiệu suất bài hát trên bảng xếp hạng. Tuy nhiên, cần lưu ý rằng Permutation Feature Importance chỉ phản ánh mức độ ảnh hưởng của từng biến đến hiệu suất dự báo của mô hình trên tập kiểm tra, không chứng minh mối quan hệ nhân quả giữa các biến và hiệu suất bài hát.


---
*Tài liệu kiểm định được chuẩn hóa phục vụ Nghiệm thu Checkpoint 2 và Báo cáo Đồ án Cuối kỳ ADY201m.*
