# BÁO CÁO KIỂM ĐỊNH THỐNG KÊ & BENCHMARK NGOÀI MẪU (THÀNH VIÊN 4)
**Vai trò:** Thành viên 4 – Tích hợp dữ liệu, Huấn luyện Model 4 (Random Forest), Đánh giá Ngoài mẫu (Out-of-Sample Benchmark) & Triển khai Hệ thống  
**Học phần:** ADY201m – Phân tích Năng lực Cạnh tranh Bài hát trên Bảng xếp hạng Âm nhạc (Spotify Top 200 VN, US, KR)  
**Tập dữ liệu:** Spotify Top 200 Weekly (2021–2026), $N = 2,098$ bản ghi phân tích sạch ($1,912$ ca khúc độc nhất), chia tách Out-of-Sample có phân cụm $70/30$ theo bài hát ($N_{\text{test}} = 410$; $N_{\text{train}} = 1,688$ cho Random Forest vì RF nhận đủ 1,688 bài bao gồm 3 bài instrumental mà OLS đã loại ra khỏi mẫu hồi quy hoàn chỉnh $1,685$).

---

## 1. MỤC ĐÍCH & GIẢ THUYẾT THỐNG KÊ

### 1.1. Bối cảnh & Mục đích Nghiên cứu
Trong chuỗi mô hình của đề tài, các Thành viên 1, 2 và 3 đã lần lượt phát triển các mô hình hồi quy tuyến tính (Model 1 Baseline Audio, Model 2 Audio + Lyrics, Model 2b + Context, Model 3 Full OLS có tương tác thị trường) nhằm ước lượng các tác động biên trung bình lên điểm thành công thương mại (`chart_score`). Tuy nhiên, bản chất cố định của hồi quy tuyến tính OLS không nắm bắt được các tương tác phi tuyến tính phức tạp (non-linear interactions) và các ngưỡng bão hòa (threshold effects).

Nhiệm vụ của Thành viên 4 là thực hiện hai bài toán kiểm định thống kê:
1. **Kiểm định Tầm quan trọng Biến bằng Hoán vị (Permutation Feature Importance Test)**: Đánh giá khách quan sức mạnh đóng góp dự báo thực tế ngoài mẫu của từng thuộc tính âm học, ca từ và thời gian trên tập kiểm định ($N_{\text{test}} = 410$) khi triệt tiêu hiện tượng thiên lệch của Gini Impurity (vốn ưu tiên các biến liên tục nhiều giá trị chia tách).
2. **Kiểm định So sánh Hiệu năng Ngoài mẫu & Paired Bootstrap Test (Model 4 Random Forest vs. Model 3 OLS)**: Đo lường mức cải thiện sai số ngoài mẫu khi chuyển từ mô hình tuyến tính sang mô hình học máy phi tuyến, kiểm chứng bằng khoảng tin cậy Bootstrap 95% có phân cụm theo ca khúc.

---

### 1.2. Giả thuyết Thống kê & Mức Ý nghĩa ($\alpha = 0.05$)

#### Bài test 1: Kiểm định Tầm quan trọng Hoán vị (Permutation Feature Importance)
* **Giả thuyết $H_0$:** Việc hoán vị ngẫu nhiên giá trị của đặc trưng $X_j$ trên tập Test không làm thay đổi sai số dự báo của mô hình Random Forest ($\Delta\text{MAE}_{X_j} = 0$, tức là $X_j$ không chứa thông tin dự báo ngoài mẫu độc lập).
* **Giả thuyết $H_1$:** Việc hoán vị ngẫu nhiên giá trị của đặc trưng $X_j$ làm gia tăng sai số dự báo một cách đáng kể ($\Delta\text{MAE}_{X_j} > 0$).
* **Tiêu chuẩn kiểm định nghiêm ngặt:** Bác bỏ $H_0$ nếu mức tăng sai số trung bình qua các lượt hoán vị độc lập vượt trội khỏi nhiễu hoán vị:
  $$\mu(\Delta\text{MAE}_{X_j}) - 2 \cdot \sigma_{X_j} > 0$$

#### Bài test 2: Kiểm định Paired Bootstrap So sánh Ngoài Mẫu (RF vs. OLS M3)
* **Giả thuyết $H_0$:** Không có sự khác biệt có ý nghĩa thống kê về năng lực dự báo ngoài mẫu giữa Model 4 (Random Forest) và Model 3 (Full OLS):
  $$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}} = 0 \quad \text{và} \quad \Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}} = 0$$
* **Giả thuyết $H_1$:** Tồn tại sự vượt trội thực chất của Model 4 so với Model 3 trên tập kiểm định ngoài mẫu:
  $$\Delta R^2 > 0 \quad \text{hoặc} \quad \Delta\text{MAE} > 0$$
* **Quy tắc bác bỏ:** Kiểm định bằng Paired Bootstrap $B = 1,000$ lần lặp có phân cụm theo bài hát (`spotify_track_id`) với hạt giống ngẫu nhiên `seed=43`. Bác bỏ $H_0$ khi và chỉ khi Khoảng Tin cậy 95% (95% CI) theo bách phân vị $[2.5\%, 97.5\%]$ không chứa giá trị $0$.

---

## 2. CODE THỰC THI & LƯU BIỂU ĐỒ

Đoạn mã Python dưới đây thực thi quá trình tính toán kiểm định Paired Bootstrap theo cụm bài hát (Cluster Bootstrap, $B=1,000$, `seed=43`) trực tiếp trên kết quả dự báo lưu trữ tại `models/test_predictions.csv`, đồng thời đối chiếu với các chỉ số chuẩn hóa trong `reports/model_test_comparison.csv` và `reports/report5/rf_benchmark_metrics.json`.

```python
"""
Execution Script: Member 4 Benchmark & Cluster Paired Bootstrap Test
Dataset: models/test_predictions.csv (N=410, 3 markets)
Outputs:
  - reports/figures/rf_vs_ols_market_comparison.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Đọc dữ liệu dự báo ngoài mẫu từ models/test_predictions.csv
df_test = pd.read_csv('models/test_predictions.csv')
print(f"Loaded Test Set: N = {len(df_test)} observations across {df_test['market'].nunique()} markets.")

ids_test = df_test['spotify_track_id']
unique_tracks = ids_test.unique()
n_tracks = len(unique_tracks)
n_boot = 1000
rng = np.random.default_rng(43)

# 2. Tạo ánh xạ track -> index để resample theo cụm bài hát với tốc độ cao
track_to_indices = {}
for idx, tid in enumerate(ids_test):
    track_to_indices.setdefault(tid, []).append(idx)

resampled_idx = []
for _ in range(n_boot):
    st = rng.choice(unique_tracks, size=n_tracks, replace=True)
    idx_pool = []
    for t in st:
        idx_pool.extend(track_to_indices[t])
    resampled_idx.append(np.array(idx_pool))

y_test = df_test['chart_score_actual'].values
pred_m3 = df_test['pred_ols_m3'].values
pred_rf = df_test['pred_rf'].values

# 3. Tính toán Paired Bootstrap toàn cầu và theo từng thị trường
diff_r2, diff_mae = [], []
for idx in resampled_idx:
    diff_r2.append(r2_score(y_test[idx], pred_rf[idx]) - r2_score(y_test[idx], pred_m3[idx]))
    diff_mae.append(mean_absolute_error(y_test[idx], pred_m3[idx]) - mean_absolute_error(y_test[idx], pred_rf[idx]))

print(f"Overall Delta R2: {np.mean(diff_r2):+.4f}, 95% CI: [{np.percentile(diff_r2, 2.5):+.4f}, {np.percentile(diff_r2, 97.5):+.4f}]")
print(f"Overall Delta MAE: {np.mean(diff_mae):+.4f}, 95% CI: [{np.percentile(diff_mae, 2.5):+.4f}, {np.percentile(diff_mae, 97.5):+.4f}]")

for mkt in ['VN', 'US', 'KR']:
    diff_mkt_r2, diff_mkt_mae = [], []
    mkt_mask = (df_test['market'] == mkt).values
    for idx in resampled_idx:
        sub = mkt_mask[idx]
        yt = y_test[idx][sub]
        ym3 = pred_m3[idx][sub]
        yrf = pred_rf[idx][sub]
        diff_mkt_r2.append(r2_score(yt, yrf) - r2_score(yt, ym3))
        diff_mkt_mae.append(mean_absolute_error(yt, ym3) - mean_absolute_error(yt, yrf))
    print(f"{mkt}: Delta R2 = {np.mean(diff_mkt_r2):+.4f} (95% CI: [{np.percentile(diff_mkt_r2, 2.5):+.4f}, {np.percentile(diff_mkt_r2, 97.5):+.4f}])")
    print(f"{mkt}: Delta MAE = {np.mean(diff_mkt_mae):+.4f} (95% CI: [{np.percentile(diff_mkt_mae, 2.5):+.4f}, {np.percentile(diff_mkt_mae, 97.5):+.4f}])")

# 4. Trực quan hóa đối chứng ngoài mẫu đạt chuẩn 300 DPI
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
Biểu đồ hiển thị mức tăng sai số tuyệt đối trung bình $\mu(\Delta\text{MAE})$ kèm độ lệch chuẩn $\pm \sigma$ khi xáo trộn từng biến độc lập trên tập Test ($N = 410$):

![Permutation Feature Importance](../figures/rf_permutation_importance.png)

### 3.2. So sánh Năng lực Dự báo Ngoài mẫu giữa OLS M3 và Random Forest
Biểu đồ đối chứng tỷ lệ phương sai giải thích ($R^2$) và sai số dự báo (MAE) giữa OLS Model 3 và Model 4 Random Forest trên toàn bộ tập Test cũng như theo từng thị trường riêng biệt:

![Out-of-Sample Benchmark across Markets](../figures/rf_vs_ols_market_comparison.png)

---

## 4. BẢNG KẾT QUẢ THỐNG KÊ CỐT LÕI

### 4.1. Bảng Tầm quan trọng Biến Hoán vị (Permutation Feature Importance) trên Test Set ($N = 410$)
*(Trích xuất chuẩn xác từ báo cáo kỹ thuật `reports/report5/rf_benchmark_metrics.json`)*:

| Thứ hạng | Đặc trưng độc lập (Feature) | Nhóm biến | Mean Importance ($\mu$) | Độ lệch chuẩn ($\sigma$) | $\mu - 2\sigma$ | Thỏa mãn tiêu chí $\mu - 2\sigma > 0$? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | `release_year` | Metadata / Thời gian | **3.4621** | $\pm 0.5799$ | **+2.3024** | **THỎA MÃN (Bác bỏ $H_0$)** |
| **2** | `rms_variability` | Âm học (Động lực học) | **0.0665** | $\pm 0.0458$ | $-0.0251$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **3** | `spectral_bandwidth` | Âm học (Âm sắc) | **0.0657** | $\pm 0.0365$ | $-0.0073$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **4** | `avg_word_length` | Ca từ (Độ dài từ) | **0.0542** | $\pm 0.0683$ | $-0.0823$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **5** | `chroma_longest_pattern` | Âm học (Hài âm) | **0.0441** | $\pm 0.0251$ | $-0.0062$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **6** | `energy` | Âm học (Năng lượng) | **0.0316** | $\pm 0.0195$ | $-0.0074$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **7** | `loudness_proxy` | Âm học (Cường độ âm) | **0.0205** | $\pm 0.0150$ | $-0.0095$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **8** | `punctuation_ratio` | Ca từ (Dấu câu) | **0.0138** | $\pm 0.0391$ | $-0.0643$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **9** | `Market_VN` | Dummy Thị trường | **0.0138** | $\pm 0.0147$ | $-0.0155$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **10** | `onset_rate` | Âm học (Tần suất nốt) | **0.0073** | $\pm 0.1082$ | $-0.2091$ | Không thỏa (Chưa tách khỏi nhiễu) |

> **Nhận định thống kê về Permutation Test**: Chỉ duy nhất biến năm phát hành (`release_year`, $\mu - 2\sigma = +2.3024 > 0$) vượt qua tiêu chí kiểm định nghiêm ngặt để bác bỏ $H_0$. Tất cả 9 đặc trưng âm học và ca từ còn lại đều có $\mu - 2\sigma < 0$, cho thấy mức đóng góp của từng đặc trưng đơn lẻ chưa đủ lớn để tách biệt một cách có ý nghĩa thống kê khỏi nhiễu hoán vị ngẫu nhiên trên tập ngoài mẫu.

---

### 4.2. Bảng Đối chứng Ngoài Mẫu (Out-of-Sample Performance) trên Test Set ($N = 410$)
*(Đối chiếu khớp 100% với `models/test_predictions.csv` và `reports/model_test_comparison.csv`)*:

| Phân vùng Thị trường | Cỡ mẫu ($N$) | Mô hình | $R^2$ Ngoài mẫu (%) | MAE | RMSE | Mức chênh lệch của Random Forest so với OLS ($\Delta$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Toàn bộ Test Set (Overall)** | **410** | **OLS Model 3**<br>**Random Forest (M4)** | $3.40\%$<br>**$6.86\%$** | $29.78$<br>**$29.13$** | $38.30$<br>**$37.61$** | $\Delta R^2 = +3.46\%$ pts<br>$\Delta\text{MAE} = -0.65$ pts (giảm sai số $2.2\%$) |
| **Việt Nam (VN)** | **105** | **OLS Model 3**<br>**Random Forest (M4)** | $2.54\%$<br>**$13.35\%$** | $35.82$<br>**$33.66$** | $44.31$<br>**$41.78$** | **$\Delta R^2 = +10.81\%$ pts**<br>**$\Delta\text{MAE} = -2.16$ pts (giảm sai số $6.0\%$)** |
| **Hoa Kỳ (US)** | **165** | **OLS Model 3**<br>**Random Forest (M4)** | $2.85\%$<br>**$3.12\%$** | $27.64$<br>**$27.13$** | $36.01$<br>**$35.96$** | $\Delta R^2 = +0.27\%$ pts<br>$\Delta\text{MAE} = -0.51$ pts |
| **Hàn Quốc (KR)** | **140** | **OLS Model 3**<br>**Random Forest (M4)** | **$3.84\%$**<br>$2.81\%$ | **$27.77$**<br>$28.09$ | **$36.00$**<br>$36.19$ | $\Delta R^2 = -1.03\%$ pts<br>$\Delta\text{MAE} = +0.32$ pts (OLS nhỉnh hơn nhẹ) |

---

### 4.3. Bảng Kiểm định Paired Bootstrap theo Cụm Bài hát ($B = 1,000$ iterations, seed=43)
*(Thực thi trực tiếp từ `models/test_predictions.csv` bằng Cluster Bootstrap theo `spotify_track_id`, đối chiếu với `reports/paired_bootstrap.csv`)*:

| Phạm vi Mẫu | Tiêu chí Kiểm định Đối chứng (RF vs. OLS M3) | Trung bình Ước lượng | Khoảng Tin cậy 95% Bootstrap (95% CI) | Chứa giá trị $0$? | Kết luận Thống kê ($\alpha = 0.05$) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Toàn bộ Test (Overall, N=410)** | $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ | $+0.0338$ ($+3.38\%$) | $[-0.0377, \, +0.1053]$ | **CÓ** | **Chưa đủ bằng chứng bác bỏ $H_0$**. Random Forest không vượt trội OLS có ý nghĩa thống kê trên toàn cầu. |
| **Toàn bộ Test (Overall, N=410)** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ | $+0.6578$ pts | $[-0.8057, \, +2.0268]$ | **CÓ** | **Chưa đủ bằng chứng bác bỏ $H_0$**. Mức giảm sai số nằm trong biên độ nhiễu lấy mẫu. |
| **Việt Nam (VN, N=105)** | $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ | $+0.1086$ ($+10.86\%$) | $[-0.0207, \, +0.2261]$ | **CÓ** | Dù $R^2$ mẫu điểm tăng từ $2.54\%$ lên $13.35\%$, nhưng khoảng tin cậy vẫn cắt qua $0$ (chưa bác bỏ $H_0$ ở $\alpha=0.05$). |
| **Việt Nam (VN, N=105)** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ | $+2.1923$ pts | $[-0.2869, \, +4.7392]$ | **CÓ** | Xu hướng giảm sai số rõ nét (+2.19 điểm MAE) nhưng cận dưới vẫn tiếp giáp ngưỡng $0$. |
| **Hoa Kỳ (US, N=165)** | $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ | $+0.0007$ | $[-0.1545, \, +0.1515]$ | **CÓ** | Hiệu năng tương đồng hoàn toàn giữa OLS và Random Forest. |
| **Hàn Quốc (KR, N=140)** | $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ | $-0.0131$ | $[-0.1194, \, +0.0814]$ | **CÓ** | Hiệu năng tương đồng hoàn toàn (OLS nhỉnh hơn không đáng kể). |

---

## 5. KẾT LUẬN & ĐIỂM NHẤN ĐƯA LÊN SLIDE THUYẾT TRÌNH

### 5.1. Ba Thông điệp Cốt lõi cho Bài Thuyết trình Báo cáo (Slide Highlights)
1. **Quy luật Thống trị của Yếu tố Thời điểm Phát hành (Recency Effect):**
   * Kết quả Permutation Importance trên tập Test khẳng định: **Chỉ duy nhất năm phát hành (`release_year`, mean importance = $3.462$, $\mu - 2\sigma = +2.302 > 0$) thỏa mãn tiêu chí vượt khỏi nhiễu hoán vị**. 
   * Toàn bộ 9 đặc trưng âm học và ca từ còn lại đều có $\mu - 2\sigma < 0$, nghĩa là không có bất kỳ một thuộc tính âm thanh hay từ ngữ riêng lẻ nào tự thân quyết định thành công thương mại nếu tách rời khỏi bối cảnh thị trường và thời điểm ra mắt.
2. **Xu hướng Phi tuyến Riêng biệt tại Thị trường Âm nhạc Việt Nam:**
   * Trong khi ở Mỹ và Hàn Quốc, OLS và Random Forest cho kết quả tương đương, thì **tại Việt Nam, Random Forest ghi nhận mức tăng $R^2$ điểm mẫu từ $2.54\%$ lên $13.35\%$ (tăng $10.81$ điểm phần trăm)** và giảm MAE $2.16$ điểm. 
   * Tuy nhiên, theo phân tích Paired Bootstrap khắt khe có phân cụm, khoảng tin cậy 95% của $\Delta R^2$ tại VN là $[-0.0207, \, +0.2261]$ vẫn chứa giá trị $0$. Do đó, về mặt học thuật, nhóm kết luận đây là **xu hướng phi tuyến mạnh mẽ mang tính gợi mở thực nghiệm** hơn là một khẳng định vượt trội có ý nghĩa thống kê tuyệt đối.
3. **Phối hợp Song hành Tuyến tính – Phi tuyến trên Web App (Streamlit Dual-Model Architecture):**
   * Đề tài đề xuất mô hình ứng dụng kết hợp: OLS Model 3 đảm nhiệm vai trò **giải thích kinh tế lượng (Inference & Interpretability)** với các hệ số kiểm định Wald và $p$-value chuẩn xác; trong khi Model 4 Random Forest đóng vai trò **công cụ mô phỏng dự báo phi tuyến (Real-time Non-linear Simulator)** phục vụ trực quan hóa trên Tab 2 và Tab 4 của ứng dụng Web.

---
*Báo cáo kiểm định thống kê chuẩn hóa bởi Thành viên 4 – Đồ án ADY201m.*
