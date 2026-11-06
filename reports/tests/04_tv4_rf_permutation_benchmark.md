# BÁO CÁO KIỂM ĐỊNH THỐNG KÊ & BENCHMARK NGOÀI MẪU (THÀNH VIÊN 4)
**Vai trò:** Thành viên 4 – Tích hợp dữ liệu, Huấn luyện Model 4 (Random Forest), Đánh giá Ngoài mẫu (Out-of-Sample Benchmark) & Triển khai Hệ thống  
**Học phần:** ADY201m – Phân tích Năng lực Cạnh tranh Bài hát trên Bảng xếp hạng Âm nhạc (Spotify Top 200 VN, US, KR)  
**Tập dữ liệu:** Spotify Top 200 Weekly (2021–2026), $N = 2,098$ bản ghi phân tích sạch ($1,912$ ca khúc độc nhất), chia tách Out-of-Sample có phân cụm $70/30$ theo bài hát ($N_{\text{test}} = 410$; $N_{\text{train}} = 1,688$ cho Random Forest vì RF nhận đủ 1,688 bài bao gồm 3 bài instrumental mà OLS đã loại ra khỏi mẫu hồi quy hoàn chỉnh $1,685$).

---

## 1. MỤC ĐÍCH & GIẢ THUYẾT THỐNG KÊ

### 1.1. Bối cảnh & Mục đích Nghiên cứu
Trong chuỗi mô hình của đề tài, các Thành viên 1, 2 và 3 đã lần lượt phát triển các mô hình hồi quy tuyến tính (Model 1 Baseline Audio, Model 2 Audio + Lyrics, Model 2b + Context, Model 3 Full OLS có tương tác thị trường) nhằm ước lượng các tác động biên trung bình lên điểm thành công thương mại (`chart_score`). Tuy nhiên, bản chất cố định của hồi quy tuyến tính OLS không nắm bắt được các tương tác phi tuyến tính phức tạp (non-linear interactions) và các ngưỡng bão hòa (threshold effects).

Nhiệm vụ của Thành viên 4 là thực hiện hai nội dung đánh giá:
1. **Permutation Feature Importance (PFI)**: Đánh giá khách quan mức độ phụ thuộc dự báo của mô hình vào từng thuộc tính âm học, ca từ và thời gian trên tập kiểm định ngoài mẫu ($N_{\text{test}} = 410$), sử dụng phương pháp đánh giá dựa trên mức suy giảm hiệu năng dự báo, thay vì phụ thuộc vào Gini Impurity vốn có thể thiên lệch khi đánh giá các biến có nhiều giá trị hoặc nhiều điểm chia tách.
2. **Out-of-Sample Benchmark & Paired Cluster Bootstrap Test**: So sánh hiệu năng dự báo ngoài mẫu giữa Random Forest (Model 4) và OLS Model 3, đồng thời đo lường độ bất định của mức chênh lệch sai số bằng phương pháp Bootstrap theo cụm bài hát phân tầng cho toàn bộ mẫu ba thị trường và từng thị trường riêng rẽ.

---

### 1.2. Phương pháp Đánh giá & Giả thuyết Thống kê

#### Bài test 1: Permutation Feature Importance (PFI)
* **Mục tiêu:** Đo lường mức gia tăng sai số tuyệt đối trung bình (MAE degradation) khi hoán vị ngẫu nhiên độc lập từng biến độc lập trên tập kiểm định Test ($N_{\text{test}} = 410$).
* **Tiêu chuẩn đánh giá bảo thủ để đánh giá mức độ vượt nhiễu:** Để nhận diện các đặc trưng có mức đóng góp thực sự tách biệt khỏi biên độ biến thiên ngẫu nhiên của thuật toán hoán vị, nghiên cứu áp dụng quy tắc thực nghiệm bảo thủ:
  $$\mu(\Delta\text{MAE}_{X_j}) - 2 \cdot \sigma_{X_j} > 0$$
  *Lưu ý:* Đây là tiêu chuẩn đánh giá bảo thủ thực nghiệm nhằm xếp hạng và diễn giải độ ổn định của PFI, không phải là một kiểm định suy luận thống kê giả thuyết có $p$-value chính thức và không mang hàm ý quan hệ nhân quả.

#### Bài test 2: Paired Cluster Bootstrap So sánh RF và OLS Model 3
* **Giả thuyết $H_0$:** Random Forest không có sai số dự báo ngoài mẫu thấp hơn OLS Model 3:
  $$\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}} \le 0$$
* **Giả thuyết $H_1$:** Random Forest có sai số dự báo ngoài mẫu thấp hơn OLS Model 3 có ý nghĩa thống kê:
  $$\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}} > 0$$
* **Chỉ tiêu mô tả bổ sung:** Chênh lệch hệ số xác định $\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ được báo cáo để mô tả mức độ giải thích biến thiên phương sai bổ sung (không phải tiêu chí chính để bác bỏ $H_0$).
* **Quy tắc kiểm định phân tầng (Stratified Cluster Bootstrap):**
  * **Toàn bộ mẫu ba thị trường (Overall):** Lấy mẫu cụm theo `spotify_track_id` trên toàn bộ tập Test ($N = 410$, $382$ cụm bài hát độc nhất).
  * **Từng thị trường (VN, US, KR):** Lấy mẫu cụm phân tầng trực tiếp trên tập con của chính thị trường đó (VN: $N=105$; US: $N=165$; KR: $N=140$). Số bài hát ở mỗi lần lặp được bảo toàn cố định, loại bỏ biến thiên ngẫu nhiên kích thước mẫu thị trường.
  * $B = 1,000$ lần lặp, cố định `seed=43`. Khoảng tin cậy 95% hai phía (95% CI) được tính theo phân vị $[2.5\%, 97.5\%]$. Việc yêu cầu cận dưới của khoảng tin cậy hai phía 95% lớn hơn 0 là một quy tắc kết luận bảo thủ đối với giả thuyết một phía $H_1$ ở mức ý nghĩa $\alpha = 0.05$.

---

## 2. CODE THỰC THI & LƯU KẾT QUẢ

Đoạn mã Python dưới đây thực hiện quy trình đánh giá và so sánh hiệu năng dự báo ngoài mẫu giữa Random Forest và OLS Model 3. Mã sử dụng các dự đoán đã được lưu trong `models/test_predictions.csv`, tính toán các chỉ số đánh giá, thực hiện Paired Cluster Bootstrap phân tầng cho Overall và từng thị trường (VN, US, KR), lưu kết quả vào `reports/rf_vs_ols_paired_bootstrap.csv` và tự động xuất biểu đồ so sánh ở độ phân giải 300 DPI.

```python
"""
Execution Script: Member 4 Benchmark & Stratified Cluster Paired Bootstrap Test
Dataset: models/test_predictions.csv (N=410, 3 markets)
Outputs:
  - reports/rf_vs_ols_paired_bootstrap.csv (Bảng CI Bootstrap chính thức)
  - reports/figures/rf_vs_ols_market_comparison.png (Biểu đồ so sánh 300 DPI)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Đọc dữ liệu tập Test
df_test = pd.read_csv('models/test_predictions.csv')
print(f"Loaded Test Set: N = {len(df_test)} observations across {df_test['market'].nunique()} markets.")

# 2. Hàm chạy Paired Cluster Bootstrap phân tầng theo từng phạm vi
def run_cluster_bootstrap(sub_df, n_boot=1000, seed=43):
    ids = sub_df['spotify_track_id'].values
    unique_tracks = np.unique(ids)
    n_tracks = len(unique_tracks)
    rng = np.random.default_rng(seed)
    
    track_to_indices = {}
    for idx, tid in enumerate(ids):
        track_to_indices.setdefault(tid, []).append(idx)
        
    y_true = sub_df['chart_score_actual'].values
    y_ols = sub_df['pred_ols_m3'].values
    y_rf = sub_df['pred_rf'].values
    
    # Điểm ước lượng thực tế trên mẫu ban đầu (Observed Point Estimates)
    obs_delta_mae = mean_absolute_error(y_true, y_ols) - mean_absolute_error(y_true, y_rf)
    obs_delta_r2 = r2_score(y_true, y_rf) - r2_score(y_true, y_ols)
    
    delta_r2_list = []
    delta_mae_list = []
    
    for _ in range(n_boot):
        sample_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        idx_pool = []
        for st in sample_tracks:
            idx_pool.extend(track_to_indices[st])
        idx_arr = np.array(idx_pool)
        
        yt = y_true[idx_arr]
        yo = y_ols[idx_arr]
        yr = y_rf[idx_arr]
        
        delta_r2_list.append(r2_score(yt, yr) - r2_score(yt, yo))
        delta_mae_list.append(mean_absolute_error(yt, yo) - mean_absolute_error(yt, yr))
        
    ci_r2 = np.percentile(delta_r2_list, [2.5, 97.5])
    ci_mae = np.percentile(delta_mae_list, [2.5, 97.5])
    
    return {
        'obs_delta_r2': obs_delta_r2,
        'delta_r2_bootstrap_mean': np.mean(delta_r2_list),
        'delta_r2_ci_lower': ci_r2[0],
        'delta_r2_ci_upper': ci_r2[1],
        'obs_delta_mae': obs_delta_mae,
        'delta_mae_bootstrap_mean': np.mean(delta_mae_list),
        'delta_mae_ci_lower': ci_mae[0],
        'delta_mae_ci_upper': ci_mae[1]
    }

# 3. Thực thi Bootstrap cho Overall và phân tầng từng thị trường
results = []
scopes = ['Overall', 'VN', 'US', 'KR']
for scope in scopes:
    sub = df_test if scope == 'Overall' else df_test[df_test['market'] == scope].reset_index(drop=True)
    res = run_cluster_bootstrap(sub, n_boot=1000, seed=43)
    res['scope'] = scope
    res['n_obs'] = len(sub)
    res['n_unique_tracks'] = int(sub['spotify_track_id'].nunique())
    results.append(res)
    print(f"Scope: {scope:7s} | Observed Delta MAE = {res['obs_delta_mae']:+.4f} | Bootstrap Mean = {res['delta_mae_bootstrap_mean']:+.4f} | 95% CI = [{res['delta_mae_ci_lower']:+.4f}, {res['delta_mae_ci_upper']:+.4f}]")

# 4. Xuất kết quả Bootstrap chính thức ra CSV
os.makedirs('reports', exist_ok=True)
bootstrap_df = pd.DataFrame(results)
bootstrap_df.to_csv('reports/rf_vs_ols_paired_bootstrap.csv', index=False)
print("Saved verified bootstrap results to reports/rf_vs_ols_paired_bootstrap.csv")

# 5. Trực quan hóa đối chứng ngoài mẫu đạt chuẩn 300 DPI tính toán động từ dữ liệu
market_labels = ['Vietnam (VN)', 'United States (US)', 'South Korea (KR)', 'Overall (Test)']
market_keys = ['VN', 'US', 'KR', 'Overall']

metrics = []
for key in market_keys:
    sub = df_test if key == 'Overall' else df_test[df_test['market'] == key]
    yt = sub['chart_score_actual'].values
    yo = sub['pred_ols_m3'].values
    yr = sub['pred_rf'].values
    metrics.append({
        'n': len(sub),
        'r2_ols': r2_score(yt, yo) * 100,
        'r2_rf': r2_score(yt, yr) * 100,
        'mae_ols': mean_absolute_error(yt, yo),
        'mae_rf': mean_absolute_error(yt, yr)
    })
m_df = pd.DataFrame(metrics)

x = np.arange(len(market_labels))
width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Subplot R2
rects1 = ax1.bar(x - width/2, m_df['r2_ols'], width, label='Model 3 (OLS)', color='#4A90E2', edgecolor='black', linewidth=0.8)
rects2 = ax1.bar(x + width/2, m_df['r2_rf'], width, label='Model 4 (Random Forest)', color='#50E3C2', edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Out-of-Sample $R^2$ (%)', fontsize=12, fontweight='bold')
ax1.set_title('Out-of-Sample Variance Explained ($R^2$)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{m}\n(N={n})' for m, n in zip(market_labels, m_df['n'])], fontsize=10)
ax1.legend(frameon=True, facecolor='white')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax1.annotate(f'{h:.2f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0E6251')

# Subplot MAE
rects3 = ax2.bar(x - width/2, m_df['mae_ols'], width, label='Model 3 (OLS)', color='#4A90E2', edgecolor='black', linewidth=0.8)
rects4 = ax2.bar(x + width/2, m_df['mae_rf'], width, label='Model 4 (Random Forest)', color='#E67E22', edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Mean Absolute Error (MAE - lower is better)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error Comparison (MAE)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{m}\n(N={n})' for m, n in zip(market_labels, m_df['n'])], fontsize=10)
ax2.legend(frameon=True, facecolor='white')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for r in rects3:
    h = r.get_height()
    ax2.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
for r in rects4:
    h = r.get_height()
    ax2.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#78281F')

plt.suptitle('Out-of-Sample Benchmark: Random Forest vs. OLS across Markets (Test Set N=410)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/rf_vs_ols_market_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved dynamic figure to reports/figures/rf_vs_ols_market_comparison.png")
```

---

## 3. BIỂU ĐỒ MINH HỌA

### 3.1. Phân cấp Tầm quan trọng Biến ngẫu nhiên (Permutation Importance)
Biểu đồ hiển thị mức tăng sai số tuyệt đối trung bình $\mu(\Delta\text{MAE})$ kèm độ lệch chuẩn $\pm \sigma$ khi xáo trộn từng biến độc lập trên tập Test ($N = 410$):

![Permutation Feature Importance](../figures/rf_permutation_importance.png)

### 3.2. So sánh Năng lực Dự báo Ngoài mẫu giữa OLS M3 và Random Forest
Biểu đồ đối chứng tỷ lệ phương sai giải thích ($R^2$) và sai số dự báo (MAE) giữa OLS Model 3 và Model 4 Random Forest trên toàn bộ tập Test cũng như theo từng thị trường riêng biệt (được sinh động từ dữ liệu thực nghiệm):

![Out-of-Sample Benchmark across Markets](../figures/rf_vs_ols_market_comparison.png)

---

## 4. BẢNG KẾT QUẢ THỐNG KÊ CỐT LÕI

### 4.1. Bảng Tầm quan trọng Biến Hoán vị (Permutation Feature Importance) trên Test Set ($N = 410$)
*(Trích xuất chuẩn xác từ báo cáo kỹ thuật `reports/report5/rf_benchmark_metrics.json`)*:

| Thứ hạng | Đặc trưng độc lập (Feature) | Nhóm biến | Mean Importance ($\mu$) | Độ lệch chuẩn ($\sigma$) | $\mu - 2\sigma$ | Thỏa mãn tiêu chí bảo thủ $\mu - 2\sigma > 0$? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | `release_year` | Metadata / Thời gian | **3.4621** | $\pm 0.5799$ | **+2.3024** | **VƯỢT TIÊU CHÍ BẢO THỦ** |
| **2** | `rms_variability` | Âm học (Động lực học) | **0.0665** | $\pm 0.0458$ | $-0.0251$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **3** | `spectral_bandwidth` | Âm học (Âm sắc) | **0.0657** | $\pm 0.0365$ | $-0.0073$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **4** | `avg_word_length` | Ca từ (Độ dài từ) | **0.0542** | $\pm 0.0683$ | $-0.0823$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **5** | `chroma_longest_pattern_duration` | Âm học (Hài âm) | **0.0441** | $\pm 0.0251$ | $-0.0062$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **6** | `energy` | Âm học (Năng lượng) | **0.0316** | $\pm 0.0195$ | $-0.0074$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **7** | `loudness_proxy` | Âm học (Cường độ âm) | **0.0205** | $\pm 0.0150$ | $-0.0095$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **8** | `punctuation_ratio` | Ca từ (Dấu câu) | **0.0138** | $\pm 0.0391$ | $-0.0643$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **9** | `Market_VN` | Dummy Thị trường | **0.0138** | $\pm 0.0147$ | $-0.0155$ | Không thỏa (Chưa tách khỏi nhiễu) |
| **10** | `onset_rate` | Âm học (Tần suất nốt) | **0.0073** | $\pm 0.1082$ | $-0.2091$ | Không thỏa (Chưa tách khỏi nhiễu) |

> **Nhận định về Permutation Feature Importance:** Chỉ duy nhất biến năm phát hành (`release_year`, $\mu - 2\sigma = +2.3024 > 0$) vượt qua tiêu chí đánh giá bảo thủ. Tất cả 9 đặc trưng còn lại, bao gồm các biến âm học, ca từ và biến kiểm soát thị trường, đều có $\mu - 2\sigma < 0$, cho thấy mức importance của từng đặc trưng riêng lẻ chưa đủ để tách biệt rõ ràng khỏi nhiễu hoán vị theo tiêu chí này. Đây không phải kết luận bác bỏ $H_0$ bằng một kiểm định suy luận thống kê có $p$-value chính thức. Ngoài ra, do phương pháp hoán vị hiện tại thực hiện theo từng dòng (row-wise), các nghiên cứu tiếp theo có thể cân nhắc Grouped Permutation Importance theo mã bài hát để bảo toàn cấu trúc nhóm đồng nhất giữa các thị trường.

---

### 4.2. Bảng Đối chứng Ngoài Mẫu (Out-of-Sample Performance) trên Test Set ($N = 410$)
*(Quy ước: $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$, giá trị dương thể hiện RF có sai số thấp hơn OLS)*:

| Phân vùng Thị trường | Cỡ mẫu ($N$) | Mô hình | $R^2$ Ngoài mẫu (%) | MAE | RMSE | Mức chênh lệch của Random Forest so với OLS ($\Delta$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Toàn bộ Test Set (Overall)** | **410** | **OLS Model 3**<br>**Random Forest (M4)** | $3.40\%$<br>**$6.86\%$** | $29.78$<br>**$29.13$** | $38.30$<br>**$37.61$** | $\Delta R^2 = +3.46$ điểm phần trăm<br>$\Delta\text{MAE} = +0.65$ điểm (giảm sai số khoảng $2.2\%$) |
| **Việt Nam (VN)** | **105** | **OLS Model 3**<br>**Random Forest (M4)** | $2.54\%$<br>**$13.35\%$** | $35.82$<br>**$33.66$** | $44.31$<br>**$41.78$** | **$\Delta R^2 = +10.81$ điểm phần trăm**<br>**$\Delta\text{MAE} = +2.16$ điểm (giảm sai số khoảng $6.0\%$)** |
| **Hoa Kỳ (US)** | **165** | **OLS Model 3**<br>**Random Forest (M4)** | $2.85\%$<br>**$3.12\%$** | $27.64$<br>**$27.13$** | $36.01$<br>**$35.96$** | $\Delta R^2 = +0.28$ điểm phần trăm<br>$\Delta\text{MAE} = +0.51$ điểm |
| **Hàn Quốc (KR)** | **140** | **OLS Model 3**<br>**Random Forest (M4)** | **$3.84\%$**<br>$2.81\%$ | **$27.77$**<br>$28.09$ | **$36.00$**<br>$36.19$ | $\Delta R^2 = -1.03$ điểm phần trăm<br>$\Delta\text{MAE} = -0.32$ điểm (OLS tốt hơn nhẹ) |

*Lưu ý: Các mức chênh lệch $\Delta R^2$ được tính từ giá trị chưa làm tròn; do đó, kết quả có thể chênh lệch nhẹ so với phép trừ trực tiếp các giá trị $R^2$ đã làm tròn trong bảng.*

---

### 4.3. Bảng Kiểm định Paired Cluster Bootstrap Phân Tầng ($B = 1,000$ iterations, seed=43)
*(Kết quả đối chiếu với `reports/rf_vs_ols_paired_bootstrap.csv`. Lấy mẫu cụm theo `spotify_track_id`, phân tầng độc lập theo từng thị trường để bảo toàn cỡ mẫu)*:

| Phạm vi Mẫu (Scope) | Kích thước Mẫu ($N$) | Số Cụm Bài Hát (Tracks) | Tiêu chí Kiểm định (RF vs. OLS M3) | Điểm Ước lượng Thực tế ($\Delta_{\text{obs}}$) | Trung bình Bootstrap (1,000 lần lặp) | Khoảng Tin cậy 95% Bootstrap (95% CI) | Chứa giá trị $0$? | Kết luận Thống kê ($\alpha = 0.05$) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Toàn bộ Test (Overall)** | **410** | **382** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ (chính)<br>$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ (bổ sung) | **$+0.6505$** pts<br>$+3.46\%$ | **$+0.6315$** pts<br>$+0.0335$ ($+3.35\%$) | **$[-0.6602, \, +1.9767]$**<br>$[-0.0377, \, +0.0997]$ | **CÓ**<br>CÓ | **Chưa đủ bằng chứng bác bỏ $H_0$**. Mức giảm sai số của RF chưa đủ bằng chứng thống kê để khẳng định trên toàn bộ mẫu kiểm định gồm ba thị trường. |
| **Việt Nam (VN)** | **105** | **105** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ (chính)<br>$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ (bổ sung) | **$+2.1633$** pts<br>$+10.81\%$ | **$+2.1521$** pts<br>$+0.1074$ ($+10.74\%$) | **$[-0.2832, \, +4.4728]$**<br>$[-0.0133, \, +0.2144]$ | **CÓ**<br>CÓ | Mức giảm sai số đạt $+2.16$ điểm MAE và $R^2$ tăng $+10.81\%$, tuy nhiên khoảng tin cậy 95% vẫn chứa giá trị 0 nên chưa đủ bằng chứng bác bỏ $H_0$. |
| **Hoa Kỳ (US)** | **165** | **165** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ (chính)<br>$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ (bổ sung) | **$+0.5114$** pts<br>$+0.28\%$ | **$+0.5117$** pts<br>$+0.0011$ ($+0.11\%$) | **$[-1.9387, \, +3.0100]$**<br>$[-0.1580, \, +0.1436]$ | **CÓ**<br>CÓ | **Chưa đủ bằng chứng bác bỏ $H_0$**. Kết quả chưa cho thấy sự khác biệt rõ ràng về hiệu năng dự báo giữa OLS và Random Forest tại thị trường Mỹ. |
| **Hàn Quốc (KR)** | **140** | **140** | $\Delta\text{MAE} = \text{MAE}_{\text{OLS}} - \text{MAE}_{\text{RF}}$ (chính)<br>$\Delta R^2 = R^2_{\text{RF}} - R^2_{\text{OLS}}$ (bổ sung) | **$-0.3200$** pts<br>$-1.03\%$ | **$-0.3687$** pts<br>$-0.0128$ ($-1.28\%$) | **$[-2.3154, \, +1.5649]$**<br>$[-0.1199, \, +0.0864]$ | **CÓ**<br>CÓ | **Chưa đủ bằng chứng bác bỏ $H_0$**. OLS nhỉnh hơn nhẹ nhưng khoảng tin cậy bao quanh 0. |

---

## 5. KẾT LUẬN & ĐIỂM NHẤN ĐƯA LÊN SLIDE THUYẾT TRÌNH

### 5.1. Ba Thông điệp Cốt lõi cho Bài Thuyết trình Báo cáo (Slide Highlights)
1. **Vai trò nổi bật của năm phát hành trong khả năng dự báo của Random Forest:**
   * Kết quả Permutation Feature Importance cho thấy `release_year` là đặc trưng có mức importance nổi bật nhất trong mô hình Random Forest ($\mu = 3.462$, $\mu - 2\sigma = +2.302 > 0$). Biến này là đặc trưng duy nhất vượt qua tiêu chí đánh giá bảo thủ.
   * Kết quả phản ánh mức độ phụ thuộc dự báo của mô hình vào thông tin năm phát hành, không chứng minh quan hệ nhân quả giữa năm phát hành và thành công thương mại của bài hát. Tất cả 9 đặc trưng còn lại, bao gồm các biến âm học, ca từ và biến kiểm soát thị trường, đều có $\mu - 2\sigma < 0$, cho thấy mức độ đóng góp đơn lẻ của từng đặc trưng chưa đủ để tách biệt rõ ràng khỏi nhiễu hoán vị trên tập Test.
2. **Tín hiệu cải thiện dự báo của Random Forest tại thị trường Việt Nam:**
   * Trên mẫu kiểm định Việt Nam, Random Forest ghi nhận mức tăng $R^2$ điểm mẫu từ $2.54\%$ lên $13.35\%$ ($+10.81$ điểm phần trăm) và giảm MAE khoảng $2.16$ điểm so với OLS Model 3.
   * Tuy nhiên, khoảng tin cậy Bootstrap 95% của cả $\Delta\text{MAE}$ ($[-0.2832, \, +4.4728]$) và $\Delta R^2$ ($[-0.0133, \, +0.2144]$) đều chứa giá trị $0$. Vì vậy, đây là kết quả mô tả mang tính gợi mở thực nghiệm, chưa đủ bằng chứng để kết luận Random Forest vượt trội có ý nghĩa thống kê tại thị trường Việt Nam.
3. **Phối hợp Song hành Tuyến tính – Phi tuyến trên Web App (Streamlit Dual-Model Architecture):**
   * Đề tài đề xuất mô hình ứng dụng kết hợp: OLS Model 3 đảm nhiệm vai trò **giải thích kinh tế lượng (Inference & Interpretability)** thông qua các hệ số hồi quy và kết quả kiểm định thống kê trong khuôn khổ các giả định của mô hình; trong khi Model 4 Random Forest đóng vai trò **công cụ mô phỏng dự báo phi tuyến (Real-time Non-linear Simulator)** phục vụ trực quan hóa trên Tab 2 và Tab 4 của ứng dụng Web.

---
*Báo cáo kiểm định thống kê chuẩn hóa bởi Thành viên 4 – Đồ án ADY201m. Dữ liệu đối chứng lưu tại `reports/rf_vs_ols_paired_bootstrap.csv`.*
