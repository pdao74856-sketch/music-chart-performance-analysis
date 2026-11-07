# Báo Cáo Kiểm Định: Tín Hiệu Âm Học Librosa — Phân Hóa Phân Phối (Kruskal-Wallis) & Bộ Lọc Đa Cộng Tuyến Thép (Audio Contract v2)

**Người thực hiện:** Thành viên 2 (Audio Features & DSP Signal Processing Lead)  
**Phân hệ phụ trách:** Trích xuất Đặc trưng Tín hiệu Âm học DSP, Kiểm định Dị biệt Mẫu & Xử lý Đa cộng tuyến  
**Dữ liệu sử dụng:** `view_clean_modeling_dataset` (Tập Train $N = 1,685$ quan sát / Toàn mẫu $N = 2,098$ bản ghi sạch đóng băng trên `origin/main`)  
**Mã kiểm định:** `reports/tests/02_tv2_audio_kruskal_vif_tests.md`  
**Script thực thi:** `reports/tests/run_tv2_audio_tests.py`

---

## 1. Mục Đích & Giả Thuyết Thống Kê

Báo cáo kiểm định độc lập này giải quyết hai bài toán phương pháp luận nền tảng của Phân hệ Âm học (DSP) trước khi bàn giao ma trận đặc trưng cho Phân hệ Mô hình hóa Kinh tế lượng (OLS Models của TV1) và Học máy (Random Forest Benchmark của TV4):

### 1.1. Bài Test 1: Kiểm định Kruskal-Wallis $H$-Test & Hiệu chỉnh Đa kiểm định (Holm-Bonferroni FWER) trên 14 đặc trưng âm học Librosa
* **Giả thuyết thống kê:**
  * **$H_0$ (Vô hiệu):** Phân phối của đặc trưng âm học $X$ là đồng nhất giữa 3 thị trường:
    $$P(X \mid \text{Market} = \text{VN}) = P(X \mid \text{Market} = \text{US}) = P(X \mid \text{Market} = \text{KR})$$
  * **$H_1$ (Đối nghịch):** Tồn tại ít nhất một cặp thị trường có phân phối đặc trưng âm học khác biệt nhau có ý nghĩa thống kê.
* **Mức ý nghĩa danh định:** $\alpha = 0.05$.
* **Căn cứ phương pháp luận (Tại sao dùng Kruskal-Wallis thay vì ANOVA 1-way?):**
  1. *Vi phạm phân phối chuẩn:* Tín hiệu âm thanh thương mại có phân phối lệch mạnh hoặc song đỉnh (bimodal), ví dụ `tempo` tập trung tại hai đỉnh 90–105 BPM và 120–130 BPM.
  2. *Vi phạm phương sai đồng nhất:* Kiểm định Levene trên `energy` cho $p < 0.001$, bác bỏ giả định homoskedasticity của ANOVA.
  $\implies$ Kruskal-Wallis là kiểm định phi tham số chuẩn tắc dựa trên thứ hạng (rank-based test), bảo đảm tính vững (robustness) trước các điểm ngoại lai của sóng âm.
* **Kiểm soát Sai số Họ Phép thử (Family-Wise Error Rate — FWER) qua Thuật toán Holm-Bonferroni:**
  * Khi tiến hành $m = 14$ phép thử thống kê độc lập cùng lúc ở mức $\alpha = 0.05$, xác suất mắc ít nhất một sai lầm loại I (False Positive) bùng nổ lên:
    $$\alpha_{\text{family}} = 1 - (1 - 0.05)^{14} \approx 51.23\%$$
  * Nhóm áp dụng quy trình kiểm định giảm bậc Holm-Bonferroni (step-down procedure): Sắp xếp các giá trị $p$-value thô tăng dần $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, đối chiếu từng bậc $i$ với ngưỡng hiệu chỉnh $\alpha / (m - i + 1)$. Quy trình này kiểm soát chặt chẽ FWER $\le 0.05$ mà vẫn bảo toàn lực kiểm định (statistical power) cao hơn so với hiệu chỉnh Bonferroni cổ điển.
* **Ý nghĩa thực tiễn & Giới hạn phương pháp luận (Study Limitations):**
  1. *Bản chất mẫu phân tầng:* Việc bác bỏ $H_0$ mô tả đặc tính phân phối biên $P(X \mid \text{Market})$ của các ca khúc lọt vào bảng xếp hạng tại 3 quốc gia, không đại diện cho toàn bộ các tác phẩm âm nhạc đại chúng nói chung.
  2. *Cảnh báo ngụy biện sinh thái (Ecological Fallacy):* Sự khác biệt phân phối **không đồng nghĩa** với việc thị hiếu người nghe hay độ dốc tác động $\frac{\partial Y}{\partial X}$ bắt buộc phải khác nhau giữa các nước. Tuy nhiên, nó là tiền đề thực nghiệm bắt buộc để biện minh cho việc đưa các biến thị trường (`Market_VN`, `Market_KR`) và số hạng tương tác âm học $\times$ thị trường vào mô hình hồi quy.
  3. *Giới hạn tính độc lập quan sát:* Tập Train gồm 1,685 quan sát ứng với 1,527 ca khúc duy nhất (1,383 bài xuất hiện ở 1 thị trường, 130 bài ở 2 thị trường, 14 bài ở cả 3 thị trường). Sự hiện diện của **144 ca khúc đa thị trường** tạo ra **158 dòng quan sát trùng lặp dư thừa (1,685 − 1,527)** giữa các nhóm kiểm định, tạo ra mức độ phụ thuộc nhẹ giữa các phân nhóm thị trường và vi phạm một phần giả định quan sát độc lập hoàn toàn của Kruskal-Wallis.

### 1.2. Bài Test 2: Kiểm định Đa cộng tuyến (Variance Inflation Factor — VIF) & Hợp đồng Kỹ thuật (Audio Contract v2)
* **Giả thuyết thống kê:**
  * **$H_0$:** Không có hiện tượng đa cộng tuyến nghiêm trọng giữa các biến âm học giải thích ($VIF_j < 5.0, \forall j$).
  * **$H_1$:** Tồn tại đa cộng tuyến nghiêm trọng ($VIF_j \ge 5.0$), làm ma trận hiệp phương sai $(X^TX)^{-1}$ tiệm cận suy biến, thổi phồng sai số chuẩn $SE(\hat{\beta})$ và đảo ngược dấu tác động thực tế.
* **Bản chất vật lý của sự trùng lặp tín hiệu số:**
  1. *Cụm Năng lượng — Âm lượng:* $\text{Loudness} \approx 20 \log_{10}(\text{RMS})$. Mối quan hệ giữa `energy` (biên độ RMS) và `loudness_proxy` (dBFS) là hàm số toán học tất định: tương quan tuyến tính $r = 0.9689$ và tương quan logarit $r = 1.000000$. Đưa cả hai vào OLS khiến VIF bùng nổ lên **$18.2034$ đối với `loudness_proxy`** và **$17.3871$ đối với `energy`**, vượt xa ngưỡng an toàn $5.0$.
  2. *Cụm Phổ tần âm sắc:* `spectral_centroid` (trọng tâm phổ) và `spectral_bandwidth` (độ rộng phổ) có tương quan vật lý $r = 0.9089$, đẩy VIF lên **$6.3232$ đối với `spectral_bandwidth`** và **$6.0221$ đối với `spectral_centroid`**.
* **Quy tắc kỹ thuật Audio Contract v2:**
  * Mô hình OLS bắt buộc chỉ sử dụng `energy_std` (chuẩn hóa RMS), **nghiêm cấm** đưa `loudness_proxy` vào ma trận hồi quy (`forbidden_cols`). `loudness_proxy` chỉ được giữ lại cho Random Forest và làm thanh trượt trực quan trên giao diện Web App.
  * Loại bỏ hoàn toàn `spectral_bandwidth_std`, chỉ giữ lại `spectral_centroid_std` làm đại diện cho độ sáng âm sắc (Acoustic Brightness).

---

## 2. Code Thực Thi & Quy Trình Kiểm Định

Đoạn code Python chuẩn hóa dưới đây chạy trực tiếp trên tập dữ liệu Train ($N = 1,685$), nạp dữ liệu sạch từ bộ đệm đồng bộ với view Supabase, tự động tính toán thống kê Kruskal-Wallis, cỡ hiệu ứng $\eta^2 = \frac{H - k + 1}{N - k}$ ($k = 3$), hiệu chỉnh đa kiểm định Holm-Bonferroni qua `statsmodels.stats.multitest.multipletests` và đo lường VIF trước/sau Audio Contract v2:

```python
"""
Independent Audio Feature Verification Script: Kruskal-Wallis & VIF
Repo: music-chart-performance-analysis | Author: Member 2 (Audio Lead)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

from src.features.prepare_features import get_split
from src.utils.supabase_client import load_clean_modeling_data

# 1. Nạp tập Train sạch (Frozen Seed 43, N = 1,685) từ nguồn đồng bộ
df_clean = load_clean_modeling_data()
split = get_split(df=df_clean)
X_tr = split.X_train

# 2. 14 biến âm học liên tục Librosa
audio_14 = [
    "tempo", "onset_rate", "tempo_stability", "energy", "rms_variability",
    "spectral_centroid", "chord_change_rate", "key_confidence",
    "chroma_repetition_determinism", "chroma_mean_pattern_duration",
    "chroma_longest_pattern_duration", "mfcc_repetition_determinism",
    "mfcc_mean_pattern_duration", "mfcc_longest_pattern_duration"
]

# 3. Thực thi Kruskal-Wallis H-test & eta^2 theo thị trường kèm hiệu chỉnh Holm
kruskal_records = []
N, k = len(X_tr), 3
mask_vn = X_tr["Market_VN"] == 1
mask_kr = X_tr["Market_KR"] == 1
mask_us = (~mask_vn) & (~mask_kr)

for feat in audio_14:
    col = f"{feat}_std"
    h_stat, p_val = stats.kruskal(
        X_tr.loc[mask_vn, col], 
        X_tr.loc[mask_us, col], 
        X_tr.loc[mask_kr, col]
    )
    eta_sq = (h_stat - k + 1) / (N - k)
    kruskal_records.append({
        "feature": feat,
        "H": round(h_stat, 2),
        "df": k - 1,
        "p_value_raw": p_val,
        "eta_squared": round(eta_sq, 4),
    })

df_kruskal = pd.DataFrame(kruskal_records).sort_values("H", ascending=False).reset_index(drop=True)

# Hiệu chỉnh Holm-Bonferroni
reject_holm, p_holm, _, _ = multipletests(df_kruskal["p_value_raw"], alpha=0.05, method="holm")
df_kruskal["p_value_holm"] = p_holm
df_kruskal["decision_holm"] = np.where(reject_holm, "Bác bỏ H0", "Không bác bỏ H0")

# 4. Kiểm định Đa cộng tuyến VIF cho Baseline Model 1 (Audio Contract v2)
m1_audio_cols = [f"{f}_std" for f in audio_14] + ["mode"]
X_m1 = sm.add_constant(X_tr[m1_audio_cols])
vif_records = []
for i, col in enumerate(X_m1.columns):
    if col != "const":
        vif_val = variance_inflation_factor(X_m1.values, i)
        vif_records.append({"feature": col, "vif": round(vif_val, 4)})
df_vif = pd.DataFrame(vif_records).sort_values("vif", ascending=False)
```

---

## 3. Kết Quả Kiểm Định, Biểu Đồ Minh Họa & Luận Giải Thực Nghiệm

Dưới đây là toàn bộ kết quả thực nghiệm chuẩn xác từ file thực thi `reports/tests/run_tv2_audio_tests.py`:

### 3.1. Phân Hệ 1: Dị Biệt Mẫu Âm Học Librosa & Dấu Ấn Đa Chiều (Kruskal-Wallis $H$-Test & $\eta^2$)

#### Biểu đồ minh họa 1 & 2: Phân bố Nhịp điệu, Năng lượng và Dấu ấn Radar 3 Quốc gia
![Audio Rhythm & Energy Distribution](../../reports/report4/figures/member2/fig1_rhythm_energy_distribution.png)  
*Hình 1: Phân bố nhịp điệu (`tempo`), mật độ nốt (`onset_rate`), năng lượng (`energy`) và độ biến thiên năng lượng (`rms_variability`) tại VN, US, KR.*

![Audio Fingerprint Radar](../../reports/report4/figures/member2/fig5_audio_fingerprint_radar.png)  
*Hình 2: Biểu đồ Radar so sánh 8 chiều đặc trưng âm học chuẩn hóa giữa 3 thị trường âm nhạc.*

#### Bảng số liệu cốt lõi: Kết quả Kruskal-Wallis & Hiệu chỉnh Holm-Bonferroni (Train $N = 1,685$, $k = 3$, $df = 2$)

| STT | Đặc Trưng Âm Học (`feature`) | Thống Kê $H$ | Bậc Tự Do ($df$) | $p$-value Thô | Ngưỡng Holm | $p$-value Holm | Cỡ Hiệu Ứng ($\eta^2$) | Phân Cấp Hiệu Ứng | Quyết Định Thống Kê |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `energy` | **139.41** | 2 | $5.35 \times 10^{-31}$ | 0.0036 | **$7.49 \times 10^{-30}$** | **0.0817** | **Trung bình (0.06 - 0.14)** | **Bác bỏ $H_0$** |
| 2 | `spectral_centroid` | **127.65** | 2 | $1.91 \times 10^{-28}$ | 0.0038 | **$2.48 \times 10^{-27}$** | **0.0747** | **Trung bình (0.06 - 0.14)** | **Bác bỏ $H_0$** |
| 3 | `key_confidence` | 40.89 | 2 | $1.32 \times 10^{-9}$ | 0.0042 | $1.59 \times 10^{-8}$ | 0.0231 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 4 | `rms_variability` | 38.93 | 2 | $3.51 \times 10^{-9}$ | 0.0045 | $3.86 \times 10^{-8}$ | 0.0220 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 5 | `onset_rate` | 29.81 | 2 | $3.37 \times 10^{-7}$ | 0.0050 | $3.37 \times 10^{-6}$ | 0.0165 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 6 | `chroma_longest_pattern_duration` | 26.53 | 2 | $1.74 \times 10^{-6}$ | 0.0056 | $1.56 \times 10^{-5}$ | 0.0146 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 7 | `mfcc_longest_pattern_duration` | 21.92 | 2 | $1.74 \times 10^{-5}$ | 0.0063 | $1.39 \times 10^{-4}$ | 0.0118 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 8 | `chroma_mean_pattern_duration` | 20.23 | 2 | $4.05 \times 10^{-5}$ | 0.0071 | $2.84 \times 10^{-4}$ | 0.0108 | Nhỏ (0.01 - 0.06) | Bác bỏ $H_0$ |
| 9 | `chord_change_rate` | 16.48 | 2 | $2.63 \times 10^{-4}$ | 0.0083 | $1.58 \times 10^{-3}$ | 0.0086 | Không đáng kể (<0.01) | Bác bỏ $H_0$ |
| 10 | `mfcc_repetition_determinism` | 10.65 | 2 | $4.86 \times 10^{-3}$ | 0.0100 | $2.43 \times 10^{-2}$ | 0.0051 | Không đáng kể (<0.01) | Bác bỏ $H_0$ |
| 11 | `mfcc_mean_pattern_duration` | 10.05 | 2 | $6.58 \times 10^{-3}$ | 0.0125 | $2.63 \times 10^{-2}$ | 0.0048 | Không đáng kể (<0.01) | Bác bỏ $H_0$ |
| 12 | `chroma_repetition_determinism` | 7.85 | 2 | 0.0197 | 0.0167 | 0.0592 | 0.0035 | Không đáng kể (<0.01) | **Không bác bỏ $H_0$** |
| 13 | `tempo_stability` | 5.90 | 2 | 0.0522 | 0.0250 | 0.1045 | 0.0023 | Không đáng kể (<0.01) | **Không bác bỏ $H_0$** |
| 14 | `tempo` | 3.32 | 2 | 0.1897 | 0.0500 | 0.1897 | 0.0008 | Không đáng kể (<0.01) | **Không bác bỏ $H_0$** |

> [!NOTE]
> **Tác động của Hiệu chỉnh Holm-Bonferroni:** Ở mức kiểm định thô từng biến, có 12/14 biến có $p < 0.05$. Tuy nhiên, khi kiểm soát sai số họ phép thử FWER qua phương pháp Holm, đặc trưng `chroma_repetition_determinism` có $p_{\text{raw}} = 0.0197$ vượt qua ngưỡng hiệu chỉnh của thứ hạng bậc 12 ($\alpha_{\text{Holm}} = \frac{0.05}{14 - 12 + 1} = 0.0167$), khiến $p_{\text{Holm}} = 0.0592 > 0.05$. Do đó, số lượng đặc trưng âm học thực sự có phân phối khác biệt giữa 3 thị trường là **chính xác 11/14 biến**, và có **3 biến không đủ bằng chứng thống kê để bác bỏ $H_0$** (`chroma_repetition_determinism`, `tempo_stability`, `tempo`).

#### Bảng số liệu mô tả bổ trợ: Trung vị Thực nghiệm theo Thị trường trên tập Train ($N = 1,685$)
*(Trích xuất trực tiếp từ Step 2b của script `reports/tests/run_tv2_audio_tests.py`)*

| Đặc Trưng Âm Học (`feature`) | Diễn Giải Kỹ Thuật | Trung Vị VN ($N=502$) | Trung Vị US ($N=632$) | Trung Vị KR ($N=551$) | Trung Vị Toàn Bộ ($N=1,685$) | Thị Trường Dẫn Đầu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `energy` | Năng lượng RMS (0 - 1) | 0.2178 | 0.2463 | **0.2666** | 0.2448 | Hàn Quốc (KR) |
| `spectral_centroid` | Độ sáng phổ Treble (Hz) | 2,060.32 | 2,207.87 | **2,404.11** | 2,231.53 | Hàn Quốc (KR) |
| `onset_rate` | Mật độ nốt gõ (onsets/phút) | 183.65 | **202.94** | 192.88 | 192.88 | Hoa Kỳ (US) |
| `rms_variability` | Độ biến thiên năng lượng RMS | 0.1080 | 0.1136 | **0.1190** | 0.1140 | Hàn Quốc (KR) |
| `tempo` | Nhịp điệu bài hát (BPM) | 123.05 | 123.05 | 117.45 | 117.45 | Đồng nhất (~120 BPM) |

#### Luận giải thực nghiệm chuyên sâu:
1. **Dấu ấn vượt trội của K-Pop về Năng lượng và Độ sáng Phổ:**
   * `energy` thể hiện sự phân hóa mẫu mạnh mẽ nhất toàn bộ tập dữ liệu với $H = 139.41, p_{\text{Holm}} = 7.49 \times 10^{-30}$, đạt cỡ hiệu ứng $\eta^2 = 0.0817$ (mức trung bình-khá). Hàn Quốc (KR) có trung vị năng lượng đạt **$0.2666$** (vượt trội so với US: $0.2463$ và VN: $0.2178$).
   * `spectral_centroid` xếp thứ hai với $H = 127.65, p_{\text{Holm}} = 2.48 \times 10^{-27}, \eta^2 = 0.0747$. Hình 2 (Radar) cho thấy KR vươn xa ở trục Centroid với trung vị đạt **$2,404.11\text{ Hz}$** (so với US: $2,207.87\text{ Hz}$ và VN: $2,060.32\text{ Hz}$).
   * *Giả thuyết chuyên môn sản xuất âm thanh:* Dưới góc độ kỹ thuật audio engineering, đây là giả thuyết giải thích hợp lý cho phong cách phối khí K-Pop hiện đại: thường sử dụng kỹ thuật nén âm mastering compression nhằm tối ưu hóa công suất âm thanh trên các thiết bị nghe cá nhân (tai nghe, điện thoại thông minh), đồng thời tăng cường dải cao (high-shelf EQ boost) trên các dải synthesizer và bộ gõ điện tử.
2. **Mật độ Tiết tấu Hoa Kỳ và Độ Phân tán Biên độ:**
   * Hoa Kỳ (US) dẫn đầu về mật độ nốt gõ `onset_rate` ($H = 29.81, p_{\text{Holm}} = 3.37 \times 10^{-6}$), với trung vị US đạt **$202.94\text{ onsets/phút}$** (> KR: $192.88$ > VN: $183.65\text{ onsets/phút}$). Một giả thuyết khả dĩ là điều này phản ánh cấu trúc tiết tấu dày đặc của các dòng nhạc thịnh hành trên Billboard Hot 100 (như Trap/Hip-hop với các cụm hi-hat rolls phân nhỏ nhịp).
   * Về độ biến thiên năng lượng `rms_variability` ($H = 38.93, p_{\text{Holm}} = 3.86 \times 10^{-8}$), KR đạt trung vị cao nhất ($0.1190$), tiếp theo là US ($0.1136$) và VN ($0.1080$), phản ánh mức độ tương phản động lực học giữa các đoạn verse và drop/chorus trong nhạc nhảy hiện đại.
3. **Tính Đồng Nhất Toàn Cầu của Nhịp Điệu (Tempo Invariance):**
   * Trái ngược với năng lượng và âm sắc, `tempo` hoàn toàn không có sự khác biệt ý nghĩa thống kê giữa 3 quốc gia ($H = 3.32, p_{\text{raw}} = 0.1897, p_{\text{Holm}} = 0.1897 > 0.05, \eta^2 = 0.0008$). 
   * Cả 3 quốc gia đều có trung vị tempo xấp xỉ nhau ($117.45 - 123.05\text{ BPM}$) và giá trị trung bình xoay quanh $119.5 - 122.3\text{ BPM}$, cùng thể hiện hình thái phân phối song đỉnh (bimodal) tại dải 90–105 BPM và 120–130 BPM.
   * *Giả thuyết đối chiếu sinh lý học:* Kết quả này phù hợp với giả thuyết về hiện tượng cộng hưởng nhịp sinh học cơ thể người (motor entrainment hypothesis), trong đó dải nhịp vận động tự nhiên của con người (nhịp tim khi vận động, nhịp bước chân nhảy múa) tạo ra vùng nhịp điệu phổ quát toàn cầu cho các ca khúc thương mại mà không bị chi phối bởi biên giới địa lý.

---

### 3.2. Phân Hệ 2: Kiểm Toán Đa Cộng Tuyến & Giải Pháp Kỹ Thuật Audio Contract v2 (VIF Test)

#### Biểu đồ minh họa 3: Ma trận Tương quan Ban đầu & Vùng Cảnh báo Đa cộng tuyến
![Correlation Heatmap Initial](../../reports/figures/correlation_heatmap_train_initial.png)  
*Hình 3: Ma trận tương quan Pearson ban đầu trên tập Train ($N = 1,685$). Vùng màu đỏ sẫm thể hiện tương quan cực đoan giữa `energy` $\leftrightarrow$ `loudness_proxy` ($r = 0.9689$) và `spectral_centroid` $\leftrightarrow$ `spectral_bandwidth` ($r = 0.9089$).*

#### Bảng số liệu cốt lõi: Kiểm toán VIF Trước vs Sau Audio Contract v2

| Đặc Trưng Âm Học (`feature`) | VIF Trước Xử Lý (Trùng lặp vật lý) | Quyết Định Xử Lý Theo Audio Contract v2 | VIF Sau Xử Lý (Model 1 OLS Baseline) | Trạng Thái Ma Trận Hiệp Phương Sai |
| :--- | :---: | :--- | :---: | :---: |
| `loudness_proxy_std` | **18.2034** | **CẤM đưa vào OLS** (Đưa vào `forbidden_cols`, chỉ giữ cho RF & Slider) | — | Triệt tiêu bùng nổ phương sai |
| `energy_std` | **17.3871** | **GIỮ LẠI CHO OLS** (Đại diện động năng chuẩn hóa RMS) | **2.0914** | **An toàn tuyệt đối ($< 3.0 < 5.0$)** |
| `spectral_bandwidth_std` | **6.3232** | **LOẠI BỎ** (Do cộng tuyến $r = 0.9089$ với `spectral_centroid`) | — | Loại bỏ cộng tuyến dải phổ |
| `spectral_centroid_std` | **6.0221** | **GIỮ LẠI CHO OLS** (Đại diện độ sáng âm sắc Brightness) | **1.2943** | **An toàn tuyệt đối ($< 3.0 < 5.0$)** |
| `chroma_longest_pattern_duration_std` | 2.7347 | Giữ nguyên (RQA Giai điệu) | **2.7336** | Điểm VIF cao nhất ma trận âm học |
| `mfcc_longest_pattern_duration_std` | 2.4201 | Giữ nguyên (RQA Âm sắc) | **2.4183** | An toàn |
| `tempo_std` | 2.1869 | Giữ nguyên (Nhịp điệu) | **2.1869** | An toàn |
| `chroma_mean_pattern_duration_std` | 2.1339 | Giữ nguyên (RQA Giai điệu) | **2.1339** | An toàn |
| `chord_change_rate_std` | 2.0530 | Giữ nguyên (Hòa âm) | **2.0530** | An toàn |
| `rms_variability_std` | 2.0170 | Giữ nguyên (Độ biến thiên RMS) | **2.0170** | An toàn |
| `mfcc_mean_pattern_duration_std` | 1.8520 | Giữ nguyên (RQA Âm sắc) | **1.8520** | An toàn |
| `tempo_stability_std` | 1.5000 | Giữ nguyên (Độ ổn định nhịp) | **1.5000** | An toàn |
| `key_confidence_std` | 1.3863 | Giữ nguyên (Độ tin cậy điệu thức) | **1.3863** | An toàn |
| `chroma_repetition_determinism_std` | 1.3686 | Giữ nguyên (Tính tất định giai điệu) | **1.3686** | An toàn |
| `mfcc_repetition_determinism_std` | 1.3661 | Giữ nguyên (Tính tất định âm sắc) | **1.3661** | An toàn |
| `onset_rate_std` | 1.2281 | Giữ nguyên (Mật độ nốt gõ) | **1.2281** | An toàn |
| `mode` | 1.1875 | Giữ nguyên (Điệu thức nhị phân $\{0, 1\}$) | **1.1875** | An toàn |

#### Luận giải bản chất toán học & Giải pháp kỹ thuật Audio Contract v2:
1. **Quan hệ hàm số tất định giữa Energy và Loudness:**
   * Trong vật lý âm học, mức áp suất âm thanh tính bằng decibel được định nghĩa qua điện áp hiệu dụng RMS:
     $$\text{Loudness (dBFS)} \approx 20 \log_{10}(\text{RMS}) = 20 \log_{10}(\text{Energy})$$
   * Thực nghiệm trên tập dữ liệu đóng băng xác nhận: Tương quan tuyến tính đạt $r = 0.9689$, và tương quan giữa $\log_{10}(\text{energy})$ với $\text{loudness\_proxy}$ đạt **chính xác $r = 1.000000$**.
   * *Hậu quả nếu phạm sai lầm mô hình hóa:* Nếu cố tình giữ cả hai biến vào ma trận hồi quy OLS, hệ số phóng đại phương sai bùng nổ lên **$VIF = 18.2034$** (đối với `loudness_proxy_std`) và **$17.3871$** (đối với `energy_std`). Hiện tượng này làm cho ma trận thông tin $X^TX$ tiệm cận suy biến, định thức $\det(X^TX) \to 0$, khiến sai số chuẩn $SE(\hat{\beta})$ bị thổi phồng giả tạo và dấu của các hệ số hồi quy bị đảo lộn ngẫu nhiên.
2. **Audio Contract v2 bảo vệ ma trận ước lượng Model 1:**
   * Phân hệ Âm học đã thiết lập hợp đồng kỹ thuật nghiêm ngặt:
     * Cấm triệt để `loudness_proxy` khỏi ma trận biến độc lập OLS (`forbidden_cols`).
     * Sử dụng biến chuẩn hóa `energy_std` làm đại diện duy nhất cho năng lượng sóng âm.
     * Loại bỏ `spectral_bandwidth_std` do cộng tuyến vật lý $r = 0.9089$ với `spectral_centroid_std`.
   * **Kết quả kiểm toán sau xử lý:** VIF của `energy_std` giảm từ $17.3871$ xuống **$2.0914$**. Biến âm học có VIF cao nhất trong toàn bộ ma trận Model 1 chỉ là `chroma_longest_pattern_duration_std` với **$VIF = 2.7336 < 3.0 \ll 5.0$**. Ma trận âm học hoàn toàn sạch đa cộng tuyến, bảo vệ tính khả nghịch cho $(X^TX)^{-1}$ và tính vững cho các ước lượng OLS.

---

## 4. Đánh Giá Tính Vững & Kết Nối Giao Diện Web App

Bên cạnh việc bảo đảm chuẩn mực thống kê, các kết quả kiểm định âm học trên được đồng bộ hóa với định hướng phát triển phần mềm ứng dụng:

```
[Người Dùng Web App] 
        │ (Kéo slider Loudness: -25 đến -5 dBFS theo thói quen tai nghe)
        ▼
[Tab 2: What-If Simulator (Checkpoint 3)]
        │
        ├─► [Thuật Toán Ánh Xạ Ngược Audio Contract v2]: Energy = 10^(Loudness / 20)
        │       │
        │       └─► [OLS Model Pipeline]: Nhận Energy chuẩn hóa (VIF = 2.09, An toàn)
        │
        └─► [Random Forest Benchmark]: Nhận trực tiếp Loudness & Energy phi tuyến
```

1. **Tab 2 (What-if Hit Song Simulator Sliders — Thiết kế chuẩn bị cho Checkpoint 3):**
   * *Yêu cầu công thái học (UX):* Người dùng và nhạc sĩ tư duy về độ lớn âm thanh bằng thang decibel trực quan (dBFS, ví dụ $-14\text{ dBFS}$ cho chuẩn mastering Spotify) thay vì ước lượng điện áp RMS (0.05 – 0.45).
   * *Giải pháp công nghệ:* Giao diện Tab 2 bố trí thanh trượt `Loudness (dBFS)`. Phía backend tự động thực hiện phép biến đổi mũ $Energy = 10^{(\text{Loudness}/20)}$ trước khi nạp vào pipeline OLS. Thiết kế này vừa bảo đảm tính tiện dụng cho người dùng, vừa bảo toàn nghiêm ngặt Audio Contract v2.
2. **Tab 3 (Audio Upload & Real-time DSP Extraction — Rào chắn năng lực DSP):**
   * Module `src/features/audio_extractor.py` trích xuất đầy đủ **17 đặc trưng âm học chuẩn hóa thô** từ file âm thanh tải lên (bao gồm cả `loudness_proxy` và `spectral_bandwidth` để phục vụ trực quan hóa dữ liệu trên giao diện), trong khi pipeline hồi quy Model 1 chọn lọc đúng **15 biến chuẩn tắc** theo Audio Contract v2.
   * **Kết quả đo lường thực nghiệm cục bộ (Benchmarking trên máy phát triển):**
     * Thử nghiệm trên file nhạc mẫu (`想見你想見你想見你.mp3`, dung lượng 3.63 MB, thời lượng 240s): Thời gian giải mã Librosa toàn bài đạt **9.61 giây**, mức tăng RAM thực tế của hệ điều hành là **134.61 MB** (tiến trình đạt đỉnh **278.86 MB**), chiếm khoảng 27.2% hạn mức RAM 1,024 MB của Streamlit Cloud.
     * Đây là kết quả đo kiểm thử nghiệm ban đầu nhằm xác nhận tính khả thi về tài nguyên máy chủ trước khi nhóm chính thức ghép nối hoàn chỉnh giao diện tại Checkpoint 3.

---

## 5. Kết Luận & Điểm Nhấn Đưa Lên Slide Thuyết Trình

### 5.1. 3 Gạch Đầu Dòng Đúc Kết Dành Cho Slide Nhóm (Khung Thuyết Trình 2 Phút của TV2)
* **Dị biệt mẫu âm học có ý nghĩa thống kê (11/14 biến bác bỏ $H_0$ sau hiệu chỉnh Holm):** Kiểm định Kruskal-Wallis có hiệu chỉnh sai số họ phép thử (FWER) xác nhận K-Pop vượt trội về Năng lượng ($H = 139.41, \eta^2 = 0.082$) và Độ sáng phổ Treble ($H = 127.65, \eta^2 = 0.075$); Pop Hoa Kỳ dẫn đầu về Mật độ nốt gõ Onset ($202.94\text{ onsets/phút}, p < 0.001$). Ngược lại, Tempo hoàn toàn đồng nhất toàn cầu quanh hằng số sinh học 120 BPM ($H = 3.32, p_{\text{Holm}} = 0.190$).
* **Audio Contract v2 triệt tiêu đa cộng tuyến vật lý:** Nhóm phát hiện tương quan tất định giữa Energy và Loudness ($r = 1.0$ trên thang log, đẩy VIF lên 18.20). Nhóm kiên quyết **cấm đưa Loudness vào OLS** và loại bỏ Spectral Bandwidth, đưa VIF của Energy về **2.0914** và toàn bộ ma trận Model 1 về **$< 2.74 < 5.0$**, bảo toàn tính khả nghịch $(X^TX)^{-1}$ cho mô hình OLS.
* **Bảo toàn tính trung thực phương pháp luận:** Báo cáo ghi nhận rõ giới hạn vi phạm một phần tính độc lập của Kruskal-Wallis do **144 ca khúc xuất hiện đa thị trường (tạo ra 158 dòng quan sát trùng lặp dư thừa: 1,685 − 1,527)**. Sự khác biệt phân phối biên $P(X \mid \text{Market})$ chỉ mô tả đặc tính mẫu phân tầng của các ca khúc lọt chart, là tiền đề kinh tế lượng để chuyển giao cho TV1 kiểm định các số hạng tương tác thị trường.

### 5.2. Chuyển Giao Mạch Thuyết Trình Sang Thành Viên 3 (Lyrics Lead)
> *"Sau khi tín hiệu âm học đã được sàng lọc đa cộng tuyến và chứng minh tính dị biệt mẫu phân tầng, câu hỏi đặt ra là: Liệu yếu tố ca từ và điệu thức âm nhạc có tạo ra sự phân hóa tương tự giữa 3 quốc gia hay không? Xin mời Thành viên 3 trình bày phân hệ Ca từ NLP."*
