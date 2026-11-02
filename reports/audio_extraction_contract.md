# Hợp Đồng Đặc Trưng Âm Học Librosa (Audio Feature Contract v2)
**Dự án:** "What Makes a Hit Song?" — So sánh các yếu tố liên quan đến hiệu suất chart nhạc giữa Việt Nam – Mỹ – Hàn Quốc (2021–2026)  
**Tác giả phụ trách:** Thành viên 2 (Audio Features & Showcase Engineer)  
**Cập nhật lần cuối:** 2026-11-03 (Phiên bản v2 — Chuẩn hóa Database Supabase, Khử đa cộng tuyến & RQA Toàn bài)  
**Trạng thái Database:** Đồng bộ 100% với `public.audio_features` và `view_clean_modeling_dataset` trên Supabase (N = 1,912 tracks audio / N = 2,098 rows modeling).

---

## 1. Nguyên Tắc Cốt Lõi Về Dữ Liệu Âm Học (Acoustic Data Principles)

1. **Tuyệt đối không sử dụng Spotify Audio Features API:**  
   Kể từ tháng 11/2024, Spotify đã chính thức đóng vĩnh viễn endpoint truy vấn `audio-features`. Do đó, dự án kiên quyết loại bỏ hoàn toàn các chỉ số "hộp đen" độc quyền thương mại (như *danceability*, *valence*, *acousticness*, *instrumentalness*, *speechiness*).
2. **Audio-derived Proxy Features qua Librosa:**  
   100% đặc trưng âm học trong nghiên cứu là các chỉ số đại diện vật lý (proxy features) được trích xuất trực tiếp từ tín hiệu sóng âm thanh thực tế (WAV/MP3) thông qua thư viện xử lý tín hiệu số chuẩn học thuật `librosa` (Python).
3. **Tính nhất quán giữa Training và Inference:**  
   Mọi công thức toán học, tham số trích xuất tín hiệu số (Sampling rate, Hop length, Frame length, Thresholds) và quy tắc khử đa cộng tuyến được áp dụng thống nhất giữa pipeline huấn luyện mô hình và module suy diễn thời gian thực trên Web App Streamlit.

---

## 2. Danh Mục 17 Đặc Trưng Âm Học Chuẩn Hóa (17 Librosa Features Contract)

Sau khi loại bỏ 4 biến trùng lặp hoàn toàn ($r = 1.0$), tập đặc trưng âm học chính thức gồm **17 biến** (chia làm 4 nhóm âm học).

### Bảng Tổng hợp Đặc trưng Âm học

| STT | Tên Biến (`feature_name`) | Nhóm Âm Học | Đơn Vị | Miền Giá Trị Thực Tế | Hiện Diện trong OLS (Model 1, 2, 3) | Hiện Diện trong RF (Model 4) & UI Sliders |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `tempo` | Nhịp điệu | BPM | $[60.0, 210.0]$ | ✅ Có (Standardized) | ✅ Có |
| 2 | `onset_rate` | Nhịp điệu | onsets/s | $[1.2, 5.8]$ | ✅ Có (Standardized) | ✅ Có |
| 3 | `tempo_stability` | Nhịp điệu | tỉ lệ | $(0.0, 1.0]$ | ✅ Có (Standardized) | ✅ Có |
| 4 | `energy` | Động năng | biên độ RMS | $[0.05, 0.45]$ | ✅ Có (Standardized) | ✅ Có |
| 5 | `rms_variability` | Động năng | độ lệch chuẩn RMS | $[0.02, 0.18]$ | ✅ Có (Standardized) | ✅ Có |
| 6 | `loudness_proxy` | Âm sắc / Phổ | dBFS | $[-33.6, -4.9]$ | ❌ Loại (Tránh VIF $\approx 18.3$) | ✅ Có (Dùng chính trên UI) |
| 7 | `spectral_centroid` | Âm sắc / Phổ | Hz | $[1000, 4200]$ | ✅ Có (Standardized) | ✅ Có |
| 8 | `spectral_bandwidth` | Âm sắc / Phổ | Hz | $[1200, 3600]$ | ⚠️ Dự phòng VIF Check ($< 5.0$) | ✅ Có |
| 9 | `chord_change_rate` | Hòa âm | hợp âm/phút | $[15.0, 95.0]$ | ✅ Có (Standardized) | ✅ Có |
| 10 | `key_confidence` | Hòa âm | tương quan Pearson | $[0.25, 0.95]$ | ✅ Có (Standardized) | ✅ Có |
| 11 | `mode` | Hòa âm | nhị phân $\{0, 1\}$ | $0$ (Minor) hoặc $1$ (Major) | ✅ Giữ nguyên 0/1 (Không scale) | ✅ Có |
| 12 | `chroma_repetition_determinism` | RQA Giai điệu | tỉ lệ | $[0.02, 0.85]$ | ✅ Có (Standardized) | ✅ Có |
| 13 | `chroma_mean_pattern_duration` | RQA Giai điệu | giây | $[3.0, 18.0]$ | ✅ Có (Standardized) | ✅ Có |
| 14 | `chroma_longest_pattern_duration`| RQA Giai điệu | giây | $[3.5, 45.0]$ | ✅ Có (Standardized) | ✅ Có |
| 15 | `mfcc_repetition_determinism` | RQA Âm sắc | tỉ lệ | $[0.05, 0.70]$ | ✅ Có (Standardized) | ✅ Có |
| 16 | `mfcc_mean_pattern_duration` | RQA Âm sắc | giây | $[3.0, 15.0]$ | ✅ Có (Standardized) | ✅ Có |
| 17 | `mfcc_longest_pattern_duration` | RQA Âm sắc | giây | $[3.5, 38.0]$ | ✅ Có (Standardized) | ✅ Có |

---

## 3. Chi Tiết Định Nghĩa & Công Thức Toán Học Từng Nhóm

### 3.1. Nhóm Nhịp Điệu & Động Năng (Rhythm & Dynamic Intensity — 5 biến)

1. **`tempo` (Tốc độ phách):**
   * *Ý nghĩa:* Tốc độ phát lại ước lượng của ca khúc, tính bằng phách trên phút (BPM).
   * *Trích xuất Librosa:* `librosa.beat.beat_track(y=y, sr=sr, hop_length=512)`.
   * *Quy ước chuẩn hóa:* Giữ lại tên `tempo`, loại bỏ cột định danh cũ `tempo_bpm`.

2. **`onset_rate` (Tần suất khởi phát âm / Mật độ nốt):**
   * *Ý nghĩa:* Đo lường mức độ dồn dập của tiết tấu và lời hát (onsets per second).
   * *Trích xuất Librosa:*
     $$\text{onset\_envelope} = \text{librosa.onset.onset\_strength}(y=y, sr=sr)$$
     $$\text{onset\_rate} = \frac{\text{len}(\text{librosa.onset.onset\_detect}(\text{onset\_envelope}))}{\text{duration\_seconds}}$$

3. **`tempo_stability` (Độ ổn định nhịp điệu):**
   * *Ý nghĩa:* Thể hiện mức độ giữ nhịp đều đặn theo phách chuẩn máy (quantized grid) hay nhịp trôi (rubato/human drift).
   * *Công thức chuẩn hóa Database Supabase (v2):*
     $$\text{tempo\_stability} = \frac{1}{1 + \text{beat\_interval\_variability}} \in (0, 1]$$
     *Trong đó:* $\text{beat\_interval\_variability} = \frac{\sigma(\Delta t_{\text{beats}})}{\mu(\Delta t_{\text{beats}})}$ là hệ số biến thiên thời gian giữa các phách liên tiếp.
     *Diễn giải:* Khi $\text{variability} \to 0$ (nhịp máy hoàn hảo DAW), $\text{tempo\_stability} \to 1.0$.

4. **`energy` (Năng lượng tổng thể RMS):**
   * *Ý nghĩa:* Cường độ năng lượng sóng âm trung bình toàn bài.
   * *Trích xuất Librosa:*
     $$\text{RMS}_t = \sqrt{\frac{1}{K}\sum_{k=1}^K y[t \cdot H + k]^2}$$
     $$\text{energy} = \frac{1}{T}\sum_{t=1}^T \text{RMS}_t$$
   * *Quy ước chuẩn hóa:* Giữ lại tên `energy`, loại bỏ alias `rms_energy`.

5. **`rms_variability` (Biến độ dao động năng lượng):**
   * *Ý nghĩa:* Độ lệch chuẩn của chuỗi năng lượng khung $\text{RMS}_t$, phản ánh sự tương phản động học giữa đoạn êm dịu (Verse) và bùng nổ (Chorus/Drop).
   * *Công thức:*
     $$\text{rms\_variability} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\text{RMS}_t - \text{energy})^2}$$

---

### 3.2. Nhóm Âm Sắc & Phổ Tần Số (Spectral & Timbre — 3 biến)

6. **`loudness_proxy` (Độ lớn biểu kiến theo thang Decibel):**
   * *Bản chất vật lý:* Ước lượng độ lớn toàn dải theo thang decibel tương đối so với thang cực đại (dBFS).
   * *Công thức chuẩn hóa Database Supabase (v2):*
     $$\text{loudness\_proxy} = 20 \times \log_{10}(\max(\text{RMS}, 10^{-10})) \quad [\text{dBFS}]$$
   * *Miền giá trị thực tế:* Giao động từ $-33.6\text{ dBFS}$ đến $-4.9\text{ dBFS}$ (Trung vị $\approx -12.3\text{ dBFS}$).
   * *Quy tắc kiểm soát đa cộng tuyến khắt khe (Energy vs Loudness):*
     * Do $\text{Corr}(\text{energy}, \text{loudness\_proxy}) = 0.968 \approx 0.971$, nếu cùng đưa vào OLS sẽ gây bùng nổ $\text{VIF} \approx 18.3 > 5.0$.
     * **Trong Model 1, 2, 3 (OLS):** **Loại bỏ hoàn toàn `loudness_proxy`**, chỉ giữ `energy` đại diện cho công suất sóng âm để bảo đảm mọi $\text{VIF} < 5.0$.
     * **Trong Model 4 (Random Forest), Tab 1 & Tab 2:** **Giữ lại `loudness_proxy`** vì mô hình cây không suy thoái do tương quan đơn điệu, đồng thời thang đo dBFS trực quan và thân thuộc hơn cho người dùng khi kéo slider.

7. **`spectral_centroid` (Trọng tâm phổ tần số — Độ sáng âm sắc):**
   * *Ý nghĩa:* Tần số trung tâm đóng vai trò trọng tâm của năng lượng phổ, biểu thị độ "sáng" (bright) hay "tối/trầm" (dark/warm) của bản phối.
   * *Công thức Librosa:*
     $$f_c(t) = \frac{\sum_{n} f_n \cdot |S(n, t)|}{\sum_{n} |S(n, t)|}$$
     $$\text{spectral\_centroid} = \frac{1}{T}\sum_{t=1}^T f_c(t) \quad [\text{Hz}]$$

8. **`spectral_bandwidth` (Độ rộng dải phổ tần số):**
   * *Ý nghĩa:* Độ rộng dải tần số bao quanh trọng tâm phổ, phản ánh mức độ lấp đầy các dải tần của bản phối.
   * *Công thức Librosa:*
     $$BW(t) = \left( \frac{\sum_n (f_n - f_c(t))^2 \cdot |S(n, t)|}{\sum_n |S(n, t)|} \right)^{1/2}$$
   * *Quy tắc kiểm định VIF thực tế (v2 Protocol):*
     * Vì $\text{Corr}(\text{spectral\_centroid}, \text{spectral\_bandwidth}) = 0.897$, trước khi fit chính thức OLS, TV1 bắt buộc chạy hàm kiểm tra VIF trên ma trận biến độc lập.
     * **Nếu $\text{VIF} \ge 5.0$:** Loại bỏ `spectral_bandwidth` khỏi ma trận hồi quy OLS (chỉ giữ `spectral_centroid`), nhưng vẫn giữ lại trong Random Forest và Web UI.
     * **Nếu $\text{VIF} < 5.0$:** Giữ lại cả 2 trong OLS và ghi nhận số liệu minh bạch vào báo cáo.

---

### 3.3. Nhóm Hòa Âm & Cấu Trúc (Tonal & Harmony — 3 biến)

9. **`chord_change_rate` (Tốc độ chuyển hợp âm):**
   * *Ý nghĩa:* Tần suất dịch chuyển hợp âm ước lượng trên mỗi phút (hợp âm/phút).
   * *Phương pháp:* Tính toán thông qua khoảng cách cosin trên ma trận Chromagram 12 bán âm được làm mịn theo thời gian.

10. **`key_confidence` (Độ tự tin dự đoán chủ âm):**
    * *Ý nghĩa:* Hệ số tương quan Pearson lớn nhất giữa vector Chroma trung bình của bài với 24 mẫu phác đồ chủ âm chuẩn Krumhansl-Schmuckler (12 Major + 12 Minor profiles).
    * *Miền giá trị:* $[0.0, 1.0]$. Giá trị càng cao thể hiện bài hát có trung tâm hòa âm rõ ràng, không bị mờ nhạt hoặc pha tạp phức điệu.

11. **`mode` (Điệu thức):**
    * *Ý nghĩa:* Mã hóa nhị phân cấu trúc điệu thức:
      $$\text{mode} = \begin{cases} 1 & \text{giọng Trưởng (Major)} \\ 0 & \text{giọng Thứ (Minor)} \end{cases}$$
    * *Quy tắc chuẩn hóa sống còn:* **Không đưa biến `mode` qua `StandardScaler`**. Biến nhị phân phải giữ nguyên $0$ và $1$ để hệ số hồi quy $\alpha$ trong OLS mang đúng ý nghĩa chênh lệch điểm trung bình giữa bài hát giọng Trưởng so với giọng Thứ.

---

### 3.4. Nhóm Cấu Trúc Lặp Lại RQA (Recurrence Quantification Analysis — 6 biến)

Được trích xuất bằng cách xây dựng Ma trận Tái phát sinh Chéo (Cross-Recurrence Matrix) trên 2 biểu diễn tín hiệu:
* **Chroma Recurrence:** Biểu diễn sự lặp lại của tiến trình hợp âm và giai điệu (Harmonic repetition).
* **MFCC Recurrence:** Biểu diễn sự lặp lại của âm sắc, hòa âm tổng phổ và kết cấu nhạc cụ (Timbral repetition).

Các thông số vận hành tiêu chuẩn:
* $SR = 22,050\text{ Hz}$, $Hop = 512$, $K_{\text{chroma}} = 12$, $K_{\text{mfcc}} = 20$.
* Ngưỡng khoảng cách tối thiểu giữa 2 lần xuất hiện: $\text{MIN\_SEPARATION\_SECONDS} = 40.0\text{s}$.
* Độ dài đường chéo tối thiểu để xác nhận là đoạn lặp: $\text{MIN\_DIAGONAL\_SECONDS} = 3.0\text{s}$.

12. **`chroma_repetition_determinism` ($DET_{\text{chroma}}$):**
    * *Ý nghĩa:* Tỉ lệ các điểm tái phát sinh nằm trên các đường chéo dài $\ge 3.0\text{s}$ trong ma trận Chroma.
    * *Miền giá trị:* $[0.0, 1.0]$. Đo lường mức độ cấu trúc giai điệu được tổ chức thành các phân đoạn điệp khúc lặp lại có chủ đích.

13. **`chroma_mean_pattern_duration`:**
    * *Ý nghĩa:* Thời lượng trung bình (tính bằng giây) của các đoạn giai điệu lặp lại được phát hiện.

14. **`chroma_longest_pattern_duration`:**
    * *Ý nghĩa:* Độ dài của phân đoạn giai điệu lặp lại dài nhất trong bài hát (thường tương ứng với 1 vòng Chorus/Hook hoàn chỉnh 8–16 ô nhịp, trung vị $\approx 10\text{–}12\text{s}$).

15. **`mfcc_repetition_determinism` ($DET_{\text{mfcc}}$):**
    * *Ý nghĩa:* Tỉ lệ các điểm tái phát sinh trên ma trận MFCC, đo lường độ lặp lại của kết cấu phối khí / loop sản xuất âm nhạc.

16. **`mfcc_mean_pattern_duration`:**
    * *Ý nghĩa:* Thời lượng trung bình (giây) của các loop âm sắc.

17. **`mfcc_longest_pattern_duration`:**
    * *Ý nghĩa:* Đoạn lặp âm sắc/bản phối dài nhất trong toàn bộ tác phẩm.

---

## 4. Danh Mục 4 Biến Bị Loại Bỏ Triệt Để (Zero Multicollinearity Rules)

Để tránh hiện tượng ma trận suy biến (Singular Matrix) và đa cộng tuyến hoàn hảo ($r = 1.0, \text{VIF} \to \infty$), 4 biến sau bị loại khỏi danh mục mô hình:

| Biến Bị Loại Bỏ | Biến Đại Diện Được Giữ Lại | Nguyên Nhân Kỹ Thuật |
| :--- | :--- | :--- |
| `tempo_bpm` | `tempo` | Trùng lặp hoàn toàn ($r = 1.0$). Giữ tên chuẩn `tempo`. |
| `rms_energy` | `energy` | Trùng lặp hoàn toàn ($r = 1.0$). Giữ tên chuẩn `energy`. |
| `dynamic_range` | `loudness_proxy` / `rms_variability` | Khắc phục lỗi gán nhầm giá trị cũ; bản chất dynamic range không phản ánh đúng loudness theo dBFS. |
| `beat_interval_variability` | `tempo_stability` | Trùng lặp toán học qua hàm nghịch đảo chuẩn hóa $1 / (1 + x)$. Giữ `tempo_stability` để trực quan theo chiều thuận. |

*Lưu ý về `key` và `duration_seconds`:*
* `key` (12 nốt chủ âm): Đòi hỏi tạo 11 biến giả (dummies). Kiểm định Chi-square độc lập cho thấy nốt chủ âm không có ý nghĩa phân hóa thị hiếu ($p = 0.532$), nên loại khỏi OLS để tránh bùng nổ số chiều.
* `duration_seconds`: Được lưu trữ cho mục đích kiểm soát, không đưa vào danh mục 17 biến âm học chính thức của mô hình cốt lõi.

---

## 5. Nguyên Tắc Xử Lý Âm Thanh Toàn Bài (Full-Track Processing Principle)

### 5.1. Cơ sở Khoa học Chuyển đổi từ Cắt đoạn ngắn sang Xử lý Toàn bài
* Trong giai đoạn thử nghiệm sơ khởi, phương án cắt 3 đoạn ngắn ($4\text{–}6\text{s}$) đã bộc lộ khuyết điểm nghiêm trọng: **6 đặc trưng RQA bị suy thoái và mất ý nghĩa thống kê**.
* *Lý do toán học:*
  Thuật toán RQA yêu cầu:
  $$\Delta t_{\text{separation}} \ge 40.0\text{s} \quad \text{và} \quad L_{\text{diagonal}} \ge 3.0\text{s}$$
  Để phát hiện được một đoạn lặp lại giữa Verse và Chorus hoặc giữa Chorus 1 và Chorus 2, tín hiệu âm thanh liên tục bắt buộc phải dài tối thiểu:
  $$T_{\text{min}} \ge 40.0 + 3.0 = 43.0\text{s}$$
  Một đoạn audio ngắn $4\text{–}6\text{s}$ không thể nào chứa đựng khoảng cách $40\text{s}$, dẫn đến $DET \to 0$ giả tạo và không thể trích xuất được `pattern_duration`.
* **Quyết định (Chốt Phương án A1):** Module suy diễn `src/features/audio_extractor.py` và Tab 3 Streamlit chuyển sang **xử lý toàn bộ bài hát** (tối đa 6 phút đầu tiên qua tham số `duration=360` của `librosa.load`).

### 5.2. Rào Chắn Giới Hạn Tải Lên (Inference Safeguards)
Để bảo vệ RAM server (giới hạn $< 1.0\text{ GB}$ trên Streamlit Cloud) và ngăn chặn tình trạng tràn bộ nhớ:
1. **Rào chắn kích thước tệp:** Giới hạn dung lượng upload tối đa $\le 15\text{ MB}$.
2. **Rào chắn thời lượng:** Tham số `duration=360` (tối đa 360 giây / 6 phút). Nếu người dùng tải lên tệp dài (như podcast, DJ non-stop mix), librosa chỉ decode đúng 6 phút đầu và dừng lại.
3. **Cơ chế Caching dựa trên Hash (SHA-256):**
   Sử dụng `@st.cache_data` kèm hàm băm SHA-256 nội dung nhị phân của tệp âm thanh. Khi người dùng thao tác chuyển tab hoặc thay đổi tùy chọn hiển thị trên UI, Streamlit sẽ trả về ngay kết quả 17 đặc trưng đã lưu đệm thay vì chạy lại thuật toán RQA tốn kém tài nguyên.

---

## 6. Hướng Dẫn Tích Hợp Trên Giao Diện Web Streamlit

### 6.1. Tab 1: Showcase Thư Viện Bài Mẫu (Radar Chart Scaling)
* Do `loudness_proxy` mang giá trị âm ($[-25.0, -6.0]\text{ dBFS}$), không thể vẽ trực tiếp lên trục Radar Chart vốn yêu cầu gốc tọa độ dương $0.0$.
* **Giải pháp chuẩn hóa:** TV2 áp dụng Min-Max Normalization về đoạn $[0.0, 1.0]$ trên tập 15 bài mẫu test cho 6 trục đặc trưng đại diện:
  1. `tempo` (Nhịp điệu)
  2. `energy` (Năng lượng RMS)
  3. `loudness_proxy` (Độ lớn dBFS)
  4. `onset_rate` (Mật độ nốt)
  5. `spectral_centroid` (Độ sáng phổ)
  6. `chord_change_rate` (Tốc độ chuyển hợp âm)

### 6.2. Tab 2: Giả Lập Sáng Tác (What-If Sliders)
* Để chống lỗi phân phối ngoại lai (Out-of-Distribution) do tương quan $r = 0.968$ giữa Energy và Loudness, giao diện chỉ hiển thị **1 thanh trượt duy nhất đại diện cho độ lớn:**
  * **Loudness Slider:** Phạm vi $[-25.0, -6.0]\text{ dBFS}$.
  * **Hàm ánh xạ ngược cho Random Forest:**
    $$\text{energy}_{\text{inferred}} = 10^{\frac{\text{Loudness}}{20}}$$
  * Nhờ đó, cả 2 mô hình (OLS chỉ dùng `energy`, RF dùng cả `energy` và `loudness_proxy`) đều nhận vector đầu vào tự nhiên và hoàn toàn nhất quán.

---
*Tài liệu này là cam kết kỹ thuật chính thức giữa Thành viên 2 với Thành viên 1 (Modeling OLS), Thành viên 3 (Lyrics & Sliders) và Thành viên 4 (Database & Deployment).*
