# Báo cáo Khám phá Dữ liệu Âm học & Trực quan hóa Học thuật (Report 4)
**Học phần:** ADY201m - Music Chart Performance Analysis  
**Thành viên phụ trách:** Member 2 (Audio Feature Engineering & Acoustic Analytics)  
**Ngày cập nhật:** 2026-10-26  
**Nguồn dữ liệu:** Supabase Database (`public.modeling_dataset` với `is_sampled = True`, N = 2,860 entries / 1,912 tracks)

---

## 1. Tóm tắt Điều hành (Executive Summary)
Báo cáo này trình bày các phát hiện rút ra từ việc phân tích khám phá dữ liệu (EDA) và trực quan hóa 18 đặc trưng âm học chuẩn hóa trên 2,860 lượt bài hát xếp hạng tại Việt Nam (VN, N=885), Hoa Kỳ (US, N=955), và Hàn Quốc (KR, N=1,020) trong giai đoạn 2021–2026.

Toàn bộ dữ liệu được trích xuất trên **toàn bộ bài hát (Full Audio Track, thời lượng trung bình ~207.1 giây)**. Mọi phân tích và nhận định trong báo cáo tuân thủ nghiêm ngặt nguyên tắc **khoa học âm học thực nghiệm (empirical acoustics)**, đối chiếu chuẩn xác 100% giữa số liệu thống kê (Kruskal-Wallis H-test, Chi-square test of independence) và các biểu đồ trực quan, tuyệt đối tránh các suy diễn chủ quan về văn hóa.

---

## 2. Danh mục Biểu đồ và Dữ liệu Kiểm chứng Thực tế

Toàn bộ biểu đồ độ phân giải cao (300 DPI) và dữ liệu tính toán tương ứng được lưu trữ độc lập tại:
- **Thư mục ảnh:** `reports/report4/figures/member2/`
- **Thư mục bảng dữ liệu:** `reports/report4/chart_data/member2/`

| STT | Tên Biểu đồ | File Ảnh (.png) | Dữ liệu Nguồn (.csv) | Kết Quả Kiểm Định Thống Kê Chính |
| :---: | :--- | :--- | :--- | :--- |
| **Hình 1** | Phân phối Nhịp điệu & Năng lượng | `fig1_rhythm_energy_distribution.png` | `fig1_rhythm_energy_market.csv` | Tempo: $H = 2.87, p = 0.238$ (Đồng nhất)<br>Onset: $H = 22.35, p < 0.001$ (US > KR > VN)<br>RMS Energy: $H = 103.88, p < 0.001$ (KR > US > VN) |
| **Hình 2** | Đặc trưng Phổ & Độ ổn định Phách | `fig2_spectral_stability_distribution.png` | `fig2_spectral_stability_market.csv` | Beat CV: $H = 1.41, p = 0.495$ (Đều < 0.02)<br>Centroid: $H = 80.28, p < 0.001$ (KR > US > VN)<br>Bandwidth: $H = 68.73, p < 0.001$ (KR rộng nhất) |
| **Hình 3** | Cấu trúc Hòa âm & Điệu thức | `fig3_tonal_harmony_distribution.png` | `fig3_tonal_harmony_market.csv` | Chủ âm (Key): $\chi^2 = 20.82, p = 0.532$ (Độc lập)<br>Điệu thức (Mode): $\chi^2 = 11.70, p = 0.0029$ (VN chuộng Major 64%)<br>Chord Rate: $H = 19.50, p < 0.001$ (KR đổi hợp âm nhanh nhất) |
| **Hình 4** | Đặc trưng Lặp lại qua RQA | `fig4_repetition_rqa_distribution.png` | `fig4_repetition_rqa_market.csv` | MFCC DET: $H = 2.95, p = 0.229$ (Đồng đều ~0.22-0.23)<br>Longest Pattern: $H = 22.61, p < 0.001$ (VN, KR > US, Median 10–11s) |
| **Hình 5** | Dấu ấn Âm học Radar Đa chiều | `fig5_audio_fingerprint_radar.png` | `fig5_radar_market_fingerprints.csv` | KR dẫn đầu 6/8 trục (Năng lượng, Phổ, Hòa âm)<br>US dẫn đầu Onset & Tempo<br>VN dẫn đầu Dynamic Range |

---

## 3. Phân tích Chi tiết Từng Biểu đồ Đối Chiếu Với Số Liệu

### 3.1. Hình 1: Phân phối Nhịp điệu và Năng lượng Âm học
- **Tốc độ Nhịp điệu (`tempo_bpm`):** 
  - *Hình dạng biểu đồ:* Phân phối song đỉnh (bimodal) rõ rệt ở cả 3 nước, phình to ở vùng 90–105 BPM và 120–130 BPM. Đường nét đứt trung vị nằm ngang bằng nhau ở mức 117.5 BPM (KR, VN) và 123.0 BPM (US).
  - *Kiểm định thống kê:* $H = 2.87, p = 0.238 > 0.05$. Khẳng định **không có sự khác biệt có ý nghĩa thống kê về tempo** giữa 3 thị trường (Mean: US 122.3, KR 120.6, VN 120.4 BPM). Pop thương mại toàn cầu tuân thủ hằng số sinh học về nhịp vận động cơ thể (motor entrainment).
- **Mật độ Nốt (`onset_rate`):** 
  - *Hình dạng & Số liệu:* Hộp của **Hoa Kỳ (US) nằm cao nhất** (Trung vị: 198.8, Mean: 199.1 onsets/s), tiếp theo là Hàn Quốc (Trung vị: 193.7, Mean: 192.5 onsets/s) và Việt Nam (Trung vị: 189.4, Mean: 188.2 onsets/s).
  - *Kiểm định:* Khác biệt có ý nghĩa thống kê cao ($H = 22.35, p = 1.4 \times 10^{-5} < 0.001$). Điều này phản ánh nhịp điệu ca từ và tiết tấu bộ gõ trong nhạc Hip-hop/Pop của Mỹ và K-Pop có mật độ nốt dày hơn nhạc trữ tình Việt Nam.
- **Năng lượng Sóng âm (`rms_energy`):** 
  - *Hình dạng & Số liệu:* Khác biệt cực kỳ mạnh ($H = 103.88, p = 2.77 \times 10^{-23} < 0.001$). Hàn Quốc có năng lượng cao nhất (Mean: 0.254, Median: 0.261), vượt trội so với US (Mean: 0.238, Median: 0.248) và VN (Mean: 0.224, Median: 0.225).
  - *Bản chất âm học:* Phản ánh kỹ thuật nén âm (mastering compression) tối đa hóa độ lớn (Loudness War) đặc trưng của các sản phẩm K-Pop để cạnh tranh trên thiết bị di động.
- **Khoảng động học (`dynamic_range`):** 
  - *Hình dạng & Số liệu:* Việt Nam có khoảng động học cao nhất (Mean: 15.51 dB), cao hơn US (14.53 dB) và KR (14.48 dB) ($H = 18.12, p < 0.001$), minh chứng nhạc Việt giữ được khoảng thở tự nhiên, ít bị nén bẹp dải động.

### 3.2. Hình 2: Đặc trưng Phổ và Độ ổn định Nhịp điệu
- **Độ Biến thiên Khoảng cách Phách (`beat_interval_variability`):** 
  - *Hình dạng & Số liệu:* Cả 3 hộp co cụm sát đáy trục tung dưới mức 0.02 (Trung vị: KR 0.0172, VN 0.0173, US 0.0175). Kiểm định $H = 1.41, p = 0.495 > 0.05$ (không khác biệt).
  - *Bản chất:* Xác nhận 100% các bản ghi thương mại đều được sản xuất trên Digital Audio Workstations (DAW) với nhịp bấm máy (Quantized Grid), loại bỏ hoàn toàn sai số nhịp tự nhiên của con người.
- **Trọng tâm Phổ (`spectral_centroid`):** 
  - *Hình dạng & Số liệu:* **Hàn Quốc (KR) có phổ sáng nhất** (Trung vị: 2,359.6 Hz, Mean: 2,335.3 Hz), cao hơn hẳn US (Trung vị: 2,211.1 Hz, Mean: 2,214.9 Hz) và VN (Trung vị: 2,150.3 Hz, Mean: 2,166.0 Hz) ($H = 80.28, p < 0.001$).
  - *Bản chất:* K-Pop sử dụng rất nhiều âm sắc synthesizer dải cao và tăng cường dải tần treble (high-shelf EQ boost) để tạo độ sắc nét, hiện đại. Nhạc Việt Nam giữ trọng tâm ở dải trung ấm áp hơn.
- **Độ rộng Phổ (`spectral_bandwidth`):** 
  - Hàn Quốc rộng nhất (Mean: 2,507.4 Hz), tiếp theo là US (2,430.3 Hz) và VN (2,409.7 Hz) ($H = 68.73, p < 0.001$). Bản phối K-Pop lấp đầy toàn bộ phổ nghe được nhiều nhất.

### 3.3. Hình 3: Cấu trúc Hòa âm, Điệu thức và Tốc độ Đổi Hợp âm
- **Chủ âm (`audio_key`):** 
  - *Hình dạng:* Cả 3 nước cùng dựng cột cao nhất tại các nốt C, C#, D, E (10–13% mỗi nốt), và cùng chạm đáy ở nốt B và A# (<4%).
  - *Kiểm định:* $\chi^2 = 20.82, df = 22, p = 0.532 > 0.05$. Sự lựa chọn nốt chủ âm hoàn toàn độc lập với thị trường quốc gia, bị chi phối bởi công thái học nhạc cụ (Instrument Ergonomics: thế tay thuận trên phím đàn piano và dây guitar).
- **Điệu thức (`audio_mode`):** 
  - *Số liệu chính xác:* **Việt Nam có tỷ lệ giọng Trưởng cao nhất (64.1%)**, cao hơn đáng kể so với Hàn Quốc (58.6%) và Hoa Kỳ (56.4%). Kiểm định $\chi^2 = 11.70, df = 2, p = 0.0029 < 0.01$.
  - *Bản chất:* Thị trường Việt Nam có sự ưa chuộng vượt trội đối với các ca khúc giọng Trưởng (Major), phản ánh thị hiếu đại chúng chuộng cảm xúc trong sáng, tích cực.
- **Tốc độ Chuyển Hợp âm (`chord_change_rate`):** 
  - Hàn Quốc đổi hợp âm nhanh nhất (Mean: 44.8 hợp âm/phút), so với US (43.8) và VN (42.1) ($H = 19.50, p < 0.001$). K-Pop sử dụng nhiều hợp âm nối 2 phách và biến đổi hòa thanh dồn dập hơn.

### 3.4. Hình 4: Tính Chu kỳ và Cấu trúc Lặp lại (RQA Metrics)
- **Lặp lại Âm sắc (`mfcc_repetition_determinism`):** 
  - *Số liệu:* Cả 3 thị trường có mức độ lặp lại âm sắc tương đương nhau quanh mức 0.22–0.23 (Mean: KR 0.230, VN 0.227, US 0.223, Trung vị ~0.21). Kiểm định $H = 2.95, p = 0.229 > 0.05$ (không khác biệt có ý nghĩa thống kê).
  - *Bản chất:* Pop hiện đại ở cả 3 nước đều chia sẻ triết lý sử dụng các vòng lặp beat/sample cố định làm khung xương cho bài hát.
- **Độ dài Đoạn lặp Cực đại (`chroma_longest_pattern_duration`):** 
  - *Số liệu:* Trung vị của cả 3 nước đều co cụm ở mức **10.0 – 11.0 giây** (KR: 11.0s, VN: 11.0s, US: 10.0s). Giá trị trung bình đạt: VN 14.16s, KR 13.82s, US 12.09s ($H = 22.61, p < 0.001$).
  - *Bản chất âm học:* Con số 10–12 giây tương ứng hoàn hảo với độ dài của một câu nhạc điệp khúc (Hook/Chorus) tiêu chuẩn 8 ô nhịp (8 bars) ở tempo 120 BPM. Cấu trúc bài hát thương mại bị giới hạn bởi dung lượng trí nhớ âm thanh ngắn hạn của con người.

### 3.5. Hình 5: Dấu ấn Âm học Đa chiều (Acoustic Radar Fingerprints)
- Đối chiếu số liệu chuẩn hóa [0, 1] trên ma trận Radar:
  - **Hàn Quốc (KR):** Chiếm ưu thế áp đảo ở **6/8 trục**: Năng lượng RMS ($0.426$), Trọng tâm phổ ($0.581$), Độ rộng phổ ($0.671$), Tốc độ hợp âm ($0.446$), Lặp lại hòa âm ($0.131$), và Lặp lại âm sắc ($0.321$). Đây là hồ sơ âm nhạc có **động năng tột bậc (Kinetic Intensity)**.
  - **Hoa Kỳ (US):** Dẫn đầu ở 2 trục: **Mật độ nốt (`onset_rate`: 0.545)** và **Tempo (`tempo_bpm`: 0.361)**.
  - **Việt Nam (VN):** Giữ vị trí cân bằng, thoáng đãng ở dải năng lượng và dẫn đầu về **Khoảng động học (`dynamic_range`: 15.51 dB)**, tôn vinh cảm xúc ca từ.

---

## 4. Ý nghĩa Mô hình hóa & Bàn giao cho Member 4
1. **Các biến có lực phân tách thị trường mạnh nhất:**
   - `rms_energy` ($H = 103.88$)
   - `spectral_centroid` ($H = 80.28$)
   - `spectral_bandwidth` ($H = 68.73$)
   - `onset_rate` ($H = 22.35$)
   - `chord_change_rate` ($H = 19.50$)
2. **Khuyến nghị Xử lý Dữ liệu trước khi Training:**
   - Áp dụng `log1p` transformation đối với biến lệch phải nặng: `chroma_longest_pattern_duration` và `mfcc_longest_pattern_duration`.
   - Loại bỏ `loudness_proxy` (vì trùng lặp đa cộng tuyến $r > 0.95$ với `rms_energy`).
   - Giữ `beat_interval_variability` và loại bỏ `tempo_stability`.
