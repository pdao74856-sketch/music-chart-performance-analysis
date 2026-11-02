# Báo cáo Kiểm kê Dữ liệu Âm học & Thống kê Mô tả (Report 3)
**Học phần:** ADY201m - Music Chart Performance Analysis  
**Thành viên phụ trách:** Member 2 (Audio Feature Engineering & Signal Processing)  
**Ngày cập nhật:** 2026-10-25  
**Nguồn dữ liệu:** Supabase Database (`public.modeling_dataset` với `is_sampled = True`)

---

## 1. Tổng quan Nghiên cứu & Phạm vi Trích xuất
Báo cáo này kiểm định toàn bộ tập dữ liệu đặc trưng âm học được trích xuất từ âm thanh chuẩn hóa (44.1 kHz, mono) của các bài hát lọt vào bảng xếp hạng âm nhạc tại 3 thị trường: **Việt Nam (VN)**, **Hoa Kỳ (US)**, và **Hàn Quốc (KR)** giai đoạn 2021–2026.

Dữ liệu âm thanh được phân tích trên **toàn bộ thời lượng bài hát (Full Audio Track)** với thời lượng trung bình là 207.1 giây (~3.45 phút/bài). Điều này giúp nắm bắt toàn diện sự phát triển cấu trúc tác phẩm từ Intro, Verse, Chorus, Bridge cho đến Outro, đặc biệt giúp các phép tính lặp lại cấu trúc (RQA) phản ánh chính xác chu kỳ điệp khúc trên toàn bộ ca khúc.

---

## 2. Kiểm kê Dữ liệu Âm học (Audio Inventory)

| Thị trường | Tổng số bản ghi BXH | Số Track duy nhất | Có Audio (has_audio) | Đầy đủ Audio & Lyrics | Thời lượng phân tích TB (s) | Tỷ lệ trích xuất thành công |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Việt Nam (VN)** | 885 | 885 | 885 | 885 | 216.4 | 100.0% |
| **Hoa Kỳ (US)** | 955 | 955 | 955 | 955 | 202.8 | 100.0% |
| **Hàn Quốc (KR)** | 1,020 | 1,020 | 1,020 | 1,020 | 203.1 | 100.0% |
| **Toàn bộ (Overall)** | **2,860** | **1,912** | **2,860** | **2,860** | **207.1** | **100.0%** |

*Ghi chú kiểm toán dữ liệu:*
- Tổng số track nhạc Spotify độc bản được lấy mẫu: **1,912 tracks**.
- Tổng số bản ghi xuất hiện trên bảng xếp hạng (chart entries): **2,860 bản ghi** (do một số ca khúc quốc tế như hit US-UK, K-Pop đồng thời lọt vào BXH Việt Nam hoặc toàn cầu).
- **Tỷ lệ hoàn thành (Completeness):** Đạt **100.0%** (0 bản ghi bị lỗi trích xuất, 0 bản ghi có giá trị rỗng/NULL trên 20 chiều đặc trưng âm học).

---

## 3. Từ điển Dữ liệu Đặc trưng Âm học (Audio Data Dictionary)

Bảng phân loại chi tiết các biến phục vụ cho Member 4 (Feature Selection & Predictive Modeling):

| Tên trường (Field Name) | Kiểu DL | Đơn vị / Miền giá trị | Phương pháp Tín hiệu & Công thức | Ý nghĩa Âm học / Cảm thụ |
| :--- | :--- | :--- | :--- | :--- |
| `tempo_bpm` | Float | BPM [58 - 235] | Dynamic Programming Beat Tracking qua Tempogram | Tốc độ nhịp điệu cơ bản của bài hát. |
| `beat_interval_variability` | Float | Tỷ số vô hướng [0.007 - 0.06] | Hệ số biến thiên khoảng cách phách: $CV = \frac{\sigma_{\Delta t}}{\mu_{\Delta t}}$ | Độ ổn định nhịp; giá trị thấp chứng tỏ nhạc thu trên Beat máy (Quantized Grid). |
| `tempo_stability` | Float | Tỷ số vô hướng [0.007 - 0.06] | Nghịch đảo độ biến động phách (đồng bộ với beat_interval_variability) | Chỉ số thể hiện tính ổn định cơ học của nhịp điệu. |
| `onset_rate` | Float | Events/sec [25 - 345] | Spectral Flux Onset Peak Picking trên khung thời gian | Mật độ nốt nhạc / sự kiện âm thanh xuất hiện trong 1 giây. |
| `rms_energy` | Float | Biên độ [0.02 - 0.57] | Căn bậc hai trung bình bình phương biên độ sóng: $\sqrt{\frac{1}{N}\sum x[n]^2}$ | Năng lượng âm học vật lý tổng thể của bản phối. |
| `rms_variability` | Float | Độ lệch chuẩn [0.007 - 0.25] | Độ lệch chuẩn của chuỗi RMS frames | Sự biến thiên độ lớn giữa các phân đoạn (đoạn êm dịu vs điệp khúc bùng nổ). |
| `dynamic_range` | Float | dB [2.2 - 193.5] | Chênh lệch năng lượng phân vị: $P_{95}(RMS_{dB}) - P_{05}(RMS_{dB})$ | Khoảng động học của bản thu âm trước và sau mastering. |
| `loudness_proxy` | Float | dB [2.2 - 193.5] | Log-scaled Energy Proxy (đồng bộ với Dynamic Range) | Đại diện cho mức âm lượng cảm nhận tổng thể. |
| `spectral_centroid` | Float | Hz [544 - 3628] | Trọng tâm phổ tần số: $\frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$ | "Độ sáng" (Brightness) của âm sắc; giá trị cao tương ứng nhiều âm treble. |
| `spectral_bandwidth` | Float | Hz [908 - 3291] | Bán kính mở rộng phổ: $\sqrt{\frac{\sum (f - C)^2 |X(f)|}{\sum |X(f)|}}$ | Độ dày và độ đầy đặn (Fullness) của dải phổ âm thanh. |
| `audio_key` | Categorical | 12 Pitch Classes (C, C#, ..., B) | Krumhansl-Schmuckler Key Finding thuật toán tương quan | Nốt chủ âm chính của tác phẩm. |
| `audio_mode` | Categorical | `major` (Trưởng) / `minor` (Thứ) | Krumhansl-Schmuckler Profile Matching | Thể thức điệu thức; ảnh hưởng trực tiếp đến sắc thái vui tươi/trầm buồn. |
| `key_confidence` | Float | Pearson $r$ [0.22 - 0.98] | Hệ số tương quan cực đại giữa Chroma vector và K-S Profile | Độ tin cậy toán học của việc nhận diện giọng điệu. |
| `chord_change_rate` | Float | Hợp âm/phút [12 - 87] | Beat-synchronous Triad Matching qua Cosine Similarity | Tốc độ chuyển đổi công năng hòa thanh (Harmonic Rhythm). |
| `chroma_repetition_determinism` | Float | Tỷ số [0.0 - 0.93] | RQA Recurrence Determinism trên ma trận Chroma Cosine Recurrence | Tỷ lệ cấu trúc hòa âm được lặp lại có chu kỳ (Verse-Chorus symmetry). |
| `chroma_mean_pattern_duration` | Float | Giây [0.0 - 17.2] | Độ dài trung bình các đường chéo lặp lại hòa âm trong ma trận RP | Thời lượng trung bình của một câu nhạc hòa âm được tái hiện. |
| `chroma_longest_pattern_duration`| Float | Giây [0.0 - 72.0] | Độ dài cực đại của đường chéo lặp lại hòa âm | Độ dài của đoạn loop hoặc điệp khúc hòa âm dài nhất. |
| `mfcc_repetition_determinism` | Float | Tỷ số [0.0 - 0.72] | RQA Recurrence Determinism trên không gian MFCC (Timbre) | Tỷ lệ âm sắc/nhạc cụ được lặp lại một cách máy móc theo vòng lặp loop. |
| `mfcc_mean_pattern_duration` | Float | Giây [0.0 - 11.7] | Độ dài trung bình các chuỗi lặp lại âm sắc trong ma trận RP | Thời lượng của một mô-típ âm sắc lặp lại. |
| `mfcc_longest_pattern_duration` | Float | Giây [0.0 - 72.0] | Độ dài cực đại của chuỗi lặp lại âm sắc | Đoạn loop âm sắc/beat nền dài nhất trong ca khúc. |

---

## 4. Tóm tắt Chỉ số Thống kê Trọng yếu (Key Statistics by Market)

| Đặc trưng (Feature) | VN Mean (SD) | US Mean (SD) | KR Mean (SD) | Overall Mean (SD) | Kiểm định Khác biệt ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `tempo_bpm` | 119.8 (26.3) | 122.4 (26.8) | 121.1 (25.1) | 121.1 (26.0) | $p = 0.092$ (Không khác biệt ý nghĩa) |
| `onset_rate` (events/s) | 185.3 (47.2) | 192.1 (45.1) | 201.5 (45.7) | 193.3 (46.5) | **$p < 0.001$** (KR cao vượt trội) |
| `rms_energy` | 0.228 (0.07) | 0.237 (0.07) | 0.251 (0.07) | 0.239 (0.07) | **$p < 0.001$** (KR > US > VN) |
| `spectral_centroid` (Hz) | 2154.5 (432) | 2276.9 (438) | 2287.1 (423) | 2242.7 (435) | **$p < 0.001$** (US, KR sáng hơn VN) |
| `chord_change_rate` | 42.1 (12.4) | 43.8 (12.2) | 46.5 (11.9) | 44.2 (12.3) | **$p < 0.001$** (KR chuyển âm nhanh nhất)|
| `mfcc_repetition_determinism`| 0.211 (0.11)| 0.239 (0.11) | 0.229 (0.11) | 0.227 (0.11) | **$p < 0.001$** (US lặp lại âm sắc cao nhất)|

---

## 5. Hướng dẫn & Lưu ý Bàn giao cho Member 4
1. **Kiểm tra Đa cộng tuyến (Multicollinearity):**
   - Cặp `rms_energy` và `loudness_proxy` có độ tương quan mạnh; khuyến nghị chọn `rms_energy` làm thước đo vật lý chính.
   - Cặp `beat_interval_variability` và `tempo_stability` cùng phản ánh tính ổn định nhịp điệu.
2. **Xử lý Biến Phân loại (Categorical Encoding):**
   - `audio_mode`: Khuyến nghị mã hóa nhị phân (`major` = 1, `minor` = 0).
   - `audio_key`: Nên dùng One-Hot Encoding hoặc biểu diễn trên Vòng cung bậc 5 (Circle of Fifths coordinates) để bảo toàn khoảng cách hòa âm.
3. **Phân phối Chuẩn & Biến đổi Dữ liệu:**
   - Hầu hết các đặc trưng RQA (`chroma_longest_pattern_duration`, `mfcc_longest_pattern_duration`) có độ lệch phải (Positive Skewness); nên áp dụng Log Transformation (`log1p`) trước khi đưa vào các mô hình tuyến tính (Linear Regression, Ridge/Lasso).
