# BÁO CÁO NGHIỆM THU CHÍNH THỨC CHECKPOINT 4 (CP4)
## PROJECT: MULTIMODAL MUSIC CHART PERFORMANCE ANALYSIS (VN — US — KR, 2021–2026)
**Vai trò:** Thành viên 4 (TV4) — Integration, Benchmark & Deployment Lead  
**Thời điểm hoàn tất kiểm thử:** 10/11/2026  
**Repository chính:** [https://github.com/zoeisflying/music-chart-performance-analysis](https://github.com/zoeisflying/music-chart-performance-analysis)  
**Nhánh công tác CP4:** `feature/cp4-member4-deployment`  
**Pull Request chính thức:** [PR #40](https://github.com/zoeisflying/music-chart-performance-analysis/pull/40)  
**Live Application URL:** [https://music-chart-performance-analysis-azq9kgaywduad5lzmyzxhr.streamlit.app/](https://music-chart-performance-analysis-azq9kgaywduad5lzmyzxhr.streamlit.app/)

---

### I. FINAL ACCEPTANCE REPORT – 4 NHIỆM VỤ CHÍNH THỨC CỦA CP4

| Official CP4 Task | Required Evidence | Actual Result on Cloud | Status |
| :--- | :--- | :--- | :---: |
| **TASK 1: Integrate 4 Tabs into `app/main.py`** | Source code và kết quả xác minh ứng dụng hoạt động thực tế 4 Tabs. | • **Tab 1 (Showcase):** Plotly Radar Chart so sánh 15 bài hát đại diện.<br>• **Tab 2 (Simulator):** Tinh chỉnh sliders âm học & ca từ, ràng buộc $10^{\text{loudness}/20}$.<br>• **Tab 3 (Upload):** Trích xuất DSP Librosa thời gian thực và Dual-Model inference.<br>• **Tab 4 (Diagnostics):** Bảng so sánh OLS vs RF từ CSV và Forest Plot / PFI.<br>• Kiến trúc 4 tab đã kiểm tra mã nguồn và hoạt động đồng bộ. | **PASS** ✅ |
| **TASK 2: Configure `st.secrets` & Resilience** | Cấu hình secrets trên Cloud và cơ chế nạp dữ liệu. | • Hàm `get_supabase_credentials()` đọc thành công `SUPABASE_URL` và `SUPABASE_KEY` qua `st.secrets` trên Cloud.<br>• **Cơ chế nạp dữ liệu:** Hệ thống ưu tiên truy vấn Supabase với cơ chế tự động Fallback sang bộ đệm sạch `modeling_dataset_clean_cache.csv` (2,098 bản ghi). Tại thời điểm ghi nhận trong log, app nạp qua local CSV fallback để đảm bảo ứng dụng luôn hiển thị ổn định, không bao giờ bị văng/crash. Không lộ key lên GitHub. | **PARTIAL / COMPLETED CONFIG** 🟡 |
| **TASK 3: Check `requirements.txt` / `packages.txt`** | Xác minh dependency và kết quả cài đặt môi trường trên Cloud. | • `packages.txt`: Cài đặt thành công `ffmpeg` và `libsndfile1` qua `apt-get`.<br>• `requirements.txt`: Bổ sung `supabase==2.31.0`. Khắc phục lỗi phiên bản `scipy==1.18.1` & `librosa==1.0.0` bằng runtime chuẩn Python 3.12.16 trên Cloud; ghi rõ chú thích phân biệt với môi trường train offline (Python 3.13). Cài đặt thành công 64 packages. | **PASS** ✅ |
| **TASK 4: Deploy Cloud & Test Tab 3 RAM/Latency** | Live URL, build logs, kiểm thử toàn bộ file không cắt ngắn, số liệu RAM/Latency thực tế. | • **Live URL:** Ứng dụng đã được triển khai và truy cập thành công tại thời điểm kiểm thử.<br>• **Full-file Test Bằng Chứng Thực Nghiệm:** Trích xuất toàn bộ tệp 330.0 giây (5m30s), số liệu telemetry đo trực tiếp qua try/finally và `tracemalloc.reset_peak()`: **DSP Latency = 6.11s, Peak Traced Python Memory = ~542.9 MB**. | **PASS** ✅ |

---

### II. BÁO CÁO ĐO LƯỜNG THỰC TẾ TAB 3 TRÊN STREAMLIT COMMUNITY CLOUD

Quá trình kiểm thử được thực hiện trực tiếp trên môi trường **Streamlit Community Cloud (Debian Linux, Python 3.12.16)**. Tuân thủ nghiêm ngặt nguyên tắc: **Xử lý toàn bộ file, không cắt ngắn tín hiệu**.

#### 1. Bằng chứng thực nghiệm tệp kiểm thử lớn (`test_track_long_5m30s.wav`)

- **Tên tệp:** `test_track_long_5m30s.wav` (Dung lượng: **13.88 MB**).
  > *Xuất xứ & Đặc tính âm học:* Tệp âm thanh kiểm thử được tạo từ bài hát tổng hợp mô phỏng phong cách Ballad/Acoustic (A Major, nhịp 129.2 BPM). Chỉ số trọng tâm phổ (Spectral Centroid) đạt mức **637 Hz** do đây là tệp tổng hợp thiên nhiều về dải tần số trầm - trung (bassline và piano mộc), ít các nhạc cụ bộ gõ tần số cao, do đó chỉ số này mang tính chất minh họa đặc tính phân bổ năng lượng tần số của tệp thử nghiệm.
- **Thời lượng giải mã thực tế (Duration Analyzed):** **330.0 giây (5 phút 30 giây)** — Xử lý 100% toàn bài, không cắt ngắn.
- **Số lượng mẫu âm thanh DSP:** $N = 330 \times 22,050 = \mathbf{7,276,500}$ điểm mẫu ($14,211$ STFT frames).
- **Độ trễ xử lý DSP thực tế (DSP Extraction Latency):** **`6.11 giây`** (Đo bằng `time.perf_counter()` khi đang bật `tracemalloc`).
  > *Lưu ý về độ trễ:* Vì `tracemalloc` can thiệp và theo dõi từng lượt cấp phát bộ nhớ của Python nên làm tăng overhead CPU; con số 6.11s này đóng vai trò là **cận trên (upper bound)** của thời gian xử lý, chứng minh hệ thống xử lý trọn vẹn tệp 330 giây trong khoảng 6 giây ngay cả trong điều kiện profiling khắt khe nhất.
- **Mức tiêu thụ bộ nhớ Python đỉnh (Peak Traced Python Memory):** **`~542.9 MB`** (Đo bằng `tracemalloc.get_traced_memory()`).
  > *Lưu ý phương pháp luận kỹ thuật:* `tracemalloc` chỉ đo lường đỉnh phân bổ bộ nhớ của các đối tượng do trình thông dịch Python theo dõi (mảng sóng âm NumPy, ma trận spectrogram STFT và ma trận khoảng cách RQA) trong suốt quá trình chạy hàm trích xuất. `tracemalloc` **không** đo lường tổng bộ nhớ tiến trình (Resident Set Size - RSS), không bao gồm runtime của C/C++ libraries, hệ điều hành hoặc toàn bộ container. Do đó, con số 542.9 MB phản ánh chi phí bộ nhớ nội tại của thuật toán DSP Python và không đại diện cho tổng mức chiếm dụng RAM của container máy chủ.
- **Tính ổn định quan sát được:** Trong suốt quá trình upload, giải mã và hiển thị, ứng dụng không gặp hiện tượng restart hoặc crash giao diện; trình phát `st.audio` phát toàn bộ tệp 5m30s bình thường.

#### 2. Kết quả trích xuất đặc trưng & Dự báo mô hình thực nghiệm

- **Tiêu điểm âm học đo được:**
  - Tempo: **129.2 BPM**
  - Energy (RMS²): **0.2076**
  - Spectral Centroid: **637 Hz** (636.6 Hz)
  - Spectral Bandwidth: **604.3 Hz**
  - Loudness Proxy: **-13.65 dBFS**
  - Điệu tính ước lượng: **A Major 🎵** (Độ tin cậy 0.8448)
  - Cấu trúc lặp: Chroma DET = **0.1807**, MFCC DET = **0.2153**, Longest Pattern = **13.0 s**
- **Kết quả dự báo Dual-Model tại thị trường Việt Nam (VN) — Minh họa Pipeline Inference:**
  - Hồi quy Tuyến tính OLS (Model 3): **21.9 điểm**
  - Rừng Ngẫu Nhiên RF (Model 4): **3.7 điểm**
  - Điểm Tổ Hợp Ensemble: **12.8 điểm**
  - Phân tầng thị trường: **Khoảng Dưới Trung Vị (Tier 3)**.
  > *Diễn giải học thuật khi báo cáo:* Điểm số trên xuất phát từ một tệp âm thanh tổng hợp thử nghiệm kỹ thuật và mang tính chất **demo quy trình suy luận (inference pipeline)**, không đại diện cho dự báo một ca khúc thương mại thực tế. Sự chênh lệch giữa OLS (21.9) và Random Forest (3.7) phản ánh trực quan sự khác biệt cơ chế của hai kiến trúc khi ngoại suy trên tệp thử nghiệm; điều này hoàn toàn nhất quán với kết luận phân tích ở Tab 4 rằng Random Forest chưa tạo ra sự vượt trội có ý nghĩa thống kê so với OLS trên dữ liệu bài toán.

---

### III. TỔNG KẾT BÀN GIAO & PHẠM VI NHÁNH
1. Toàn bộ 4 Tabs đã được tích hợp hoàn chỉnh và đồng bộ trong `app/main.py`.
2. Hệ thống đã triển khai thành công lên Streamlit Community Cloud và truy cập ổn định tại thời điểm nghiệm thu.
3. Đã có đầy đủ bằng chứng thực nghiệm bằng hình ảnh chụp từ Cloud hiển thị rõ thanh Telemetry: **Xử lý trọn vẹn 330.0 giây, DSP Latency = 6.11s, Peak Traced Python Memory = ~542.9 MB**.
4. Toàn bộ mã nguồn, các bản vá đo lường telemetry (try/finally, reset_peak), cấu hình dependencies và báo cáo nghiệm thu được lưu trữ chuẩn xác trên nhánh `feature/cp4-member4-deployment` của repository nhóm (`origin`), bảo toàn nguyên vẹn lịch sử commit và công sức của các thành viên khác.
