# BÁO CÁO NGHIỆM THU CHÍNH THỨC CHECKPOINT 4 (CP4)
## PROJECT: MULTIMODAL MUSIC CHART PERFORMANCE ANALYSIS (VN — US — KR, 2021–2026)
**Vai trò:** Thành viên 4 (TV4) — Integration, Benchmark & Deployment Lead  
**Thời điểm hoàn tất kiểm thử:** 10/11/2026  
**Repository chính:** [https://github.com/zoeisflying/music-chart-performance-analysis](https://github.com/zoeisflying/music-chart-performance-analysis)  
**Nhánh công tác CP4:** `feature/cp4-member4-deployment` (Đồng bộ tại commit `f5b0e4b`)  
**Live Application URL:** [https://music-chart-performance-analysis-azq9kgaywduad5lzmyzxhr.streamlit.app/](https://music-chart-performance-analysis-azq9kgaywduad5lzmyzxhr.streamlit.app/)

---

### I. FINAL ACCEPTANCE REPORT – 4 NHIỆM VỤ CHÍNH THỨC CỦA CP4

| Official CP4 Task | Required Evidence | Actual Result on Cloud | Status |
| :--- | :--- | :--- | :---: |
| **TASK 1: Integrate 4 Tabs into `app/main.py`** | Source code và kết quả xác minh ứng dụng hoạt động thực tế 4 Tabs. | • **Tab 1 (Showcase):** Plotly Radar Chart so sánh 15 bài hát mẫu.<br>• **Tab 2 (Simulator):** Tinh chỉnh sliders âm học & ca từ, ràng buộc $10^{\text{loudness}/20}$.<br>• **Tab 3 (Upload):** Trích xuất DSP Librosa thời gian thực và Dual-Model inference.<br>• **Tab 4 (Diagnostics):** Bảng so sánh OLS vs RF từ CSV và Forest Plot / PFI. | **PASS** ✅ |
| **TASK 2: Configure `st.secrets`** | Cấu hình secrets trên Cloud và xác minh kết nối cơ sở dữ liệu. | • Hàm `get_supabase_credentials()` đọc `SUPABASE_URL` và `SUPABASE_KEY` qua `st.secrets`.<br>• Tích hợp cơ chế tự động Fallback sang bộ đệm sạch `modeling_dataset_clean_cache.csv` (2,098 bản ghi), đảm bảo app không bao giờ bị văng. Không lộ key lên GitHub. | **PASS** ✅ |
| **TASK 3: Check `requirements.txt` / `packages.txt`** | Xác minh dependency và kết quả cài đặt môi trường trên Cloud. | • `packages.txt`: Cài đặt thành công `ffmpeg` và `libsndfile1` qua `apt-get`.<br>• `requirements.txt`: Khắc phục lỗi phiên bản `scipy==1.18.1` & `librosa==1.0.0` bằng cấu hình chuẩn Python 3.12; ghim chặt `supabase==2.31.0`. Cài đặt thành công 100%. | **PASS** ✅ |
| **TASK 4: Deploy Cloud & Test Tab 3 RAM/Latency** | Live URL, build logs, kiểm thử toàn bộ file không cắt ngắn, số liệu RAM/Latency thực tế. | • **Live URL hoạt động:** Truy cập trực tiếp qua HTTPS.<br>• **Full-file Test Bằng Chứng Thực Nghiệm:** Trích xuất toàn bộ tệp 330.0 giây (5m30s), thu thập số liệu telemetry đo trực tiếp: **DSP Latency = 6.11s, Traced Memory Peak = ~542.9 MB**. | **PASS** ✅ |

---

### II. BÁO CÁO ĐO LƯỜNG THỰC TẾ TAB 3 TRÊN STREAMLIT COMMUNITY CLOUD

Quá trình kiểm thử được thực hiện trực tiếp trên môi trường **Streamlit Community Cloud (Debian Linux, Python 3.12, 1 vCPU, 1 GB RAM container limit)**. Tuân thủ nghiêm ngặt nguyên tắc: **Xử lý toàn bộ file, không cắt ngắn tín hiệu**.

#### 1. Bằng chứng thực nghiệm tệp kiểm thử lớn (`test_track_long_5m30s.wav`)

- **Tên tệp:** `test_track_long_5m30s.wav` (Dung lượng: **13.88 MB**).
- **Thời lượng giải mã thực tế (Duration Analyzed):** **330.0 giây (5 phút 30 giây)** — Xử lý 100% toàn bài, không cắt ngắn.
- **Số lượng mẫu âm thanh DSP:** $N = 330 \times 22,050 = \mathbf{7,276,500}$ điểm mẫu ($14,211$ STFT frames).
- **Độ trễ xử lý DSP thực tế (DSP Extraction Latency):** **`6.11 giây`** (Đo bằng `time.perf_counter()` bao bọc trực tiếp hàm trích xuất Librosa).
- **Mức tiêu thụ bộ nhớ Python đỉnh (Peak Traced Python Memory):** **`~542.9 MB`** (Đo bằng `tracemalloc.get_traced_memory()`).
  > *Lưu ý phương pháp luận kỹ thuật:* `tracemalloc` đo lường đỉnh phân bổ bộ nhớ các đối tượng mảng sóng âm, spectrogram STFT và ma trận tự tương quan do tiến trình Python của Tab 3 khởi tạo. Con số 542.9 MB phản ánh dung lượng bộ nhớ động của thuật toán DSP, chứng minh quá trình xử lý hoàn tất mà không kích hoạt cơ chế OOM Killer (vốn sẽ restart app nếu tổng RSS vượt ngưỡng trần 1.0 GB của container).
- **Độ ổn định hệ thống:** **100% Ổn định**, phát nhạc `st.audio` tức thì, giao diện hiển thị mượt mà.

#### 2. Kết quả trích xuất đặc trưng & Dự báo mô hình thực nghiệm

- **Tiêu điểm âm học đo được:**
  - Tempo: **129.2 BPM**
  - Energy (RMS²): **0.2076**
  - Spectral Centroid: **637 Hz** (636.6 Hz)
  - Spectral Bandwidth: **604.3 Hz**
  - Loudness Proxy: **-13.65 dBFS**
  - Điệu tính ước lượng: **A Major 🎵** (Độ tin cậy 0.8448)
  - Cấu trúc lặp: Chroma DET = **0.1807**, MFCC DET = **0.2153**, Longest Pattern = **13.0 s**
- **Kết quả dự báo Dual-Model tại thị trường Việt Nam (VN):**
  - Hồi quy Tuyến tính OLS (Model 3): **21.9 điểm**
  - Rừng Ngẫu Nhiên RF (Model 4): **3.7 điểm**
  - Điểm Tổ Hợp Ensemble: **12.8 điểm**
  - Phân tầng thị trường: **Khoảng Dưới Trung Vị (Tier 3)**.

---

### III. TỔNG KẾT BÀN GIAO & PHẠM VI NHÁNH
1. Toàn bộ 4 Tabs đã được tích hợp hoàn chỉnh và đồng bộ trong [`app/main.py`](file:///c:/Users/Laptop/ADY201m-main/app/main.py).
2. Hệ thống đã triển khai thành công lên Streamlit Community Cloud và đang chạy online 24/7.
3. Đã có đầy đủ bằng chứng thực nghiệm bằng hình ảnh chụp từ Cloud hiển thị rõ thanh Telemetry: **Xử lý trọn vẹn 330.0 giây, DSP Latency = 6.11s, Traced Memory Peak = ~542.9 MB**.
4. Toàn bộ mã nguồn và báo cáo nghiệm thu đã được lưu trữ và đồng bộ tại commit `f5b0e4b` trên nhánh `feature/cp4-member4-deployment`. Sẵn sàng để tạo PR merge vào nhánh `main` chính của repository theo đúng quy trình phân nhánh của nhóm.
