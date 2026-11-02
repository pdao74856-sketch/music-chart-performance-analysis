# Báo Cáo Kỹ Thuật Tích Hợp Dữ Liệu & Kiểm Soát Chất Lượng — Member 4 (Report 3)

* **Học phần:** ADY201m — Advanced Data Analytics / Applied Data Science Project  
* **Thành viên phụ trách:** Member 4 (Data Integration, Quality Control & Report Coordination)  
* **Ngày cập nhật:** 2026-10-26  
* **Nguồn dữ liệu:** Kho dữ liệu tích hợp từ Member 1 (Chart), Member 2 (Audio), Member 3 (Lyrics), và Lấy mẫu phân tầng (Sampling Provenance).

---

## 1. Mục Tiêu & Trách Nhiệm Cốt Lõi Của Member 4

Member 4 giữ vai trò là xương sống tích hợp (Integration Backbone) và kiểm toán viên chất lượng dữ liệu (Quality Auditor) của toàn bộ dự án, chịu trách nhiệm:
1. **Kiểm tra & Chuẩn hóa Quan hệ Cơ sở Dữ liệu 3NF:** Xác thực tính toàn vẹn tham chiếu giữa `songs`, `chart_weekly`, `chart_performance`, `youtube_mapping`, `audio_features`, `lyrics_features`, và `song_quarter_sampling`.
2. **Thiết kế Lược đồ & VIEW Tích hợp:** Xây dựng `sql/04_modeling_dataset.sql` định nghĩa VIEW chuẩn `public.modeling_dataset` tại grain cấp bài hát $\times$ thị trường `(spotify_track_id, market)`.
3. **Phát triển Pipeline Tích hợp Tự động hóa:** Xây dựng `src/integration/build_analysis_dataset.py` và bộ kiểm thử tự động `src/integration/validate_analysis_dataset.py`.
4. **Kiểm soát Khuyết thiếu (Missingness) & Loại trừ (Exclusion):** Phân loại rành mạch các nguyên nhân thiếu dữ liệu, đảm bảo nguyên tắc bảo toàn `NULL` (không gán 0 tùy tiện).
5. **Xây dựng Từ điển Dữ liệu Tổng thể:** Đóng gói toàn bộ định nghĩa, công thức, đơn vị và lưu ý của 4 thành viên vào `reports/report3/data_dictionary.xlsx`.
6. **Tổng hợp Báo cáo Nghiên cứu:** Biên soạn báo cáo tổng thể Report 3 (`reports/report3/report3.md`).

---

## 2. Đối Chiếu & Giải Trình Cỡ Mẫu (Sample Alignment Audit)

| Thực thể dữ liệu | Cỡ mẫu ghi nhận | Khóa chính (Primary Key) | Giải trình cấu trúc dữ liệu |
| :--- | :---: | :--- | :--- |
| **`songs`** | 15.037 | `spotify_track_id` | Toàn bộ các bài hát độc bản được crawl từ Kworb seed pages. |
| **`chart_weekly`** | 161.734 | `(spotify_track_id, market, chart_date)` | Toàn bộ lịch sử thứ hạng tuần Spotify Top 200 (2021–2026). |
| **`chart_performance`** | 10.887 | `(spotify_track_id, market)` | Tổng hợp hiệu suất lifetime của toàn bộ bài hát từng lên bảng xếp hạng. |
| **`song_quarter_sampling`** | 2.760 | `(spotify_track_id, market, year, quarter)` | Phân bổ ngẫu nhiên phân tầng: 40 bài/quý $\times$ 23 quý $\times$ 3 thị trường. |
| **Số bài hát độc bản trong mẫu** | **1.912** | `spotify_track_id` | 1.912 bài hát độc bản được chọn làm đại diện cho 23 quý nghiên cứu. |
| **Tập mẫu mô hình hóa (`modeling_dataset`)** | **2.860** | `(spotify_track_id, market)` | **Đúng 2.860 bản ghi xếp hạng** khi join 1.912 bài mẫu với `chart_performance`. |

### Giải trình toán học về con số 2.860:
Một bài hát trong 1.912 bài được lấy mẫu ở một quý tại một quốc gia (ví dụ: ca khúc hit toàn cầu của BTS hoặc Taylor Swift) có thể đồng thời lọt vào bảng xếp hạng của cả 3 quốc gia trong giai đoạn 2021–2026:
- **Hàn Quốc (KR):** 1.020 bản ghi
- **Hoa Kỳ (US):** 955 bản ghi
- **Việt Nam (VN):** 885 bản ghi
- **Tổng cộng:** $1.020 + 955 + 885 = \mathbf{2.860 \text{ bản ghi}}$.
- Không có bất kỳ hiện tượng nhân dòng (fan-out) nào do YouTube mapping vì câu lệnh join ép điều kiện `ym.is_selected = TRUE`.

---

## 3. Kết Quả Kiểm Thử Chất Lượng Tích Hợp (11/11 Test Suites Passed)

Script kiểm định tự động `src/integration/validate_analysis_dataset.py` đã thực thi trên tập dữ liệu `modeling_dataset.csv` và xuất nhật ký tại `logs/integration_validation_log.csv`:
* **Khóa chính duy nhất:** 0 bản ghi trùng lặp trên `(spotify_track_id, market)`.
* **Khớp nối công thức mục tiêu:** $|\text{ChartScore} - (101 - \text{lifetime\_average\_rank})| = 0.000000$ trên 100% bản ghi.
* **Độ phủ dữ liệu âm học (Audio Completeness):** Đạt **100.0%** (0 NULL trên toàn bộ 20 đặc trưng).
* **Bảo toàn nguyên tắc NULL trong NLP:**
  - 4 bản ghi thiếu lời do là nhạc không lời (`unavailable_instrumental`, 3 track độc bản của Ryuichi Sakamoto và Jimin).
  - 217 bản ghi có `chorus_repetition_count = NULL` do không nhận diện được đoạn điệp khúc tin cậy (không tự gán 0).
  - 291 bản ghi có `pronoun_ratio = NULL` do bài hát song ngữ `mixed` hoặc ngôn ngữ ngoài từ điển đơn ngữ.
* **Độc lập thời gian:** Biến `release_year` được tính từ `songs.release_date`, phát hiện đúng 339 bản ghi nhạc cũ (catalog tracks phát hành trước 2020) hoàn toàn độc lập với năm lấy mẫu `year` (2021–2026).

---

## 4. Danh Mục Bàn Giao Của Member 4

1. **SQL Schema Migration:** [`sql/04_modeling_dataset.sql`](file:///c:/Users/Laptop/ADY201m-main/sql/04_modeling_dataset.sql)
2. **Script Tích Hợp Dữ Liệu:** [`src/integration/build_analysis_dataset.py`](file:///c:/Users/Laptop/ADY201m-main/src/integration/build_analysis_dataset.py)
3. **Script Kiểm Định Chất Lượng:** [`src/integration/validate_analysis_dataset.py`](file:///c:/Users/Laptop/ADY201m-main/src/integration/validate_analysis_dataset.py)
4. **Nhật Ký Kiểm Định Tích Hợp:** [`logs/integration_validation_log.csv`](file:///c:/Users/Laptop/ADY201m-main/logs/integration_validation_log.csv)
5. **Tập Dữ Liệu Mô Hình Hóa Master:** [`data/processed/modeling_dataset.csv`](file:///c:/Users/Laptop/ADY201m-main/data/processed/modeling_dataset.csv) ($N = 2.860$)
6. **Tập Dữ Liệu Complete-Case:** [`data/processed/modeling_dataset_complete_case.csv`](file:///c:/Users/Laptop/ADY201m-main/data/processed/modeling_dataset_complete_case.csv) ($N = 2.332$)
7. **Tập Dữ Liệu Chuỗi Thời Gian Theo Quý:** [`data/processed/quarterly_feature_trends.csv`](file:///c:/Users/Laptop/ADY201m-main/data/processed/quarterly_feature_trends.csv) ($N = 2.760$)
8. **Báo Cáo Khuyết Thiếu Theo Thị Trường:** [`reports/report3/missingness_by_market.csv`](file:///c:/Users/Laptop/ADY201m-main/reports/report3/missingness_by_market.csv)
9. **Kiểm Kê Bản Ghi Bị Loại:** [`reports/report3/excluded_records.csv`](file:///c:/Users/Laptop/ADY201m-main/reports/report3/excluded_records.csv)
10. **Bảng Thống Kê Mô Tả Tổng Hợp:** [`reports/report3/descriptive_statistics.csv`](file:///c:/Users/Laptop/ADY201m-main/reports/report3/descriptive_statistics.csv)
11. **Từ Điển Dữ Liệu Toàn Diện (Excel):** [`reports/report3/data_dictionary.xlsx`](file:///c:/Users/Laptop/ADY201m-main/reports/report3/data_dictionary.xlsx)
12. **Báo Cáo Nhóm Tổng Hợp (Report 3):** [`reports/report3/report3.md`](file:///c:/Users/Laptop/ADY201m-main/reports/report3/report3.md)

---

## 5. Hướng Dẫn Tái Lập Pipeline

```bash
# 1. Chạy tích hợp dataset và xuất báo cáo kiểm toán
python src/integration/build_analysis_dataset.py

# 2. Chạy bộ kiểm thử tự động 11 test suites
python src/integration/validate_analysis_dataset.py
```
