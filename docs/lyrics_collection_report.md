# Báo Cáo Thu Thập và Kiểm Kê Lyrics (Member 3)

## 1. Tổng Quan Nguồn Dữ Liệu

- **Bảng lấy mẫu gốc trên Supabase:** `public.song_quarter_sampling` (`sampled = true`: **2,760** dòng quý-thị trường, gồm 920 dòng cho mỗi thị trường `VN`, `US`, `KR`) kết hợp với `public.songs`.
- **Tổng số bài hát duy nhất (`spotify_track_id`):** **1,912** bài hát.
- **Số cặp `(spotify_track_id, market)` duy nhất:** **2,098** quan sát (`US`: 797, `KR`: 693, `VN`: 608; trong đó có 169 bài xuất hiện ở từ 2 thị trường trở lên).

## 2. Kết Quả Thu Thập Lyrics (`data/raw/lyrics_collection_results.json`)

| Trạng thái (`lyrics_status`) | Số lượng bài hát | Tỷ lệ (%) | Ghi chú |
| :--- | ---: | ---: | :--- |
| `success` | 1,909 | 99.84% | Có file `.txt` hợp lệ, đọc được (UTF-8), không rỗng tại `data/raw/lyrics/<spotify_track_id>.txt` |
| `not_found` | 3 | 0.16% | Cả 3 bài đều là nhạc không lời (Instrumental): `6WY4wvlmgccWapnIg14Vy0` (*Merry Christmas Mr. Lawrence* - Ryuichi Sakamoto), `1Px44tw80Q4rcCl8FxDGOi` (*Who (Instrumental)* - Jimin), `7M30L99pePlQ3wh8IxVUsD` (*With You - Instrumental* - Jimin) |
| `failed` | 0 | 0.00% | Đã khôi phục hoàn tất qua bước retry (`lyrics_retry.py`) và bổ sung thủ công (`update_manual_lyrics.py`) |
| **Tổng cộng** | **1,912** | **100.00%** | Không có file mồ côi (orphan) hoặc trùng lặp Track ID |

### Phân Bố Nguồn Thu Thập (`source`)

- **LRCLIB:** 1,879 bài (98.27% tổng số mẫu; 98.43% số bài có lời)
- **Genius:** 29 bài (1.52%)
- **Manual (bổ sung và kiểm chứng thủ công):** 1 bài (0.05%)
- **Không có lời (`not_found` - Instrumental):** 3 bài (0.16%)

## 3. Phân Bố Ngôn Ngữ (`language_detection.py` v2.1.0)

| Nhãn ngôn ngữ (`language`) | Số bài hát | Tỷ lệ (%) | Trạng thái trích xuất đặc trưng (`extraction_status`) |
| :--- | ---: | ---: | :--- |
| `en` (Tiếng Anh) | 1,095 | 57.27% | `success` (đầy đủ bộ feature ngôn ngữ) |
| `vi` (Tiếng Việt) | 367 | 19.19% | `success` (đầy đủ bộ feature ngôn ngữ) |
| `ko` (Tiếng Hàn - Hangul) | 211 | 11.04% | `success` (đầy đủ bộ feature ngôn ngữ) |
| `mixed` (Đa ngôn ngữ $\ge 2$ ngôn ngữ $\ge 30\%$) | 154 | 8.05% | `success` (`pronoun_ratio = NULL` theo quy tắc) |
| `es` (Tiếng Tây Ban Nha) | 43 | 2.25% | `success` (đầy đủ bộ feature ngôn ngữ) |
| Ngôn ngữ khác (`ja`: 12, `sw`: 6, `cy`: 4, `id`: 3, `tl`: 2, `fr`: 2, `so`: 2, `zh`: 2, `af`: 1, `no`: 1, `de`: 1, `fi`: 1, `pt`: 1, `ro`: 1) | 39 | 2.04% | `language_limited` (chỉ trích xuất các feature cơ bản, `title_word_in_lyrics` và `pronoun_ratio` để `NULL`) |
| `unknown` (3 bài Instrumental `not_found`) | 3 | 0.16% | `not_found` (toàn bộ feature để `NULL`) |

Chi tiết kỹ thuật đầy đủ về kiểm kê, quy tắc ngôn ngữ, trích xuất đặc trưng và kiểm định chất lượng được trình bày trong `reports/report3/member3.md` và `reports/report4/member3.md`.
