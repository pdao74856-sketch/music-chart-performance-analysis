# Báo Cáo Kỹ Thuật Toàn Diện & Từ Điển Dữ Liệu — Member 3: Lyrics Feature Extraction & Validation (Report 3)

## 1. Mục Tiêu và Phạm Vi Công Việc

Báo cáo này tài liệu hóa toàn bộ quy trình kiểm toán (audit), kiểm kê dữ liệu thô (`lyrics_inventory`), chuẩn hóa nhận diện ngôn ngữ (`language_detection` v2.1.0), trích xuất 10 đặc trưng lời bài hát (`extract_lyrics_features` v1.1.0), kiểm định chất lượng dữ liệu (`validate_lyrics_features`), đồng bộ cơ sở dữ liệu Supabase (`public.lyrics_features`), và đóng gói dữ liệu bàn giao cho **Member 4** và **Member 1** trong dự án `music-chart-performance-analysis`.

---

## 2. Bảng Kiểm Toán Toàn Bộ Repository (Stage 1 Audit Table)

| Hạng mục yêu cầu | Trạng thái ban đầu | Bằng chứng kiểm tra thực tế | Vấn đề phát hiện | Hành động xử lý đã thực hiện | Trạng thái sau hoàn thành |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw Lyrics & Collection Metadata** (`data/raw/lyrics/`, `lyrics_collection_results.json`) | Đã có dữ liệu gốc | 1,912 bài trong JSON; 1,909 file `.txt` trong `data/raw/lyrics/`; 3 bài `not_found` trong `lyrics_missing_inventory.csv`. | Cả 3 bài `not_found` đều là bài nhạc không lời (Instrumental) nhưng chưa được gắn nhãn phân loại rõ trong inventory; `docs/lyrics_collection_report.md` bị rỗng (0 byte). | Giữ nguyên tuyệt đối 1,909 file `.txt` gốc; phân loại rõ 3 bài `unavailable_instrumental`; hoàn thiện `docs/lyrics_collection_report.md`. | **DONE AND VERIFIED** |
| **2. Kiểm kê Raw Lyrics** (`reports/report3/lyrics_inventory.csv`) | Chưa đúng vị trí & thiếu cột | `build_lyrics_inventory.py` chỉ xuất ra `data/processed/lyrics_inventory.csv`, thiếu các cột `artist`, `market`, `is_readable`, `validation_status`, `error_reason`; `word_count` đếm bằng `text.split()` lệch với tokenizer của pipeline trích xuất. | Lệch `word_count` giữa inventory và feature CSV (ví dụ `00imgaPlYRrMGn9o83hfmk`: 379 vs 378); thiếu đường dẫn chuẩn `reports/report3/lyrics_inventory.csv`. | Nâng cấp `src/lyrics/build_lyrics_inventory.py` dùng chung hàm `tokenize()` từ `extract_lyrics_features.py`, bổ sung đầy đủ các cột và xuất đồng thời ra `reports/report3/lyrics_inventory.csv` và `data/processed/lyrics_inventory.csv`. | **DONE AND VERIFIED** |
| **3. Language Detection** (`src/lyrics/language_detection.py`) | Có thay đổi chưa commit gây lỗi | `HEAD` (`v2.0`) lưu 154 bài `mixed` và 26 bài `unknown` trong JSON/checkpoint. Bản working-tree chưa commit sửa `detect_language()` làm giảm `mixed` từ 154 xuống còn 12 bài. | 1) Lỗi mẫu số ở working-tree (dòng 221–226): chia `script_count / sum(script_counts_total.values())` (bỏ quên ký tự Latin ở mẫu số!), khiến 123 bài Hàn-Anh `mixed` bị ép thành `ko` và 3 bài thành `ja`.<br>2) Cộng đúp `latin_weight` làm 11 bài `mixed` (gồm nhạc Việt-Anh) bị ép thành `en`/`es`.<br>3) Ở bản `HEAD`, 25 bài thuần Hangul Hàn Quốc và 1 bài tiếng Trung bị gán `unknown` vì mọi dòng đơn lẻ đều dài $9\text{--}14 < 15$ ký tự (`MIN_TEXT_LENGTH = 15`). | Sửa lỗi trong `src/lyrics/language_detection.py` (`v2.1.0`): khôi phục chính xác 154 bài `mixed`, bổ sung fallback cấp bài hát có bằng chứng ($\ge 70\%$ ký tự script trên tổng ký tự toàn bài) khi mọi dòng đơn lẻ $< 15$ ký tự. Kết quả: 25 bài Hangul `unknown -> ko`, 1 bài Hanzi `unknown -> zh`, giữ nguyên 154 bài `mixed`. Đồng bộ toàn bộ JSON, checkpoint, `lyrics_status.csv`. | **DONE AND VERIFIED** |
| **4. Feature Extraction & YAML Rules** (`extract_lyrics_features.py`, `language_rules.yaml`) | Thiếu feature & tắt chorus | `extract_lyrics_features.py` chỉ xuất 9 feature (thiếu `repetition_ratio` dù `validate_lyrics_features.py` và SQL schema yêu cầu); `chorus_repetition_count` bị gán cứng `None`; `chorus_repetition_results.csv` có dòng rác `test_song_001` và gán `0` cho 118 bài không nhãn khi không tìm thấy đoạn lặp. | Khi chạy `validate_lyrics_features.py` ban đầu bị crash ngay (`Feature CSV is missing required columns: repetition_ratio`); gán `0` cho chorus khi không nhận diện được chorus vi phạm quy tắc `NULL`; từ điển đại từ tiếng Hàn chỉ có gốc từ đơn âm tiết khiến `pronoun_ratio` của `ko` bị thấp giả tạo (2.58%). | Nâng cấp `language_rules.yaml` và `extract_lyrics_features.py` (`v1.1.0`): bổ sung `repetition_ratio`, tích hợp thuật toán `detect_chorus_repetition()` (chỉ trả về `0` khi có nhãn `[Chorus]` tường minh mà không lặp, trả về `NULL` khi không đủ bằng chứng nhận diện chorus), loại bỏ nhãn `[Section]` khỏi đếm từ, mở rộng từ điển đại từ tiếng Hàn có trợ từ (`나는, 난, 나를...`) và tiếng Anh rút gọn (`i'm, you're...`). | **DONE AND VERIFIED** |
| **5. Validation & Unit Tests** (`validate_lyrics_features.py`, `tests/`) | Chưa hoàn thiện & chưa có test | `validate_lyrics_features.py` không trả về exit code lỗi, không đối chiếu chéo với `lyrics_inventory.csv` và Supabase DB; chưa có thư mục `tests/`. | Không phát hiện được lệch giữa inventory, log, CSV và DB; thiếu bộ unit test kiểm thử 11 kịch bản biên. | Hoàn thiện `src/lyrics/validate_lyrics_features.py` (kiểm tra đầy đủ miền giá trị, công thức, quy tắc `NULL`, đối chiếu chéo 4 nguồn `CSV - Log - Inventory - Supabase DB`, trả về exit code chuẩn CI); tạo `tests/test_lyrics_pipeline.py` gồm 11 unit tests (`11/11 PASS`). | **DONE AND VERIFIED** |
| **6. Database & Supabase** (`sql/03_lyrics_features.sql`, bảng `lyrics_features`) | Bảng rỗng (`count = 0`) | Bảng `public.lyrics_features` đã tồn tại trên Supabase với 13 cột đúng theo migration nhưng có `0` dòng dữ liệu; chưa có script upsert và chưa có `sql/03_lyrics_features.sql`. | Dữ liệu đặc trưng chưa được đưa lên Supabase cho Member 4 và Member 1 sử dụng. | Tạo `sql/03_lyrics_features.sql` tương thích tuyệt đối với schema hiện hành; tạo và chạy `src/lyrics/upsert_lyrics_features.py` thực hiện upsert 1,912 bài hát lên Supabase, chạy lần 2 kiểm chứng tính idempotent (0 duplicate, 0 mismatch). | **DONE AND VERIFIED** |
| **7. Statistics, Visualization & Reports** (`reports/report3/`, `reports/report4/`) | Chưa tồn tại | Chưa có `lyrics_statistics.csv`, chưa có thư mục `figures/member3/`, `chart_data/member3/`, `reports/report3/member3.md`, `reports/report4/member3.md` và các file handoff. | Thiếu toàn bộ đầu ra thống kê, biểu đồ và báo cáo bàn giao của Giai đoạn 7, 8, 9. | Tạo `src/lyrics/generate_statistics_and_charts.py`, sinh `lyrics_statistics.csv` (90 dòng thống kê), 9 biểu đồ PNG 300-DPI, 9 file CSV dữ liệu biểu đồ, 3 file CSV bàn giao cho Member 1 & Member 4, cùng 2 báo cáo Markdown hoàn chỉnh. | **DONE AND VERIFIED** |

---

## 3. Nguồn Dữ Liệu, Quy Trình Kiểm Kê và Ánh Xạ Track ID (Stage 2)

### 3.1. Nguồn dữ liệu và quy ước ánh xạ

- **Khóa định danh chuẩn:** `spotify_track_id` (chuỗi 22 ký tự Base62 của Spotify), liên kết khóa ngoại trực tiếp tới bảng `public.songs(spotify_track_id)` và bảng hiệu suất `public.chart_performance(spotify_track_id, market)`.
- **Quy ước lưu trữ file lyrics thô:** Mỗi bài hát có trạng thái `lyrics_status = "success"` được lưu thành đúng 1 file văn bản mã hóa UTF-8 tại đường dẫn `data/raw/lyrics/<spotify_track_id>.txt`.
- **Ánh xạ thị trường (`market`):** Được trích xuất từ trường `sampling_provenance` trong `data/raw/lyrics_collection_results.json` (đối chiếu khớp 100% với 2,760 dòng có `sampled = true` trên bảng `public.song_quarter_sampling` của Supabase). Các bài hát xuất hiện ở nhiều thị trường được lưu chuỗi thị trường phân tách bằng dấu chấm phẩy (ví dụ: `KR;VN`, `KR;US`, `KR;US;VN`) ở bảng cấp bài hát, và được bung thành từng cặp `(spotify_track_id, market)` (tổng cộng **2,098 cặp**) khi phân tích theo thị trường hoặc join với `chart_performance`.

### 3.2. Kết quả kiểm kê thực tế (`reports/report3/lyrics_inventory.csv`)

| Phân loại kiểm kê | Số lượng bài hát | Tỷ lệ (%) | Chi tiết xác minh |
| :--- | ---: | ---: | :--- |
| **Tổng số bài hát trong tập mẫu (`spotify_track_id`)** | **1,912** | **100.00%** | Khớp 100% với `song_quarter_sampling` (`sampled = true`) trên Supabase |
| **Lyrics hợp lệ, đọc được & đầy đủ đặc trưng ngôn ngữ (`valid`)** | **1,870** | **97.80%** | Thuộc 5 nhãn ngôn ngữ hỗ trợ đầy đủ (`en`: 1,095, `vi`: 367, `ko`: 211, `mixed`: 154, `es`: 43) |
| **Lyrics hợp lệ, đọc được nhưng giới hạn ngôn ngữ (`valid_language_limited`)** | **39** | **2.04%** | Có file `.txt` hợp lệ, tính được 8 đặc trưng cơ bản nhưng ngôn ngữ nằm ngoài bộ từ điển stopword/đại từ (`ja`: 12, `sw`: 6, `cy`: 4, `id`: 3, `tl`: 2, `fr`: 2, `so`: 2, `zh`: 2, `af`: 1, `no`: 1, `de`: 1, `fi`: 1, `pt`: 1, `ro`: 1) |
| **Không có lời do là nhạc không lời (`unavailable_instrumental`)** | **3** | **0.16%** | `6WY4wvlmgccWapnIg14Vy0` (*Merry Christmas Mr. Lawrence* - Ryuichi Sakamoto, KR), `1Px44tw80Q4rcCl8FxDGOi` (*Who (Instrumental)* - Jimin, KR), `7M30L99pePlQ3wh8IxVUsD` (*With You - Instrumental* - Jimin, VN) |
| **File không tồn tại khi `lyrics_status = success` (`missing_file`)** | **0** | **0.00%** | Toàn bộ 1,909 bài `success` đều có file `.txt` tồn tại trên ổ đĩa |
| **File rỗng hoặc chỉ chứa khoảng trắng (`empty_file`)** | **0** | **0.00%** | Không có file rỗng |
| **File lỗi encoding hoặc không đọc được (`read_error`)** | **0** | **0.00%** | 1,909/1,909 file đọc thành công bằng `utf-8-sig` |
| **File mồ côi trên đĩa không có trong metadata (`orphan_files`)** | **0** | **0.00%** | Đúng 1,909 file `.txt` ánh xạ 1-1 với 1,909 bài `success` |
| **Track ID trùng lặp hoặc nhiều Track ID trỏ cùng 1 file** | **0** | **0.00%** | Đảm bảo tính duy nhất của khóa chính `spotify_track_id` |

---

## 4. Phân Tích Chuyên Sâu & Xử Lý Language Detection (Stage 3)

### 4.1. Tái hiện và giải thích nguyên nhân sự cố `154 mixed` vs `12 mixed`

Khi kiểm tra lịch sử Git (`git diff HEAD -- src/lyrics/language_detection.py`) và chạy thực nghiệm đối chiếu giữa phiên bản đã commit (`HEAD`, commit `767d9a1`) và phiên bản sửa đổi chưa commit trên working tree, nguyên nhân chính xác của sự chênh lệch giữa **154 bài `mixed`** (trong JSON/checkpoint) và **12 bài `mixed`** (khi gọi hàm `detect_language()` ở working tree trước khi sửa) đã được xác định hoàn toàn:

1. **Nguyên nhân 1 — Lỗi mẫu số trong kiểm tra `script_counts_total` (Working-Tree dòng 221–226):**
   - Trong bản working-tree chưa commit, đoạn mã sau đã được thêm vào trước bước kiểm tra `mixed`:
     ```python
     if script_counts_total:
         script_lang, script_count = script_counts_total.most_common(1)[0]
         script_total = sum(script_counts_total.values())
         if script_count / script_total >= MAIN_LANGUAGE_THRESHOLD:
             return script_lang, distribution
     ```
   - Hàm `count_script_chars()` chỉ đếm các ký tự thuộc 3 hệ chữ tượng hình/âm tiết Đông Á (`ko` — Hangul, `ja` — Kana, `zh` — Hanzi) và **hoàn toàn không đếm ký tự chữ cái Latin**.
   - Do đó, đối với bất kỳ bài hát K-Pop song ngữ Hàn–Anh nào (ví dụ: *Make A Wish* - NCT U có 52% tiếng Hàn và 42% tiếng Anh, hoặc *Next Level* - aespa có 59% tiếng Anh và 35% tiếng Hàn), biến `script_counts_total` chỉ chứa duy nhất khóa `{'ko': N}` (vì không có chữ Nhật hay chữ Hán). Khi đó tỷ số `script_count / script_total` luôn bằng $N / N = 1.0 \ge 0.70$, khiến **123 bài hát song ngữ Hàn–Anh (`mixed`) bị ép sai thành `'ko'`** (và **3 bài** có ký tự tiếng Nhật bị ép thành `'ja'`) trước khi kịp đi tới nhánh kiểm tra `mixed`!
2. **Nguyên nhân 2 — Lỗi cộng đúp trọng số `latin_weight` (Working-Tree dòng 230–249):**
   - Đối với các bài hát dùng chữ Latin có tỷ lệ ngôn ngữ đứng đầu $< 70\%$, bản working-tree chạy lại `detect(latin_text)` trên toàn bộ khối dòng Latin rồi **cộng bồi thêm một lần nữa** tổng số ký tự Latin của cả bài vào `weighted_counts[latin_language]` (đồng thời danh sách `("en", "fr", "es", "de", "it", "pt")` bỏ sót `"vi"`). Việc cộng đúp này đã đẩy tỷ lệ tiếng Anh vượt ngưỡng 70% một cách nhân tạo, biến **11 bài `mixed` (bao gồm các bài song ngữ Việt–Anh như *Don't Break My Heart* - Binz, *1 Cọng Tóc Mai* - Tóc Tiên, *Gái Độc Thân* - tlinh)** thành `'en'` hoặc `'es'`.
3. **Nguyên nhân 3 — Thay đổi nhánh fallback cuối hàm và `detect_romanized_hint`:**
   - Làm thay đổi thêm **5 bài** còn lại (`3` bài sang `'unknown'`, `2` bài sang ngôn ngữ khác), tổng cộng làm giảm số bài `mixed` từ $154$ xuống đúng $154 - 123 - 3 - 11 - 3 - 2 = 12$ bài.
4. **Vì sao khi chạy `language_detection.py` trước đây file JSON không bị đổi sang 12 bài `mixed`?**
   - Vì hàm `main()` trong `language_detection.py` có câu lệnh `if track_id in checkpoint: continue` mà không có cờ `--recompute`, nên nó đọc thẳng kết quả cũ từ `data/raw/language_detection_v2_checkpoint.json` (vốn được sinh từ bản `HEAD` với 154 bài `mixed`).

### 4.2. Cải tiến có kiểm soát trong `language_detection.py` (Phiên bản `v2.1.0`)

- **Khôi phục thuật toán dòng chuẩn của `v2.0` (`HEAD`)** cho mọi bài hát có `total_weight > 0` (loại bỏ hoàn toàn lỗi mẫu số `script_counts_total` và lỗi cộng đúp `latin_weight`), bảo toàn nguyên vẹn **154 bài `mixed`** thực tế.
- **Giải quyết triệt để 26 bài `unknown` của bản `v2.0` bằng bằng chứng ký tự Unicode:**
  - Qua kiểm tra toàn bộ 26 bài bị gán `unknown` ở bản `v2.0`, cả 26 bài đều có độ dài dòng lớn nhất $\le 14$ ký tự (`max_line_w` từ 9 đến 14 ký tự) nên bị điều kiện `weight < MIN_TEXT_LENGTH` (`15`) của `v2.0` bỏ qua toàn bộ các dòng (`total_weight == 0`). Trong tiếng Hàn (Hangul) và tiếng Trung (Hanzi), mỗi ký tự đã là một âm tiết hoàn chỉnh, nên một câu hát 9–14 âm tiết chỉ gồm 9–14 ký tự.
  - Phiên bản `v2.1.0` bổ sung quy tắc fallback cấp bài hát **chỉ khi `total_weight == 0`**: tính tổng số ký tự script (`ko`, `ja`, `zh`) chia cho **tổng toàn bộ ký tự chữ trong bài (`total_script_chars + total_latin_chars`)**. Chỉ khi tổng số ký tự $\ge 15$ và một hệ chữ chiếm $\ge 70\%$ tổng ký tự toàn bài thì mới gán nhãn ngôn ngữ đó.
  - **Kết quả thay đổi nhãn (`v2.0` $\to$ `v2.1.0`):** Có đúng **26 bài hát** thay đổi nhãn (tất cả đều có bằng chứng ký tự từ $79.1\%$ đến $100.0\%$):
    - **25 bài `unknown -> ko`:** *Hi spring Bye* (IU), *Traffic light* (LEE MU JIN), *Winter Sleep* (IU), *Drama* (IU), *Our Blues, Our Life* (Lim Young Woong), *Dear my X* (KyoungSeo), *Loving You* (Lim Young Woong), *니가 왜 거기서 나와* (YOUNGTAK), *Free* (Xydo), *Gradation* (10CM), *Event Horizon* (Younha), *If We Ever Meet Again* (Lim Young Woong), *A Psalm of Life* (Lim Young Woong), *Polaroid* (Lim Young Woong), *Father* (Lim Young Woong), *Nest* (Lim Young Woong), *Heavenly fate* (LEECHANGSUB), *A Letter* (BUMJIN), *가시* (BUZZ), *Landing in Love* (HANRORO), *Let Me Love My Youth* (HANRORO), *Paradise of Rumors* (AKMU), *Blossom Parade* (PLAVE), *My whole world* (Car, the garden), *사랑은 봄비처럼...이별은 겨울비처럼* (Lim Hyun Jung).
    - **1 bài `unknown -> zh`:** `1XZE0InCx3SdCnLpVsPpZl` (*跳楼机* - LBI利比, 488 ký tự Hán tự / 98.39%).
    - **3 bài Instrumental (`not_found`)** không có lời bài hát giữ nguyên nhãn `unknown` (`None` trong JSON).
- **Lưu ý về các bản lời phiên âm Latin (Romanized Japanese/Korean):** Có 19 bài hát trong tập dữ liệu (ví dụ: *Lemon* - Kenshi Yonezu, *Still With You* - Jung Kook, *Sudden Shower* - ECLIPSE) mà nguồn LRCLIB trả về bản phiên âm chữ cái Latin (Romaji / Romanized Korean, không có ký tự Hangul hay Kana). `langdetect` nhận diện các chuỗi phiên âm này thành các mã ngôn ngữ ngẫu nhiên như `sw`, `cy`, `so`, `id`, `fi`, `af`, `ro`, `de`. Pipeline **cố ý không ép các file phiên âm chữ Latin này thành `ko` hay `ja`**, vì nếu gán `ko` cho một văn bản viết bằng chữ cái Latin, bộ tách từ và từ điển đại từ Hangul (`나, 내, 너...`) sẽ chạy trên chữ Latin và sinh ra kết quả sai (`pronoun_ratio = 0.0` giả tạo thay vì `NULL`). Thay vào đó, việc giữ chúng ngoài tập 5 ngôn ngữ chính giúp `extract_lyrics_features.py` tự động đánh dấu `extraction_status = "language_limited"` và đặt `pronoun_ratio = NULL`, `title_word_in_lyrics = NULL` một cách an toàn.

---

## 5. Quy Tắc Tokenization Theo Ngôn Ngữ và Định Nghĩa 10 Đặc Trưng (Stage 4)

### 5.1. Tiền xử lý và Chuẩn hóa văn bản (`normalize_text`, `clean_token`)

1. **Loại bỏ nhãn cấu trúc khi tách từ:** Mọi thẻ cấu trúc nằm trong ngoặc vuông như `[Chorus]`, `[Verse 1]`, `[Intro]` được loại bỏ trước khi tách từ (`ANY_SECTION_TAG_RE.sub(" ", text)`) để các từ khóa metadata như `"chorus"`, `"verse"` không bị tính nhầm vào `word_count` hay `title_word_in_lyrics`.
2. **Chuẩn hóa Unicode & chữ thường:** Toàn bộ văn bản và token được đưa về dạng chuẩn **Unicode NFC** (`unicodedata.normalize("NFC", ...)`), đảm bảo các chữ cái tiếng Việt có dấu tổ hợp và dựng sẵn được quy về duy nhất một biểu diễn, sau đó chuyển về chữ thường bằng `.casefold()`.
3. **Làm sạch token:** Cắt bỏ các dấu câu bao quanh đầu/cuối token (`"'’‘“”.,!?;:()[]{}<>…—–-·`), nhưng **giữ nguyên dấu nháy đơn và dấu gạch nối nằm bên trong từ** (ví dụ: `don't`, `i'm`, `well-known`). Mọi chuỗi chỉ chứa dấu câu hoặc ký tự trang trí không có chữ cái/chữ số Unicode (`[\p{L}\p{N}]`) đều bị loại bỏ, tuyệt đối không tính là từ.

### 5.2. Quy tắc Tokenization theo từng ngôn ngữ (`src/lyrics/language_rules.yaml` v1.1.0)

| Nhãn ngôn ngữ | Phương pháp (`method`) | Quy tắc chi tiết | Hạn chế phương pháp luận (được ghi nhận trong báo cáo) |
| :--- | :--- | :--- | :--- |
| **Tiếng Anh (`en`) & Tây Ban Nha (`es`)** | `unicode_word_regex` | Regex `[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*` sau đó chuẩn hóa NFC + `casefold()`. | Các từ rút gọn (`don't`, `we're`) được tính là 1 token duy nhất; nhất quán giữa `word_count`, `unique_word_count`, `avg_word_length` và `pronoun_ratio`. |
| **Tiếng Việt (`vi`)** | `whitespace_delimited_unicode_tokens` | Tách theo khoảng trắng sau khi loại bỏ dấu câu bao quanh và chuẩn hóa NFC. | Mỗi token tương ứng với một **tiếng (âm tiết)**, không phải từ ghép đa âm tiết. Do đó `word_count` cao hơn và `avg_word_length` nhỏ hơn so với cách tách từ ghép. |
| **Tiếng Hàn (`ko`)** | `whitespace_delimited_unicode_tokens` | Tách theo khoảng trắng tạo thành các đơn vị **eojeol** Hangul. | Không sử dụng bộ phân tích hình thái học (Mecab/Kiwi). Một eojeol thường bao gồm gốc từ gắn liền với tiểu từ/đuôi động từ, làm tăng `lexical_diversity` và giảm `word_count`. |
| **Đa ngôn ngữ (`mixed`)** | `unicode_script_aware_word_regex` | Giữ chung cả token chữ Latin và token chữ Hangul/CJK trong cùng dãy token chuẩn hóa. | Không áp dụng từ điển đại từ đơn ngữ cho bài `mixed` (`pronoun_ratio = NULL`) để tránh sai số dương giả / âm giả giữa hai ngôn ngữ. |
| **Ngôn ngữ khác / `unknown`** | `unicode_word_regex` | Tách token chữ/số cơ bản; gắn trạng thái `extraction_status = "language_limited"`. | Chỉ trích xuất các đặc trưng thống kê bề mặt; đặt `pronoun_ratio = NULL` và `title_word_in_lyrics = NULL`. |

### 5.3. Định nghĩa toán học, công thức, mẫu số và quy tắc xử lý `NULL` của 10 Features

Giả sử bài hát sau khi chuẩn hóa và tách từ có dãy token hợp lệ $T = (t_1, t_2, \dots, t_W)$ với $W = |T|$ và tập token phân biệt $U = \text{set}(T)$ với $V = |U|$.

| STT | Tên Feature | Kiểu dữ liệu (SQL / Python) | Công thức & Định nghĩa chính xác | Mẫu số | Quy tắc phân biệt `0` và `NULL` |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `word_count` | `INTEGER` / `int` | $W = |T|$: Tổng số token từ hợp lệ sau chuẩn hóa. | Không áp dụng | `NULL` khi file không tồn tại, lỗi đọc, hoặc rỗng/chỉ có khoảng trắng. Bằng `0` chỉ khi file có ký tự hiển thị nhưng chỉ toàn dấu câu (`extraction_status = failed`). |
| 2 | `unique_word_count` | `INTEGER` / `int` | $V = |\text{set}(T)|$: Số lượng token phân biệt sau chuẩn hóa chữ thường NFC. | Không áp dụng | `NULL` khi file không tồn tại, lỗi đọc, hoặc rỗng. Luôn thỏa $0 \le V \le W$. |
| 3 | `lexical_diversity` | `DOUBLE PRECISION` / `float` | $\text{TTR} = \dfrac{V}{W} = \dfrac{\text{unique\_word\_count}}{\text{word\_count}}$ | $W = \text{word\_count}$ | Chỉ xác định khi $W > 0$ (miền giá trị $(0, 1]$). Bắt buộc là `NULL` khi $W = 0$ hoặc `word_count IS NULL` (không chia cho 0). |
| 4 | `repetition_ratio` | `DOUBLE PRECISION` / `float` | $\text{Rep} = \dfrac{W - V}{W} = 1 - \text{lexical\_diversity}$ | $W = \text{word\_count}$ | Chỉ xác định khi $W > 0$ (miền giá trị $[0, 1)$). Bằng `0.0` khi mọi token trong bài đều khác nhau ($V = W$); bằng `NULL` khi $W = 0$ hoặc `NULL`. |
| 5 | `avg_word_length` | `DOUBLE PRECISION` / `float` | $\bar{L} = \dfrac{1}{W} \sum_{i=1}^{W} \text{len}(t_i)$ (đơn vị: ký tự Unicode/token). | $W = \text{word\_count}$ | Chỉ xác định khi $W > 0$. Bằng `NULL` khi $W = 0$ hoặc `NULL`. |
| 6 | `line_count` | `INTEGER` / `int` | Số dòng văn bản không rỗng sau khi tách theo ký tự xuống dòng và `.strip()`. | Không áp dụng | `NULL` khi lyrics rỗng, thiếu file hoặc lỗi đọc. |
| 7 | `chorus_repetition_count` | `INTEGER` / `int` | Số lần lặp lại **tăng thêm** ($k - 1$) của khối điệp khúc sau lần xuất hiện đầu tiên. | Không áp dụng | Xem Mục 5.4: Bằng `0` **chỉ khi** có đúng 1 nhãn `[Chorus]`/`[Refrain]`/`[Hook]` tường minh mà khối đó không lặp lại. Bằng `NULL` khi bài không có nhãn `[Chorus]` và không tìm thấy khối đa dòng lặp lại ($\ge 3$ dòng, $\ge 2$ dòng phân biệt, $\ge 6$ từ, độ tương đồng $\ge 0.80$). |
| 8 | `title_word_in_lyrics` | `BOOLEAN` / `bool` | `True` nếu $\exists w \in T_{\text{title\_content}}$ sao cho $w \in \text{set}(T)$; ngược lại `False`. Trong đó $T_{\text{title\_content}}$ là tập token chữ cái của `title` sau khi loại bỏ `title_stopwords` theo ngôn ngữ. | Không áp dụng | `NULL` khi `word_count == 0`/`NULL`, ngôn ngữ ngoài tập hỗ trợ (`en, vi, ko, mixed, es`), hoặc tiêu đề chỉ chứa số/stopwords (ví dụ: `"505"`, `"As It Was"`, `"I AM"`). |
| 9 | `punctuation_ratio` | `DOUBLE PRECISION` / `float` | $\dfrac{C_{\text{punct}}}{C_{\text{total}}}$, với $C_{\text{punct}}$ là số ký tự thuộc phạm trù dấu câu Unicode (`Pc, Pd, Pe, Pf, Pi, Po, Ps`) trong văn bản đã `.strip()`. | $C_{\text{total}} = \text{len}(\text{text.strip()})$ (tổng ký tự văn bản, tính cả khoảng trắng nội bộ) | Không trộn lẫn tỷ lệ trên ký tự và tỷ lệ trên token. Bằng `0.0` khi lời bài hát không có dấu câu nào; bằng `NULL` khi lời bài hát rỗng/thiếu. |
| 10 | `pronoun_ratio` | `DOUBLE PRECISION` / `float` | $\dfrac{P}{W}$, với $P$ là tổng số vị trí token trong $T$ khớp chính xác với từ điển đại từ của ngôn ngữ (`pronouns.<lang>`). | $W = \text{word\_count}$ | Sử dụng thuật toán **khớp cụm dài nhất trước** (*longest-phrase-first exact token match*, ví dụ `"chúng tôi"` tính 2 vị trí token một lần, không đếm trùng `"tôi"`). Bằng `0.0` khi bài thuộc `en, vi, ko, es` có $W > 0$ nhưng không dùng đại từ nào; bằng `NULL` khi ngôn ngữ là `mixed`, `unknown` hoặc ngôn ngữ khác. |

### 5.4. Quy tắc nhận diện điệp khúc (`detect_chorus_repetition`)

1. **Trường hợp 1 — Có nhãn cấu trúc tường minh (`[Chorus]`, `[Refrain]`, `[Hook]`, `[Điệp khúc]`):**
   - Nếu xuất hiện $k \ge 2$ nhãn điệp khúc trong bài $\implies$ `chorus_repetition_count` $= k - 1$.
   - Nếu xuất hiện đúng $k = 1$ nhãn điệp khúc: kiểm tra xem nội dung lời có khối đa dòng lặp lại hay không; nếu không có khối lặp thêm nhưng bài có ít nhất 2 dòng lời hợp lệ, trả về `0` (vì điệp khúc đã được định danh tường minh và xác nhận xuất hiện 1 lần duy nhất, lặp thêm 0 lần).
2. **Trường hợp 2 — Lyrics không có nhãn cấu trúc (chiếm 98.5% dữ liệu từ LRCLIB):**
   - Chuẩn hóa từng dòng (NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng).
   - Tìm khối dòng liên tiếp dài từ `min_block_lines = 3` đến `max_block_lines = 10` dòng, có ít nhất `min_unique_lines = 2` dòng phân biệt (chặn lỗi nhận diện một câu cảm thán đơn lẻ lặp lại liên tiếp như `"woah oh oh"` làm điệp khúc) và ít nhất `min_block_words = 6` từ, xuất hiện lặp lại không chồng lấp ở ít nhất 2 vị trí trong bài với độ tương đồng chuỗi từ `SequenceMatcher.ratio() >= 0.80`.
   - Chọn nhóm khối lặp có điểm cấu trúc `length * len(occurrences)` lớn nhất làm điệp khúc chính và trả về `len(occurrences) - 1`.
   - **Nếu không tìm thấy khối lặp nào thỏa mãn:** Trả về `None` (`NULL`), tuyệt đối **không gán bằng `0`**.

---

## 6. Kết Quả Chạy Pipeline, Validation và Unit Tests (Stage 5)

### 6.1. Thống kê kết quả trích xuất thực tế (`reports/report3/lyrics_features_extracted.csv`)

- **Tổng số dòng đầu vào (`lyrics_status.csv`):** 1,912 bài hát.
- **Số dòng đầu ra (`lyrics_features_extracted.csv`):** 1,912 bài hát duy nhất (0 trùng lặp).
- **Trạng thái trích xuất (`extraction_status`):**
  - `success`: **1,870 bài (97.80%)** — Trích xuất đầy đủ bộ đặc trưng cho 5 nhãn ngôn ngữ (`en`: 1,095, `vi`: 367, `ko`: 211, `mixed`: 154, `es`: 43).
  - `language_limited`: **39 bài (2.04%)** — Trích xuất đầy đủ 8 đặc trưng cơ bản (`word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, `line_count`, `chorus_repetition_count`, `punctuation_ratio`), đặt `NULL` cho `title_word_in_lyrics` và `pronoun_ratio`.
  - `not_found`: **3 bài (0.16%)** — 3 bài Instrumental không có lời bài hát, toàn bộ 10 đặc trưng đều là `NULL`.
- **Độ phủ từng đặc trưng trên 1,912 bài hát:**
  - `word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, `line_count`, `punctuation_ratio`: **1,909 / 1,912 (99.84%)** hợp lệ (`3` bài `NULL`).
  - `title_word_in_lyrics`: **1,843 / 1,912 (96.39%)** hợp lệ (`True`: 1,537 bài / 83.40%; `False`: 306 bài / 16.60%; `69` bài `NULL` gồm 3 bài `not_found`, 39 bài `language_limited`, và 27 bài có tiêu đề chỉ gồm số hoặc stopwords như `"505"`, `"906090"`, `"34+35"`, `"As It Was"`, `"You & Me"`, `"I AM"`).
  - `chorus_repetition_count`: **1,764 / 1,912 (92.26%)** hợp lệ (`148` bài `NULL` gồm 3 bài `not_found` và 145 bài không có nhãn `[Chorus]` và không tìm thấy khối đa dòng lặp lại).
  - `pronoun_ratio`: **1,716 / 1,912 (89.75%)** hợp lệ (`196` bài `NULL` gồm 154 bài `mixed`, 39 bài `language_limited`, và 3 bài `not_found`).

### 6.2. Kết quả chạy Validation (`src/lyrics/validate_lyrics_features.py --check-db`)

- **Tổng số dòng kiểm tra:** 1,912 dòng.
- **Số Track ID trùng lặp:** 0.
- **Tỷ lệ đạt chuẩn (`validation_status == "pass"`):** **1,912 / 1,912 (100.00%)**.
- **Số lỗi / cảnh báo (`issues`):** **0 lỗi**.
- **Kiểm tra đối chiếu chéo 4 nguồn (`Features CSV` $\leftrightarrow$ `Extraction Log` $\leftrightarrow$ `Inventory CSV` $\leftrightarrow$ `Supabase lyrics_features`):** **PASS 100%** (khớp tuyệt đối cả 1,912 `spotify_track_id`, `language`, `lyrics_status`, `word_count` và các cột đặc trưng).

### 6.3. Kết quả chạy Unit Tests (`tests/test_lyrics_pipeline.py`)

Đã xây dựng và thực thi thành công **11 kịch bản kiểm thử tự động** (`Ran 11 tests in 0.727s — OK`):
1. `test_01_empty_and_whitespace_lyrics`: Chuỗi rỗng và chuỗi chỉ chứa khoảng trắng/xuống dòng trả về `None` cho toàn bộ 10 đặc trưng, không chia cho 0.
2. `test_02_punctuation_only_lyrics`: Chuỗi chỉ chứa dấu câu (`"!!! ??? ..."` ) trả về `word_count = 0`, `unique_word_count = 0`, `line_count = 2`, `punctuation_ratio > 0.5`, và `None` cho toàn bộ các tỷ số có mẫu số `word_count`.
3. `test_03_highly_repetitive_lyrics`: Kiểm chứng chính xác công thức `lexical_diversity`, `repetition_ratio` và hằng đẳng thức `lexical_diversity + repetition_ratio == 1.0`.
4. `test_04_title_word_in_lyrics_true_false_and_stopwords`: Kiểm tra trường hợp tiêu đề xuất hiện (`True`), không xuất hiện (`False`), và tiêu đề chỉ gồm stopwords (`"As It Was"` $\to$ `None`).
5. `test_05_vietnamese_diacritics_and_pronouns`: Kiểm tra chuẩn hóa dấu tiếng Việt NFC, đếm âm tiết và khớp cụm đại từ dài nhất (`"chúng tôi"` tính 2 vị trí token, không đếm trùng `"tôi"`).
6. `test_06_english_contractions_and_hyphens`: Kiểm tra giữ nguyên từ rút gọn (`i'm`, `don't`, `we're`) và từ ghép gạch nối (`well-known`).
7. `test_07_korean_lyrics_and_language_detection`: Kiểm tra nhận diện ngôn ngữ tiếng Hàn trên các dòng ngắn $< 15$ ký tự Hangul và khớp đại từ tiếng Hàn có trợ từ.
8. `test_08_multilingual_mixed_lyrics`: Kiểm tra nhận diện bài hát song ngữ Hàn–Anh (`mixed`) và đảm bảo `pronoun_ratio` trả về `None`.
9. `test_09_lyrics_without_chorus_label_and_no_repetition`: Kiểm tra bài không có nhãn `[Chorus]` và không có đoạn lặp trả về `None` (không bao giờ giả định bằng `0`), trong khi bài có 1 nhãn `[Chorus]` tường minh không lặp trả về `0`.
10. `test_10_verbatim_chorus_repetition_tagged_and_untagged`: Kiểm tra đếm số lần lặp thêm của điệp khúc trên cả bài có nhãn `[Chorus]` và bài không nhãn.
11. `test_11_missing_file_and_encoding_error_pipeline`: Kiểm tra tích hợp CLI pipeline khi gặp file không tồn tại (`missing_file`), file rỗng (`empty_file`) và file hỏng byte UTF-8 (`read_error`).

---

## 7. Đồng Bộ Cơ Sở Dữ Liệu Supabase (Stage 6)

- **File SQL Schema:** `sql/03_lyrics_features.sql` (tương thích 100% với `supabase/migrations/20260926090000_create_lyrics_features.sql`).
- **Script Upsert & Xác minh:** `src/lyrics/upsert_lyrics_features.py` (ghi log kết quả tại `logs/supabase_upsert_log.json`, không lưu bất kỳ thông tin nhạy cảm hay credentials nào).
- **Kết quả thực thi thực tế trên Supabase (`public.lyrics_features`):**
  - **Lần chạy 1 (Populate từ bảng rỗng):** `Existing before = 0` $\to$ `Inserted = 1,912`, `Updated = 0`, `Skipped = 0`, `Errors = 0`. Đọc ngược lại từ Supabase: **1,912 dòng**, `Read-back mismatches = 0`.
  - **Lần chạy 2 (Kiểm chứng Idempotency khi chạy lại):** `Existing before = 1,912` $\to$ `Inserted = 0`, `Updated = 1,912`, `Skipped = 0`, `Errors = 0`. Đọc ngược lại từ Supabase: **1,912 dòng** (không sinh bản ghi trùng lặp, bảo toàn 100% giá trị `NULL`).

---

## 8. Từ Điển Dữ Liệu (Data Dictionary)

### 8.1. Bảng `public.lyrics_features` (Supabase) & `reports/report3/lyrics_features_extracted.csv`

| Tên cột | Kiểu dữ liệu SQL | Có trong Supabase? | Có trong CSV? | Miền giá trị hợp lệ | Mô tả chi tiết |
| :--- | :--- | :-: | :-: | :--- | :--- |
| `spotify_track_id` | `TEXT PRIMARY KEY` | Có | Có | Chuỗi 22 ký tự (FK $\to$ `songs`) | Mã định danh duy nhất của bài hát trên Spotify. |
| `title` | — | Không (ở bảng `songs`) | Có | Chuỗi văn bản | Tên bài hát gốc từ bảng `songs`. |
| `artist` | — | Không (ở bảng `songs`) | Có | Chuỗi văn bản | Tên nghệ sĩ chính từ bảng `songs`. |
| `market` | — | Không (ở bảng `sampling`) | Có | `VN`, `US`, `KR` (hoặc ghép `;`) | Các thị trường mà bài hát được lấy mẫu trong `song_quarter_sampling`. |
| `language` | `TEXT` | Có | Có | Mã ISO-639-1 / `mixed` / `unknown` | Ngôn ngữ lời bài hát được xác định bởi `language_detection.py` v2.1.0. |
| `source` | — | Không | Có | `LRCLIB`, `Genius`, `manual`, `""` | Nguồn thu thập lời bài hát. |
| `lyrics_status` | `TEXT` | Có | Có | `success`, `not_found` | Trạng thái thu thập lời bài hát. |
| `extraction_status` | — | Không | Có | `success`, `language_limited`, `not_found`, ... | Trạng thái trích xuất đặc trưng của pipeline. |
| `word_count` | `INTEGER` | Có | Có | Số nguyên $\ge 1$ hoặc `NULL` (`0` nếu chỉ có dấu câu) | Tổng số token từ hợp lệ theo quy tắc ngôn ngữ. |
| `unique_word_count` | `INTEGER` | Có | Có | Số nguyên $1 \le V \le W$ hoặc `NULL` | Số lượng token từ phân biệt sau khi chuẩn hóa chữ thường NFC. |
| `lexical_diversity` | `DOUBLE PRECISION` | Có | Có | Số thực $\in (0.0, 1.0]$ hoặc `NULL` | Độ đa dạng từ vựng ($\text{unique\_word\_count} / \text{word\_count}$). |
| `repetition_ratio` | `DOUBLE PRECISION` | Có | Có | Số thực $\in [0.0, 1.0)$ hoặc `NULL` | Tỷ lệ lặp lại token ($(\text{word\_count} - \text{unique\_word\_count}) / \text{word\_count}$). |
| `avg_word_length` | `DOUBLE PRECISION` | Có | Có | Số thực $> 0.0$ hoặc `NULL` | Độ dài trung bình của một token tính bằng số ký tự Unicode. |
| `line_count` | `INTEGER` | Có | Có | Số nguyên $\ge 1$ hoặc `NULL` | Số dòng lời bài hát không rỗng. |
| `chorus_repetition_count` | `INTEGER` | Có | Có | Số nguyên $\ge 0$ hoặc `NULL` | Số lần lặp lại tăng thêm của điệp khúc (`NULL` khi không đủ bằng chứng nhận diện điệp khúc). |
| `title_word_in_lyrics` | `BOOLEAN` | Có | Có | `true`, `false`, hoặc `NULL` | Có ít nhất một từ nội dung (không phải stopword) của tiêu đề xuất hiện trong lời bài hát hay không. |
| `punctuation_ratio` | `DOUBLE PRECISION` | Có | Có | Số thực $\in [0.0, 1.0]$ hoặc `NULL` | Tỷ lệ ký tự dấu câu Unicode trên tổng số ký tự của văn bản lời bài hát đã `.strip()`. |
| `pronoun_ratio` | `DOUBLE PRECISION` | Có | Có | Số thực $\in [0.0, 1.0]$ hoặc `NULL` | Tỷ lệ vị trí token khớp từ điển đại từ trên `word_count` (chỉ tính cho `en, vi, ko, es`; `NULL` cho `mixed`/khác). |
| `rules_version` | — | Không | Có | `"1.1.0"` | Phiên bản bộ quy tắc `language_rules.yaml` đã sử dụng. |

---

## 9. Tài Liệu Bàn Giao Cho Member 4 và Member 1 (Stage 9 Handoff)

### 9.1. Bàn giao cho Member 4 (Database Integration & Cross-Module QA)

1. **Bảng Supabase `public.lyrics_features`:** Đã được populate và xác minh trực tiếp toàn bộ **1,912 bản ghi** (`1,909` bài có đặc trưng lyrics, `3` bài Instrumental lưu `NULL` cho các cột đặc trưng).
2. **Danh sách bài hát hợp lệ và danh sách giới hạn/thiếu:**
   - `reports/report3/handoff_member4_valid_tracks.csv`: **1,909 bài hát** có lyrics hợp lệ (`lyrics_status = "success"`).
   - `reports/report3/handoff_member4_missing_and_limited_tracks.csv`: **42 bài hát** cần lưu ý đặc biệt, gồm **3 bài** `not_found` (Instrumental) và **39 bài** `language_limited` (có đầy đủ 8 đặc trưng cơ bản nhưng `title_word_in_lyrics` và `pronoun_ratio` là `NULL`).
3. **Cấu hình, từ điển và logs:**
   - Bộ quy tắc & từ điển đại từ/stopwords: `src/lyrics/language_rules.yaml` (v1.1.0).
   - Log trích xuất theo từng Track ID: `logs/lyrics_extraction_log.csv`.
   - Log kết quả upsert Supabase: `logs/supabase_upsert_log.json`.
   - Báo cáo kiểm định chất lượng: `reports/report3/lyrics_feature_validation.csv`.

### 9.2. Bàn giao cho Member 1 (Phân tích tương quan với `chart_performance` & `ChartScore`)

1. **File Handoff chuyên dụng để Join với `chart_performance`:**
   - Đường dẫn: `reports/report3/handoff_member1_chart_join.csv` (**2,098 dòng**).
   - **Khóa đối chiếu kép:** `(spotify_track_id, market)` — khớp **100% (2,098 / 2,098 cặp)** với các bài hát được lấy mẫu trong `public.chart_performance`.
2. **Quy tắc xử lý Missingness khi phân tích `ChartScore` (Tránh thiên lệch chọn mẫu):**
   - **Không làm mất các bài không có lyrics:** Cả 3 bài Instrumental (`6WY4wvlmgccWapnIg14Vy0` tại KR, `1Px44tw80Q4rcCl8FxDGOi` tại KR, `7M30L99pePlQ3wh8IxVUsD` tại VN) vẫn được giữ nguyên trong `handoff_member1_chart_join.csv` và trên bảng `public.lyrics_features` với cờ `has_valid_lyrics = False` (`lyrics_status = "not_found"`). Khi join từ `chart_performance` sang `lyrics_features`, hãy dùng **`LEFT JOIN` theo `spotify_track_id`** để không làm rớt các bài Instrumental khỏi thống kê tổng thể của bảng xếp hạng.
   - **Cờ trạng thái hỗ trợ lọc có kiểm soát:**
     - `has_valid_lyrics = True` (**2,095 / 2,098** quan sát thị trường; **1,909 / 1,912** bài duy nhất): Dùng cho mọi phân tích trên 7 đặc trưng cơ bản (`word_count`, `unique_word_count`, `lexical_diversity`, `repetition_ratio`, `avg_word_length`, `line_count`, `punctuation_ratio`).
     - `is_full_feature_set = True` (`extraction_status == "success"`, **2,054 / 2,098** quan sát thị trường; **1,870 / 1,912** bài duy nhất): Các bài thuộc 5 nhãn ngôn ngữ chính (`en, vi, ko, mixed, es`).
     - Đối với `chorus_repetition_count` ($n = 1,943 / 2,098$ quan sát thị trường) và `pronoun_ratio` ($n = 1,883 / 2,098$ quan sát thị trường, do loại trừ các bài `mixed`), cần lọc `WHERE chorus_repetition_count IS NOT NULL` hoặc `WHERE pronoun_ratio IS NOT NULL` riêng cho từng phân tích đơn biến tương ứng, tuyệt đối **không điền `0` vào các ô `NULL`**.

---

## 10. Hướng Dẫn Chạy Lại Toàn Bộ Pipeline (Reproducibility Guide)

Chạy tuần tự các lệnh sau từ thư mục gốc của repository (`music-chart-performance-analysis`):

```powershell
# Bước 1: Kiểm tra/chạy nhận diện ngôn ngữ (v2.1.0) và tạo bảng trạng thái lyrics_status.csv
python -X utf8 src/lyrics/language_detection.py --only-unknown
python -X utf8 src/lyrics/create_lyrics_handoff.py

# Bước 2: Tạo bảng kiểm kê chi tiết raw lyrics (reports/report3/lyrics_inventory.csv)
python -X utf8 src/lyrics/build_lyrics_inventory.py

# Bước 3: Trích xuất 10 đặc trưng lyrics (reports/report3/lyrics_features_extracted.csv)
python -X utf8 src/lyrics/extract_lyrics_features.py

# Bước 4: Đồng bộ (Upsert) lên bảng public.lyrics_features trên Supabase và xác minh
python -X utf8 src/lyrics/upsert_lyrics_features.py

# Bước 5: Chạy kiểm định chất lượng dữ liệu & đối chiếu chéo CSV - Log - Inventory - Supabase DB
python -X utf8 src/lyrics/validate_lyrics_features.py --check-db

# Bước 6: Chạy bộ 11 Unit Tests kiểm thử toàn diện pipeline
python -X utf8 -m unittest discover -s tests -v

# Bước 7: Tạo bảng thống kê mô tả, 9 biểu đồ PNG, dữ liệu biểu đồ và các file Handoff
python -X utf8 src/lyrics/generate_statistics_and_charts.py
```
