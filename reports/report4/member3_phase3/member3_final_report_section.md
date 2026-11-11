# CHƯƠNG CHUYÊN SÂU: PHÂN HỆ KHAI PHÁ CA TỪ NLP, ĐIỆU THỨC ÂM HỌC & TÍCH HỢP HỒI QUY MODEL 2 (THÀNH VIÊN 3)

**Học phần:** ADY201m — AI, Data Science with Python & SQL  
**Đề tài:** What Makes a Hit Song? So sánh các yếu tố ảnh hưởng đến độ phổ biến âm nhạc giữa Việt Nam, Hoa Kỳ và Hàn Quốc (Spotify Top 200 Weekly, 2021–2026)  
**Tác giả phụ trách:** Thành viên 3 (Lyrics & Text NLP Lead | Model 2 OLS Regression)  
**Phạm vi tài liệu:** Chương đóng góp học thuật độc lập của Thành viên 3 phục vụ tích hợp trực tiếp vào Báo cáo Tổng kết Chung của Đề tài (`reports/final_report.docx`).

---

## 1. Thu Thập Dữ Liệu Ca Từ & Tiền Xử Lý Văn Bản Đa Ngôn Ngữ

### 1.1. Xuất Xứ Thu Thập & Độ Phủ Dữ Liệu
Trong quy trình tích hợp đa phương thức của đề tài, Thành viên 3 chịu trách nhiệm thu thập, làm sạch và trích xuất các đặc trưng xử lý ngôn ngữ tự nhiên (NLP) từ ca từ của toàn bộ các ca khúc lọt vào bảng xếp hạng Spotify Weekly Top 200 tại ba quốc gia Việt Nam (`VN`), Hoa Kỳ (`US`) và Hàn Quốc (`KR`) giai đoạn 2021–2026.

Từ danh mục gốc gồm **$1,912$ ca khúc độc nhất** (`spotify_track_id`), quy trình thu thập tự động kết hợp đối soát bán tự động đã thu thập thành công ca từ cho **$1,909$ ca khúc, đạt độ phủ thực nghiệm $99.84\%$**:
* **LRCLIB API (Nguồn tự động sơ cấp):** Thu thập thành công $1,836$ bài hát ($96.03\%$).
* **Genius API / Web Scraping (Nguồn bổ trợ):** Thu thập thành công $29$ bài hát ($1.52\%$).
* **Nhập liệu thủ công có kiểm chứng (Manual Entry):** Bổ sung $44$ bài hát ($2.30\%$) cho các ca khúc Indie/Underground Việt Nam và các bản phối khí đặc thù không có trên cơ sở dữ liệu quốc tế.
* **Tác phẩm không lời (Instrumental / Not Found):** Đúng **$3$ ca khúc** (`lyrics_status = not_found`, gồm 1 bài tại VN và 2 bài tại KR) được xác định là nhạc không lời hoặc không tồn tại văn bản lời hát. Theo quy tắc tiền xử lý chuẩn tắc của đề tài (`src/features/prepare_features.py`), 3 quan sát này được loại bỏ khỏi mẫu hồi quy hoàn chỉnh, bảo đảm $100\%$ các bài hát đưa vào huấn luyện đều có dữ liệu ca từ hợp lệ.

### 1.2. Chuẩn Hóa Văn Bản & Tách Từ Đa Ngôn Ngữ
Văn bản ca từ được xử lý qua đường ống chuẩn hóa chuẩn tắc (`src/lyrics/extract_lyrics_features.py`):
1. **Chuẩn hóa Unicode:** Áp dụng dạng chuẩn Unicode NFC (`unicodedata.normalize('NFC', text)`) nhằm đồng nhất các ký tự có dấu trong tiếng Việt (tổ hợp vs dựng sẵn) và các khối ký tự Hangul tiếng Hàn.
2. **Loại bỏ siêu dữ liệu cấu trúc bài hát:** Sử dụng biểu thức chính quy (`regex`) bóc tách toàn bộ các thẻ tiêu đề đoạn không thuộc lời hát chính thức như `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`, `[Điệp khúc]`.
3. **Phân tách từ ngữ thích ứng ngôn ngữ (Language-Aware Tokenization):**
   * *Tiếng Anh (`en`):* Tách theo ranh giới từ `\b\p{L}+\b`, chuyển về chữ thường (`casefold`), loại bỏ dấu câu bao quanh.
   * *Tiếng Việt (`vi`):* Do tiếng Việt là ngôn ngữ đơn lập (isolating language), mỗi âm tiết (*tiếng*) được phân tách tự nhiên qua khoảng trắng và đóng vai trò là một đơn vị từ vựng độc lập (word token).
   * *Tiếng Hàn (`ko`):* Phân tách theo khoảng trắng thành các cụm ngữ pháp *eojeol* (cụm gồm gốc từ kết hợp trợ từ hoặc đuôi biến tố).
   * *Ca khúc đa ngữ (`mixed`):* Áp dụng quy tắc bóc tách ký tự Unicode mở rộng `\p{L}+` bảo toàn đồng thời ký tự Latin có dấu, ký tự Hangul và tiếng Anh chêm xen.

---

## 2. Định Nghĩa & Trích Xuất Các Đặc Trưng Ca Từ NLP

Từ kho văn bản đã làm sạch, đường ống NLP trích xuất 10 chỉ số văn bản thô, phản ánh 3 chiều kích cấu trúc: Độ dài & Dung lượng, Tính đa dạng từ vựng, và Kết cấu hình thức bài hát.

**Bảng 1. Định nghĩa vận hành, thang đo và vai trò của các đặc trưng ca từ.**

| Tên Đặc Trưng | Ký hiệu / Thang đo | Định nghĩa toán học / Thuật toán trích xuất | Vai trò trong Đề tài |
| :--- | :--- | :--- | :--- |
| `word_count` | Từ (tokens) / bài | Tổng số lượng tokens từ ngữ sau khi chuẩn hóa văn bản. | **Đặc trưng chính Model 2** (chuẩn hóa Z-score sau biến đổi $\ln(1 + x)$) |
| `unique_word_count` | Từ phân biệt / bài | Số lượng surface tokens duy nhất, không trùng lặp trong bài hát. | Sàng lọc VIF: **Loại bỏ** (cộng tuyến tích số với `word_count`) |
| `lexical_diversity` | Tỷ số $[0, 1]$ | Tỷ lệ loại-từ trên tổng từ (Type-Token Ratio: $\text{TTR} = \text{unique} / \text{word}$). | Sàng lọc VIF: **Loại bỏ** (đồng nhất thức đại số $1 - \text{Repetition}$) |
| `repetition_ratio` | Tỷ số $[0, 1]$ | Tỷ lệ ca từ lặp lại: $(\text{word} - \text{unique}) / \text{word} = 1 - \text{TTR}$. | **Đặc trưng chính Model 2** (đo lường độ lặp lại hook ca từ) |
| `avg_word_length` | Ký tự / từ | Tổng số ký tự chữ cái chia cho tổng số từ: $\sum \text{len}(w_i) / \text{word\_count}$. | **Đặc trưng chính Model 2** (đo lường độ phức tạp hình thái học) |
| `line_count` | Dòng thơ / bài | Tổng số dòng ca từ thực tế có nội dung (không tính dòng trống). | **Đặc trưng chính Model 2** (đo lường độ dài kết cấu bài hát) |
| `chorus_repetition_count`| Lần lặp điệp khúc | Số lần khối điệp khúc lặp lại (dựa trên thuật toán so khớp khối nhiều dòng tương đồng $\ge 70\%$). Gán giá trị `NULL` nếu không tìm thấy khối lặp. | **Đặc trưng chính Model 2** (điền $0.0$ kỹ thuật cho các trường hợp `NULL`) |
| `punctuation_ratio` | Ký tự / tổng ký tự | Tỷ số giữa các ký tự dấu câu biểu cảm (`!`, `?`, `...`) trên tổng số ký tự. | **Đặc trưng chính Model 2** (đo lường mật độ cảm xúc văn bản) |

---

## 3. Phân Định Rõ 5 Phạm Vi Dữ Liệu Quan Sát (Data Scopes)

Để đảm bảo tính trung thực học thuật và không nhầm lẫn cỡ mẫu giữa các phép thử độc lập và mô hình hồi quy, đề tài phân định rõ **5 phạm vi dữ liệu quan sát**:

1. **Scope A — Quan sát Toàn thị trường ($N = 2,098$ dòng / $1,912$ ca khúc độc nhất):** Gồm `VN = 608`, `US = 797`, `KR = 693`. Do có $169$ bài hát xuất hiện đồng thời trên nhiều bảng xếp hạng, mẫu này được dùng cho phân tích mô tả trực quan và kiểm định độ nhạy GEE có kiểm soát cấu trúc cụm bài hát ($N = 2,095$ dòng ca từ hợp lệ sau khi loại 3 bài không lời).
2. **Scope B — Ca khúc Độc nhất ($N = 1,912$ bài hát `spotify_track_id`):** Dùng để kiểm toán xuất xứ nguồn thu thập dữ liệu ca từ.
3. **Scope C — Ca khúc Độc quyền Thị trường ($N = 1,743$ bài hát độc nhất):** Gồm `VN = 488`, `US = 721`, `KR = 534`. Mẫu này loại trừ triệt để $169$ bài hát đa thị trường nhằm **bảo toàn hoàn toàn giả định quan sát độc lập (Independence of Observations)** cho các kiểm định phi tham số suy luận chính Kruskal-Wallis ($N = 1,740$ cho từ vựng; $N = 1,601$ cho điệp khúc) và Pearson Chi-Square ($N = 1,743$).
4. **Scope D — Mẫu Hồi quy Đóng băng ($N = 2,095$ dòng $\to$ Train $N = 1,685$ dòng / $1,527$ cụm bài; Test $N = 410$ dòng / $382$ cụm bài):** Được phân chia qua `GroupShuffleSplit(seed=43)` theo `spotify_track_id`, bảo đảm không rò rỉ dữ liệu giữa huấn luyện và kiểm định ngoài mẫu. Dùng để kiểm toán VIF và huấn luyện chuỗi mô hình OLS Model 1, Model 2, Model 3.
5. **Scope E — Mẫu Khảo sát Lịch sử Report 4 EDA ($N = 2,860$ dòng):** Dữ liệu gộp ban đầu từ `fig3_tonal_harmony_market.csv`, dùng riêng cho mục đích đối soát hòa giải các tuyên bố lịch sử cũ.

---

## 4. Xử Lý Giá Trị Trung Vị Dự Phòng Ca Từ (CP1 — Market-Specific Fallback Medians)

Tệp siêu dữ liệu: [`data/processed/lyrics_market_medians.json`](file:///C:/Users/Biiiii/Desktop/music-chart-performance-analysis/data/processed/lyrics_market_medians.json)

Để phục vụ triển khai ứng dụng thực tế (Tab 3 Audio Upload — nơi người dùng tải file âm thanh và hệ thống không thể bóc tách ca từ tự động nếu thiếu mô hình nhận dạng giọng nói ASR), Thành viên 3 đã xây dựng bộ tham số trung vị dự phòng theo từng thị trường.

**Nguyên tắc kỹ thuật khắt khe:**
* Toàn bộ giá trị trung vị **chỉ được tính toán độc quyền trên tập Train** ($N = 1,685$ dòng / $1,527$ cụm bài), tuyệt đối không sử dụng tập Test nhằm triệt tiêu hoàn toàn nguy cơ rò rỉ thông tin (data leakage).
* Tệp JSON bao gồm đầy đủ 6 đặc trưng Model 2 cho cả 3 thị trường:
  * **Hàn Quốc (KR, $n = 551$ dòng Train):** `word_count` $= 302.0$, `repetition_ratio` $= 0.5787$, `avg_word_length` $= 3.4726$, `line_count` $= 51.0$, `chorus_repetition_count` $= 1.0$, `punctuation_ratio` $= 0.0488$.
  * **Hoa Kỳ (US, $n = 632$ dòng Train):** `word_count` $= 361.0$, `repetition_ratio` $= 0.6274$, `avg_word_length` $= 3.8426$, `line_count` $= 50.0$, `chorus_repetition_count` $= 1.0$, `punctuation_ratio` $= 0.0465$.
  * **Việt Nam (VN, $n = 502$ dòng Train):** `word_count` $= 392.0$, `repetition_ratio` $= 0.6221$, `avg_word_length` $= 3.3544$, `line_count` $= 52.0$, `chorus_repetition_count` $= 1.0$, `punctuation_ratio` $= 0.0307$.
* Toàn bộ 18 giá trị trung vị lưu trữ trong tệp JSON đã được kiểm chứng đối soát chéo độc lập với `get_split().raw_train`, đạt độ chính xác số học $100\%$.

---

## 5. Kiểm Toán Đa Cộng Tuyến (VIF) & Biến Đổi Tham Số Hồi Quy

### 5.1. Triệt Tiêu Đa Cộng Tuyến Tuyến Tính & Cộng Tuyến Tích Số
Trước khi ước lượng mô hình hồi quy OLS trên tập Train ($N = 1,685$), 8 biến ca từ ứng viên được sàng lọc qua Hệ số Phóng đại Phương sai ($\text{VIF}_j = 1 / (1 - R_j^2)$). Kết quả kiểm toán làm rõ hai cơ chế đa cộng tuyến:
1. **Đồng nhất thức đại số tuyến tính hoàn hảo:**
   $$\text{lexical\_diversity} = \frac{\text{unique\_word\_count}}{\text{word\_count}}, \quad \text{repetition\_ratio} = 1 - \text{lexical\_diversity}$$
   Sau khi chuẩn hóa Z-score, $\text{repetition\_ratio\_std} \equiv -\text{lexical\_diversity\_std}$. Nếu giữ cả hai biến cùng hệ số chặn `const`, ma trận hiệp phương sai bị suy biến hoàn toàn với $\text{VIF} = 1.0008 \times 10^{15}$.  
   $\to$ **Quyết định:** Loại bỏ vĩnh viễn `lexical_diversity_std` (`forbidden_cols`), giữ lại `repetition_ratio_std` để đo lường trực tiếp mức độ lặp lại của hook bài hát.
2. **Cộng tuyến tích số (Multiplicative Collinearity):**
   $$\text{unique\_word\_count} = \text{word\_count} \times (1 - \text{repetition\_ratio})$$
   Sau khi loại bỏ `lexical_diversity`, VIF của `word_count_std` và `unique_word_count_std` vẫn ở mức báo động nghiêm trọng ($16.42$ và $14.80$), vượt xa ngưỡng an toàn $5.0$.  
   $\to$ **Quyết định:** Loại bỏ tiếp `unique_word_count_std`. Ngay lập tức, VIF của `word_count_std` hạ từ $16.42$ xuống **$3.40$** (và đạt **$3.77$** trong ma trận 24 biến Model 2 sau khi lấy log). Toàn bộ 6 biến ca từ của Model 2 đều đạt $\text{VIF} \le 3.77 < 5.0$, bảo vệ tuyệt đối tính khả nghịch của $(X^\top X)^{-1}$.

### 5.2. Biến Đổi $\ln(1 + \text{word\_count})$ & Nén Đòn Bẩy (Leverage Reduction)
Biến `word_count` thô trên tập Train có độ lệch phải rất lớn ($\text{Skewness} = +1.5765$) và đuôi nhọn nặng ($\text{Kurtosis} = +6.4205$) do các bài rap có lượng từ lên tới $1,000$ – $1,985$ từ.

Áp dụng phép biến đổi logarithm tự nhiên $\text{word\_count}_{\text{log}} = \ln(1 + \text{word\_count})$ mang lại hai chuyển biến phương pháp luận quan trọng:
* Giảm gần một nửa độ bất đối xứng (Skewness hạ từ $+1.58$ xuống **$-0.79$**) và nén $58.8\%$ độ nhọn đuôi (Kurtosis hạ từ $+6.42$ xuống **$+2.65$**).
* Trên nhóm 9 bài hát rap có dung lượng $\ge 1,000$ từ, giá trị đòn bẩy đường chéo ma trận $H$ ($h_{ii}$) trong Model 2 giảm từ **$0.0764$ xuống $0.0502$** (giảm $34.3\%$), đưa đòn bẩy trung bình từ $0.0366$ (vượt ngưỡng cảnh báo $2p/N = 0.0297$) về mức an toàn **$0.0253 < 2p/N$**. Khoảng cách Cook tối đa giảm từ $0.0032$ về **$0.0018 < 4/N = 0.0024$**, triệt tiêu nguy cơ các ca khúc rap cá biệt làm sai lệch toàn bộ hệ số OLS.

---

## 6. Phân Tích Thống Kê Dị Biệt Ca Từ & Điệu Thức Âm Học (Phase 2.4 Audit)

### 6.1. Kiểm Định Kruskal-Wallis Trên Mẫu Độc Quyền Thị Trường ($N = 1,743$)
Trên mẫu Scope C gồm $1,743$ ca khúc độc quyền thị trường ($1,740$ bài đủ dữ liệu từ vựng; $1,601$ bài đủ dữ liệu điệp khúc), kiểm định phi tham số Kruskal-Wallis có hiệu chỉnh sai số họ phép thử FWER bằng thuật toán Holm-Bonferroni ở mức $\alpha = 0.05$ cho kết quả chính thức:

**Bảng 2. Kết quả kiểm định Kruskal-Wallis và cỡ hiệu ứng $\varepsilon^2$ trên mẫu ca khúc độc quyền thị trường.**

| STT | Đặc Trưng Ca Từ | Cỡ Mẫu ($N$) | Thống Kê $H$ ($df=2$) | $p$-value Thô | $p$-value Holm | Cỡ Hiệu Ứng ($\varepsilon^2$) | Quyết Định Thống Kê |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `avg_word_length` | 1,740 | **557.0660** | $1.08 \times 10^{-121}$ | **$6.50 \times 10^{-121}$** | **0.3196** (Rất lớn) | **Bác bỏ $H_0$** |
| 2 | `word_count` | 1,740 | **269.8847** | $2.48 \times 10^{-59}$ | **$1.24 \times 10^{-58}$** | **0.1542** (Lớn) | **Bác bỏ $H_0$** |
| 3 | `lexical_diversity`* | 1,740 | **147.7139** | $8.40 \times 10^{-33}$ | **$3.36 \times 10^{-32}$** | **0.0839** (Trung bình) | **Bác bỏ $H_0$** |
| 4 | `repetition_ratio`* | 1,740 | **147.7139** | $8.40 \times 10^{-33}$ | **$3.36 \times 10^{-32}$** | **0.0839** (Trung bình) | **Bác bỏ $H_0$** |
| 5 | `unique_word_count` | 1,740 | **135.7840** | $3.27 \times 10^{-30}$ | **$6.55 \times 10^{-30}$** | **0.0770** (Trung bình) | **Bác bỏ $H_0$** |
| 6 | `chorus_repetition_count` | 1,601 | 2.3681 | 0.3060 | 0.3060 | 0.0002 (Không đáng kể) | **Không bác bỏ $H_0$** |

*\* Ghi chú: `lexical_diversity` và `repetition_ratio` có thống kê $H$ bằng nhau tuyệt đối do phụ thuộc đại số tuyến tính.*

**Luận giải thực nghiệm chuyên sâu:**
1. **Độ dài từ trung bình (`avg_word_length`, $\varepsilon^2 = 0.3196$):** Hoa Kỳ (US) có trung vị độ dài từ cao nhất ($3.8530$ ký tự/từ), vượt trội so với Việt Nam ($3.3194$) và Hàn Quốc ($3.2927$). Cỡ hiệu ứng cực lớn này giải thích xấp xỉ $32\%$ biến thiên thứ hạng nhưng là một **nhiễu ngôn ngữ học bề mặt (morphological artifact)** bắt nguồn từ quy ước phân tách token: tiếng Anh chứa nhiều từ ghép đa âm tiết dài; tiếng Việt phân tách từng âm tiết đơn lập ngắn; tiếng Hàn phân tách theo cụm *eojeol* pha trộn ký tự Hangul và tiếng Anh.
2. **Tổng số lượng từ (`word_count`, $\varepsilon^2 = 0.1542$):** Việt Nam (VN) dẫn đầu về độ dài bài hát với trung vị **$421.0\text{ từ}$** (trung bình $456.41\text{ từ}$), áp đảo Hoa Kỳ ($356.0\text{ từ}$) và Hàn Quốc ($286.0\text{ từ}$). Điều này phản ánh văn hóa thưởng thức âm nhạc tại Việt Nam ưa chuộng các thể loại trữ tình tự sự (Ballad, R&B) nhiều ca từ kể chuyện và mật độ lời dày đặc của Rap Việt.
3. **Tính bất biến của cấu trúc lặp điệp khúc (`chorus_repetition_count`):** Hoàn toàn không có sự khác biệt có ý nghĩa thống kê giữa 3 quốc gia ($p = 0.3060, \varepsilon^2 = 0.0002$). Cả 3 thị trường đều có **trung vị bằng đúng $1.0000$ lần lặp bổ sung** (tương ứng kết cấu 2 đoạn điệp khúc tiêu chuẩn trong bài hát pop). Điệp khúc lặp lại là quy chuẩn sáng tác thương mại toàn cầu nhằm định vị giai điệu (hook recall) trong tâm trí người nghe, không bị phân hóa theo biên giới địa lý.

### 6.2. Kiểm Định Chi-Square Điệu Thức Âm Học (`audio_mode`)
Trên cùng mẫu độc quyền thị trường ($N = 1,743$), bảng tiếp biến $3 \times 2$ kiểm định tính độc lập giữa `market` và điệu thức âm nhạc (`Major` vs. `Minor`):

**Bảng 3. Bảng tiếp biến $3 \times 2$ và kiểm định Chi-Square điệu thức âm học.**

| Thị Trường | Quan Sát Điệu Trưởng (Major) | Tỷ Lệ Nội Bộ (%) | Quan Sát Điệu Thứ (Minor) | Tỷ Lệ Nội Bộ (%) | Tổng Thị Trường ($N_m$) | Kết Quả Kiểm Định Pearson |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Việt Nam (VN)** | **336** (Kỳ vọng: 292.02) | **68.85%** | 152 (Kỳ vọng: 195.98) | 31.15% | 488 | \multirow{3}{*}{**$\chi^2(2) = 24.3142$**<br>**$p = 5.25 \times 10^{-6}$**<br>**$\text{Cramér's V} = 0.1181$**<br>(Hiệu ứng nhỏ, Bác bỏ $H_0$)} |
| **Hoa Kỳ (US)** | **396** (Kỳ vọng: 431.44) | **54.92%** | 325 (Kỳ vọng: 289.56) | 45.08% | 721 | |
| **Hàn Quốc (KR)** | **311** (Kỳ vọng: 319.54) | **58.24%** | 223 (Kỳ vọng: 214.46) | 41.76% | 534 | |
| **Toàn Bộ Mẫu** | **1,043** | **59.84%** | **700** | **40.16%** | **1,743** | |

*Nhận xét:* Bác bỏ $H_0$ với $p = 5.25 \times 10^{-6}$, phản ánh thị trường Việt Nam có sự thiên lệch mạnh mẽ về Điệu Trưởng ($68.85\%$), trong khi Hoa Kỳ có tỷ lệ bài hát Điệu Thứ cao nhất ($45.08\%$) do thị hiếu tiêu thụ các dòng nhạc Trap, Hip-hop và Alternative R&B u tối. Hệ số $\text{Cramér's V} = 0.1181$ khẳng định mối liên hệ này có ý nghĩa thống kê nhưng ở mức độ tác động khiêm tốn.

### 6.3. Phân Tích Độ Nhạy GEE & Kiểm Toán Dữ Liệu Thiếu Điệp Khúc
1. **Mô hình Rank-GEE Toàn mẫu ($N = 2,095$ dòng / $1,909$ cụm bài):** Áp dụng mô hình phương trình ước lượng tổng quát với sai số chuẩn bánh kẹp vững phân cụm theo `spotify_track_id` xác nhận chiều hướng và mức ý nghĩa tương đồng hoàn toàn với Kruskal-Wallis (`avg_word_length`: $\text{Wald }\chi^2 = 772.35, p < 10^{-160}$; `word_count`: $\text{Wald }\chi^2 = 290.08, p < 10^{-60}$; `chorus_repetition`: $\text{Wald }\chi^2 = 3.33, p = 0.1891$).
2. **Kiểm toán dữ liệu thiếu có hệ thống ở điệp khúc:** Hiện tượng không nhận diện được đoạn điệp khúc (`NULL` ở $142$ bài độc quyền) gắn liền có hệ thống với thị trường: **Hoa Kỳ có tỷ lệ khuyết lên tới $12.76\%$ ($92/721$)**, cao gấp đôi Việt Nam ($5.74\%$) và Hàn Quốc ($4.12\%$) ($\chi^2 = 35.8621, p = 1.63 \times 10^{-8}$). Các bài rap Mỹ thường có kết cấu đọc liên tục (through-composed) không lặp lại ca từ cố định. Nếu gán bằng 0 trong kiểm định phân phối, giá trị $p$ bị kéo lệch xuống $0.0212$. Do đó, phân tích trường hợp hoàn chỉnh ($p = 0.3060$) bắt buộc phải là kết quả chính thức; việc điền $0.0$ chỉ là giải pháp kỹ thuật trong tiền xử lý hồi quy OLS.

---

## 7. Mô Hình Hồi Quy Model 2 OLS & Kiểm Định Wald Đồng Thời

### 7.1. Đặc Tả Mô Hình & Ước Lượng Sai Số Chuẩn Vững Theo Cụm
Mô hình ước lượng điểm thành công bảng xếp hạng $\text{ChartScore} = 101 - \text{lifetime\_average\_rank}$ trên tập Train ($N = 1,685$ dòng / $1,527$ cụm bài hát duy nhất).

Để xử lý hiện tượng phương sai thay đổi (Breusch-Pagan $\text{LM} = 84.07, p = 3.99 \times 10^{-6}$) và sự phụ thuộc giữa các quan sát của cùng một ca khúc xuất hiện trên nhiều thị trường, toàn bộ các mô hình sử dụng **sai số chuẩn vững theo cụm bài hát** (Cluster-Robust Standard Errors grouped by `spotify_track_id`; Cameron & Miller, 2015) thông qua hàm chuẩn tắc `fit_ols_cluster`:
$$\text{Model 2}: \quad \text{ChartScore}_i = \beta_0 + \sum_{j=1}^{14} \beta_j \text{Audio}_{ji} + \beta_{\text{mode}} \text{Mode}_i + \beta_{\text{year}} \text{Year}_i + \beta_{\text{VN}} \text{Market\_VN}_i + \beta_{\text{KR}} \text{Market\_KR}_i + \sum_{m=1}^{6} \gamma_m \text{Lyrics}_{mi} + \varepsilon_i$$

**Bảng 4. Hệ số hồi quy OLS với sai số chuẩn cụm cho biến `mode` và khối 6 biến ca từ trong Model 2 Canonical (Train $N = 1,685$).**

| Tên Biến Dự Báo | Hệ số Ước lượng ($\hat{\beta}$) | Sai số Chuẩn Cụm ($SE$) | Thống kê $t$ | $p$-value | Khoảng Tin Cậy 95% | VIF | Diễn giải kinh tế lượng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `mode` (Major=1, Minor=0) | **+3.4233** | 2.1304 | +1.6069 | 0.1081 | $[-0.7521,\ +7.5987]$ | 1.2014 | Điệu Trưởng có điểm trung bình cao hơn 3.42 điểm, nhưng chưa đạt $p < 0.05$. |
| `word_count_std` | **-2.0718** | 1.9776 | -1.0476 | 0.2948 | $[-5.9477,\ +1.8042]$ | 3.7737 | Tăng 1 SD lượng từ log đi kèm điểm giảm 2.07 điểm (không ý nghĩa). |
| `repetition_ratio_std` | **+1.9631** | 1.3462 | +1.4582 | 0.1448 | $[-0.6755,\ +4.6017]$ | 1.7669 | Tỷ lệ lặp lại ca từ có xu hướng dương nhưng không ý nghĩa riêng lẻ. |
| `avg_word_length_std` | **-0.8968** | 1.0262 | -0.8739 | 0.3822 | $[-2.9082,\ +1.1146]$ | 1.2150 | Độ dài từ hình thái học không có liên hệ với thứ hạng bài hát. |
| `line_count_std` | **+2.5327** | 1.5103 | +1.6770 | 0.0935 | $[-0.4274,\ +5.4927]$ | 2.7631 | Số dòng lời có xu hướng dương ở mức gợi ý ($p < 0.10$). |
| `punctuation_ratio_std` | **+2.0943** | 1.1159 | +1.8767 | 0.0606 | $[-0.0929,\ +4.2815]$ | 1.3776 | Mật độ dấu câu có xu hướng dương ở mức gợi ý ($p = 0.0606$). |
| `chorus_repetition_count_std`| **+0.9848** | 1.0448 | +0.9425 | 0.3459 | $[-1.0630,\ +3.0325]$ | 1.2983 | Lặp điệp khúc không có liên hệ thống kê với điểm thứ hạng. |

### 7.2. Kiểm Định Wald Đồng Thời Khối Ca Từ & Giải Mã Nghịch Lý Thống Kê
Để kiểm tra xem khối ca từ xét trên tổng thể có đóng góp phần giải thích nào vào điểm thứ hạng hay không, kiểm định Wald đồng thời (Joint Wald Block Test) được thực thi trên ma trận hiệp phương sai bánh kẹp:
$$H_0: \gamma_1 = \gamma_2 = \gamma_3 = \gamma_4 = \gamma_5 = \gamma_6 = 0$$

**Kết quả kiểm định chính thức:**
* Trong **Model 2 Canonical có biến kiểm soát (24 biến)**:
  $$\text{Wald }\chi^2(6) = \mathbf{13.5932}, \quad p = \mathbf{0.0345} \quad (\text{Bác bỏ } H_0 \text{ ở mức } \alpha = 0.05)$$
* Trong **Model 2 Lồng nhau không có biến kiểm soát (21 biến)**:
  $$\text{Wald }\chi^2(6) = \mathbf{15.5722}, \quad p = \mathbf{0.0162} \quad (\text{Bác bỏ } H_0 \text{ ở mức } \alpha = 0.05)$$

> [!IMPORTANT]
> **Giải mã Nghịch lý: Khối Ca Từ Có Ý Nghĩa Đồng Thời Nhưng Từng Biến Riêng Lẻ Không Đạt $p < 0.05$:**  
> Trong Bảng 4, không có hệ số ca từ nào đạt $p < 0.05$ riêng lẻ. Tuy nhiên, kiểm định Wald 6 bậc tự do lại đạt ý nghĩa thống kê rõ rệt ($p = 0.0345$).  
> *Nguyên nhân toán học:* Các thuộc tính ca từ chia sẻ mối tương quan hiệp phương sai nội bộ vừa phải ($\text{VIF} \in [1.77, 3.77]$) — một bài hát có nhiều từ (`word_count`) thường có nhiều dòng (`line_count`) và độ lặp lại cao (`repetition_ratio`). Khi thực hiện kiểm định $t$ đơn lẻ, phương sai giải thích bị phân mảnh cho từng biến, làm thổi phồng sai số chuẩn $SE(\hat{\beta}_j)$ và làm giảm lực kiểm định của từng hệ số. Ngược lại, kiểm định Wald đánh giá dạng toàn phương $\hat{\gamma}^\top \hat{\Sigma}_{\text{lyrics}}^{-1} \hat{\gamma}$, kiểm tra xem **toàn bộ véc-tơ hệ số ca từ có đồng thời bằng véc-tơ 0 hay không**. Kết quả $p = 0.0345$ chứng minh rằng xét trên tổng thể, thông tin văn bản NLP đóng góp phần giải thích có ý nghĩa thống kê vào điểm thành công của bài hát trong tập huấn luyện.

---

## 8. So Sánh Mô Hình & Thực Tế Dự Báo Ngoài Mẫu (Out-of-Sample Reality)

Đánh giá hiệu năng dự báo ngoài mẫu trên tập Test độc lập gồm **$N = 410$ quan sát / $382$ cụm bài hát** hoàn toàn chưa từng xuất hiện trong tập Train:

**Bảng 5. So sánh hiệu năng dự báo giữa Model 1 và Model 2 trên tập Test ($N = 410$).**

| Chỉ số Đánh giá | Model 1 (Audio Baseline + Controls) | Model 2 (Audio + Lyrics + Controls) | Chênh lệch ($\Delta = \text{M2} - \text{M1}$) | Khoảng Tin Cậy Ghép Cặp (Cluster Bootstrap 95%) |
| :--- | :---: | :---: | :---: | :---: |
| **Số biến dự báo ($p$)** | 18 | 24 | $+6$ biến | — |
| **$R^2$ Train (In-sample)** | $0.0480$ | $0.0562$ | $+0.0082$ | — |
| **$R^2$ Test (Out-of-sample)** | **$0.04699$** | **$0.03649$** | **$-0.01050$** | $[-0.037, +0.029]$ (Chứa 0) |
| **Adjusted $R^2$ Test** | **$+0.00312$** | **$-0.02357$** | **$-0.02669$** | — |
| **MAE Test (Sai số tuyệt đối)** | **$29.7387$** | **$30.0569$** | **$+0.3182$** | $[-0.49, +0.19]$ (Chứa 0, $\Delta\text{MAE} = -0.15$) |
| **RMSE Test (Căn bậc hai MSE)** | **$38.0461$** | **$38.2550$** | **$+0.2089$** | — |

**Luận giải phương pháp luận cốt lõi:**
1. **Tại sao khối biến có ý nghĩa thống kê trong mẫu ($p = 0.0345$) nhưng không cải thiện dự báo ngoài mẫu?**  
   Trong kinh tế lượng và khoa học dữ liệu, ý nghĩa thống kê trong mẫu (In-sample statistical significance) phản ánh khả năng bác bỏ giả thuyết vô hiệu $H_0$ trên dữ liệu đã quan sát, nhưng **không đồng nghĩa với năng lực khái quát hóa ngoài mẫu (Out-of-sample predictive power)**. Thêm 6 biến ca từ làm tăng độ phức tạp mô hình. Khi áp dụng lên 410 bài hát mới, nhiễu ngẫu nhiên của các từ vựng bề mặt khiến mô hình bị phạt phương sai (variance penalty), làm $R^2$ Test giảm từ $4.70\%$ xuống $3.65\%$ và Adjusted $R^2$ rơi xuống mức âm ($-0.0236$).
2. **Kiểm định Bootstrap ghép cặp (Paired Cluster Bootstrap):**  
   Quy trình lấy mẫu lại $1,000$ lần có hoàn lại theo cụm bài hát của TV1 xác nhận mức chênh lệch sai số giữa Model 1 và Model 2 là $\Delta\text{MAE} = -0.15$ với khoảng tin cậy $95\%$ là $[-0.49; +0.19]$ (chứa giá trị 0). Do đó, nhóm **bác bỏ luận điểm rằng việc bổ sung ca từ cải thiện hiệu năng dự báo thứ hạng bài hát mới**.
3. **Ý nghĩa kinh tế và quản trị âm nhạc:**  
   Toàn bộ các đặc trưng âm học nội sinh và ca từ NLP chỉ giải thích được tối đa $3.6\% – 4.7\%$ biến thiên thứ hạng bài hát trên Spotify. Khoảng $95\%$ phương sai thành công thương mại còn lại được quyết định bởi các yếu tố ngoại sinh: danh tiếng nghệ sĩ (artist equity), quy mô ngân sách tiếp thị truyền thông, sự cộng hưởng của các trào lưu nhảy TikTok, và đặc biệt là vị trí sắp xếp trên các playlist biên tập định hướng của Spotify.

---

## 9. Giới Hạn Phương Pháp Luận & Khuyến Nghị Nghiên Cứu

1. **Thiên lệch chọn mẫu (Selection Bias / Survivorship Bias):**  
   Mọi quan sát đều là các bài hát đã lọt vào Spotify Top 200 Weekly. Nghiên cứu chỉ mô tả phân phối trong tập hợp các bài hát đã thành công, tuyệt đối không được ngoại suy thành mô hình dự báo bài hát bất kỳ chưa phát hành có trở thành hit hay không (ngụy biện virality).
2. **Nhiễu loạn phân tách token đa ngôn ngữ:**  
   Thuật toán tokenization dựa trên khoảng trắng tạo ra sự chênh lệch cơ học giữa âm tiết tiếng Việt, từ tiếng Anh và cụm *eojeol* tiếng Hàn. Các nghiên cứu tương lai cần áp dụng mô hình nhúng ngữ nghĩa tiền huấn luyện đa ngôn ngữ (Multilingual Sentence-BERT hoặc RoBERTa) để bóc tách tầng ngữ nghĩa sâu thay vì đếm surface tokens bề mặt.
3. **Cơ chế khuyết dữ liệu điệp khúc:**  
   Dữ liệu khuyết điệp khúc tập trung ở dòng nhạc rap Hoa Kỳ ($12.76\%$). Việc điền $0.0$ trong OLS là giải pháp kỹ thuật bắt buộc để bảo toàn cỡ mẫu, nhưng cần được ghi nhận là một giới hạn đo lường khi phân tích các thể loại âm nhạc có kết cấu xuyên suốt.
4. **Quan hệ tương quan, không mang hàm ý nhân quả:**  
   Mọi hệ số hồi quy chỉ phản ánh mối tương quan thống kê có điều kiện, không đo lường tác động can thiệp. Một nhạc sĩ không thể tăng thứ hạng bảng xếp hạng chỉ bằng cách viết thêm từ ngữ hay chuyển điệu thức bài hát sang Điệu Trưởng.

---

## 10. Tài Liệu Tham Khảo (Supported References)

1. **Cameron, A. C., & Miller, D. L. (2015).** A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.
2. **Holm, S. (1979).** A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
3. **Cochran, W. G. (1954).** Some methods for strengthening the common $\chi^2$ tests. *Biometrics*, 10(4), 417–451.
4. **Cramér, H. (1946).** *Mathematical Methods of Statistics*. Princeton University Press.
