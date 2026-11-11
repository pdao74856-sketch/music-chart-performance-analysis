# THUYẾT TRÌNH — THÀNH VIÊN 3 (LYRICS & TEXT NLP LEAD)
**Đề tài:** What Makes a Hit Song? — So sánh Yếu tố Ảnh hưởng đến Độ Phổ biến Âm nhạc giữa Việt Nam, Hoa Kỳ và Hàn Quốc (Spotify Top 200, 2021–2026)  
**Môn học:** ADY201m — AI, Data Science with Python & SQL  
**Thời lượng phân bổ:** ~2 phút (2 slide cô đọng). Phạm vi: Phân hệ Ca từ NLP, Điệu thức Âm học, Kiểm toán Đa cộng tuyến VIF và Mô hình Hồi quy Model 2 OLS.  
**Nguyên tắc học thuật:** Bám sát chuẩn mực Data Science Methodology; sử dụng số liệu thực nghiệm đã xác minh từ bộ đệm canonical ($N = 2,098$); tuyệt đối không ngụy biện rằng ca từ giúp tăng độ chính xác dự báo ngoài mẫu khi số liệu Test không hỗ trợ; chỉ khẳng định quan hệ tương quan, không khẳng định nhân quả.

---

## PHẦN 1. NỘI DUNG 2 SLIDE TRÌNH BÀY

### SLIDE 1 — KHAI PHÁ DỮ LIỆU CA TỪ NLP & DỊ BIỆT THỊ TRƯỜNG (0:00 – 1:00, 60 giây)
* **Quy mô thu thập dữ liệu ca từ:**
  * Thu thập thành công $1,909 / 1,912$ ca khúc độc nhất ($99.84\%$ độ phủ qua LRCLIB, Genius và nhập thủ công; 3 bài instrumental không lời được loại bỏ chuẩn tắc).
  * Trích xuất các thuộc tính hình thái học, cấu trúc bài hát và độ đa dạng từ vựng.
* **Sàng lọc Đa cộng tuyến (VIF) trước khi mô hình hóa (Train $N = 1,685$):**
  * Phát hiện và triệt tiêu **đồng nhất thức đại số tuyến tính**: $\text{repetition\_ratio} \equiv 1 - \text{lexical\_diversity}$ ($\text{VIF} \approx 1.0008 \times 10^{15}$) $\to$ Loại bỏ `lexical_diversity`, giữ `repetition_ratio`.
  * Triệt tiêu **cộng tuyến tích số**: $\text{unique\_word} = \text{word} \times (1 - \text{repetition})$ ($\text{VIF} > 14.8$) $\to$ Loại bỏ `unique_word_count`, giữ `word_count`.
  * Khối 6 biến ca từ của Model 2 đều an toàn tuyệt đối: $\text{VIF} \le 3.77 < 5.0$. Biến đổi $\ln(1 + \text{word\_count})$ nén $34.3\%$ đòn bẩy $h_{ii}$ của các bản rap cực đoan.
* **Kiểm định Kruskal-Wallis & Chi-Square trên mẫu độc quyền thị trường ($N = 1,743$):**

| Đặc trưng kiểm định | Thống kê kiểm định | $p$-value sau hiệu chỉnh Holm | Cỡ hiệu ứng | Diễn giải thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| `avg_word_length` | $H = 557.07$ ($df=2$) | **$6.50 \times 10^{-121}$** | $\varepsilon^2 = 0.3196$ (Rất lớn) | Bản chất tokenization: US từ đa âm tiết ($3.85$) $>$ VN âm tiết đơn ($3.32$) $\approx$ KR Hangul ($3.29$). |
| `word_count` | $H = 269.88$ ($df=2$) | **$1.24 \times 10^{-58}$** | $\varepsilon^2 = 0.1542$ (Lớn) | Mật độ tự sự: VN lời dài nhất ($421.0\text{ từ}$) $>$ US ($356.0$) $>$ KR ($286.0\text{ eojeols}$). |
| `chorus_repetition_count` | $H = 2.37$ ($df=2$) | **$0.3060$** (Không ý nghĩa) | $\varepsilon^2 = 0.0002$ (Không đáng kể) | **Quy chuẩn Pop toàn cầu**: Cả 3 nước đồng nhất trung vị $= 1.0$ lần lặp điệp khúc. |
| `audio_mode` (Điệu thức) | $\chi^2 = 24.31$ ($df=2$) | **$5.25 \times 10^{-6}$** | Cramér's V $= 0.1181$ (Nhỏ) | VN chuộng Điệu Trưởng nhất ($68.85\%$) $>$ KR ($58.24\%$) $>$ US ($54.92\%$). |

---

### SLIDE 2 — MÔ HÌNH HÓA MODEL 2 OLS, KIỂM ĐỊNH WALD ĐỒNG THỜI & THỰC TẾ NGOÀI MẪU (1:00 – 2:00, 60 giây)
* **Đặc tả Mô hình Model 2 OLS (Audio + Lyrics + Controls):**
  * Ước lượng $\text{ChartScore} = 101 - \text{lifetime\_average\_rank}$ trên tập Train ($N = 1,685$ dòng / $1,527$ cụm bài).
  * 24 biến độc lập: 15 biến âm học (14 biến Librosa + binary `mode`) + 3 biến bối cảnh (`release_year_std`, `Market_VN`, `Market_KR`) + 6 biến ca từ chuẩn hóa Z-score.
  * Toàn bộ suy luận sử dụng **sai số chuẩn vững theo cụm** (Cluster-Robust SE theo `spotify_track_id`) do TV1 chuyển giao.
* **Kiểm định Wald Đồng thời Khối Ca từ (Joint Wald Block Test):**
  * $H_0: \beta_{\text{word}} = \beta_{\text{rep}} = \beta_{\text{len}} = \beta_{\text{line}} = \beta_{\text{chorus}} = \beta_{\text{punc}} = 0$.
  * Trong Model 2 Canonical: **$\text{Wald }\chi^2(6) = 13.5932, p = 0.0345 < 0.05$** $\to$ **Bác bỏ $H_0$**.
  * Trong Model 2 lồng nhau (Audio + Lyrics không controls): **$\text{Wald }\chi^2(6) = 15.5722, p = 0.0162 < 0.05$**.
  * *Bản chất thống kê:* Dù không có hệ số ca từ nào đạt $p < 0.05$ riêng lẻ do chia sẻ hiệp phương sai (song length, verse count, hook repetition), véc-tơ 6 biến ca từ xét tổng thể đóng góp thông tin giải thích có ý nghĩa thống kê trong mẫu Train.
* **Thực tế Đánh giá Ngoài mẫu (Test $N = 410$ bài hát chưa từng thấy):**

| Chỉ số đánh giá | Model 1 (Audio + Controls) | Model 2 (Audio + Lyrics + Controls) | Chênh lệch ($\Delta = \text{M2} - \text{M1}$) | Kết luận khoa học |
| :--- | :---: | :---: | :---: | :--- |
| **$R^2$ Test** | **$0.0470$** | **$0.0365$** | **$-0.0105$** | Ca từ **không cải thiện** khả năng giải thích phương sai ngoài mẫu. |
| **Adjusted $R^2$ Test** | $+0.0031$ | $-0.0236$ | $-0.0267$ | Bị phạt nặng do bổ sung 6 bậc tự do không đem lại sức mạnh dự báo. |
| **MAE Test** | **$29.74$** | **$30.06$** | **$+0.32$** (Tăng sai số) | Bootstrap ghép cặp xác nhận $\Delta\text{MAE} = -0.15$ $[-0.49; +0.19]$ (chứa 0). |
| **RMSE Test** | **$38.05$** | **$38.26$** | **$+0.21$** (Tăng sai số) | Sai số dự báo tăng nhẹ trên các bài hát mới. |

* **Thông điệp kết luận & Đóng góp ứng dụng:**
  1. *Trung thực học thuật:* Ca từ có sự phân hóa văn hóa sâu sắc và có ý nghĩa thống kê trong mẫu, nhưng **không giúp dự báo thứ hạng bài hát mới**. Thứ hạng thương mại thực tế bị chi phối bởi danh tiếng nghệ sĩ, chiến dịch marketing và phân phối playlist.
  2. *Triển khai thực tế:* Kết quả đã được đóng gói thành công vào **Tab 2 What-If Simulator** (cung cấp 2 thanh trượt ca từ `word_count`, `repetition_ratio`) và bộ dữ liệu trung vị dự phòng `lyrics_market_medians.json` phục vụ Tab 3 Audio Upload.

---

## PHẦN 2. KỊCH BẢN LỜI NÓI THUYẾT TRÌNH (BÁM SÁT 2 PHÚT)

> **[0:00 – 1:00] SLIDE 1 — KHAI PHÁ DỮ LIỆU CA TỪ NLP & DỊ BIỆT THỊ TRƯỜNG**  
> *"Kính thưa Thầy Cô và các bạn, em là Thành viên 3, phụ trách Phân hệ Ca từ NLP và Mô hình Hồi quy Model 2.*  
> *Từ 1,912 bài hát trên Spotify Top 200, nhóm đã thu thập thành công ca từ cho 1,909 bài, đạt độ phủ 99.8%. Trước khi đưa vào mô hình, em thực hiện kiểm toán đa cộng tuyến VIF trên tập Train và phát hiện hai điểm nghẽn nghiêm trọng: tỷ lệ lặp lại ca từ là nghịch đảo đại số hoàn hảo của độ đa dạng từ vựng, đẩy VIF lên 1 triệu tỷ; đồng thời số từ duy nhất cộng tuyến tích số với tổng số từ với VIF trên 14.8. Em đã kiên quyết loại bỏ hai biến này, đưa toàn bộ 6 biến ca từ của Model 2 về mức VIF an toàn dưới 3.77.*  
> *Khi kiểm định phi tham số Kruskal-Wallis trên 1,743 bài hát độc quyền thị trường, 5 trên 6 đặc trưng ca từ có phân phối khác biệt sâu sắc giữa ba quốc gia với p nhỏ hơn 10 mũ trừ 29 sau hiệu chỉnh Holm. Tuy nhiên, sự phân hóa mạnh nhất ở độ dài từ thực chất phản ánh đặc thù phân tách token: tiếng Anh là từ đa âm tiết, tiếng Việt là âm tiết đơn, còn tiếng Hàn là cụm eojeol. Đáng chú ý, số lần lặp điệp khúc hoàn toàn không có sự khác biệt thống kê với p bằng 0.306, cả ba quốc gia đều có trung vị đúng bằng 1 lần lặp, khẳng định đây là cấu trúc công nghiệp chung của nhạc Pop toàn cầu.*  
> *Về mặt điệu thức, kiểm định Chi-square xác nhận Việt Nam chuộng Điệu Trưởng nhất với gần 69%, so với 55% ở Hoa Kỳ và 58% ở Hàn Quốc."*  

> **[1:00 – 2:00] SLIDE 2 — MÔ HÌNH HÓA MODEL 2, KIỂM ĐỊNH WALD & THỰC TẾ NGOÀI MẪU**  
> *"Sang bước Mô hình hóa, em kế thừa 18 biến âm học và bối cảnh từ Model 1 của bạn Thành viên 1, kết hợp cùng 6 biến ca từ chuẩn hóa Z-score để xây dựng Model 2 gồm 24 biến, ước lượng bằng sai số chuẩn vững theo cụm bài hát.*  
> *Về mặt suy luận thống kê trong mẫu Train, kiểm định Wald đồng thời cho khối 6 biến ca từ đạt Chi-square 13.59 với p bằng 0.0345, bác bỏ giả thuyết H0. Mặc dù từng biến ca từ riêng lẻ không có p dưới 0.05 do chia sẻ phương sai về độ dài bài hát và mật độ lặp, nhưng xét toàn bộ khối, ca từ đóng góp thông tin giải thích có ý nghĩa thống kê.*  
> *Tuy nhiên, trên tập Test ngoài mẫu gồm 410 bài hát chưa từng thấy, kết quả lại rất thực tế: R-squared của Model 2 giảm từ 4.7% xuống 3.6%, sai số MAE tăng từ 29.74 lên 30.06 điểm. Kiểm định Bootstrap theo cụm của bạn Thành viên 1 xác nhận mức chênh lệch sai số có khoảng tin cậy chứa 0. Điều này chứng minh rằng ca từ không giúp cải thiện khả năng dự báo thứ hạng bài hát mới.*  
> *Nhóm em báo cáo trung thực kết quả này: thành công thương mại trên bảng xếp hạng phụ thuộc phần lớn vào các yếu tố ngoại sinh như độ nổi tiếng của nghệ sĩ hay chiến dịch marketing, thay vì chỉ dựa vào ca từ hay âm thanh.*  
> *Cuối cùng, em đã đóng gói các thuộc tính ca từ vào Tab 2 Simulator và tệp giá trị trung vị dự phòng cho Tab 3 của ứng dụng Web. Em xin trân trọng cảm ơn Thầy Cô."*

---

## PHẦN 3. BỘ CÂU HỎI & TRẢ LỜI BẢO VỆ CHUYÊN SÂU (Q&A CHO TV3)

**Q1. Vì sao kiểm định Wald đồng thời khối ca từ có ý nghĩa ($p = 0.0345$) nhưng không có hệ số ca từ nào đạt $p < 0.05$ trong Model 2?**  
*Trả lời:* Trong Model 2, các biến ca từ như tổng số từ (`word_count_std`), số dòng (`line_count_std`) và độ lặp lại (`repetition_ratio_std`) có mối tương quan hiệp phương sai nội bộ vừa phải ($\text{VIF} \in [1.77, 3.77]$). Khi kiểm định $t$ đơn lẻ, phương sai giải thích bị phân mảnh cho từng biến khiến sai số chuẩn riêng lẻ tăng lên, làm $p$-value riêng lẻ nằm ở mức $0.06$ đến $0.38$. Tuy nhiên, kiểm định Wald đánh giá dạng toàn phương $\hat{\beta}^\top \hat{\Sigma}^{-1} \hat{\beta}$ trên toàn bộ véc-tơ hệ số ca từ cùng lúc. Kết quả $p = 0.0345$ khẳng định khối ca từ xét tổng thể có liên quan đến ChartScore trong tập huấn luyện.

**Q2. Tại sao kiểm định Wald khối ca từ trong Model 2 Canonical có $p = 0.0345$, nhưng sang Model 3 của TV1 lại thành $p = 0.068$?**  
*Trả lời:* Trong Model 2, hệ số ca từ ước lượng độ dốc gộp (pooled slope) trên toàn bộ $1,685$ quan sát. Sang Model 3, do TV1 đưa thêm các biến tương tác thị trường cho 3 biến ca từ, hệ số chính chỉ còn đại diện cho độ dốc riêng của thị trường tham chiếu Hoa Kỳ ($n \approx 640$). Cỡ mẫu hiệu dụng giảm làm sai số chuẩn tăng lên, khiến giá trị Wald Chi-square giảm từ $13.59$ xuống $11.74$ ($p = 0.068$, sau hiệu chỉnh Holm là $0.113$).

**Q3. Tại sao R² Test của Model 2 lại thấp hơn Model 1 dù thêm nhiều biến hơn?**  
*Trả lời:* Hiện tượng này phản ánh chi phí phương sai (variance cost) trong học máy và kinh tế lượng: Bổ sung 6 biến ca từ làm tăng độ phức tạp của mô hình. Vì tín hiệu ca từ bề mặt trên bài hát mới rất yếu và chứa nhiều nhiễu ngẫu nhiên, mô hình bị overfitting nhẹ trên tập Train, dẫn tới sai số ngoài mẫu trên tập Test tăng nhẹ ($+0.32$ điểm MAE) và $R^2$ giảm. Đây là bằng chứng thực nghiệm quan trọng cho thấy ca từ thuần túy không quyết định thứ hạng bài hát.

**Q4. Vì sao lại loại bỏ `lexical_diversity` và `unique_word_count` khỏi mô hình OLS?**  
*Trả lời:* `lexical_diversity` là Type-Token Ratio ($\text{unique} / \text{word}$), và tỷ lệ lặp lại $\text{repetition\_ratio} = 1 - \text{TTR}$. Sau khi chuẩn hóa Z-score, hai biến này đồng nhất đại số hoàn toàn ($\text{Z}_{\text{rep}} \equiv -\text{Z}_{\text{div}}$), làm ma trận $(X^\top X)$ bị suy biến với VIF xấp xỉ 1 triệu tỷ. Mặt khác, $\text{unique\_word\_count} = \text{word\_count} \times (1 - \text{repetition\_ratio})$, tạo ra cộng tuyến tích số với VIF $> 14.8$. Em đã loại bỏ cả hai để bảo vệ tính khả nghịch của mô hình OLS.

**Q5. Tại sao sự khác biệt về `avg_word_length` lại có cỡ hiệu ứng lớn nhất ($\varepsilon^2 = 0.320$) nhưng lại bị coi là "nhiễu ngôn ngữ học" (linguistic artifact)?**  
*Trả lời:* Thuật toán tokenization tách từ bằng khoảng trắng. Tiếng Việt là ngôn ngữ đơn lập, mỗi âm tiết (*tiếng*) là một token ngắn (2–5 chữ cái). Tiếng Anh là ngôn ngữ biến hình, có nhiều từ đa âm tiết ghép dài. Tiếng Hàn phân tách theo cụm ngữ pháp *eojeol*. Sự khác biệt cực lớn này đo lường quy ước chữ viết của ngôn ngữ chứ không đo độ phức tạp học thuật hay chất lượng nghệ thuật của bài hát.
