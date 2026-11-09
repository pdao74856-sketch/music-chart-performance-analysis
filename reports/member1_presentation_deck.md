# THUYẾT TRÌNH — THÀNH VIÊN 1 (MODELING & ANALYSIS LEAD)
**Đề tài:** What Makes a Hit Song? — Spotify VN / US / KR, 2021–2026
**Môn:** ADY201m — AI, Data Science with Python & SQL
**Khung giờ:** 11:30 – 18:30 (7 phút, 4 slide). Phạm vi: chỉ phần TV1 đã làm theo CP1–CP3 và file test `reports/tests/01_tv1_ols_econometric_tests.md`.
**Bám khung:** Data Science Methodology (Data Preparation → Modeling → Evaluation → Deployment) và thuật ngữ bài 6.7–6.8 Regression Analysis: biến phụ thuộc Y, biến giải thích X, hằng số β₀, hệ số (weight) β, error term ε, t statistic, p-value < 0,05, bác bỏ null hypothesis, statistically significant, dummy variable, statsmodels/OLS, R-squared.
**Ngôn ngữ:** chỉ nói "liên quan / có tương quan", không nói nhân quả.

---

## PHẦN 1. NỘI DUNG 4 SLIDE

### SLIDE 1 — DATA PREPARATION: CHIA TRAIN/TEST KHÔNG RÒ RỈ, CHUẨN HÓA, KIỂM TRA ĐA CỘNG TUYẾN (11:30–13:00, 1′30″)
- **Chia dữ liệu theo bài hát:** `GroupShuffleSplit` theo `spotify_track_id`, seed 43
  - Train: 1.685 dòng / 1.527 bài. Test: 410 dòng / 382 bài (≈ 80/20)
  - Một bài có thể xuất hiện ở tối đa 3 thị trường; riêng Train có 144 bài như vậy (130 bài ở 2 thị trường, 14 bài ở 3)
- **Chuẩn hóa Z-score** (`StandardScaler`, fit trên Train): z = (x − μ) / σ; biến nhị phân `mode` giữ 0/1
- **Kiểm tra đa cộng tuyến (VIF) trên 24 biến chính của Model 3:** cao nhất 3,77 (`word_count`) < 5
  - Dùng `energy` thay `loudness_proxy` (loudness = 20·log₁₀(energy), hàm tất định)
  - Loại `spectral_bandwidth` (r = 0,909 với `spectral_centroid`)

### SLIDE 2 — MODELING: CHUỖI MÔ HÌNH HỒI QUY TUYẾN TÍNH BỘI LỒNG NHAU + SAI SỐ CHUẨN CỤM (13:00–14:45, 1′45″)
- **Biến phụ thuộc (Y):** ChartScore = 101 − LifetimeAverageRank (liên tục, [−99, 100]); các biến giải thích (X): âm học, ca từ, bối cảnh
- **4 bậc mô hình** (`statsmodels` OLS):
  - Model 1: Audio (16 hệ số) → Model 2: + Lyrics (22) → Model 2b: + Context = `release_year` + 2 biến giả (dummy) thị trường (25) → Model 3: + 10 số hạng tương tác Feature × Market (35)
- **Kiểm tra error term (Breusch–Pagan, Train N = 1.685):**
  - LM = 84,07, χ²(34), p = 3,99 × 10⁻⁶; F(34; 1650) = 2,55, p = 2,73 × 10⁻⁶ → bác bỏ null hypothesis phương sai sai số đồng đều
- **Quyết định:** dùng **sai số chuẩn vững theo cụm** (cluster-robust SE, cụm = `spotify_track_id`; Cameron & Miller, 2015) cho cả Model 1–3
  - Lý do: phương sai không đồng đều + các dòng của cùng một bài có sai số tương quan → SE thường bị đánh giá thấp, p-value dễ "có ý nghĩa" giả

### SLIDE 3 — EVALUATION: ĐÁNH GIÁ NGOÀI MẪU, KIỂM ĐỊNH WALD THEO KHỐI, ĐỘ DỐC `energy` THEO THỊ TRƯỜNG (14:45–17:00, 2′15″)
- **Hiệu năng trên tập Test (N = 410), baseline = trung bình Train theo thị trường (R² = 0,0018, MAE 30,27):**

| Mô hình | R² Test | MAE Test |
|---|---|---|
| Model 1 (Audio) | −0,0013 | 30,63 |
| Model 2 (+ Lyrics) | −0,0016 | 30,79 |
| Model 2b (+ Context) | 0,0365 [−0,0151; 0,0822] | 30,06 |
| Model 3 (+ Interactions) | 0,0340 [−0,0178; 0,0824] | 29,78 |

- **So sánh ghép cặp (bootstrap theo cụm bài hát):**
  - M1 → M2: ΔMAE −0,15 [−0,49; +0,19]
  - **M2 → M2b: ΔMAE +0,73 [+0,13; +1,38]; ΔR² +0,038 [+0,001; +0,078]** — bước duy nhất có khoảng tin cậy không chứa 0 (sát biên, chưa hiệu chỉnh đa so sánh)
  - M2b → M3: ΔMAE +0,28 [−0,10; +0,63] → chứa 0
  - (ΔMAE > 0: mô hình sau có MAE thấp hơn)
- **Joint Wald test theo khối (Model 3, SE cụm, hiệu chỉnh Holm):**
  - Audio: χ²(15) = 39,84, p = 0,0005 (Holm 0,0014) · Context: χ²(3) = 39,27, p = 1,5 × 10⁻⁸ (Holm 6,1 × 10⁻⁸)
  - Lyrics: χ²(6) = 11,74, p = 0,068 (Holm 0,113) · Interactions: χ²(10) = 17,91, p = 0,057 (Holm 0,113)
- **Hệ số góc của `energy` theo thị trường** (ChartScore thay đổi theo 1 độ lệch chuẩn của `energy`):
  - US −4,62 [−8,43; −0,81], p = 0,017 · VN +0,07, p = 0,976 · KR −0,93, p = 0,660
  - Kiểm định chung 2 bậc tự do cho `energy` × Market: p = 0,148 (Holm 0,593) → chỉ là gợi ý

### SLIDE 4 — CHẨN ĐOÁN, GIỚI HẠN VÀ DEPLOYMENT (TAB 4) (17:00–18:30, 1′30″)
- **Ramsey RESET** (ŷ² và ŷ³, Train): F(2; 1648) = 12,37, p = 4,67 × 10⁻⁶; Wald cụm χ²(2) = 9,22, p = 0,00996 → gợi ý dạng hàm tuyến tính chưa đủ (phi tuyến hoặc thiếu biến)
- **Độ nhạy:** Cook's D tối đa 0,0152; 95/1.685 quan sát vượt ngưỡng 4/n (5,6%); bỏ các bài trước 1980 thì p của khối tương tác 0,057 → 0,194
- **Giới hạn:** R² Test chỉ 3,4% (R² Train 6,6%); `release_year` có thể phản ánh thời gian xuất hiện lâu trên bảng xếp hạng; kết quả là **tương quan, không phải nhân quả**
- **Deployment:** Tab 4 trên web app đọc trực tiếp các kết quả trên từ file trong repo (không tính lại, không nhập tay)

---

## PHẦN 2. KỊCH BẢN LỜI NÓI (≈ 7 PHÚT)

> **[11:30 – 13:00] SLIDE 1 — DATA PREPARATION**
> "Em là Thành viên 1, phụ trách phần mô hình hóa. Theo Data Science Methodology, em bắt đầu từ bước Data Preparation, vì đây là bước quyết định kết quả đánh giá có trung thực hay không.
> Thứ nhất là cách chia dữ liệu. Một bài hát có thể xuất hiện ở tối đa ba thị trường, nên nếu chia theo từng dòng thì cùng một bài sẽ nằm ở cả Train và Test, gây rò rỉ dữ liệu. Em chia theo `spotify_track_id`, hạt giống 43: Train có 1.685 dòng ứng với 1.527 bài, Test có 410 dòng ứng với 382 bài, khoảng 80/20. Riêng Train có 144 bài xuất hiện ở nhiều thị trường.
> Thứ hai, em chuẩn hóa Z-score, tức là trừ trung bình và chia độ lệch chuẩn, với tham số chỉ tính trên Train.
> Thứ ba là kiểm tra đa cộng tuyến bằng VIF. Với 24 biến chính của Model 3, VIF lớn nhất là 3,77, thấp hơn ngưỡng 5. Để đạt được điều này, em dùng `energy` thay cho `loudness_proxy`, vì loudness chính là 20 log10 của energy, và loại `spectral_bandwidth` vì tương quan 0,909 với `spectral_centroid`."
>
> **[13:00 – 14:45] SLIDE 2 — MODELING**
> "Sang bước Modeling, biến phụ thuộc Y là ChartScore bằng 101 trừ thứ hạng trung bình, là biến liên tục, và các biến giải thích X là đặc trưng âm học, ca từ và bối cảnh. Em dùng thư viện statsmodels để xây bốn bậc hồi quy tuyến tính bội lồng nhau: Model 1 chỉ có âm học, Model 2 thêm ca từ, Model 2b thêm bối cảnh gồm năm phát hành và biến giả, tức dummy, cho thị trường, và Model 3 thêm mười số hạng tương tác giữa đặc trưng và thị trường. Cách lồng nhau này giúp tách xem mỗi nhóm biến đóng góp thêm bao nhiêu.
> Trước khi tin vào p-value, em kiểm tra error term, tức phần mô hình không giải thích được, bằng Breusch–Pagan: LM bằng 84,07 với bậc tự do 34, p cỡ 4 nhân 10 mũ trừ 6. Em bác bỏ null hypothesis rằng phương sai sai số đồng đều. Thêm nữa, các dòng của cùng một bài hát có sai số tương quan với nhau. Vì vậy em dùng sai số chuẩn vững theo cụm bài hát cho cả ba mô hình. Nếu dùng sai số thường, sai số chuẩn bị đánh giá thấp và p-value dễ cho kết luận có ý nghĩa một cách giả tạo."
>
> **[14:45 – 17:00] SLIDE 3 — EVALUATION**
> "Bước Evaluation, em đánh giá trên tập Test mà mô hình chưa thấy. Mốc so sánh là baseline, dự đoán bằng trung bình Train theo thị trường, với R² bằng 0,0018.
> Model 1 và Model 2 có R² âm nhẹ, nghĩa là dự đoán không tốt hơn việc đoán trung bình. Khi thêm bối cảnh ở Model 2b, R-squared lên 0,0365 và MAE giảm từ 30,79 xuống 30,06. Em kiểm tra bằng bootstrap theo cụm bài hát: bước từ Model 2 lên 2b là bước duy nhất mà khoảng tin cậy 95% của mức giảm MAE, từ 0,13 đến 1,38, không chứa 0. Nhưng đây là bước sát biên và em chưa hiệu chỉnh cho nhiều phép so sánh. Các bước còn lại đều có khoảng tin cậy chứa 0, kể cả khi thêm tương tác ở Model 3.
> Em cũng dùng kiểm định Wald đồng thời cho từng khối biến. Khối Âm học và khối Bối cảnh có ý nghĩa thống kê, statistically significant, ngay cả sau hiệu chỉnh Holm. Khối Ca từ có p bằng 0,068 và khối Tương tác có p bằng 0,057, chưa đạt mức 5%.
> Cuối cùng là hệ số góc của `energy` theo thị trường. Ở Mỹ, hệ số hồi quy, tức weight của `energy`, là âm 4,62: `energy` cao hơn một độ lệch chuẩn đi kèm ChartScore thấp hơn khoảng 4,6 điểm. Ở Việt Nam hệ số gần 0, ở Hàn Quốc là âm 0,93, đều không có ý nghĩa thống kê, vì p-value lớn hơn 0,05. Kiểm định chung cho `energy` nhân thị trường có p bằng 0,148, nên em chỉ xem đây là gợi ý, không phải bằng chứng chắc chắn về khác biệt giữa các thị trường."
>
> **[17:00 – 18:30] SLIDE 4 — DIAGNOSTICS, GIỚI HẠN, DEPLOYMENT**
> "Về chẩn đoán, kiểm định Ramsey RESET cho F bằng 12,37 và p cỡ 5 nhân 10 mũ trừ 6; dùng sai số cụm thì p bằng 0,00996. Kết quả này gợi ý dạng hàm tuyến tính chưa đủ, có thể do quan hệ phi tuyến hoặc thiếu biến, nhưng bản thân kiểm định không cho biết nguyên nhân. Em cũng kiểm tra độ nhạy: khoảng cách Cook lớn nhất chỉ 0,0152, và nếu bỏ các bài trước 1980 thì p của khối tương tác tăng từ 0,057 lên 0,194.
> Về giới hạn, R² Test chỉ 3,4%, mô hình giải thích được rất ít biến thiên của ChartScore. Biến năm phát hành có thể phản ánh việc bài hát đã ở trên bảng xếp hạng lâu, chứ không hẳn là đặc tính âm nhạc. Mọi kết quả của em là tương quan, không phải nhân quả.
> Ở bước Deployment, toàn bộ kết quả kiểm định này được đưa lên Tab 4 của web app, đọc trực tiếp từ file trong repo nên số liệu trên app và trên slide luôn trùng nhau. Em xin chuyển sang bạn tiếp theo. Em cảm ơn Thầy Cô."

---

## PHẦN 3. Q&A (CHỈ PHẦN TV1)

**Q1. Vì sao R² Test thấp, thậm chí âm ở Model 1 và 2?**
R² Test âm nghĩa là mô hình dự đoán bài hát mới kém hơn việc dùng trung bình. Điều này cho thấy đặc trưng âm học và ca từ, riêng chúng, liên quan rất ít đến ChartScore của bài chưa từng thấy. Chúng em báo cáo đúng như vậy, và không dùng R² Train (6,6%) để thay cho kết quả Test.

**Q2. Vì sao dùng sai số chuẩn vững theo cụm mà không dùng sai số OLS thường?**
Breusch–Pagan bác bỏ phương sai đồng đều (p ≈ 4 × 10⁻⁶), và cùng một bài hát có thể có tới 3 dòng, sai số của chúng tương quan. Khi đó SE thường bị thấp, t-statistic bị thổi phồng. Cụm theo `spotify_track_id` xử lý cả hai vấn đề.

**Q3. Vì sao dùng `energy` mà không dùng `loudness_proxy`?**
`loudness_proxy` = 20·log₁₀(energy), là hàm tất định của `energy`. Đưa cả hai vào hồi quy tuyến tính gây đa cộng tuyến, VIF vượt xa 5. Hồi quy tuyến tính cần ma trận biến độc lập không bị suy biến nên em chỉ giữ `energy`.

**Q4. Ramsey RESET có p nhỏ, có nghĩa là quan hệ chắc chắn phi tuyến không?**
Không. RESET chỉ gợi ý dạng hàm tuyến tính chưa đủ; nguyên nhân có thể là phi tuyến hoặc thiếu biến. Vì vậy em nêu đây là giới hạn của mô hình, không kết luận thêm.

**Q5. Có thể nói âm nhạc "khiến" bài hát lên chart không?**
Không. Đây là phân tích tương quan trên dữ liệu quan sát. Hệ số cho biết mức liên quan sau khi kiểm soát các biến còn lại, không đo tác động can thiệp, và mô hình không có biến như nghệ sĩ hay ngân sách quảng bá.

**Q6. Vì sao bước M2 → M2b "có ý nghĩa" mà em vẫn thận trọng?**
Khoảng tin cậy của ΔMAE [+0,13; +1,38] và ΔR² [+0,001; +0,078] chỉ vừa qua 0, chưa hiệu chỉnh đa so sánh, và R² Test của Model 2b vẫn có khoảng tin cậy chứa 0. Hơn nữa `release_year` có thể là biến phản ánh thời gian xuất hiện trên bảng xếp hạng.

---

## PHẦN 4. THAY ĐỔI SO VỚI BẢN ĐÃ COMMIT (`9a2fb0c`)
1. **Sửa số sai:** "144 bài, 102 bài ở 2 nước, 42 bài ở 3 nước, 330 dòng" → Train: 144 bài (130 ở 2 thị trường, 14 ở 3).
2. **Sửa lỗi nặng ở Slide 2 cũ:** bản cũ gán khoảng tin cậy của R² Model 2b cho ΔR² và kết luận "chứa 0". Số đúng: ΔR² +0,038 [+0,001; +0,078], ΔMAE +0,73 [+0,13; +1,38], không chứa 0.
3. Sửa r(centroid, bandwidth) 0,897 → 0,909.
4. Bỏ các cụm đã nhân quả/phóng đại: "khán giả Mỹ ưa chuộng…", "phân kỳ văn hóa rõ nét… khẳng định", "chi phối bởi viral marketing, nghệ sĩ danh tiếng" (không đo được).
5. Sửa tên môn: "Applied Data Analytics" → "AI, Data Science with Python & SQL".
6. Sắp xếp lại theo Data Science Methodology và bỏ phần RF, Tab khác (chỉ giữ 1 câu chuyển giao).
7. Thêm kết quả bổ sung: baseline, Cook's D, độ nhạy pre-1980, kiểm định chung energy × Market (p = 0,148).
