# BÁO CÁO THUYẾT TRÌNH FINAL — PHÂN HỆ THÀNH VIÊN 1 (MODELING & ECONOMETRIC LEAD)
**Dự án:** What Makes a Hit Song? — Phân Tích Hiệu Suất Bảng Xếp Hạng Spotify Việt Nam, Hoa Kỳ & Hàn Quốc (2021–2026)  
**Môn học:** ADY201m — Applied Data Analytics (Chuẩn IBM Data Science Methodology & Toolkit)  
**Thành viên:** Member 1 (Phụ trách: Xử lý Đa cộng tuyến VIF, Đặc tả Mô hình OLS Model 3, Kiểm định Chẩn đoán Kinh tế lượng, Đối chuẩn Ngoài mẫu Random Forest, Streamlit Tab 4 & Unit Tests)  
**Thời lượng trình bày:** 5 — 7 phút  

---

## PHẦN 1. NỘI DUNG 4 SLIDE TRÌNH CHIẾU (DECK SLIDES)
*(Dành cho Member 3 ráp trực tiếp vào file trình chiếu chung `reports/presentation_deck.pptx`)*

```mermaid
flowchart LR
    S1["Slide 1: VIF Audit & Ma trận OLS Model 3"] --> S2["Slide 2: Khuyết tật & Cluster-Robust SE (Cameron-Miller)"]
    S2 --> S3["Slide 3: Test N=410, Nested Bootstrap & RF"]
    S3 --> S4["Slide 4: Dashboard Tab 4 & Unit Test Suite"]
```

---

### 🖥️ SLIDE 1: KIỂM SOÁT ĐA CỘNG TUYẾN & THIẾT KẾ MA TRẬN OLS MODEL 3
* **Tiêu đề Slide:** **Kiểm Soát Đa Cộng Tuyến (VIF Audit) & Đặc Tả Ma Trận Hồi Quy Model 3**
* **Bố cục hiển thị:**
  1. **Kiểm toán đa cộng tuyến thực tế (Multicollinearity & VIF Audit):**
     * Phát hiện cặp biến âm học Librosa có tương quan Pearson thực nghiệm rất cao: `spectral_centroid` và `spectral_bandwidth` ($r = 0.897$).
     * *Quyết định phương pháp luận:* Loại bỏ `spectral_bandwidth`, giữ lại `spectral_centroid` (đại diện độ sáng âm thanh) và `energy_std` (năng lượng vật lý tín hiệu).
     * *Kết quả VIF:* Toàn bộ 24 biến chính của Model 3 trên tập Train ($N = 1,685$) đều có $\text{VIF} < 5.0$ (cao nhất là `word_count_std` $= 3.77$, `energy_std` $= 2.28$).
  2. **Đặc tả ma trận hồi quy Model 3 (Full OLS with Market Interactions):**
     * **Grain dữ liệu:** Mỗi dòng quan sát = 1 cặp `(spotify_track_id × market)` trên tập sạch $2,098$ dòng ($1,912$ bài hát độc bản).
     * **Cấu trúc 34 bậc tự do (DOF):**
       * Khối Âm học: $15$ biến đặc trưng DSP Librosa.
       * Khối Ca từ: $6$ biến xử lý ngôn ngữ tự nhiên NLP.
       * Khối Bối cảnh: $3$ biến (`release_year_std`, `Market_VN`, `Market_KR`).
       * Khối Tương tác thị trường: $10$ biến tương tác chéo giữa âm học/ca từ với biến giả quốc gia.

---

### 🖥️ SLIDE 2: CĂN CỨ CHO CLUSTER-ROBUST SE & KIỂM ĐỊNH CHẨN ĐOÁN OLS
* **Tiêu đề Slide:** **Kiểm Định Khuyết Tật Giả Định OLS & Căn Cứ Bắt Buộc Dùng Cluster-Robust SE**
* **Bố cục hiển thị:**
  1. **Khuyết tật phương sai sai số & Cấu trúc tương quan nội cụm:**
     * **Kiểm định Breusch–Pagan (Train $N=1,685$):**
       * Thống kê LM: $84.07$ ($p = 3.99 \times 10^{-6}$)
       * Thống kê F: $2.55$ ($p = 2.73 \times 10^{-6}$, $df = [34, 1650]$)
       * *Kết luận:* Bác bỏ triệt để giả định phương sai đồng nhất ($p < 0.001$), xác nhận phương sai sai số thay đổi (Heteroskedasticity).
     * **Cấu trúc dữ liệu lặp lại đa thị trường:**
       * Có **144 ca khúc xuất hiện đồng thời ở nhiều thị trường** (102 bài ở 2 nước, 42 bài ở cả 3 nước), chiếm **330 dòng quan sát**.
       * Gây hiện tượng tương quan nội cụm giữa các quan sát của cùng bài hát: $\text{Cov}(u_{it}, u_{is}) \ne 0$.
     * **Chuẩn hóa phương pháp luận bắt buộc:**
       * Nếu dùng OLS cổ điển hay White SE, sai số chuẩn sẽ bị đánh giá thấp (underestimated), thổi phồng giả tạo giá trị thống kê $t$.
       * Bắt buộc áp dụng **Sai số chuẩn vững theo cụm track (`spotify_track_id`) theo Cameron & Miller (2015)** cho cả Model 1, Model 2 và Model 3.
  2. **Kiểm định dạng hàm & Điểm ảnh hưởng:**
     * **Ramsey RESET Test:** $F(2, 1648) = 12.37$ ($p = 4.67 \times 10^{-6}$); Wald cụm $\chi^2 = 9.22$ ($p = 0.00996$) $\rightarrow$ Bác bỏ dạng hàm tuyến tính chuẩn xác, phát hiện dấu hiệu quan hệ phi tuyến.
     * **Cook's Distance:** Ngưỡng $4/n = 0.00237$, Max $= 0.0152$, có $95$ quan sát ($5.6\%$) vượt ngưỡng nhưng không có outlier đòn bẩy phá vỡ mô hình ($\text{Max} \ll 0.5$).
     * **Q-Q Residuals:** Phân phối phần dư có đuôi lệch do biến phụ thuộc $\text{ChartScore} \in [-99, 100]$ bị chặn hai đầu (censored/bounded); suy luận thống kê dựa trên SE cụm và bootstrap, không phụ thuộc giả định phần dư chuẩn.

---

### 🖥️ SLIDE 3: HIỆU NĂNG NGOÀI MẪU (TEST N=410), NESTED BOOTSTRAP & RANDOM FOREST
* **Tiêu đề Slide:** **Hiệu Năng Ngoài Mẫu, Kiểm Định Bootstrap Ghép Cặp & Đối Chuẩn Random Forest**
* **Bố cục hiển thị:**
  1. **Hiệu năng ngoài mẫu trên tập Test độc lập ($N = 410$, $382$ ca khúc độc bản):**
     * $R^2$ Test OLS Model 3 đạt $0.0344$, MAE $= 17.47$.
     * Random Forest Model 4 Benchmark đạt $R^2 = 0.0694$, MAE $= 16.82$.
     * Khoảng tin cậy 95% của $R^2$ ở các mô hình OLS đều bao hàm giá trị 0.
  2. **Kiểm định ghép cặp lồng nhau qua Cluster Bootstrap (Overall):**
     * $M1 \rightarrow M2$: $\Delta R^2 = -0.0013$ ($[-0.018, +0.016]$ chứa 0), $\Delta \text{MAE} = -0.01$ (chứa 0) $\rightarrow$ Khối ca từ NLP chưa tạo khác biệt ngoài mẫu.
     * $M2 \rightarrow M2b$: $\Delta R^2 = +0.0354$ ($[+0.001, +0.078]$ **không chứa 0**), $\Delta \text{MAE} = +0.61$ **không chứa 0** $\rightarrow$ Biến bối cảnh `release_year` tạo bước nhảy thực chất.
     * $M2b \rightarrow M3$: $\Delta R^2 = -0.0021$ ($[-0.025, +0.019]$ chứa 0), $\Delta \text{MAE} = -0.09$ (chứa 0) $\rightarrow$ 10 biến tương tác thị trường có dấu hiệu quá khớp (overfitting).
  3. **Đối chuẩn Random Forest vs OLS Model 3:**
     * $\Delta R^2 = +0.035$, 95% CI $[-0.038, +0.105]$ **chứa 0**.
     * $\Delta \text{MAE} = +0.65$, 95% CI $[-0.53, +1.09]$ **chứa 0**.
     * *Kết luận học thuật:* Chưa có bằng chứng thống kê ngoài mẫu về sự vượt trội của mô hình học máy phi tuyến phức tạp so với mô hình OLS cô đọng (nguyên tắc parsimony).

---

### 🖥️ SLIDE 4: PHÂN HỆ TAB 4 STREAMLIT & BỘ KIỂM THỬ TỰ ĐỘNG (UNIT TEST SUITE)
* **Tiêu đề Slide:** **Dashboard Chẩn Đoán Tab 4 & Quy Trình Kiểm Thử Tự Động (Unit Tests)**
* **Bố cục hiển thị:**
  1. **Phân hệ Streamlit Tab 4 (`app/components/tab_diagnostics.py`):**
     * Kiến trúc Offline-first: Đọc số liệu 100% động từ repository qua `@st.cache_data`, không phụ thuộc Supabase runtime, không hardcode số liệu thống kê.
     * Logic 3 trạng thái (`True / False / None`) xử lý an toàn missing data trong khoảng tin cậy CI.
     * 5 khối chức năng hoàn chỉnh: Bảng đối chuẩn ngoài mẫu; Bảng Bootstrap ghép cặp; Đồ thị hệ số OLS & Random Forest Feature Importance; Expander chẩn đoán (BP, RESET, Joint Wald, Q-Q Plot); Hộp cảnh báo giới hạn phương pháp luận (Correlation $\ne$ Causation, Unobserved Factors).
  2. **Bộ kiểm thử tự động (Unit Test Suite - PR #34 đã merge):**
     * Tự động hóa kiểm thử tính nhất quán của số liệu đóng băng (Train $N=1,685$, Test $N=410$, 144 bài đa thị trường, bậc tự do kiểm định F và Wald).

---

## PHẦN 2. KỊCH BẢN THUYẾT TRÌNH (SCRIPT 5–6 PHÚT)
*(Khẩu khí chững chạc, học thuật, đúng trọng tâm công việc được giao, không lan man)*

> **[00:00 — 00:30] MỞ ĐẦU & GIỚI THIỆU PHẠM VI TRÁCH NHIỆM:**  
> *"Kính thưa Thầy/Cô và Hội đồng chấm, em là Thành viên 1. Trong đề tài nghiên cứu 'What Makes a Hit Song?', trách nhiệm cốt lõi của em là **Xử lý Đa cộng tuyến ma trận biến, Đặc tả Mô hình Kinh tế lượng OLS Model 3, Thực hiện Kiểm định Chẩn đoán Khuyết tật Mô hình, Đối chuẩn Hiệu năng Ngoài mẫu với Random Forest, Xây dựng Phân hệ Tab 4 trên Streamlit và Thiết kế Bộ Kiểm thử Tự động**.*
>
> ---
>
> **[00:30 — 01:45] SLIDE 1: XỬ LÝ ĐA CỘNG TUYẾN & MA TRẬN BIẾN MODEL 3:**  
> *(Chuyển sang Slide 1)*  
> *Bước vào giai đoạn mô hình hóa, thách thức đầu tiên về kinh tế lượng là kiểm soát đa cộng tuyến. Khi phân tích ma trận tương quan giữa 17 đặc trưng âm học do nhóm tự trích xuất từ Librosa, em phát hiện cặp biến `spectral_centroid` và `spectral_bandwidth` có hệ số tương quan thực nghiệm rất cao, đạt $r = 0.897$.*  
> *Dựa trên nguyên tắc kinh tế lượng về tính độc lập của biến giải thích, em đã quyết định loại bỏ `spectral_bandwidth` và giữ lại `spectral_centroid` cùng `energy_std`. Kết quả kiểm toán VIF trên toàn bộ 24 biến chính của tập Train $N = 1,685$ quan sát cho thấy: toàn bộ VIF đều giảm xuống dưới ngưỡng an toàn 5.0, trong đó biến có VIF cao nhất chỉ là `word_count_std` đạt $3.77$, và `energy_std` đạt $2.28$.*  
> *Từ đây, ma trận hồi quy Model 3 được thiết lập hoàn chỉnh với 34 bậc tự do, kết hợp giữa 15 biến âm học, 6 biến ca từ NLP, 3 biến bối cảnh và 10 biến tương tác thị trường trên grain dữ liệu chuẩn: mỗi dòng là một cặp track và market.*
>
> ---
>
> **[01:45 — 03:15] SLIDE 2: KHUYẾT TẬT OLS & BẮT BUỘC CLUSTER-ROBUST SE:**  
> *(Chuyển sang Slide 2)*  
> *Sau khi ước lượng mô hình, em tiến hành kiểm định chẩn đoán toàn diện các giả định kinh tế lượng cổ điển:*  
> *Đầu tiên, kiểm định Breusch–Pagan cho giá trị thống kê $LM = 84.07$ với $p = 3.99 \times 10^{-6}$, bác bỏ triệt để giả định phương sai sai số đồng nhất.*  
> *Đặc biệt, về mặt cấu trúc dữ liệu, mẫu nghiên cứu có **144 bài hát xuất hiện đồng thời ở từ 2 đến 3 thị trường, tạo ra 330 dòng quan sát lặp lại**. Hiện tượng này gây ra tương quan sai số nội cụm: sai số của cùng một bài hát giữa các thị trường không độc lập nhau ($\text{Cov}(u_{it}, u_{is}) \ne 0$). Nếu sử dụng OLS cổ điển hay White SE thông thường, sai số chuẩn sẽ bị đánh giá thấp và làm thổi phồng giả tạo giá trị thống kê t.*  
> *Vì vậy, em đã chuẩn hóa bắt buộc việc áp dụng **Sai số chuẩn vững theo cụm track (`spotify_track_id`) theo phương pháp luận Cameron & Miller (2015)** cho cả Model 1, Model 2 và Model 3.*  
> *Bên cạnh đó, kiểm định Ramsey RESET cho giá trị $p = 0.00996$, bác bỏ dạng hàm tuyến tính thuần túy và báo hiệu sự tồn tại của các mối quan hệ phi tuyến trong dữ liệu âm nhạc.*
>
> ---
>
> **[03:15 — 04:30] SLIDE 3: ĐÁNH GIÁ NGOÀI MẪU & ĐỐI CHUẨN RANDOM FOREST:**  
> *(Chuyển sang Slide 3)*  
> *Để đánh giá năng lực dự báo thực tế và tránh bẫy quá khớp (overfitting), em tiến hành kiểm thử trên tập Test độc lập gồm $410$ quan sát ngoài mẫu:*  
> *Thứ nhất, qua kiểm định Paired Cluster Bootstrap lồng nhau, khi đi từ Model 2 lên Model 2b, khoảng tin cậy 95% của $\Delta R^2$ là $[+0.001, +0.078]$ và $\Delta \text{MAE}$ là $[+0.01, +1.28]$ đều hoàn toàn **không chứa 0**, chứng minh biến năm phát hành `release_year` mang lại cải thiện thực chất. Tuy nhiên, khi bổ sung tiếp 10 biến tương tác thị trường ở Model 3, khoảng tin cậy lại bao hàm giá trị 0, cho thấy các tương tác phức tạp có xu hướng quá khớp trên tập Train.*  
> *Thứ hai, khi đối chuẩn OLS Model 3 với mô hình học máy phi tuyến Random Forest, mức chênh lệch $\Delta R^2 = +0.035$ và $\Delta \text{MAE} = +0.65$ đều có khoảng tin cậy 95% bao hàm giá trị 0. Kết quả này chứng minh: mô hình phi tuyến phức tạp chưa đem lại sự vượt trội có ý nghĩa thống kê ngoài mẫu so với mô hình OLS cô đọng, củng cố nguyên tắc parsimony trong kinh tế lượng.*
>
> ---
>
> **[04:30 — 05:30] SLIDE 4: DASHBOARD TAB 4 & KIỂM THỬ TỰ ĐỘNG:**  
> *(Chuyển sang Slide 4)*  
> *Toàn bộ phân tích, đồ thị và kiểm định kinh tế lượng này đã được em đóng gói thành công thành **Phân hệ Tab 4: Chẩn Đoán & Benchmark trên giao diện Streamlit (`tab_diagnostics.py`)**.*  
> *Phân hệ hoạt động theo kiến trúc Offline-first hoàn chỉnh, đọc số liệu động từ repository, áp dụng logic 3 trạng thái an toàn cho dữ liệu khuyết thiếu và tuyệt đối tuân thủ ngôn ngữ tương quan thống kê, cảnh báo rõ ràng rằng tương quan không đồng nghĩa với quan hệ nhân quả.*  
> *Đồng thời, em đã xây dựng một bộ kiểm thử tự động (Unit Test Suite) trên nhánh `feature/tv1-tests-report` đã merge thành công vào nhánh main, đảm bảo mọi chỉ số thống kê đều được khóa cứng và kiểm chứng tự động.*  
> *Em xin chân thành cảm ơn Thầy/Cô và xin chuyển giao phần trình bày tiếp theo cho Thành viên 2 ạ."*

---

## PHẦN 3. BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN CHUYÊN MÔN (Q&A DEFENSE)

### ❓ Câu 1: Tại sao phải loại bỏ `spectral_bandwidth` thay vì giữ lại cả hai?
> **Trả lời:**  
> *"Dạ thưa Thầy/Cô, tương quan Pearson thực nghiệm giữa `spectral_centroid` và `spectral_bandwidth` lên tới $0.897$. Nếu giữ cả hai, ma trận biến sẽ rơi vào tình trạng đa cộng tuyến nghiêm trọng, làm thổi phồng phương sai sai số chuẩn (VIF vượt ngưỡng), khiến các hệ số hồi quy bị bất ổn định và đảo lộn dấu thống kê. Về mặt vật lý âm học, `spectral_centroid` đo lường trọng tâm năng lượng tần số (độ sáng), trong khi `spectral_bandwidth` đo độ rộng dải tần xung quanh trọng tâm đó. Bằng cách giữ `spectral_centroid` kết hợp với `energy_std`, mô hình vừa bảo toàn được thông tin về độ sáng và mức năng lượng tín hiệu, vừa triệt tiêu được đa cộng tuyến, đưa toàn bộ VIF về dưới mức 3.8."*

### ❓ Câu 2: Tại sao White SE (Robust SE thông thường) chưa đủ mà bắt buộc phải dùng Cluster-Robust SE theo bài hát?
> **Trả lời:**  
> *"Dạ thưa Thầy/Cô, kiểm định Breusch–Pagan ($p = 2.73 \times 10^{-6}$) đã chứng minh phương sai thay đổi. Tuy nhiên, White SE cổ điển chỉ xử lý được phương sai thay đổi mà vẫn giả định các quan sát độc lập nhau. Trong tập dữ liệu của nhóm, có 144 bài hát xuất hiện đồng thời ở từ 2 đến 3 thị trường ($330$ dòng quan sát). Cùng một bài hát thì sai số giữa các thị trường chắc chắn có tương quan: $\text{Cov}(u_{it}, u_{is}) \ne 0$. Theo Cameron & Miller (2015), nếu dùng White SE trong trường hợp có tương quan cụm, sai số chuẩn vẫn bị chệch xuống dưới (underestimated), làm t-statistic bị thổi phồng giả tạo. Do đó, nhóm bắt buộc phải phân cụm theo `spotify_track_id` để suy luận thống kê được chuẩn xác."*

### ❓ Câu 3: Kiểm định Ramsey RESET cho $p = 0.010$ (bác bỏ tuyến tính) và $R^2$ Test chỉ đạt ~3.5%, tại sao nhóm vẫn báo cáo OLS?
> **Trả lời:**  
> *"Dạ thưa Thầy/Cô, đây là lý do nhóm xây dựng kiến trúc song hành (Dual-Model Architecture). Trong kinh tế lượng thực chứng, $R^2$ ở mức 3–7% là đặc tính hoàn toàn bình thường và trung thực đối với dữ liệu âm nhạc thương mại, bởi vì thành bại của một bài hát chịu sự chi phối rất lớn của các yếu tố ngoại sinh chưa quan sát được (unobserved factors) như danh tiếng nghệ sĩ, ngân sách marketing hay thuật toán đề xuất TikTok. OLS cung cấp cho chúng ta hệ số góc chuẩn hóa có thể giải thích được về mặt cơ chế (interpretability), trong khi Random Forest đóng vai trò đối chuẩn benchmark phi tuyến. Việc kiểm định ghép cặp Bootstrap giữa RF và OLS có khoảng tin cậy chứa 0 chứng minh mô hình OLS đã cô đọng và phản ánh tương đối trọn vẹn phần tín hiệu dự báo được ngoài mẫu."*
