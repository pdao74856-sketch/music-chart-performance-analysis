# Báo Cáo Kiểm Định: Chẩn đoán OLS, So sánh Mô hình Lồng nhau & Kiểm định Wald (Model 3)

**Người thực hiện:** Thành viên 1 (Modeling & Analysis Lead)
**Dữ liệu:** `view_clean_modeling_dataset` (2.098 dòng; sau khi loại 3 bài instrumental còn 2.095). Train 1.685 dòng / 1.527 bài; Test 410 dòng / 382 bài; split theo `spotify_track_id`, seed 43.
**Lưu ý ngôn ngữ:** mọi kết quả là tương quan, không phải nhân quả.

---

## 1. Mục đích & Giả thuyết thống kê

Mức ý nghĩa α = 0,05. Ba nhóm kiểm định:

| Kiểm định | H₀ | Câu hỏi |
| :--- | :--- | :--- |
| Breusch–Pagan | Phương sai sai số đồng nhất | Sai số chuẩn OLS thường có dùng được không? |
| So sánh mô hình lồng nhau (Test, bootstrap theo bài) | Thêm một khối biến không cải thiện dự báo ngoài mẫu | Khối nào (audio, lyrics, bối cảnh, tương tác) thực sự cải thiện dự báo trên bài chưa thấy? |
| Wald đồng thời + Holm | Toàn bộ hệ số của một khối biến = 0 | Khối nào có liên quan đến ChartScore (trong mẫu Train)? |

## 2. Code thực thi

Mọi kết quả lấy từ script đã commit, không tính tay:
- `src/modeling/evaluate_diagnostics.py` → `reports/diagnostics/diagnostics_summary.json` (Breusch–Pagan, Cook's distance, độ nhạy bỏ bài trước 1980).
- `src/modeling/evaluate_diagnostics_reset.py` → `reports/diagnostics/reset_test.json` (Ramsey RESET).
- `src/modeling/paired_bootstrap_nested.py` → `reports/diagnostics/paired_bootstrap_nested.csv`.
- Wald theo khối, hệ số, độ dốc: `reports/model_wald_blocks.csv`, `reports/model3_coefficients.csv`, `reports/model_market_slopes.csv`; hiệu năng Test: `reports/model_test_comparison.csv`.
- VIF ma trận 24 biến chính của M3: `src/modeling/compute_vif_final_energy.py` → `reports/vif_final_energy.csv`.
- Hàm dùng chung: `src/utils/ols_cluster_fit.py` (`fit_ols_cluster`, `wald_block`), sai số chuẩn cụm theo `spotify_track_id`.

```
python src/modeling/evaluate_diagnostics.py
python src/modeling/evaluate_diagnostics_reset.py
python src/modeling/paired_bootstrap_nested.py
```

## 3. Biểu đồ minh họa

![Forest plot các hiệu ứng chính của Model 3](../figures/forest_main.png)
*Hình 1: Hệ số hồi quy chuẩn hóa của Model 3 với khoảng tin cậy 95% theo sai số chuẩn cụm.*

![Độ dốc theo thị trường](../figures/forest_market_slopes.png)
*Hình 2: Độ dốc theo từng thị trường (US là nhóm tham chiếu); khoảng tin cậy chưa hiệu chỉnh đa kiểm định.*

![Phân phối năm phát hành (Train)](../figures/release_year_distribution_train.png)
*Hình 3: Phân phối `release_year` trong tập Train; biến này cũng là thành phần chính của khối bối cảnh.*

Hình phụ (chẩn đoán phần dư): `../diagnostics/qq_residuals.png`, `../diagnostics/residuals_vs_fitted.png`.

## 4. Bảng kết quả thống kê cốt lõi

**4.1 Breusch–Pagan (Train, n = 1.685)**

| Thống kê | Giá trị | p-value | Quyết định |
| :--- | :---: | :---: | :--- |
| LM | 84,07 | 3,99 × 10⁻⁶ | Bác bỏ H₀ |
| F | 2,55 | 2,73 × 10⁻⁶ | Bác bỏ H₀ |

Phương sai sai số không đồng nhất. Ngoài ra, một bài hát có thể có tối đa 3 dòng (3 thị trường) nên sai số trong cùng bài không độc lập. Vì vậy mọi suy luận dùng sai số chuẩn vững theo cụm (Cameron & Miller, 2015).

**4.2 Hiệu năng ngoài mẫu của các mô hình lồng nhau (Test, n = 410)**

| Mô hình | Số tham số | R² Train | R² Test [95% CI] | MAE Test [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| Baseline: trung bình Train | 1 | – | −0,001 | 30,37 |
| Baseline: trung bình theo thị trường | 3 | – | 0,002 | 30,27 |
| M1 (audio) | 16 | 0,016 | −0,001 [−0,033; 0,024] | 30,63 [28,33; 33,13] |
| M2 (+ lyrics) | 22 | 0,026 | −0,002 [−0,037; 0,029] | 30,79 [28,48; 33,23] |
| M2b (+ năm phát hành, dummy thị trường) | 25 | 0,056 | 0,036 [−0,015; 0,082] | 30,06 [27,90; 32,52] |
| M3 (+ 10 tương tác) | 35 | 0,066 | 0,034 [−0,018; 0,082] | 29,78 [27,48; 32,25] |

So sánh cặp (bootstrap theo bài, 1.000 lần, seed 43; chênh lệch dương = mô hình sau tốt hơn):

| Bước | ΔMAE [95% CI] | ΔR² [95% CI] | Kết luận |
| :--- | :---: | :---: | :--- |
| M1 → M2 (thêm lyrics) | −0,15 [−0,49; +0,19] | −0,000 [−0,018; +0,016] | Không cải thiện phát hiện được |
| M2 → M2b (thêm bối cảnh) | **+0,73 [+0,13; +1,38]** | **+0,038 [+0,001; +0,078]** | Cải thiện duy nhất phát hiện được (CI sát 0) |
| M2b → M3 (thêm tương tác) | +0,28 [−0,10; +0,63] | −0,002 [−0,025; +0,019] | Không cải thiện phát hiện được |

Các CI chưa hiệu chỉnh đa so sánh; R² Test của M2b và M3 đều có CI chứa 0, nên đây là bằng chứng thăm dò.

**4.3 Wald đồng thời theo khối, Model 3 (Train; 4 khối thuộc họ Holm)**

| Khối | df | Wald χ² | p thô | p Holm |
| :--- | :---: | :---: | :---: | :---: |
| Audio | 15 | 39,84 | 0,00048 | 0,0014 |
| Lyrics | 6 | 11,74 | 0,068 | 0,113 |
| Context (năm + thị trường) | 3 | 39,27 | 1,52 × 10⁻⁸ | 6,07 × 10⁻⁸ |
| Tương tác (biến × thị trường) | 10 | 17,91 | 0,0566 | 0,113 |

Độ nhạy khi bỏ 17 dòng trước 1980: p khối tương tác 0,057 → 0,194; p khối lyrics 0,068 → 0,124.

**4.4 Độ dốc energy theo thị trường (p chưa hiệu chỉnh)**

| Thị trường | Độ dốc (per 1 SD) | 95% CI | p |
| :--- | :---: | :---: | :---: |
| US (tham chiếu) | −4,62 | [−8,43; −0,81] | 0,017 |
| VN | +0,07 | [−4,59; +4,73] | 0,976 |
| KR | −0,93 | [−5,09; +3,22] | 0,660 |

Hệ số tương tác energy × VN = +4,69 (p = 0,083), energy × KR = +3,69 (p = 0,131).

**4.5 Ramsey RESET (Train, ŷ² và ŷ³) — nêu như một hạn chế**

F thường = 12,37 (df 2; 1.648; p = 4,67 × 10⁻⁶); Wald theo cụm χ²(2) = 9,22 (p = 0,010). Có dấu hiệu dạng hàm tuyến tính của M3 chưa đủ hoặc thiếu biến; RESET không cho biết nguồn gốc cụ thể.

**4.6 VIF của ma trận 24 biến chính Model 3 (Train, bản dùng `energy_std`)**

Ngưỡng tham chiếu VIF < 5. Cao nhất: `word_count_std` 3,77; `chroma_longest_pattern_duration_std` 2,77; `line_count_std` 2,76; `mfcc_longest_pattern_duration_std` 2,48; `energy_std` 2,28. Mọi biến đều < 5 (đầy đủ trong `reports/vif_final_energy.csv`). `loudness_proxy` không đưa vào OLS vì loudness_proxy = 20·log10(energy); `spectral_bandwidth` bị loại do tương quan r = 0,909 với `spectral_centroid`. Lưu ý: file `reports/vif_after.csv` cũ là bản giữ loudness, không phải bản dùng cho Model 3 hiện tại.

## 5. Kết luận & điểm nhấn đưa lên slide

- Phương sai sai số không đồng nhất (Breusch–Pagan LM = 84,07, p < 0,001) và một bài có thể xuất hiện ở nhiều thị trường → dùng sai số chuẩn vững theo cụm bài hát.
- Ngoài mẫu: audio và lyrics không cải thiện dự báo (R² Test của M1, M2 ≈ 0); bước duy nhất cải thiện phát hiện được là thêm bối cảnh (năm phát hành + thị trường), ΔR² = +0,038 [+0,001; +0,078]. Tuy vậy R² Test cuối cùng vẫn thấp (M3 = 0,034, CI chứa 0).
- Khối audio có liên quan đến ChartScore trong mẫu (Wald χ² = 39,84; Holm p = 0,0014); khối tương tác chỉ ở mức gợi ý (Holm p = 0,113). Độ dốc energy theo thị trường: US −4,62 (p = 0,017), VN +0,07, KR −0,93 (chưa hiệu chỉnh đa kiểm định).
- Hạn chế: RESET (Wald cụm p = 0,010) gợi ý dạng hàm tuyến tính chưa đủ.
- Hai slide chính: (1) M1/M2 ≈ 0 → M2b cải thiện (bối cảnh); (2) Forest plot và độ dốc energy theo thị trường.

**Gắn với Web App:** các kết quả này giải thích bảng hệ số / forest plot ở Tab 4 và việc app hiển thị khoảng bất định thay vì một con số tuyệt đối. (TV4 xác nhận lại tên tab trước khi ghi lên slide.)
