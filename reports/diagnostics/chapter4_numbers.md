# Bảng Tổng Hợp Số Liệu Chính Chương 4 (Kết Quả Mô Hình Hóa)

> **Ghi chú tính toàn vẹn dữ liệu**: Mọi con số trong tài liệu này được trích xuất nguyên văn (verbatim copy) trực tiếp từ các tệp báo cáo gốc tương ứng trong thư mục `reports/`. Tuyệt đối không tự tính toán lại hay làm tròn thủ công.

---

## 1. So Sánh Phân Cấp Các Mô Hình Trên Tập Huấn Luyện (Train Set N = 1,685)
**Nguồn tệp**: `reports/model_comparison_table.csv`

| model | num_features | r_squared | adj_r_squared | aic | bic | delta_r2 | nested_wald_test | wald_chi2 | wald_df | wald_p_value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 (Audio Baseline) | 15 | 0.0164600428074898 | 0.0076205584708284 | 17146.72102804533 | 17233.59336152991 | 0.0 | N/A (Baseline) | nan | nan | nan |
| M2 (Audio + Lyrics) | 21 | 0.0256251978834268 | 0.0133210061549554 | 17142.945674157847 | 17262.39513269915 | 0.009165155075937 | M1 -> M2 (Lyrics 6 DOF) | 15.572202420189404 | 6.0 | 0.0162439404179222 |
| M2b (+ Context) | 24 | 0.0561824293684813 | 0.0425368741304353 | 17095.25628207143 | 17230.99430314109 | 0.0305572314850545 | M2 -> M2b (Context 3 DOF) | 38.14688516527615 | 3.0 | 2.6311540918826226e-08 |
| M3 (Full + Interactions) | 34 | 0.0656061910692936 | 0.0463520156125397 | 17098.347459252815 | 17288.38068875034 | 0.0094237617008122 | M2b -> M3 (Interactions 10 DOF) | 17.906097459872683 | 10.0 | 0.0565684537819801 |

---

## 2. Đánh Giá Khả Năng Tổng Quát Hóa Trên Tập Kiểm Tra Độc Lập (Test Set N = 410, 382 Track)
**Nguồn tệp**: `reports/model_test_comparison.csv`

| model | sample | num_params | n | r2 | r2_ci_95 | mae | mae_ci_95 | rmse | rmse_ci_95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline: Global Train Mean | Overall | 1 | 410 | -0.0009982363249323 | [-0.0196, -0.0000] | 30.36740820149499 | [28.01, 32.94] | 38.99213705505627 | [36.33, 41.94] |
| Baseline: Global Train Mean | Market_US | 1 | 165 | -0.0096315877461876 | [-0.0734, -0.0000] | 28.9772986477052 | [25.40, 32.49] | 36.70645504633581 | [32.45, 40.66] |
| Baseline: Global Train Mean | Market_KR | 1 | 140 | -0.0010484749360508 | [-0.0480, -0.0000] | 28.004256068989807 | [24.10, 32.10] | 36.73283211770215 | [31.94, 41.39] |
| Baseline: Global Train Mean | Market_VN | 1 | 105 | -0.002884985651266 | [-0.0593, -0.0000] | 35.70273558174302 | [30.77, 40.82] | 44.94879633684448 | [39.78, 50.25] |
| Baseline: Market Train Mean | Overall | 3 | 410 | 0.0018055475109061 | [-0.0229, 0.0163] | 30.26713685566141 | [27.91, 32.85] | 38.937490512287 | [36.29, 41.87] |
| Baseline: Market Train Mean | Market_US | 3 | 165 | -7.217637230105822e-06 | [-0.0324, -0.0000] | 28.245896697905767 | [24.46, 31.85] | 36.53108293191848 | [32.15, 40.71] |
| Baseline: Market Train Mean | Market_KR | 3 | 140 | -0.0061875439528602 | [-0.0657, -0.0000] | 28.378811639434893 | [24.56, 32.51] | 36.82699883852812 | [32.06, 41.59] |
| Baseline: Market Train Mean | Market_VN | 3 | 105 | -6.48171164838729e-05 | [-0.0532, -0.0000] | 35.96113834424609 | [31.05, 41.13] | 44.88555258271238 | [39.72, 49.98] |
| Model 1 (Audio Baseline) | Overall | 16 | 410 | -0.0012631824102566 | [-0.0326, 0.0241] | 30.63363421062485 | [28.33, 33.13] | 38.99729696952562 | [36.28, 41.92] |
| Model 1 (Audio Baseline) | Market_US | 16 | 165 | -0.0150291126189292 | [-0.1007, 0.0290] | 29.29738937739211 | [25.85, 32.78] | 36.80444124126508 | [32.65, 40.94] |
| Model 1 (Audio Baseline) | Market_KR | 16 | 140 | -0.0030901248271424 | [-0.0656, 0.0393] | 28.178956589803427 | [24.39, 32.27] | 36.7702715549148 | [32.03, 41.70] |
| Model 1 (Audio Baseline) | Market_VN | 16 | 105 | 0.0037749327721952 | [-0.0632, 0.0450] | 36.00635101441915 | [31.23, 41.12] | 44.79930064945761 | [39.56, 50.05] |
| Model 2 (Audio + Lyrics) | Overall | 22 | 410 | -0.0015948088187289 | [-0.0374, 0.0290] | 30.78615004801293 | [28.48, 33.23] | 39.00375454386714 | [36.28, 41.86] |
| Model 2 (Audio + Lyrics) | Market_US | 22 | 165 | -0.0197885097940333 | [-0.1053, 0.0330] | 29.328713922523324 | [25.91, 32.74] | 36.89062699458119 | [32.75, 41.05] |
| Model 2 (Audio + Lyrics) | Market_KR | 22 | 140 | -0.006310165710099 | [-0.0819, 0.0461] | 28.455625107213773 | [24.55, 32.50] | 36.82924278090479 | [32.21, 41.56] |
| Model 2 (Audio + Lyrics) | Market_VN | 22 | 105 | 0.0106255139048094 | [-0.0592, 0.0493] | 36.18377340437164 | [31.50, 41.19] | 44.645002850487394 | [39.62, 49.72] |
| Model 2b (+ Context) | Overall | 25 | 410 | 0.0364907220102003 | [-0.0151, 0.0822] | 30.056883592682087 | [27.90, 32.52] | 38.25501113536262 | [35.55, 41.05] |
| Model 2b (+ Context) | Market_US | 25 | 165 | 0.0160161700968886 | [-0.0986, 0.1112] | 27.90635491403672 | [24.42, 31.70] | 36.23722730554724 | [31.36, 40.55] |
| Model 2b (+ Context) | Market_KR | 25 | 140 | 0.0207555620576996 | [-0.0668, 0.0812] | 28.367028931468816 | [24.63, 32.42] | 36.33058713375298 | [31.97, 41.15] |
| Model 2b (+ Context) | Market_VN | 25 | 105 | 0.061331356294829 | [-0.0192, 0.1025] | 35.68942535026631 | [31.21, 40.24] | 43.485919438443176 | [38.75, 48.05] |
| Model 3 (Full OLS) | Overall | 35 | 410 | 0.0340120200165667 | [-0.0178, 0.0824] | 29.78037932116539 | [27.48, 32.25] | 38.30418651339223 | [35.43, 41.08] |
| Model 3 (Full OLS) | Market_US | 35 | 165 | 0.0284581296526423 | [-0.0926, 0.1312] | 27.638712832089347 | [24.14, 31.32] | 36.00739810813319 | [31.08, 40.47] |
| Model 3 (Full OLS) | Market_KR | 35 | 140 | 0.0383563140190962 | [-0.0644, 0.1102] | 27.771610656128463 | [24.10, 31.84] | 36.002607181914875 | [31.50, 40.89] |
| Model 3 (Full OLS) | Market_VN | 35 | 105 | 0.0253812632385891 | [-0.0487, 0.0726] | 35.824213452619844 | [31.13, 40.51] | 44.31082947551766 | [39.46, 49.05] |

---

## 3. Kiểm Định Wald Đồng Thời Theo Khối Biến Trên Mô Hình 3 (Train Cluster SE)
**Nguồn tệp**: `reports/model_wald_blocks.csv`

| test_scope | block | model | num_features | chi2 | df | p_value | p_holm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model 3 Core Blocks | Audio Block (Main, M3) | M3 (Full OLS) | 15 | 39.84316878358558 | 15 | 0.0004788851775703 | 0.0014366555327109 |
| Model 3 Core Blocks | Lyrics Block (Main, M3) | M3 (Full OLS) | 6 | 11.744047449842077 | 6 | 0.0679285162325311 | 0.1131369077167799 |
| Model 3 Core Blocks | Context Block (M3) | M3 (Full OLS) | 3 | 39.27441885219337 | 3 | 1.5181924199127472e-08 | 6.072769679650989e-08 |
| Model 3 Core Blocks | Interactions Block (M3) | M3 (Full OLS) | 10 | 17.90609745545936 | 10 | 0.0565684538583899 | 0.1131369077167799 |
| Full Feature Family (with Interactions) | Audio Full Block (Main + Interactions, 19 DOF) | M3 (Full OLS) | 19 | 44.32985829660175 | 19 | 0.0008499714890837 | nan |
| Full Feature Family (with Interactions) | Lyrics Full Block (Main + Interactions, 12 DOF) | M3 (Full OLS) | 12 | 19.68474945982705 | 12 | 0.0732894187624932 | nan |
| Model Specific Joint Significance | Audio Block in M1 (15 DOF) | M1 (Audio Baseline) | 15 | 30.79817162823773 | 15 | 0.00934744728003 | nan |
| Model Specific Joint Significance | Lyrics Block in M2 (6 DOF) | Model 2 (Audio + Lyrics) | 6 | 15.57222523731996 | 6 | 0.0162437967438195 | nan |

### Tiêu Chuẩn Thông Tin Phân Cấp Mô Hình (Train N = 1,685)
**Nguồn tệp**: `reports/model_wald_blocks.csv`

| model | num_params | r2_train | adj_r2_train | aic | bic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M1 (Audio Baseline) | 16 | 0.0164600414914375 | 0.0076205571429482 | 17146.72103029999 | 17233.59336378457 |
| Model 2 (Audio + Lyrics) | 22 | 0.0256252111756947 | 0.0133210196150751 | 17142.945651171343 | 17262.395109712645 |
| Model 2b (+ Context) | 25 | 0.0561824133036225 | 0.0425368578333134 | 17095.256310752065 | 17230.994331821727 |
| M3 (Full OLS) | 35 | 0.0656061759797724 | 0.0463520002120829 | 17098.34748646387 | 17288.380715961397 |

---

## 4. Kiểm Định Wald Cho Từng Biến Tương Tác Của Mô Hình 3 (2 DOF)
**Nguồn tệp**: `reports/model3_wald_features.csv`

| feature | chi2 | df | p_raw | p_holm | significant_05 | q_bh | evidence_tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| spectral_centroid | 6.352276388952458 | 2 | 0.0417465613280957 | 0.2087328066404785 | False | 0.2087328066404784 | gợi ý |
| energy | 3.8173899619706746 | 2 | 0.1482737604665132 | 0.5930950418660531 | False | 0.370684401166283 | không có bằng chứng |
| avg_word_length | 2.766791405192768 | 2 | 0.2507257159240657 | 0.7521771477721972 | False | 0.4178761932067761 | không có bằng chứng |
| word_count | 2.0780080492033983 | 2 | 0.3538068894923297 | 0.7521771477721972 | False | 0.4422586118654121 | không có bằng chứng |
| repetition_ratio | 0.2196045810137986 | 2 | 0.8960112677191238 | 0.8960112677191238 | False | 0.8960112677191238 | không có bằng chứng |

---

## 5. So Sánh Từng Cặp Bằng Bootstrap Theo Cụm (Paired Cluster Bootstrap Test)
**Nguồn tệp**: `reports/paired_bootstrap.csv`

| model | baseline | sample | delta_mae | delta_mae_ci_95 | pct_mae_better | delta_r2 | delta_r2_ci_95 | pct_r2_better |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model 1 (Audio Baseline) | Baseline: Global Train Mean | Overall | -0.2662260091298627 | [-0.72, +0.17] | 11.7 | -0.0002649460853243 | [-0.0271, +0.0254] | 47.8 |
| Model 1 (Audio Baseline) | Baseline: Global Train Mean | Market_US | -0.320090729686914 | [-1.07, +0.31] | 17.0 | -0.0053975248727415 | [-0.0581, +0.0365] | 39.6 |
| Model 1 (Audio Baseline) | Baseline: Global Train Mean | Market_KR | -0.1747005208136194 | [-1.00, +0.67] | 32.7 | -0.0020416498910915 | [-0.0481, +0.0480] | 44.9 |
| Model 1 (Audio Baseline) | Baseline: Global Train Mean | Market_VN | -0.303615432676132 | [-1.16, +0.58] | 25.5 | 0.0066599184234612 | [-0.0360, +0.0515] | 61.5 |
| Model 1 (Audio Baseline) | Baseline: Market Train Mean | Overall | -0.366497354963446 | [-0.91, +0.21] | 8.3 | -0.0030687299211628 | [-0.0339, +0.0290] | 41.1 |
| Model 1 (Audio Baseline) | Baseline: Market Train Mean | Market_US | -1.0514926794863442 | [-1.97, -0.14] | 1.0 | -0.0150218949816991 | [-0.0787, +0.0392] | 29.3 |
| Model 1 (Audio Baseline) | Baseline: Market Train Mean | Market_KR | 0.1998550496314663 | [-0.66, +1.08] | 66.4 | 0.0030974191257178 | [-0.0512, +0.0629] | 52.0 |
| Model 1 (Audio Baseline) | Baseline: Market Train Mean | Market_VN | -0.0452126701730648 | [-0.95, +0.90] | 46.5 | 0.003839749888679 | [-0.0461, +0.0535] | 56.6 |
| Model 2 (Audio + Lyrics) | Baseline: Global Train Mean | Overall | -0.4187418465179426 | [-1.01, +0.15] | 7.399999999999999 | -0.0005965724937966 | [-0.0333, +0.0315] | 46.8 |
| Model 2 (Audio + Lyrics) | Baseline: Global Train Mean | Market_US | -0.3514152748181267 | [-1.35, +0.54] | 21.0 | -0.0101569220478456 | [-0.0728, +0.0438] | 34.599999999999994 |
| Model 2 (Audio + Lyrics) | Baseline: Global Train Mean | Market_KR | -0.4513690382239659 | [-1.61, +0.64] | 21.3 | -0.0052616907740481 | [-0.0722, +0.0531] | 42.2 |
| Model 2 (Audio + Lyrics) | Baseline: Global Train Mean | Market_VN | -0.4810378226286218 | [-1.51, +0.55] | 17.7 | 0.0135104995560755 | [-0.0369, +0.0559] | 70.6 |
| Model 2 (Audio + Lyrics) | Baseline: Market Train Mean | Overall | -0.519013192351526 | [-1.14, +0.13] | 6.2 | -0.003400356329635 | [-0.0376, +0.0308] | 42.9 |
| Model 2 (Audio + Lyrics) | Baseline: Market Train Mean | Market_US | -1.082817224617557 | [-2.24, -0.06] | 1.9 | -0.0197812921568032 | [-0.0918, +0.0410] | 25.4 |
| Model 2 (Audio + Lyrics) | Baseline: Market Train Mean | Market_KR | -0.0768134677788801 | [-1.22, +1.05] | 43.2 | -0.0001226217572387 | [-0.0660, +0.0643] | 47.9 |
| Model 2 (Audio + Lyrics) | Baseline: Market Train Mean | Market_VN | -0.2226350601255546 | [-1.22, +0.80] | 34.9 | 0.0106903310212933 | [-0.0427, +0.0614] | 67.0 |
| Model 2b (+ Context) | Baseline: Global Train Mean | Overall | 0.3105246088129014 | [-0.55, +1.15] | 75.6 | 0.0374889583351326 | [-0.0098, +0.0841] | 93.7 |
| Model 2b (+ Context) | Baseline: Global Train Mean | Market_US | 1.0709437336684786 | [-0.54, +2.67] | 90.1 | 0.0256477578430762 | [-0.0729, +0.1347] | 70.19999999999999 |
| Model 2b (+ Context) | Baseline: Global Train Mean | Market_KR | -0.3627728624790087 | [-1.63, +0.81] | 27.800000000000004 | 0.0218040369937505 | [-0.0515, +0.0865] | 71.8 |
| Model 2b (+ Context) | Baseline: Global Train Mean | Market_VN | 0.0133102314767086 | [-1.24, +1.29] | 50.3 | 0.064216341946095 | [-0.0018, +0.1189] | 97.0 |
| Model 2b (+ Context) | Baseline: Market Train Mean | Overall | 0.2102532629793181 | [-0.57, +1.00] | 68.0 | 0.0346851744992942 | [-0.0114, +0.0788] | 93.0 |
| Model 2b (+ Context) | Baseline: Market Train Mean | Market_US | 0.3395417838690484 | [-1.18, +1.91] | 68.7 | 0.0160233877341187 | [-0.0902, +0.1189] | 61.8 |
| Model 2b (+ Context) | Baseline: Market Train Mean | Market_KR | 0.011782707966077 | [-1.28, +1.15] | 47.0 | 0.0269431060105599 | [-0.0465, +0.0987] | 74.9 |
| Model 2b (+ Context) | Baseline: Market Train Mean | Market_VN | 0.2717129939797758 | [-0.82, +1.45] | 67.80000000000001 | 0.0613961734113128 | [+0.0021, +0.1068] | 97.7 |
| Model 3 (Full OLS) | Baseline: Global Train Mean | Overall | 0.5870288803295978 | [-0.29, +1.53] | 89.4 | 0.0350102563414991 | [-0.0145, +0.0843] | 92.2 |
| Model 3 (Full OLS) | Baseline: Global Train Mean | Market_US | 1.3385858156158503 | [-0.41, +3.03] | 92.7 | 0.03808971739883 | [-0.0740, +0.1544] | 75.8 |
| Model 3 (Full OLS) | Baseline: Global Train Mean | Market_KR | 0.2326454128613448 | [-1.30, +1.58] | 58.9 | 0.039404788955147 | [-0.0502, +0.1164] | 80.60000000000001 |
| Model 3 (Full OLS) | Baseline: Global Train Mean | Market_VN | -0.1214778708768236 | [-1.33, +1.07] | 41.6 | 0.0282662488898551 | [-0.0296, +0.0842] | 82.1 |
| Model 3 (Full OLS) | Baseline: Market Train Mean | Overall | 0.4867575344960144 | [-0.34, +1.40] | 85.5 | 0.0322064725056606 | [-0.0159, +0.0791] | 90.0 |
| Model 3 (Full OLS) | Baseline: Market Train Mean | Market_US | 0.6071838658164204 | [-1.08, +2.40] | 77.4 | 0.0284653472898724 | [-0.0876, +0.1384] | 70.8 |
| Model 3 (Full OLS) | Baseline: Market Train Mean | Market_KR | 0.6072009833064307 | [-0.87, +1.89] | 79.3 | 0.0445438579719564 | [-0.0430, +0.1265] | 83.2 |
| Model 3 (Full OLS) | Baseline: Market Train Mean | Market_VN | 0.1369248916262435 | [-0.95, +1.24] | 61.3 | 0.025446080355073 | [-0.0272, +0.0791] | 81.89999999999999 |

