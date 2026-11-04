# RUN STATUS - MODELING PIPELINE EXECUTION LOG

| Pha | Trạng thái | Thời gian hoàn thành | Kết quả chính |
|---|---|---|---|
| **PHA 0** | Hoàn thành | 2026-11-03 22:32 | Git diff của `prepare_features.py` và `supabase_client.py` đã duyệt |
| **PHA 1** | Hoàn thành | 2026-11-03 22:18 | `audio_mode`: major 1.275 (60.77%), minor 823 (39.23%), NULL 0. Fallback strict `ValueError`. DEFAULT_SELECT_COLUMNS 32 cols. Data fingerprint: `902bace8d98815bc`. |
| **PHA 2** | Hoàn thành | 2026-11-03 19:10 | Sao lưu `reports/sensitivity_loudness_proxy/`. Đổi OLS sang `energy`. Train VIF = 2.085 (< 5.0). Kruskal-Wallis H = 139.40, p = 5.35e-31 (< 0.001), η² = 0.0817. |
| **PHA 3** | Hoàn thành | 2026-11-04 19:35 | Re-run split 382/382 tracks (0 diff). `train_model3_full_ols.py` exit 0. `mlr_pipeline.pkl` reload khớp 0.00e+00 (< 1e-9). Test R² M3 = 0.034012, MAE = 29.780379, RMSE = 38.304187. |
| **PHA 4** | Hoàn thành | 2026-11-04 19:34 | Tích hợp Wald 19 DOF (Audio full, χ²=44.33, p=8.50e-4) & 12 DOF (Lyrics full, χ²=19.68, p=0.0733). Wald M1 Audio 15 DOF (χ²=30.80, p=0.0093). Wald M2 Lyrics 6 DOF (χ²=15.57, p=0.0162). Tiêu đề Forest plot (b) đã bỏ chữ "Moderation". |
| **PHA 5** | Hoàn thành | 2026-11-04 19:37 | Train Pearson/Spearman r = 1.000000 giữa `energy` và 10^(loudness_proxy/20). Slope a = 1.000000. Đã ghi nhận `pip freeze`. |
| **PHA 6** | Hoàn thành | 2026-11-04 19:41 | Hoàn tất bảng đối chiếu hiệu năng Test & Hệ số M3. Xác nhận 0 commit, 0 staged. |
