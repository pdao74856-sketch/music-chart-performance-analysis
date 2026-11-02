-- ============================================================================
-- SQL Schema Migration: 05_view_clean_modeling_dataset.sql
-- Project: "What Makes a Hit Song? Comparing chart performance between
--           Vietnam, US, and South Korea (2021–2026)"
-- Course: ADY201m - Applied Data Analytics
-- Milestone: Checkpoint 0 (CP 0 — 23:59 Thứ Hai 02/11/2026) CRITICAL PATH
-- Owner: Member 4 (Data Integration, Quality Control, Benchmark & Deploy Lead)
-- Specification Reference: Plan v2 (Librosa Features + Cluster-Robust SE + VIF Spectral)
-- Target Database: Supabase PostgreSQL (schema: public)
-- Target Grain: 1 row = 1 (spotify_track_id, market) [Tối đa 3 dòng / bài hát]
-- Target Sample Size: Exactly N = 2,098 clean observations
--   - South Korea (KR): 693
--   - United States (US): 797
--   - Vietnam (VN):       608
-- ============================================================================
-- Architecture & Methodological Notes:
-- 1. Strictly Audio-derived Librosa Features:
--    Spotify closed audio feature endpoints in 11/2024. This project completely
--    avoids Spotify API and proprietary metrics (danceability, valence).
--    All 17 acoustic signals are open-source audio proxies extracted via Librosa.
--
-- 2. Multicollinearity Prevention (r = 1.0 pairs eliminated):
--    - tempo_bpm -> aliased to `tempo`, duplicate dropped.
--    - rms_energy -> aliased to `energy`, duplicate dropped.
--    - beat_interval_variability -> replaced by standardized `tempo_stability` in (0, 1].
--    - unique_word_count & lexical_diversity -> replaced by `repetition_ratio` (1 - TTR)
--      to eliminate infinite VIF from exact linear dependence.
--
-- 3. OLS vs. Random Forest Feature Strategy:
--    - loudness_proxy (in dBFS) and spectral_bandwidth are retained in this View
--      so they are available for Random Forest (Model 4), Tab 1 Showcase, and Tab 2 Sliders.
--    - OLS Model 1, 2, 3 exclude loudness_proxy during Python preprocessing
--      (prepare_features.py) because Corr(energy, loudness_proxy) = 0.971 (VIF > 18.0).
--    - spectral_bandwidth undergoes empirical VIF check in CP 1.5 (threshold = 5.0)
--      before deciding its inclusion in OLS main effects.
--
-- 4. Non-linear Transform Policy:
--    Logarithmic transformation log(1 + x) (np.log1p) on lyrics metrics (word_count,
--    line_count, chorus_repetition_count) is performed in the Python preprocessing
--    pipeline (prepare_features.py / lyrics_processor.py), NOT hardcoded in this View.
--
-- 5. Zero Data Leakage Contract:
--    Excludes all post-chart performance variables:
--    lifetime_average_rank, peak_rank, weeks_on_chart, weeks_in_top10, streams.
-- ============================================================================

BEGIN;

-- 1. Xóa View cũ nếu đã tồn tại để cập nhật schema mới nhất
DROP VIEW IF EXISTS public.view_clean_modeling_dataset CASCADE;

-- 2. Khởi tạo View chuẩn hóa dữ liệu phục vụ huấn luyện mô hình
CREATE OR REPLACE VIEW public.view_clean_modeling_dataset AS
SELECT 
    -- ------------------------------------------------------------------------
    -- PHẦN 1: ĐỊNH DANH & HIỂN THỊ (METADATA & IDENTIFIERS)
    -- Phục vụ GroupShuffleSplit / GroupKFold cluster theo bài hát & Tab 1 Showcase
    -- ------------------------------------------------------------------------
    spotify_track_id,
    market,
    title,
    artist,

    -- ------------------------------------------------------------------------
    -- PHẦN 2: BIẾN MỤC TIÊU & BIẾN KIỂM SOÁT BỐI CẢNH (TARGET & CONTEXT)
    -- ChartScore = 101 - LifetimeAverageRank, miền giá trị liên tục [1.0, 100.0]
    -- ------------------------------------------------------------------------
    chart_score,
    release_year,

    -- ------------------------------------------------------------------------
    -- PHẦN 3: 17 ĐẶC TRƯNG ÂM HỌC LIBROSA (X_AUDIO)
    -- Tín hiệu sóng âm thanh thực tế, hoàn toàn phi-Spotify
    -- ------------------------------------------------------------------------
    
    -- 3.1. Nhóm Nhịp điệu & Động năng (5 biến)
    tempo_bpm                       AS tempo,
    onset_rate,
    tempo_stability,                -- Chuẩn hóa: 1 / (1 + beat_interval_variability)
    rms_energy                      AS energy,
    rms_variability,

    -- 3.2. Nhóm Âm sắc & Phổ tần số (3 biến)
    loudness_proxy,                 -- Độ lớn biểu kiến: 20 * log10(RMS) [dBFS]
    spectral_centroid,              -- Trọng tâm phổ tần số (độ sáng âm sắc)
    spectral_bandwidth,             -- Độ rộng dải phổ tần số

    -- 3.3. Nhóm Hòa âm & Cấu trúc (3 biến)
    chord_change_rate,              -- Tốc độ chuyển hợp âm (từ chromagram)
    key_confidence,                 -- Độ tự tin dự đoán tông nhạc [0.0, 1.0]
    CASE 
        WHEN audio_mode::text IN ('1', 'major', 'Major') THEN 1
        ELSE 0
    END                             AS mode, -- Điệu thức: Major=1, Minor=0

    -- 3.4. Nhóm Chroma / Cấu trúc giai điệu lặp RQA (6 biến)
    chroma_repetition_determinism,  -- Tính xác định pattern giai điệu lặp
    chroma_mean_pattern_duration,   -- Thời lượng trung bình pattern lặp
    chroma_longest_pattern_duration,-- Pattern lặp dài nhất trong bài
    mfcc_repetition_determinism,    -- Tính xác định pattern âm sắc MFCC lặp
    mfcc_mean_pattern_duration,     -- Thời lượng trung bình pattern MFCC lặp
    mfcc_longest_pattern_duration,  -- Pattern MFCC lặp dài nhất

    -- ------------------------------------------------------------------------
    -- PHẦN 4: 6 ĐẶC TRƯNG CA TỪ NLP TINH GỌN (X_LYRICS)
    -- Khử triệt để đa cộng tuyến hoàn hảo (UniqueWordCount & LexicalDiversity dropped)
    -- ------------------------------------------------------------------------
    word_count,                     -- Tổng số từ (log1p xử lý trong Python)
    repetition_ratio,               -- Tỷ lệ lặp lại ca từ (1 - TTR)
    avg_word_length,                -- Chiều dài từ trung bình (ký tự/từ)
    line_count,                     -- Số dòng lời bài hát
    chorus_repetition_count,        -- Số lần điệp khúc lặp lại
    punctuation_ratio               -- Tỷ lệ dấu câu

FROM public.modeling_dataset
WHERE is_sampled = TRUE 
  AND has_both_rows = TRUE;

-- 3. Ghi chú metadata cho View trong PostgreSQL Catalog
COMMENT ON VIEW public.view_clean_modeling_dataset IS 
    'Canonical clean multimodal modeling dataset for ADY201m (N=2,098; KR=693, US=797, VN=608). Filtered by is_sampled=TRUE AND has_both_rows=TRUE. Contains 17 Librosa audio features and 6 NLP lyrics features with zero data leakage.';

COMMIT;

-- ============================================================================
-- AUDIT & VERIFICATION QUERIES (DÀNH CHO THÀNH VIÊN 4 & KIỂM ĐỊNH CHECKPOINT 0)
-- Có thể copy và thực thi trực tiếp trên Supabase SQL Editor
-- ============================================================================

-- [AUDIT 1] Kiểm tra tổng số dòng và phân bổ theo thị trường
-- Kỳ vọng: Tổng = 2,098 dòng (KR = 693, US = 797, VN = 608)
-- ----------------------------------------------------------------------------
-- SELECT 
--     market,
--     COUNT(*) AS row_count,
--     COUNT(DISTINCT spotify_track_id) AS unique_songs,
--     ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) AS pct_share
-- FROM public.view_clean_modeling_dataset
-- GROUP BY market
-- ORDER BY market;

-- [AUDIT 2] Kiểm tra tính toàn vẹn khóa chính (spotify_track_id, market)
-- Kỳ vọng: 0 dòng trả về (Không có trùng lặp grain)
-- ----------------------------------------------------------------------------
-- SELECT 
--     spotify_track_id, 
--     market, 
--     COUNT(*) AS duplicate_count
-- FROM public.view_clean_modeling_dataset
-- GROUP BY spotify_track_id, market
-- HAVING COUNT(*) > 1;

-- [AUDIT 3] Kiểm tra độ trọn vẹn dữ liệu (Missing Value Audit)
-- Kỳ vọng: Tất cả các cột đều trả về 0 giá trị NULL
-- ----------------------------------------------------------------------------
-- SELECT 
--     COUNT(*) AS total_rows,
--     COUNT(*) - COUNT(spotify_track_id)             AS null_track_id,
--     COUNT(*) - COUNT(market)                       AS null_market,
--     COUNT(*) - COUNT(chart_score)                  AS null_chart_score,
--     COUNT(*) - COUNT(release_year)                 AS null_release_year,
--     COUNT(*) - COUNT(tempo)                        AS null_tempo,
--     COUNT(*) - COUNT(onset_rate)                   AS null_onset_rate,
--     COUNT(*) - COUNT(tempo_stability)              AS null_tempo_stability,
--     COUNT(*) - COUNT(energy)                       AS null_energy,
--     COUNT(*) - COUNT(loudness_proxy)               AS null_loudness_proxy,
--     COUNT(*) - COUNT(rms_variability)              AS null_rms_variability,
--     COUNT(*) - COUNT(spectral_centroid)            AS null_spectral_centroid,
--     COUNT(*) - COUNT(spectral_bandwidth)           AS null_spectral_bandwidth,
--     COUNT(*) - COUNT(chord_change_rate)            AS null_chord_change_rate,
--     COUNT(*) - COUNT(key_confidence)               AS null_key_confidence,
--     COUNT(*) - COUNT(mode)                         AS null_mode,
--     COUNT(*) - COUNT(chroma_repetition_determinism) AS null_chroma_det,
--     COUNT(*) - COUNT(chroma_mean_pattern_duration)  AS null_chroma_mean,
--     COUNT(*) - COUNT(chroma_longest_pattern_duration) AS null_chroma_max,
--     COUNT(*) - COUNT(mfcc_repetition_determinism)   AS null_mfcc_det,
--     COUNT(*) - COUNT(mfcc_mean_pattern_duration)    AS null_mfcc_mean,
--     COUNT(*) - COUNT(mfcc_longest_pattern_duration) AS null_mfcc_max,
--     COUNT(*) - COUNT(word_count)                   AS null_word_count,
--     COUNT(*) - COUNT(repetition_ratio)             AS null_repetition_ratio,
--     COUNT(*) - COUNT(avg_word_length)              AS null_avg_word_length,
--     COUNT(*) - COUNT(line_count)                   AS null_line_count,
--     COUNT(*) - COUNT(chorus_repetition_count)      AS null_chorus_rep,
--     COUNT(*) - COUNT(punctuation_ratio)            AS null_punct_ratio
-- FROM public.view_clean_modeling_dataset;

-- [AUDIT 4] Kiểm tra miền giá trị hợp lệ (Sanity Range Check)
-- Kỳ vọng: Tất cả min/max đều nằm trong giới hạn lý thuyết & thực nghiệm
-- ----------------------------------------------------------------------------
-- SELECT 
--     MIN(chart_score) AS min_chart_score, MAX(chart_score) AS max_chart_score,
--     MIN(mode) AS min_mode, MAX(mode) AS max_mode,
--     MIN(tempo) AS min_tempo, MAX(tempo) AS max_tempo,
--     MIN(loudness_proxy) AS min_loudness_dbfs, MAX(loudness_proxy) AS max_loudness_dbfs,
--     MIN(tempo_stability) AS min_tempo_stab, MAX(tempo_stability) AS max_tempo_stab,
--     MIN(repetition_ratio) AS min_rep_ratio, MAX(repetition_ratio) AS max_rep_ratio
-- FROM public.view_clean_modeling_dataset;

-- [AUDIT 5] Thống kê phân bố cụm bài hát (Cluster Distribution for Cluster-Robust SE)
-- Đếm số bài xuất hiện ở 1, 2 hoặc cả 3 thị trường
-- ----------------------------------------------------------------------------
-- WITH track_market_counts AS (
--     SELECT spotify_track_id, COUNT(DISTINCT market) AS markets_per_track
--     FROM public.view_clean_modeling_dataset
--     GROUP BY spotify_track_id
-- )
-- SELECT 
--     markets_per_track,
--     COUNT(*) AS unique_songs,
--     SUM(markets_per_track) AS total_rows_generated
-- FROM track_market_counts
-- GROUP BY markets_per_track
-- ORDER BY markets_per_track;
