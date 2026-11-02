-- ============================================================================
-- SQL Schema Migration: 05_view_clean_modeling_dataset.sql
-- Project: What Makes a Hit Song? (Music Chart Performance Analysis)
-- Owner: Member 4 (Data Integration, Quality Control & Deploy Lead)
-- Target Database: Supabase PostgreSQL (public schema)
-- Target Grain: 1 row = 1 (spotify_track_id, market)
-- ============================================================================
-- Description:
--   Creates the canonical clean modeling view `public.view_clean_modeling_dataset`.
--   Filters exactly sampled tracks with complete multimodal features:
--     WHERE is_sampled = TRUE AND has_both_rows = TRUE
--   Features Included:
--     1. Identifiers & Display: spotify_track_id, market, title, artist
--     2. Target & Controls: chart_score, release_year
--     3. 17 Audio Features Librosa (dBFS normalized, tempo stability 1/(1+x))
--     4. 6 NLP Lyrics Features (perfect multicollinearity eliminated)
--   Excluded:
--     - Leakage columns (lifetime_average_rank, peak_rank, weeks_on_chart, streams)
--     - Redundant r=1.0 pairs (tempo_bpm, rms_energy, dynamic_range, beat_interval_variability)
--     - Degenerate collinear features (unique_word_count, lexical_diversity)
-- ============================================================================

BEGIN;

DROP VIEW IF EXISTS public.view_clean_modeling_dataset CASCADE;

CREATE OR REPLACE VIEW public.view_clean_modeling_dataset AS
SELECT 
    -- 1. Định danh & hiển thị (Phục vụ Tab 1 Showcase & Navigation)
    spotify_track_id,
    market,
    title,
    artist,

    -- 2. Biến mục tiêu & biến kiểm soát
    chart_score,
    release_year,

    -- 3. 17 Audio Features Librosa (Đã chuẩn hóa công thức toán học)
    -- Nhóm Nhịp điệu & Động năng (5 biến)
    COALESCE(tempo, tempo_bpm) AS tempo,
    onset_rate,
    tempo_stability,
    COALESCE(energy, rms_energy) AS energy,
    loudness_proxy,
    rms_variability,

    -- Nhóm Âm sắc & Phổ tần số (2 biến)
    spectral_centroid,
    spectral_bandwidth,

    -- Nhóm Hòa âm & Cấu trúc (3 biến)
    chord_change_rate,
    key_confidence,
    CASE 
        WHEN mode::text IN ('1', 'major', 'Major') THEN 1
        WHEN audio_mode::text IN ('1', 'major', 'Major') THEN 1
        ELSE 0
    END AS mode,

    -- Nhóm Chroma / Cấu trúc giai điệu lặp RQA (6 biến)
    chroma_repetition_determinism,
    chroma_mean_pattern_duration,
    chroma_longest_pattern_duration,
    mfcc_repetition_determinism,
    mfcc_mean_pattern_duration,
    mfcc_longest_pattern_duration,

    -- 4. 6 Lyrics Features NLP tinh gọn (Loại bỏ triệt để đa cộng tuyến)
    word_count,
    repetition_ratio,
    avg_word_length,
    line_count,
    chorus_repetition_count,
    punctuation_ratio

FROM public.modeling_dataset
WHERE is_sampled = TRUE 
  AND has_both_rows = TRUE;

-- Comment giải thích View trên Database Supabase
COMMENT ON VIEW public.view_clean_modeling_dataset IS 
    'Clean modeling cohort (is_sampled=TRUE, has_both_rows=TRUE) containing 17 Librosa audio features and 6 NLP lyrics features with zero data leakage.';

COMMIT;
