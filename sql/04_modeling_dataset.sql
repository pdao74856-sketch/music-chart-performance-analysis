-- ============================================================================
-- SQL Schema Migration: 04_modeling_dataset.sql
-- Project: What Makes a Hit Song? (Music Chart Performance Analysis)
-- Owner: Member 4 (Data Integration, Quality Control & Report Coordination)
-- Grain: 1 row = 1 (spotify_track_id, market)
-- ============================================================================
-- Description:
--   Creates the canonical unified VIEW `public.modeling_dataset` combining:
--   1. Target variables & chart history from `public.chart_performance`
--   2. Song metadata from `public.songs` (with derived release_year)
--   3. Acoustic signal features from `public.audio_features` (20 dimensions)
--   4. YouTube candidate verification from `public.youtube_mapping` (is_selected = True)
--   5. Lexical & NLP features from `public.lyrics_features` (10 dimensions)
--   6. Sampling filter `is_sampled = True` linking to `public.song_quarter_sampling`
-- ============================================================================

BEGIN;

-- 1. Drop existing view if present
DROP VIEW IF EXISTS public.modeling_dataset CASCADE;

-- 2. Create the unified modeling view
CREATE VIEW public.modeling_dataset AS
SELECT
    -- Identifiers and Grain Keys
    cp.spotify_track_id,
    cp.market,

    -- Target Variables & Chart Performance (Member 1)
    cp.lifetime_average_rank,
    cp.chart_score,
    cp.peak_rank,
    cp.weeks_on_chart,
    cp.weeks_in_top10,

    -- Metadata & Provenance (Songs table)
    s.title,
    s.artist,
    s.album,
    s.release_date,
    EXTRACT(YEAR FROM s.release_date)::INTEGER AS release_year,

    -- Audio Features: Rhythm & Energy (Member 2)
    af.tempo,
    af.tempo_bpm,
    af.tempo_stability,
    af.beat_interval_variability,
    af.onset_rate,
    af.energy,
    af.rms_energy,
    af.rms_variability,
    af.dynamic_range,
    af.loudness_proxy,

    -- Audio Features: Spectral & Duration (Member 2)
    af.spectral_centroid,
    af.spectral_bandwidth,
    af.duration_seconds,
    af.duration_analyzed_seconds,

    -- Audio Features: Tonal & Harmony (Member 2)
    af.key AS audio_key,
    af.mode AS audio_mode,
    af.key_confidence,
    af.chord_change_rate,

    -- Audio Features: Recurrence Quantification Analysis - RQA (Member 2)
    af.chroma_repetition_determinism,
    af.chroma_mean_pattern_duration,
    af.chroma_longest_pattern_duration,
    af.mfcc_repetition_determinism,
    af.mfcc_mean_pattern_duration,
    af.mfcc_longest_pattern_duration,

    -- YouTube Audio Mapping Audit (Member 2 / Youtube Matcher)
    af.youtube_mapping_id,
    ym.is_selected AS audio_mapping_is_selected,

    -- Lyrics Features: Structural & Lexical (Member 3)
    lf.word_count,
    lf.unique_word_count,
    lf.lexical_diversity,
    lf.repetition_ratio,
    lf.avg_word_length,
    lf.line_count,
    lf.chorus_repetition_count,
    lf.title_word_in_lyrics,
    lf.punctuation_ratio,
    lf.pronoun_ratio,
    lf.language,
    lf.lyrics_status,

    -- Data Quality Flags & Integration Audit
    af.spotify_track_id IS NOT NULL AS has_audio,
    lf.spotify_track_id IS NOT NULL AS has_lyrics_row,
    (
        af.spotify_track_id IS NOT NULL
        AND lf.spotify_track_id IS NOT NULL
    ) AS has_both_rows,
    (
        EXISTS (
            SELECT 1
            FROM public.song_quarter_sampling q
            WHERE q.spotify_track_id = cp.spotify_track_id
        )
    ) AS is_sampled

FROM public.chart_performance cp
JOIN public.songs s
    ON s.spotify_track_id = cp.spotify_track_id
LEFT JOIN public.audio_features af
    ON af.spotify_track_id = cp.spotify_track_id
LEFT JOIN public.youtube_mapping ym
    ON ym.id = af.youtube_mapping_id
   AND ym.is_selected = TRUE
LEFT JOIN public.lyrics_features lf
    ON lf.spotify_track_id = cp.spotify_track_id;

-- 3. Documentation Comments
COMMENT ON VIEW public.modeling_dataset IS
    'Unified modeling view at (spotify_track_id, market) grain integrating target ChartScore with 20 audio and 10 lyrics features.';

COMMIT;

-- ============================================================================
-- AUDIT & VERIFICATION QUERIES
-- ============================================================================

-- Query 1: Total records and distribution by market for sampled modeling cohort
-- Expected: 2,860 rows total (KR: 1,020; US: 955; VN: 885)
-- SELECT market, COUNT(*) AS count, COUNT(DISTINCT spotify_track_id) AS unique_tracks
-- FROM public.modeling_dataset
-- WHERE is_sampled = TRUE
-- GROUP BY market
-- ORDER BY market;

-- Query 2: Verify zero fan-out / duplicate check on primary grain
-- Expected: 0 rows returned
-- SELECT spotify_track_id, market, COUNT(*)
-- FROM public.modeling_dataset
-- GROUP BY spotify_track_id, market
-- HAVING COUNT(*) > 1;

-- Query 3: Audio completeness check on sampled cohort
-- Expected: 100% audio completeness (missing_audio = 0)
-- SELECT market,
--        COUNT(*) AS total_entries,
--        COUNT(tempo_bpm) AS audio_present,
--        COUNT(*) - COUNT(tempo_bpm) AS missing_audio
-- FROM public.modeling_dataset
-- WHERE is_sampled = TRUE
-- GROUP BY market;

-- Query 4: Lyrics completeness & status distribution on sampled cohort
-- Expected: 2,856 success, 4 not_found (3 unique instrumental tracks)
-- SELECT market, lyrics_status, COUNT(*) AS count
-- FROM public.modeling_dataset
-- WHERE is_sampled = TRUE
-- GROUP BY market, lyrics_status
-- ORDER BY market, lyrics_status;
