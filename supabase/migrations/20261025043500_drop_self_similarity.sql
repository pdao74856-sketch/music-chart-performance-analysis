-- Migration: Drop obsolete self_similarity column from audio_features table
-- and recreate modeling_dataset view without self_similarity.

BEGIN;

-- 1) Drop view depending on self_similarity
DROP VIEW IF EXISTS public.modeling_dataset;

-- 2) Drop obsolete self_similarity column
ALTER TABLE public.audio_features
    DROP COLUMN IF EXISTS self_similarity;

-- 3) Recreate public.modeling_dataset view without self_similarity
CREATE VIEW public.modeling_dataset AS
SELECT
    cp.spotify_track_id,
    cp.market,
    cp.lifetime_average_rank,
    cp.chart_score,
    cp.peak_rank,
    cp.weeks_on_chart,
    cp.weeks_in_top10,
    s.title,
    s.artist,
    s.album,
    s.release_date,
    EXTRACT(YEAR FROM s.release_date)::INTEGER AS release_year,
    af.tempo,
    af.energy,
    af.tempo_bpm,
    af.onset_rate,
    af.beat_interval_variability,
    af.rms_energy,
    af.rms_variability,
    af.dynamic_range,
    af.loudness_proxy,
    af.tempo_stability,
    af.spectral_centroid,
    af.spectral_bandwidth,
    af.duration_seconds,
    af.duration_analyzed_seconds,
    af.key AS audio_key,
    af.mode AS audio_mode,
    af.key_confidence,
    af.chord_change_rate,
    af.chroma_repetition_determinism,
    af.chroma_mean_pattern_duration,
    af.chroma_longest_pattern_duration,
    af.mfcc_repetition_determinism,
    af.mfcc_mean_pattern_duration,
    af.mfcc_longest_pattern_duration,
    af.youtube_mapping_id,
    ym.is_selected AS audio_mapping_is_selected,
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
LEFT JOIN public.lyrics_features lf
    ON lf.spotify_track_id = cp.spotify_track_id;

COMMIT;
