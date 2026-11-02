-- Migration: Add standardized RQA repetition features, key confidence, and analyzed duration;
-- drop obsolete self_similarity column; and update the public.modeling_dataset view.

BEGIN;

-- 1) Add new feature columns to public.audio_features
ALTER TABLE public.audio_features
    ADD COLUMN IF NOT EXISTS chroma_repetition_determinism DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS chroma_mean_pattern_duration DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS chroma_longest_pattern_duration DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mfcc_repetition_determinism DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mfcc_mean_pattern_duration DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mfcc_longest_pattern_duration DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS key_confidence DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS duration_analyzed_seconds DOUBLE PRECISION;

-- 2) Drop obsolete self_similarity column (superseded by RQA repetition features)
ALTER TABLE public.audio_features
    DROP COLUMN IF EXISTS self_similarity;

-- 3) Add documentation comments
COMMENT ON COLUMN public.audio_features.chroma_repetition_determinism IS
    'Fraction of recurrence points in diagonal lines (>= 3s) from 12-dim pitch-class chroma (harmonic repetition).';

COMMENT ON COLUMN public.audio_features.chroma_mean_pattern_duration IS
    'Mean duration in seconds of repeating harmonic patterns detected via chroma recurrence.';

COMMENT ON COLUMN public.audio_features.chroma_longest_pattern_duration IS
    'Longest repeating harmonic pattern duration in seconds detected via chroma recurrence.';

COMMENT ON COLUMN public.audio_features.mfcc_repetition_determinism IS
    'Fraction of recurrence points in diagonal lines (>= 3s) from 20 standardized MFCCs (timbral/arrangement repetition).';

COMMENT ON COLUMN public.audio_features.mfcc_mean_pattern_duration IS
    'Mean duration in seconds of repeating timbral/production patterns detected via MFCC recurrence.';

COMMENT ON COLUMN public.audio_features.mfcc_longest_pattern_duration IS
    'Longest repeating timbral/production pattern duration in seconds detected via MFCC recurrence.';

COMMENT ON COLUMN public.audio_features.key_confidence IS
    'Pearson correlation coefficient between the song chroma profile and the best-matching Krumhansl-Schmuckler key profile.';

COMMENT ON COLUMN public.audio_features.duration_analyzed_seconds IS
    'Exact duration in seconds of the audio waveform analyzed during feature extraction.';

-- 4) Recreate public.modeling_dataset view with all new features included (without self_similarity)
DROP VIEW IF EXISTS public.modeling_dataset;

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
