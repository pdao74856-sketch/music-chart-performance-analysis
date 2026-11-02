-- ============================================================================
-- SQL Schema Migration: 03_lyrics_features.sql
-- Project: music-chart-performance-analysis (Member 3)
-- Compatible with: supabase/migrations/20260926090000_create_lyrics_features.sql
-- ============================================================================
-- Notes:
-- 1. Non-destructive migration: uses CREATE TABLE IF NOT EXISTS so existing
--    production tables and foreign key relationships are preserved.
-- 2. Primary key & foreign key: spotify_track_id references songs(spotify_track_id),
--    preventing duplicate rows on re-runs via ON CONFLICT (spotify_track_id) DO UPDATE.
-- 3. Nullable feature columns: unmeasurable or unavailable features are stored
--    as NULL (never coerced to 0).
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.lyrics_features (
    spotify_track_id TEXT PRIMARY KEY
        REFERENCES public.songs(spotify_track_id),

    word_count INTEGER,
    unique_word_count INTEGER,
    lexical_diversity DOUBLE PRECISION,
    repetition_ratio DOUBLE PRECISION,
    avg_word_length DOUBLE PRECISION,
    line_count INTEGER,
    chorus_repetition_count INTEGER,
    title_word_in_lyrics BOOLEAN,
    punctuation_ratio DOUBLE PRECISION,
    pronoun_ratio DOUBLE PRECISION,

    language TEXT,
    lyrics_status TEXT
);

COMMENT ON TABLE public.lyrics_features IS
    'Track-level lyrics features extracted by Member 3 pipeline (1 row per unique sampled spotify_track_id).';
COMMENT ON COLUMN public.lyrics_features.spotify_track_id IS
    'Primary key and foreign key to public.songs(spotify_track_id).';
COMMENT ON COLUMN public.lyrics_features.chorus_repetition_count IS
    'Additional occurrences of the identified chorus block (>= 0 when identified, NULL when chorus cannot be reliably identified).';
COMMENT ON COLUMN public.lyrics_features.title_word_in_lyrics IS
    'TRUE/FALSE whether at least one non-stopword alphabetic title token appears in lyrics; NULL when unmeasurable.';
COMMENT ON COLUMN public.lyrics_features.pronoun_ratio IS
    'Matched pronoun token positions divided by word_count for supported single languages (en, vi, ko, es); NULL for mixed/other/unavailable.';
