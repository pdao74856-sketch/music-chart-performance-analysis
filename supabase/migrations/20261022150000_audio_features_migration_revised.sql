-- Migration: align the existing Supabase audio_features table with the finalized
-- 13-feature schema for the music-chart-performance project.
--
-- Existing legacy columns are intentionally preserved:
--   tempo, energy, loudness_proxy, tempo_stability
--
-- The modeling_dataset view is temporarily dropped and recreated because
-- PostgreSQL does not allow changing the type of columns used by an existing
-- view. The entire migration runs in one transaction, so a failure rolls back
-- the view removal and all schema changes.
--
-- Safe to rerun after successful completion. Run with a role that can alter
-- public tables, create indexes/constraints, and create the view.

BEGIN;

-- 1) Read the exact key types from the parent tables and create audio_features
--    only if it does not already exist.
DO $$
DECLARE
    track_id_type TEXT;
    mapping_id_type TEXT;
BEGIN
    IF to_regclass('public.songs') IS NULL THEN
        RAISE EXCEPTION 'Required table public.songs does not exist';
    END IF;

    IF to_regclass('public.youtube_mapping') IS NULL THEN
        RAISE EXCEPTION 'Required table public.youtube_mapping does not exist';
    END IF;

    SELECT format_type(a.atttypid, a.atttypmod)
      INTO track_id_type
      FROM pg_attribute a
     WHERE a.attrelid = 'public.songs'::regclass
       AND a.attname = 'spotify_track_id'
       AND NOT a.attisdropped;

    SELECT format_type(a.atttypid, a.atttypmod)
      INTO mapping_id_type
      FROM pg_attribute a
     WHERE a.attrelid = 'public.youtube_mapping'::regclass
       AND a.attname = 'id'
       AND NOT a.attisdropped;

    IF track_id_type IS NULL THEN
        RAISE EXCEPTION 'Could not find public.songs.spotify_track_id';
    END IF;

    IF mapping_id_type IS NULL THEN
        RAISE EXCEPTION 'Could not find public.youtube_mapping.id';
    END IF;

    EXECUTE format($ddl$
        CREATE TABLE IF NOT EXISTS public.audio_features (
            spotify_track_id %s PRIMARY KEY
                REFERENCES public.songs(spotify_track_id) ON DELETE CASCADE,
            youtube_mapping_id %s
                REFERENCES public.youtube_mapping(id) ON DELETE SET NULL,
            tempo_bpm DOUBLE PRECISION,
            onset_rate DOUBLE PRECISION,
            beat_interval_variability DOUBLE PRECISION,
            rms_energy DOUBLE PRECISION,
            rms_variability DOUBLE PRECISION,
            dynamic_range DOUBLE PRECISION,
            spectral_centroid DOUBLE PRECISION,
            spectral_bandwidth DOUBLE PRECISION,
            duration_seconds DOUBLE PRECISION,
            chord_change_rate DOUBLE PRECISION,
            self_similarity DOUBLE PRECISION,
            key TEXT,
            mode TEXT,
            audio_status TEXT DEFAULT 'pending',
            audio_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    $ddl$, track_id_type, mapping_id_type);
END $$;

-- 2) Add finalized feature columns if they are missing.
--    Existing legacy columns are intentionally retained.
ALTER TABLE public.audio_features
    ADD COLUMN IF NOT EXISTS tempo_bpm DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS onset_rate DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS beat_interval_variability DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rms_energy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rms_variability DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dynamic_range DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS spectral_centroid DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS spectral_bandwidth DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS chord_change_rate DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS self_similarity DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS audio_status TEXT,
    ADD COLUMN IF NOT EXISTS audio_error TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE public.audio_features
    ALTER COLUMN audio_status SET DEFAULT 'pending';

-- 3) Drop the existing view before changing the types of key and mode.
--    No dependent views were found for modeling_dataset.
DROP VIEW IF EXISTS public.modeling_dataset;

-- 4) Convert key and mode to TEXT only if the table is empty.
--    Numeric values are not automatically reinterpreted as musical labels.
DO $$
DECLARE
    key_type TEXT;
    mode_type TEXT;
    row_count BIGINT;
BEGIN
    SELECT format_type(a.atttypid, a.atttypmod)
      INTO key_type
      FROM pg_attribute a
     WHERE a.attrelid = 'public.audio_features'::regclass
       AND a.attname = 'key'
       AND NOT a.attisdropped;

    SELECT format_type(a.atttypid, a.atttypmod)
      INTO mode_type
      FROM pg_attribute a
     WHERE a.attrelid = 'public.audio_features'::regclass
       AND a.attname = 'mode'
       AND NOT a.attisdropped;

    IF key_type IS NULL THEN
        ALTER TABLE public.audio_features ADD COLUMN key TEXT;
        key_type := 'text';
    END IF;

    IF mode_type IS NULL THEN
        ALTER TABLE public.audio_features ADD COLUMN mode TEXT;
        mode_type := 'text';
    END IF;

    IF key_type <> 'text' OR mode_type <> 'text' THEN
        SELECT COUNT(*)
          INTO row_count
          FROM public.audio_features;

        IF row_count <> 0 THEN
            RAISE EXCEPTION
                'audio_features contains % rows; refusing to convert key/mode to TEXT automatically. Review and migrate their numeric values explicitly first.',
                row_count;
        END IF;

        IF key_type <> 'text' THEN
            ALTER TABLE public.audio_features
                ALTER COLUMN key TYPE TEXT USING key::TEXT;
        END IF;

        IF mode_type <> 'text' THEN
            ALTER TABLE public.audio_features
                ALTER COLUMN mode TYPE TEXT USING mode::TEXT;
        END IF;
    END IF;
END $$;

-- 5) Recreate modeling_dataset with its existing columns and joins.
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
    af.onset_rate,
    af.loudness_proxy,
    af.tempo_stability,
    af.spectral_centroid,
    af.spectral_bandwidth,
    af.key AS audio_key,
    af.mode AS audio_mode,
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

-- 6) Add expected foreign keys only if an FK for that column does not exist.
DO $$
DECLARE
    track_attnum SMALLINT;
    mapping_attnum SMALLINT;
BEGIN
    SELECT attnum
      INTO track_attnum
      FROM pg_attribute
     WHERE attrelid = 'public.audio_features'::regclass
       AND attname = 'spotify_track_id'
       AND NOT attisdropped;

    SELECT attnum
      INTO mapping_attnum
      FROM pg_attribute
     WHERE attrelid = 'public.audio_features'::regclass
       AND attname = 'youtube_mapping_id'
       AND NOT attisdropped;

    IF track_attnum IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.conrelid = 'public.audio_features'::regclass
          AND c.contype = 'f'
          AND c.conkey = ARRAY[track_attnum]::SMALLINT[]
    ) THEN
        ALTER TABLE public.audio_features
            ADD CONSTRAINT audio_features_spotify_track_id_fkey
            FOREIGN KEY (spotify_track_id)
            REFERENCES public.songs(spotify_track_id)
            ON DELETE CASCADE;
    END IF;

    IF mapping_attnum IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        WHERE c.conrelid = 'public.audio_features'::regclass
          AND c.contype = 'f'
          AND c.conkey = ARRAY[mapping_attnum]::SMALLINT[]
    ) THEN
        ALTER TABLE public.audio_features
            ADD CONSTRAINT audio_features_youtube_mapping_id_fkey
            FOREIGN KEY (youtube_mapping_id)
            REFERENCES public.youtube_mapping(id)
            ON DELETE SET NULL;
    END IF;
END $$;

-- 7) A YouTube mapping may be used by at most one audio_features row.
CREATE UNIQUE INDEX IF NOT EXISTS audio_features_youtube_mapping_id_uidx
    ON public.audio_features (youtube_mapping_id)
    WHERE youtube_mapping_id IS NOT NULL;

-- 8) Document feature definitions and units.
COMMENT ON COLUMN public.audio_features.tempo_bpm IS
    'Estimated tempo in beats per minute.';

COMMENT ON COLUMN public.audio_features.onset_rate IS
    'Detected onsets per minute; keep the unit consistent across all tracks.';

COMMENT ON COLUMN public.audio_features.beat_interval_variability IS
    'Standard deviation of consecutive detected beat intervals, in seconds.';

COMMENT ON COLUMN public.audio_features.rms_energy IS
    'Mean frame-level RMS amplitude; depends on audio mastering/normalization.';

COMMENT ON COLUMN public.audio_features.rms_variability IS
    'Standard deviation of frame-level RMS amplitude.';

COMMENT ON COLUMN public.audio_features.dynamic_range IS
    'P90 minus P10 of frame-level RMS expressed in dBFS; project-defined proxy.';

COMMENT ON COLUMN public.audio_features.spectral_centroid IS
    'Mean frame-level spectral centroid in Hz.';

COMMENT ON COLUMN public.audio_features.spectral_bandwidth IS
    'Mean frame-level spectral bandwidth in Hz.';

COMMENT ON COLUMN public.audio_features.duration_seconds IS
    'Duration of the exact audio file analyzed, in seconds.';

COMMENT ON COLUMN public.audio_features.chord_change_rate IS
    'Valid chord changes per minute of valid analyzed duration, estimated using Chordino/Vamp.';

COMMENT ON COLUMN public.audio_features.self_similarity IS
    'Project-defined mean cosine similarity between selected non-identical segments using chroma vectors.';

COMMENT ON COLUMN public.audio_features.key IS
    'Estimated tonic/key label, e.g. C or F#; categorical, not ordinal.';

COMMENT ON COLUMN public.audio_features.mode IS
    'Estimated scale mode, normalized to major/minor/unknown according to pipeline policy.';

COMMENT ON COLUMN public.audio_features.audio_status IS
    'Extraction status, e.g. pending, success, failed, not_found, rejected.';

COMMENT ON COLUMN public.audio_features.audio_error IS
    'Error details when audio processing fails.';

COMMIT;