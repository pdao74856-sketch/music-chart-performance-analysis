CREATE TABLE IF NOT EXISTS lyrics_features (
    spotify_track_id TEXT PRIMARY KEY
        REFERENCES songs(spotify_track_id),

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