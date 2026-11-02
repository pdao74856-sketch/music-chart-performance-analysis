#!/usr/bin/env python3
"""Comprehensive unit tests for Member 3 lyrics pipeline (Stage 5).

Can be run with standard library unittest:
    python -m unittest discover -s tests -v
or directly:
    python tests/test_lyrics_pipeline.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lyrics.extract_lyrics_features import (
    calculate_features,
    detect_chorus_repetition,
    load_yaml,
    normalize_text,
    tokenize,
)
from src.lyrics.language_detection import detect_language
from src.lyrics.validate_lyrics_features import validate_row

CONFIG_PATH = ROOT / "src/lyrics/language_rules.yaml"


class TestLyricsFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(CONFIG_PATH)

    def test_01_empty_and_whitespace_lyrics(self) -> None:
        """Case 1: Empty or whitespace-only lyrics must return NULL for all features (no div-by-zero)."""
        for empty_text in ("", "   ", "\n\t \n  "):
            features = calculate_features(empty_text, "My Song", "en", self.config)
            for name, val in features.items():
                self.assertIsNone(
                    val, f"Expected None for {name} on empty input, got {val!r}"
                )

    def test_02_punctuation_only_lyrics(self) -> None:
        """Case 2: Punctuation-only lyrics have 0 word tokens and NULL token-denominator ratios."""
        text = "!!! ??? ... ---\n... !!!"
        features = calculate_features(text, "Hello World", "en", self.config)
        self.assertEqual(features["word_count"], 0)
        self.assertEqual(features["unique_word_count"], 0)
        self.assertIsNone(features["lexical_diversity"])
        self.assertIsNone(features["repetition_ratio"])
        self.assertIsNone(features["avg_word_length"])
        self.assertEqual(features["line_count"], 2)
        self.assertIsNone(features["chorus_repetition_count"])
        self.assertIsNone(features["title_word_in_lyrics"])
        self.assertIsNotNone(features["punctuation_ratio"])
        self.assertGreater(features["punctuation_ratio"], 0.5)
        self.assertIsNone(features["pronoun_ratio"])

    def test_03_highly_repetitive_lyrics(self) -> None:
        """Case 3: Highly repetitive lyrics yield low lexical_diversity and high repetition_ratio."""
        line = "baby baby baby oh"
        text = "\n".join([line] * 8)
        features = calculate_features(text, "Baby", "en", self.config)
        self.assertEqual(features["word_count"], 32)
        self.assertEqual(features["unique_word_count"], 2)
        self.assertAlmostEqual(features["lexical_diversity"], 2 / 32)
        self.assertAlmostEqual(features["repetition_ratio"], 30 / 32)
        self.assertAlmostEqual(
            features["lexical_diversity"] + features["repetition_ratio"], 1.0
        )

    def test_04_title_word_in_lyrics_true_false_and_stopwords(self) -> None:
        """Case 4: Title content word presence, absence, and all-stopword title NULL handling."""
        lyrics = "Walking through the midnight rain alone in the city"
        feat_true = calculate_features(lyrics, "Midnight Rain", "en", self.config)
        self.assertIs(feat_true["title_word_in_lyrics"], True)

        feat_false = calculate_features(lyrics, "Golden Sunshine", "en", self.config)
        self.assertIs(feat_false["title_word_in_lyrics"], False)

        # Title with only stopwords ('As It Was') has no content word -> None
        feat_stop = calculate_features(lyrics, "As It Was", "en", self.config)
        self.assertIsNone(feat_stop["title_word_in_lyrics"])

    def test_05_vietnamese_diacritics_and_pronouns(self) -> None:
        """Case 5: Vietnamese lyrics with NFC diacritics, syllable tokens, and pronouns."""
        # Combine NFC and NFD input to verify NFC normalization consistency
        vi_text = (
            "Anh yêu em giữa mùa thu Hà Nội\n"
            "Chúng tôi đi qua những con đường nhỏ\n"
            "Em có nhớ những ngày xưa"
        )
        features = calculate_features(vi_text, "Mùa Thu Hà Nội", "vi", self.config)
        self.assertEqual(features["word_count"], 22)
        self.assertIs(features["title_word_in_lyrics"], True)
        # 'anh' (1), 'em' (2), 'chúng tôi' (2 token positions) -> 5 pronoun positions / 22 tokens
        self.assertAlmostEqual(features["pronoun_ratio"], 5 / 22)

    def test_06_english_contractions_and_hyphens(self) -> None:
        """Case 6: English contractions (don't, I'm) and hyphens kept as single tokens."""
        en_text = "I'm sure you don't mind a well-known song when we're together"
        tokens = tokenize(normalize_text(en_text, self.config), "en", self.config)
        self.assertIn("i'm", tokens)
        self.assertIn("don't", tokens)
        self.assertIn("well-known", tokens)
        self.assertIn("we're", tokens)
        features = calculate_features(en_text, "Well-Known Song", "en", self.config)
        self.assertEqual(features["word_count"], len(tokens))
        self.assertIs(features["title_word_in_lyrics"], True)
        # 'i'm', 'you', 'we're' -> 3 pronouns out of 11 tokens
        self.assertAlmostEqual(features["pronoun_ratio"], 3 / len(tokens))

    def test_07_korean_lyrics_and_language_detection(self) -> None:
        """Case 7: Korean Hangul tokenization, pronouns, and short-line language detection."""
        ko_text = (
            "나도 한때는 그이의 손을 잡고\n"
            "내가 온 세상 주인공 된 듯\n"
            "너를 사랑해 우리 함께 걸어가요\n"
            "그대 나를 보며 웃어주네요"
        )
        lang, _ = detect_language(ko_text)
        self.assertEqual(lang, "ko")
        features = calculate_features(ko_text, "주인공 (Hero)", "ko", self.config)
        self.assertGreater(features["word_count"], 10)
        self.assertIs(features["title_word_in_lyrics"], True)
        self.assertIsNotNone(features["pronoun_ratio"])
        self.assertGreater(features["pronoun_ratio"], 0.1)

    def test_08_multilingual_mixed_lyrics(self) -> None:
        """Case 8: Bilingual Korean-English lyrics detected as mixed; pronoun_ratio is NULL."""
        mixed_text = (
            "I want to hold your hand tonight in the city light\n"
            "Take me to the stars where we never say goodbye\n"
            "Everything I need is right here by my side\n"
            "오늘 밤도 너와 함께 이 길을 걸어가고 싶어\n"
            "우리들의 이야기는 끝나지 않을 거야 영원히\n"
            "아름다운 별빛 아래에서 너의 손을 잡을게"
        )
        lang, _ = detect_language(mixed_text)
        self.assertEqual(lang, "mixed")
        features = calculate_features(mixed_text, "City Light 별빛", "mixed", self.config)
        self.assertGreater(features["word_count"], 20)
        self.assertIs(features["title_word_in_lyrics"], True)
        self.assertIsNone(features["pronoun_ratio"])

    def test_09_lyrics_without_chorus_label_and_no_repetition(self) -> None:
        """Case 9: Unlabeled lyrics without repeated blocks must return NULL (never 0) for chorus."""
        unlabeled_unique = (
            "Morning light wakes up the quiet street\n"
            "Coffee brewing in the small kitchen corner\n"
            "Buses roll past the old brick station\n"
            "People hurry toward the downtown bridge\n"
            "Clouds drift slowly over silver roofs\n"
            "Evening brings a gentle autumn breeze"
        )
        self.assertIsNone(detect_chorus_repetition(unlabeled_unique, self.config))
        # However, if an explicit [Chorus] tag is present once without repetition, it returns 0
        tagged_once = "[Chorus]\n" + unlabeled_unique
        self.assertEqual(detect_chorus_repetition(tagged_once, self.config), 0)

    def test_10_verbatim_chorus_repetition_tagged_and_untagged(self) -> None:
        """Case 10: Verbatim repeated chorus block is detected with and without [Chorus] tags."""
        chorus_block = (
            "We are dancing in the pouring rain tonight\n"
            "Holding on until the morning light arrives\n"
            "Nothing can tear our hearts apart anymore\n"
            "Dancing in the pouring rain forevermore"
        )
        verse_1 = (
            "Walking down the empty boulevard alone\n"
            "Thinking about the words you said to me"
        )
        verse_2 = (
            "Shadows fall across the neon window pane\n"
            "Every memory comes rushing back again"
        )
        untagged_song = f"{verse_1}\n{chorus_block}\n{verse_2}\n{chorus_block}\n{chorus_block}"
        self.assertEqual(detect_chorus_repetition(untagged_song, self.config), 2)

        tagged_song = (
            f"[Verse 1]\n{verse_1}\n[Chorus]\n{chorus_block}\n"
            f"[Verse 2]\n{verse_2}\n[Chorus]\n{chorus_block}"
        )
        self.assertEqual(detect_chorus_repetition(tagged_song, self.config), 1)

    def test_11_missing_file_and_encoding_error_pipeline(self) -> None:
        """Case 11: Pipeline handles missing files, empty files, and invalid UTF-8 encodings."""
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            lyrics_dir = tmp_dir / "lyrics"
            lyrics_dir.mkdir()

            # 1. Valid file
            (lyrics_dir / "track_ok.txt").write_text(
                "Hello my friend I love you\nWe are here together tonight",
                encoding="utf-8",
            )
            # 2. Empty file
            (lyrics_dir / "track_empty.txt").write_text("   \n  ", encoding="utf-8")
            # 3. Bad encoding file (invalid UTF-8 bytes)
            (lyrics_dir / "track_bad_enc.txt").write_bytes(b"\xff\xfe\x00\x80\x81\xfe")

            status_csv = tmp_dir / "status.csv"
            with status_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "spotify_track_id",
                        "title",
                        "artist",
                        "market",
                        "language",
                        "source",
                        "lyrics_status",
                        "lyrics_file",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "spotify_track_id": "track_ok",
                        "title": "Hello Friend",
                        "artist": "Artist A",
                        "market": "US",
                        "language": "en",
                        "source": "LRCLIB",
                        "lyrics_status": "success",
                        "lyrics_file": str(lyrics_dir / "track_ok.txt"),
                    },
                    {
                        "spotify_track_id": "track_missing",
                        "title": "Ghost Track",
                        "artist": "Artist B",
                        "market": "US",
                        "language": "en",
                        "source": "LRCLIB",
                        "lyrics_status": "success",
                        "lyrics_file": str(lyrics_dir / "does_not_exist.txt"),
                    },
                    {
                        "spotify_track_id": "track_empty",
                        "title": "Silent Track",
                        "artist": "Artist C",
                        "market": "VN",
                        "language": "vi",
                        "source": "LRCLIB",
                        "lyrics_status": "success",
                        "lyrics_file": str(lyrics_dir / "track_empty.txt"),
                    },
                    {
                        "spotify_track_id": "track_bad_enc",
                        "title": "Corrupt Track",
                        "artist": "Artist D",
                        "market": "KR",
                        "language": "ko",
                        "source": "LRCLIB",
                        "lyrics_status": "success",
                        "lyrics_file": str(lyrics_dir / "track_bad_enc.txt"),
                    },
                ])

            out_csv = tmp_dir / "features.csv"
            log_csv = tmp_dir / "log.csv"
            cmd = [
                sys.executable,
                str(ROOT / "src/lyrics/extract_lyrics_features.py"),
                "--root",
                str(ROOT),
                "--input",
                str(status_csv),
                "--lyrics-dir",
                str(lyrics_dir),
                "--output",
                str(out_csv),
                "--log",
                str(log_csv),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = {r["spotify_track_id"]: r for r in csv.DictReader(f)}

            self.assertEqual(rows["track_ok"]["extraction_status"], "success")
            self.assertEqual(rows["track_missing"]["extraction_status"], "missing_file")
            self.assertEqual(rows["track_empty"]["extraction_status"], "empty_file")
            self.assertEqual(rows["track_bad_enc"]["extraction_status"], "read_error")

            # Ensure failed/missing/empty rows have empty (NULL) features, not 0
            for tid in ("track_missing", "track_empty", "track_bad_enc"):
                self.assertEqual(rows[tid]["word_count"], "")
                self.assertEqual(validate_row(rows[tid], lyrics_dir=lyrics_dir), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
