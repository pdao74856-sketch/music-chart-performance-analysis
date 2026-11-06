import json
import unittest
from pathlib import Path

import pandas as pd

from src.features.lyrics_processor import (
    MODEL2_FEATURES, process_lyrics_text, process_metadata,
)
from src.modeling.train_model2_lyrics import AUDIO, _prepare, build_train_medians

ROOT = Path(__file__).resolve().parents[1]


class TestLyricsProcessorAndTrainMedians(unittest.TestCase):
    def test_processor_returns_six_features_and_stable_id(self):
        row = process_lyrics_text("track-a", "Anh yêu em, em yêu anh!", language="vi")
        self.assertEqual(row["spotify_track_id"], "track-a")
        self.assertEqual(set(MODEL2_FEATURES), set(row) - {"spotify_track_id", "extraction_status", "error_reason"})
        self.assertEqual(row["word_count"], 6)
        self.assertGreater(row["punctuation_ratio"], 0)

    def test_empty_missing_and_korean_text_are_safe(self):
        empty = process_lyrics_text("e", "  \n", language="en")
        missing = process_lyrics_text("m", None, language="ko")
        korean = process_lyrics_text("k", "너를 사랑해 우리 함께 걸어가요", language="ko")
        self.assertEqual(empty["extraction_status"], "empty_file")
        self.assertEqual(missing["extraction_status"], "missing_file")
        self.assertTrue(all(empty[f] is None for f in MODEL2_FEATURES))
        self.assertGreater(korean["word_count"], 0)

    def test_unrecognized_tokens_match_canonical_failure_status(self):
        row = process_lyrics_text("punctuation", "!!! ??? …", language="en")
        self.assertEqual(row["word_count"], 0)
        self.assertEqual(row["extraction_status"], "failed")
        self.assertEqual(row["error_reason"], "no_tokens_recognized_after_tokenization")

    def test_metadata_rejects_missing_or_blank_track_ids_before_string_conversion(self):
        with self.subTest("valid string ID accepted"):
            # An empty directory means a valid ID is handled as an ordinary missing file.
            import tempfile
            with tempfile.TemporaryDirectory() as lyrics_dir:
                result = process_metadata(pd.DataFrame({"spotify_track_id": ["track-ok"]}), lyrics_dir)
                self.assertEqual(result.loc[0, "spotify_track_id"], "track-ok")
                self.assertEqual(result.loc[0, "extraction_status"], "missing_file")
        for invalid in (float("nan"), None, "", "   "):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "invalid spotify_track_id"):
                    process_metadata(pd.DataFrame({"spotify_track_id": [invalid]}), ".")

    def test_train_medians_match_preserved_artifact(self):
        df = pd.read_csv(ROOT / "data/processed/modeling_dataset.csv")
        train_ids = set(json.loads((ROOT / "models/train_track_ids.json").read_text())["track_ids"])
        rebuilt = build_train_medians(df, train_ids)
        stored = json.loads((ROOT / "data/processed/lyrics_market_medians.json").read_text())
        self.assertEqual(set(rebuilt["markets"]), {"VN", "US", "KR"})
        self.assertEqual(rebuilt["markets"], stored["markets"])
        test_ids = set(json.loads((ROOT / "models/test_track_ids.json").read_text())["track_ids"])
        self.assertFalse(train_ids & test_ids)

    def test_market_imputation_and_scaling_can_use_train_only(self):
        rows = []
        for market in ("VN", "US", "KR"):
            for i in range(2):
                row = {c: float(i + 1) for c in AUDIO}
                row.update({c: float(i + 2) for c in MODEL2_FEATURES})
                row.update({"market": market, "release_year": 2023 + i,
                            "audio_mode": "major", "spotify_track_id": f"{market}{i}"})
                rows.append(row)
        train = pd.DataFrame(rows)
        test = train.iloc[[0, 2, 4]].copy()
        test["word_count"] = float("nan")
        test["tempo_bpm"] = 1e9
        med = {"markets": {m: {"medians": {c: float(2) for c in MODEL2_FEATURES}}
                           for m in ("VN", "US", "KR")}}
        _, scaler, _ = _prepare(train, med, fit=True)
        combined = pd.concat([train, test], ignore_index=True)
        X, _, _ = _prepare(combined, med, scaler=scaler, fit=False)
        tempo_index = AUDIO.index("tempo_bpm")
        self.assertEqual(float(scaler.mean_[tempo_index]), float(train.tempo_bpm.mean()))
        self.assertEqual(len(X), len(train) + len(test))
        self.assertFalse(X.isna().any().any())


if __name__ == "__main__":
    unittest.main()
