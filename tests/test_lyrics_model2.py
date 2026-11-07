import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import prepare_features
from src.features.lyrics_processor import MODEL2_FEATURES, process_lyrics_text, process_metadata
from src.modeling import train_model2_lyrics as runner
from src.utils.ols_cluster_fit import fit_ols_cluster, metrics


class TestLyricsProcessor(unittest.TestCase):
    def test_processor_returns_six_features_and_stable_id(self):
        row = process_lyrics_text("track-a", "Anh yêu em, em yêu anh!", language="vi")
        self.assertEqual(row["spotify_track_id"], "track-a")
        self.assertEqual(
            set(MODEL2_FEATURES),
            set(row) - {"spotify_track_id", "extraction_status", "error_reason"},
        )
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
        with tempfile.TemporaryDirectory() as lyrics_dir:
            result = process_metadata(pd.DataFrame({"spotify_track_id": ["track-ok"]}), lyrics_dir)
            self.assertEqual(result.loc[0, "spotify_track_id"], "track-ok")
            self.assertEqual(result.loc[0, "extraction_status"], "missing_file")
        for invalid in (float("nan"), None, "", "   "):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "invalid spotify_track_id"):
                    process_metadata(pd.DataFrame({"spotify_track_id": [invalid]}), ".")


class TestCanonicalLyricsModels(unittest.TestCase):
    def _fake_split(self):
        rng = np.random.default_rng(57)
        all_features = runner.MODEL2_FEATURES + [
            f"{feature}__x__{market}"
            for feature in ("energy_std", "spectral_centroid_std", "avg_word_length_std",
                            "word_count_std", "repetition_ratio_std")
            for market in ("VN", "KR")
        ]
        train_n, test_n = 120, 60
        train_market = pd.Series(np.resize(["US", "VN", "KR"], train_n))
        test_market = pd.Series(np.resize(["US", "VN", "KR"], test_n))

        def design(n, market):
            frame = pd.DataFrame(rng.normal(size=(n, len(all_features))), columns=all_features)
            frame["mode"] = rng.integers(0, 2, size=n).astype(float)
            frame["Market_VN"] = market.eq("VN").astype(float)
            frame["Market_KR"] = market.eq("KR").astype(float)
            return frame

        train_x, test_x = design(train_n, train_market), design(test_n, test_market)
        train_y = pd.Series(rng.normal(size=train_n) + 2 * train_x.word_count_std)
        test_y = pd.Series(rng.normal(size=test_n) + 2 * test_x.word_count_std)
        return SimpleNamespace(
            feature_names=all_features,
            X_train=train_x,
            X_test=test_x,
            y_train=train_y,
            y_test=test_y,
            ids_train=pd.Series([f"train-{i}" for i in range(train_n)]),
            ids_test=pd.Series([f"test-{i}" for i in range(test_n)]),
            market_train=train_market,
            market_test=test_market,
            raw_train=pd.DataFrame(index=range(train_n)),
            raw_test=pd.DataFrame(index=range(test_n)),
            meta={"initial_rows": 183, "cleaned_rows": 180, "train_rows": 120,
                  "test_rows": 60, "train_tracks": 120, "test_tracks": 60,
                  "excluded_tracks": [{"spotify_track_id": "dropped", "reason": "test row"}],
                  "seed": 43, "test_size": 0.2, "winsorize_year_q01": 1998.0,
                  "data_source": "local cache",
                  "data_path": "data/processed/modeling_dataset_clean_cache.csv",
                  "data_fingerprint": "fake-fingerprint", "source_rows": 183},
        )

    def test_load_member3_split_forces_local_loader_and_passes_dataframe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "modeling_dataset_clean_cache.csv"
            cache_path.touch()
            source = pd.DataFrame({"row": range(runner.EXPECTED_CACHE_ROWS)})
            source.attrs["data_source"] = "cache"
            source.attrs["data_fingerprint"] = "cache-fingerprint"
            expected_split = self._fake_split()

            with (patch.object(runner, "CACHE_PATH", cache_path),
                  patch.object(runner, "ROOT", Path(temp_dir)),
                  patch.object(runner, "load_clean_modeling_data", return_value=source) as loader,
                  patch.object(runner, "get_split", return_value=expected_split) as get_split):
                actual_split = runner.load_member3_split()

            loader.assert_called_once_with(source="local")
            get_split.assert_called_once()
            self.assertIs(get_split.call_args.kwargs["df"], source)
            self.assertIs(actual_split, expected_split)
            self.assertEqual(actual_split.meta["data_source"], "local cache")
            self.assertEqual(actual_split.meta["source_rows"], 2098)

    def test_load_member3_split_stops_when_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "missing_cache.csv"
            with (patch.object(runner, "CACHE_PATH", cache_path),
                  patch.object(runner, "load_clean_modeling_data") as loader,
                  patch.object(runner, "get_split") as get_split):
                with self.assertRaisesRegex(FileNotFoundError, "Required Member 3 cache"):
                    runner.load_member3_split()
            loader.assert_not_called()
            get_split.assert_not_called()

    def test_load_member3_split_stops_on_wrong_cache_row_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "modeling_dataset_clean_cache.csv"
            cache_path.touch()
            source = pd.DataFrame({"row": range(runner.EXPECTED_CACHE_ROWS - 1)})
            with (patch.object(runner, "CACHE_PATH", cache_path),
                  patch.object(runner, "load_clean_modeling_data", return_value=source) as loader,
                  patch.object(runner, "get_split") as get_split):
                with self.assertRaisesRegex(ValueError, "Expected 2098 rows.*got 2097"):
                    runner.load_member3_split()
            loader.assert_called_once_with(source="local")
            get_split.assert_not_called()

    def test_models_use_exact_features_and_exclude_interactions(self):
        split = self._fake_split()
        model1, model2 = runner._validate_and_select_features(split)
        self.assertEqual(model1, [
            "tempo_std", "onset_rate_std", "tempo_stability_std", "energy_std",
            "rms_variability_std", "spectral_centroid_std", "chord_change_rate_std",
            "key_confidence_std", "chroma_repetition_determinism_std",
            "chroma_mean_pattern_duration_std", "chroma_longest_pattern_duration_std",
            "mfcc_repetition_determinism_std", "mfcc_mean_pattern_duration_std",
            "mfcc_longest_pattern_duration_std", "mode", "release_year_std",
            "Market_VN", "Market_KR",
        ])
        self.assertEqual(model2, model1 + [
            "word_count_std", "repetition_ratio_std", "avg_word_length_std",
            "line_count_std", "chorus_repetition_count_std", "punctuation_ratio_std",
        ])
        self.assertEqual(len(runner.LYRICS_FEATURES), 6)
        self.assertFalse(any("__x__" in feature for feature in model1 + model2))

    def test_missing_required_column_stops_before_fitting(self):
        split = self._fake_split()
        split.X_test = split.X_test.drop(columns=["release_year_std"])
        with self.assertRaisesRegex(ValueError, "X_test=.*release_year_std"):
            runner._validate_and_select_features(split)

    def test_get_split_applies_lyrics_drop_and_chorus_fill_rules(self):
        config = prepare_features.DEFAULT_CONFIG.copy()
        rows = []
        for i in range(24):
            row = {
                config["id_col"]: f"track-{i:02d}",
                config["market_col"]: ("US", "VN", "KR")[i % 3],
                config["target_col"]: float((i % 4) * 30 - 30),
                config["title_col"]: f"Title {i}",
                config["artist_col"]: f"Artist {i}",
                config["year_col"]: 1990 + i,
                config["audio_mode_col"]: "major" if i % 2 else "minor",
            }
            row.update({col: float(i + j + 1) for j, col in enumerate(config["audio_cols"])})
            row.update({col: float(i + j + 2) for j, col in enumerate(config["lyrics_cols"])})
            rows.append(row)

        lyrics = config["lyrics_cols"]
        for col in lyrics:
            rows[1][col] = np.nan  # all lyric features null: dropped
        rows[2]["word_count"] = np.nan  # word_count null: dropped even with other lyrics present
        rows[3]["chorus_repetition_count"] = np.nan  # retained and filled with zero
        source = pd.DataFrame(rows)

        # Exercise get_split's real cleaning, splitting, filling, scaling and design-matrix code.
        # The post-split validation suite is orthogonal to these missing-value rules.
        with patch.object(prepare_features, "run_validation_checks", return_value={}):
            split = prepare_features.get_split(df=source, seed=43)

        self.assertEqual(split.meta["initial_rows"], 24)
        self.assertEqual(split.meta["cleaned_rows"], 22)
        self.assertEqual(split.meta["train_rows"] + split.meta["test_rows"], 22)
        self.assertEqual(
            {item["spotify_track_id"] for item in split.meta["excluded_tracks"]},
            {"track-01", "track-02"},
        )
        retained_ids = set(split.raw_train["spotify_track_id"]) | set(split.raw_test["spotify_track_id"])
        self.assertEqual(retained_ids, {f"track-{i:02d}" for i in range(24)} - {"track-01", "track-02"})

        retained_raw = pd.concat([split.raw_train, split.raw_test], ignore_index=True)
        chorus_row = retained_raw.loc[retained_raw["spotify_track_id"].eq("track-03")].iloc[0]
        self.assertEqual(chorus_row["chorus_repetition_count"], 0.0)
        split_part = split.raw_train if "track-03" in set(split.ids_train) else split.raw_test
        design_part = split.X_train if "track-03" in set(split.ids_train) else split.X_test
        row_idx = split_part.index[split_part["spotify_track_id"].eq("track-03")][0]
        scale_col = config["audio_cols"] + lyrics + [config["year_col"]]
        chorus_scale_idx = scale_col.index("chorus_repetition_count")
        expected_z = (0.0 - split.scaler.mean_[chorus_scale_idx]) / split.scaler.scale_[chorus_scale_idx]
        self.assertAlmostEqual(
            float(design_part.loc[row_idx, "chorus_repetition_count_std"]), expected_z
        )

    def test_get_split_errors_on_other_partial_lyrics_missingness(self):
        config = prepare_features.DEFAULT_CONFIG.copy()
        rows = []
        for i in range(24):
            row = {
                config["id_col"]: f"track-{i:02d}",
                config["market_col"]: ("US", "VN", "KR")[i % 3],
                config["target_col"]: float((i % 4) * 30 - 30),
                config["title_col"]: f"Title {i}",
                config["artist_col"]: f"Artist {i}",
                config["year_col"]: 1990 + i,
                config["audio_mode_col"]: "major" if i % 2 else "minor",
            }
            row.update({col: float(i + j + 1) for j, col in enumerate(config["audio_cols"])})
            row.update({col: float(i + j + 2) for j, col in enumerate(config["lyrics_cols"])})
            rows.append(row)
        rows[3]["repetition_ratio"] = np.nan
        with patch.object(prepare_features, "run_validation_checks", return_value={}):
            with self.assertRaisesRegex(ValueError, "Remaining nulls found after cleaning"):
                prepare_features.get_split(df=pd.DataFrame(rows), seed=43)

    def test_run_writes_outputs_summary_and_exact_six_term_wald(self):
        split = self._fake_split()
        fit_calls = []
        metric_calls = []

        def record_fit(y, X, groups):
            fit_calls.append({
                "target": list(y), "track_ids": list(groups), "columns": list(X.columns),
            })
            return fit_ols_cluster(y, X, groups)

        def record_metrics(y_true, y_pred, market, track_ids, n_boot=1000, seed=43):
            metric_calls.append({
                "target": list(y_true), "track_ids": list(track_ids), "market": list(market),
            })
            return metrics(y_true, y_pred, market, track_ids, n_boot=n_boot, seed=seed)

        with tempfile.TemporaryDirectory() as output_dir:
            with (patch.object(runner, "load_member3_split", return_value=split),
                  patch.object(runner, "fit_ols_cluster", side_effect=record_fit),
                  patch.object(runner, "metrics", side_effect=record_metrics)):
                result = runner.run(Path(output_dir))
            output_path = Path(output_dir)
            expected_files = {
                "member3_model_comparison.csv", "member3_model_test_metrics.csv",
                "member3_model_coefficients.csv", "member3_lyrics_joint_wald.json",
                "member3_model_run_summary.json",
            }
            self.assertEqual({path.name for path in output_path.iterdir()}, expected_files)
            comparison = pd.read_csv(output_path / "member3_model_comparison.csv")
            self.assertEqual(comparison["N"].tolist(), [60, 60])
            metrics_frame = pd.read_csv(output_path / "member3_model_test_metrics.csv")
            self.assertEqual(metrics_frame.groupby("model")["Market"].apply(list).to_dict(), {
                "Model 1 (audio + controls)": ["Overall", "KR", "US", "VN"],
                "Model 2 (audio + lyrics + controls)": ["Overall", "KR", "US", "VN"],
            })
            wald = json.loads((output_path / "member3_lyrics_joint_wald.json").read_text())
            self.assertEqual(wald["df"], 6)
            self.assertEqual(len(wald["tested_features"]), 6)
            self.assertEqual(wald["tested_features"], runner.LYRICS_FEATURES)
            summary = json.loads((output_path / "member3_model_run_summary.json").read_text())
            self.assertEqual(summary["source_pipeline"], "src.features.prepare_features.get_split()")
            self.assertEqual(summary["data_source"], "local cache")
            self.assertEqual(summary["source_rows"], 183)
            self.assertEqual(summary["train_observations"], 120)
            self.assertEqual(summary["test_observations"], 60)
            self.assertTrue(summary["interaction_terms_excluded"])
            self.assertEqual(summary["excluded_rows_reported_by_get_split"], 1)
            self.assertEqual(result["wald"]["tested_features"], runner.LYRICS_FEATURES)
            self.assertEqual(len(fit_calls), 2)
            self.assertEqual(fit_calls[0]["track_ids"], list(split.ids_train))
            self.assertEqual(fit_calls[1]["track_ids"], list(split.ids_train))
            self.assertEqual(fit_calls[0]["track_ids"], fit_calls[1]["track_ids"])
            self.assertEqual(fit_calls[0]["target"], list(split.y_train))
            self.assertEqual(fit_calls[1]["target"], list(split.y_train))
            self.assertEqual(fit_calls[0]["columns"], runner.MODEL1_FEATURES)
            self.assertEqual(fit_calls[1]["columns"], runner.MODEL2_FEATURES)
            self.assertEqual(len(metric_calls), 2)
            for metric_call in metric_calls:
                self.assertEqual(metric_call["track_ids"], list(split.ids_test))
                self.assertEqual(metric_call["target"], list(split.y_test))
                self.assertEqual(metric_call["market"], list(split.market_test))
            self.assertEqual(metric_calls[0]["track_ids"], metric_calls[1]["track_ids"])


if __name__ == "__main__":
    unittest.main()
