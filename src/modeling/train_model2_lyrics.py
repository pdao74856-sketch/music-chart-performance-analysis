"""Train a fair audio-only vs audio-plus-lyrics OLS comparison.

Uses the frozen track-ID split, Train-only market medians and Train-only scaling.
Run from repository root: python -m src.modeling.train_model2_lyrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.features.lyrics_processor import MODEL2_FEATURES
from src.utils.ols_cluster_fit import fit_ols_cluster, metrics, wald_block

ROOT = Path(__file__).resolve().parents[2]
AUDIO = ["tempo_bpm", "onset_rate", "tempo_stability", "rms_energy",
         "rms_variability", "spectral_centroid", "chord_change_rate",
         "key_confidence", "chroma_repetition_determinism",
         "chroma_mean_pattern_duration", "chroma_longest_pattern_duration",
         "mfcc_repetition_determinism", "mfcc_mean_pattern_duration",
         "mfcc_longest_pattern_duration"]


def build_train_medians(df: pd.DataFrame, train_ids: set[str]) -> dict:
    train = df[df.spotify_track_id.isin(train_ids)]
    markets = {}
    for market, group in train.groupby("market", sort=True):
        med = group[MODEL2_FEATURES].apply(pd.to_numeric, errors="coerce").median()
        if med.isna().any():
            raise ValueError(f"No valid Train median for {market}: {med[med.isna()].index.tolist()}")
        markets[market] = {
            "n_rows": int(len(group)),
            "medians": {k: float(v) for k, v in med.items()},
            "valid_counts": {k: int(group[k].notna().sum()) for k in MODEL2_FEATURES},
        }
    if set(markets) != {"VN", "US", "KR"}:
        raise ValueError(f"Expected VN/US/KR Train rows, got {sorted(markets)}")
    return {"source": "data/processed/modeling_dataset.csv",
            "split_source": "models/train_track_ids.json",
            "train_unique_tracks": int(train.spotify_track_id.nunique()),
            "train_rows": int(len(train)), "features": MODEL2_FEATURES,
            "markets": markets}


def _prepare(df: pd.DataFrame, medians: dict, scaler: StandardScaler | None = None,
             fit: bool = False):
    out = df.copy()
    for col in MODEL2_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    missing = out.groupby("market")[MODEL2_FEATURES].agg(lambda x: int(x.isna().sum()))
    for market, values in medians["markets"].items():
        mask = out.market.eq(market)
        for col in MODEL2_FEATURES:
            out.loc[mask, col] = out.loc[mask, col].fillna(values["medians"][col])
    if out[MODEL2_FEATURES].isna().any().any():
        raise ValueError("Lyrics median missing for one or more markets/features")
    continuous = AUDIO + MODEL2_FEATURES + ["release_year"]
    out["tempo_bpm"] = pd.to_numeric(out.tempo_bpm, errors="coerce")
    out["tempo_bpm"] = out["tempo_bpm"]
    out["word_count"] = np.log1p(out.word_count.astype(float))
    if fit:
        scaler = StandardScaler().fit(out[continuous])
    assert scaler is not None
    z = pd.DataFrame(scaler.transform(out[continuous]), columns=continuous, index=out.index)
    # Model 1 and Model 2 use identical rows and shared training-fitted transforms.
    x = pd.DataFrame(index=out.index)
    for col in AUDIO + MODEL2_FEATURES + ["release_year"]:
        x[f"{col}_std"] = z[col]
    mode = out["audio_mode"].astype(str).str.lower()
    x["mode"] = mode.map({"major": 1.0, "minor": 0.0, "1": 1.0, "0": 0.0})
    if x["mode"].isna().any():
        raise ValueError("Invalid audio_mode")
    x["Market_VN"] = out.market.eq("VN").astype(float)
    x["Market_KR"] = out.market.eq("KR").astype(float)
    return x, scaler, missing


def run(data_path: Path | None = None, output_dir: Path | None = None) -> dict:
    data_path = data_path or ROOT / "data/processed/modeling_dataset.csv"
    output_dir = output_dir or ROOT / "reports/report4"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    train_ids = set(json.loads((ROOT / "models/train_track_ids.json").read_text(encoding="utf-8"))["track_ids"])
    test_ids = set(json.loads((ROOT / "models/test_track_ids.json").read_text(encoding="utf-8"))["track_ids"])
    if train_ids & test_ids:
        raise ValueError("Frozen train/test IDs overlap")
    if df.spotify_track_id.duplicated().any() and df.duplicated(["spotify_track_id", "market"]).any():
        raise ValueError("Duplicate track-market observations in modeling data")
    df = df[df.spotify_track_id.isin(train_ids | test_ids)].copy().reset_index(drop=True)
    train_mask = df.spotify_track_id.isin(train_ids)
    test_mask = df.spotify_track_id.isin(test_ids)
    med = build_train_medians(df, train_ids)
    # Preserve existing untracked artifact when its content already validates.
    median_path = ROOT / "data/processed/lyrics_market_medians.json"
    if median_path.exists():
        current = json.loads(median_path.read_text(encoding="utf-8"))
        if current.get("markets") != med["markets"]:
            raise ValueError("Existing median artifact does not match recalculated Train-only values")
    else:
        median_path.write_text(json.dumps(med, indent=2) + "\n", encoding="utf-8")
    # Fit every transform only on official Train observations, then apply to both splits.
    _, scaler, _ = _prepare(df.loc[train_mask].copy(), med, fit=True)
    X, _, _ = _prepare(df, med, scaler=scaler, fit=False)
    # Missingness is recorded pre-imputation; after-imputation must be exactly zero.
    missing_records = []
    for subset_name, mask in (("train", train_mask), ("test", test_mask)):
        part = df.loc[mask]
        for market in ("VN", "US", "KR"):
            market_rows = part[part.market.eq(market)]
            for col in MODEL2_FEATURES:
                missing_records.append({"split": subset_name, "market": market,
                    "feature": col, "n": len(market_rows),
                    "missing_before": int(market_rows[col].isna().sum()),
                    "missing_after": 0})
    pd.DataFrame(missing_records).to_csv(output_dir / "member3_model2_missingness.csv", index=False)

    results, predictions = {}, {}
    y = pd.to_numeric(df.chart_score, errors="raise")
    ids = df.spotify_track_id
    m1_cols = [f"{c}_std" for c in AUDIO] + ["mode", "release_year_std", "Market_VN", "Market_KR"]
    m2_cols = m1_cols + [f"{c}_std" for c in MODEL2_FEATURES]
    for name, cols in (("Model 1 (audio + controls)", m1_cols),
                       ("Model 2 (audio + lyrics + controls)", m2_cols)):
        fit = fit_ols_cluster(y.loc[train_mask], X.loc[train_mask, cols], ids.loc[train_mask])
        results[name] = fit
        pred = fit.predict(sm.add_constant(X.loc[test_mask, cols], has_constant="add"))
        predictions[name] = np.asarray(pred)
    # Fair test rows identical by construction.
    score_rows = []
    for name, pred in predictions.items():
        actual = y.loc[test_mask].to_numpy()
        err = actual - pred
        r2 = 1 - np.sum(err**2) / np.sum((actual - actual.mean())**2)
        score_rows.append({"model": name, "N": len(actual), "R2": r2,
            "adjusted_R2": 1 - (1-r2)*(len(actual)-1)/(len(actual)-len(m1_cols if name.startswith("Model 1") else m2_cols)-1),
            "RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(np.abs(err)))})
    test_ids_rows = df.loc[test_mask, "spotify_track_id"].to_numpy()
    test_markets = df.loc[test_mask, "market"].to_numpy()
    market_metrics = []
    for name, pred in predictions.items():
        m = metrics(y.loc[test_mask], pred, market=test_markets,
                    track_ids=test_ids_rows, n_boot=1000, seed=43)
        m.insert(0, "model", name)
        market_metrics.append(m)
    pd.DataFrame(score_rows).to_csv(output_dir / "member3_model_comparison.csv", index=False)
    pd.concat(market_metrics, ignore_index=True).to_csv(output_dir / "member3_model_test_metrics.csv", index=False)
    coefficients = []
    for name, fit in results.items():
        ci = fit.conf_int()
        for term in fit.params.index:
            coefficients.append({"model": name, "term": term, "coefficient": fit.params[term],
                "cluster_se": fit.bse[term], "p_value": fit.pvalues[term],
                "ci_lower": ci.loc[term, 0], "ci_upper": ci.loc[term, 1]})
    pd.DataFrame(coefficients).to_csv(output_dir / "member3_model_coefficients.csv", index=False)
    lyrics_terms = [f"{c}_std" for c in MODEL2_FEATURES]
    wald = wald_block(results["Model 2 (audio + lyrics + controls)"], lyrics_terms)
    wald.update({"method": "cluster-robust Wald chi-square", "alpha": 0.05,
                 "conclusion": "reject H0" if wald["p_value"] < 0.05 else "fail to reject H0"})
    (output_dir / "member3_lyrics_joint_wald.json").write_text(json.dumps(wald, indent=2) + "\n", encoding="utf-8")
    med["test_unique_tracks"] = len(test_ids)
    med["observed"] = {"total_rows": len(df), "train_rows": int(train_mask.sum()),
                       "test_rows": int(test_mask.sum()),
                       "train_market_rows": df[train_mask].groupby("market").size().to_dict(),
                       "test_market_rows": df[test_mask].groupby("market").size().to_dict(),
                       "fair_comparison_train_rows": int(train_mask.sum()),
                       "fair_comparison_test_rows": int(test_mask.sum())}
    (output_dir / "member3_model_run_summary.json").write_text(json.dumps(med, indent=2) + "\n", encoding="utf-8")
    return {"medians": med, "comparison": score_rows, "wald": wald}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
